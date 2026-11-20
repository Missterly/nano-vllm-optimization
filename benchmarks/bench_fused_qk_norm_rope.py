"""Check and benchmark the standalone Qwen3-0.6B Q/K RMSNorm + RoPE fusion."""

import argparse
from collections.abc import Callable
from statistics import median

import torch

from nanovllm.layers.fused_qk_norm_rope import fused_qk_norm_rope
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.rotary_embedding import RotaryEmbedding


HEAD_DIM = 128
QUERY_HEADS = 16
KV_HEADS = 8
MAX_POSITION = 40960
ATOL = 0.002
RTOL = 0.02

QKOperation = Callable[[], tuple[torch.Tensor, torch.Tensor]]


def make_case(
    tokens: int,
    query_norm: RMSNorm,
    key_norm: RMSNorm,
    rope: RotaryEmbedding,
    generator: torch.Generator,
) -> tuple[QKOperation, QKOperation]:
    device = query_norm.weight.device
    query_width = QUERY_HEADS * HEAD_DIM
    kv_width = KV_HEADS * HEAD_DIM
    packed_qkv = torch.randn(
        tokens, query_width + 2 * kv_width,
        device=device, dtype=torch.bfloat16, generator=generator,
    )
    query, key, _ = packed_qkv.split((query_width, kv_width, kv_width), dim=-1)
    query = query.view(tokens, QUERY_HEADS, HEAD_DIM)
    key = key.view(tokens, KV_HEADS, HEAD_DIM)
    positions = torch.randint(0, 2048, (tokens,), device=device, generator=generator)
    positions[0] = 0
    if tokens > 1:
        positions[-1] = MAX_POSITION - 1

    def original() -> tuple[torch.Tensor, torch.Tensor]:
        return rope(positions, query_norm(query), key_norm(key))

    def fused() -> tuple[torch.Tensor, torch.Tensor]:
        return fused_qk_norm_rope(
            query, key, positions, query_norm.weight, key_norm.weight,
            rope.cos_sin_cache, query_norm.eps,
        )

    return original, fused


def check_correctness(tokens: int, original: QKOperation, fused: QKOperation) -> None:
    reference = original()
    result = fused()
    for name, actual, expected in zip(("Q", "K"), result, reference, strict=True):
        torch.testing.assert_close(
            actual, expected, atol=ATOL, rtol=RTOL,
            msg=lambda message: f"tokens={tokens}, tensor={name}\n{message}",
        )
        max_abs = (actual.float() - expected.float()).abs().max().item()
        exact = (actual == expected).float().mean().item()
        print(f"  tokens={tokens:4d}  {name}  max_abs={max_abs:.6g}  exact={exact:.5%}")


def count_cuda_kernels(operation: QKOperation) -> int:
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as profile:
        operation()
        torch.cuda.synchronize()
    return sum(event.device_type.name == "CUDA" for event in profile.events())


def capture_graph(operation: QKOperation, calls: int) -> torch.cuda.CUDAGraph:
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(calls):
            operation()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    return graph


def benchmark(
    tokens: int, original: QKOperation, fused: QKOperation, calls: int, repeats: int,
) -> None:
    operations = (original, fused)
    for operation in operations:
        for _ in range(12):
            operation()
    torch.cuda.synchronize()

    original_kernels, fused_kernels = (count_cuda_kernels(operation) for operation in operations)
    graphs = [capture_graph(operation, calls) for operation in operations]
    samples = ([], [])
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for repeat in range(repeats):
        order = (0, 1) if repeat % 2 == 0 else (1, 0)
        for index in order:
            start.record()
            graphs[index].replay()
            end.record()
            end.synchronize()
            samples[index].append(start.elapsed_time(end) * 1000 / calls)

    original_us, fused_us = map(median, samples)
    reduction = 1 - fused_us / original_us
    print(
        f"  tokens={tokens:4d}  original={original_us:8.3f} us  fused={fused_us:8.3f} us"
        f"  latency reduction={reduction:7.2%}  kernels={original_kernels}->{fused_kernels}"
    )


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return number


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=positive_int, nargs="+", default=[1, 8, 16, 32, 64, 128, 256, 512, 1024])
    parser.add_argument("--check-only", action="store_true", help="Skip timing after the correctness checks.")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--calls-per-graph", type=positive_int, default=128)
    parser.add_argument("--repeats", type=positive_int, default=11)
    args = parser.parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        parser.error("a CUDA device with BF16 support is required")

    device = torch.device("cuda", torch.cuda.current_device())
    generator = torch.Generator(device=device).manual_seed(args.seed)
    query_norm = RMSNorm(HEAD_DIM).to(device=device, dtype=torch.bfloat16)
    key_norm = RMSNorm(HEAD_DIM).to(device=device, dtype=torch.bfloat16)
    query_norm.weight.uniform_(0.25, 1.75, generator=generator)
    key_norm.weight.uniform_(0.25, 1.75, generator=generator)
    with torch.device(device):
        rope = RotaryEmbedding(HEAD_DIM, HEAD_DIM, MAX_POSITION, 1_000_000)

    cases = {
        tokens: make_case(tokens, query_norm, key_norm, rope, generator)
        for tokens in dict.fromkeys(args.tokens)
    }
    print(f"{torch.cuda.get_device_name(device)} | PyTorch {torch.__version__} | BF16 | seed={args.seed}")
    print(f"Correctness against torch.compile (atol={ATOL}, rtol={RTOL})")
    for tokens, operations in cases.items():
        check_correctness(tokens, *operations)
    print("All correctness checks passed.", flush=True)

    if not args.check_only:
        print(f"CUDA Graph replay median ({args.repeats} rounds, {args.calls_per_graph} calls per graph)")
        for tokens, operations in cases.items():
            benchmark(tokens, *operations, args.calls_per_graph, args.repeats)


if __name__ == "__main__":
    main()
