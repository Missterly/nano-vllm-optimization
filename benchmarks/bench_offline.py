"""Benchmark an offline batch without timing tokenization or text decoding."""

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import sys
import tempfile
import time
from datetime import datetime, timezone
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


BENCHMARK_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCHMARK_DIR.parent

LATENCIES = {"ttft_ms": "TTFT", "tpot_ms": "TPOT", "itl_ms": "ITL", "e2e_latency_ms": "E2E Latency"}
THROUGHPUTS = {
    "request_throughput_req_s": "Request Throughput (req/s)",
    "output_throughput_tok_s": "Output Throughput (tok/s)",
    "total_throughput_tok_s": "Total Throughput (tok/s)",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_benchmark_arguments(parser)
    args = parser.parse_args(argv)
    validate_benchmark_args(parser, args)
    return args


def add_benchmark_arguments(parser, *, num_requests=128, repeat=1):
    parser.add_argument("--model", type=Path, default=Path("~/huggingface/Qwen3-0.6B"))
    parser.add_argument(
        "--engine-dir", type=Path, default=REPO_ROOT / "nanovllm",
        help="Path to the engine's Python package directory",
    )
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--max-num-batched-tokens", type=int, default=1024)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--num-requests", type=int, default=num_requests)
    parser.add_argument("--workload", choices=("fixed", "random"), default="fixed")
    parser.add_argument("--input-len", type=int, default=1024)
    parser.add_argument("--output-len", type=int, default=128)
    for kind in ("input", "output"):
        parser.add_argument(f"--min-{kind}-len", type=int, default=128)
        parser.add_argument(f"--max-{kind}-len", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--temperature", type=float, default=1e-5, help="Positive; default is near-greedy, not exact greedy")
    parser.add_argument("--warmup-requests", type=int, default=4)
    parser.add_argument("--repeat", type=int, default=repeat)
    parser.add_argument("--output-dir", type=Path, default=BENCHMARK_DIR / "results")


def validate_benchmark_args(parser, args, *, replay=False):
    for name in ("max_num_seqs", "max_num_batched_tokens", "max_model_len", "num_requests", "repeat"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.warmup_requests < 0:
        parser.error("--warmup-requests cannot be negative")
    if not 0 < args.gpu_memory_utilization <= 1:
        parser.error("--gpu-memory-utilization must be in (0, 1]")
    if not math.isfinite(args.temperature) or args.temperature <= 1e-10:
        parser.error("--temperature must be finite and > 1e-10 (the engine does not support greedy sampling)")
    if not replay:
        if not 0 <= args.seed <= 2**64 - args.repeat:
            parser.error("--seed and --seed + --repeat - 1 must fit in an unsigned 64-bit integer")
        try:
            validate_lengths(args, args.max_model_len)
        except ValueError as exc:
            parser.error(str(exc))


def length_ranges(args):
    if args.workload == "fixed":
        return (args.input_len, args.input_len), (args.output_len, args.output_len)
    return (args.min_input_len, args.max_input_len), (args.min_output_len, args.max_output_len)


def validate_lengths(args, context_limit):
    input_range, output_range = length_ranges(args)
    for kind, (minimum, maximum) in (("input", input_range), ("output", output_range)):
        if not 1 <= minimum <= maximum:
            raise ValueError(f"{kind} lengths must satisfy 1 <= min <= max")
    if input_range[1] + output_range[1] > context_limit:
        raise ValueError(f"maximum input + output length exceeds context limit ({context_limit})")


def generate_workload(args, num_requests, seed, vocab_size):
    rng = random.Random(seed)
    input_range, output_range = length_ranges(args)
    workload = []
    for _ in range(num_requests):
        input_len = rng.randint(*input_range)
        output_len = rng.randint(*output_range)
        prompt = [rng.randrange(vocab_size) for _ in range(input_len)]
        workload.append((prompt, output_len))
    return workload


def run_batch(llm, workload, params):
    if not llm.is_finished():
        raise RuntimeError("Benchmark requires an idle engine before each batch")
    for (prompt, _), sampling_params in zip(workload, params, strict=True):
        llm.add_request(prompt, sampling_params)
    outputs = {}
    while not llm.is_finished():
        finished, *_ = llm.step()
        for seq_id, token_ids, *metrics in finished:
            if seq_id in outputs:
                raise RuntimeError(f"Duplicate result for sequence {seq_id}")
            outputs[seq_id] = token_ids, metrics[0] if metrics else None
    if len(outputs) != len(workload):
        raise RuntimeError(f"Expected {len(workload)} results, got {len(outputs)}")
    return outputs


def measure_batch(llm, cuda, workload, params):
    cuda.synchronize()
    cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    outputs = run_batch(llm, workload, params)
    cuda.synchronize()
    elapsed = time.perf_counter() - started
    memory = {
        "peak_gpu_memory_allocated_mib": cuda.max_memory_allocated() / 2**20,
        "peak_gpu_memory_reserved_mib": cuda.max_memory_reserved() / 2**20,
    }
    return outputs, elapsed, memory


def describe(values):
    ordered = sorted(values)
    stats = {"mean": statistics.fmean(ordered) if ordered else None}
    for percentile in (50, 95, 99):
        value = None
        if ordered:
            rank = (len(ordered) - 1) * percentile / 100
            lower, upper = math.floor(rank), math.ceil(rank)
            value = ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)
        stats[f"p{percentile}"] = value
    return stats


def collect_metrics(workload, outputs, elapsed):
    requests = []
    all_itls = []
    for request_id, ((prompt, output_limit), seq_id) in enumerate(zip(workload, sorted(outputs), strict=True)):
        tokens, metrics = outputs[seq_id]
        count = len(tokens)
        ttft = metrics["time_to_first_token"] if metrics is not None else None
        tpot = metrics["time_per_output_token"] if metrics is not None and count >= 2 else None
        itls = metrics["inter_token_latencies"] if metrics is not None and count >= 2 else []
        e2e = ttft if count == 1 else None
        if ttft is not None and tpot is not None:
            e2e = ttft + (count - 1) * tpot
        all_itls.extend(itls)
        requests.append({
            "request_id": request_id,
            "input_tokens": len(prompt),
            "output_tokens": count,
            "requested_output_tokens": output_limit,
            "ttft_ms": ttft,
            "tpot_ms": tpot,
            "mean_itl_ms": statistics.fmean(itls) if itls else None,
            "max_itl_ms": max(itls) if itls else None,
            "e2e_latency_ms": e2e,
        })

    total_input = sum(row["input_tokens"] for row in requests)
    total_output = sum(row["output_tokens"] for row in requests)
    summary = {
        "elapsed_s": elapsed,
        "total_requests": len(requests),
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "request_throughput_req_s": len(requests) / elapsed,
        "output_throughput_tok_s": total_output / elapsed,
        "total_throughput_tok_s": (total_input + total_output) / elapsed,
    }
    for metric in LATENCIES:
        samples = all_itls if metric == "itl_ms" else [row[metric] for row in requests if row[metric] is not None]
        summary.update({f"{metric.removesuffix('_ms')}_{stat}_ms": value for stat, value in describe(samples).items()})
    return summary, requests


def across_runs(summaries):
    metrics = ("elapsed_s", *THROUGHPUTS, *(name for name in summaries[0] if name.endswith(("_ms", "_mib"))))
    result = {}
    for metric in metrics:
        values = [row[metric] for row in summaries if row[metric] is not None]
        result[metric] = {
            "count": len(values),
            "mean": statistics.fmean(values) if values else None,
            "stddev": statistics.stdev(values) if len(values) > 1 else None,
        }
    return result


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def save_results(output_dir, config, summaries, requests, aggregate):
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(tempfile.mkdtemp(prefix=f"offline_{stamp}_", dir=output_dir))
    write_csv(run_dir / "summary.csv", [config | row for row in summaries])
    write_csv(run_dir / "requests.csv", requests)
    report = {"config": config, "runs": summaries, "across_runs": aggregate}
    (run_dir / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return run_dir


def display(value):
    return "N/A" if value is None else f"{value:.3f}"


def print_run(summary, repeat):
    print(f"\nRun {summary['run']}/{repeat} (seed={summary['seed']})")
    print(f"Requests: {summary['total_requests']} | Input tokens: {summary['total_input_tokens']} | "
          f"Output tokens: {summary['total_output_tokens']}")
    print(f"Elapsed Time (s): {summary['elapsed_s']:.3f}")
    for key, label in THROUGHPUTS.items():
        print(f"{label}: {display(summary[key])}")
    print(f"\n{'Latency (ms)':<17}{'Mean':>11}{'P50':>11}{'P95':>11}{'P99':>11}")
    for key, label in LATENCIES.items():
        values = [display(summary[f"{key.removesuffix('_ms')}_{stat}_ms"]) for stat in ("mean", "p50", "p95", "p99")]
        print(f"{label:<17}" + "".join(f"{value:>11}" for value in values))
    print(f"\nPyTorch CUDA peak allocated (MiB): {summary['peak_gpu_memory_allocated_mib']:.2f}")
    print(f"PyTorch CUDA peak reserved (MiB):  {summary['peak_gpu_memory_reserved_mib']:.2f}")


def load_engine(engine_dir):
    package_dir = engine_dir.expanduser().resolve()
    initializer = package_dir / "__init__.py"
    if not initializer.is_file():
        raise SystemExit(f"nano-vLLM package not found: {initializer}")

    engine = sys.modules.get("nanovllm")
    if engine is None:
        # Both packages use absolute nanovllm imports internally.
        spec = spec_from_file_location("nanovllm", initializer, submodule_search_locations=[str(package_dir)])
        engine = module_from_spec(spec)
        sys.modules[spec.name] = engine
        spec.loader.exec_module(engine)
    elif Path(engine.__file__).resolve() != initializer:
        raise SystemExit(f"nano-vLLM already loaded from {engine.__file__}; run each engine in its own process")
    return engine.LLM, engine.SamplingParams


def main():
    args = parse_args()
    model = args.model.expanduser().resolve()
    if not model.is_dir():
        raise SystemExit(f"Model directory not found: {model} (use --model)")

    import torch

    LLM, SamplingParams = load_engine(args.engine_dir)

    print("========== Offline Benchmark ==========", flush=True)
    print(f"Model: {model.name} | Max seqs: {args.max_num_seqs} | Token budget: {args.max_num_batched_tokens}")
    input_range, output_range = length_ranges(args)
    print(f"Workload: {args.workload} | Requests: {args.num_requests} | "
          f"Input: {input_range[0]}-{input_range[1]} | Output: {output_range[0]}-{output_range[1]}", flush=True)
    llm = LLM(
        str(model),
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        tensor_parallel_size=1,
    )
    engine_config = llm.model_runner.config
    try:
        validate_lengths(args, engine_config.max_model_len)
    except ValueError as exc:
        raise SystemExit(f"Invalid workload: {exc}") from exc
    vocab_size = engine_config.hf_config.vocab_size
    config = vars(args) | {
        "model": str(model),
        "engine_dir": str(args.engine_dir.expanduser().resolve()),
        "output_dir": str(args.output_dir.expanduser().resolve()),
        "base_seed": args.seed,
        "effective_max_model_len": engine_config.max_model_len,
        "vocab_size": vocab_size,
        "ignore_eos": True,
    }
    del config["seed"]

    def sampling_params(workload):
        return [SamplingParams(temperature=args.temperature, max_tokens=limit, ignore_eos=True) for _, limit in workload]

    if args.warmup_requests:
        print(f"Warming up with {args.warmup_requests} requests...", flush=True)
        warmup = generate_workload(args, args.warmup_requests, f"offline-warmup:{args.seed}", vocab_size)
        torch.manual_seed(args.seed)
        run_batch(llm, warmup, sampling_params(warmup))

    summaries, requests = [], []
    for run_index in range(args.repeat):
        run = run_index + 1
        seed = args.seed + run_index
        workload = generate_workload(args, args.num_requests, seed, vocab_size)
        fingerprint = hashlib.sha256(json.dumps(workload, separators=(",", ":")).encode("ascii")).hexdigest()
        torch.manual_seed(seed)
        timestamp = datetime.now(timezone.utc).isoformat()
        print(f"Running batch {run}/{args.repeat}...", flush=True)
        outputs, elapsed, memory = measure_batch(llm, torch.cuda, workload, sampling_params(workload))
        summary, request_rows = collect_metrics(workload, outputs, elapsed)
        summary = {"run": run, "seed": seed, "timestamp_utc": timestamp, "workload_sha256": fingerprint} | summary | memory
        summaries.append(summary)
        requests.extend({"run": run, "seed": seed} | row for row in request_rows)
        print_run(summary, args.repeat)

    aggregate = across_runs(summaries)
    run_dir = save_results(args.output_dir, config, summaries, requests, aggregate)
    if args.repeat > 1:
        print("\nAcross runs (mean +/- sample SD)")
        for metric, label in ({"elapsed_s": "Elapsed Time (s)"} | THROUGHPUTS).items():
            stats = aggregate[metric]
            print(f"{label}: {display(stats['mean'])} +/- {display(stats['stddev'])}")
    print(f"\nResults: {run_dir}")


if __name__ == "__main__":
    main()
