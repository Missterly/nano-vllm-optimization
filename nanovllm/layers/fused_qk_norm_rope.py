import torch
import triton
import triton.language as tl


_HEAD_DIM = 128


@triton.jit
def _fused_qk_norm_rope_kernel(
    query,
    key,
    positions,
    query_weight,
    key_weight,
    cos_sin_cache,
    query_out,
    key_out,
    query_token_stride,
    query_head_stride,
    key_token_stride,
    key_head_stride,
    position_stride,
    cache_position_stride,
    NUM_QUERY_HEADS: tl.constexpr,
    NUM_KEY_HEADS: tl.constexpr,
    HALF_HEAD_DIM: tl.constexpr,
    EPS: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1)
    head_dim: tl.constexpr = 2 * HALF_HEAD_DIM

    if head < NUM_QUERY_HEADS:
        input_head = query + token * query_token_stride + head * query_head_stride
        weight = query_weight
        output_head = query_out + (token * NUM_QUERY_HEADS + head) * head_dim
    else:
        key_head = head - NUM_QUERY_HEADS
        input_head = key + token * key_token_stride + key_head * key_head_stride
        weight = key_weight
        output_head = key_out + (token * NUM_KEY_HEADS + key_head) * head_dim

    col = tl.arange(0, HALF_HEAD_DIM)
    left = tl.load(input_head + col).to(tl.float32)
    right = tl.load(input_head + HALF_HEAD_DIM + col).to(tl.float32)
    mean_square = tl.sum(left * left + right * right, axis=0) / head_dim
    inv_rms = tl.rsqrt(mean_square + EPS)
    left_weight = tl.load(weight + col).to(tl.float32)
    right_weight = tl.load(weight + HALF_HEAD_DIM + col).to(tl.float32)

    # The compiled RMSNorm multiplies in FP32 and returns BF16 to RoPE.
    left = (left * inv_rms * left_weight).to(tl.bfloat16).to(tl.float32)
    right = (right * inv_rms * right_weight).to(tl.bfloat16).to(tl.float32)

    position = tl.load(positions + token * position_stride).to(tl.int64)
    cache_row = cos_sin_cache + position * cache_position_stride
    cos = tl.load(cache_row + col)
    sin = tl.load(cache_row + HALF_HEAD_DIM + col)
    tl.store(output_head + col, left * cos - right * sin)
    tl.store(output_head + HALF_HEAD_DIM + col, right * cos + left * sin)


def fused_qk_norm_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    positions: torch.Tensor,
    query_weight: torch.Tensor,
    key_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply Qwen3's per-head RMSNorm and full RoPE in one inference kernel.

    Q and K are BF16 [tokens, heads, 128] CUDA views. Token and head strides
    may vary, and head elements must be contiguous. Positions must be in
    [0, max_position) for the FP32 [max_position, 1, 128] RoPE cache.
    The outputs are contiguous. RMSNorm rounding follows the compiled path.
    """
    if query.ndim != 3 or key.ndim != 3:
        raise ValueError("query and key must be [tokens, heads, 128] tensors")
    num_tokens, num_query_heads, query_head_dim = query.shape
    key_tokens, num_key_heads, key_head_dim = key.shape
    if key_tokens != num_tokens or query_head_dim != _HEAD_DIM or key_head_dim != _HEAD_DIM:
        raise ValueError("query and key must have matching token counts and head dimension 128")
    if num_query_heads == 0 or num_key_heads == 0:
        raise ValueError("query and key must each have at least one head")
    if positions.shape != (num_tokens,) or positions.dtype not in (torch.int32, torch.int64):
        raise ValueError("positions must contain one integer index per token")
    if query_weight.shape != (_HEAD_DIM,) or key_weight.shape != (_HEAD_DIM,):
        raise ValueError("each RMSNorm weight must have shape [128]")
    if cos_sin_cache.shape[1:] != (1, _HEAD_DIM) or cos_sin_cache.dtype != torch.float32:
        raise ValueError("cos_sin_cache must be an FP32 tensor of shape [max_position, 1, 128]")
    if any(tensor.dtype != torch.bfloat16 for tensor in (query, key, query_weight, key_weight)):
        raise ValueError("query, key, and RMSNorm weights must be BF16")
    if any(tensor.stride(-1) != 1 for tensor in (query, key, query_weight, key_weight, cos_sin_cache)):
        raise ValueError("head elements, weights, and RoPE cache columns must be contiguous")
    if not query.is_cuda or any(
        tensor.device != query.device for tensor in (key, positions, query_weight, key_weight, cos_sin_cache)
    ):
        raise ValueError("all inputs must be on the same CUDA device")

    query_out = torch.empty_like(query, memory_format=torch.contiguous_format)
    key_out = torch.empty_like(key, memory_format=torch.contiguous_format)
    if num_tokens == 0:
        return query_out, key_out

    _fused_qk_norm_rope_kernel[(num_tokens, num_query_heads + num_key_heads)](
        query,
        key,
        positions,
        query_weight,
        key_weight,
        cos_sin_cache,
        query_out,
        key_out,
        query.stride(0),
        query.stride(1),
        key.stride(0),
        key.stride(1),
        positions.stride(0),
        cos_sin_cache.stride(0),
        NUM_QUERY_HEADS=num_query_heads,
        NUM_KEY_HEADS=num_key_heads,
        HALF_HEAD_DIM=_HEAD_DIM // 2,
        EPS=eps,
        num_warps=1,
        enable_fp_fusion=False,
    )
    return query_out, key_out
