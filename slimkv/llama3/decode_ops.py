"""Small inference-only Triton fusions; Torch remains the CPU/gradient fallback."""
from functools import lru_cache


@lru_cache(None)
def _kernels():
    try:
        from .decode_kernels import triton, rope_kernel, merge_kernel
    except ImportError:
        return None
    return triton, rope_kernel, merge_kernel


def fused_rope(q, k, rope, query_positions, raw_indices, sink, history):
    import torch
    if (not q.is_cuda or q.dtype not in (torch.bfloat16, torch.float16)
            or (torch.is_grad_enabled() and (q.requires_grad or k.requires_grad))):
        return None
    kernels = _kernels()
    if kernels is None:
        return None
    triton, kernel, _ = kernels
    oq, ok = q.new_empty(q.shape), k.new_empty(k.shape)
    block = 256
    grid = triton.cdiv(q.numel(), block) + triton.cdiv(k.numel(), block)
    kernel[(grid,)](q, k, rope.cos_cached, rope.sin_cached, query_positions,
                    raw_indices if raw_indices is not None else query_positions, oq, ok,
                    q.shape[1], k.shape[1], q.shape[2], k.shape[2], q.shape[3], q.shape[0],
                    *q.stride()[:3], *k.stride()[:3], query_positions.stride(0), sink, history,
                    BLOCK=block, num_warps=4, enable_fp_fusion=False)
    return oq, ok


def fused_merge(raw, raw_lse, latent, latent_lse):
    import torch
    if (not raw.is_cuda or raw.dtype not in (torch.bfloat16, torch.float16)
            or (torch.is_grad_enabled() and (raw.requires_grad or latent.requires_grad))):
        return None
    kernels = _kernels()
    if kernels is None:
        return None
    triton, _, kernel = kernels
    out = raw.new_empty(raw.shape)
    kernel[(triton.cdiv(raw.numel(), 256),)](
        raw, latent, raw_lse, latent_lse, out, raw.shape[1], raw.shape[2], raw.shape[3], raw.shape[0],
        *raw.stride()[:3], *latent.stride()[:3], *raw_lse.stride(), *latent_lse.stride(),
        BLOCK=256, num_warps=4, enable_fp_fusion=False)
    return out
