"""Exact inference-only prefill bookkeeping and rotary fusion."""
from dataclasses import dataclass
from functools import lru_cache

import torch


@dataclass
class PrefillMetadata:
    raw_indices: torch.Tensor
    beacon_indices: torch.Tensor
    roles: torch.Tensor
    positions: torch.Tensor
    replicas: bool


def prepare_metadata(model, ids, mask, positions, past, labels):
    config = model.config
    if (not getattr(config, 'beacon_prefill_fast', True) or model.training
            or torch.is_grad_enabled() or labels is not None or not ids.is_cuda
            or ids.shape[1] <= 1 or getattr(config, '_slimkv_decode_step', False)
            or config._attn_implementation != 'flash_attention_2' or mask is not None
            or config.beacon_pos != 'interleave' or config.beacon_parallel_window != 1
            or config.beacon_window != config.beacon_stride
            or not model.memory.latent_cache_active or past[0][2] < 0
            or not getattr(config, 'beacon_skip_rope_for_beacon_k', False)
            or getattr(config, 'beacon_chunk_fixed_rope_for_beacon_k', False)
            or getattr(config, 'beacon_enhance_k_pos_residual', False)
            or getattr(config, 'raw_nokrope', False)
            or getattr(config, 'beacon_skip_rope_for_raw_k', False)):
        return None
    roles = past[0][3]
    current = roles[-ids.shape[1]:]
    return PrefillMetadata((current == 0).nonzero().flatten(),
                           (current != 0).nonzero().flatten(), roles, positions,
                           bool((current == 2).any()))


@lru_cache(None)
def _kernel():
    try:
        from .prefill_kernels import triton, prefill_rope_kernel
    except ImportError:
        return None
    return triton, prefill_rope_kernel


def fused_prefill_rope(q, k, rope, metadata, sink):
    # Memory constructs these positions as arange(KLEN), repeated per batch.
    # Its known upper bound replaces a per-layer GPU scalar synchronization.
    if (not q.is_cuda or not k.is_cuda or q.dtype != torch.bfloat16 or k.dtype != q.dtype
            or q.shape[-1] != k.shape[-1] or q.shape[-1] % 2
            or q.numel() >= 2**31 or k.numel() >= 2**31
            or type(rope).__name__ not in ('RotaryEmbedding', 'LinearScalingRotaryEmbedding', 'Llama3RotaryEmbedding')
            or metadata.positions.shape != (q.shape[0], k.shape[2])
            or k.shape[2] > rope.max_seq_len_cached
            or metadata.roles.numel() > k.shape[2]
            or q.stride(-1) != 1 or k.stride(-1) != 1):
        return None
    kernels = _kernel()
    if kernels is None:
        return None
    triton, kernel = kernels
    oq, ok = torch.empty_like(q), torch.empty_like(k)
    block = 1024
    grid = triton.cdiv(q.numel(), block) + triton.cdiv(k.numel(), block)
    kernel[(grid,)](q, k, rope.cos_cached, rope.sin_cached,
                    metadata.positions, metadata.roles, oq, ok,
                    q.shape[0], q.shape[1], k.shape[1], q.shape[2], k.shape[2], q.shape[-1],
                    *q.stride()[:3], *k.stride()[:3], *oq.stride()[:3], *ok.stride()[:3],
                    metadata.positions.stride(0), metadata.roles.numel(), sink,
                    BLOCK=block, num_warps=4, enable_fp_fusion=False)
    return oq, ok
