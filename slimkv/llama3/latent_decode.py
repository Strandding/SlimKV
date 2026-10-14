"""Backend selection and eager attention for reconstruction-free Llama decode."""
import math

import torch


def uses_latent_decode(config):
    return getattr(config, 'beacon_decode_backend', 'reconstruct') in ('latent', 'fa2')


def uses_fa2_decode(config):
    backend = getattr(config, 'beacon_decode_backend', 'reconstruct')
    return backend == 'fa2' or (backend == 'latent' and config._attn_implementation == 'flash_attention_2')


def eager_forward(q, k, v, scale, causal=False):
    """BSHD inputs, BSHD output and BHQ log-sum-exp; K/V ranks may differ."""
    batch, queries, heads, rank = q.shape
    keys, kv_heads = k.shape[1:3]
    if (k.shape[0] != batch or k.shape[-1] != rank or v.shape[:3] != k.shape[:3]
            or kv_heads < 1 or heads % kv_heads or keys < 1):
        raise ValueError('Invalid eager attention shapes or GQA head counts.')
    groups = heads // kv_heads
    # Group query heads so shared K/V need no expanded copies.
    grouped_q = q.transpose(1, 2).reshape(batch, kv_heads, groups * queries, rank)
    # Preserve FP32 reference arithmetic even inside a caller's autocast scope.
    with torch.autocast(device_type=q.device.type, enabled=False):
        scores = (grouped_q.float() @ k.transpose(1, 2).transpose(-1, -2).float()) * scale
        scores = scores.reshape(batch, heads, queries, keys)
        if causal:
            query_positions = torch.arange(queries, device=q.device) + keys - queries
            mask = torch.arange(keys, device=q.device)[None, :] > query_positions[:, None]
            scores = scores.masked_fill(mask, -math.inf)
        lse = torch.logsumexp(scores, dim=-1)
        probabilities = torch.softmax(scores, dim=-1)
        # Bottom-right causality can leave the first query with no visible keys.
        probabilities = torch.where(torch.isfinite(lse)[..., None], probabilities, 0.0)
        grouped_p = probabilities.reshape(batch, kv_heads, groups * queries, keys)
        output = grouped_p @ v.transpose(1, 2).float()
    output = output.reshape(batch, heads, queries, v.shape[-1]).transpose(1, 2)
    return output.to(q.dtype), lse
