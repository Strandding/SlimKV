"""Inference-only global latent attention with raw/Beacon softmax merging."""
import importlib
import math
import os
import sys

import torch
import torch.nn.functional as F

from ..memory import cat_tensor
from .memory import ReconstructionCache, ReconstructionCacheUpdate
from .latent_decode import eager_forward, uses_latent_decode, uses_fa2_decode


def flash_forward(q, k, v, scale, causal=False):
    """BSHD tensors; preserve the original head scale despite transient padding."""
    width = max(q.shape[-1], k.shape[-1], v.shape[-1])
    if width > 320:
        raise ValueError(f'FA2 decode supports global ranks up to 320, found {width}')
    width = 320 if width > 256 else (width + 7) // 8 * 8
    # Dense FA2 accepts arbitrary leading strides; avoid copying raw K/V when
    # their existing last dimension already matches the selected width.
    q, k, v = [F.pad(x, (0, width - x.shape[-1])) if x.shape[-1] != width else x
               for x in (q, k, v)]
    if width == 320:
        path = os.environ.get('SLIMKV_FA2_DECODE_EXTENSION_DIR')
        if path and path not in sys.path:
            sys.path.insert(0, path)
        extension = importlib.import_module('slimkv_fa2_decode_320')
        out, lse = extension.forward(q, k, v, scale, causal)
    else:
        from flash_attn.flash_attn_interface import _flash_attn_forward
        out, lse, _, _ = _flash_attn_forward(q, k, v, 0.0, scale, causal, -1, -1, 0.0, None, False)
    if causal and k.shape[1] < q.shape[1]:
        # FA2 uses +inf for fully masked rows; a zero-mass branch has log Z=-inf.
        lse = lse.clone()
        lse[..., :q.shape[1] - k.shape[1]] = -math.inf
    return out, lse


def project_query(q, weight, kv_heads):
    """Q in BHQD; global U_K slices follow the original GQA head grouping."""
    batch, heads, length, dim = q.shape
    rank = weight.shape[-1]
    grouped = q.reshape(batch, kv_heads, heads // kv_heads * length, dim)
    return (grouped @ weight.reshape(kv_heads, dim, rank)).reshape(batch, heads, length, rank)


def project_value(output, weight, kv_heads):
    """Latent PV in BHQR; apply each original KV head's global U_V slice."""
    batch, heads, length, rank = output.shape
    dim = weight.shape[0] // kv_heads
    grouped = output.reshape(batch, kv_heads, heads // kv_heads * length, rank)
    return (grouped @ weight.reshape(kv_heads, dim, rank).transpose(-1, -2)).reshape(batch, heads, length, dim)


def merge_attention(raw_out, raw_lse, latent_out, latent_lse):
    """Outputs in BHQD and LSE in BHQ; merge before rounding to model dtype."""
    from .decode_ops import fused_merge
    fused = fused_merge(raw_out, raw_lse, latent_out, latent_lse)
    if fused is not None:
        return fused
    total = torch.logaddexp(raw_lse, latent_lse)
    raw_mass = torch.exp(raw_lse - total).unsqueeze(-1)
    latent_mass = torch.exp(latent_lse - total).unsqueeze(-1)
    return (raw_mass * raw_out.float() + latent_mass * latent_out.float()).to(raw_out.dtype)


def eligible(attention, hidden_states, past_key_value, output_attentions):
    return (getattr(attention.config, '_slimkv_decode_step', False)
            and uses_latent_decode(attention.config)
            and isinstance(past_key_value[0], ReconstructionCache)
            and not attention.training and not output_attentions
            and hidden_states.shape[0] > 0 and hidden_states.shape[1] in (1, 2))


def decode(attention, hidden, position_ids, past_key_value, forward=None):
    """One raw token, optionally followed by its newly inserted Beacon."""
    pk, pv, size, indices = past_key_value
    if forward is None:
        forward = flash_forward if uses_fa2_decode(attention.config) else eager_forward
    batch, length, _ = hidden.shape
    if size not in (0, 1) or length != size + 1:
        raise ValueError('Latent decode expects one raw token followed by at most one Beacon.')
    uk, uv = attention.beacon_k_up_proj, attention.beacon_v_up_proj
    if not isinstance(uk, torch.nn.Linear) or not isinstance(uv, torch.nn.Linear):
        raise ValueError('Latent decode currently requires globally coupled Linear up projections.')
    raw_hidden = hidden[:, :1]
    q = attention.q_proj(hidden)
    new_k = attention.k_proj(raw_hidden).reshape(batch, 1, attention.num_key_value_heads,
                                                attention.head_dim).transpose(1, 2)
    new_v = attention.v_proj(raw_hidden).reshape_as(new_k.transpose(1, 2)).transpose(1, 2)
    kl = vl = None
    if size:
        if 'q' in attention.config.beacon_param:
            q = torch.cat([q[:, :1], attention.beacon_q_proj(hidden[:, 1:])], dim=1)
        kd = attention.beacon_kv_down_proj if attention._use_shared_kv_down_proj() else attention.beacon_k_down_proj
        vd = attention.beacon_kv_down_proj if attention._use_shared_kv_down_proj() else attention.beacon_v_down_proj
        kl, vl = kd(hidden[:, 1:]).unsqueeze(1), vd(hidden[:, 1:]).unsqueeze(1)
    update = ReconstructionCacheUpdate(new_k, new_v, size, indices, kl, vl)
    update.raw_only = True
    q = q.reshape(batch, length, attention.num_heads, attention.head_dim).transpose(1, 2)
    raw_k = cat_tensor([pk.sink, pk.raw, new_k], dim=2)
    raw_v = cat_tensor([pv.sink, pv.raw, new_v], dim=2)
    latent_k = cat_tensor([pk.history, pk.window, kl], dim=2)
    latent_v = cat_tensor([pv.history, pv.window, vl], dim=2)
    sink_len = 0 if pk.sink is None else pk.sink.shape[2]
    history_len = 0 if pk.history is None else pk.history.shape[2]
    rope = attention.rotary_emb
    if (type(rope).__name__ in ('RotaryEmbedding', 'LinearScalingRotaryEmbedding', 'Llama3RotaryEmbedding')
            and position_ids.shape[-1] <= rope.max_seq_len_cached):
        # Full-coverage positions are arange(total_length), so the length is a
        # CPU-known upper bound. No per-layer CUDA max().item() is necessary.
        from ..modeling_utils import rotate_half

        def rotate(x, ids):
            cosine = rope.cos_cached[ids].to(x.dtype).unsqueeze(1)
            sine = rope.sin_cached[ids].to(x.dtype).unsqueeze(1)
            return x * cosine + rotate_half(x) * sine

        from .decode_ops import fused_rope
        fused = fused_rope(q, raw_k, rope, position_ids[:, -length:], pk.window_raw_indices,
                           sink_len, history_len)
        if fused is not None:
            q, raw_k = fused
        else:
            positions = [position_ids[0, :sink_len]]
            if pk.window_raw_indices is not None:
                positions.append(pk.window_raw_indices + sink_len + history_len)
            positions.append(position_ids[0, -length:][:1])
            raw_positions = torch.cat(positions).unsqueeze(0).expand(batch, -1)
            q, raw_k = rotate(q, position_ids[:, -length:]), rotate(raw_k, raw_positions)
    else:
        # Other RoPE variants can add query temperature or change frequencies.
        # Preserve their API with dummy keys carrying the actual Q positions.
        positions = [position_ids[0, :sink_len]]
        if pk.window_raw_indices is not None:
            positions.append(pk.window_raw_indices + sink_len + history_len)
        positions.append(position_ids[0, -length:][:1])
        raw_positions = torch.cat(positions).unsqueeze(0).expand(batch, -1)
        rope_positions = torch.cat([raw_positions, position_ids[:, -length:]], dim=1)
        dummy = new_k.new_zeros(batch, attention.num_key_value_heads, length, attention.head_dim)
        q, rotated = rope(q, torch.cat([raw_k, dummy], dim=2), rope_positions)
        raw_k = rotated[:, :, :-length]
    scale = 1.0 / math.sqrt(attention.head_dim)
    raw_out, raw_lse = forward(q.transpose(1, 2), raw_k.transpose(1, 2),
                               raw_v.transpose(1, 2), scale, False)
    raw_out = raw_out[..., :attention.head_dim].transpose(1, 2)
    if latent_k is None:
        output = raw_out
    else:
        latent_q = project_query(q, uk.weight, attention.num_key_value_heads)
        latent_out, latent_lse = forward(latent_q.transpose(1, 2), latent_k.transpose(1, 2),
                                         latent_v.transpose(1, 2), scale, bool(size))
        latent_out = project_value(latent_out[..., :uv.in_features].transpose(1, 2),
                                   uv.weight, attention.num_key_value_heads)
        if uk.bias is not None:
            bias = uk.bias.reshape(attention.num_key_value_heads, attention.head_dim)
            bias = bias.repeat_interleave(attention.num_key_value_groups, dim=0)
            latent_lse = latent_lse + (q.float() * bias[None, :, None].float()).sum(-1) * scale
        if uv.bias is not None:
            bias = uv.bias.reshape(attention.num_key_value_heads, attention.head_dim)
            latent_out = latent_out + bias.repeat_interleave(attention.num_key_value_groups, dim=0)[None, :, None]
        output = merge_attention(raw_out, raw_lse, latent_out, latent_lse)
    output = output.transpose(1, 2).reshape(batch, length, attention.hidden_size)
    return attention.o_proj_with_beacon(output, size, indices), None, update
