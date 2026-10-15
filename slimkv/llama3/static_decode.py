"""Whole-model FA2 decode graphs with append-only KV and device cache lengths.

Persistent Beacon views retain their independently trained K/V ranks. Padded
FA2 workspaces and graph allocations are additional memory, not KV savings.
"""
import importlib
import math
import os
import sys
from functools import lru_cache

import torch

from .fa2_decode import project_query, project_value, merge_attention
from .decode_ops import fused_rope
from .latent_decode import uses_fa2_decode
from .paired_decode import paired_attention


@lru_cache(None)
def cache_extension():
    path = os.environ.get('SLIMKV_FA2_DECODE_EXTENSION_DIR')
    if path and path not in sys.path:
        sys.path.insert(0, path)
    try:
        extension = importlib.import_module('slimkv_fa2_decode_320')
    except ImportError:
        return None
    return extension if getattr(extension, 'supports_cache_seqlens', False) else None


def _length(x):
    return 0 if x is None else x.shape[2]


def _rotate(x, rope, positions):
    cosine = rope.cos_cached[positions].to(x.dtype).unsqueeze(1)
    sine = rope.sin_cached[positions].to(x.dtype).unsqueeze(1)
    half = torch.cat([-x[..., x.shape[-1] // 2:], x[..., :x.shape[-1] // 2]], dim=-1)
    return x * cosine + half * sine


class StaticDecode:
    """One aligned B1 stream; unsupported calls retain the ordinary model path."""

    def __init__(self, model, capacity, raw_capacity=None):
        self.model = model
        self.capacity = capacity
        self.layers = []
        self.graphs = {}
        self.epoch = None
        device = model.device
        self.raw_index = torch.empty(1, dtype=torch.long, device=device)
        self.latent_index = torch.empty(1, dtype=torch.long, device=device)
        self.raw_cu = torch.zeros(2, dtype=torch.int32, device=device)
        self.latent_cu = torch.zeros(2, dtype=torch.int32, device=device)
        self.raw_lengths = self.raw_cu[1:]
        self.latent_lengths = self.latent_cu[1:]
        self.paired_lengths = torch.empty(2, dtype=torch.int32, device=device)
        self.paired_enabled = False
        self.sm_count = torch.cuda.get_device_properties(device).multi_processor_count if device.type == 'cuda' else 0
        self.query_cu = {q: torch.tensor([0, q], dtype=torch.int32, device=device) for q in (1, 2)}
        self.raw_capacity = max(model.config.beacon_window + model.config.beacon_sink_size + 1,
                                raw_capacity or 0)
        self.raw_bucket = self.raw_capacity
        for layer in model.model.layers:
            attn = layer.self_attn
            template = attn.q_proj.weight
            heads, dim = attn.num_key_value_heads, attn.head_dim
            rk, rv = attn.beacon_k_up_proj.in_features, attn.beacon_v_up_proj.in_features
            width = max(rk, rv)
            width = 320 if width > 256 else (width + 7) // 8 * 8
            raw_k = template.new_empty(1, heads, self.raw_capacity, dim)
            raw_v = torch.empty_like(raw_k)
            self.layers.append(dict(
                raw_k=raw_k, raw_v=raw_v, rotated_k=torch.empty_like(raw_k),
                latent_k=template.new_empty(1, 1, capacity, rk),
                latent_v=template.new_empty(1, 1, capacity, rv),
                padded_k=template.new_zeros(1, capacity, 1, width),
                padded_v=template.new_zeros(1, capacity, 1, width),
                width=width))
        self.extension = cache_extension()

    def attention(self, q, k, v, lengths, cumulative, scale, causal):
        if causal and self.paired_enabled:
            return paired_attention(self.extension.forward, q, k, v, scale,
                                    self.paired_lengths, self.sm_count)
        return self.extension.forward(q, k, v, scale, causal, 0, True, lengths)

    @staticmethod
    def eligible(model, ids, past, kwargs):
        return (ids.is_cuda and ids.shape[0] == 1 and ids.shape[1] in (1, 2)
                and not torch.is_grad_enabled() and uses_fa2_decode(model.config)
                and getattr(model.config, 'beacon_decode_graph', False)
                and getattr(model.config, 'beacon_decode_graph_full', True)
                and model.dtype == torch.bfloat16
                and model.config.beacon_sink_size in (0, 1)
                and all(max(layer.self_attn.beacon_k_up_proj.in_features,
                            layer.self_attn.beacon_v_up_proj.in_features) > 192
                        for layer in model.model.layers)
                and model.memory._fast_decode_active
                and not kwargs.get('output_attentions') and not kwargs.get('output_hidden_states')
                and kwargs.get('labels') is None and kwargs.get('inputs_embeds') is None
                and kwargs.get('return_dict', True) is not False
                and all(type(layer.self_attn.rotary_emb).__name__ in
                        ('RotaryEmbedding', 'LinearScalingRotaryEmbedding', 'Llama3RotaryEmbedding')
                        for layer in model.model.layers)
                and all(past[0][0].shape[2] + ids.shape[1] <= layer.self_attn.rotary_emb.max_seq_len_cached
                        for layer in model.model.layers)
                and all(isinstance(layer.self_attn.beacon_k_up_proj, torch.nn.Linear)
                        and isinstance(layer.self_attn.beacon_v_up_proj, torch.nn.Linear)
                        for layer in model.model.layers)
                and cache_extension() is not None)

    def initialize(self, past):
        for state, layer, (pk, pv, _, _) in zip(self.layers, self.model.model.layers, past):
            sink, raw, history = _length(pk.sink), _length(pk.raw), _length(pk.history)
            for kind, cache in [('k', pk), ('v', pv)]:
                raw_buffer = state['raw_' + kind]
                for offset, tensor in [(0, cache.sink), (sink, cache.raw)]:
                    if tensor is not None:
                        raw_buffer[:, :, offset:offset + _length(tensor)].copy_(tensor)
                latent_buffer = state['latent_' + kind]
                for offset, tensor in [(0, cache.history), (history, cache.window)]:
                    if tensor is not None:
                        latent_buffer[:, :, offset:offset + _length(tensor)].copy_(tensor)
                count = history + _length(cache.window)
                if count:
                    state['padded_' + kind][:, :count, :, :latent_buffer.shape[-1]].copy_(
                        latent_buffer[:, :, :count].transpose(1, 2))
            positions = torch.cat([
                torch.arange(sink, device=state['raw_k'].device),
                pk.window_raw_indices + sink + history if raw else
                torch.empty(0, dtype=torch.long, device=state['raw_k'].device)])
            if sink + raw:
                state['rotated_k'][:, :, :sink + raw].copy_(
                    _rotate(state['raw_k'][:, :, :sink + raw], layer.self_attn.rotary_emb,
                            positions.unsqueeze(0)))
        self.epoch = self.model.memory._static_decode_epoch

    def forward(self, ids, positions, refresh_sink=False, return_normalized_hidden=False):
        model = self.model
        size = ids.shape[1] - 1
        hidden = model.model.embed_tokens(ids[:, :1])
        if size:
            hidden = torch.cat([hidden, model.model.beacon_embed_tokens(ids[:, 1:] - model.config.vocab_size)], dim=1)
        for state, layer in zip(self.layers, model.model.layers):
            residual = hidden
            normalized = layer.input_layernorm(hidden)
            attn = layer.self_attn
            q = attn.q_proj(normalized)
            if size and 'q' in attn.config.beacon_param:
                q = torch.cat([q[:, :1], attn.beacon_q_proj(normalized[:, 1:])], dim=1)
            q = q.reshape(1, 1 + size, attn.num_heads, attn.head_dim).transpose(1, 2)
            raw_k = attn.k_proj(normalized[:, :1]).reshape(1, 1, attn.num_key_value_heads,
                                                        attn.head_dim).transpose(1, 2)
            raw_v = attn.v_proj(normalized[:, :1]).reshape_as(raw_k.transpose(1, 2)).transpose(1, 2)
            state['raw_k'].index_copy_(2, self.raw_index, raw_k)
            state['raw_v'].index_copy_(2, self.raw_index, raw_v)
            unrotated_k = raw_k
            rotated = fused_rope(q, raw_k, attn.rotary_emb, positions, None, 0, 0)
            if rotated is None:
                q, raw_k = _rotate(q, attn.rotary_emb, positions), _rotate(raw_k, attn.rotary_emb, positions[:, :1])
            else:
                q, raw_k = rotated
            state['rotated_k'].index_copy_(2, self.raw_index, raw_k)
            uk, uv = attn.beacon_k_up_proj, attn.beacon_v_up_proj
            if size:
                kd = attn.beacon_kv_down_proj if attn._use_shared_kv_down_proj() else attn.beacon_k_down_proj
                vd = attn.beacon_kv_down_proj if attn._use_shared_kv_down_proj() else attn.beacon_v_down_proj
                kl, vl = kd(normalized[:, 1:]).unsqueeze(1), vd(normalized[:, 1:]).unsqueeze(1)
                state['latent_k'].index_copy_(2, self.latent_index, kl)
                state['latent_v'].index_copy_(2, self.latent_index, vl)
                state['padded_k'][..., :uk.in_features].index_copy_(1, self.latent_index, kl.transpose(1, 2))
                state['padded_v'][..., :uv.in_features].index_copy_(1, self.latent_index, vl.transpose(1, 2))
            scale = 1.0 / math.sqrt(attn.head_dim)
            raw_out, raw_lse = self.attention(
                q.transpose(1, 2), state['rotated_k'][:, :, :self.raw_bucket].transpose(1, 2),
                state['raw_v'][:, :, :self.raw_bucket].transpose(1, 2),
                self.raw_lengths, self.raw_cu, scale, False)
            latent_q = project_query(q, uk.weight, attn.num_key_value_heads).transpose(1, 2)
            width = state['width']
            if latent_q.shape[-1] != width:
                latent_q = torch.nn.functional.pad(latent_q, (0, width - latent_q.shape[-1]))
            latent_out, latent_lse = self.attention(
                latent_q, state['padded_k'], state['padded_v'],
                self.latent_lengths, self.latent_cu, scale, bool(size))
            latent_out = project_value(latent_out[..., :uv.in_features].transpose(1, 2),
                                       uv.weight, attn.num_key_value_heads)
            if uk.bias is not None:
                bias = uk.bias.reshape(attn.num_key_value_heads, attn.head_dim)
                bias = bias.repeat_interleave(attn.num_key_value_groups, dim=0)
                latent_lse = latent_lse + (q.float() * bias[None, :, None].float()).sum(-1) * scale
            if uv.bias is not None:
                bias = uv.bias.reshape(attn.num_key_value_heads, attn.head_dim)
                latent_out = latent_out + bias.repeat_interleave(attn.num_key_value_groups, dim=0)[None, :, None]
            merged = merge_attention(raw_out.transpose(1, 2), raw_lse, latent_out, latent_lse)
            merged = merged.transpose(1, 2).reshape(1, 1 + size, attn.hidden_size)
            output = attn.o_proj(merged)
            if size and 'o' in attn.config.beacon_param:
                # Match o_proj_with_beacon's GEMM shapes before selecting rows.
                beacon_output = attn.beacon_o_proj(merged)
                output = torch.cat([output[:, :1], beacon_output[:, 1:]], dim=1)
            if refresh_sink and model.config.beacon_sink_size:
                # Preserve Memory's existing first-window sink update exactly.
                state['raw_k'][:, :, :1].copy_(unrotated_k)
                state['raw_v'][:, :, :1].copy_(raw_v)
                zero_positions = positions[:, :1] * 0
                state['rotated_k'][:, :, :1].copy_(_rotate(unrotated_k, attn.rotary_emb, zero_positions))
            hidden = layer._decode_tail(output, residual)
        hidden = model.model.norm(hidden)
        return hidden if return_normalized_hidden else model.lm_head(hidden).float()

    def run(self, ids, positions, past, return_normalized_hidden=False):
        pk = past[0][0]
        sink, history = _length(pk.sink), _length(pk.history)
        raw = sink + _length(pk.raw)
        latent = history + _length(pk.window)
        size = ids.shape[1] - 1
        self.raw_index.fill_(raw)
        self.latent_index.fill_(latent)
        self.raw_lengths.fill_(raw + 1)
        self.raw_bucket = min(self.raw_capacity, (raw + 128) // 128 * 128)
        self.latent_lengths.fill_(latent + size)
        self.paired_enabled = self.sm_count > 0 and latent + size > 1
        if size and self.paired_enabled:
            self.paired_lengths.fill_(latent + size)
            self.paired_lengths[:1].fill_(latent + size - 1)
        if self.epoch != self.model.memory._static_decode_epoch:
            self.initialize(past)
        positions = positions[:, -ids.shape[1]:]
        refresh_sink = pk.history is None and not self.model.memory._decode_uncompressed
        graph_key = (size, refresh_sink, self.raw_bucket, return_normalized_hidden,
                     bool(getattr(self.model.config, 'beacon_decode_compile', False)), self.paired_enabled)
        if graph_key not in self.graphs:
            buffers = (ids.clone(), positions.clone())
            sink_snapshot = [(state[name][:, :, :1], state[name][:, :, :1].clone())
                             for state in self.layers for name in ('raw_k', 'raw_v', 'rotated_k')]

            def restore_sink():
                if refresh_sink:
                    for target, saved in sink_snapshot:
                        target.copy_(saved)

            current = torch.cuda.current_stream(ids.device)
            stream = torch.cuda.Stream(device=ids.device)
            stream.wait_stream(current)
            with torch.cuda.stream(stream):
                for _ in range(3):
                    restore_sink()
                    self.forward(*buffers, refresh_sink=refresh_sink,
                                 return_normalized_hidden=return_normalized_hidden)
            current.wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.stream(stream):
                restore_sink()
            with torch.cuda.graph(graph, stream=stream):
                output = self.forward(*buffers, refresh_sink=refresh_sink,
                                      return_normalized_hidden=return_normalized_hidden)
            current.wait_stream(stream)
            restore_sink()
            self.graphs[graph_key] = graph, buffers, output
        graph, buffers, output = self.graphs[graph_key]
        buffers[0].copy_(ids)
        buffers[1].copy_(positions)
        graph.replay()
        self.commit(past, size)
        # Returned logits own their storage, like the ordinary forward result.
        return output.clone()

    def commit(self, past, size):
        """Publish valid-rank views and advance window metadata after KV writes."""
        pk = past[0][0]
        sink, history = _length(pk.sink), _length(pk.history)
        raw = sink + _length(pk.raw)
        latent = history + _length(pk.window)
        memory = self.model.memory
        for i, state in enumerate(self.layers):
            memory.sink_activations[i] = (state['raw_k'][:, :, :sink], state['raw_v'][:, :, :sink])
            if memory._is_full_window:
                memory.beacon_activations[i] = (state['latent_k'][:, :, :latent + size],
                                                state['latent_v'][:, :, :latent + size])
                memory.raw_activations[i] = (None, None)
                memory.window_beacon_activations[i] = (None, None)
            else:
                memory.beacon_activations[i] = tuple(state['latent_' + x][:, :, :history] if history else None for x in ('k', 'v'))
                memory.raw_activations[i] = (state['raw_k'][:, :, sink:raw + 1], state['raw_v'][:, :, sink:raw + 1])
                memory.window_beacon_activations[i] = tuple(state['latent_' + x][:, :, history:latent + size]
                                                          if latent + size > history else None for x in ('k', 'v'))
        memory._window_roles = None if memory._is_full_window else memory._fast_window_roles
        memory._window_raw_indices = None if memory._is_full_window else memory._fast_window_raw_indices
