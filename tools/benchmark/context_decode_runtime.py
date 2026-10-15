"""Fixed-context B1 decode with equivalent graph and cache policies for every route.

This runner specializes to a completed Beacon window and at most one window of
continuation. It does not replace general generation, training or scheduling.
"""
from contextlib import nullcontext
from functools import lru_cache
from types import SimpleNamespace
from typing import Optional
import math

import torch

from slimkv.llama3.static_decode import _rotate, cache_extension
from slimkv.llama3.fa2_decode import project_query, project_value, merge_attention
from slimkv.llama3.paired_decode import paired_attention

ATTENTION = 'slimkv_bench/attention'
RECONSTRUCTION = 'slimkv_bench/kv_reconstruction'
ROPE = 'slimkv_bench/rope'


def apply_rope(q, k, rope, qpositions, kpositions, roles=None, profile=False):
    if profile and q.is_cuda:
        return profile_ops()[0](q, k, rope.cos_cached, rope.sin_cached, qpositions, kpositions, roles)
    if q.is_cuda and q.dtype == torch.bfloat16:
        from tools.benchmark.context_decode_kernels import triton, rotary_kernel
        oq, ok = torch.empty_like(q), torch.empty_like(k)
        block = 256
        grid = triton.cdiv(q.numel(), block)+triton.cdiv(k.numel(), block)
        rotary_kernel[(grid,)](q, k, rope.cos_cached, rope.sin_cached,
                              qpositions, kpositions, roles if roles is not None else kpositions,
                              oq, ok, q.shape[1], k.shape[1], q.shape[2], k.shape[2], q.shape[3],
                              *q.stride()[1:3], *k.stride()[1:3],
                              *oq.stride()[1:3], *ok.stride()[1:3], roles is not None,
                              BLOCK=block, num_warps=4, enable_fp_fusion=False)
        return oq, ok
    oq, ok = _rotate(q, rope, qpositions), _rotate(k, rope, kpositions)
    if roles is not None:
        ok = torch.where(roles[None, None, :, None], k, ok)
    return oq, ok


@lru_cache(None)
def profile_ops():
    # Profiling-only Torch operators give driver-launched Triton/pybind kernels
    # a CPU correlation as well as the enclosing GPU annotation. Timed Graph
    # replay uses the same kernels directly, without these Python wrappers.
    if hasattr(torch.ops.slimkv_benchmark, 'rotary'):
        return torch.ops.slimkv_benchmark.rotary.default, torch.ops.slimkv_benchmark.attention.default

    @torch.library.custom_op('slimkv_benchmark::rotary', mutates_args=(), device_types='cuda')
    def rotary(q: torch.Tensor, k: torch.Tensor, cosine: torch.Tensor, sine: torch.Tensor,
               qpositions: torch.Tensor, kpositions: torch.Tensor,
               roles: Optional[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        return apply_rope(q, k, SimpleNamespace(cos_cached=cosine, sin_cached=sine),
                          qpositions, kpositions, roles)

    @torch.library.custom_op('slimkv_benchmark::attention', mutates_args=(), device_types='cuda')
    def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, lengths: torch.Tensor,
                  scale: float, causal: bool) -> tuple[torch.Tensor, torch.Tensor]:
        out, lse = cache_extension().forward(q, k, v, scale, causal, 0, True, lengths)
        return out, lse
    return rotary, attention


def count(x):
    return 0 if x is None else x.shape[2]


@lru_cache(None)
def profiled_pair():
    if hasattr(torch.ops.slimkv_benchmark, 'paired_attention'):
        return torch.ops.slimkv_benchmark.paired_attention.default

    @torch.library.custom_op('slimkv_benchmark::paired_attention', mutates_args=(), device_types='cuda')
    def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                  lengths: torch.Tensor, scale: float, splits: int) -> tuple[torch.Tensor, torch.Tensor]:
        return tuple(cache_extension().forward(q, k, v, scale, False, splits, True, lengths))
    return attention


class OptimizedDecodeRuntime:
    """Shared complete-model CUDA Graph execution, with per-route numerical checks."""

    def __init__(self, model, mode, decode_steps, memory=None, raw_cache=None,
                 graph=True, compile_tail=False):
        self.model, self.mode = model, mode
        self.decode_steps = decode_steps
        self.use_graph, self.compile_tail = graph, compile_tail
        self.graphs = {}
        self.pool = torch.cuda.graph_pool_handle() if graph and model.device.type == 'cuda' else None
        self.scope = False
        self.extension = cache_extension() if model.device.type == 'cuda' else None
        self.sm_count = torch.cuda.get_device_properties(model.device).multi_processor_count if self.extension else 0
        self.paired_lengths = torch.empty(2, dtype=torch.int32, device=model.device)
        if model.device.type == 'cuda' and self.extension is None:
            raise RuntimeError('Optimized comparison requires the FA2 decode extension with device cache lengths.')
        self.token = torch.empty((1, 1), dtype=torch.long, device=model.device)
        self.index = self.token.new_empty(1)
        self.positions = {q: self.token.new_empty(1, q) for q in (1, 2)}
        self.lengths = torch.empty(1, dtype=torch.int32, device=model.device)
        self.layers = []
        self.memory = memory
        self.raw_cache = raw_cache
        if mode == 'full_kv':
            self.prefix = raw_cache.get_seq_length()
            self.ratio, self.history, self.sink = 0, 0, 0
            capacity = (self.prefix + decode_steps + 127) // 128 * 128
            for k, v in zip(raw_cache.key_cache, raw_cache.value_cache):
                shape = (*k.shape[:2], capacity, k.shape[-1])
                self.layers.append({'k': k.new_empty(shape), 'v': v.new_empty(shape)})
        else:
            if memory.raw_activations[0][0] is not None or memory.window_beacon_activations[0][0] is not None:
                raise ValueError('Optimized comparison requires a completed Beacon window.')
            if decode_steps > model.config.beacon_window:
                raise ValueError('Continuation must fit in one Beacon window.')
            self.ratio = memory.config.beacon_ratio[0]
            self.history = count(memory.beacon_activations[0][0])
            self.sink = count(memory.sink_activations[0][0])
            self.prefix = self.history + self.sink
            for layer in model.model.layers:
                attn = layer.self_attn
                template = attn.q_proj.weight
                heads, dim = attn.num_key_value_heads, attn.head_dim
                rk, rv = attn.beacon_k_up_proj.in_features, attn.beacon_v_up_proj.in_features
                width = max(rk, rv)
                width = 320 if width > 256 else (width + 7) // 8 * 8
                raw_cap = (self.sink + decode_steps + 127)//128*128
                self.raw_capacity = raw_cap
                latent_cap = (self.history + (decode_steps+self.ratio-1)//self.ratio + 63)//64*64
                capacity = (self.prefix + decode_steps + (decode_steps+self.ratio-1)//self.ratio + 127)//128*128
                state = {'raw_k': template.new_zeros(1, heads, raw_cap, dim),
                         'raw_v': template.new_zeros(1, heads, raw_cap, dim),
                         'rotated_k': template.new_zeros(1, heads, raw_cap, dim),
                         'latent_k': template.new_zeros(1, 1, latent_cap, rk),
                         'latent_v': template.new_zeros(1, 1, latent_cap, rv),
                         'padded_k': template.new_zeros(1, latent_cap, 1, width),
                         'padded_v': template.new_zeros(1, latent_cap, 1, width),
                         'width': width}
                if mode == 'reconstruct':
                    state.update(k=template.new_empty(1, heads, capacity, dim),
                                 v=template.new_empty(1, heads, capacity, dim))
                self.layers.append(state)
            self.raw_positions = self.token.new_zeros(self.raw_capacity)
            self.latent_lengths = torch.empty_like(self.lengths)
            self.latent_index = self.token.new_empty(1)
            self.raw_slots = torch.arange(decode_steps, device=model.device)
            self.raw_slots = self.raw_slots + self.raw_slots//self.ratio + self.prefix
            self.beacon_slots = torch.arange((decode_steps+self.ratio-1)//self.ratio, device=model.device)
            self.beacon_slots = self.prefix + (self.beacon_slots+1)*(self.ratio+1)-1
            self.current_slots = {q: self.token.new_empty(q) for q in (1, 2)}
            self.key_positions = torch.arange(capacity, device=model.device)[None]
            roles = torch.zeros(capacity, dtype=torch.bool, device=model.device)
            roles[self.sink:self.prefix] = True
            roles[self.beacon_slots] = True
            self.roles = roles
        self.reset()

    def marked(self, label):
        return torch.profiler.record_function(label) if self.scope else nullcontext()

    def reset(self):
        self.step_index = 0
        if self.mode == 'full_kv':
            for state, k, v in zip(self.layers, self.raw_cache.key_cache, self.raw_cache.value_cache):
                state['k'][:, :, :self.prefix].copy_(k[:, :, :self.prefix])
                state['v'][:, :, :self.prefix].copy_(v[:, :, :self.prefix])
        else:
            for state, sink, history, layer in zip(self.layers, self.memory.sink_activations,
                                                  self.memory.beacon_activations, self.model.model.layers):
                if self.sink:
                    state['rotated_k'][:, :, :self.sink].copy_(_rotate(
                        sink[0], layer.self_attn.rotary_emb,
                        torch.arange(self.sink, device=self.token.device)[None]))
                for name, j in [('k', 0), ('v', 1)]:
                    if self.sink:
                        state['raw_'+name][:, :, :self.sink].copy_(sink[j])
                    flat = history[j].transpose(1, 2).reshape(1, self.history, -1).unsqueeze(1)
                    state['latent_'+name][:, :, :self.history].copy_(flat)
                    state['padded_'+name][:, :self.history, :, :flat.shape[-1]].copy_(flat.transpose(1, 2))
            if self.sink:
                self.raw_positions[:self.sink].copy_(torch.arange(self.sink, device=self.token.device))

    def attention(self, q, k, v, lengths, causal=False):
        if self.extension is not None:
            scale = 1/math.sqrt(self.model.config.hidden_size//self.model.config.num_attention_heads)
            if causal and q.shape[1] == 2:
                forward = self.extension.forward
                if self.scope:
                    forward = lambda q, k, v, scale, causal, splits, pack, lengths: profiled_pair()(q, k, v, lengths, scale, splits)
                return paired_attention(forward, q, k, v, scale, self.paired_lengths, self.sm_count)
            if self.scope:
                return profile_ops()[1](q, k, v, lengths, scale, causal)
            return self.extension.forward(q, k, v, scale, causal, 0, True, lengths)
        from slimkv.llama3.latent_decode import eager_forward
        n = int(lengths[0])
        return eager_forward(q, k[:, :n], v[:, :n],
                             1/math.sqrt(self.model.config.hidden_size//self.model.config.num_attention_heads), causal)

    def tail(self, layer, output, residual):
        if self.compile_tail:
            from slimkv.llama3.compile_decode import compiled_tail
            return compiled_tail()(layer, output, residual)
        hidden = residual + output
        return hidden + layer.mlp(layer.post_attention_layernorm(hidden))

    def forward(self, size):
        model = self.model
        hidden = model.model.embed_tokens(self.token)
        if size:
            beacon = self.token.new_full((1, 1), 0)
            hidden = torch.cat([hidden, model.model.beacon_embed_tokens(beacon)], dim=1)
        for state, layer in zip(self.layers, model.model.layers):
            residual = hidden
            normalized = layer.input_layernorm(hidden)
            with self.marked(ATTENTION):
                if self.mode == 'full_kv':
                    output = self.full_attention(layer.self_attn, normalized, state)
                elif self.mode == 'latent':
                    output = self.latent_attention(layer.self_attn, normalized, state, size)
                else:
                    output = self.reconstruction_attention(layer.self_attn, normalized, state, size)
            hidden = self.tail(layer, output, residual)
        hidden = model.model.norm(hidden)
        return model.lm_head(hidden).float()[:, :1]

    def full_attention(self, attn, hidden, state):
        q = attn.q_proj(hidden).view(1, 1, attn.num_heads, attn.head_dim).transpose(1, 2)
        k = attn.k_proj(hidden).view(1, 1, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
        v = attn.v_proj(hidden).view_as(k.transpose(1, 2)).transpose(1, 2)
        with self.marked(ROPE):
            q, k = apply_rope(q, k, attn.rotary_emb, self.positions[1], self.positions[1], profile=self.scope)
        state['k'].index_copy_(2, self.index, k)
        state['v'].index_copy_(2, self.index, v)
        out, _ = self.attention(q.transpose(1, 2), state['k'][:, :, :self.bucket].transpose(1, 2),
                                state['v'][:, :, :self.bucket].transpose(1, 2), self.lengths)
        return attn.o_proj(out.reshape(1, 1, -1))

    def latent_attention(self, attn, hidden, state, size):
        length = 1+size
        q = attn.q_proj(hidden)
        if size and 'q' in attn.config.beacon_param:
            q = torch.cat([q[:, :1], attn.beacon_q_proj(hidden[:, 1:])], dim=1)
        q = q.view(1, length, attn.num_heads, attn.head_dim).transpose(1, 2)
        k = attn.k_proj(hidden[:, :1]).view(1, 1, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
        v = attn.v_proj(hidden[:, :1]).view_as(k.transpose(1, 2)).transpose(1, 2)
        state['raw_k'].index_copy_(2, self.index, k)
        state['raw_v'].index_copy_(2, self.index, v)
        if size:
            kd = attn.beacon_kv_down_proj if attn._use_shared_kv_down_proj() else attn.beacon_k_down_proj
            vd = attn.beacon_kv_down_proj if attn._use_shared_kv_down_proj() else attn.beacon_v_down_proj
            for name, down in [('k', kd), ('v', vd)]:
                latent = down(hidden[:, 1:]).unsqueeze(1)
                state['latent_'+name].index_copy_(2, self.latent_index, latent)
                rank = latent.shape[-1]
                state['padded_'+name][..., :rank].index_copy_(1, self.latent_index, latent.transpose(1, 2))
        with self.marked(ROPE):
            q, rotated = apply_rope(q, k, attn.rotary_emb, self.positions[length],
                                    self.positions[length][:, :1], profile=self.scope)
            state['rotated_k'].index_copy_(2, self.index, rotated)
        raw, raw_lse = self.attention(q.transpose(1, 2), state['rotated_k'][:, :, :self.raw_bucket].transpose(1, 2),
                                      state['raw_v'][:, :, :self.raw_bucket].transpose(1, 2), self.lengths)
        latent_q = project_query(q, attn.beacon_k_up_proj.weight, attn.num_key_value_heads).transpose(1, 2)
        width = state['width']
        if latent_q.shape[-1] != width:
            latent_q = torch.nn.functional.pad(latent_q, (0, width-latent_q.shape[-1]))
        out, lse = self.attention(latent_q, state['padded_k'][:, :self.latent_bucket],
                                  state['padded_v'][:, :self.latent_bucket], self.latent_lengths, bool(size))
        out = project_value(out[..., :attn.beacon_v_up_proj.in_features].transpose(1, 2),
                            attn.beacon_v_up_proj.weight, attn.num_key_value_heads)
        if attn.beacon_k_up_proj.bias is not None:
            bias = attn.beacon_k_up_proj.bias.view(attn.num_key_value_heads, attn.head_dim)
            bias = bias.repeat_interleave(attn.num_key_value_groups, dim=0)
            lse = lse+(q.float()*bias[None, :, None].float()).sum(-1)/math.sqrt(attn.head_dim)
        if attn.beacon_v_up_proj.bias is not None:
            bias = attn.beacon_v_up_proj.bias.view(attn.num_key_value_heads, attn.head_dim)
            out = out+bias.repeat_interleave(attn.num_key_value_groups, dim=0)[None, :, None]
        merged = merge_attention(raw.transpose(1, 2), raw_lse, out, lse)
        merged = merged.transpose(1, 2).reshape(1, length, -1)
        output = attn.o_proj(merged)
        if size and 'o' in attn.config.beacon_param:
            beacon_output = attn.beacon_o_proj(merged)
            output = torch.cat([output[:, :1], beacon_output[:, 1:]], dim=1)
        return output

    def reconstruction_attention(self, attn, hidden, state, size):
        length = 1+size
        q = attn.q_proj(hidden)
        k, v = attn.k_proj(hidden), attn.v_proj(hidden)
        if size:
            if 'q' in attn.config.beacon_param:
                q = torch.cat([q[:, :1], attn.beacon_q_proj(hidden)[:, 1:]], dim=1)
            kd = attn.beacon_kv_down_proj if attn._use_shared_kv_down_proj() else attn.beacon_k_down_proj
            vd = attn.beacon_kv_down_proj if attn._use_shared_kv_down_proj() else attn.beacon_v_down_proj
            kl, vl = kd(hidden), vd(hidden)
            with self.marked(RECONSTRUCTION):
                bk, bv = attn.beacon_k_up_proj(kl), attn.beacon_v_up_proj(vl)
            k, v = torch.cat([k[:, :1], bk[:, 1:]], dim=1), torch.cat([v[:, :1], bv[:, 1:]], dim=1)
            state['latent_k'].index_copy_(2, self.latent_index, kl[:, 1:].unsqueeze(1))
            state['latent_v'].index_copy_(2, self.latent_index, vl[:, 1:].unsqueeze(1))
        q = q.view(1, length, attn.num_heads, attn.head_dim).transpose(1, 2)
        k = k.view(1, length, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
        v = v.view_as(k.transpose(1, 2)).transpose(1, 2)
        state['raw_k'].index_copy_(2, self.index, k[:, :, :1])
        state['raw_v'].index_copy_(2, self.index, v[:, :, :1])
        with self.marked(RECONSTRUCTION):
            for name, up in [('k', attn.beacon_k_up_proj), ('v', attn.beacon_v_up_proj)]:
                history = up(state['latent_'+name][:, :, :self.history].squeeze(1))
                history = history.view(1, self.history, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
                state[name][:, :, self.sink:self.prefix].copy_(history)
                if self.old_beacons:
                    window = up(state['latent_'+name][:, :, self.history:self.history+self.old_beacons].squeeze(1))
                    window = window.view(1, self.old_beacons, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
                    state[name].index_copy_(2, self.beacon_slots[:self.old_beacons], window)
                if self.sink:
                    state[name][:, :, :self.sink].copy_(state['raw_'+name][:, :, :self.sink])
                state[name].index_copy_(2, self.raw_slots,
                                        state['raw_'+name][:, :, self.sink:self.sink+self.decode_steps])
            state['k'].index_copy_(2, self.current_slots[length], k)
            state['v'].index_copy_(2, self.current_slots[length], v)
        with self.marked(ROPE):
            q, k = apply_rope(q, state['k'][:, :, :self.bucket], attn.rotary_emb,
                              self.positions[length], self.key_positions[:, :self.bucket],
                              self.roles[:self.bucket], profile=self.scope)
        out, _ = self.attention(q.transpose(1, 2), k.transpose(1, 2),
                                state['v'][:, :, :self.bucket].transpose(1, 2), self.lengths, bool(size))
        # Preserve original projection GEMM shapes even when only raw logits are used.
        output = attn.o_proj(out.reshape(1, length, -1))
        if size and 'o' in attn.config.beacon_param:
            beacon_output = attn.beacon_o_proj(out.reshape(1, length, -1))
            output = torch.cat([output[:, :1], beacon_output[:, 1:]], dim=1)
        return output

    def run(self, token, graph=None):
        if self.step_index >= self.decode_steps:
            raise ValueError('Decode continuation exceeded the reserved capacity.')
        i = self.step_index
        size = int(self.mode != 'full_kv' and (i+1)%self.ratio == 0)
        length = 1+size
        self.token.copy_(token)
        if self.mode == 'full_kv':
            end = self.prefix+i+1
            self.index.fill_(end-1)
            self.positions[1].fill_(end-1)
            self.lengths.fill_(end)
            self.bucket = (end+127)//128*128
            key = (1, self.bucket)
        else:
            self.old_beacons = i//self.ratio
            self.raw_length = self.sink+i+1
            self.raw_bucket = min(self.raw_capacity, (self.raw_length+127)//128*128)
            total = self.prefix+i+self.old_beacons+length
            self.index.fill_(self.raw_length-1)
            self.latent_index.fill_(self.history+self.old_beacons)
            self.positions[length].copy_(torch.arange(total-length, total, device=self.token.device)[None])
            self.raw_positions[self.raw_length-1:self.raw_length].fill_(total-length)
            self.current_slots[length].copy_(self.positions[length][0])
            latent_length = self.history+self.old_beacons+size
            self.latent_lengths.fill_(latent_length)
            if size:
                pair_length = latent_length if self.mode == 'latent' else total
                self.paired_lengths.fill_(pair_length)
                self.paired_lengths[:1].fill_(pair_length-1)
            self.latent_bucket = (latent_length+63)//64*64
            self.bucket = (total+127)//128*128
            self.lengths.fill_(self.raw_length if self.mode == 'latent' else total)
            key = (length, self.raw_bucket if self.mode == 'latent' else self.old_beacons,
                   self.latent_bucket if self.mode == 'latent' else self.bucket)
        # Static shapes are chosen outside the graph. Graph construction is warmup.
        enabled = self.use_graph if graph is None else graph
        if enabled:
            if key not in self.graphs:
                current = torch.cuda.current_stream()
                stream = torch.cuda.Stream()
                stream.wait_stream(current)
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        self.forward(size)
                current.wait_stream(stream)
                graph_obj = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph_obj, stream=stream, pool=self.pool):
                    output = self.forward(size)
                current.wait_stream(stream)
                self.graphs[key] = graph_obj, output
            graph_obj, output = self.graphs[key]
            graph_obj.replay()
        else:
            output = self.forward(size)
        self.step_index += 1
        return output.clone()
