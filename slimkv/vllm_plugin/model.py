"""vLLM scheduling and paged storage with SlimKV's original attention math."""
from typing import Iterable

import torch
from torch import nn

from slimkv.llama3.configuration_llama import LlamaConfig
from slimkv.llama3.memory import LlamaMemory
from slimkv.llama3.modeling_llama import LlamaForCausalLM
from slimkv.modeling_utils import get_rope


def normalize_rope_config(values):
    """Recover legacy RoPE fields without changing their checkpoint values."""
    values = dict(values)
    rope = dict(values.get('rope_parameters') or values.get('rope_scaling') or {})
    values['rope_theta'] = rope.get('rope_theta', values.get('rope_theta', 10000.0))
    kind = rope.get('rope_type', rope.get('type', 'default'))
    values['rope_scaling'] = None if kind == 'default' else rope
    return values


class SlimKVLlamaForCausalLM(nn.Module):
    has_inner_state = True

    def __init__(self, *, vllm_config, prefix=''):
        super().__init__()
        parallel = vllm_config.parallel_config
        model_config = vllm_config.model_config
        scheduler = vllm_config.scheduler_config
        self.native_paged = bool(getattr(model_config.hf_config, 'slimkv_native_paged', False))
        if parallel.tensor_parallel_size != 1 or parallel.pipeline_parallel_size != 1:
            raise ValueError('SlimKV currently requires single-device execution.')
        if not self.native_paged and (scheduler.max_num_seqs != 1 or scheduler.enable_chunked_prefill):
            raise ValueError('This adapter requires B1 and unchunked prefill.')
        if vllm_config.cache_config.enable_prefix_caching:
            raise ValueError('SlimKV cache ownership does not support prefix sharing yet.')
        if not model_config.enforce_eager or vllm_config.speculative_config is not None:
            raise ValueError('Use enforce_eager=True and disable speculative decoding.')
        if model_config.dtype != torch.bfloat16 or model_config.quantization is not None:
            raise ValueError('SlimKV requires the original BF16 checkpoint without quantization.')
        values = normalize_rope_config(model_config.hf_config.to_dict())
        if values.get('model_type') != 'llama':
            raise ValueError('This adapter supports SlimKV Llama checkpoints.')
        if not values.get('beacon_skip_rope_for_beacon_k'):
            raise ValueError('Direct latent decode requires the trained no-Beacon-K-RoPE checkpoint.')
        if any(values.get('beacon_' + kind + '_compression') != 'lowrank_svd' for kind in ('k', 'v')):
            raise ValueError('SlimKV requires converted low-rank K and V weights.')
        if len(values.get('beacon_ratio', [])) != 1:
            raise ValueError('Select one token ratio through hf_overrides.')
        if self.native_paged:
            window, ratio = values['beacon_window'], values['beacon_ratio'][0]
            if (values['beacon_stride'] != window or window % 16 or ratio <= 0
                    or window % ratio or values['beacon_pos'] != 'interleave'
                    or values['beacon_attn'] != 'full-coverage'
                    or not values.get('beacon_attend_prev', True)
                    or values.get('beacon_sink_size', 0) not in (0, 1)
                    or values.get('beacon_chunk_fixed_rope_for_beacon_k', False)):
                raise ValueError('Paged SlimKV requires nonoverlapping aligned windows, '
                                 'interleaved full-coverage Beacons and at most one sink token.')
        values.update(beacon_cache_mode='latent', beacon_decode_backend='latent',
                      beacon_decode_fast=True, beacon_decode_graph=True,
                      beacon_decode_graph_full=True, beacon_ratio_mix='sequence')
        config = LlamaConfig(**values)
        config._attn_implementation = 'flash_attention_2'
        self.model = LlamaForCausalLM(config).eval()
        self.model.memory = LlamaMemory(config)
        # HF checkpoint loading builds these nonpersistent buffers on CPU.
        # GPU initialization changes rounding in pow/sin/cos, then BF16 Q/K.
        with torch.device('cpu'):
            canonical = get_rope(self.model.model.layers[0].self_attn.head_dim,
                                 config.rope_theta, config.max_position_embeddings,
                                 config.rope_scaling)
        device = self.model.model.embed_tokens.weight.device
        values = {name: getattr(canonical, name).to(device)
                  for name in ('inv_freq', 'cos_cached', 'sin_cached')}
        for layer in self.model.model.layers:
            for name, value in values.items():
                getattr(layer.self_attn.rotary_emb, name).copy_(value)
        if self.native_paged:
            from .backend import create_page_layers
            self.raw_pages, self.beacon_pages = create_page_layers(self.model, vllm_config, prefix)

    def embed_input_ids(self, input_ids):
        return self.model.model.embed_tokens(input_ids)

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        if intermediate_tensors is not None or inputs_embeds is not None:
            raise ValueError('SlimKV expects token IDs on a single device.')
        if input_ids.ndim != 1 or positions.ndim != 1 or input_ids.shape != positions.shape:
            raise ValueError('Expected one flat, unpadded token stream.')
        if input_ids.numel() == 0:
            return self.model.model.embed_tokens.weight.new_empty(0, self.model.config.hidden_size)
        if self.native_paged:
            from vllm.forward_context import get_forward_context
            context = get_forward_context()
            if context.attn_metadata is not None:
                return self._paged_forward(input_ids, context.attn_metadata)
        self._request_heads = None
        if int(positions[0]) == 0:
            self.model.memory.reset()
        output = self.model(input_ids=input_ids[None],
                            attention_mask=torch.ones_like(input_ids)[None],
                            return_normalized_hidden=True, return_dict=True)
        hidden = output.last_hidden_state[0]
        self._native_hidden = output.native_last_hidden_state
        self._native_roles = None
        if self.model.memory._all_beacon_sizes[-1] > 0:
            self._native_roles = self.model.memory._beacon_indices[-self._native_hidden.shape[1]:].clone()
        if hidden.shape[0] != input_ids.shape[0]:
            raise RuntimeError('SlimKV returned the wrong number of raw-token hidden states.')
        return hidden

    def _paged_forward(self, input_ids, metadata):
        from .paged_model import restore_memory, flush_memory, release_workspace_views
        beacon_meta = metadata[self.beacon_pages[0].name]
        starts = beacon_meta.query_start_loc_cpu.tolist()
        lengths = beacon_meta.seq_lens_cpu.tolist()
        outputs, self._request_heads = [], []
        for request in range(beacon_meta.num_reqs):
            begin, end = starts[request:request + 2]
            if begin == end:
                continue
            previous = lengths[request] - (end - begin)
            raw_table = [metadata[layer.name].block_table_tensor[request] for layer in self.raw_pages]
            beacon_table = [metadata[layer.name].block_table_tensor[request] for layer in self.beacon_pages]
            restore_memory(self.model, previous, self.raw_pages, self.beacon_pages, raw_table, beacon_table)
            ids = input_ids[begin:end][None]
            output = self.model(input_ids=ids, attention_mask=torch.ones_like(ids),
                                return_normalized_hidden=True, return_dict=True)
            hidden = output.last_hidden_state[0]
            if hidden.shape[0] != end - begin:
                raise RuntimeError('Paged request returned an incorrect raw-token count.')
            roles = None
            if self.model.memory._all_beacon_sizes[-1] > 0:
                roles = self.model.memory._beacon_indices[-output.native_last_hidden_state.shape[1]:].clone()
            self._request_heads.append((output.native_last_hidden_state, roles))
            flush_memory(self.model, previous, lengths[request], self.raw_pages, self.beacon_pages,
                         raw_table, beacon_table)
            release_workspace_views(self.model.memory)
            outputs.append(hidden)
        return torch.cat(outputs, dim=0)

    def compute_logits(self, hidden_states):
        if self._request_heads is not None and hidden_states.shape[0] == len(self._request_heads):
            rows = []
            for hidden, roles in self._request_heads:
                logits = self.model.lm_head(hidden).float()
                if roles is not None:
                    logits = logits[:, roles == 0]
                rows.append(logits[:, -1])
            return torch.cat(rows, dim=0)
        if hidden_states.shape[0] == 1:
            # Preserve the original GEMM shape, including its Beacon rows.
            # Changing the shape can change BF16 rounding before sampling.
            logits = self.model.lm_head(self._native_hidden).float()
            if self._native_roles is not None:
                logits = logits[:, self._native_roles == 0]
            return logits[:, -1]
        return self.model.lm_head(hidden_states.to(self.model.lm_head.weight.dtype)).float()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params = dict(self.model.named_parameters())
        loaded = set()
        with torch.no_grad():
            for name, value in weights:
                if name.endswith('.rotary_emb.inv_freq'):
                    continue
                if name not in params:
                    raise ValueError(f'Unexpected SlimKV checkpoint tensor: {name}')
                param = params[name]
                if param.shape != value.shape:
                    raise ValueError(f'Checkpoint shape mismatch for {name}: {value.shape} versus {param.shape}')
                param.copy_(value)
                loaded.add('model.' + name)
        return loaded
