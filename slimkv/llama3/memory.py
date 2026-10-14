"""Inference-only latent Beacon storage with ordinary full-dimensional attention."""
from ..memory import Memory, cat_tensor, slice_tensor


def _length(tensor):
    return 0 if tensor is None else tensor.shape[2]


class ReconstructionCache:
    """A layer-local cache read; materialization is never retained by Memory."""

    def __init__(self, sink, history, raw, window, window_roles, num_heads, head_dim, kind,
                 window_raw_indices=None):
        self.sink = sink
        self.history = history
        self.raw = raw
        self.window = window
        self.window_roles = window_roles
        self.kind = kind
        self.window_raw_indices = window_raw_indices
        batch = next((x.shape[0] for x in (sink, history, raw, window) if x is not None), 0)
        length = sum(_length(x) for x in (sink, history, raw, window))
        self.shape = (batch, num_heads, length, head_dim)

    def materialize(self, attention):
        up = attention.beacon_k_up_proj if self.kind == 'k' else attention.beacon_v_up_proj

        def reconstruct(latent):
            if latent is None:
                return None
            batch, heads, length, dim = latent.shape
            flat = latent.transpose(1, 2).reshape(batch, length, heads * dim)
            full = up(flat)
            return full.reshape(batch, length, attention.num_key_value_heads,
                                attention.head_dim).transpose(1, 2)

        history = reconstruct(self.history)
        if self.window is None:
            recent = self.raw
        else:
            window = reconstruct(self.window)
            recent = window.new_empty(window.shape[0], window.shape[1],
                                      self.window_roles.numel(), window.shape[-1])
            recent[:, :, self.window_roles] = window
            if self.raw is not None:
                recent[:, :, ~self.window_roles] = self.raw
        return cat_tensor([self.sink, history, recent], dim=2)


class ReconstructionCacheUpdate(tuple):
    """Keep the legacy four-item cache interface; carry newly projected latents."""

    def __new__(cls, key, value, beacon_size, beacon_indices, key_latents, value_latents):
        result = super().__new__(cls, (key, value, beacon_size, beacon_indices))
        result.key_latents = key_latents
        result.value_latents = value_latents
        return result


class LlamaMemory(Memory):
    """Use latent K/V for supported Llama inference; preserve full-cache training."""

    def reset(self):
        super().reset()
        self.window_beacon_activations = [(None, None) for _ in range(self.config.num_hidden_layers)]
        self._window_roles = None
        self._latent_cache_active = None
        self._cache_decode_backend = None
        self._window_raw_indices = None
        self._fast_decode_active = False
        self._decode_unpadded = None
        self._decode_role_buffer = None
        self._decode_bool_buffer = None
        self._decode_raw_indices = None
        self._decode_positions = None

    def prepare(self, input_ids, attention_mask, labels, skip_first=None, skip_last=None):
        self._fast_decode_active = (
            not self.training and labels is None and input_ids is not None
            and input_ids.ndim == 2 and input_ids.shape[1] == 1
            and self.latent_cache_active and self.finish
            and self._cache_decode_backend == 'fa2'
            and getattr(self.config, 'beacon_decode_fast', True)
            and getattr(self.config, '_slimkv_decode_fast_step', False)
            and skip_first is None and skip_last is None
            and self.beacon_skip_first is None and self.beacon_skip_last is None
        )
        # Prefill retains the legacy CPU bookkeeping. Once decoding starts, keep
        # input history on the GPU; a later multi-token call can return to CPU.
        self._input_storage_device = input_ids.device if self._fast_decode_active else self._cpu
        if not self._fast_decode_active:
            self._decode_unpadded = None
            self._decode_role_buffer = None
            self._decode_bool_buffer = None
            self._decode_raw_indices = None
        return super().prepare(input_ids, attention_mask, labels, skip_first, skip_last)

    def _step(self, ignore_memory=False):
        if not self._fast_decode_active or ignore_memory:
            return super()._step(ignore_memory)
        import torch
        start = self._start_idx
        end = self.all_sequence_length
        full = end >= start + self.config.beacon_window
        if self._is_full_window:
            ratio = self.set_compression_ratio(start_idx=start, end_idx=end)
            self._interleave_compression_ratio = ratio
            previous_roles = 0
            previous_raw = 0
        else:
            ratio = self._interleave_compression_ratio
            previous_roles = self._beacon_indices.numel()
            previous_raw = 0 if self._window_raw_indices is None else self._window_raw_indices.numel()
        size = (self._interleave_remainder + 1) // ratio
        self._interleave_remainder = (self._interleave_remainder + 1) % ratio
        token = self.all_input_ids[:, -1:]
        if size:
            # Decode inserts at most one Beacon, immediately after the raw token.
            token = torch.cat([token, token.new_full((token.shape[0], 1), self.config.vocab_size)], dim=1)
        if self._decode_role_buffer is None:
            capacity = 2 * self.config.beacon_window + 2
            self._decode_role_buffer = torch.zeros(capacity, dtype=torch.long, device=token.device)
            self._decode_bool_buffer = torch.zeros(capacity, dtype=torch.bool, device=token.device)
            self._decode_raw_indices = torch.empty(self.config.beacon_window + 1, dtype=torch.long, device=token.device)
            if previous_roles:
                self._decode_role_buffer[:previous_roles].copy_(self._beacon_indices)
                self._decode_bool_buffer[:previous_roles].copy_(self._window_roles)
            if previous_raw:
                self._decode_raw_indices[:previous_raw].copy_(self._window_raw_indices)
        count = previous_roles + 1 + size
        self._decode_role_buffer[previous_roles:count].fill_(0)
        self._decode_bool_buffer[previous_roles:count].fill_(False)
        if size:
            self._decode_role_buffer[count - 1:count].fill_(1)
            self._decode_bool_buffer[count - 1:count].fill_(True)
        self._decode_raw_indices[previous_raw:previous_raw + 1].fill_(previous_roles)
        indices = self._decode_role_buffer[:count]
        past = [self._get_layer_past_key_value(i, size, indices, False)
                for i in range(self.config.num_hidden_layers)]
        length = past[0][0].shape[2] + 1 + size
        if self._decode_positions is None or self._decode_positions.numel() < length:
            capacity = max(4096, 1 << (length - 1).bit_length())
            self._decode_positions = torch.arange(capacity, dtype=torch.long, device=token.device)
        positions = self._decode_positions[:length].unsqueeze(0).expand(token.shape[0], -1)
        self._beacon_indices = indices
        self._fast_window_roles = self._decode_bool_buffer[:count]
        self._fast_window_raw_indices = self._decode_raw_indices[:previous_raw + 1]
        self._is_full_window = full
        self._raw_size_to_cache = 0 if full else -1
        self._all_beacon_sizes.append(size)
        self._start_idx = start + self.config.beacon_stride if full else start
        self._end_idx = end
        self._step_idx += 1
        # FA2 decode has already checked padding. Its raw/latent branches handle
        # causality without constructing an attention mask or re-scanning roles.
        return token, None, positions, past, None

    @property
    def latent_cache_active(self):
        return self._latent_cache_active is True

    def _select_cache_mode(self):
        mode = getattr(self.config, 'beacon_cache_mode', 'auto')
        backend = getattr(self.config, 'beacon_decode_backend', 'reconstruct')
        if backend not in ('reconstruct', 'fa2'):
            raise ValueError(f'Unknown beacon_decode_backend: {backend}')
        if self._cache_decode_backend is None:
            self._cache_decode_backend = backend
        elif backend != self._cache_decode_backend:
            raise RuntimeError('Reset Memory before changing the decode backend.')
        if mode not in ('auto', 'full', 'latent'):
            raise ValueError(f'Unknown beacon_cache_mode: {mode}')
        supported = (
            self.config.beacon_pos == 'interleave'
            and self.config.beacon_window == self.config.beacon_stride
            and self.config.beacon_parallel_window == 1
            and all(r > 0 for r in self.config.beacon_ratio)
            and all(p in self.config.beacon_param for p in ('k', 'v'))
            and self.config.beacon_k_compression == 'lowrank_svd'
            and self.config.beacon_v_compression == 'lowrank_svd'
            and not getattr(self.config, 'beacon_chunk_fixed_rope_for_beacon_k', False)
            and not getattr(self.config, 'beacon_enhance_k_pos_residual', False)
            and self.beacon_skip_first is None and self.beacon_skip_last is None
        )
        if not self.training and mode == 'latent' and not supported:
            raise ValueError('Latent Beacon cache requires interleave, positive ratios, one '
                             'non-overlapping window, low-rank K/V, and no fixed-RoPE, '
                             'K residual or beacon_skip overrides.')
        active = not self.training and mode != 'full' and supported
        if not self.training and backend == 'fa2' and (
            not active or self.config.beacon_attn != 'full-coverage'
            or not self.config.beacon_attend_prev
            or not getattr(self.config, 'beacon_skip_rope_for_beacon_k', False)
            or getattr(self.config, 'raw_nokrope', False)
            or getattr(self.config, 'beacon_skip_rope_for_raw_k', False)
            or getattr(self.config, 'beacon_headwise_up_proj', False)
            or getattr(self.config, 'beacon_latent_rpb', False)
        ):
            raise ValueError('FA2 decode requires latent storage, full-coverage, attend_prev, '
                             'raw K RoPE, unrotated Beacon K, global up projections '
                             'and no latent relative-position bias.')
        if self._latent_cache_active is None:
            self._latent_cache_active = active
        elif active != self._latent_cache_active:
            raise RuntimeError('Reset Memory before changing its training state or cache mode.')

    def _get_layer_past_key_value(self, layer_idx, beacon_size, beacon_indices, ignore_memory):
        self._select_cache_mode()
        if not self.latent_cache_active:
            return super()._get_layer_past_key_value(layer_idx, beacon_size, beacon_indices, ignore_memory)
        heads = self.config.num_key_value_heads
        head_dim = self.config.hidden_size // self.config.num_attention_heads
        pairs = [(None, None)] * 4 if ignore_memory else [
            self.sink_activations[layer_idx], self.beacon_activations[layer_idx],
            self.raw_activations[layer_idx], self.window_beacon_activations[layer_idx],
        ]
        key = ReconstructionCache(*(x[0] for x in pairs), self._window_roles, heads, head_dim, 'k', self._window_raw_indices)
        value = ReconstructionCache(*(x[1] for x in pairs), self._window_roles, heads, head_dim, 'v', self._window_raw_indices)
        return key, value, beacon_size, beacon_indices

    def get_memory_size(self):
        sink, history, raw = super().get_memory_size()
        if self.latent_cache_active:
            raw += _length(self.window_beacon_activations[0][0])
        return sink, history, raw

    def update_memory(self, past_key_values):
        if not self.latent_cache_active:
            return super().update_memory(past_key_values)
        for idx, update in enumerate(past_key_values):
            if not isinstance(update, ReconstructionCacheUpdate):
                raise TypeError('Latent Memory requires projected Beacon latents from Llama attention.')
            key, value, _, indices = update
            old_raw_k, old_raw_v = self.raw_activations[idx]
            old_window_k, old_window_v = self.window_beacon_activations[idx]
            # Boolean indexing owns its storage, unlike a view into a full projection.
            if getattr(update, 'raw_only', False):
                new_raw_k, new_raw_v = key, value
            else:
                current_roles = indices[-key.shape[2]:] != 0
                new_raw_k, new_raw_v = key[:, :, ~current_roles], value[:, :, ~current_roles]
            raw_k = cat_tensor([old_raw_k, new_raw_k], dim=2)
            raw_v = cat_tensor([old_raw_v, new_raw_v], dim=2)
            window_k = cat_tensor([old_window_k, update.key_latents], dim=2)
            window_v = cat_tensor([old_window_v, update.value_latents], dim=2)
            if self.beacon_activations[idx][0] is None and self.config.beacon_sink_size > 0:
                self.sink_activations[idx] = (
                    slice_tensor(key, end=self.config.beacon_sink_size, dim=2).clone(),
                    slice_tensor(value, end=self.config.beacon_sink_size, dim=2).clone(),
                )
            if self._is_full_window:
                old_k, old_v = self.beacon_activations[idx]
                self.beacon_activations[idx] = (
                    cat_tensor([old_k, window_k], dim=2),
                    cat_tensor([old_v, window_v], dim=2),
                )
                self.raw_activations[idx] = (None, None)
                self.window_beacon_activations[idx] = (None, None)
            else:
                self.raw_activations[idx] = (raw_k, raw_v)
                self.window_beacon_activations[idx] = (window_k, window_v)
        self._window_roles = None if self._is_full_window else (
            self._fast_window_roles if self._fast_decode_active else indices != 0)
        if self._cache_decode_backend == 'fa2':
            self._window_raw_indices = None if self._is_full_window else (
                self._fast_window_raw_indices if self._fast_decode_active else (indices == 0).nonzero().flatten())

    def cache_bytes(self):
        """Logical persistent K/V bytes; attention workspaces are excluded."""
        groups = {
            'sink': self.sink_activations, 'raw': self.raw_activations,
            'beacon': self.beacon_activations, 'window_beacon': self.window_beacon_activations,
        }
        return {name: sum(x.numel() * x.element_size() for pair in pairs for x in pair if x is not None)
                for name, pairs in groups.items()}

    def output(self, model_outputs):
        model_outputs = super().output(model_outputs)
        if self.latent_cache_active:
            # generate() carries this value into the next iteration. Do not let
            # its returned cache retain the full current-window projections.
            model_outputs['past_key_values'] = tuple(
                self._get_layer_past_key_value(idx, 0, self._beacon_indices, False)
                for idx in range(self.config.num_hidden_layers)
            )
        return model_outputs
