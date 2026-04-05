"""
SlimKV – Simplified Memory for windowed KV-cache compression.

Fixed config: window=2048, stride=2048, compression_ratio=8, interleave, full-coverage.
"""

import torch
import torch.nn.functional as F


class Memory:
    def __init__(self, config, anchor_token_id, num_layers, k_seq_dim=2, v_seq_dim=2):
        self.config = config
        self.anchor_token_id = anchor_token_id
        self.num_layers = num_layers
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim

        # Fixed hyper-parameters (simplification)
        self.window = 2048
        self.stride = 2048
        self.compression_ratio = 8

        self.reset()

    def reset(self):
        """Reset state for a new sequence."""
        self._all_input_ids = None
        self._all_attention_mask = None
        self._all_labels = None
        self._start_idx = 0
        self._step_idx = 0

        # Per-layer KV caches: anchor history + raw cache
        self.anchor_kv = [(None, None) for _ in range(self.num_layers)]
        self.raw_kv = [(None, None) for _ in range(self.num_layers)]

        # Loss accumulation
        self._total_loss = None
        self._total_tokens = 0

    def prepare(self, input_ids, attention_mask, labels):
        """Store the full sequence for chunked iteration."""
        self._all_input_ids = input_ids
        self._all_attention_mask = attention_mask if attention_mask is not None else torch.ones_like(input_ids)
        # Pre-shift labels for next-token prediction (same as activation_beacon):
        # labels[i] = input_ids[i+1], so forward can use cross_entropy(logits, labels) directly
        if labels is not None:
            self._all_labels = torch.cat([labels[:, 1:], labels.new_full((labels.shape[0], 1), -100)], dim=1)
        else:
            self._all_labels = None
        self._all_seq_len = input_ids.shape[1]
        self._start_idx = 0
        self._step_idx = 0
        self._total_loss = None
        self._total_tokens = 0

        # Reset KV caches
        self.anchor_kv = [(None, None) for _ in range(self.num_layers)]
        self.raw_kv = [(None, None) for _ in range(self.num_layers)]

    @property
    def finish(self):
        return self._start_idx >= self._all_seq_len

    def step(self):
        """Prepare inputs for one chunk. Returns (input_ids, attn_mask, position_ids, past_kv, labels)."""
        start = self._start_idx
        end = min(start + self.stride, self._all_seq_len)
        device = self._all_input_ids.device
        bsz = self._all_input_ids.shape[0]

        # Slice raw tokens for this chunk
        raw_ids = self._all_input_ids[:, start:end]
        raw_mask = self._all_attention_mask[:, start:end]
        raw_labels = self._all_labels[:, start:end] if self._all_labels is not None else None

        # Determine anchor count
        raw_len = raw_ids.shape[1]
        is_full_window = (raw_len == self.stride)
        anchor_size = raw_len // self.compression_ratio if is_full_window else 0

        # Insert anchor tokens (interleave)
        if anchor_size > 0:
            total_len = raw_len + anchor_size
            input_ids = raw_ids.new_full((bsz, total_len), self.anchor_token_id)
            # Positions of raw tokens: skip every (compression_ratio+1)-th position
            # for interleave pattern: [raw]*ratio, anchor, [raw]*ratio, anchor, ...
            raw_indices = torch.arange(total_len, device=device)
            raw_indices = raw_indices[raw_indices % (self.compression_ratio + 1) != self.compression_ratio]
            raw_indices = raw_indices[:raw_len].unsqueeze(0).expand(bsz, -1)
            input_ids.scatter_(1, raw_indices, raw_ids)

            attn_mask = raw_mask.new_ones(bsz, total_len)
            attn_mask.scatter_(1, raw_indices, raw_mask)

            if raw_labels is not None:
                labels = raw_labels.new_full((bsz, total_len), -100)
                labels.scatter_(1, raw_indices, raw_labels)
            else:
                labels = None

            # Anchor indices: True where input_ids == anchor_token_id
            anchor_indices = (input_ids[0] == self.anchor_token_id).long()
        else:
            input_ids = raw_ids
            attn_mask = raw_mask
            labels = raw_labels
            anchor_indices = torch.zeros(raw_len, dtype=torch.long, device=device)

        # Skip loss for the very first window (no anchor history to validate)
        if self._step_idx == 0 and labels is not None:
            labels = labels.clone()
            labels[:] = -100

        # Build past_key_values from cached KV
        past_key_values = []
        for layer_idx in range(self.num_layers):
            anchor_k, anchor_v = self.anchor_kv[layer_idx]
            raw_k, raw_v = self.raw_kv[layer_idx]
            key = _cat_kv(anchor_k, raw_k, dim=self.k_seq_dim)
            value = _cat_kv(anchor_v, raw_v, dim=self.v_seq_dim)
            past_key_values.append((key, value, anchor_size, anchor_indices))

        # Build attention mask (causal + memory)
        input_len = input_ids.shape[1]
        mem_size = past_key_values[0][0].shape[self.k_seq_dim] if past_key_values[0][0] is not None else 0
        total_len = mem_size + input_len

        # Position ids
        position_ids = torch.arange(total_len, dtype=torch.long, device=device).unsqueeze(0).expand(bsz, -1)

        # 4D causal attention mask
        full_attn_mask = torch.cat([attn_mask.new_ones(bsz, mem_size), attn_mask], dim=1)
        causal_mask = _make_causal_mask(input_len, mem_size, device, self.config.torch_dtype)
        # Apply padding mask
        padding_mask = full_attn_mask[:, None, None, :].expand(bsz, 1, input_len, total_len)
        min_val = torch.finfo(self.config.torch_dtype).min
        causal_mask = causal_mask.masked_fill(padding_mask == 0, min_val)

        self._start_idx = end
        self._step_idx += 1
        self._current_anchor_size = anchor_size
        self._current_anchor_indices = anchor_indices
        self._is_full_window = is_full_window

        return input_ids, causal_mask, position_ids, past_key_values, labels

    def update_memory(self, past_key_values):
        """Extract anchor and raw KV from the returned past_key_values after one chunk."""
        for layer_idx, (key, value, anchor_size, anchor_indices) in enumerate(past_key_values):
            # past_key_values returned from forward are the NEW keys/values only (pre-RoPE)
            prev_raw_k, prev_raw_v = self.raw_kv[layer_idx]

            if not self._is_full_window:
                # Accumulate raw activations
                self.raw_kv[layer_idx] = (
                    _cat_kv(prev_raw_k, key, dim=self.k_seq_dim),
                    _cat_kv(prev_raw_v, value, dim=self.v_seq_dim),
                )
            else:
                # Full window: separate anchor and raw tokens
                full_key = _cat_kv(prev_raw_k, key, dim=self.k_seq_dim)
                full_value = _cat_kv(prev_raw_v, value, dim=self.v_seq_dim)

                # Extract anchor KV
                anchor_mask = (anchor_indices == 1)
                prev_anchor_k, prev_anchor_v = self.anchor_kv[layer_idx]
                new_anchor_k = full_key[:, :, anchor_mask]
                new_anchor_v = full_value[:, :, anchor_mask]
                self.anchor_kv[layer_idx] = (
                    _cat_kv(prev_anchor_k, new_anchor_k, dim=self.k_seq_dim),
                    _cat_kv(prev_anchor_v, new_anchor_v, dim=self.v_seq_dim),
                )

                # No raw cache to keep (stride == window)
                self.raw_kv[layer_idx] = (None, None)

    def update_loss(self, loss, labels):
        """Accumulate loss across chunks."""
        if loss is None or labels is None:
            return
        valid_tokens = (labels != -100).sum().item()
        if valid_tokens == 0:
            return
        if self._total_loss is None:
            self._total_loss = loss * valid_tokens
            self._total_tokens = valid_tokens
        else:
            self._total_loss = self._total_loss + loss * valid_tokens
            self._total_tokens += valid_tokens

    def output(self, last_outputs):
        """Return final outputs with accumulated loss."""
        if self._total_loss is not None and self._total_tokens > 0:
            last_outputs.loss = self._total_loss / self._total_tokens
        elif last_outputs.loss is None:
            # Fallback: all labels were -100 (e.g., very short sequence with only 1 window)
            # Return zero loss to avoid Trainer crash
            last_outputs.loss = torch.tensor(0.0, device=last_outputs.logits.device, requires_grad=True)
        return last_outputs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cat_kv(a, b, dim=2):
    if a is None:
        return b
    if b is None:
        return a
    return torch.cat([a, b], dim=dim)


def _make_causal_mask(tgt_len, mem_len, device, dtype):
    """Lower-triangular causal mask for (tgt_len) query attending to (mem_len + tgt_len) keys."""
    min_val = torch.finfo(dtype).min
    total = mem_len + tgt_len
    # tgt queries can see all memory + causal within input
    causal = torch.full((tgt_len, tgt_len), min_val, device=device, dtype=dtype)
    causal.masked_fill_(torch.arange(tgt_len, device=device).unsqueeze(1) >= torch.arange(tgt_len, device=device).unsqueeze(0), 0)
    # Memory part: all visible
    mem_mask = torch.zeros(tgt_len, mem_len, device=device, dtype=dtype)
    mask = torch.cat([mem_mask, causal], dim=-1)  # [tgt_len, total]
    return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, tgt_len, total]
