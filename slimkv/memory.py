"""
SlimKV – Simplified Memory for windowed KV-cache compression.

Fixed config: stride=2048, compression_ratio=8, interleave, full-coverage.
Interleave placement uses per-chunk ceil compression:
anchor_size = ceil(raw_len / compression_ratio).
Raw tokens are distributed so earlier gaps are larger and later gaps are
smaller, and each chunk ends with an anchor token.
With this fixed setup, historical memory consists of anchor KV only.
"""

import torch

IGNORE_INDEX = -100


class Memory:
    def __init__(self, config, anchor_token_id, num_layers, k_seq_dim=2, v_seq_dim=2):
        self.config = config
        self.anchor_token_id = anchor_token_id
        self.num_layers = num_layers
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.torch_dtype = _resolve_torch_dtype(getattr(config, "torch_dtype", None))

        # Fixed hyper-parameters (simplification)
        self.stride = 2048
        self.compression_ratio = 8

        self.reset()

    def reset(self):
        """Reset state for a new sequence."""
        self._all_input_ids = None
        self._all_labels = None
        self._start_idx = 0
        self._step_idx = 0

        # Per-layer KV caches: anchor history only.
        self.anchor_kv = [(None, None) for _ in range(self.num_layers)]

        # Loss accumulation
        self._total_loss = None
        self._total_tokens = 0

    def prepare(self, input_ids, attention_mask, labels):
        """Store the full sequence for chunked iteration."""
        self._all_input_ids = input_ids
        if attention_mask is not None and not torch.all(attention_mask == 1):
            raise ValueError(
                "SlimKV memory expects no padding tokens when micro-batch size is 1. "
                "Got attention_mask containing 0."
            )
        # Pre-shift labels for next-token prediction (same as activation_beacon):
        # labels[i] = input_ids[i+1], so forward can use cross_entropy(logits, labels) directly
        if labels is not None:
            self._all_labels = torch.cat(
                [labels[:, 1:], labels.new_full((labels.shape[0], 1), IGNORE_INDEX)],
                dim=1,
            )
        else:
            self._all_labels = None
        self._all_seq_len = input_ids.shape[1]
        self._start_idx = 0
        self._step_idx = 0
        self._total_loss = None
        self._total_tokens = 0

        # Reset KV caches
        self.anchor_kv = [(None, None) for _ in range(self.num_layers)]

    @property
    def finish(self):
        return self._start_idx >= self._all_seq_len

    def step(self):
        """Prepare one chunk.

        Returns: (input_ids, causal_mask, position_ids, past_kv, labels).
        """
        start = self._start_idx
        end = min(start + self.stride, self._all_seq_len)
        device = self._all_input_ids.device
        bsz = self._all_input_ids.shape[0]

        # Slice raw tokens for this chunk
        raw_ids = self._all_input_ids[:, start:end]
        raw_labels = self._all_labels[:, start:end] if self._all_labels is not None else None

        # Determine anchor count for this chunk with ceil compression.
        raw_len = raw_ids.shape[1]
        anchor_size = _compute_anchor_size(
            raw_len=raw_len,
            compression_ratio=self.compression_ratio,
        )

        # Insert anchor tokens (interleave)
        if anchor_size > 0:
            total_len = raw_len + anchor_size
            input_ids = raw_ids.new_full((bsz, total_len), self.anchor_token_id)
            raw_indices = _build_interleaved_raw_indices(
                raw_len=raw_len,
                anchor_size=anchor_size,
                device=device,
            )
            raw_indices = raw_indices.unsqueeze(0).expand(bsz, -1)
            input_ids.scatter_(1, raw_indices, raw_ids)

            if raw_labels is not None:
                labels = raw_labels.new_full((bsz, total_len), IGNORE_INDEX)
                labels.scatter_(1, raw_indices, raw_labels)
            else:
                labels = None

            # Anchor indices: True where input_ids == anchor_token_id
            anchor_indices = (input_ids[0] == self.anchor_token_id).long()
        else:
            input_ids = raw_ids
            labels = raw_labels
            anchor_indices = torch.zeros(raw_len, dtype=torch.long, device=device)

        # Skip loss for the very first window (no anchor history to validate)
        if self._step_idx == 0 and labels is not None:
            labels = labels.clone()
            labels[:] = IGNORE_INDEX

        # Build past_key_values from cached KV
        past_key_values = [
            (anchor_k, anchor_v, anchor_size, anchor_indices)
            for anchor_k, anchor_v in self.anchor_kv
        ]

        # Build causal attention mask (no PAD masking in micro-batch-size=1 mode)
        input_len = input_ids.shape[1]
        mem_size = past_key_values[0][0].shape[self.k_seq_dim] if past_key_values[0][0] is not None else 0

        # Position ids
        position_ids = torch.arange(mem_size + input_len, dtype=torch.long, device=device).unsqueeze(0).expand(bsz, -1)

        # 4D causal attention mask
        causal_mask = _make_causal_mask(input_len, mem_size, device, self.torch_dtype)

        self._start_idx = end
        self._step_idx += 1

        return input_ids, causal_mask, position_ids, past_key_values, labels

    def update_memory(self, past_key_values):
        """Extract and append anchor KV from the current chunk."""
        for layer_idx, (key, value, anchor_size, anchor_indices) in enumerate(past_key_values):
            # past_key_values returned from forward are the NEW keys/values only (pre-RoPE)
            if anchor_size == 0:
                # Empty chunk guard (normally unreachable).
                continue

            anchor_mask = (anchor_indices == 1)
            prev_anchor_k, prev_anchor_v = self.anchor_kv[layer_idx]
            new_anchor_k = key[:, :, anchor_mask]
            new_anchor_v = value[:, :, anchor_mask]
            self.anchor_kv[layer_idx] = (
                _cat_kv(prev_anchor_k, new_anchor_k, dim=self.k_seq_dim),
                _cat_kv(prev_anchor_v, new_anchor_v, dim=self.v_seq_dim),
            )

    def update_loss(self, loss, labels):
        """Accumulate loss across chunks."""
        if loss is None or labels is None:
            return
        valid_tokens = (labels != IGNORE_INDEX).sum().item()
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


def _compute_anchor_size(raw_len, compression_ratio):
    if raw_len <= 0:
        return 0
    return (raw_len + compression_ratio - 1) // compression_ratio


def _build_interleaved_raw_indices(
    raw_len,
    anchor_size,
    device,
):
    """Return raw-token positions for ceil-compression interleave insertion.

    We create ``anchor_size`` raw spans before anchors:
    - earlier spans use ceil(raw_len / anchor_size),
    - later spans use floor(raw_len / anchor_size).
    This guarantees the final token in the chunk is an anchor.
    """
    if raw_len <= 0:
        return torch.empty((0,), device=device, dtype=torch.long)
    if anchor_size <= 0:
        return torch.arange(raw_len, device=device, dtype=torch.long)

    base_span, remainder = divmod(raw_len, anchor_size)
    spans = [base_span + 1] * remainder + [base_span] * (anchor_size - remainder)

    raw_positions = []
    pos = 0
    for span in spans:
        raw_positions.extend(range(pos, pos + span))
        pos += span + 1  # one anchor slot after every raw span

    return torch.tensor(raw_positions, device=device, dtype=torch.long)


def _resolve_torch_dtype(dtype):
    if isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, str):
        dtype_map = {
            "torch.bfloat16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "torch.float16": torch.float16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "half": torch.float16,
            "torch.float32": torch.float32,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        return dtype_map.get(dtype.lower(), torch.float32)
    return torch.float32


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
