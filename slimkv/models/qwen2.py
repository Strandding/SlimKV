"""
SlimKV – Qwen2-specific patched forward functions.

Supports:
  anchor_kv_type = "full" | "lowrank"
  skip_anchor_rope_k = True | False
  shared_kv_down = True | False  (lowrank only)
"""

import math
import torch
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.qwen2.modeling_qwen2 import rotate_half


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _apply_rope(x, cos, sin):
    """Apply RoPE to a single tensor. x: [bsz, heads, seq_len, head_dim]."""
    return (x * cos) + (rotate_half(x) * sin)


def repeat_kv(hidden_states, n_rep):
    if n_rep == 1:
        return hidden_states
    bsz, num_kv_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(bsz, num_kv_heads, n_rep, seq_len, head_dim)
    return hidden_states.reshape(bsz, num_kv_heads * n_rep, seq_len, head_dim)


# ---------------------------------------------------------------------------
# Anchor QKV projection (dispatches based on config)
# ---------------------------------------------------------------------------

def _project_anchor_k(attn, hidden_states):
    """Project anchor K: full-rank uses separate anchor_k_proj, lowrank uses down+up."""
    cfg = attn._slimkv_config
    if cfg["anchor_kv_type"] == "full":
        return attn.anchor_k_proj(hidden_states)
    else:
        down = attn.anchor_kv_down if cfg["shared_kv_down"] else attn.anchor_k_down
        return attn.anchor_k_up(down(hidden_states))


def _project_anchor_v(attn, hidden_states):
    """Project anchor V: full-rank uses separate anchor_v_proj, lowrank uses down+up."""
    cfg = attn._slimkv_config
    if cfg["anchor_kv_type"] == "full":
        return attn.anchor_v_proj(hidden_states)
    else:
        down = attn.anchor_kv_down if cfg["shared_kv_down"] else attn.anchor_v_down
        return attn.anchor_v_up(down(hidden_states))


# ---------------------------------------------------------------------------
# Patched attention forward
# ---------------------------------------------------------------------------

def patched_attn_forward(
    self,
    hidden_states,
    attention_mask=None,
    position_ids=None,
    past_key_value=None,
    output_attentions=False,
    use_cache=False,
    **kwargs,
):
    """Qwen2 attention forward with anchor-token aware projection + optional RoPE skip."""
    bsz, q_len, _ = hidden_states.size()
    past_key, past_value, anchor_size, anchor_indices = past_key_value
    head_dim = self.head_dim
    num_heads = getattr(self, "num_heads", None)
    if num_heads is None:
        num_heads = self.q_proj.out_features // head_dim
    num_kv_heads = getattr(self, "num_key_value_heads", None)
    if num_kv_heads is None:
        num_kv_heads = self.k_proj.out_features // head_dim

    kv_seq_len = q_len
    if past_key is not None:
        kv_seq_len += past_key.shape[2]

    # ---- 1. QKV projection ----
    if anchor_size > 0:
        cur_mask = anchor_indices[-q_len:]  # 1=anchor, 0=raw

        base_q = self.q_proj(hidden_states)
        anchor_q = self.anchor_q_proj(hidden_states)
        query_states = torch.where((cur_mask == 0)[:, None], base_q, anchor_q)

        base_k = self.k_proj(hidden_states)
        anchor_k = _project_anchor_k(self, hidden_states)
        key_states = torch.where((cur_mask == 0)[:, None], base_k, anchor_k)

        base_v = self.v_proj(hidden_states)
        anchor_v = _project_anchor_v(self, hidden_states)
        value_states = torch.where((cur_mask == 0)[:, None], base_v, anchor_v)
    else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

    # ---- 2. Reshape ----
    query_states = query_states.view(bsz, q_len, num_heads, head_dim)
    key_states = key_states.view(bsz, q_len, num_kv_heads, head_dim)
    value_states = value_states.view(bsz, q_len, num_kv_heads, head_dim)

    # Qwen3 applies per-head RMSNorm before RoPE.
    if hasattr(self, "q_norm"):
        query_states = self.q_norm(query_states)
    if hasattr(self, "k_norm"):
        key_states = self.k_norm(key_states)

    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    # ---- 3. Cache pre-RoPE KV (incremental) ----
    new_past_key_value = (key_states, value_states, anchor_size, anchor_indices)

    # ---- 4. Concat with cached KV ----
    if past_key is not None:
        key_states = torch.cat([past_key, key_states], dim=2)
        value_states = torch.cat([past_value, value_states], dim=2)

    # ---- 5. RoPE ----
    # We cache pre-RoPE KV and re-apply RoPE after concat, so Q and K have different seq_lens.
    # Q: [bsz, heads, q_len, head_dim], K: [bsz, kv_heads, kv_seq_len, head_dim]
    # position_ids: [bsz, kv_seq_len] — last q_len entries correspond to Q positions.
    if hasattr(self, "rotary_emb"):
        # Qwen2-style rotary module on attention layer.
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        # [seq_len, head_dim] -> [bsz, kv_seq_len, head_dim]
        cos = cos[position_ids]
        sin = sin[position_ids]
    elif hasattr(self, "_slimkv_model_rotary_emb"):
        # Qwen3-style rotary module on model.
        cos, sin = self._slimkv_model_rotary_emb(value_states, position_ids)
    else:
        raise AttributeError("No rotary embedding module found for SlimKV patched attention.")

    k_cos = cos.unsqueeze(1)  # [bsz, 1, kv_seq_len, head_dim]
    k_sin = sin.unsqueeze(1)
    q_cos = k_cos[:, :, -q_len:, :]  # Q positions are the last q_len
    q_sin = k_sin[:, :, -q_len:, :]

    if hasattr(self, "rotary_fn"):
        query_states, _ = self.rotary_fn(query_states, query_states, cos[:, -q_len:, :], sin[:, -q_len:, :])
    else:
        query_states = _apply_rope(query_states, q_cos, q_sin)

    cfg = self._slimkv_config
    if cfg["skip_anchor_rope_k"] and anchor_size > 0:
        # Build full-length anchor mask aligned with key_states
        prefix_len = kv_seq_len - len(anchor_indices)
        if prefix_len > 0:
            prefix_indices = anchor_indices.new_ones(prefix_len)
            full_indices = torch.cat([prefix_indices, anchor_indices])
        else:
            full_indices = anchor_indices
        anchor_k_mask = (full_indices == 1)
        beacon_k_backup = key_states[:, :, anchor_k_mask].clone()
        if hasattr(self, "rotary_fn"):
            _, key_states = self.rotary_fn(key_states, key_states, cos, sin)
        else:
            key_states = _apply_rope(key_states, k_cos, k_sin)
        key_states[:, :, anchor_k_mask] = beacon_k_backup
    else:
        if hasattr(self, "rotary_fn"):
            _, key_states = self.rotary_fn(key_states, key_states, cos, sin)
        else:
            key_states = _apply_rope(key_states, k_cos, k_sin)

    # ---- 6. GQA repeat ----
    num_groups = getattr(self, "num_key_value_groups", None)
    if num_groups is None:
        num_groups = num_heads // num_kv_heads
    key_states = repeat_kv(key_states, num_groups)
    value_states = repeat_kv(value_states, num_groups)

    # ---- 7. Attention ----
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states)

    # ---- 8. Output ----
    attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
    attn_output = self.o_proj(attn_output)

    return attn_output, None, new_past_key_value


# ---------------------------------------------------------------------------
# Patched model forward
# ---------------------------------------------------------------------------

def _layer_forward(layer, hidden_states, attention_mask, position_ids, past_key_value):
    residual = hidden_states
    hidden_states = layer.input_layernorm(hidden_states)
    hidden_states, _, new_past_key_value = layer.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        use_cache=True,
    )
    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = layer.post_attention_layernorm(hidden_states)
    hidden_states = layer.mlp(hidden_states)
    hidden_states = residual + hidden_states

    return hidden_states, new_past_key_value


def patched_causal_lm_forward(
    self,
    input_ids=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    labels=None,
    use_cache=True,
    **kwargs,
):
    hidden_states = self.model.embed_tokens(input_ids)

    all_new_past_kv = []
    use_grad_ckpt = getattr(self.model, "gradient_checkpointing", False) and self.training

    for i, layer in enumerate(self.model.layers):
        layer_past_kv = past_key_values[i] if past_key_values is not None else None

        if use_grad_ckpt:
            hidden_states, new_past_kv = torch.utils.checkpoint.checkpoint(
                _layer_forward, layer, hidden_states, attention_mask,
                position_ids, layer_past_kv, use_reentrant=False,
            )
        else:
            hidden_states, new_past_kv = _layer_forward(
                layer, hidden_states, attention_mask, position_ids, layer_past_kv,
            )
        all_new_past_kv.append(new_past_kv)

    hidden_states = self.model.norm(hidden_states)
    logits = self.lm_head(hidden_states)

    # Labels are pre-shifted in Memory.prepare(), no shift needed
    loss = None
    if labels is not None:
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )

    return CausalLMOutputWithPast(
        loss=loss, logits=logits, past_key_values=all_new_past_kv,
    )
