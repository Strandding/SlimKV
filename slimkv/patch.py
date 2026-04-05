"""
SlimKV – Monkey-patch a HuggingFace causal-LM for anchor-token KV compression.

Configurable options:
  anchor_kv_type:      "full" (separate full-rank K/V) or "lowrank" (down+up)
  skip_anchor_rope_k:  skip RoPE for anchor token keys
  shared_kv_down:      share the down projection between K and V (lowrank only)
"""

import types
import torch
import torch.nn as nn

from .models.qwen2 import patched_attn_forward, patched_causal_lm_forward


# ---------------------------------------------------------------------------
# Parameter injection
# ---------------------------------------------------------------------------

def _build_lowrank_layers(in_features, out_features, latent_dim, bias=False):
    down = nn.Linear(in_features, latent_dim, bias=False)
    up = nn.Linear(latent_dim, out_features, bias=bias)
    down.weight.data.zero_()
    up.weight.data.zero_()
    if up.bias is not None:
        up.bias.data.zero_()
    return down, up


def inject_anchor_params(attn, anchor_kv_type, latent_dim, shared_kv_down):
    """Attach anchor projections to one attention layer."""
    # Derive dimensions from actual module weights instead of config-level
    # assumptions. This keeps the patch compatible with models like Qwen3
    # where hidden_size != num_attention_heads * head_dim.
    in_features = attn.q_proj.in_features
    q_dim = attn.q_proj.out_features
    k_dim = attn.k_proj.out_features
    v_dim = attn.v_proj.out_features

    # Anchor Q – always full rank, initialised from base q_proj
    attn.anchor_q_proj = nn.Linear(in_features, q_dim, bias=attn.q_proj.bias is not None)
    attn.anchor_q_proj.weight.data.copy_(attn.q_proj.weight.data)
    if attn.q_proj.bias is not None:
        attn.anchor_q_proj.bias.data.copy_(attn.q_proj.bias.data)

    if anchor_kv_type == "full":
        # Full-rank: separate K/V with same shape as base, initialised from base weights
        attn.anchor_k_proj = nn.Linear(in_features, k_dim, bias=attn.k_proj.bias is not None)
        attn.anchor_k_proj.weight.data.copy_(attn.k_proj.weight.data)
        if attn.k_proj.bias is not None:
            attn.anchor_k_proj.bias.data.copy_(attn.k_proj.bias.data)

        attn.anchor_v_proj = nn.Linear(in_features, v_dim, bias=attn.v_proj.bias is not None)
        attn.anchor_v_proj.weight.data.copy_(attn.v_proj.weight.data)
        if attn.v_proj.bias is not None:
            attn.anchor_v_proj.bias.data.copy_(attn.v_proj.bias.data)

    elif anchor_kv_type == "lowrank":
        if shared_kv_down:
            # Shared down projection for K and V
            shared_down = nn.Linear(in_features, latent_dim, bias=False)
            shared_down.weight.data.zero_()
            shared_down._is_hf_initialized = True
            attn.anchor_kv_down = shared_down
            # Separate up projections
            _, attn.anchor_k_up = _build_lowrank_layers(
                in_features, k_dim, latent_dim, bias=attn.k_proj.bias is not None,
            )
            _, attn.anchor_v_up = _build_lowrank_layers(
                in_features, v_dim, latent_dim, bias=attn.v_proj.bias is not None,
            )
        else:
            attn.anchor_k_down, attn.anchor_k_up = _build_lowrank_layers(
                in_features, k_dim, latent_dim, bias=attn.k_proj.bias is not None,
            )
            attn.anchor_v_down, attn.anchor_v_up = _build_lowrank_layers(
                in_features, v_dim, latent_dim, bias=attn.v_proj.bias is not None,
            )
    else:
        raise ValueError(f"Unknown anchor_kv_type: {anchor_kv_type}")


# ---------------------------------------------------------------------------
# Windowed forward (replaces model.forward)
# ---------------------------------------------------------------------------

def _slimkv_forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
    memory = self.memory

    # Synchronize sequence lengths across all ranks so every rank iterates
    # the same number of chunks.  Without this, ranks with shorter sequences
    # finish the while-loop earlier and the subsequent DeepSpeed gradient
    # allreduce deadlocks.
    if torch.distributed.is_initialized():
        import math
        local_steps = math.ceil(input_ids.shape[1] / memory.stride)
        steps_tensor = torch.tensor([local_steps], device=input_ids.device)
        torch.distributed.all_reduce(steps_tensor, op=torch.distributed.ReduceOp.MAX)
        target_len = int(steps_tensor.item()) * memory.stride

        pad_len = target_len - input_ids.shape[1]
        if pad_len > 0:
            input_ids = torch.nn.functional.pad(input_ids, (0, pad_len), value=0)
            if attention_mask is not None:
                attention_mask = torch.nn.functional.pad(attention_mask, (0, pad_len), value=0)
            if labels is not None:
                labels = torch.nn.functional.pad(labels, (0, pad_len), value=-100)

    memory.prepare(input_ids, attention_mask, labels)

    outputs = None
    while not memory.finish:
        chunk_ids, chunk_mask, chunk_pos, past_kv, chunk_labels = memory.step()

        outputs = self._patched_forward(
            input_ids=chunk_ids,
            attention_mask=chunk_mask,
            position_ids=chunk_pos,
            past_key_values=past_kv,
            labels=chunk_labels,
            use_cache=True,
        )

        memory.update_memory(outputs.past_key_values)
        memory.update_loss(outputs.loss, chunk_labels)

    return memory.output(outputs)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def patch_model(model, memory, anchor_token_id,
                anchor_kv_type="full", latent_dim=64,
                skip_anchor_rope_k=False, shared_kv_down=False):
    """Apply all SlimKV patches. Currently supports Qwen2."""
    # Store config on model for attention forward to read
    model._slimkv_config = {
        "anchor_kv_type": anchor_kv_type,
        "skip_anchor_rope_k": skip_anchor_rope_k,
        "shared_kv_down": shared_kv_down,
    }

    # ---- Inject anchor parameters ----
    for layer in model.model.layers:
        inject_anchor_params(
            layer.self_attn, anchor_kv_type, latent_dim, shared_kv_down,
        )
        # Store config ref on each attention module
        layer.self_attn._slimkv_config = model._slimkv_config
        # Qwen3 keeps rotary embedding at model-level instead of attention-level.
        if not hasattr(layer.self_attn, "rotary_emb") and hasattr(model.model, "rotary_emb"):
            layer.self_attn._slimkv_model_rotary_emb = model.model.rotary_emb

    # ---- Move new params to model device / dtype ----
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    for layer in model.model.layers:
        attn = layer.self_attn
        for name, mod in attn.named_modules():
            if "anchor" in name:
                mod.to(device=device, dtype=dtype)

    # ---- Replace attention forward (per-layer) ----
    for layer in model.model.layers:
        layer.self_attn.forward = types.MethodType(patched_attn_forward, layer.self_attn)

    # ---- Replace model forward ----
    model._patched_forward = types.MethodType(patched_causal_lm_forward, model)
    model.forward = types.MethodType(_slimkv_forward, model)

    model.memory = memory
    model._anchor_token_id = anchor_token_id

    return model
