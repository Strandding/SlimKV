# coding=utf-8
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch Llama model."""
import inspect
import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from transformers.integrations import is_deepspeed_zero3_enabled
from ..llama3.configuration_llama import LlamaConfig


if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

    _flash_supports_window_size = "window_size" in list(inspect.signature(flash_attn_func).parameters)

from .memory import Memory
from ..modeling_utils import optional_grad_ctx, compute_loss, build_beacon_loss_weights, get_rope, rotate_half, ModelOutput


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LlamaConfig"

# Copied from transformers.models.llama.modeling_llama._get_unpad_data
def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


# Copied from transformers.models.llama.modeling_llama.LlamaRMSNorm with Llama->Llama
class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


# Copied from transformers.models.llama.modeling_llama.LlamaMLP with Llama->Llama
class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        mlp_bias = getattr(config, "mlp_bias", False)
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def make_layer_past_key_value(
    raw_key: Optional[torch.Tensor] = None,
    raw_value: Optional[torch.Tensor] = None,
    beacon_key: Optional[torch.Tensor] = None,
    beacon_value: Optional[torch.Tensor] = None,
    beacon_size: int = 0,
    beacon_indices: Optional[torch.Tensor] = None,
    raw_prefix_size: int = 0,
):
    return {
        "raw_key": raw_key,
        "raw_value": raw_value,
        "beacon_key": beacon_key,
        "beacon_value": beacon_value,
        "beacon_size": beacon_size,
        "beacon_indices": beacon_indices,
        "raw_prefix_size": raw_prefix_size,
    }


def unpack_layer_past_key_value(past_key_value):
    if past_key_value is None:
        return (
            None,  # raw_key
            None,  # raw_value
            None,  # beacon_key
            None,  # beacon_value
            0,     # beacon_size
            None,  # beacon_indices
            0,     # raw_prefix_size
        )

    if isinstance(past_key_value, dict):
        return (
            past_key_value.get("raw_key", None),
            past_key_value.get("raw_value", None),
            past_key_value.get("beacon_key", None),
            past_key_value.get("beacon_value", None),
            past_key_value.get("beacon_size", 0),
            past_key_value.get("beacon_indices", None),
            past_key_value.get("raw_prefix_size", 0),
        )

    if isinstance(past_key_value, (tuple, list)) and len(past_key_value) == 4:
        raw_key, raw_value, beacon_size, beacon_indices = past_key_value
        return raw_key, raw_value, None, None, beacon_size, beacon_indices, 0

    raise TypeError(f"Unsupported past_key_value type: {type(past_key_value)}")


class HeadwiseUpProj(nn.Module):
    """Head-wise independent up projection for low-rank beacon K/V."""

    def __init__(
        self,
        num_heads: int,
        latent_head_dim: int,
        head_dim: int,
        bias: bool = True,
        dtype=None,
        device=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.latent_head_dim = latent_head_dim
        self.head_dim = head_dim
        self.weight = nn.Parameter(
            torch.empty(num_heads, head_dim, latent_head_dim, dtype=dtype, device=device)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(num_heads, head_dim, dtype=dtype, device=device))
        else:
            self.register_parameter("bias", None)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        hidden_states = hidden_states.reshape(batch_size, seq_len, self.num_heads, self.latent_head_dim)
        projected = torch.einsum("bshl,hdl->bshd", hidden_states, self.weight)
        if self.bias is not None:
            projected = projected + self.bias.unsqueeze(0).unsqueeze(0)
        return projected.reshape(batch_size, seq_len, self.num_heads * self.head_dim)


class GroupedQueryUpProj(nn.Module):
    """Up-project latent outputs per KV head; query heads reuse KV-head weights by GQA mapping."""

    def __init__(
        self,
        num_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        latent_head_dim: int,
        bias: bool = True,
        dtype=None,
        device=None,
    ):
        super().__init__()
        if num_heads % num_key_value_heads != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must be divisible by num_key_value_heads ({num_key_value_heads})."
            )
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.latent_head_dim = latent_head_dim
        self.num_key_value_groups = num_heads // num_key_value_heads
        self.weight = nn.Parameter(
            torch.empty(num_key_value_heads, head_dim, latent_head_dim, dtype=dtype, device=device)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(num_key_value_heads, head_dim, dtype=dtype, device=device))
        else:
            self.register_parameter("bias", None)

    def forward(self, latent_states: torch.Tensor) -> torch.Tensor:
        # latent_states: [batch, num_heads, q_len, latent_head_dim]
        head_to_kv = torch.arange(self.num_heads, device=latent_states.device) // self.num_key_value_groups
        grouped_weight = self.weight.index_select(0, head_to_kv)
        output = torch.einsum("bhql,hdl->bhqd", latent_states, grouped_weight)
        if self.bias is not None:
            grouped_bias = self.bias.index_select(0, head_to_kv)
            output = output + grouped_bias.unsqueeze(0).unsqueeze(2)
        return output


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        config_head_dim = getattr(config, "head_dim", None)
        self.head_dim = config_head_dim if config_head_dim is not None else self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.beacon_hybrid_latent_attn = bool(getattr(config, "beacon_hybrid_latent_attn", True))

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

        self.rotary_emb = get_rope(self.head_dim, config.rope_theta, config.max_position_embeddings, getattr(config, "rope_scaling", None))

        # NOTE: add extra parameters for beacon tokens
        # skip post initialization to speed up loading
        has_beacon_kv = ("k" in config.beacon_param) or ("v" in config.beacon_param)
        if ("k" in config.beacon_param) != ("v" in config.beacon_param):
            raise ValueError("Llama3 beacon requires both `k` and `v` in beacon_param.")
        if has_beacon_kv and self._use_hybrid_latent_attn():
            self.beacon_k_latent_head_dim = self._get_beacon_k_latent_head_dim()
            self.beacon_v_latent_head_dim = self._get_beacon_v_latent_head_dim()
            self.beacon_k_latent_global_dim = self.num_key_value_heads * self.beacon_k_latent_head_dim
            self.beacon_v_latent_global_dim = self.num_key_value_heads * self.beacon_v_latent_head_dim

        if "q" in config.beacon_param:
            self.beacon_q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=self.q_proj.bias is not None)
            # NOTE: initialize the beacon parameters as zero
            self.beacon_q_proj.weight.data.zero_()
            self.beacon_q_proj._is_hf_initialized = True
        if "k" in config.beacon_param:
            if self._use_hybrid_latent_attn():
                self.beacon_k_latent_down_proj = nn.Linear(
                    self.hidden_size,
                    self.beacon_k_latent_global_dim,
                    bias=self.k_proj.bias is not None,
                )
                # Shared K-up matrix U_K: global-latent -> concatenated KV heads.
                self.beacon_k_latent_up_proj = nn.Linear(
                    self.beacon_k_latent_global_dim,
                    self.num_key_value_heads * self.head_dim,
                    bias=False,
                )
            else:
                self.beacon_k_proj = nn.Linear(
                    self.hidden_size,
                    self.num_key_value_heads * self.head_dim,
                    bias=self.k_proj.bias is not None,
                )
                self.beacon_k_proj.weight.data.zero_()
                self.beacon_k_proj._is_hf_initialized = True
        if "v" in config.beacon_param:
            if self._use_hybrid_latent_attn():
                self.beacon_v_latent_proj = nn.Linear(
                    self.hidden_size,
                    self.beacon_v_latent_global_dim,
                    bias=self.v_proj.bias is not None,
                )
                # Shared V-up matrix U_V: global-latent -> concatenated KV heads.
                self.beacon_v_latent_up_proj = nn.Linear(
                    self.beacon_v_latent_global_dim,
                    self.num_key_value_heads * self.head_dim,
                    bias=self.v_proj.bias is not None,
                )
            else:
                self.beacon_v_proj = nn.Linear(
                    self.hidden_size,
                    self.num_key_value_heads * self.head_dim,
                    bias=self.v_proj.bias is not None,
                )
                self.beacon_v_proj.weight.data.zero_()
                self.beacon_v_proj._is_hf_initialized = True
        if "o" in config.beacon_param:
            self.beacon_o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=self.o_proj.bias is not None)
            self.beacon_o_proj.weight.data.zero_()
            self.beacon_o_proj._is_hf_initialized = True

    def _use_hybrid_latent_attn(self):
        return (
            getattr(self.config, "beacon_hybrid_latent_attn", False)
            and ("k" in self.config.beacon_param and "v" in self.config.beacon_param)
        )

    def _get_layer_latent_dim_from_list(self, field_name: str):
        values = getattr(self.config, field_name, None)
        if values is None:
            raise ValueError(
                f"`{field_name}` must be provided when `beacon_hybrid_latent_attn=True`."
            )
        if not isinstance(values, (list, tuple)):
            raise ValueError(f"`{field_name}` must be a list/tuple, found {type(values)}.")
        if len(values) != self.config.num_hidden_layers:
            raise ValueError(
                f"`{field_name}` length must equal num_hidden_layers ({self.config.num_hidden_layers}), "
                f"found {len(values)}."
            )
        if self.layer_idx is None:
            raise ValueError(f"layer_idx must be set to use `{field_name}`.")
        value = values[self.layer_idx]
        if not isinstance(value, int):
            raise ValueError(f"`{field_name}[{self.layer_idx}]` must be int, found {type(value)}.")
        return value

    def _get_beacon_k_latent_head_dim(self):
        latent_head_dim = self._get_layer_latent_dim_from_list("beacon_k_latent_head_dim_list")
        if latent_head_dim <= 0:
            raise ValueError(f"beacon_k_latent_head_dim must be positive, found {latent_head_dim}.")
        if latent_head_dim > self.head_dim:
            raise ValueError(
                f"beacon_k_latent_head_dim must be <= head_dim ({self.head_dim}), found {latent_head_dim}."
            )
        return latent_head_dim

    def _project_beacon_key(self, hidden_states):
        if self._use_hybrid_latent_attn():
            return self.beacon_k_latent_down_proj(hidden_states)
        return self.beacon_k_proj(hidden_states)

    def _get_beacon_v_latent_head_dim(self):
        latent_head_dim = self._get_layer_latent_dim_from_list("beacon_v_latent_head_dim_list")
        if latent_head_dim <= 0:
            raise ValueError(f"beacon_v_latent_head_dim must be positive, found {latent_head_dim}.")
        if latent_head_dim > self.head_dim:
            raise ValueError(
                f"beacon_v_latent_head_dim must be <= head_dim ({self.head_dim}), found {latent_head_dim}."
            )
        return latent_head_dim

    def _project_beacon_value(self, hidden_states):
        if self._use_hybrid_latent_attn():
            return self.beacon_v_latent_proj(hidden_states)
        return self.beacon_v_proj(hidden_states)

    def _init_beacon_proj(self, missing_keys):
        """Initialize the beacon projection weight with that of the ordinal projection."""
        beacon_param = self.config.beacon_param
        
        if is_deepspeed_zero3_enabled():
            # FIXME: after deepspeed initialization, some weights becomes non-zero
            # For Llama, there are rows that are full of zeros
            # For Llama, there are values bigger than 1e29...

            import deepspeed
            if "q" in beacon_param:
                params = [self.beacon_q_proj.weight, self.q_proj.weight]
                if self.q_proj.bias is not None:
                    params.extend([self.beacon_q_proj.bias, self.q_proj.bias])
                with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
                    # FIXME: after deepspeed initialization, some weights becomes non-zero, but there are rows that are full of zeros
                    if (self.beacon_q_proj.weight.sum(-1) == 0).any() or (self.beacon_q_proj.weight > 1e29).any():
                        self.beacon_q_proj.weight.data[:] = self.q_proj.weight.data
                        if self.q_proj.bias is not None:
                            self.beacon_q_proj.bias.data[:] = self.q_proj.bias.data
            if "k" in beacon_param and hasattr(self, "beacon_k_proj"):
                params = [self.beacon_k_proj.weight, self.k_proj.weight]
                if self.k_proj.bias is not None:
                    params.extend([self.beacon_k_proj.bias, self.k_proj.bias])
                with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
                    if (self.beacon_k_proj.weight.sum(-1) == 0).any() or (self.beacon_k_proj.weight > 1e29).any():
                        self.beacon_k_proj.weight.data[:] = self.k_proj.weight.data
                        if self.k_proj.bias is not None:
                            self.beacon_k_proj.bias.data[:] = self.k_proj.bias.data
            if "v" in beacon_param and hasattr(self, "beacon_v_proj"):
                params = [self.beacon_v_proj.weight, self.v_proj.weight]
                if self.v_proj.bias is not None:
                    params.extend([self.beacon_v_proj.bias, self.v_proj.bias])
                with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
                    if (self.beacon_v_proj.weight.sum(-1) == 0).any() or (self.beacon_v_proj.weight > 1e29).any():
                        self.beacon_v_proj.weight.data[:] = self.v_proj.weight.data
                        if self.v_proj.bias is not None:
                            self.beacon_v_proj.bias.data[:] = self.v_proj.bias.data
            if "o" in beacon_param:
                params = [self.beacon_o_proj.weight, self.o_proj.weight]
                if self.o_proj.bias is not None:
                    params.extend([self.beacon_o_proj.bias, self.o_proj.bias])
                with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
                    # FIXME: after deepspeed initialization, some weights becomes non-zero, but there are rows that are full of zeros
                    if (self.beacon_o_proj.weight.sum(-1) == 0).any() or (self.beacon_o_proj.weight > 1e29).any():
                        self.beacon_o_proj.weight.data[:] = self.o_proj.weight.data
                        if self.o_proj.bias is not None:
                            self.beacon_o_proj.bias.data[:] = self.o_proj.bias.data
        else:
            # only copy the value in-place, without tieing the weight
            if "q" in beacon_param and any("beacon_q_proj" in missing_key for missing_key in missing_keys):
                # FIXME: some beacon weights are not initialized as zero for llama model, why? 
                # if (self.beacon_q_proj.weight == 0).all():
                    self.beacon_q_proj.weight.data[:] = self.q_proj.weight.data
                    if self.q_proj.bias is not None:
                        self.beacon_q_proj.bias.data[:] = self.q_proj.bias.data
            if (
                "k" in beacon_param
                and hasattr(self, "beacon_k_proj")
                and any("beacon_k_proj" in missing_key for missing_key in missing_keys)
            ):
                self.beacon_k_proj.weight.data[:] = self.k_proj.weight.data
                if self.k_proj.bias is not None:
                    self.beacon_k_proj.bias.data[:] = self.k_proj.bias.data
            if (
                "v" in beacon_param
                and hasattr(self, "beacon_v_proj")
                and any("beacon_v_proj" in missing_key for missing_key in missing_keys)
            ):
                self.beacon_v_proj.weight.data[:] = self.v_proj.weight.data
                if self.v_proj.bias is not None:
                    self.beacon_v_proj.bias.data[:] = self.v_proj.bias.data
            if "o" in beacon_param and any("beacon_o_proj" in missing_key for missing_key in missing_keys):
                # if (self.beacon_o_proj.weight == 0).all():
                    self.beacon_o_proj.weight.data[:] = self.o_proj.weight.data
                    if self.o_proj.bias is not None:
                        self.beacon_o_proj.bias.data[:] = self.o_proj.bias.data

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _cat_kv_states(self, states: List[Optional[torch.Tensor]]):
        states = [x for x in states if x is not None]
        if len(states) == 0:
            return None
        if len(states) == 1:
            return states[0]
        return torch.cat(states, dim=2)

    def _apply_rotary_query_and_key(self, query_states, key_states, query_position_ids, key_position_ids, rope_module):
        seq_len = max(
            query_position_ids.max().item() + 1,
            key_position_ids.max().item() + 1,
            key_states.shape[2],
        )
        if seq_len > rope_module.max_seq_len_cached:
            rope_module._set_cos_sin_cache(seq_len=seq_len, device=key_states.device, dtype=key_states.dtype)
        q_cos = rope_module.cos_cached[query_position_ids].to(dtype=query_states.dtype).unsqueeze(1)
        q_sin = rope_module.sin_cached[query_position_ids].to(dtype=query_states.dtype).unsqueeze(1)
        k_cos = rope_module.cos_cached[key_position_ids].to(dtype=key_states.dtype).unsqueeze(1)
        k_sin = rope_module.sin_cached[key_position_ids].to(dtype=key_states.dtype).unsqueeze(1)
        query_states = (query_states * q_cos) + (rotate_half(query_states) * q_sin)
        key_states = (key_states * k_cos) + (rotate_half(key_states) * k_sin)
        return query_states, key_states

    def _project_query_with_beacon(self, hidden_states, beacon_size, beacon_indices):
        if beacon_size <= 0 or beacon_indices is None or "q" not in self.config.beacon_param:
            return self.q_proj(hidden_states)
        cur_beacon_indices = beacon_indices[-hidden_states.shape[1]:]
        ordinal_query_states = self.q_proj(hidden_states)
        beacon_query_states = self.beacon_q_proj(hidden_states)
        query_states = torch.where((cur_beacon_indices == 0)[:, None], ordinal_query_states, beacon_query_states)
        if (cur_beacon_indices == 2).any():
            assert (cur_beacon_indices == 2).sum() <= (cur_beacon_indices == 1).sum(), \
                "Replicate beacon count exceeds source beacon count in Q projection"
            query_states[:, cur_beacon_indices == 2] = beacon_query_states[:, cur_beacon_indices == 1][:, :(cur_beacon_indices == 2).sum()]
        return query_states

    def _project_query_to_beacon_latent_with_k_up(self, query_states):
        """Project Q to global K-latent space via shared K-up matrix slices."""
        head_to_kv = torch.arange(self.num_heads, device=query_states.device) // self.num_key_value_groups
        k_up_weight = self.beacon_k_latent_up_proj.weight.view(
            self.num_key_value_heads, self.head_dim, self.beacon_k_latent_global_dim
        )
        grouped_k_up = k_up_weight.index_select(0, head_to_kv)
        return torch.einsum("bhqd,hdl->bhql", query_states, grouped_k_up)

    def _apply_rotary_with_beacon(self, query_states, key_states, position_ids, beacon_size, beacon_indices):
        """Apply standard RoPE to Q and K."""
        return self.rotary_emb(query_states, key_states, position_ids)

    def qkv_proj_with_beacon(self, hidden_states, beacon_size, beacon_indices):
        if beacon_size > 0:
            # NOTE: when beacon_pos == "interleave", the beacon_indices points to all beacon tokens in the current window (cached activations + input_ids), so we shall slice out the part corresponding to the input_ids
            cur_beacon_indices = beacon_indices[-hidden_states.shape[1]:]

            # NOTE: there is slight redundant computation because ordinal tokens should never be projected by beacon matrices, but we are doing this for efficiency
            if "q" in self.config.beacon_param:
                ordinal_query_states = self.q_proj(hidden_states)
                beacon_query_states = self.beacon_q_proj(hidden_states)
                query_states = torch.where((cur_beacon_indices == 0)[:, None], ordinal_query_states, beacon_query_states)
                if (cur_beacon_indices == 2).any():
                    # beacon_indices == 2 means the beacon token is used to replicate the ones in previous window for parallel encoding
                    # we should slice out all beacon tokens then copy them to the replicate beacon tokens
                    query_states[:, cur_beacon_indices == 2] = beacon_query_states[:, cur_beacon_indices == 1][:, :(cur_beacon_indices == 2).sum()]
            else:
                query_states = self.q_proj(hidden_states)

            if "k" in self.config.beacon_param:
                ordinal_key_states = self.k_proj(hidden_states)
                beacon_key_states = self._project_beacon_key(hidden_states)
                key_states = torch.where((cur_beacon_indices == 0)[:, None], ordinal_key_states, beacon_key_states)
                if (cur_beacon_indices == 2).any():
                    # beacon_indices == 2 means the beacon token is used to replicate the ones in previous window for parallel encoding
                    # we should slice out all beacon tokens then copy them to the replicate beacon tokens
                    key_states[:, cur_beacon_indices == 2] = beacon_key_states[:, cur_beacon_indices == 1][:, :(cur_beacon_indices == 2).sum()]
            else:
                key_states = self.k_proj(hidden_states)

            if "v" in self.config.beacon_param:
                ordinal_value_states = self.v_proj(hidden_states)
                beacon_value_states = self._project_beacon_value(hidden_states)
                value_states = torch.where((cur_beacon_indices == 0)[:, None], ordinal_value_states, beacon_value_states)
                if (cur_beacon_indices == 2).any():
                    # beacon_indices == 2 means the beacon token is used to replicate the ones in previous window for parallel encoding
                    # we should slice out all beacon tokens then copy them to the replicate beacon tokens
                    value_states[:, cur_beacon_indices == 2] = beacon_value_states[:, cur_beacon_indices == 1][:, :(cur_beacon_indices == 2).sum()]
            else:
                value_states = self.v_proj(hidden_states)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        return query_states, key_states, value_states

    def o_proj_with_beacon(self, attn_output, beacon_size, beacon_indices):
        if beacon_size > 0:
            # NOTE: when beacon_pos == "interleave", the beacon_indices points to all beacon tokens in the current window (cached activations + input_ids), so we shall slice out the part corresponding to the input_ids
            cur_beacon_indices = beacon_indices[-attn_output.shape[1]:]

            if "o" in self.config.beacon_param:
                ordinal_attn_output = self.o_proj(attn_output)
                beacon_attn_output = self.beacon_o_proj(attn_output)
                attn_output = torch.where((cur_beacon_indices == 0)[:, None], ordinal_attn_output, beacon_attn_output)
            else:
                attn_output = self.o_proj(attn_output)
        else:
            attn_output = self.o_proj(attn_output)
        return attn_output

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        bsz, q_len, _ = hidden_states.size()
        (
            past_raw_key,
            past_raw_value,
            past_beacon_key,
            past_beacon_value,
            beacon_size,
            beacon_indices,
            raw_prefix_size,
        ) = unpack_layer_past_key_value(past_key_value)

        if position_ids is None:
            # Fallback for non-beacon usage (e.g. native generation).
            # In beacon mode, Memory always provides position_ids; this path
            # does not account for the split raw-prefix/beacon/raw-suffix layout.
            total_guess_len = (
                (past_raw_key.shape[2] if past_raw_key is not None else 0)
                + (past_beacon_key.shape[2] if past_beacon_key is not None else 0)
                + q_len
            )
            position_ids = torch.arange(total_guess_len, device=hidden_states.device).unsqueeze(0).expand(bsz, -1)

        if beacon_size > 0 and beacon_indices is not None:
            cur_beacon_indices = beacon_indices[-q_len:]
        else:
            cur_beacon_indices = torch.zeros(q_len, dtype=torch.long, device=hidden_states.device)

        current_raw_mask = cur_beacon_indices == 0
        current_beacon_attn_mask = cur_beacon_indices != 0
        current_beacon_cache_mask = torch.logical_or(cur_beacon_indices == 1, cur_beacon_indices == -1)

        query_states = self._project_query_with_beacon(hidden_states, beacon_size, beacon_indices)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        ordinal_key_states = self.k_proj(hidden_states)
        ordinal_value_states = self.v_proj(hidden_states)

        if "k" in self.config.beacon_param:
            beacon_key_states = self._project_beacon_key(hidden_states)
            if (cur_beacon_indices == 2).any():
                assert (cur_beacon_indices == 2).sum() <= (cur_beacon_indices == 1).sum(), \
                    "Replicate beacon count exceeds source beacon count in K projection"
                beacon_key_states[:, cur_beacon_indices == 2] = beacon_key_states[:, cur_beacon_indices == 1][:, :(cur_beacon_indices == 2).sum()]
        else:
            beacon_key_states = ordinal_key_states

        if "v" in self.config.beacon_param:
            beacon_value_states = self._project_beacon_value(hidden_states)
            if (cur_beacon_indices == 2).any():
                assert (cur_beacon_indices == 2).sum() <= (cur_beacon_indices == 1).sum(), \
                    "Replicate beacon count exceeds source beacon count in V projection"
                beacon_value_states[:, cur_beacon_indices == 2] = beacon_value_states[:, cur_beacon_indices == 1][:, :(cur_beacon_indices == 2).sum()]
        else:
            beacon_value_states = ordinal_value_states

        current_raw_key = ordinal_key_states[:, current_raw_mask]
        current_raw_value = ordinal_value_states[:, current_raw_mask]
        current_beacon_key_for_attn = beacon_key_states[:, current_beacon_attn_mask]
        current_beacon_value_for_attn = beacon_value_states[:, current_beacon_attn_mask]
        current_beacon_key_for_cache = beacon_key_states[:, current_beacon_cache_mask]
        current_beacon_value_for_cache = beacon_value_states[:, current_beacon_cache_mask]

        current_raw_key = current_raw_key.view(bsz, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2).contiguous()
        current_raw_value = current_raw_value.view(bsz, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2).contiguous()

        if self._use_hybrid_latent_attn():
            current_beacon_key_for_attn = current_beacon_key_for_attn.view(
                bsz, -1, 1, self.beacon_k_latent_global_dim
            ).transpose(1, 2).contiguous()
            current_beacon_value_for_attn = current_beacon_value_for_attn.view(
                bsz, -1, 1, self.beacon_v_latent_global_dim
            ).transpose(1, 2).contiguous()
            current_beacon_key_for_cache = current_beacon_key_for_cache.view(
                bsz, -1, 1, self.beacon_k_latent_global_dim
            ).transpose(1, 2).contiguous()
            current_beacon_value_for_cache = current_beacon_value_for_cache.view(
                bsz, -1, 1, self.beacon_v_latent_global_dim
            ).transpose(1, 2).contiguous()
        else:
            current_beacon_key_for_attn = current_beacon_key_for_attn.view(
                bsz, -1, self.num_key_value_heads, self.head_dim
            ).transpose(1, 2).contiguous()
            current_beacon_value_for_attn = current_beacon_value_for_attn.view(
                bsz, -1, self.num_key_value_heads, self.head_dim
            ).transpose(1, 2).contiguous()
            current_beacon_key_for_cache = current_beacon_key_for_cache.view(
                bsz, -1, self.num_key_value_heads, self.head_dim
            ).transpose(1, 2).contiguous()
            current_beacon_value_for_cache = current_beacon_value_for_cache.view(
                bsz, -1, self.num_key_value_heads, self.head_dim
            ).transpose(1, 2).contiguous()

        past_raw_len = 0 if past_raw_key is None else past_raw_key.shape[2]
        past_beacon_len = 0 if past_beacon_key is None else past_beacon_key.shape[2]
        if raw_prefix_size < 0:
            raw_prefix_size = 0
        if raw_prefix_size > past_raw_len:
            raw_prefix_size = past_raw_len

        past_raw_prefix_key = None if past_raw_key is None else past_raw_key[:, :, :raw_prefix_size]
        past_raw_suffix_key = None if past_raw_key is None else past_raw_key[:, :, raw_prefix_size:]
        past_raw_prefix_value = None if past_raw_value is None else past_raw_value[:, :, :raw_prefix_size]
        past_raw_suffix_value = None if past_raw_value is None else past_raw_value[:, :, raw_prefix_size:]

        raw_key_states = self._cat_kv_states([past_raw_prefix_key, past_raw_suffix_key, current_raw_key])
        raw_value_states = self._cat_kv_states([past_raw_prefix_value, past_raw_suffix_value, current_raw_value])
        beacon_key_states = self._cat_kv_states([past_beacon_key, current_beacon_key_for_attn])
        beacon_value_states = self._cat_kv_states([past_beacon_value, current_beacon_value_for_attn])

        past_raw_suffix_len = 0 if past_raw_suffix_key is None else past_raw_suffix_key.shape[2]
        # Tail layout is [past_raw_suffix, current_q].
        # The cached suffix is always raw; only current_q needs beacon masking.
        tail_beacon_mask = torch.cat(
            [
                torch.zeros(past_raw_suffix_len, dtype=torch.bool, device=hidden_states.device),
                current_beacon_attn_mask.to(device=hidden_states.device),
            ],
            dim=0,
        )

        full_beacon_mask = torch.cat(
            [
                torch.zeros(raw_prefix_size, dtype=torch.bool, device=hidden_states.device),
                torch.ones(past_beacon_len, dtype=torch.bool, device=hidden_states.device),
                tail_beacon_mask,
            ],
            dim=0,
        )
        kv_seq_len = full_beacon_mask.shape[0]
        raw_pos_mask = ~full_beacon_mask
        beacon_pos_mask = full_beacon_mask
        raw_seq_len = int(raw_pos_mask.sum().item())
        beacon_seq_len = int(beacon_pos_mask.sum().item())

        if raw_key_states is None or raw_value_states is None:
            raise ValueError("Raw KV states should never be empty in beacon attention.")
        if raw_key_states.shape[2] != raw_seq_len:
            raise ValueError(f"Raw key length mismatch: tensor={raw_key_states.shape[2]} mask={raw_seq_len}.")
        if raw_value_states.shape[2] != raw_seq_len:
            raise ValueError(f"Raw value length mismatch: tensor={raw_value_states.shape[2]} mask={raw_seq_len}.")
        if beacon_seq_len > 0:
            if beacon_key_states is None or beacon_value_states is None:
                raise ValueError("Beacon KV states missing while beacon positions exist.")
            if beacon_key_states.shape[2] != beacon_seq_len:
                raise ValueError(f"Beacon key length mismatch: tensor={beacon_key_states.shape[2]} mask={beacon_seq_len}.")
            if beacon_value_states.shape[2] != beacon_seq_len:
                raise ValueError(f"Beacon value length mismatch: tensor={beacon_value_states.shape[2]} mask={beacon_seq_len}.")

        if position_ids.shape[-1] != kv_seq_len:
            raise ValueError(
                f"position_ids length should be {kv_seq_len}, but found {position_ids.shape[-1]}."
            )
        query_position_ids = position_ids[:, -q_len:]
        raw_position_ids = position_ids[:, raw_pos_mask]
        beacon_position_ids = position_ids[:, beacon_pos_mask] if beacon_seq_len > 0 else None

        query_raw_states, raw_key_states = self._apply_rotary_query_and_key(
            query_states,
            raw_key_states,
            query_position_ids,
            raw_position_ids,
            self.rotary_emb,
        )
        raw_key_states_expanded = repeat_kv(raw_key_states, self.num_key_value_groups)
        raw_logits = torch.matmul(query_raw_states, raw_key_states_expanded.transpose(2, 3)) / math.sqrt(self.head_dim)

        beacon_logits = None
        if beacon_seq_len > 0:
            if self._use_hybrid_latent_attn():
                # Hybrid latent attention follows Eq. QK^T = (Q' U_K) C_K^T:
                # Q is first RoPE-encoded in original space, then projected by U_K.
                query_beacon_states = self._project_query_to_beacon_latent_with_k_up(query_raw_states)
                beacon_key_states_global = beacon_key_states.squeeze(1)  # [bsz, beacon_len, latent_global]
                beacon_logits = torch.matmul(
                    query_beacon_states, beacon_key_states_global.transpose(1, 2).unsqueeze(1)
                ) / math.sqrt(self.head_dim)
            else:
                query_beacon_states, beacon_key_states = self._apply_rotary_query_and_key(
                    query_states,
                    beacon_key_states,
                    query_position_ids,
                    beacon_position_ids,
                    self.rotary_emb,
                )
                beacon_key_states_expanded = repeat_kv(beacon_key_states, self.num_key_value_groups)
                beacon_logits = torch.matmul(query_beacon_states, beacon_key_states_expanded.transpose(2, 3)) / math.sqrt(self.head_dim)

        attn_weights = hidden_states.new_empty((bsz, self.num_heads, q_len, kv_seq_len), dtype=query_states.dtype)
        attn_weights[..., raw_pos_mask] = raw_logits
        if beacon_logits is not None:
            attn_weights[..., beacon_pos_mask] = beacon_logits

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        raw_attn_weights = attn_weights[..., raw_pos_mask]
        raw_value_states_expanded = repeat_kv(raw_value_states, self.num_key_value_groups)
        attn_output = torch.matmul(raw_attn_weights, raw_value_states_expanded)

        if beacon_seq_len > 0:
            beacon_attn_weights = attn_weights[..., beacon_pos_mask]
            if self._use_hybrid_latent_attn():
                beacon_value_states_global = beacon_value_states.squeeze(1)  # [bsz, beacon_len, latent_global]
                beacon_latent_output = torch.matmul(
                    beacon_attn_weights, beacon_value_states_global
                )  # [bsz, num_heads, q_len, latent_global]

                head_to_kv = torch.arange(self.num_heads, device=hidden_states.device) // self.num_key_value_groups
                v_up_weight = self.beacon_v_latent_up_proj.weight.view(
                    self.num_key_value_heads, self.head_dim, self.beacon_v_latent_global_dim
                )
                grouped_v_up = v_up_weight.index_select(0, head_to_kv)  # [num_heads, head_dim, latent_global]
                beacon_attn_output = torch.einsum("bhql,hdl->bhqd", beacon_latent_output, grouped_v_up)
                if self.beacon_v_latent_up_proj.bias is not None:
                    v_up_bias = self.beacon_v_latent_up_proj.bias.view(self.num_key_value_heads, self.head_dim)
                    grouped_bias = v_up_bias.index_select(0, head_to_kv)  # [num_heads, head_dim]
                    beacon_attn_output = beacon_attn_output + grouped_bias.unsqueeze(0).unsqueeze(2)
            else:
                beacon_value_states_expanded = repeat_kv(beacon_value_states, self.num_key_value_groups)
                beacon_attn_output = torch.matmul(beacon_attn_weights, beacon_value_states_expanded)
            attn_output = attn_output + beacon_attn_output

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj_with_beacon(attn_output, beacon_size, beacon_indices)

        next_past_key_value = make_layer_past_key_value(
            raw_key=current_raw_key,
            raw_value=current_raw_value,
            beacon_key=current_beacon_key_for_cache,
            beacon_value=current_beacon_value_for_cache,
            beacon_size=beacon_size,
            beacon_indices=beacon_indices,
            raw_prefix_size=0,
        )

        if beacon_indices is None:
            accumulated_raw_key = self._cat_kv_states([past_raw_key, current_raw_key])
            accumulated_raw_value = self._cat_kv_states([past_raw_value, current_raw_value])
            accumulated_beacon_key = self._cat_kv_states([past_beacon_key, current_beacon_key_for_cache])
            accumulated_beacon_value = self._cat_kv_states([past_beacon_value, current_beacon_value_for_cache])
            next_past_key_value = make_layer_past_key_value(
                raw_key=accumulated_raw_key,
                raw_value=accumulated_raw_value,
                beacon_key=accumulated_beacon_key,
                beacon_value=accumulated_beacon_value,
                beacon_size=beacon_size,
                beacon_indices=beacon_indices,
                raw_prefix_size=raw_prefix_size,
            )

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, next_past_key_value


class LlamaSdpaAttention(LlamaAttention):
    """
    Llama attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `LlamaAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    # Adapted from LlamaAttention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if self._use_hybrid_latent_attn():
            logger.warning_once(
                "Hybrid latent attention is eager-only. Falling back from SDPA to eager attention."
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        if isinstance(past_key_value, dict):
            (
                past_raw_key,
                past_raw_value,
                past_beacon_key,
                past_beacon_value,
                beacon_size,
                beacon_indices,
                _,
            ) = unpack_layer_past_key_value(past_key_value)
            if past_beacon_key is not None or past_beacon_value is not None or beacon_indices is not None:
                logger.warning_once(
                    "Split beacon cache is not supported in SDPA path. Falling back to eager attention."
                )
                return super().forward(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            past_key_value = (past_raw_key, past_raw_value, beacon_size, beacon_indices)

        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "LlamaModel is using LlamaSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
        bsz, q_len, _ = hidden_states.size()
        kv_seq_len = hidden_states.shape[-2]
        past_key, past_value, beacon_size, beacon_indices = past_key_value
        if past_key is not None:
            past_seq_len = past_key.shape[2]
            kv_seq_len += past_seq_len
        else:
            past_seq_len = 0

        query_states, key_states, value_states = self.qkv_proj_with_beacon(hidden_states, beacon_size, beacon_indices)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # return keys and values before rope
        # NOTE: incrementally return keys and values for beacon efficiency
        past_key_value = (key_states, value_states, beacon_size, beacon_indices)

        if past_key is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key, key_states], dim=2)
            value_states = torch.cat([past_value, value_states], dim=2)

        # In no-beacon native generation, `generate()` expects an accumulated KV cache.
        # Beacon path keeps using incremental cache and is updated by Memory.update_memory.
        if beacon_indices is None:
            past_key_value = (key_states, value_states, beacon_size, beacon_indices)
        
        query_states, key_states = self._apply_rotary_with_beacon(
            query_states,
            key_states,
            position_ids,
            beacon_size,
            beacon_indices,
        )

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
            is_causal=self.is_causal and attention_mask is None and q_len > 1,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj_with_beacon(attn_output, beacon_size, beacon_indices)

        return attn_output, None, past_key_value


class LlamaFlashAttention2(LlamaAttention):
    """
    Llama flash attention module. This module inherits from `LlamaAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if self._use_hybrid_latent_attn():
            logger.warning_once(
                "Hybrid latent attention is eager-only. Falling back from FlashAttention2 to eager attention."
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        if isinstance(past_key_value, dict):
            (
                past_raw_key,
                past_raw_value,
                past_beacon_key,
                past_beacon_value,
                beacon_size,
                beacon_indices,
                _,
            ) = unpack_layer_past_key_value(past_key_value)
            if past_beacon_key is not None or past_beacon_value is not None or beacon_indices is not None:
                logger.warning_once(
                    "Split beacon cache is not supported in FlashAttention2 path. Falling back to eager attention."
                )
                return super().forward(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            past_key_value = (past_raw_key, past_raw_value, beacon_size, beacon_indices)

        output_attentions = False

        bsz, q_len, _ = hidden_states.size()
        kv_seq_len = hidden_states.shape[-2]

        past_key, past_value, beacon_size, beacon_indices = past_key_value
        if past_key is not None:
            past_seq_len = past_key.shape[2]
            kv_seq_len += past_seq_len
        else:
            past_seq_len = 0

        query_states, key_states, value_states = self.qkv_proj_with_beacon(hidden_states, beacon_size, beacon_indices)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # return keys and values before rope
        # NOTE: incrementally return keys and values for beacon efficiency
        past_key_value = (key_states, value_states, beacon_size, beacon_indices)

        if past_key is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key, key_states], dim=2)
            value_states = torch.cat([past_value, value_states], dim=2)

        # In no-beacon native generation, `generate()` expects an accumulated KV cache.
        # Beacon path keeps using incremental cache and is updated by Memory.update_memory.
        if beacon_indices is None:
            past_key_value = (key_states, value_states, beacon_size, beacon_indices)

        query_states, key_states = self._apply_rotary_with_beacon(
            query_states,
            key_states,
            position_ids,
            beacon_size,
            beacon_indices,
        )

        # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
        # to be able to avoid many of these transpose/reshape/view.
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        dropout_rate = self.attention_dropout if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (LlamaRMSNorm handles it correctly)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = self._flash_attention_forward(
            query_states, 
            key_states, 
            value_states, 
            attention_mask, 
            q_len, 
            dropout=dropout_rate
        )

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj_with_beacon(attn_output, beacon_size, beacon_indices)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def _flash_attention_forward(
        self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`float`):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        """
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in LlamaFlashAttention2 __init__.
            causal = self.is_causal and query_length != 1

        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=causal,
            )

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            attn_output = flash_attn_func(
                query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal
            )

        return attn_output

    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )


LLAMA_ATTENTION_CLASSES = {
    "eager": LlamaAttention,
    "sdpa": LlamaSdpaAttention,
    "flash_attention_2": LlamaFlashAttention2,
}


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = LLAMA_ATTENTION_CLASSES[config._attn_implementation](config, layer_idx)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


LLAMA_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`LlamaConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare Llama Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaPreTrainedModel(PreTrainedModel):
    config_class = LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, HeadwiseUpProj):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, GroupedQueryUpProj):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


LLAMA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance;
            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.

            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare Llama Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaModel(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

        # BEACON: add beacon embedding
        self.beacon_embed_tokens = nn.Embedding(1, config.hidden_size, self.padding_idx)
        self.beacon_embed_tokens._is_hf_initialized = True

        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._attn_implementation = config._attn_implementation
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def _init_beacon_embed(self, missing_keys):
        """Initialize the beacon token embedding with that of the eos token."""
        if is_deepspeed_zero3_enabled():
            import deepspeed
            params = [self.beacon_embed_tokens.weight, self.embed_tokens.weight]
            with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
                # deepspeed will initialize the parameters to zero
                if (self.beacon_embed_tokens.weight == 0).all():
                    if self.config.beacon_embed_init == "bos":
                        self.beacon_embed_tokens.weight.data[:] = self.embed_tokens.weight.data[self.config.bos_token_id]
                    elif self.config.beacon_embed_init == "eos":
                        if isinstance(self.config.eos_token_id, list):
                            eos_token_id = self.config.eos_token_id[0]
                        else:
                            eos_token_id = self.config.eos_token_id
                        self.beacon_embed_tokens.weight.data[:] = self.embed_tokens.weight.data[eos_token_id]
                    else:
                        raise NotImplementedError(f"Make sure beacon_embed_init is either eos or bos, found {self.config.beacon_embed_init}")
        else:
            if any("beacon_embed_tokens" in missing_key for missing_key in missing_keys):
                if self.config.beacon_embed_init == "bos":
                    self.beacon_embed_tokens.weight.data[:] = self.embed_tokens.weight.data[self.config.bos_token_id]
                elif self.config.beacon_embed_init == "eos":
                    if isinstance(self.config.eos_token_id, list):
                        eos_token_id = self.config.eos_token_id[0]
                    else:
                        eos_token_id = self.config.eos_token_id
                    self.beacon_embed_tokens.weight.data[:] = self.embed_tokens.weight.data[eos_token_id]
                else:
                    raise NotImplementedError(f"Make sure beacon_embed_init is either eos or bos, found {self.config.beacon_embed_init}")

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # BEACON: always use cache
        use_cache = True

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if past_key_values is None:
            past_key_values = [
                make_layer_past_key_value(
                    raw_key=None,
                    raw_value=None,
                    beacon_key=None,
                    beacon_value=None,
                    beacon_size=0,
                    beacon_indices=None,
                    raw_prefix_size=0,
                )
                for _ in range(self.config.num_hidden_layers)
            ]

        _, _, _, _, beacon_size, beacon_indices, _ = unpack_layer_past_key_value(past_key_values[0])

        # BEACON: separately embed ordinal tokens and beacon tokens because ordinal tokens do not receive gradients
        if beacon_size > 0:
            # NOTE: when beacon_pos == "interleave", the beacon_indices points to all beacon tokens in the current window (cached activations + input_ids), so we shall slice out the part corresponding to the input_ids
            cur_beacon_indices = beacon_indices[-input_ids.shape[1]:]

            ordinal_input_ids = input_ids[:, cur_beacon_indices == 0]
            beacon_input_ids = input_ids[:, cur_beacon_indices > 0]
            ordinal_inputs_embeds = self.embed_tokens(ordinal_input_ids)
            beacon_input_embeds = self.beacon_embed_tokens(beacon_input_ids - self.config.vocab_size)
            # create a new embedding tensor
            inputs_embeds = beacon_input_embeds.new_zeros(*input_ids.shape, beacon_input_embeds.shape[-1])
            inputs_embeds[:, cur_beacon_indices == 0] = ordinal_inputs_embeds
            inputs_embeds[:, cur_beacon_indices > 0] = beacon_input_embeds

        else:
            inputs_embeds = self.embed_tokens(input_ids)

        # embed positions
        hidden_states = inputs_embeds

        # print(f"input_ids:          {input_ids}")
        # print(f"beacon_indices:     {beacon_indices}")
        # print(f"position_ids:       {position_ids}")
        # print(f"attention_mask:\n{attention_mask == 0}")
        # x = input()
        # if x == "s":
        #     return

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        # BEACON: use per-layer cache objects
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # BEACON: slice out the past_key_value of the corresponding layer
            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_value,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class LlamaForCausalLM(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        """Override the default from_pretrained to extend vocab size according to beacon_size."""
        kwargs.update(output_loading_info=True)
        model, loading_info = super().from_pretrained(*args, **kwargs)

        # NOTE: set memory after from_pretrained because there may be another transformer model inside the Memory object, which may cause weird erros during loading
        config = model.config
        model.memory = Memory(
            model_config=config,
            k_seq_dim=2,
            v_seq_dim=2,
        )

        missing_keys = loading_info["missing_keys"]
        # NOTE: the beacon parameters may or may not be loaded from the checkpoint
        # if it is loaded from the checkpoint, we should not re-initilize it
        model.model._init_beacon_embed(missing_keys)
        # initialize weights of possible q,k,v,o,mlp
        for layer in model.model.layers:
            layer.self_attn._init_beacon_proj(missing_keys)

        return model

    def _native_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        loss_weights: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, ModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # when we directly call _native_forward, the past_key_values would be None
        if past_key_values is None:
            # NOTE: set beacon size to 0 to avoid using any beacon parameters, see Qwen2Attention.forward
            past_key_values = [
                make_layer_past_key_value(
                    raw_key=None,
                    raw_value=None,
                    beacon_key=None,
                    beacon_value=None,
                    beacon_size=0,
                    beacon_indices=None,
                    raw_prefix_size=0,
                )
                for _ in range(self.config.num_hidden_layers)
            ]

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        batch_loss = None
        token_loss = None
        loss_denominator = None
        
        if labels is not None:
            loss, batch_loss, token_loss, loss_denominator = compute_loss(logits, labels, shift=False, weights=loss_weights)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return ModelOutput(
            loss=loss,
            batch_loss=batch_loss,
            token_loss=token_loss,
            loss_denominator=loss_denominator,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def _beacon_forward(self, 
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        # t1 = time.time()

        # initialize cache
        self.memory.prepare(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            labels=labels
        )

        # t2 = time.time()

        while not self.memory.finish:

            # t3 = time.time()

            input_ids, attention_mask, position_ids, past_key_values, labels = self.memory.step()
            loss_weights = build_beacon_loss_weights(
                input_ids=input_ids,
                labels=labels,
                beacon_token_id=self.config.vocab_size,
                loss_mode=getattr(self.config, "beacon_loss_mode", None),
                loss_weight=getattr(self.config, "beacon_loss_weight", None),
            )

            # t4 = time.time()

            outputs = self._native_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                labels=labels,
                loss_weights=loss_weights,
            )

            # t5 = time.time()

            # update past_key_values
            self.memory.update_memory(outputs.past_key_values)

            # t6 = time.time()

            if labels is not None:
                # update loss
                self.memory.update_loss(outputs.batch_loss, outputs.loss_denominator)

        # t8 = time.time()

        # output loss, past_key_values, and perplexity
        outputs = self.memory.output(outputs)

        # t9 = time.time()

        # print(f"output time:            {t9-t8}")
        # input()

        return outputs

    def forward(self, **kwargs):
        """Forward computation over a batch of sequences.
        """
        # only allow gradient when training
        with optional_grad_ctx(with_grad=self.training):
            return self._beacon_forward(**kwargs)

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        **kwargs,
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            # Beacon generation only needs the latest position id at decode steps.
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "cache_position": cache_position,
            }
        )

        for extra_key in ("position_embeddings", "num_logits_to_keep"):
            if extra_key in kwargs and kwargs[extra_key] is not None:
                model_inputs[extra_key] = kwargs[extra_key]

        reserved_kwargs = {"position_ids", "use_cache", "position_embeddings", "num_logits_to_keep"}
        for k, v in kwargs.items():
            if k in reserved_kwargs or v is None:
                continue
            if k not in model_inputs:
                model_inputs[k] = v

        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            if isinstance(layer_past, dict):
                reordered_layer = {}
                for key, value in layer_past.items():
                    if key in {"raw_key", "raw_value", "beacon_key", "beacon_value"} and torch.is_tensor(value):
                        reordered_layer[key] = value.index_select(0, beam_idx.to(value.device))
                    else:
                        reordered_layer[key] = value
                reordered_past += (reordered_layer,)
            else:
                reordered_past += (
                    tuple(
                        past_state.index_select(0, beam_idx.to(past_state.device))
                        if torch.is_tensor(past_state) and i in (0, 1)
                        else past_state
                        for i, past_state in enumerate(layer_past)
                    ),
                )
        return reordered_past
