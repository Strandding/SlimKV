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
from .configuration_llama import LlamaConfig


if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

    _flash_supports_window_size = "window_size" in list(inspect.signature(flash_attn_func).parameters)

from .memory import LlamaMemory, ReconstructionCache, ReconstructionCacheUpdate
from .fa2_decode import eligible as fa2_eligible, decode as fa2_decode
from .latent_decode import uses_latent_decode, uses_fa2_decode
from ..modeling_utils import optional_grad_ctx, compute_loss, build_beacon_loss_weights, get_rope, ModelOutput


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


class HeadwiseUpProj(nn.Module):
    """Grouped-coupled up projection for low-rank beacon K/V.

    Each group couples `group_size` KV heads with one matrix:
    (group_size * latent_head_dim) -> (group_size * head_dim).
    """

    def __init__(
        self,
        num_heads: int,
        latent_head_dim: int,
        head_dim: int,
        group_size: int = 1,
        bias: bool = True,
        dtype=None,
        device=None,
    ):
        super().__init__()
        if group_size <= 0:
            raise ValueError(f"group_size must be positive, found {group_size}.")
        if num_heads % group_size != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must be divisible by group_size ({group_size})."
            )
        self.num_heads = num_heads
        self.latent_head_dim = latent_head_dim
        self.head_dim = head_dim
        self.group_size = group_size
        self.num_groups = num_heads // group_size
        self.group_input_dim = group_size * latent_head_dim
        self.group_output_dim = group_size * head_dim
        self.weight = nn.Parameter(
            torch.empty(self.num_groups * self.group_output_dim, self.group_input_dim, dtype=dtype, device=device)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(self.num_groups * self.group_output_dim, dtype=dtype, device=device))
        else:
            self.register_parameter("bias", None)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        hidden_states = hidden_states.reshape(
            batch_size, seq_len, self.num_groups, self.group_input_dim
        )
        up_weight = self.weight.view(self.num_groups, self.group_output_dim, self.group_input_dim)
        projected = torch.einsum("bsgi,goi->bsgo", hidden_states, up_weight)
        if self.bias is not None:
            up_bias = self.bias.view(self.num_groups, self.group_output_dim)
            projected = projected + up_bias.unsqueeze(0).unsqueeze(0)
        return projected.reshape(batch_size, seq_len, self.num_heads * self.head_dim)


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
        self.beacon_k_compression = getattr(config, "beacon_k_compression", "none")
        self.beacon_v_compression = getattr(config, "beacon_v_compression", "none")
        self.beacon_chunk_fixed_rope_for_beacon_k = bool(
            getattr(config, "beacon_chunk_fixed_rope_for_beacon_k", False)
        )
        self.beacon_enhance_k_pos_residual = bool(getattr(config, "beacon_enhance_k_pos_residual", False))
        self.beacon_probe_prev_raw8_attn = bool(getattr(config, "beacon_probe_prev_raw8_attn", False))
        probe_prev_raw_tokens = getattr(config, "beacon_probe_prev_raw_tokens", 8)
        self.beacon_probe_prev_raw_tokens = int(probe_prev_raw_tokens) if probe_prev_raw_tokens is not None else 8
        if self.beacon_probe_prev_raw_tokens <= 0:
            self.beacon_probe_prev_raw_tokens = 8
        self.beacon_probe_chunk2_heatmap = bool(getattr(config, "beacon_probe_chunk2_heatmap", False))
        probe_chunk_idx = getattr(config, "beacon_probe_chunk_idx", 2)
        try:
            probe_chunk_idx = int(probe_chunk_idx)
        except (TypeError, ValueError):
            probe_chunk_idx = 2
        if probe_chunk_idx <= 0:
            probe_chunk_idx = 2
        self.beacon_probe_chunk_idx = probe_chunk_idx
        self._beacon_probe_records = []
        self._beacon_chunk2_heatmap_record = None
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        for _name, _val in [("beacon_k_compression", self.beacon_k_compression),
                            ("beacon_v_compression", self.beacon_v_compression)]:
            if _val not in ["none", "lowrank_svd"]:
                raise ValueError(
                    f"Unsupported {_name}: {_val}. Expected one of ['none', 'lowrank_svd']."
                )
        self._validate_beacon_pos_residual_config()

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

        self.rotary_emb = get_rope(self.head_dim, config.rope_theta, config.max_position_embeddings, getattr(config, "rope_scaling", None))

        # NOTE: add extra parameters for beacon tokens
        # skip post initialization to speed up loading
        if "q" in config.beacon_param:
            self.beacon_q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=self.q_proj.bias is not None)
            # NOTE: initialize the beacon parameters as zero
            self.beacon_q_proj.weight.data.zero_()
            self.beacon_q_proj._is_hf_initialized = True
        if "k" in config.beacon_param:
            if self._use_lowrank_beacon_k():
                if self._use_shared_kv_down_proj():
                    # shared down_proj will be created in the V block below
                    _, self.beacon_k_up_proj = self._build_lowrank_beacon_k_layers()
                else:
                    self.beacon_k_down_proj, self.beacon_k_up_proj = self._build_lowrank_beacon_k_layers()
            else:
                self.beacon_k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=self.k_proj.bias is not None)
                self.beacon_k_proj.weight.data.zero_()
                self.beacon_k_proj._is_hf_initialized = True
        if "v" in config.beacon_param:
            if self._use_lowrank_beacon_v():
                if self._use_shared_kv_down_proj():
                    # create ONE shared down_proj for both K and V
                    self.beacon_kv_down_proj, self.beacon_v_up_proj = self._build_lowrank_beacon_v_layers()
                else:
                    self.beacon_v_down_proj, self.beacon_v_up_proj = self._build_lowrank_beacon_v_layers()
            else:
                self.beacon_v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=self.v_proj.bias is not None)
                self.beacon_v_proj.weight.data.zero_()
                self.beacon_v_proj._is_hf_initialized = True
        if "o" in config.beacon_param:
            self.beacon_o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=self.o_proj.bias is not None)
            self.beacon_o_proj.weight.data.zero_()
            self.beacon_o_proj._is_hf_initialized = True

    # ---- shared KV down-proj helper ----

    def _use_shared_kv_down_proj(self):
        return (
            getattr(self.config, "beacon_shared_kv_down_proj", False)
            and self._use_lowrank_beacon_k()
            and self._use_lowrank_beacon_v()
            and self._get_beacon_k_latent_head_dim() == self._get_beacon_v_latent_head_dim()
            and getattr(self.config, "beacon_skip_rope_for_beacon_k", False)
        )

    # ---- K low-rank helpers ----

    def _use_headwise_up_proj(self):
        return getattr(self.config, "beacon_headwise_up_proj", False)

    def _get_beacon_up_proj_group_size(self):
        group_size = getattr(self.config, "beacon_up_proj_group_size", 1)
        if group_size is None:
            group_size = 1
        if group_size <= 0:
            raise ValueError(f"beacon_up_proj_group_size must be positive, found {group_size}.")
        if self.num_key_value_heads % group_size != 0:
            raise ValueError(
                f"beacon_up_proj_group_size ({group_size}) must divide num_key_value_heads "
                f"({self.num_key_value_heads})."
            )
        return group_size

    def _use_lowrank_beacon_k(self):
        return "k" in self.config.beacon_param and self.beacon_k_compression == "lowrank_svd"

    def _get_layerwise_latent_head_dim(self, field_name):
        latent_head_dim_list = getattr(self.config, field_name, None)
        if latent_head_dim_list is None:
            return None
        if not isinstance(latent_head_dim_list, (list, tuple)):
            raise ValueError(
                f"{field_name} must be a list/tuple, found {type(latent_head_dim_list).__name__}."
            )
        if len(latent_head_dim_list) != self.config.num_hidden_layers:
            raise ValueError(
                f"{field_name} length must equal num_hidden_layers "
                f"({self.config.num_hidden_layers}), found {len(latent_head_dim_list)}."
            )
        if self.layer_idx is None:
            logger.warning_once(
                f"layer_idx is None while {field_name} is set; fallback to layer 0 latent dim."
            )
            latent_head_dim = latent_head_dim_list[0]
        else:
            latent_head_dim = latent_head_dim_list[self.layer_idx]
        if not isinstance(latent_head_dim, int):
            raise ValueError(
                f"Each element in {field_name} must be int, "
                f"but layer {self.layer_idx} is {type(latent_head_dim).__name__}."
            )
        return latent_head_dim

    def _get_beacon_k_latent_head_dim(self):
        latent_head_dim = self._get_layerwise_latent_head_dim("beacon_k_latent_head_dim_list")
        if latent_head_dim is None:
            latent_head_dim = getattr(self.config, "beacon_k_latent_head_dim", None)
        if latent_head_dim is None:
            latent_head_dim = self.head_dim
        if latent_head_dim <= 0:
            raise ValueError(f"beacon_k_latent_head_dim must be positive, found {latent_head_dim}.")
        if latent_head_dim > self.head_dim:
            raise ValueError(
                f"beacon_k_latent_head_dim must be <= head_dim ({self.head_dim}), found {latent_head_dim}."
            )
        return latent_head_dim

    def _build_lowrank_beacon_k_layers(self, latent_head_dim=None, dtype=None, device=None):
        if latent_head_dim is None:
            latent_head_dim = self._get_beacon_k_latent_head_dim()
        if latent_head_dim <= 0:
            raise ValueError(f"beacon_k_latent_head_dim must be positive, found {latent_head_dim}.")
        if latent_head_dim > self.head_dim:
            raise ValueError(
                f"beacon_k_latent_head_dim must be <= head_dim ({self.head_dim}), found {latent_head_dim}."
            )

        global_rank = self.num_key_value_heads * latent_head_dim
        out_features = self.num_key_value_heads * self.head_dim
        if global_rank > min(self.hidden_size, out_features):
            raise ValueError(
                f"Global rank {global_rank} exceeds the maximum feasible rank "
                f"{min(self.hidden_size, out_features)} for beacon_k_proj."
            )

        down_proj = nn.Linear(self.hidden_size, global_rank, bias=False, dtype=dtype, device=device)
        if self._use_headwise_up_proj():
            up_proj = HeadwiseUpProj(
                num_heads=self.num_key_value_heads,
                latent_head_dim=latent_head_dim,
                head_dim=self.head_dim,
                group_size=self._get_beacon_up_proj_group_size(),
                bias=self.k_proj.bias is not None,
                dtype=dtype,
                device=device,
            )
        else:
            up_proj = nn.Linear(global_rank, out_features, bias=self.k_proj.bias is not None, dtype=dtype, device=device)
        return down_proj, up_proj

    def _project_beacon_key(self, hidden_states, return_latent=False):
        if self._use_lowrank_beacon_k():
            if self._use_shared_kv_down_proj():
                latent = self.beacon_kv_down_proj(hidden_states)
            else:
                latent = self.beacon_k_down_proj(hidden_states)
            full = self.beacon_k_up_proj(latent)
            return (full, latent) if return_latent else full
        return self.beacon_k_proj(hidden_states)

    # ---- V low-rank helpers ----

    def _use_lowrank_beacon_v(self):
        return "v" in self.config.beacon_param and self.beacon_v_compression == "lowrank_svd"

    def _get_beacon_v_latent_head_dim(self):
        latent_head_dim = self._get_layerwise_latent_head_dim("beacon_v_latent_head_dim_list")
        if latent_head_dim is None:
            latent_head_dim = getattr(self.config, "beacon_v_latent_head_dim", None)
        if latent_head_dim is None:
            latent_head_dim = self.head_dim
        if latent_head_dim <= 0:
            raise ValueError(f"beacon_v_latent_head_dim must be positive, found {latent_head_dim}.")
        if latent_head_dim > self.head_dim:
            raise ValueError(
                f"beacon_v_latent_head_dim must be <= head_dim ({self.head_dim}), found {latent_head_dim}."
            )
        return latent_head_dim

    def _build_lowrank_beacon_v_layers(self, latent_head_dim=None, dtype=None, device=None):
        if latent_head_dim is None:
            latent_head_dim = self._get_beacon_v_latent_head_dim()
        if latent_head_dim <= 0:
            raise ValueError(f"beacon_v_latent_head_dim must be positive, found {latent_head_dim}.")
        if latent_head_dim > self.head_dim:
            raise ValueError(
                f"beacon_v_latent_head_dim must be <= head_dim ({self.head_dim}), found {latent_head_dim}."
            )

        global_rank = self.num_key_value_heads * latent_head_dim
        out_features = self.num_key_value_heads * self.head_dim
        if global_rank > min(self.hidden_size, out_features):
            raise ValueError(
                f"Global rank {global_rank} exceeds the maximum feasible rank "
                f"{min(self.hidden_size, out_features)} for beacon_v_proj."
            )

        down_proj = nn.Linear(self.hidden_size, global_rank, bias=False, dtype=dtype, device=device)
        if self._use_headwise_up_proj():
            up_proj = HeadwiseUpProj(
                num_heads=self.num_key_value_heads,
                latent_head_dim=latent_head_dim,
                head_dim=self.head_dim,
                group_size=self._get_beacon_up_proj_group_size(),
                bias=self.v_proj.bias is not None,
                dtype=dtype,
                device=device,
            )
        else:
            up_proj = nn.Linear(global_rank, out_features, bias=self.v_proj.bias is not None, dtype=dtype, device=device)
        return down_proj, up_proj

    def _project_beacon_value(self, hidden_states, return_latent=False):
        if self._use_lowrank_beacon_v():
            if self._use_shared_kv_down_proj():
                latent = self.beacon_kv_down_proj(hidden_states)
            else:
                latent = self.beacon_v_down_proj(hidden_states)
            full = self.beacon_v_up_proj(latent)
            return (full, latent) if return_latent else full
        return self.beacon_v_proj(hidden_states)

    # ---- SVD conversion: K ----

    def convert_beacon_k_proj_to_lowrank(
        self,
        latent_head_dim,
        svd_device="cpu",
        svd_partition="global",
        return_singular_values=False,
        svd_factors=None,
    ):
        if "k" not in self.config.beacon_param:
            raise ValueError("Cannot convert beacon K projection because `k` is not enabled in beacon_param.")
        if self._use_lowrank_beacon_k():
            raise ValueError("beacon_k_proj is already using low-rank parameterization.")
        if not hasattr(self, "beacon_k_proj"):
            raise ValueError("Dense beacon_k_proj was not found on the source checkpoint.")
        if svd_partition not in {"global", "per_head"}:
            raise ValueError(
                f"Unsupported svd_partition={svd_partition}. Expected one of {{'global', 'per_head'}}."
            )

        svd_device = torch.device(svd_device)
        dense_weight = self.beacon_k_proj.weight.detach().float().to(svd_device)
        global_rank = self.num_key_value_heads * latent_head_dim
        max_rank = min(dense_weight.shape)
        if global_rank > max_rank:
            raise ValueError(
                f"Requested global rank {global_rank} exceeds the maximum feasible rank {max_rank}."
            )

        down_proj, up_proj = self._build_lowrank_beacon_k_layers(
            latent_head_dim=latent_head_dim,
            dtype=self.beacon_k_proj.weight.dtype,
            device=self.beacon_k_proj.weight.device,
        )
        if svd_partition == "global":
            if svd_factors is None:
                u, s, vh = torch.linalg.svd(dense_weight, full_matrices=False)
            else:
                u, s, vh = svd_factors
            singular_values = s.detach().cpu()
            u = u[:, :global_rank]
            s = s[:global_rank]
            vh = vh[:global_rank, :]
            sqrt_s = s.sqrt()
            up_weight = u * sqrt_s.unsqueeze(0)
            down_weight = sqrt_s.unsqueeze(1) * vh

            down_proj.weight.data.copy_(down_weight.to(dtype=down_proj.weight.dtype, device=down_proj.weight.device))
            if isinstance(up_proj, HeadwiseUpProj):
                up_weight = up_weight.to(dtype=up_proj.weight.dtype, device=up_proj.weight.device)
                grouped_up = up_proj.weight.view(
                    up_proj.num_groups, up_proj.group_output_dim, up_proj.group_input_dim
                )
                for group_idx in range(up_proj.num_groups):
                    row_start = group_idx * up_proj.group_output_dim
                    row_end = row_start + up_proj.group_output_dim
                    col_start = group_idx * up_proj.group_input_dim
                    col_end = col_start + up_proj.group_input_dim
                    grouped_up[group_idx].copy_(up_weight[row_start:row_end, col_start:col_end])
                if up_proj.bias is not None and self.beacon_k_proj.bias is not None:
                    dense_bias = self.beacon_k_proj.bias.data.to(dtype=up_proj.bias.dtype, device=up_proj.bias.device)
                    grouped_bias = up_proj.bias.view(up_proj.num_groups, up_proj.group_output_dim)
                    for group_idx in range(up_proj.num_groups):
                        row_start = group_idx * up_proj.group_output_dim
                        row_end = row_start + up_proj.group_output_dim
                        grouped_bias[group_idx].copy_(dense_bias[row_start:row_end])
            else:
                up_proj.weight.data.copy_(up_weight.to(dtype=up_proj.weight.dtype, device=up_proj.weight.device))
                if up_proj.bias is not None and self.beacon_k_proj.bias is not None:
                    up_proj.bias.data.copy_(self.beacon_k_proj.bias.data.to(dtype=up_proj.bias.dtype, device=up_proj.bias.device))
        else:
            if not isinstance(up_proj, HeadwiseUpProj) or up_proj.group_size != 1:
                raise ValueError(
                    "svd_partition='per_head' requires headwise up projection with beacon_up_proj_group_size=1."
                )
            down_weight = torch.empty(
                global_rank, self.hidden_size, dtype=dense_weight.dtype, device=svd_device
            )
            grouped_up = up_proj.weight.view(up_proj.num_groups, up_proj.group_output_dim, up_proj.group_input_dim)
            singular_values = []
            for head_idx in range(self.num_key_value_heads):
                row_start = head_idx * self.head_dim
                row_end = row_start + self.head_dim
                head_weight = dense_weight[row_start:row_end, :]
                u, s, vh = torch.linalg.svd(head_weight, full_matrices=False)
                singular_values.append(s.detach().cpu())
                u = u[:, :latent_head_dim]
                s = s[:latent_head_dim]
                vh = vh[:latent_head_dim, :]
                sqrt_s = s.sqrt()
                up_head = u * sqrt_s.unsqueeze(0)
                down_head = sqrt_s.unsqueeze(1) * vh

                latent_start = head_idx * latent_head_dim
                latent_end = latent_start + latent_head_dim
                down_weight[latent_start:latent_end, :] = down_head
                grouped_up[head_idx].copy_(up_head.to(dtype=up_proj.weight.dtype, device=up_proj.weight.device))

            down_proj.weight.data.copy_(down_weight.to(dtype=down_proj.weight.dtype, device=down_proj.weight.device))
            if up_proj.bias is not None and self.beacon_k_proj.bias is not None:
                dense_bias = self.beacon_k_proj.bias.data.to(dtype=up_proj.bias.dtype, device=up_proj.bias.device)
                grouped_bias = up_proj.bias.view(up_proj.num_groups, up_proj.group_output_dim)
                for head_idx in range(self.num_key_value_heads):
                    row_start = head_idx * self.head_dim
                    row_end = row_start + self.head_dim
                    grouped_bias[head_idx].copy_(dense_bias[row_start:row_end])
            singular_values = torch.cat(singular_values, dim=0)

        del dense_weight
        if svd_device.type == "cuda":
            torch.cuda.empty_cache()

        del self.beacon_k_proj
        self.config.beacon_k_compression = "lowrank_svd"
        self.config.beacon_k_latent_head_dim = latent_head_dim
        self.beacon_k_compression = "lowrank_svd"
        self.beacon_k_down_proj = down_proj
        self.beacon_k_up_proj = up_proj
        if return_singular_values:
            return {
                "global_rank": global_rank,
                "singular_values": singular_values,
            }
        return global_rank

    # ---- SVD conversion: V ----

    def convert_beacon_v_proj_to_lowrank(
        self,
        latent_head_dim,
        svd_device="cpu",
        svd_partition="global",
        return_singular_values=False,
        svd_factors=None,
    ):
        if "v" not in self.config.beacon_param:
            raise ValueError("Cannot convert beacon V projection because `v` is not enabled in beacon_param.")
        if self._use_lowrank_beacon_v():
            raise ValueError("beacon_v_proj is already using low-rank parameterization.")
        if not hasattr(self, "beacon_v_proj"):
            raise ValueError("Dense beacon_v_proj was not found on the source checkpoint.")
        if svd_partition not in {"global", "per_head"}:
            raise ValueError(
                f"Unsupported svd_partition={svd_partition}. Expected one of {{'global', 'per_head'}}."
            )

        svd_device = torch.device(svd_device)
        dense_weight = self.beacon_v_proj.weight.detach().float().to(svd_device)
        global_rank = self.num_key_value_heads * latent_head_dim
        max_rank = min(dense_weight.shape)
        if global_rank > max_rank:
            raise ValueError(
                f"Requested global rank {global_rank} exceeds the maximum feasible rank {max_rank}."
            )

        down_proj, up_proj = self._build_lowrank_beacon_v_layers(
            latent_head_dim=latent_head_dim,
            dtype=self.beacon_v_proj.weight.dtype,
            device=self.beacon_v_proj.weight.device,
        )
        if svd_partition == "global":
            if svd_factors is None:
                u, s, vh = torch.linalg.svd(dense_weight, full_matrices=False)
            else:
                u, s, vh = svd_factors
            singular_values = s.detach().cpu()
            u = u[:, :global_rank]
            s = s[:global_rank]
            vh = vh[:global_rank, :]
            sqrt_s = s.sqrt()
            up_weight = u * sqrt_s.unsqueeze(0)
            down_weight = sqrt_s.unsqueeze(1) * vh

            down_proj.weight.data.copy_(down_weight.to(dtype=down_proj.weight.dtype, device=down_proj.weight.device))
            if isinstance(up_proj, HeadwiseUpProj):
                up_weight = up_weight.to(dtype=up_proj.weight.dtype, device=up_proj.weight.device)
                grouped_up = up_proj.weight.view(
                    up_proj.num_groups, up_proj.group_output_dim, up_proj.group_input_dim
                )
                for group_idx in range(up_proj.num_groups):
                    row_start = group_idx * up_proj.group_output_dim
                    row_end = row_start + up_proj.group_output_dim
                    col_start = group_idx * up_proj.group_input_dim
                    col_end = col_start + up_proj.group_input_dim
                    grouped_up[group_idx].copy_(up_weight[row_start:row_end, col_start:col_end])
                if up_proj.bias is not None and self.beacon_v_proj.bias is not None:
                    dense_bias = self.beacon_v_proj.bias.data.to(dtype=up_proj.bias.dtype, device=up_proj.bias.device)
                    grouped_bias = up_proj.bias.view(up_proj.num_groups, up_proj.group_output_dim)
                    for group_idx in range(up_proj.num_groups):
                        row_start = group_idx * up_proj.group_output_dim
                        row_end = row_start + up_proj.group_output_dim
                        grouped_bias[group_idx].copy_(dense_bias[row_start:row_end])
            else:
                up_proj.weight.data.copy_(up_weight.to(dtype=up_proj.weight.dtype, device=up_proj.weight.device))
                if up_proj.bias is not None and self.beacon_v_proj.bias is not None:
                    up_proj.bias.data.copy_(self.beacon_v_proj.bias.data.to(dtype=up_proj.bias.dtype, device=up_proj.bias.device))
        else:
            if not isinstance(up_proj, HeadwiseUpProj) or up_proj.group_size != 1:
                raise ValueError(
                    "svd_partition='per_head' requires headwise up projection with beacon_up_proj_group_size=1."
                )
            down_weight = torch.empty(
                global_rank, self.hidden_size, dtype=dense_weight.dtype, device=svd_device
            )
            grouped_up = up_proj.weight.view(up_proj.num_groups, up_proj.group_output_dim, up_proj.group_input_dim)
            singular_values = []
            for head_idx in range(self.num_key_value_heads):
                row_start = head_idx * self.head_dim
                row_end = row_start + self.head_dim
                head_weight = dense_weight[row_start:row_end, :]
                u, s, vh = torch.linalg.svd(head_weight, full_matrices=False)
                singular_values.append(s.detach().cpu())
                u = u[:, :latent_head_dim]
                s = s[:latent_head_dim]
                vh = vh[:latent_head_dim, :]
                sqrt_s = s.sqrt()
                up_head = u * sqrt_s.unsqueeze(0)
                down_head = sqrt_s.unsqueeze(1) * vh

                latent_start = head_idx * latent_head_dim
                latent_end = latent_start + latent_head_dim
                down_weight[latent_start:latent_end, :] = down_head
                grouped_up[head_idx].copy_(up_head.to(dtype=up_proj.weight.dtype, device=up_proj.weight.device))

            down_proj.weight.data.copy_(down_weight.to(dtype=down_proj.weight.dtype, device=down_proj.weight.device))
            if up_proj.bias is not None and self.beacon_v_proj.bias is not None:
                dense_bias = self.beacon_v_proj.bias.data.to(dtype=up_proj.bias.dtype, device=up_proj.bias.device)
                grouped_bias = up_proj.bias.view(up_proj.num_groups, up_proj.group_output_dim)
                for head_idx in range(self.num_key_value_heads):
                    row_start = head_idx * self.head_dim
                    row_end = row_start + self.head_dim
                    grouped_bias[head_idx].copy_(dense_bias[row_start:row_end])
            singular_values = torch.cat(singular_values, dim=0)

        del dense_weight
        if svd_device.type == "cuda":
            torch.cuda.empty_cache()

        del self.beacon_v_proj
        self.config.beacon_v_compression = "lowrank_svd"
        self.config.beacon_v_latent_head_dim = latent_head_dim
        self.beacon_v_compression = "lowrank_svd"
        self.beacon_v_down_proj = down_proj
        self.beacon_v_up_proj = up_proj
        if return_singular_values:
            return {
                "global_rank": global_rank,
                "singular_values": singular_values,
            }
        return global_rank

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
            if "k" in beacon_param and not self._use_lowrank_beacon_k():
                params = [self.beacon_k_proj.weight, self.k_proj.weight]
                if self.k_proj.bias is not None:
                    params.extend([self.beacon_k_proj.bias, self.k_proj.bias])
                with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
                    # FIXME: after deepspeed initialization, some weights becomes non-zero, but there are rows that are full of zeros
                    if (self.beacon_k_proj.weight.sum(-1) == 0).any() or (self.beacon_k_proj.weight > 1e29).any():
                        self.beacon_k_proj.weight.data[:] = self.k_proj.weight.data
                        if self.k_proj.bias is not None:
                            self.beacon_k_proj.bias.data[:] = self.k_proj.bias.data
            if "v" in beacon_param and not self._use_lowrank_beacon_v():
                params = [self.beacon_v_proj.weight, self.v_proj.weight]
                if self.v_proj.bias is not None:
                    params.extend([self.beacon_v_proj.bias, self.v_proj.bias])
                with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
                    # FIXME: after deepspeed initialization, some weights becomes non-zero, but there are rows that are full of zeros
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
            if "k" in beacon_param and not self._use_lowrank_beacon_k() and any("beacon_k_proj" in missing_key for missing_key in missing_keys):
                # if (self.beacon_k_proj.weight == 0).all():
                    self.beacon_k_proj.weight.data[:] = self.k_proj.weight.data
                    if self.k_proj.bias is not None:
                        self.beacon_k_proj.bias.data[:] = self.k_proj.bias.data
            if "v" in beacon_param and not self._use_lowrank_beacon_v() and any("beacon_v_proj" in missing_key for missing_key in missing_keys):
                # if (self.beacon_v_proj.weight == 0).all():
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

    def pop_beacon_probe_records(self):
        if not self._beacon_probe_records:
            return []
        records = self._beacon_probe_records
        self._beacon_probe_records = []
        return records

    def clear_beacon_probe_records(self):
        self._beacon_probe_records = []

    def pop_beacon_chunk2_heatmap_record(self):
        if self._beacon_chunk2_heatmap_record is None:
            return None
        record = self._beacon_chunk2_heatmap_record
        self._beacon_chunk2_heatmap_record = None
        return record

    def clear_beacon_chunk2_heatmap_record(self):
        self._beacon_chunk2_heatmap_record = None

    def _validate_beacon_pos_residual_config(self):
        if self.beacon_chunk_fixed_rope_for_beacon_k:
            if getattr(self.config, "beacon_parallel_window", 1) > 1:
                raise ValueError(
                    "beacon_chunk_fixed_rope_for_beacon_k does not support beacon_parallel_window>1."
                )
            if getattr(self.config, "beacon_pos", None) != "interleave":
                raise ValueError(
                    "beacon_chunk_fixed_rope_for_beacon_k currently supports beacon_pos='interleave' only."
                )
            if "k" not in getattr(self.config, "beacon_param", []):
                raise ValueError(
                    "beacon_chunk_fixed_rope_for_beacon_k requires 'k' in beacon_param."
                )
        if not self.beacon_enhance_k_pos_residual:
            return
        if not getattr(self.config, "beacon_skip_rope_for_beacon_k", False):
            raise ValueError(
                "beacon_enhance_k_pos_residual=True requires beacon_skip_rope_for_beacon_k=True."
            )
        if getattr(self.config, "beacon_parallel_window", 1) > 1:
            raise ValueError(
                "beacon_enhance_k_pos_residual does not support beacon_parallel_window>1."
            )
        if getattr(self.config, "beacon_pos", None) != "interleave":
            raise ValueError(
                "beacon_enhance_k_pos_residual currently supports beacon_pos='interleave' only."
            )
        if "k" not in getattr(self.config, "beacon_param", []):
            raise ValueError(
                "beacon_enhance_k_pos_residual requires 'k' in beacon_param."
            )

    def _should_enhance_beacon_k_pos_residual(self, beacon_size, beacon_indices):
        return (
            self.beacon_enhance_k_pos_residual
            and beacon_size is not None
            and beacon_size > 0
            and beacon_indices is not None
        )

    def _should_apply_chunk_fixed_beacon_k_rope(self, beacon_size, beacon_indices):
        return (
            self.beacon_chunk_fixed_rope_for_beacon_k
            and beacon_size is not None
            and beacon_size > 0
            and beacon_indices is not None
        )

    def _apply_chunk_fixed_rope_to_current_beacon_k(
        self,
        key_states: torch.Tensor,
        position_ids: torch.LongTensor,
        beacon_indices: torch.LongTensor,
        q_len: int,
        current_key_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply RoPE to current-step beacon K with one fixed anchor position per chunk.

        Anchor position: the first token position in the current chunk.
        """
        if beacon_indices.dim() != 1:
            raise ValueError(
                f"beacon_indices must be 1D, found shape={tuple(beacon_indices.shape)}"
            )
        if beacon_indices.shape[0] < q_len:
            raise ValueError(
                f"beacon_indices length {beacon_indices.shape[0]} is smaller than q_len {q_len}."
            )

        cur_beacon_indices = beacon_indices[-q_len:].to(device=key_states.device)
        beacon_positions = torch.nonzero(cur_beacon_indices == 1, as_tuple=False).flatten()
        if beacon_positions.numel() == 0:
            return key_states

        current_start = key_states.shape[2] - q_len
        anchor_local = 0
        anchor_global = current_start + anchor_local

        beacon_count = beacon_positions.numel()
        anchor_pos = position_ids[:, anchor_global].unsqueeze(1).expand(-1, beacon_count)
        selected_current = current_key_states.index_select(2, beacon_positions) if current_key_states is not None else key_states.index_select(2, beacon_positions + current_start)

        # Reuse rotary_emb implementation: rotate selected beacon K with fixed anchor positions.
        dummy_q = torch.zeros(
            (selected_current.shape[0], self.num_heads, selected_current.shape[2], self.head_dim),
            dtype=selected_current.dtype,
            device=selected_current.device,
        )
        _, selected_rot = self.rotary_emb(dummy_q, selected_current, anchor_pos)

        if current_key_states is not None:
            current_key_states.index_copy_(2, beacon_positions, selected_rot)
        key_states.index_copy_(2, beacon_positions + current_start, selected_rot)
        return key_states

    def _enhance_beacon_key_with_raw_pos_residual_from_rope(
        self,
        key_states: torch.Tensor,
        key_states_rope: torch.Tensor,
        beacon_indices: torch.LongTensor,
        q_len: int,
        current_key_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Enhance current-step beacon K using the nearest previous raw-token RoPE residual.

        For each beacon in the current step, take only the last raw token before it:
            delta = (k_rope - k) at that raw token,
            k_beacon <- k_beacon + delta.

        This reuses the already computed `key_states_rope` to avoid an extra RoPE call.
        """
        if beacon_indices.dim() != 1:
            raise ValueError(
                f"beacon_indices must be 1D, found shape={tuple(beacon_indices.shape)}"
            )
        if beacon_indices.shape[0] < q_len:
            raise ValueError(
                f"beacon_indices length {beacon_indices.shape[0]} is smaller than q_len {q_len}."
            )

        cur_beacon_indices = beacon_indices[-q_len:].to(device=key_states.device)
        beacon_positions = torch.nonzero(cur_beacon_indices == 1, as_tuple=False).flatten()
        if beacon_positions.numel() == 0:
            return key_states

        current_start = key_states.shape[2] - q_len
        raw_mask = cur_beacon_indices == 0

        token_positions = torch.arange(q_len, device=key_states.device)
        # previous raw position for each token (or -1 if none)
        last_raw = torch.where(raw_mask, token_positions, token_positions.new_full((q_len,), -1))
        last_raw = torch.cummax(last_raw, dim=0).values
        last_raw_prev = torch.cat([token_positions.new_full((1,), -1), last_raw[:-1]], dim=0)

        residual_cur = key_states_rope[:, :, current_start:, :] - key_states[:, :, current_start:, :]
        prev_raw_idx = last_raw_prev.index_select(0, beacon_positions)
        valid_prev_raw = prev_raw_idx >= 0
        if not valid_prev_raw.any():
            return key_states

        valid_idx = torch.nonzero(valid_prev_raw, as_tuple=False).flatten()
        valid_beacon_pos = beacon_positions.index_select(0, valid_idx)
        valid_beacon_pos_global = valid_beacon_pos + current_start

        valid_prev_raw_idx = prev_raw_idx.index_select(0, valid_idx)
        delta = residual_cur.index_select(2, valid_prev_raw_idx)

        updated_key = key_states.index_select(2, valid_beacon_pos_global) + delta
        key_states.index_copy_(2, valid_beacon_pos_global, updated_key)
        if current_key_states is not None and current_key_states is not key_states:
            updated_current = current_key_states.index_select(2, valid_beacon_pos) + delta
            current_key_states.index_copy_(2, valid_beacon_pos, updated_current)

        return key_states

    def _collect_beacon_prev_raw_attn_probe(
        self,
        attn_probs: torch.Tensor,
        beacon_indices: Optional[torch.LongTensor],
        q_len: int,
        kv_seq_len: int,
    ) -> None:
        if not self.beacon_probe_prev_raw8_attn or beacon_indices is None or q_len <= 0:
            return
        if beacon_indices.dim() != 1 or beacon_indices.shape[0] < q_len:
            return

        cur_beacon_indices = beacon_indices[-q_len:].to(device=attn_probs.device)
        beacon_query_positions = torch.nonzero(cur_beacon_indices == 1, as_tuple=False).flatten()
        if beacon_query_positions.numel() == 0:
            return

        prefix_len = kv_seq_len - len(beacon_indices)
        if prefix_len < 0:
            return
        if prefix_len > 0:
            sink_size = min(getattr(self.config, "beacon_sink_size", 0), prefix_len)
            beacon_history_size = prefix_len - sink_size
            prefix_indices = torch.cat(
                [
                    beacon_indices.new_zeros(sink_size),
                    beacon_indices.new_ones(beacon_history_size),
                ]
            )
            full_beacon_indices = torch.cat([prefix_indices, beacon_indices]).to(device=attn_probs.device)
        else:
            full_beacon_indices = beacon_indices.to(device=attn_probs.device)

        if full_beacon_indices.shape[0] != kv_seq_len:
            return

        raw_positions = torch.nonzero(full_beacon_indices == 0, as_tuple=False).flatten()
        if raw_positions.numel() == 0:
            return

        bsz = attn_probs.shape[0]
        current_start = kv_seq_len - q_len
        for beacon_query_pos in beacon_query_positions.tolist():
            query_global_pos = current_start + beacon_query_pos
            prev_raw = raw_positions[raw_positions < query_global_pos]
            if prev_raw.numel() == 0:
                continue

            selected_raw = prev_raw[-self.beacon_probe_prev_raw_tokens :]
            probs = attn_probs[:, :, beacon_query_pos, selected_raw]
            probs_mean = probs.mean(dim=1)
            probs_sum_mean = probs_mean.sum(dim=-1)
            selected_raw_list = selected_raw.detach().cpu().tolist()

            for batch_idx in range(bsz):
                self._beacon_probe_records.append(
                    {
                        "layer_idx": int(self.layer_idx if self.layer_idx is not None else -1),
                        "batch_idx": int(batch_idx),
                        "beacon_query_local_idx": int(beacon_query_pos),
                        "beacon_query_global_idx": int(query_global_pos),
                        "prev_raw_positions": [int(x) for x in selected_raw_list],
                        "prev_raw_probs_mean": probs_mean[batch_idx].detach().to(torch.float32).cpu().tolist(),
                        "prev_raw_probs_sum_mean": float(probs_sum_mean[batch_idx].item()),
                    }
                )

    def _collect_beacon_chunk2_heatmap_probe(
        self,
        attn_probs: torch.Tensor,
        q_len: int,
        past_key: Optional[torch.Tensor],
        beacon_indices: Optional[torch.LongTensor],
    ) -> None:
        if not self.beacon_probe_chunk2_heatmap:
            return
        # Probe-only safeguard: never collect during training.
        if self.training:
            return
        if self._beacon_chunk2_heatmap_record is not None:
            return

        runtime_chunk_idx = int(getattr(self.config, "_beacon_runtime_chunk_idx", -1))
        raw_probe_mode = runtime_chunk_idx < 0 and beacon_indices is None
        if raw_probe_mode:
            # Raw-model probe: capture prefill attention heatmap once.
            if past_key is not None or q_len <= 1:
                return
        else:
            # Beacon probe: capture the configured chunk only.
            if past_key is None or q_len <= 1:
                return
            if runtime_chunk_idx != self.beacon_probe_chunk_idx:
                return
        if attn_probs.shape[0] <= 0:
            return

        # [heads, q_len, kv_seq_len] -> [q_len, kv_seq_len]
        heatmap = attn_probs[0].mean(dim=0).detach().to(torch.float32).cpu()
        beacon_query_positions = []
        raw_query_positions = []
        beacon_key_positions = []
        kv_seq_len = int(attn_probs.shape[-1])
        if beacon_indices is not None and beacon_indices.dim() == 1 and beacon_indices.shape[0] >= q_len:
            cur_beacon_indices = beacon_indices[-q_len:].to(device=attn_probs.device)
            beacon_query_mask = torch.logical_or(cur_beacon_indices == 1, cur_beacon_indices == 2)
            raw_query_mask = torch.logical_not(beacon_query_mask)
            beacon_query_positions = (
                torch.nonzero(beacon_query_mask, as_tuple=False).flatten().detach().cpu().tolist()
            )
            raw_query_positions = (
                torch.nonzero(raw_query_mask, as_tuple=False).flatten().detach().cpu().tolist()
            )

            if len(beacon_indices) <= kv_seq_len:
                prefix_len = kv_seq_len - len(beacon_indices)
                if prefix_len > 0:
                    sink_size = min(int(getattr(self.config, "beacon_sink_size", 0)), prefix_len)
                    beacon_history_size = prefix_len - sink_size
                    prefix_indices = torch.cat(
                        [
                            beacon_indices.new_zeros(sink_size),
                            beacon_indices.new_ones(beacon_history_size),
                        ]
                    )
                    full_beacon_indices = torch.cat([prefix_indices, beacon_indices]).to(device=attn_probs.device)
                else:
                    full_beacon_indices = beacon_indices.to(device=attn_probs.device)
                if full_beacon_indices.shape[0] == kv_seq_len:
                    beacon_key_mask = torch.logical_or(full_beacon_indices == 1, full_beacon_indices == 2)
                    beacon_key_positions = (
                        torch.nonzero(beacon_key_mask, as_tuple=False).flatten().detach().cpu().tolist()
                    )
        else:
            # Raw-model probe: all queries are raw tokens; no beacon keys.
            raw_query_positions = list(range(int(heatmap.shape[0])))
        self._beacon_chunk2_heatmap_record = {
            "layer_idx": int(self.layer_idx if self.layer_idx is not None else -1),
            "chunk_idx": runtime_chunk_idx,
            "q_len": int(heatmap.shape[0]),
            "kv_seq_len": int(heatmap.shape[1]),
            "beacon_query_positions": [int(x) for x in beacon_query_positions],
            "raw_query_positions": [int(x) for x in raw_query_positions],
            "beacon_key_positions": [int(x) for x in beacon_key_positions],
            "heatmap": heatmap.tolist(),
        }

    def _apply_rotary_with_beacon(
        self,
        query_states,
        key_states,
        position_ids,
        beacon_size,
        beacon_indices,
        current_key_states=None,
    ):
        """Apply RoPE to Q and K with beacon-K custom handling.

        Two optional beacon-K behaviors are supported:
        1) beacon_skip_rope_for_beacon_k: keep beacon K un-rotated.
        2) beacon_chunk_fixed_rope_for_beacon_k:
           - beacon K in each chunk uses one fixed anchor position (chunk-first token)
           - the same fixed-RoPE beacon K is used for attention and KV cache
        """
        skip_beacon_k_rope = getattr(self.config, "beacon_skip_rope_for_beacon_k", False)
        chunk_fixed_beacon_k_rope = self.beacon_chunk_fixed_rope_for_beacon_k
        raw_nokrope = bool(getattr(self.config, "raw_nokrope", False))

        # No Beacon metadata means native raw mode. With metadata, historical
        # Beacon keys still need custom handling even when no new Beacon is added.
        if beacon_indices is None:
            if raw_nokrope:
                query_states_rope, _ = self.rotary_emb(query_states, key_states, position_ids)
                return query_states_rope, key_states
            return self.rotary_emb(query_states, key_states, position_ids)
        if not skip_beacon_k_rope and not chunk_fixed_beacon_k_rope:
            return self.rotary_emb(query_states, key_states, position_ids)

        # key_states shape: [bsz, num_kv_heads, kv_seq_len, head_dim]
        # beacon_indices covers [raw_cache, current_input] but key_states is
        # [sink, beacon_history, raw_cache, current_input].
        # We need to build a full-length mask aligned with key_states.
        kv_seq_len = key_states.shape[2]
        prefix_len = kv_seq_len - len(beacon_indices)
        if prefix_len > 0:
            # prefix = [sink (0), beacon_history (1)]
            sink_size = getattr(self.config, "beacon_sink_size", 0)
            beacon_history_size = prefix_len - sink_size
            prefix_indices = torch.cat([
                beacon_indices.new_zeros(sink_size),
                beacon_indices.new_ones(beacon_history_size),
            ])
            full_beacon_indices = torch.cat([prefix_indices, beacon_indices]).to(device=key_states.device)
        else:
            full_beacon_indices = beacon_indices.to(device=key_states.device)

        q_len = query_states.shape[2]

        # Apply RoPE once. If enabled, reuse this result to compute residuals
        # and custom-update current-step beacon K in full-rank space.
        query_states, key_states_rope = self.rotary_emb(query_states, key_states, position_ids)

        if self._should_enhance_beacon_k_pos_residual(beacon_size, beacon_indices):
            if q_len > 0:
                key_states = self._enhance_beacon_key_with_raw_pos_residual_from_rope(
                    key_states=key_states,
                    key_states_rope=key_states_rope,
                    beacon_indices=beacon_indices,
                    q_len=q_len,
                    current_key_states=current_key_states,
                )

        if skip_beacon_k_rope and not chunk_fixed_beacon_k_rope:
            # Keep beacon K un-rotated for both current and cached parts.
            beacon_mask = torch.logical_or(full_beacon_indices == 1, full_beacon_indices == 2)
            beacon_k_backup = key_states[:, :, beacon_mask].clone()
            key_states_rope[:, :, beacon_mask] = beacon_k_backup
            return query_states, key_states_rope

        if chunk_fixed_beacon_k_rope:
            # Restore historical beacon K from cache first. Cache stores chunk-fixed
            # beacon K (already rotated with each chunk's anchor).
            current_start = key_states.shape[2] - q_len
            if current_start > 0:
                history_beacon_mask = torch.logical_or(
                    full_beacon_indices[:current_start] == 1,
                    full_beacon_indices[:current_start] == 2,
                )
                if history_beacon_mask.any():
                    history_pos = torch.nonzero(history_beacon_mask, as_tuple=False).flatten()
                    key_states_rope[:, :, history_pos] = key_states[:, :, history_pos]

            # Apply chunk-fixed RoPE to current-step beacon K on both attention path
            # and cache path so they stay identical.
            if (
                current_key_states is not None
                and self._should_apply_chunk_fixed_beacon_k_rope(beacon_size, beacon_indices)
                and q_len > 0
            ):
                self._apply_chunk_fixed_rope_to_current_beacon_k(
                    key_states=key_states_rope,
                    position_ids=position_ids,
                    beacon_indices=beacon_indices,
                    q_len=q_len,
                    current_key_states=current_key_states,
                )

        return query_states, key_states_rope
    
    # def qkv_proj_with_beacon(self, hidden_states, beacon_size, beacon_indices):
    #     if beacon_size > 0:
    #         # NOTE: when beacon_pos == "interleave", the beacon_indices points to all beacon tokens in the current window (cached activations + input_ids), so we shall slice out the part corresponding to the input_ids
    #         cur_beacon_indices = beacon_indices[-hidden_states.shape[1]:]

    #         ordinal_hidden_states = hidden_states[:, cur_beacon_indices == 0]
    #         beacon_hidden_states = hidden_states[:, cur_beacon_indices == 1]

    #         if "q" in self.config.beacon_param:
    #             ordinal_query_states = self.q_proj(ordinal_hidden_states)
    #             beacon_query_states = self.beacon_q_proj(beacon_hidden_states)
    #             query_states = beacon_query_states.new_zeros((ordinal_query_states.shape[0], cur_beacon_indices.shape[0], ordinal_query_states.shape[2]))
    #             query_states[:, cur_beacon_indices == 0] = ordinal_query_states
    #             query_states[:, cur_beacon_indices == 1] = beacon_query_states
    #             # NOTE: replicate hidden states for beacon tokens in case of parallel windows
    #             if (cur_beacon_indices == 2).any():
    #                 query_states[:, cur_beacon_indices == 2] = beacon_query_states[:, :(cur_beacon_indices == 2).sum()]

    #         else:
    #             query_states = self.q_proj(hidden_states)

    #         if "k" in self.config.beacon_param:
    #             ordinal_key_states = self.k_proj(ordinal_hidden_states)
    #             beacon_key_states = self.beacon_k_proj(beacon_hidden_states)
    #             key_states = beacon_key_states.new_zeros((ordinal_key_states.shape[0], cur_beacon_indices.shape[0], ordinal_key_states.shape[2]))
    #             key_states[:, cur_beacon_indices == 0] = ordinal_key_states
    #             key_states[:, cur_beacon_indices == 1] = beacon_key_states
    #             # NOTE: replicate hidden states for beacon tokens in case of parallel windows
    #             if (cur_beacon_indices == 2).any():
    #                 key_states[:, cur_beacon_indices == 2] = beacon_key_states[:, :(cur_beacon_indices == 2).sum()]

    #         else:
    #             key_states = self.k_proj(hidden_states)
            
    #         if "v" in self.config.beacon_param:
    #             ordinal_value_states = self.v_proj(ordinal_hidden_states)
    #             beacon_value_states = self.beacon_v_proj(beacon_hidden_states)
    #             value_states = beacon_value_states.new_zeros((ordinal_value_states.shape[0], cur_beacon_indices.shape[0], ordinal_value_states.shape[2]))
    #             value_states[:, cur_beacon_indices == 0] = ordinal_value_states
    #             value_states[:, cur_beacon_indices == 1] = beacon_value_states
    #             # NOTE: replicate hidden states for beacon tokens in case of parallel windows
    #             if (cur_beacon_indices == 2).any():
    #                 value_states[:, cur_beacon_indices == 2] = beacon_value_states[:, :(cur_beacon_indices == 2).sum()]
    #         else:
    #             value_states = self.v_proj(hidden_states)

    #     else:
    #         query_states = self.q_proj(hidden_states)
    #         key_states = self.k_proj(hidden_states)
    #         value_states = self.v_proj(hidden_states)

    #     return query_states, key_states, value_states

    # def o_proj_with_beacon(self, attn_output, beacon_size, beacon_indices):
    #     if beacon_size > 0:
    #         # NOTE: when beacon_pos == "interleave", the beacon_indices points to all beacon tokens in the current window (cached activations + input_ids), so we shall slice out the part corresponding to the input_ids
    #         cur_beacon_indices = beacon_indices[-attn_output.shape[1]:]

    #         if "o" in self.config.beacon_param:
    #             ordinal_attn_output = self.o_proj(attn_output[:, cur_beacon_indices == 0])
    #             beacon_attn_output = self.beacon_o_proj(attn_output[:, cur_beacon_indices == 1])
    #             attn_output = beacon_attn_output.new_zeros(attn_output.shape)
    #             attn_output[:, cur_beacon_indices == 0] = ordinal_attn_output
    #             attn_output[:, cur_beacon_indices == 1] = beacon_attn_output
    #             # NOTE: replicate hidden states for beacon tokens in case of parallel windows
    #             # if (cur_beacon_indices == 2).any():
    #             #     attn_output[:, cur_beacon_indices == 2] = beacon_attn_output[:, :(cur_beacon_indices == 2).sum()]
    #         else:
    #             attn_output = self.o_proj(attn_output)
    #     else:
    #         attn_output = self.o_proj(attn_output)
    #     return attn_output

    def qkv_proj_with_beacon(self, hidden_states, beacon_size, beacon_indices, return_latents=False):
        key_latent, value_latent = None, None
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
                if return_latents:
                    beacon_key_states, key_latent = self._project_beacon_key(hidden_states, return_latent=True)
                else:
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
                if return_latents:
                    beacon_value_states, value_latent = self._project_beacon_value(hidden_states, return_latent=True)
                else:
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

        if return_latents:
            return query_states, key_states, value_states, key_latent, value_latent
        return query_states, key_states, value_states

    def _incremental_cache(self, key, value, beacon_size, indices, latents):
        if latents is None:
            return key, value, beacon_size, indices
        roles = indices[-key.shape[2]:] != 0

        def pack(latent):
            if latent is None:
                return None
            selected = latent[:, roles]
            batch, length, rank = selected.shape
            if uses_latent_decode(self.config):
                return selected.unsqueeze(1).contiguous()
            return selected.reshape(batch, length, self.num_key_value_heads,
                                    rank // self.num_key_value_heads).transpose(1, 2).contiguous()

        return ReconstructionCacheUpdate(key, value, beacon_size, indices,
                                         pack(latents[0]), pack(latents[1]))

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
        kv_seq_len = hidden_states.shape[-2]
        past_key, past_value, beacon_size, beacon_indices = past_key_value
        latent_cache = isinstance(past_key, ReconstructionCache)
        if fa2_eligible(self, hidden_states, past_key_value, output_attentions):
            return fa2_decode(self, hidden_states, position_ids, past_key_value)
        if latent_cache:
            past_key, past_value = past_key.materialize(self), past_value.materialize(self)

        if past_key is not None:
            past_seq_len = past_key.shape[2]
            kv_seq_len += past_seq_len
        else:
            past_seq_len = 0

        projected = self.qkv_proj_with_beacon(
            hidden_states, beacon_size, beacon_indices, return_latents=latent_cache
        )
        query_states, key_states, value_states = projected[:3]
        cache_latents = projected[3:] if latent_cache else None

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        current_key_states = key_states

        # return keys and values before rope
        # NOTE: incrementally return keys and values for beacon efficiency
        past_key_value = self._incremental_cache(
            current_key_states, value_states, beacon_size, beacon_indices, cache_latents
        )

        if past_key is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key, current_key_states], dim=2)
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
            current_key_states=current_key_states,
        )

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        self._collect_beacon_prev_raw_attn_probe(
            attn_probs=attn_weights,
            beacon_indices=beacon_indices,
            q_len=q_len,
            kv_seq_len=kv_seq_len,
        )
        self._collect_beacon_chunk2_heatmap_probe(
            attn_probs=attn_weights,
            q_len=q_len,
            past_key=past_key,
            beacon_indices=beacon_indices,
        )
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj_with_beacon(attn_output, beacon_size, beacon_indices)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


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
        latent_cache = isinstance(past_key, ReconstructionCache)
        if fa2_eligible(self, hidden_states, past_key_value, output_attentions):
            return fa2_decode(self, hidden_states, position_ids, past_key_value)
        if latent_cache:
            past_key, past_value = past_key.materialize(self), past_value.materialize(self)
        if past_key is not None:
            past_seq_len = past_key.shape[2]
            kv_seq_len += past_seq_len
        else:
            past_seq_len = 0

        projected = self.qkv_proj_with_beacon(
            hidden_states, beacon_size, beacon_indices, return_latents=latent_cache
        )
        query_states, key_states, value_states = projected[:3]
        cache_latents = projected[3:] if latent_cache else None

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        current_key_states = key_states

        # return keys and values before rope
        # NOTE: incrementally return keys and values for beacon efficiency
        past_key_value = self._incremental_cache(
            current_key_states, value_states, beacon_size, beacon_indices, cache_latents
        )

        if past_key is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key, current_key_states], dim=2)
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
            current_key_states=current_key_states,
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
        output_attentions = False

        bsz, q_len, _ = hidden_states.size()
        kv_seq_len = hidden_states.shape[-2]

        past_key, past_value, beacon_size, beacon_indices = past_key_value
        latent_cache = isinstance(past_key, ReconstructionCache)
        if fa2_eligible(self, hidden_states, past_key_value, output_attentions):
            return fa2_decode(self, hidden_states, position_ids, past_key_value)
        if latent_cache:
            past_key, past_value = past_key.materialize(self), past_value.materialize(self)
        if past_key is not None:
            past_seq_len = past_key.shape[2]
            kv_seq_len += past_seq_len
        else:
            past_seq_len = 0

        projected = self.qkv_proj_with_beacon(
            hidden_states, beacon_size, beacon_indices, return_latents=latent_cache
        )
        query_states, key_states, value_states = projected[:3]
        cache_latents = projected[3:] if latent_cache else None

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        current_key_states = key_states

        # return keys and values before rope
        # NOTE: incrementally return keys and values for beacon efficiency
        past_key_value = self._incremental_cache(
            current_key_states, value_states, beacon_size, beacon_indices, cache_latents
        )

        if past_key is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key, current_key_states], dim=2)
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
            current_key_states=current_key_states,
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
        self._decode_graphs = None

    def _apply(self, fn, recurse=True):
        self._decode_graphs = None
        return super()._apply(fn, recurse=recurse)

    def _decode_tail(self, attention_output, residual):
        hidden = residual + attention_output
        return hidden + self.mlp(self.post_attention_layernorm(hidden))

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
        graph_step = (not self.training and hidden_states.is_cuda
                      and hidden_states.shape[0] > 0 and hidden_states.shape[1] in (1, 2)
                      and not output_attentions
                      and getattr(self.self_attn.config, '_slimkv_decode_graph_step', False)
                      and getattr(self.self_attn.config, 'beacon_decode_graph', False))
        if graph_step:
            if self._decode_graphs is None:
                from .decode_graph import DecodeGraphs
                self._decode_graphs = DecodeGraphs()
            hidden_states = self._decode_graphs('input_norm', self.input_layernorm, hidden_states)
        else:
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
        if graph_step:
            hidden_states = self._decode_graphs('tail', self._decode_tail, hidden_states, residual)
        else:
            hidden_states = residual + hidden_states
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
    _supports_sdpa = False
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

    def pop_beacon_probe_records(self):
        records = []
        for layer in self.layers:
            self_attn = layer.self_attn
            if hasattr(self_attn, "pop_beacon_probe_records"):
                layer_records = self_attn.pop_beacon_probe_records()
                if layer_records:
                    records.extend(layer_records)
        return records

    def clear_beacon_probe_records(self):
        for layer in self.layers:
            self_attn = layer.self_attn
            if hasattr(self_attn, "clear_beacon_probe_records"):
                self_attn.clear_beacon_probe_records()

    def pop_beacon_chunk2_heatmap_records(self):
        records = []
        for layer in self.layers:
            self_attn = layer.self_attn
            if hasattr(self_attn, "pop_beacon_chunk2_heatmap_record"):
                layer_record = self_attn.pop_beacon_chunk2_heatmap_record()
                if layer_record is not None:
                    records.append(layer_record)
        records.sort(key=lambda x: x.get("layer_idx", -1))
        return records

    def clear_beacon_chunk2_heatmap_records(self):
        for layer in self.layers:
            self_attn = layer.self_attn
            if hasattr(self_attn, "clear_beacon_chunk2_heatmap_record"):
                self_attn.clear_beacon_chunk2_heatmap_record()

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

        past_key, past_value, beacon_size, beacon_indices = past_key_values[0]

        # Eager attention in this codepath consumes an additive 4D mask
        # of shape [bsz, 1, q_len, kv_len]. Generation can provide a 2D
        # padding mask [bsz, kv_len], so convert it here for compatibility.
        if (
            attention_mask is not None
            and attention_mask.dim() == 2
            and getattr(self.config, "_attn_implementation", None) == "eager"
        ):
            bsz = attention_mask.shape[0]
            q_len = seq_length
            kv_len = attention_mask.shape[-1]
            min_dtype = torch.finfo(torch.float32).min
            device = attention_mask.device

            # Padding mask: keys with mask==0 should be blocked for all queries.
            additive_mask = torch.zeros((bsz, 1, q_len, kv_len), dtype=torch.float32, device=device)
            key_padding = attention_mask[:, None, None, :].eq(0)
            additive_mask = additive_mask.masked_fill(key_padding, min_dtype)

            # Causal mask for current query block against full key length.
            past_tokens = max(0, kv_len - q_len)
            causal = torch.full((q_len, kv_len), min_dtype, dtype=torch.float32, device=device)
            causal = torch.triu(causal, diagonal=1 + past_tokens)
            attention_mask = additive_mask + causal[None, None, :, :]

        # BEACON: separately embed ordinal tokens and beacon tokens because ordinal tokens do not receive gradients
        if (beacon_size == 1 and input_ids.ndim == 2 and input_ids.shape[1] == 2
                and getattr(self.config, '_slimkv_decode_step', False)
                and uses_fa2_decode(self.config)
                and getattr(self.config, 'beacon_decode_fast', True)):
            inputs_embeds = torch.cat([self.embed_tokens(input_ids[:, :1]),
                                       self.beacon_embed_tokens(input_ids[:, 1:] - self.config.vocab_size)], dim=1)
        elif beacon_size > 0:
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
        # BEACON: still use tuple to organize cache
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

    def pop_beacon_probe_records(self):
        if hasattr(self.model, "pop_beacon_probe_records"):
            return self.model.pop_beacon_probe_records()
        return []

    def clear_beacon_probe_records(self):
        if hasattr(self.model, "clear_beacon_probe_records"):
            self.model.clear_beacon_probe_records()

    def pop_beacon_chunk2_heatmap_records(self):
        if hasattr(self.model, "pop_beacon_chunk2_heatmap_records"):
            return self.model.pop_beacon_chunk2_heatmap_records()
        return []

    def clear_beacon_chunk2_heatmap_records(self):
        if hasattr(self.model, "clear_beacon_chunk2_heatmap_records"):
            self.model.clear_beacon_chunk2_heatmap_records()

    @staticmethod
    def _validate_lowrank_beacon_kv_loading(config, loading_info):
        beacon_param = getattr(config, "beacon_param", [])
        missing_keys = loading_info.get("missing_keys", [])
        unexpected_keys = loading_info.get("unexpected_keys", [])
        lowrank_enabled = (
            getattr(config, "beacon_k_compression", "none") == "lowrank_svd"
            or getattr(config, "beacon_v_compression", "none") == "lowrank_svd"
        )

        # validate K
        shared_kv = getattr(config, "beacon_shared_kv_down_proj", False)
        if "k" in beacon_param and getattr(config, "beacon_k_compression", "none") == "lowrank_svd":
            if shared_kv:
                # shared mode: K uses beacon_kv_down_proj (validated in V block) + beacon_k_up_proj
                missing_lowrank_keys = [
                    k for k in missing_keys if "beacon_k_up_proj" in k
                ]
            else:
                missing_lowrank_keys = [
                    k for k in missing_keys
                    if "beacon_k_down_proj" in k or "beacon_k_up_proj" in k
                ]
            if missing_lowrank_keys:
                unexpected_dense_keys = [k for k in unexpected_keys if "beacon_k_proj" in k]
                message = (
                    "Low-rank beacon K mode requires a converted checkpoint. "
                    f"Missing low-rank weights: {missing_lowrank_keys[:8]}"
                )
                if unexpected_dense_keys:
                    message += (
                        f". Dense beacon_k_proj weights were found instead: {unexpected_dense_keys[:8]}. "
                        "Run the offline conversion script before training or finetuning."
                    )
                else:
                    message += ". Run the offline conversion script before training or finetuning."
                raise ValueError(message)

        # validate V
        if "v" in beacon_param and getattr(config, "beacon_v_compression", "none") == "lowrank_svd":
            if shared_kv:
                # shared mode: V uses beacon_kv_down_proj + beacon_v_up_proj
                missing_lowrank_keys = [
                    k for k in missing_keys
                    if "beacon_kv_down_proj" in k or "beacon_v_up_proj" in k
                ]
            else:
                missing_lowrank_keys = [
                    k for k in missing_keys
                    if "beacon_v_down_proj" in k or "beacon_v_up_proj" in k
                ]
            if missing_lowrank_keys:
                unexpected_dense_keys = [k for k in unexpected_keys if "beacon_v_proj" in k]
                message = (
                    "Low-rank beacon V mode requires a converted checkpoint. "
                    f"Missing low-rank weights: {missing_lowrank_keys[:8]}"
                )
                if unexpected_dense_keys:
                    message += (
                        f". Dense beacon_v_proj weights were found instead: {unexpected_dense_keys[:8]}. "
                        "Run the offline conversion script before training or finetuning."
                    )
                else:
                    message += ". Run the offline conversion script before training or finetuning."
                raise ValueError(message)

        if lowrank_enabled:
            missing_beacon_keys = [k for k in missing_keys if "beacon_" in k]
            if missing_beacon_keys:
                raise ValueError(
                    "Low-rank beacon mode does not allow auto-initializing missing beacon weights. "
                    f"Missing keys: {missing_beacon_keys[:8]}. "
                    "Please provide a fully converted checkpoint."
                )

    @staticmethod
    def _auto_merge_kv_down_proj(model, loading_info):
        """When beacon_shared_kv_down_proj is enabled but the checkpoint has
        separate beacon_k_down_proj / beacon_v_down_proj, automatically merge
        them into beacon_kv_down_proj by averaging."""
        missing_keys = loading_info.get("missing_keys", [])
        unexpected_keys = loading_info.get("unexpected_keys", [])

        has_separate_k = any("beacon_k_down_proj" in k for k in unexpected_keys)
        has_separate_v = any("beacon_v_down_proj" in k for k in unexpected_keys)
        needs_shared = any("beacon_kv_down_proj" in k for k in missing_keys)

        if not (has_separate_k and has_separate_v and needs_shared):
            return

        logger.info("Auto-merging beacon_k_down_proj + beacon_v_down_proj -> beacon_kv_down_proj (averaged)")

        from safetensors.torch import load_file as _load_file
        import os
        import glob

        model_path = model.config._name_or_path
        if not os.path.isdir(model_path):
            return

        k_weights = {}
        v_weights = {}
        st_files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
        for st_file in st_files:
            sd = _load_file(st_file)
            for key, tensor in sd.items():
                if "beacon_k_down_proj" in key:
                    k_weights[key] = tensor
                elif "beacon_v_down_proj" in key:
                    v_weights[key] = tensor

        if not k_weights or not v_weights:
            logger.warning("Could not find separate K/V down_proj weights in checkpoint; skipping auto-merge.")
            return

        merged_count = 0
        for v_key, v_tensor in v_weights.items():
            k_key = v_key.replace("beacon_v_down_proj", "beacon_k_down_proj")
            shared_key = v_key.replace("beacon_v_down_proj", "beacon_kv_down_proj")
            if k_key in k_weights:
                avg_weight = (k_weights[k_key].to(v_tensor.dtype) + v_tensor) / 2
                parts = shared_key.split(".")
                obj = model
                for part in parts[:-1]:
                    if part.isdigit():
                        obj = obj[int(part)]
                    else:
                        obj = getattr(obj, part)
                param_name = parts[-1]
                getattr(obj, param_name).data.copy_(avg_weight)
                merged_count += 1

        loading_info["missing_keys"] = [
            k for k in loading_info["missing_keys"] if "beacon_kv_down_proj" not in k
        ]
        loading_info["unexpected_keys"] = [
            k for k in loading_info["unexpected_keys"]
            if "beacon_k_down_proj" not in k and "beacon_v_down_proj" not in k
        ]
        logger.info(f"Auto-merged {merged_count} beacon_kv_down_proj weights from separate K/V down_proj.")

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        """Override the default from_pretrained to extend vocab size according to beacon_size."""
        kwargs.update(output_loading_info=True)
        model, loading_info = super().from_pretrained(*args, **kwargs)

        # NOTE: set memory after from_pretrained because there may be another transformer model inside the Memory object, which may cause weird erros during loading
        config = model.config
        model.memory = LlamaMemory(
            model_config=config,
            k_seq_dim=2,
            v_seq_dim=2,
        )
        model.memory.train(model.training)

        # Auto-merge separate K/V down_proj into shared beacon_kv_down_proj
        # when beacon_shared_kv_down_proj is enabled but checkpoint has separate weights.
        if getattr(config, "beacon_shared_kv_down_proj", False):
            cls._auto_merge_kv_down_proj(model, loading_info)

        cls._validate_lowrank_beacon_kv_loading(config, loading_info)
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
        num_logits_to_keep: int = 0,
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
            past_key_values = [(None, None, 0, None) for _ in range(self.config.num_hidden_layers)]

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
        if num_logits_to_keep > 0 and labels is None and not self.training:
            hidden_states = hidden_states[:, -num_logits_to_keep:]
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

    def _beacon_forward(self, input_ids=None, attention_mask=None, **kwargs):
        decode_step = (not self.training and kwargs.get('labels') is None
                       and input_ids is not None and input_ids.ndim == 2 and input_ids.shape[1] == 1
                       and self.memory.all_input_ids is not None)
        if decode_step and uses_latent_decode(self.config):
            old_mask = self.memory.all_attention_mask
            if self.memory._decode_unpadded is None:
                self.memory._decode_unpadded = not bool((old_mask == 0).any())
            if not self.memory._decode_unpadded or (attention_mask is not None and (attention_mask == 0).any()):
                raise ValueError('Latent decode requires an unpadded batch with aligned sequence lengths.')
        self.config._slimkv_decode_step = decode_step
        requested_attentions = kwargs.get('output_attentions')
        if requested_attentions is None:
            requested_attentions = self.config.output_attentions
        requested_hidden = kwargs.get('output_hidden_states')
        if requested_hidden is None:
            requested_hidden = self.config.output_hidden_states
        self.config._slimkv_decode_fast_step = decode_step and not requested_attentions
        self.config._slimkv_decode_graph_step = (
            decode_step and uses_latent_decode(self.config)
            and not requested_hidden and not requested_attentions)
        try:
            return self._beacon_forward_impl(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        finally:
            self.config._slimkv_decode_step = False
            self.config._slimkv_decode_fast_step = False
            self.config._slimkv_decode_graph_step = False

    def _beacon_forward_impl(self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        # t1 = time.time()
        self.config._beacon_runtime_chunk_idx = -1
        keep_last_logits = num_logits_to_keep > 0 and labels is None and not self.training
        # Interleaved Beacons can occupy the last position. Retain enough raw
        # positions before Memory removes Beacon logits, even at token ratio 1.
        chunk_logits = 2 * num_logits_to_keep if keep_last_logits and self.config.beacon_pos == 'interleave' else 0

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
            self.config._beacon_runtime_chunk_idx = int(self.memory._step_idx)
            loss_weights = build_beacon_loss_weights(
                input_ids=input_ids,
                labels=labels,
                beacon_token_id=self.config.vocab_size,
                loss_mode=getattr(self.config, "beacon_loss_mode", None),
                loss_weight=getattr(self.config, "beacon_loss_weight", None),
            )

            # t4 = time.time()

            static = None
            if self.config._slimkv_decode_graph_step and self.memory._fast_decode_active:
                from .static_decode import StaticDecode
                if StaticDecode.eligible(self, input_ids, past_key_values, {
                        'labels': labels, 'inputs_embeds': inputs_embeds,
                        'output_attentions': output_attentions, 'output_hidden_states': output_hidden_states,
                        'return_dict': return_dict}):
                    first = past_key_values[0][0]
                    latent_length = sum(0 if x is None else x.shape[2] for x in (first.history, first.window))
                    if latent_length > 0:
                        required = latent_length + past_key_values[0][2]
                        capacity = max(64, (required + 63) // 64 * 64)
                        static = getattr(self, '_static_decode', None)
                        if (static is None or static.capacity != capacity
                                or static.layers[0]['raw_k'].device != input_ids.device):
                            static = self._static_decode = StaticDecode(self, capacity)
            if static is not None:
                logits = static.run(input_ids, position_ids, past_key_values)
                outputs = ModelOutput(logits=logits, past_key_values=())
            else:
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
                    num_logits_to_keep=chunk_logits,
                )

            # t5 = time.time()

            # update past_key_values
            if static is None:
                self.memory.update_memory(outputs.past_key_values)
                if hasattr(self, '_static_decode'):
                    # Beacon insertion and diagnostic/multi-token calls use the
                    # ordinary path. Refresh static buffers before the next Q1.
                    self._static_decode.epoch = None

            # t6 = time.time()

            if labels is not None:
                # update loss
                self.memory.update_loss(outputs.batch_loss, outputs.loss_denominator)

            # t7 = time.time()

            # print(f"step time: {t4-t3}, forward time: {t5-t4}, update time: {t6-t5}, loss time: {t7-t6}")
            # input()

        # t8 = time.time()

        # output loss, past_key_values, and perplexity
        outputs = self.memory.output(outputs)
        if keep_last_logits:
            outputs['logits'] = outputs['logits'][:, -num_logits_to_keep:]
        self.config._beacon_runtime_chunk_idx = -1

        # t9 = time.time()

        # print(f"output time:            {t9-t8}")
        # input()

        return outputs

    def forward(self, **kwargs):
        """Forward computation over a batch of sequences.
        """
        # only allow gradient when training
        with optional_grad_ctx(with_grad=self.training):
            # we can disable beacon to use the original llama
            if hasattr(self, "_enable_beacon") and self._enable_beacon == False:
                return self._native_forward(**kwargs)
            else:
                return self._beacon_forward(**kwargs)

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        num_logits_to_keep=None,
        **kwargs,
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            # NOTE:
            # - no_beacon/native path in this repository keeps unrotated KV and applies RoPE on concatenated K,
            #   so it requires full-length position_ids during generation.
            # - beacon path follows the existing incremental behavior and only needs the last position id.
            if past_key_values and not (hasattr(self, "_enable_beacon") and self._enable_beacon is False):
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

        if num_logits_to_keep is not None:
            model_inputs['num_logits_to_keep'] = num_logits_to_keep
        for extra_key in ("position_embeddings",):
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
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past
