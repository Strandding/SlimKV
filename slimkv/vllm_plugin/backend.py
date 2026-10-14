"""vLLM allocation and metadata for compact Beacon and local raw KV pages."""
from dataclasses import dataclass

import torch
from torch import nn

from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backend import AttentionBackend, AttentionMetadataBuilder
from vllm.v1.kv_cache_interface import FullAttentionSpec, ChunkedLocalAttentionSpec


@dataclass(frozen=True, kw_only=True)
class BeaconSpec(FullAttentionSpec):
    token_ratio: int
    raw_width: int

    @property
    def storage_block_size(self):
        return self.block_size // self.token_ratio + (2 * self.raw_width + self.head_size - 1) // self.head_size

    @property
    def real_page_size_bytes(self):
        return self.storage_block_size * self.head_size * 2

    @classmethod
    def merge(cls, specs):
        assert all(spec == specs[0] for spec in specs)
        return specs[0]


class PageMetadataBuilder(AttentionMetadataBuilder):
    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        if common_prefix_len:
            raise ValueError('Shared prefixes are not enabled for SlimKV pages.')
        return common_attn_metadata


class PageStoreImpl:
    """Storage is consumed by SlimKV's existing FA2 attention implementation."""


class RawPageBackend(AttentionBackend):
    supported_dtypes = [torch.bfloat16]
    supported_kv_cache_dtypes = ['auto', 'bfloat16']

    @staticmethod
    def get_name():
        return 'SLIMKV_RAW'

    @staticmethod
    def get_impl_cls():
        return PageStoreImpl

    @staticmethod
    def get_builder_cls():
        return PageMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str='auto'):
        return num_blocks, block_size, 2, num_kv_heads, head_size

    @staticmethod
    def get_kv_cache_stride_order(include_num_layers_dimension=False):
        return (1, 0, 2, 3, 4, 5) if include_num_layers_dimension else (0, 1, 2, 3, 4)

    @classmethod
    def indexes_kv_by_block_stride(cls):
        return True


class BeaconPageBackend(RawPageBackend):
    @staticmethod
    def get_name():
        return 'SLIMKV_BEACON'

    @staticmethod
    def get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str='auto'):
        return num_blocks, block_size, head_size

    @staticmethod
    def get_kv_cache_stride_order(include_num_layers_dimension=False):
        return (1, 0, 2, 3) if include_num_layers_dimension else (0, 1, 2)


class PageLayer(nn.Module, AttentionLayerBase):
    def __init__(self, name, spec, vllm_config, beacon=False):
        super().__init__()
        self.name = name
        self.spec = spec
        self.beacon = beacon
        self.kv_cache = torch.empty(0)
        self.impl = PageStoreImpl()
        context = vllm_config.compilation_config.static_forward_context
        if name in context:
            raise ValueError(f'Duplicate SlimKV cache layer: {name}')
        context[name] = self

    def get_attn_backend(self):
        return BeaconPageBackend if self.beacon else RawPageBackend

    def get_kv_cache_spec(self, vllm_config):
        return self.spec


def create_page_layers(model, vllm_config, prefix):
    ratio = model.config.beacon_ratio[0]
    first = model.model.layers[0].self_attn
    raw_width = first.num_key_value_heads * first.head_dim
    raw_block = 16
    page_bytes = raw_block * 2 * raw_width * 2
    max_width = max(layer.self_attn.beacon_k_up_proj.in_features +
                    layer.self_attn.beacon_v_up_proj.in_features for layer in model.model.layers)
    beacon_block = (((page_bytes // 2 - 2 * raw_width) // max_width) * ratio // raw_block) * raw_block
    if beacon_block < raw_block or beacon_block % ratio:
        raise ValueError('Cannot represent the selected token ratio in compact pages.')
    raw_layers, beacon_layers = nn.ModuleList(), nn.ModuleList()
    for index, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        raw_spec = ChunkedLocalAttentionSpec(block_size=raw_block,
            num_kv_heads=attn.num_key_value_heads, head_size=attn.head_dim,
            dtype=torch.bfloat16, attention_chunk_size=model.config.beacon_window)
        beacon_spec = BeaconSpec(block_size=beacon_block, num_kv_heads=1,
            head_size=attn.beacon_k_up_proj.in_features + attn.beacon_v_up_proj.in_features,
            head_size_v=0, dtype=torch.bfloat16, token_ratio=ratio, raw_width=raw_width,
            page_size_padded=page_bytes)
        raw_layers.append(PageLayer(f'{prefix}slimkv.{index}.raw', raw_spec, vllm_config))
        beacon_layers.append(PageLayer(f'{prefix}slimkv.{index}.beacon', beacon_spec, vllm_config, beacon=True))
    return raw_layers, beacon_layers
