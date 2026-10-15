"""Optional decoder-tail compilation with the original variance reduction."""
from functools import lru_cache
import warnings

import torch
from .latent_decode import uses_fa2_decode


def eligible(layer, hidden):
    config = layer.self_attn.config
    return (getattr(config, 'beacon_decode_compile', False)
            and not layer.training and not torch.is_grad_enabled()
            and hidden.is_cuda and hidden.dtype == torch.bfloat16
            and hidden.shape[0] == 1 and hidden.shape[1] in (1, 2)
            and getattr(config, '_slimkv_decode_graph_step', False)
            and getattr(config, 'beacon_decode_graph', False)
            and uses_fa2_decode(config)
            and config.hidden_act == 'silu')


@lru_cache(None)
def compiled_tail():
    # Other Torch releases keep the original path until numerically validated.
    if tuple(int(x) for x in torch.__version__.split('.')[:2]) != (2, 11):
        warnings.warn('SlimKV decode compilation is validated with Torch 2.11; using the original tail.',
                      stacklevel=2)
        return None

    @torch.library.custom_op('slimkv::decode_variance', mutates_args=())
    def variance(hidden: torch.Tensor) -> torch.Tensor:
        return hidden.float().pow(2).mean(-1, keepdim=True)

    @variance.register_fake
    def variance_fake(hidden):
        return torch.empty((*hidden.shape[:-1], 1), device=hidden.device, dtype=torch.float32)

    def tail(layer, attention_output, residual):
        hidden = residual + attention_output
        norm = layer.post_attention_layernorm
        normalized = hidden.float() * torch.rsqrt(variance(hidden) + norm.variance_epsilon)
        normalized = norm.weight * normalized.to(hidden.dtype)
        return hidden + layer.mlp(normalized)

    return torch.compile(tail, fullgraph=True, dynamic=False, options={
        'emulate_precision_casts': True,
        'epilogue_fusion': False,
        'triton.cudagraphs': False,
    })
