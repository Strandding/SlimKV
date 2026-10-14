"""Lazy registration of SlimKV's vLLM model implementation."""


def register():
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        'SlimKVLlamaForCausalLM',
        'slimkv.vllm_plugin.model:SlimKVLlamaForCausalLM',
    )
