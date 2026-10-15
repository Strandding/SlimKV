# Quick start

[中文](zh/quickstart.md)
## Required checkpoint

Use a local trained SlimKV checkpoint with `config.json`, the complete `.safetensors`/PyTorch weight shards, tokenizer files, and any shard index. Checkpoints retain `beacon_*` fields and tensor names. Model weights are not distributed with this repository. Use an existing SlimKV checkpoint or follow [training](training.md).

The minimal command from the repository root is:

```bash
python -m main.inference.infer \
  --model_name_or_path /path/to/slimkv-checkpoint \
  --prompt 'Explain why a KV cache helps language model inference.' \
  --max_new_tokens 32
```

CUDA defaults: BF16, FA2 prefill, latent cache and direct latent decode, prefill optimizations, CUDA Graphs and optional decoder-tail compilation. Install FA2 and any required [width-320 extension](benchmark.md#building-the-width-320-extension) first. Greedy generation computes only the requested final-token logits. Ranks, token ratio, RoPE policy and window come from the checkpoint. `--device cpu --dtype fp32` defaults to eager reconstruction. For Qwen2, use `--chat_template qwen --beacon_cache_mode auto --beacon_decode_backend reconstruct`; see [implementation status](implementation_status.md).

Use `--prompt_file /path/to/context.txt` for a long context. The file is used as one user message. Short prompts may never complete a Beacon window and therefore do not demonstrate cache savings. The model still computes prefill in the original attention dimension.

## Python API

```python
import torch
from slimkv import ModelArgs, apply_chat_template, get_model_and_tokenizer

model, tokenizer = get_model_and_tokenizer(
    ModelArgs(model_name_or_path='/path/to/slimkv-checkpoint',
              enable_beacon=True, dtype='bf16', attn_impl='flash_attention_2',
              beacon_cache_mode='latent', beacon_decode_backend='latent',
              beacon_decode_fast=True, beacon_decode_graph=True,
              beacon_decode_graph_full=True, beacon_decode_compile=True),
    device='cuda:0',
)
prompt = apply_chat_template(
    'llama-3', messages=[{'role': 'user', 'content': 'Summarize the following context: ...'}],
    tokenizer=tokenizer, add_generation_prompt=True,
).encoded
ids = torch.tensor([prompt['input_ids']], device='cuda:0')
mask = torch.tensor([prompt['attention_mask']], device='cuda:0')
model.memory.reset()
with torch.inference_mode():
    output = model.generate(input_ids=ids, attention_mask=mask,
                            max_new_tokens=32, do_sample=False, num_logits_to_keep=1)
print(tokenizer.decode(output[0, ids.shape[1]:], skip_special_tokens=True))
model.memory.reset()
```

Memory persists across forward calls. Reset it between independent requests and before switching cache/backend modes. Resetting does not release all CUDA Graph workspaces; avoid enabling Graph if its memory cost is unnecessary.

## Backend and compilation options

For eager decode over compressed Beacon K/V, no FlashAttention installation is needed:

```bash
python -m main.inference.infer \
  --model_name_or_path /path/to/slimkv-checkpoint \
  --attn_impl eager --beacon_cache_mode latent --beacon_decode_backend latent \
  --prompt_file /path/to/context.txt
```

For FlashAttention, install FA2 and build any required width-320 extension:

```bash
export SLIMKV_FA2_DECODE_EXTENSION_DIR="$PWD/.cache/fa2_decode_320"
python -m main.inference.infer \
  --model_name_or_path /path/to/slimkv-checkpoint \
  --attn_impl flash_attention_2 \
  --beacon_cache_mode latent --beacon_decode_backend latent \
  --beacon_decode_graph --prompt_file /path/to/context.txt
```

The CUDA CLI enables Graphs and compilation by default. `--no-beacon_decode_graph` disables Graphs; `--no-beacon_decode_graph_full` retains only layer graphs. `--no-beacon_decode_compile` disables compilation, and `--no-beacon_prefill_fast` disables prefill optimizations. Compilation currently runs only with Torch 2.11, B1 BF16 FA2 decode and Graphs; training, larger batches and unsupported configurations keep their original paths. Initial compilation adds startup time. The Python API keeps its existing defaults, so select the options explicitly as above. Both paths use original-dimension prefill and direct global-latent Beacon attention during eligible decode steps. `beacon_decode_backend=latent` follows `attn_impl`; `reconstruct` restores Beacon K/V first, while the legacy value `fa2` explicitly selects latent FA2 decode. The model supports aligned, equal-length unpadded dense batches; this CLI intentionally demonstrates one request. Ragged serving uses the separate experimental [vLLM integration](vllm.md); Qwen latent decode is unavailable. [Build, validation, benchmark](benchmark.md).

## No-checkpoint smoke test

```bash
python -m unittest discover -s tests -v
```

Tests use small CPU models, exercise prefill/decode/rollover/cache/checkpoint compatibility, and run the inference CLI against a temporary tiny checkpoint and tokenizer. Passing them does not validate a trained model's quality.
