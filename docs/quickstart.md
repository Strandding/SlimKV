# Quick start

[中文](zh/quickstart.md)
## Required checkpoint

Use a local trained SlimKV checkpoint with `config.json`, the complete `.safetensors`/PyTorch weight shards, tokenizer files, and any shard index. Checkpoints retain `beacon_*` fields and tensor names. Model weights are not distributed with this repository. Use an existing SlimKV checkpoint or follow [training](training.md).

The minimal command from the repository root is:

```bash
python -m main.infer \
  --model_name_or_path /path/to/slimkv-checkpoint \
  --prompt 'Explain why a KV cache helps language model inference.' \
  --max_new_tokens 32
```

Defaults: `cuda:0`, BF16, eager prefill, greedy generation, `beacon_cache_mode=auto`, reconstruction decode. No rank, ratio, RoPE policy, or window override is applied. The loader reads these from the checkpoint. For CPU use `--device cpu --dtype fp32`; for Qwen2 use `--chat_template qwen`, with the limitations in [implementation status](implementation_status.md).

Use `--prompt_file /path/to/context.txt` for a long context. The file is used as one user message. Short prompts may never complete a Beacon window and therefore do not demonstrate cache savings. The model still computes prefill in the original attention dimension.

## Python API

```python
import torch
from slimkv import ModelArgs, apply_chat_template, get_model_and_tokenizer

model, tokenizer = get_model_and_tokenizer(
    ModelArgs(model_name_or_path='/path/to/slimkv-checkpoint',
              enable_beacon=True, dtype='bf16', attn_impl='eager'),
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
                            max_new_tokens=32, do_sample=False)
print(tokenizer.decode(output[0, ids.shape[1]:], skip_special_tokens=True))
model.memory.reset()
```

Memory persists across forward calls. Reset it between independent requests and before switching cache/backend modes. Resetting does not release all CUDA Graph workspaces; avoid enabling Graph if its memory cost is unnecessary.

## Optional reconstruction-free decode

For eager decode over compressed Beacon K/V, no FlashAttention installation is needed:

```bash
python -m main.infer \
  --model_name_or_path /path/to/slimkv-checkpoint \
  --attn_impl eager --beacon_cache_mode latent --beacon_decode_backend latent \
  --prompt_file /path/to/context.txt
```

For FlashAttention, install FA2 and build any required width-320 extension:

```bash
export SLIMKV_FA2_DECODE_EXTENSION_DIR="$PWD/.cache/fa2_decode_320"
python -m main.infer \
  --model_name_or_path /path/to/slimkv-checkpoint \
  --attn_impl flash_attention_2 \
  --beacon_cache_mode latent --beacon_decode_backend latent \
  --beacon_decode_graph --prompt_file /path/to/context.txt
```

Graph is optional and disabled by default. Both paths use original-dimension prefill and direct global-latent Beacon attention during eligible decode steps. `beacon_decode_backend=latent` follows `attn_impl`; `reconstruct` restores Beacon K/V first, while the legacy value `fa2` explicitly selects latent FA2 decode. The model supports aligned, equal-length unpadded dense batches; this CLI intentionally demonstrates one request. Ragged/paged serving and Qwen latent decode are not implemented. [Build, validation, benchmark](benchmark.md).

## No-checkpoint smoke test

```bash
python -m unittest discover -s tests -v
```

Tests use small CPU models, exercise prefill/decode/rollover/cache/checkpoint compatibility, and run the inference CLI against a temporary tiny checkpoint and tokenizer. Passing them does not validate a trained model's quality.
