# vLLM integration

[中文](zh/vllm.md)

SlimKV can use vLLM's scheduler and KV allocator on a single GPU with a converted BF16 Llama checkpoint. Prefill retains the original attention dimensions. Decode uses the existing reconstruction-free FA2 implementation with the checkpoint's actual per-layer K/V ranks.

## Installation

Use a separate environment from the training and standalone inference environment. The integration was tested with Python 3.12, PyTorch 2.11.0+cu129, Transformers 5.8.0, FlashAttention 2.8.3 and vLLM `0.24.1.dev0+gee0da84ab.d20260731.cu129`. It uses vLLM V1 internals from that revision. Compatibility with other revisions has not been tested.

In an environment with the tested vLLM and CUDA dependencies installed, run this from the repository root:

```bash
uv pip install -e ./integrations/vllm
```

This installation profile exposes the same SlimKV source and registers its vLLM plugin. It keeps the main package's Transformers 4.40.x dependency for training and standalone inference separate. Install one profile per environment.

Build the [width-320 FA2 extension](benchmark.md#building-the-width-320-extension) when the checkpoint's ranks require it, and export `SLIMKV_FA2_DECODE_EXTENSION_DIR` as described there. Use the checkpoint's existing weights, configuration and tokenizer.

## Inference

```bash
python -m main.inference.infer_vllm \
  --model /path/to/slimkv-checkpoint \
  --token-ratio 8 --max-num-seqs 4 \
  --prompt 'Explain why a KV cache helps language model inference.' \
  --prompt 'What does feature compression change in a KV cache?'
```

For longer inputs, pass `--prompts-json /path/to/prompts.json`, a JSON list of strings. Each string becomes an independent chat request. This entry point does not truncate inputs. Set `--max-model-len` to a length supported by the checkpoint and available memory.

The Python API uses:

```python
from vllm import LLM, SamplingParams
from slimkv.vllm_plugin import register

register()
llm = LLM(
    model='/path/to/slimkv-checkpoint',
    dtype='bfloat16', enforce_eager=True, tensor_parallel_size=1,
    max_model_len=16384, max_num_seqs=4, max_num_batched_tokens=16384,
    gpu_memory_utilization=0.55,
    enable_chunked_prefill=False, enable_prefix_caching=False,
    async_scheduling=False, generation_config='auto',
    disable_hybrid_kv_cache_manager=False,
    hf_overrides={
        'architectures': ['SlimKVLlamaForCausalLM'],
        'slimkv_native_paged': True,
        'beacon_ratio': [8], 'beacon_ratio_mix': 'sequence',
    },
)
outputs = llm.generate(
    [{'prompt_token_ids': [128000, 9906]}],
    SamplingParams(temperature=0, max_tokens=32),
)
```

Supply tokenized prompts appropriate for your checkpoint. The CLI applies its chat template. `enforce_eager=True` disables vLLM's outer graph capture. SlimKV retains its own validated decode graphs.

## Cache ownership and scheduling

vLLM owns the physical pages, block tables and request lifetimes. Each layer has two storage groups: current-window raw K/V and compressed Beacon K/V. Raw pages cover nonoverlapping attention windows. Beacon pages preserve history at the checkpoint's exact K/V ranks, with the original raw sink token stored in the first page. Page padding does not change the ranks or the attention calculation.

Each scheduled request restores transient SlimKV workspaces from its block tables, executes the original B1 attention math, writes new KV values back and releases the request KV views. Completed requests return their pages to vLLM. Multiple requests can enter and leave a running batch, and prefill can share a scheduling step with other requests' decode. The current implementation executes those requests sequentially within each step. It does not yet provide a fused batch or an attention kernel that directly reads paged KV.

The validated configuration is single-device Llama, BF16, interleaved full-coverage Beacons, nonoverlapping windows and at most one sink token. Keep prefix caching, speculative decoding, quantization and chunked prefill disabled. Prompt log-probability output, tensor/pipeline parallelism and Qwen are not supported by this entry point.

## Validation and performance

On H20 with the original f4 checkpoint and t8 compression, both 4,009- and 6,194-token inputs matched the standalone implementation bitwise across prefill and 63 fixed decode steps. Eight fixed LongBench-v1 samples also retained identical generated tokens and official scores. Their continuous scheduling reached four simultaneous requests and five mixed prefill/decode steps.

The native paged B1 prototype measured **32.73 and 32.59 decode tokens/s**, with **0.727 and 1.139 seconds** for prefill. Measurements used one warmup and four runs, fixed decode inputs and synchronized host timing of `LLMEngine.step`, including scheduler, sampler and teacher-forcing overhead.

Page-boundary and window-transition checks at 399, 1,023 and 4,095 prefix tokens also matched all 17 logit steps bitwise. Physical pages were reused across requests, and the allocator's free-block count returned to its initial value after every request.

The [standalone FA2 path](benchmark.md#b1-decode-graphs) remains the faster implementation at 82.38 and 82.09 tokens/s under its documented protocol. The native prototype gathers pages into contiguous workspaces every step. Removing this traffic and batching the original attention work are the next performance tasks.
