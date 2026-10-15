# LongBench-v1 evaluation

[中文](zh/evaluation.md)

The Llama3 preset evaluates 14 English and code subsets. The training corpus is Chinese-only, so Chinese evaluation subsets are excluded. The synthetic subsets `passage_count` and `passage_retrieval_en` are excluded separately. `lcc` and `repobench-p` are included. Use `TASKS=all` to evaluate all LongBench-v1 subsets.

## Prepare data

Download the official [LongBench-v1 archive](https://huggingface.co/datasets/zai-org/LongBench/tree/main) and extract the selected task files:

```bash
python -m tools.data.download_longbench --output_dir data/datasets/longbench
```

The downloader pins revision `5e628be450b7e67fb7ae6e201bd6d8f7056f7672` and records archive and task-file SHA256 hashes in `dataset_manifest.json`. If `data.zip` is already available locally, add `--archive /path/to/data.zip`. Existing task files with different contents are never overwritten. This uses v1 JSONL files, excluding LongBench-E and LongBench-v2.

## Run SlimKV

Use a trained Llama3 SlimKV checkpoint containing weights, config and tokenizer. Feature ranks come from the checkpoint. An f4 checkpoint with `BEACON_RATIO=4` gives nominal t4f4, or 16× token-feature compression.

```bash
MODEL_NAME_OR_PATH=/path/to/slimkv-f4-checkpoint \
EVAL_DATA=/path/to/longbench \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/evaluation/eval_longbench_slimkv.sh
```

The preset uses BF16 greedy inference, token ratio 4, latent KV storage, original-dimension FA2 prefill, latent-space FA2 Beacon decode, CUDA Graphs, optional B1 decoder-tail compilation and `num_logits_to_keep=1`. Beacon Keys skip RoPE. It reads layer ranks and projection structure from the checkpoint.

### Select attention and decode paths

`ATTN_IMPL` chooses the original-dimension attention implementation. `DECODE_BACKEND` selects how compressed Beacon memory is used during decode. All combinations below retain latent KV storage.

| Environment settings | Prefill | Beacon decode | Default Graph setting |
| --- | --- | --- | --- |
| `ATTN_IMPL=eager` | Eager | Reconstruct, then eager attention | Off |
| `ATTN_IMPL=eager DECODE_BACKEND=latent` | Eager | Reconstruction-free latent eager | Off |
| `ATTN_IMPL=flash_attention_2 DECODE_BACKEND=reconstruct` | FA2 | Reconstruct, then FA2 | Off |
| `ATTN_IMPL=flash_attention_2` | FA2 | Reconstruction-free latent FA2 | On |

`DECODE_BACKEND=reconstruct` rebuilds Beacon K/V before attention. `DECODE_BACKEND=latent` computes attention directly over compressed Beacon K/V using the selected `ATTN_IMPL`. Eager supports both paths without a FlashAttention installation or width-320 extension. Latent eager uses FP32 attention scores and weighted sums, supports different K/V ranks and keeps the original head scaling.

The legacy `DECODE_BACKEND=fa2` explicitly selects latent FA2 decode regardless of prefill choice, preserving older commands. `DECODE_GRAPH=False` disables the optional Graphs independently.

Result directories include both attention choices, keeping different paths separate by default. When using `python -m main.evaluation.eval_longbench` directly, select both `--attn_impl` and `--beacon_decode_backend` explicitly; the script's automatic pairing applies to its environment variables.

FA2 must be installed for either FlashAttention path. Global latent ranks up to 256 use standard FA2. Ranks from 257 to 320 require the BF16 SM90 extension and `SLIMKV_FA2_DECODE_EXTENSION_DIR`; see [build instructions](benchmark.md#building-the-width-320-extension). Latent eager has no width-320 limit. Both direct latent paths require globally coupled Linear up projections and the [supported cache/RoPE configuration](implementation_status.md#latent-cache-and-direct-decode-requirements). Other projection structures should use reconstruction decode. Direct latent decode currently supports Llama3, not Qwen2.

The **final chat-formatted input** is limited to 31500 tokens. Longer inputs retain the first 15750 and last 15750 tokens, without decoding and re-encoding. Few-shot and code tasks retain their existing template behavior. The direct Python entry keeps its historical truncation before chat formatting unless `--truncate_after_chat True` is supplied.

Batch size is an upper limit of 32. Only inputs with exactly equal token lengths share a batch, preserving every selected input without padding. Most distinct lengths run at batch 1. `BATCH_SIZE` and `MAX_BATCH_TOKENS` control memory use. The default token budget is `BATCH_SIZE × MAX_LENGTH`; choose limits for your GPU and checkpoint.

## Small runs and resume

Run the first two rows of two subsets:

```bash
MODEL_NAME_OR_PATH=/path/to/slimkv-f4-checkpoint \
EVAL_DATA=/path/to/longbench \
TASKS="qasper hotpotqa" MAX_SAMPLES_PER_TASK=2 RESULT_DIR=smoke \
bash scripts/evaluation/eval_longbench_slimkv.sh
```

To evaluate t8f4, set `BEACON_RATIO=8` and use a separate result directory. Additional evaluator arguments can follow the script name. `bash scripts/evaluation/eval_longbench_slimkv.sh --help` lists the environment controls.

Set `LOAD_RESULT=True` to reuse completed task files and their original timing records. Saved configuration must match, apart from `load_result`. Incomplete tasks rerun from the beginning. Keep checkpoint weights and task files unchanged when resuming, and use a new `RESULT_DIR` for a different protocol. Timing records are not invented for older prediction files that lack them.

## Results and timing

Results default to `data/results/longbench/slimkv-t4-flash_attention_2-latent/`:

| File | Contents |
| --- | --- |
| `<task>.json` | Score on the first line, followed by predictions and original sample indices |
| `<task>.timing.json` | Generation time, token counts, batch sizes and peak allocated GPU memory |
| `task_timings.csv` | One row per completed timed subset, updated after each subset |
| `scores.json` | Subset scores and their unweighted average |
| `progress.json` | Snapshot updated after each batch, with completed tasks and current sample count |
| `config.json`, `run_manifest.json` | Arguments, source hashes, checkpoint config/task-file hashes, software and hardware |

`time_sec` measures generation wall time, including prefill, decode, data transfer, text decoding and progress writes. Model loading and dataset preprocessing are reported separately. This is not attention kernel time. The preset uses one process on one GPU; progress is a saved snapshot and does not prove a process is still alive. Config hashes do not identify weight bytes, so retain the exact checkpoint with each run.

The average covers only the selected tasks. The English preset is not the full bilingual LongBench score. `BEACON_RATIO` controls token compression, while feature compression comes from checkpoint ranks. Their product does not describe total GPU memory reduction.

Independent batches reset `model.memory`. QA tasks use the existing newline stopping rule, and each subset keeps its generation cap and metric. Batched floating-point computation can produce different generated tokens from batch 1.

## Other entry points

The older `eval_longbench_llama31_8B.sh` and Qwen presets remain available with their existing defaults. For reproducible length-selected latency fixtures, use [fixture preparation](benchmark.md#preparing-small-longbench-v1-fixtures).

| Evaluation | Entry point | Inputs |
| --- | --- | --- |
| Needle-in-a-Haystack | `python -m main.evaluation.eval_needle --help` | Local haystack and optional judge |
| Synthetic passkey retrieval | `python -m main.evaluation.eval_passkey --help` | Seed, lengths and depths |
| LM perplexity | `python -m main.evaluation.eval_lm --help` | Local data and matching tokenization/stride |
| Diagnostics | `main.analysis.eval_longbench_beacon_probe`, `main.analysis.eval_longbench_chunk2_heatmap` | Probe and attention analysis |

## Decode compression policy

`DECODE_COMPRESS=True` is the default. Set `DECODE_COMPRESS=False` to keep prompt compression while disabling new Beacon insertion during decode. The evaluator argument is `--beacon_decode_compress False`. This setting is independent of `ATTN_IMPL`, `DECODE_BACKEND`, Graphs and compilation. Existing prompt KV uses the same compression settings, and generated raw KV grows with output length. The default result directory gains `-rawdecode` when disabled, keeping both policies separate.
