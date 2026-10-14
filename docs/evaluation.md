# Accuracy evaluation

[中文](zh/evaluation.md)
Evaluation entry points run from the repository root. A trained SlimKV checkpoint and the exact task data are required. Default reconstruction, optional latent FA2 decode and the independent latency prototype have different semantics; see [implementation status](implementation_status.md).

## LongBench-v1

Prepare official task JSONL files from [THUDM/LongBench](https://huggingface.co/datasets/THUDM/LongBench), with each task at `<dataset_dir>/<task>.jsonl`. Use LongBench-v1, not LongBench-v2. Keep the downloaded dataset revision and hashes with your results.

A small, bounded run:

```bash
python -m main.eval_longbench \
  --model_name_or_path /path/to/slimkv-checkpoint \
  --eval_data /path/to/longbench \
  --output_dir data/results/longbench --result_dir smoke \
  --enable_beacon --attn_impl eager --dtype bf16 --chat_template llama-3 \
  --beacon_ratio 8 --beacon_ratio_mix sequence \
  --tasks qasper hotpotqa --max_samples_per_task 2 \
  --sample_seed 42 --sample_shuffle False --batch_size 1
```

This selects the first two original rows of each task; it does not reproduce the historical length-filtered sample selection. For those benchmark fixtures, use [fixture preparation](benchmark.md#preparing-small-longbench-v1-fixtures). Evaluation honors its existing `max_length` and task truncation/template behavior; preserve these settings when comparing runs.

For the full preset:

```bash
MODEL_NAME_OR_PATH=/path/to/slimkv-checkpoint \
EVAL_DATA=/path/to/longbench \
CUDA_VISIBLE_DEVICES=0 NPROC_PER_NODE=1 \
BEACON_RATIO=8 ATTN_IMPL=eager \
bash scripts/evaluation/eval_longbench_llama31_8B.sh
```

The preset's task set excludes Chinese subsets. Check the task list rather than calling it the full bilingual benchmark. `BEACON_RATIO` controls **token** compression; feature compression comes from checkpoint ranks. t8f4 nominally means 8× token and 4× feature compression, not 32× reduction in total GPU memory.

## Controls and outputs

`--beacon_cache_mode full` and `--beacon_cache_mode latent` compare caches using the same checkpoint. `--beacon_decode_backend fa2 --attn_impl flash_attention_2` opts into eligible latent decode and standard FA2 prefill; build requirements are in [benchmark](benchmark.md). Use B1 for general QA evaluation: padded variable-length batches do not satisfy latent FA2 decode eligibility.

Independent requests must reset `model.memory`; the evaluation entry resets between batches. Preserve greedy decoding, chat template, token ratio, RoPE flags, task-specific generation caps and stopping rules. QA can stop on newline; code completion is handled separately. Floating-point path changes need not preserve generated tokens.

Under `<output_dir>/<result_dir>/`, task `.json` files contain the score on the first line followed by prediction records, and `config.json` saves arguments. `run_manifest.json` records argument configuration, actual source hashes, checkpoint config and task-file hashes, software versions and GPU properties. Aggregate logs are written under `output_dir`. Config hashes do not identify weight bytes: retain/hash exact shards separately. Use environment-based Hugging Face authentication rather than putting credentials in logged CLI flags.

## Other accuracy entry points

| Evaluation | Entry point | Additional inputs / caveats |
|---|---|---|
| Needle-in-a-Haystack | `python -m main.eval_needle --help` | Local haystack, optional judge dependencies/API credentials; current judging and three-seed aggregation do not reproduce the paper |
| Synthetic passkey retrieval | `python -m main.eval_passkey --help` | Generated contexts, explicit seed and length/depth settings; not the paper NIAH protocol |
| LM perplexity | `python -m main.eval_lm --help` | Local evaluation data and matching tokenization/stride |
| Generation / MMLU / topic | `main.eval_generation`, `main.eval_mmlu`, `main.eval_topic` | Existing entry points; inspect `--help` and required schemas; not all are paper experiments |
| Probe / heatmap | `main.eval_longbench_beacon_probe`, `main.eval_longbench_chunk2_heatmap` | Diagnostic ablations, not official aggregate quality evaluation |

A CPU-only smoke test is `python -m unittest discover -s tests -v`; it downloads no data. It proves local mechanics, not trained-model quality.
