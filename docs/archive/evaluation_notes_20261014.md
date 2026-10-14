# Historical evaluation and profiling instructions

[中文原始记录](evaluation_notes_20261014_zh.md)

English companion to the previous evaluation guide. These commands and B1 support statements describe the earlier implementation; see [current evaluation](../evaluation.md), [benchmark](../benchmark.md) and [status](../implementation_status.md). The historical commands are preserved below. No old prototype result is promoted to complete paper reproduction.

## Ordinary evaluation and controls

Llama3 can store latent Beacon K/V and reconstruct before standard attention; optional FA2 performs global-latent decode. Qwen2 still stores full-dimensional K/V. `slimkv/latency` uses a separate prototype loader.

The LongBench launcher expects a trained checkpoint and local task data. Independent requests call `model.memory.reset()`; the main evaluator resets per batch. Task output files start with a score then prediction records. Cache modes: default auto uses latent only when compatible, full forces full cache, explicit latent rejects incompatible configuration. Requirements are interleave/window=stride/single-window/positive ratio/low-rank K and V, with no chunk-fixed K RoPE/K residual/beacon_skip. Training is full-cache. Reset before mode/training-state changes; use greedy controls. Small subsets use `--tasks qasper hotpotqa --max_samples_per_task 2`. SDPA had CPU regression; eager and standard FA2 prefill/latent decode had separate GPU records.

`BEACON_RATIO` means token ratio. f4 with token4/8 nominally means t4f4/t8f4, while raw/sink lower actual compression.

## Reconstruction benchmarking

JSONL fixtures use original LongBench-v1 fields plus task/source_index. Existing task templates and Llama-3 chat formatting are used; code completion omits chat formatting. Quality uses official caps/greedy/QA newline rules; output includes predictions/tokens/scores.

Timing uses fixed generation, no EOS stop, prefill first token and subsequent decode. Reset every call; warm modes, then symmetric full→compact→latent→latent→compact→full, two trials/mode/round. Synchronize host wall time; tokenization/scoring/IO/logits recording excluded. Compact full only frees sink backing views inside the harness; it preserves values/math. Report live/peak allocation and logical/unique K/V, excluding allocator reserved bytes. The optional `--disable_optional_flash_import` disables an unused broken optional extension in the harness only.

## Latent FA2 and build

Backend fa2 is opt-in, reconstruct remains default. Training/prefill keep original dimensions. Historical/new Beacon K/V participate directly in decode; raw/sink stay full dimension, branches merge by LSE into one softmax. Q2 inserted-Beacon causality is separately tested.

The historical validated scope was Llama global Linear up/full-coverage/attend_prev/no Beacon K RoPE/raw K RoPE/no latent positional bias, BF16 unpadded B1 with one raw input per step. Batches/multiple raw tokens reconstructed then. Reset cache/backend changes. Later aligned-batch support is described in current status.

FA2 2.8.3 serves dimensions≤256. Unequal K/V ranks temporarily pad to a common width; >256 uses the compiled BF16 SM90 width320 forward extension. Actual rank remains in cache; f4 max K/V304/312, scale1/sqrt128. Runtime binary ABI must match Torch/Python/CUDA/C++; tested Torch2.11.0+cu129/TF4.40.2 reused an existing FA2 binary. No backward/other GPU build. Standard FA2 prefill is selected by attn_impl=flash_attention_2; CUDA BF16/FP16+Triton fuse RoPE/merge; CPU/grad/noTriton retain Torch. JIT belongs to warmup.

Performance alternates compact/reconstruct/fa2 then reverse, using the compact warmup's identical fixed decoder inputs. Forward/argmax/sync are timed; quality greedy is separate. Saved teacher-forced logits and kernel coverage do not imply BF16 bitwise equality between different attention algorithms. Optional forced_tokens_json maps sample indices to input token lists; reference_decode_source switches an archived compatible decoder on the same weights, adding a fourth mode/reverse control. skip_quality limits the run to performance; default validation saves16steps.

Fast B1 bookkeeping retains GPU token history/roles/raw indices/positions; beacon_decode_fast=false selects the older bookkeeping. Graph is off by default, opt-in for input norm and post-attention residual/postnorm/MLP/residual, Q1/Q2 captured separately during warmup. Training/CPU/prefill/hidden-or-attention diagnostics use applicable ordinary paths; multiple-token Memory can resume single-token decode. Dtype/device changes clear graphs. main.eval_longbench accepts the same backend flags.

compare_b1 tests compact/reconstruct/legacy/fast/fast+Graph forward and reverse, four medians over two rounds, with saved FA2 logit parity checks. First Graph warmup added~516MiB live allocation; all formal modes retain the graphs, so their peak differences do not isolate overhead.

## Profiling and GPU checks

Unprofiled timing is separate from Nsys/NCU. Nsys traces cuda/nvtx only with sampling/context-switches disabled, fork-before-exec and node-level Graph traces; ProfilerAPI delimits warmed model capture. phase=prefill selects prefill; mode=capture saves real layer0/2/28 Q/K/V/output/LSE. Kernel-mode replay supplies NCU. The archived NCU command collects LaunchStats/SpeedOfLight/MemoryWorkloadAnalysis/SchedulerStats/WarpStateStats/SourceCounters, with source import and one kernel launch. Durations diagnose mechanisms, not unprofiled speedups.

validate_decode_ops independently checks fused operators. profile_fa2 decode_variant legacy/fast/graph chooses the path; no extra OSRT or CUDA memory events are requested. Graph replay does not rerun Python MLP NVTX wrappers, so missing MLP range does not mean removed computation. validate_b1_decode verifies transitions/replay; quick BF16 suits memcheck.

## Other accuracy and historical prototype

NIAH defaults to ROUGE; gpt_eval requires optional judge dependencies and environment API credentials. Judge/seeds were not the paper's final protocol. Other entries include passkey/InfBench/perplexity.

The archived latency command uses one repeat and zero warmup to avoid repeated-memory accumulation, but still cannot substitute for the paper speed table: the reconstruct control stores full K/V and the hybrid FA2 fallback can lose masks. Use eager for that prototype. Probe/heatmap tools and scripts/evaluation preserve diagnostic functionality; generated figures/predictions are local artifacts.

scripts/vllm targets original-model baselines with config/vllm_models.sh, not a SlimKV paged cache or serving model integration.

## Preserved command blocks

The following appear in the original guide in order: LongBench launcher; reconstruction benchmark; FA2 build/validation; latent FA2 benchmark; Nsys; NCU; accuracy help; eager prototype. Paths are placeholders, not public weight downloads.

```bash
MODEL_NAME_OR_PATH=/path/to/checkpoint-2000 \
EVAL_DATA=/path/to/long-llm/longbench \
CUDA_VISIBLE_DEVICES=0 NPROC_PER_NODE=1 \
BEACON_RATIO=8 \
bash scripts/evaluation/eval_longbench_llama31_8B.sh
```

```bash
python -m tools.benchmark_reconstruction \
  --model_name_or_path /path/to/checkpoint-2000 \
  --samples_jsonl /path/to/samples.jsonl \
  --output_dir data/results/reconstruction-benchmark \
  --attn_impl eager --beacon_ratio 8 \
  --decode_tokens 64 --warmup 1 --rounds 2 \
  --perf_sample_indices 0 2
```

```bash
git clone --depth 1 --branch v2.8.3 \
  https://github.com/Dao-AILab/flash-attention.git .cache/flash-attention-v2.8.3
git -C .cache/flash-attention-v2.8.3 submodule update --init csrc/cutlass
python -m tools.build_fa2_decode \
  --fa2_source .cache/flash-attention-v2.8.3 \
  --output_dir .cache/fa2_decode_320 --jobs 2
export SLIMKV_FA2_DECODE_EXTENSION_DIR="$PWD/.cache/fa2_decode_320"
python -m tools.validate_fa2_decode \
  --extension_dir "$SLIMKV_FA2_DECODE_EXTENSION_DIR" \
  --output data/results/fa2-kernel-validation.json
```

```bash
python -m tools.benchmark_fa2_decode \
  --model_name_or_path /path/to/checkpoint-2000 \
  --samples_jsonl /path/to/samples.jsonl \
  --output_dir data/results/fa2-benchmark \
  --attn_impl flash_attention_2 --beacon_ratio 8 \
  --decode_tokens 64 --warmup 1 --rounds 2 \
  --perf_sample_indices 0 2
```

```bash
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --cuda-graph-trace=node --trace-fork-before-exec=true \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  -o data/results/slimkv-decode \
  python -m tools.profile_fa2 --mode model --phase decode \
  --model_name_or_path /path/to/checkpoint-2000 \
  --samples_jsonl /path/to/samples.jsonl --sample_index 2 \
  --decode_variant graph --output_dir data/results/slimkv-decode-data
```

```bash
ncu --nvtx --nvtx-include 'kernel_probe/' \
  --kernel-name-base demangled --kernel-name 'regex:.*flash_fwd.*' \
  --launch-count 1 --replay-mode kernel \
  --section LaunchStats --section SpeedOfLight \
  --section MemoryWorkloadAnalysis --section SchedulerStats \
  --section WarpStateStats --section SourceCounters --import-source yes \
  --export data/results/slimkv-kernel \
  python -m tools.profile_fa2 --mode kernel \
  --case_file /path/to/captured-case.pt \
  --output_dir data/results/slimkv-kernel-data
```

```bash
python -m main.eval_needle --help
python -m main.eval_passkey --help
python -m main.eval_infbench --help
python -m main.eval_lm --help
```

```bash
python -m main.benchmark_latency \
  --lowrank_model_or_ckpt /path/to/checkpoint-2000 \
  --mode beacon_lowrank_hybrid_latent \
  --attn_impl eager --beacon_ratio 8 \
  --prompt_len 8192 --gen_len 128 --repeat 1 --warmup 0 \
  --output_json /tmp/slimkv-latency.json
```
