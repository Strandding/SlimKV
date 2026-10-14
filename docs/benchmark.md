# Latency and memory benchmarking

[中文](zh/benchmark.md)
Use tools-based controls for the current model; `main.benchmark_latency` is an independent historical prototype with known repeat/reset and baseline-semantic issues. Do not use its numbers as paper reproduction evidence. All paths and flags below are invoked from the repository root.

## Preparing small LongBench-v1 fixtures

Supply official local task JSONL files, then:

```bash
python -m tools.prepare_longbench \
  --dataset_dir /path/to/longbench \
  --output data/results/fixtures/samples.jsonl \
  --tasks qasper hotpotqa multifieldqa_en lcc \
  --max_samples_per_task 2 --min_length 2500 --max_length 6000
```

The tool selects the first matching rows in original order and adds `task` and `source_index`. It preserves the official row fields and writes source/output hashes in `samples.manifest.json`. The length filter refers to the dataset's `length`, not the tokenized chat prompt. No context is truncated. Eight rows in this task order reproduce the historical fixture identities if the dataset revision matches.

Performance tools expect a local Llama3 checkpoint and long prompts: the current fixture contract requires more than two Beacon windows and at most 31,550 prompt tokens. They are not arbitrary short-prompt benchmarks. Quality uses task generation caps and stopping rules; timing uses fixed-length decode separately.

## Reconstruction control

```bash
python -m tools.benchmark_reconstruction \
  --model_name_or_path /path/to/slimkv-checkpoint \
  --samples_jsonl data/results/fixtures/samples.jsonl \
  --output_dir data/results/reconstruction-benchmark \
  --attn_impl eager --beacon_ratio 8 \
  --decode_tokens 64 --warmup 1 --rounds 2 --perf_sample_indices 0 2
```

The same model alternates full, compact full and latent-reconstruction cache modes with symmetric ordering. `full_compact` clones the first sink view inside the harness to release excess backing storage; it changes neither values nor attention mathematics. Every call resets Memory. This is a reconstruction control using the no-Beacon-K-RoPE checkpoint, not the paper's K-RoPE-retaining trained reconstruction variant.

## Building the width-320 extension

Install FA2 2.8.3 and a compatible development toolchain first. This build compiles BF16 SM90 **forward only**; no backward or A100/other-architecture support is claimed.

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

The build manifest includes Torch/CUDA, upstream FA2/CUTLASS commits and source/binary SHA256. Standard FA2 serves widths ≤256; larger actual ranks are temporarily padded to 320 for computation. Persistent cache stores actual ranks, with no padding. A failed/mismatched extension import must be resolved before claiming a validated latent backend.

## Current FA2 and Graph comparison

```bash
python -m tools.benchmark_fa2_decode \
  --model_name_or_path /path/to/slimkv-checkpoint \
  --samples_jsonl data/results/fixtures/samples.jsonl \
  --output_dir data/results/fa2-benchmark \
  --attn_impl flash_attention_2 --beacon_ratio 8 \
  --decode_tokens 64 --warmup 1 --rounds 2 \
  --perf_sample_indices 0 2 --compare_b1
```

`--compare_b1` compares compact full, reconstruction, legacy FA2, GPU bookkeeping and partial Graph in one model. Omit it for the three-path full/reconstruction/FA2 comparison. `--reference_decode_source` separately loads an archived decoder for a same-model control; it cannot be combined with `--compare_b1`.

Timing uses synchronized host wall time and includes forward, argmax, and synchronization boundaries. Tokenization, scoring, file IO, diagnostics and logit saving are outside the timing interval. Warmup includes Triton compilation and Graph capture. Each generated length of 64 contains the prefill first token and 63 decode steps, with no early EOS in the timing phase. All modes receive the same teacher-forced decode inputs; quality is evaluated separately with normal greedy generation. Four measurements per mode (two symmetric rounds) produce medians; retain every raw trial.

Results include `results.json`, predictions and logit tensors, checkpoint config/input/tool hashes, software/GPU metadata and actual kernel coverage. Live/peak CUDA allocation differs from unique persistent KV storage and allocator reserved bytes. Prewarmed Graph workspaces remain live in all comparison modes, so mode-to-mode peak differences are not isolated Graph memory cost.

## Recorded local results (not paper reproduction)

H20, private f4 Llama-3.1-8B BF16, Torch 2.11.0+cu129, Transformers 4.40.2, FA2 2.8.3, Triton 3.6.0; token ratio 8, window=stride 1024.

| Input tokens | Compact full FA2 decode | Legacy fused latent FA2 | GPU bookkeeping | Partial Graph |
|---|---:|---:|---:|---:|
| 4,009 | 26.343 ms/token | 24.772 | 24.263 | 18.381 |
| 6,194 | 26.397 ms/token | 24.671 | 24.227 | 18.671 |

Same-model symmetric four-median protocol; reference commit `f6f5cb6`. A separate eager→standard-FA2 prefill control measured 867.59→736.25 ms and 1398.39→1168.39 ms on these fixtures. Persistent KV after prefill was 132.906 / 30.563 MiB in the latent modes versus 179.875 / 103.125 MiB in compact full. Beacon features are reduced 4× on average; raw/sink and model/workspace memory remain. These cache measurements are not peak CUDA usage or a 4× whole-GPU saving. First B1 Graph warmup added 540,804,096 live bytes (~515.75 MiB) without changing KV bytes.

The eight-sample quality test and two saved 17×128256 logit arrays provide local correctness evidence; they do not reproduce complete LongBench or 128K efficiency. [Machine-readable evidence](results/validation_20261014.json) contains audited summaries and hashes of the underlying source results. Raw logs/tensors/traces remain local and are not bundled as a public artifact release.

## Aligned dense batch scaling

```bash
python -m tools.benchmark_batch_decode \
  --model_name_or_path /path/to/slimkv-checkpoint \
  --samples_jsonl data/results/fixtures/samples.jsonl \
  --forced_tokens_json /path/to/fixed-tokens.json \
  --output_dir data/results/batch-scaling \
  --batch_sizes 1 2 4 8 --sample_indices 0 2 --rounds 2
```

Fixed tokens are JSON mapping sample indices to decoder input-token lists, for example `{"0": [123, 456], "2": [789, 10]}`; supply at least 63 token IDs per fixture for the default 64-token generation run (the short example describes the schema only). Obtain the lists from the warmup/reference run's `forced_token_ids` in saved `logits_<index>.pt`. Do not use arbitrary tokens for a reported same-input comparison. Exact fixed-input hashes are recorded.

The tool repeats each prompt across independent cache rows, prewarms all batch shapes and retains their Graph workspaces. This measures controlled dense scaling, not independent-request or ragged serving throughput. Four measurements per batch/fixture, 48 total for the recorded six batches. Same-batch Graph/eager logits are checked on all rows; BF16 across different batch sizes is not asserted to be bitwise identical.

| Batch | 4,009 tokens: step ms / aggregate tok/s | 6,194 tokens: step ms / aggregate tok/s |
|---|---:|---:|
| 1 | 18.183 / 55.00 | 17.939 / 55.74 |
| 2 | 18.646 / 107.26 | 18.506 / 108.08 |
| 4 | 18.763 / 213.19 | 18.477 / 216.49 |
| 8 | 20.139 / 397.23 | 19.092 / 419.03 |
| 16 | 25.827 / 619.50 | 20.387 / 784.81 |
| 32 | 40.917 / 782.08 | 24.714 / 1294.82 |

A separate 16-step Nsys capture of the 6,194-token fixture measured mean SMs Active 19.21% / 30.69% / 54.92% at B1/B8/B32. The union of GPU kernel intervals covered 31.86% / 41.94% / 66.40% of the kernel span. These are different metrics, and neither reaches sustained 100%. Standalone capture-process peak CUDA allocation was 22.38 / 34.49 / 75.99 GiB; throughput runs retained every prewarmed batch Graph and are not isolated memory measurements. B64 was not tested.

## Profiling and GPU validation

Use `nsys` from your PATH (or the actual installation path). Capture only after warmup:

```bash
nsys profile --sample=none --cpuctxsw=none --trace=cuda,nvtx \
  --trace-fork-before-exec=true --cuda-graph-trace=node \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --force-overwrite=false -o data/results/nsys-decode \
  python -m tools.profile_fa2 --mode model --phase decode \
    --model_name_or_path /path/to/slimkv-checkpoint \
    --samples_jsonl data/results/fixtures/samples.jsonl \
    --output_dir data/results/nsys-inputs --decode_variant graph --batch_size 1
```

Inspect `python -m tools.profile_fa2 --help` for sample and step settings. Add `--gpu-metrics-devices=0 --gpu-metrics-frequency=10000` only on supported devices with permission to collect metrics. Percentages are interpreted in the actual kernel span, not over setup/teardown. Graph node tracing exposes GPU work; missing Python MLP ranges does not mean MLP computation disappeared.

The profiler can save actual per-layer Q/K/V cases. `--mode kernel --case_file ...` replays them for NCU; inspect `ncu --help` for your installed version and collect the real main/combined kernel. Profiling durations are diagnostics and must not replace unprofiled benchmark medians. The historical NCU cases showed width-320 Q2 register spilling, but host launch gaps and small operations also contributed substantially to whole-model latency.

Independent GPU entries are `tools.validate_decode_ops`, `tools.validate_b1_decode`, and `tools.validate_fa2_decode` (`--batch_sizes` for aligned batches). Record GPU model, dtype, shapes, extension SHA and memcheck logs; CPU tests mock FA2 algebra and do not validate CUDA binaries. No long training or full evaluation is started by CPU CI.
