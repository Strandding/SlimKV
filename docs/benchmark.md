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

The tool selects the first matching rows in original order and adds `task` and `source_index`. It preserves the official row fields and writes source/output hashes in `samples.manifest.json`. The length filter refers to the dataset's `length`, not the tokenized chat prompt. No context is truncated. The selected rows and their source indices are recorded in the manifest.

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

The build manifest includes Torch/CUDA, upstream FA2/CUTLASS commits and source/binary SHA256. The extension supports width 128 for raw KV and widths 200–256/320 for static latent decode. Ordinary decode uses standard FA2 at widths ≤256. Larger actual ranks are temporarily padded to 320 for computation. Persistent cache stores actual ranks, with no padding.

## Prefill indexing and rotary fusion

`--beacon_prefill_fast True` is the default for Llama. During inference, unpadded interleaved FA2 windows with a latent cache share token indices across layers and skip raw-cache copies at completed windows. BF16 Q/K rotary operations use one Triton kernel while leaving Beacon K unrotated. Multiplications still round to BF16 before addition, matching the original operations. Projection shapes, weights, ranks, attention scale and masks stay the same. Attention itself runs in the original feature dimensions.

Training, padded inputs and unsupported configurations use the existing path. Unsupported rotary kernels also fall back to the existing rotary operations. For a direct comparison, pass `--beacon_prefill_fast False` to `main.eval_longbench`, or set `model.config.beacon_prefill_fast = False`. The first eligible call compiles the Triton kernel.

On H20, with the BF16 Llama-3.1-8B f4 checkpoint and t4f4, one warmup followed by four alternating measurements gives these prefill medians. These are synchronized forward times, including all prompt windows and final-token logits, without decode.

| Input tokens | Original (s) | Optimized (s) | Speedup |
| ---: | ---: | ---: | ---: |
| 4,009 | 0.796 | 0.695 | 1.14× |
| 6,194 | 1.283 | 1.112 | 1.15× |
| 31,500 | 7.427 | 6.585 | 1.13× |

The logits and every layer's persistent KV tensors match bitwise on these inputs, window-boundary cases and B4/B32 batches of distinct 31,500-token prompts. A single paired validation run takes 26.28 → 24.54 s at B4 and 196.98 → 187.12 s at B32; these batch timings are individual observations, rather than repeated medians. Large batches spend more time in matrix multiplications, which this optimization leaves unchanged.

Two prompts followed by 128 fixed decode steps also retain bitwise logits and final KV across Beacon insertions and window transitions. Greedy token sequences and official scores match on eight fixed LongBench-v1 samples. The CUDA test in `tests/test_llama_prefill.py` checks BF16 rotary rounding, tensor strides, sinks and Beacon history; it is skipped in CPU CI.

## B1 decode graphs

With the rebuilt extension, `--beacon_decode_graph True` also enables whole-model graphs for eligible B1 single-token latent FA2 decode. KV buffers use fixed addresses and append in place. Beacon insertion, larger batches and diagnostic calls use the ordinary path. Set `--beacon_decode_graph_full False` to retain only the layer graphs.

On H20 with the BF16 Llama-3.1-8B f4 checkpoint and t8f4, the same two prompts and 63 teacher-forced decode steps give the following medians after one warmup and four measurements. Timing includes forward and argmax.

| Input tokens | Layer graphs (tokens/s) | Whole-model graphs (tokens/s) | Prefill (s) | Peak allocated (GiB) |
| ---: | ---: | ---: | ---: | ---: |
| 4,009 | 54.13 | 82.38 | 0.76 | 22.66 |
| 6,194 | 55.08 | 82.09 | 1.21 | 22.72 |

Peak allocation includes the model, cache, FA workspaces and retained graphs. In this comparison, all 64 logit steps match the original implementation bitwise. Greedy outputs and official scores also match on eight fixed LongBench-v1 samples from Qasper, HotpotQA, MultiFieldQA-en and LCC.

## Optional decoder-tail compilation

`--beacon_decode_compile True` enables the validated Torch 2.11 decoder-tail compiler for B1 BF16 FA2 Graph decode. It retains the original FP32 variance reduction and BF16 rounding boundaries, and reuses the existing CUDA Graphs. Other Torch versions and unsupported shapes retain the original path. The first eligible call compiles the tail before Graph capture.

The CUDA inference CLI and `eval_longbench_slimkv.sh` enable this option by default. Disable it with `--no-beacon_decode_compile` in `main.infer`, `DECODE_COMPILE=False` in the shell preset, or `--beacon_decode_compile False` in `main.eval_longbench`. The Python API and comparison tools preserve their original defaults. Training is unchanged.

On H20 with the same f4 checkpoint, t8f4, B1 and 63 fixed decode steps, four alternating measurements after warmup give 87.76 → 89.48 tokens/s for a 4,009-token prompt and 87.62 → 89.39 tokens/s for a 6,194-token prompt. Compilation is outside the timing interval. All 64 logit steps and final persistent KV tensors match bitwise, and greedy tokens and official scores match on eight fixed LongBench-v1 samples. These controls use the current source; the historical 82 tokens/s result is a separate baseline.

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

`--compare_b1` compares compact full, reconstruction, legacy FA2, GPU bookkeeping and Graph decode in one model. Omit it for the three-path full/reconstruction/FA2 comparison. `--reference_decode_source` separately loads an archived decoder for a same-model control; it cannot be combined with `--compare_b1`.

Timing uses synchronized host wall time and includes forward, argmax, and synchronization boundaries. Tokenization, scoring, file IO, diagnostics and logit saving are outside the timing interval. Warmup includes Triton compilation and Graph capture. Each generated length of 64 contains the prefill first token and 63 decode steps, with no early EOS in the timing phase. All modes receive the same teacher-forced decode inputs; quality is evaluated separately with normal greedy generation. Four measurements per mode (two symmetric rounds) produce medians; retain every raw trial.

Results include `results.json`, predictions and logit tensors, checkpoint config/input/tool hashes, software/GPU metadata and actual kernel coverage. Live/peak CUDA allocation differs from unique persistent KV storage and allocator reserved bytes. Prewarmed Graph workspaces remain live in all comparison modes, so mode-to-mode peak differences are not isolated Graph memory cost.

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

The tool repeats each prompt across independent cache rows, prewarms all batch shapes and retains their Graph workspaces. This measures controlled dense scaling, not independent-request or ragged serving throughput. Use `--rounds` to set the number of paired measurements for each batch and fixture. Same-batch Graph/eager logits are checked on all rows; BF16 across different batch sizes is not asserted to be bitwise identical.

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

The profiler can save actual per-layer Q/K/V cases. `--mode kernel --case_file ...` replays them for NCU; inspect `ncu --help` for your installed version and collect the real main/combined kernel. Profiling durations are diagnostics and must not replace unprofiled benchmark medians. Inspect register spilling, memory traffic, small operations and host launch gaps alongside kernel occupancy.

Independent GPU entries are `tools.validate_decode_ops`, `tools.validate_b1_decode`, and `tools.validate_fa2_decode` (`--batch_sizes` for aligned batches). Record GPU model, dtype, shapes, extension SHA and memcheck logs; CPU tests mock FA2 algebra and do not validate CUDA binaries. No long training or full evaluation is started by CPU CI.
