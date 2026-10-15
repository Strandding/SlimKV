# Latency and memory benchmarking

[Chinese](zh/benchmark.md)
Use the tools below with the current model implementation. Run commands from the repository root.

## Before you start

Follow the [README walkthrough](../README.md#-quick-start) to install FA2, build the SM90 extension and download `Zihant0513/SlimKV-Llama3.1-8B-r4`. Run the commands below in that environment, with `MODEL_NAME_OR_PATH` and `SLIMKV_FA2_DECODE_EXTENSION_DIR` exported. The fixed-context runner generates synthetic tokens; LongBench files are needed only for fixture-based tools.

For a smaller first check:

```bash
CONTEXT_LENGTHS=32768 BEACON_RATIOS=4 REPETITIONS=1 \
OUTPUT_DIR=data/results/speed-smoke \
  bash scripts/benchmark/benchmark_context_decode.sh
```

For the full matrix, use `OUTPUT_DIR=data/results/speed-32k-128k bash scripts/benchmark/benchmark_context_decode.sh`. Each invocation requires a fresh output directory. Both attention GPU kernel time and end-to-end decode time are written locally to `results.json` and `summary.csv`; generated files under `data/results/` are ignored by Git. For experiments and results, see the [paper](https://arxiv.org/abs/2610.02953).

## Preparing small LongBench-v1 fixtures

Supply official local task JSONL files, then:

```bash
python -m tools.data.prepare_longbench \
  --dataset_dir /path/to/longbench \
  --output data/results/fixtures/samples.jsonl \
  --tasks qasper hotpotqa multifieldqa_en lcc \
  --max_samples_per_task 2 --min_length 2500 --max_length 6000
```

The tool selects the first matching rows in original order and adds `task` and `source_index`. It preserves the official row fields and writes source/output hashes in `samples.manifest.json`. The length filter refers to the dataset's `length`, not the tokenized chat prompt. No context is truncated. The selected rows and their source indices are recorded in the manifest.

Performance tools expect a local Llama3 checkpoint and long prompts: the current fixture contract requires more than two Beacon windows and at most 31,550 prompt tokens. They are not arbitrary short-prompt benchmarks. Quality uses task generation caps and stopping rules; timing uses fixed-length decode separately.

## Reconstruction control

```bash
python -m tools.benchmark.benchmark_reconstruction \
  --model_name_or_path "$MODEL_NAME_OR_PATH" \
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
python -m tools.validation.validate_fa2_decode \
  --extension_dir "$SLIMKV_FA2_DECODE_EXTENSION_DIR" \
  --output data/results/fa2-kernel-validation.json
```

The build manifest includes Torch/CUDA, upstream FA2/CUTLASS commits and source/binary SHA256. The extension supports width 128 for raw KV and widths 200–256/320 for static latent decode. Ordinary decode uses standard FA2 at widths ≤256. Larger actual ranks are temporarily padded to 320 for computation. Persistent cache stores actual ranks, with no padding.

## Prefill indexing and rotary fusion

`--beacon_prefill_fast True` is the default for Llama. During inference, unpadded interleaved FA2 windows with a latent cache share token indices across layers and skip raw-cache copies at completed windows. BF16 Q/K rotary operations use one Triton kernel while leaving Beacon K unrotated. Multiplications still round to BF16 before addition, matching the original operations. Projection shapes, weights, ranks, attention scale and masks stay the same. Attention itself runs in the original feature dimensions.

Training, padded inputs and unsupported configurations use the existing path. Unsupported rotary kernels also fall back to the existing rotary operations. For a direct comparison, pass `--beacon_prefill_fast False` to `main.evaluation.eval_longbench`, or set `model.config.beacon_prefill_fast = False`. The first eligible call compiles the Triton kernel.

## B1 decode graphs

With the rebuilt extension, `--beacon_decode_graph True` also enables whole-model graphs for eligible B1 single-token latent FA2 decode, including steps that insert a Beacon. KV buffers use fixed addresses and append in place. Larger batches and diagnostic calls use the ordinary path. Set `--beacon_decode_graph_full False` to retain only the layer graphs.

On Beacon insertion steps, the raw-token and Beacon queries use two single-query sequences with shared KV storage. Their valid lengths differ by one, preserving the raw query's exclusion of the new Beacon. FA2 packs each query's GQA heads for better KV reuse. The split count stays the same as the original two-query call to preserve its softmax reduction order. Persistent ranks and cache contents are unchanged.

## Optional decoder-tail compilation

`--beacon_decode_compile True` enables the validated Torch 2.11 decoder-tail compiler for B1 BF16 FA2 Graph decode. It retains the original FP32 variance reduction and BF16 rounding boundaries, and reuses the existing CUDA Graphs. Other Torch versions and unsupported shapes retain the original path. The first eligible call compiles the tail before Graph capture.

The CUDA inference CLI and `eval_longbench_slimkv.sh` enable this option by default. Disable it with `--no-beacon_decode_compile` in `main.inference.infer`, `DECODE_COMPILE=False` in the shell preset, or `--beacon_decode_compile False` in `main.evaluation.eval_longbench`. The Python API and comparison tools preserve their original defaults. Training is unchanged.

## Current FA2 and Graph comparison

```bash
python -m tools.benchmark.benchmark_fa2_decode \
  --model_name_or_path "$MODEL_NAME_OR_PATH" \
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
python -m tools.benchmark.benchmark_batch_decode \
  --model_name_or_path "$MODEL_NAME_OR_PATH" \
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
  python -m tools.benchmark.profile_fa2 --mode model --phase decode \
    --model_name_or_path "$MODEL_NAME_OR_PATH" \
    --samples_jsonl data/results/fixtures/samples.jsonl \
    --output_dir data/results/nsys-inputs --decode_variant graph --batch_size 1
```

Inspect `python -m tools.benchmark.profile_fa2 --help` for sample and step settings. Add `--gpu-metrics-devices=0 --gpu-metrics-frequency=10000` only on supported devices with permission to collect metrics. Percentages are interpreted in the actual kernel span, not over setup/teardown. Graph node tracing exposes GPU work; missing Python MLP ranges does not mean MLP computation disappeared.

The profiler can save actual per-layer Q/K/V cases. `--mode kernel --case_file ...` replays them for NCU; inspect `ncu --help` for your installed version and collect the real main/combined kernel. Profiling durations are diagnostics and must not replace unprofiled benchmark medians. Inspect register spilling, memory traffic, small operations and host launch gaps alongside kernel occupancy.

Independent GPU entries are `tools.validation.validate_decode_ops`, `tools.validation.validate_b1_decode`, and `tools.validation.validate_fa2_decode` (`--batch_sizes` for aligned batches). Record GPU model, dtype, shapes, extension SHA and memcheck logs; CPU tests mock FA2 algebra and do not validate CUDA binaries. No long training or full evaluation is started by CPU CI.

## Fixed-context decode comparison

Compare Full KV, reconstruction and direct latent attention with the same
checkpoint, BF16, B1 and FlashAttention 2:

```bash
MODEL_NAME_OR_PATH="$MODEL_NAME_OR_PATH" \
  bash scripts/benchmark/benchmark_context_decode.sh
```

The launcher defaults to `EXECUTION_MODE=optimized`: all three methods use
preallocated caches, complete-model CUDA Graphs and the same optional exact
Torch 2.11 decoder-tail compilation. Set `DECODE_COMPILE=False` to disable
compilation for every method. Warmup and graph construction are excluded from
timing. Graph capture inside a measured interval is treated as an error.

The default matrix covers 32K, 64K, 96K and 128K input tokens, both t4f4 and
t8f4, and exactly 128 single-token decode forwards after prefill. K = 1024.
Every method receives identical saved synthetic inputs and the same continuation
inputs from Full KV. EOS does not shorten the run. Prefill is excluded.

The optimized runner specializes to a completed Beacon window followed by at
most one window of continuation. It shares model weights and execution policies
across methods. Each optimized method must match its original implementation's
logits and final persistent KV bitwise, both without capture and under Graph
replay, before timing. These numerical checks do not replace LongBench accuracy
evaluation. The runner is a fixed-context performance tool, not a serving scheduler.

Full KV uses the checkpoint's raw backbone without Beacons, retains every raw
token and applies the checkpoint's RoPE to each new key once. Reconstruction
rebuilds historical Beacon K/V at every decode step. Original-dimensional
reconstruction workspaces are temporary attention storage, not additional
compressed persistent KV. All paths prefill with original-dimensional FA2.
The 128K case appends 128 canonical RoPE positions without changing existing
tables or frequencies.

Two measurements are saved separately:

- **Attention GPU kernel time:** summed real GPU kernel durations inside
  attention, including QKV/output projections, reconstruction, RoPE, cache
  operations and latent mappings. Reconstruction and RoPE are also reported
  separately and already included in the total. Norm, MLP and LM head are excluded.
  A separate uncaptured profiler pass executes the same optimized operations and
  verifies predictions. CUDA launch correlations attribute each GPU operation
  to its CPU scope. GPU annotation ranges are a fallback for older profilers
  and are never counted as execution time. FA2 time must not exceed the enclosing attention total.
- **End-to-end decode time:** synchronized wall time for all 128 model forwards
  and argmax operations. Cache restoration, tokenization, prefill, graph capture,
  compilation and profiler overhead are excluded. Four repetitions follow one
  warmup, with alternating method order. Summaries use median times.

For the original non-Graph implementation comparison, use:

```bash
EXECUTION_MODE=matched MODEL_NAME_OR_PATH="$MODEL_NAME_OR_PATH" \
  bash scripts/benchmark/benchmark_context_decode.sh
```

`matched` disables Graphs and compilation for every method. It compares the
existing model paths: Full KV uses an append cache, while SlimKV uses its normal
non-Graph cache implementation. These paths have different allocation and
launch overheads, so this mode does not isolate algorithmic attention cost.

Outputs default to `data/results/context_decode_<execution_mode>` and require
a new directory. They include `results.json`, `summary.csv`, exact input and
continuation IDs, numerical validation, source/configuration hashes and individual
timings. Set `REFERENCE_DIR` to an earlier result directory to reuse and verify
its exact inputs and continuation. Override `OUTPUT_DIR`, `CONTEXT_LENGTHS`,
`BEACON_RATIOS`, `REPETITIONS` or `PROMPT_FILE` as needed.

Implementation: `tools/benchmark/benchmark_context_decode.py` and
`context_decode_runtime.py`. Use the Transformers 4.40 environment and the
compatible FA2 decode extension. Direct Python invocation defaults to `matched`;
pass `--execution_mode optimized` for the optimized runner.

### Selecting decode token compression

`DECODE_COMPRESS=True` is the default. For prompt-only token compression, run the same launcher with `DECODE_COMPRESS=False` (Python: `--no-beacon_decode_compress`). Prompt token/feature compression stays the same. During decode, SlimKV adds only raw KV and no new Beacons; Full KV is unchanged. The optimized runner also supports continuations longer than one Beacon window when decode compression is disabled. Each method is validated against its original path with the same policy. Results record this flag, and the default output directory gains `_raw_decode` when disabled.
