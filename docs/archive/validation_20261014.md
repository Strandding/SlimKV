# Historical validation log (2026-10-14)

[中文原始记录](validation_20261014_zh.md)

English companion to the original internal record. Each section describes the implementation and environment **at that stage**. Batch restrictions, test counts, machine paths and future-work statements are historical, not the current support contract. Use [implementation status](../implementation_status.md) and [benchmark](../benchmark.md) for current behavior. Raw weights, logs and large artifacts remain local.

## Initial implementation and execution paths

The main implementation provides progressive Beacon compression, Llama3/Qwen2 low-rank K/V projections, layer-wise rank lists, SVD initialization, frozen-backbone training and long-context evaluations. Compatible Llama3 inference stores latent Beacon K/V and defaults to reconstruction; an optional latent FA2 decode path was subsequently validated on H20 f4 weights. Multi-model quality and the full paper efficiency protocol remain incomplete. `slimkv/latency` is a separate prototype.

| Historical path | Beacon cache | Computation | Model |
|---|---|---|---|
| Compatible Llama3 reconstruction | Latent K/V; full raw/sink | Reconstructed standard attention | Llama3 |
| Llama3 + fa2 decode | Global latent K/V; full raw/sink | Original-dimension prefill; two FA2 decode branches merged by LSE | Llama3, H20 BF16 |
| Llama3 training/full/incompatible auto; Qwen2 | Full K/V | Standard attention | Llama3/Qwen2 |
| Latency prototype | Global latent K/V | Joint raw/Beacon logits and softmax | Llama eager prototype |

`beacon_*` fields and weight keys remain compatible. Headwise/grouped up-projection is available in some ordinary experiments but cannot be assumed compatible with the hybrid loader. `auto` selects latent storage only for eligible inference; `full` preserves full caching; explicit `latent` rejects incompatible settings. Training retains full-cache mathematics and gradients. Both historical and incomplete-window Beacon states use latent storage; exported cache references do not retain reconstructed full K/V.

Eligibility at this stage: interleave, window=stride, one-window encoding, positive token ratio, low-rank K and V; no chunk-fixed K RoPE, K residual or beacon_skip. Qwen2 remains full-cache. Fourfold average feature compression applies to Beacon K/V, not raw/sink, temporary workspaces or total GPU memory.

## Historical Beacon K RoPE correction

Llama's ordinary path was corrected when no new Beacon was inserted. CPU regressions cover raw-only calls, new Beacon insertion, the final training window, prefill tails, decode and chunk-fixed historical memory. BF16 generation was checked on a small subset; full quality evaluation was not performed.

On H20, separate source snapshots `eeaca5e` (before) and `c8c8935` (after) used the same f4 `checkpoint-2000`, BF16 eager, Llama-3 chat template, greedy generation, token ratio 8 and window=stride 1024. Memory reset before every request; weights and ranks were unchanged. For each of Qasper/HotpotQA, select the first two official LongBench-v1 rows with dataset length 2500–6000, without context truncation. Generation caps were 128/32 with the existing QA newline stop rule.

| Task / original index | Input tokens | Before F1 | After F1 |
|---|---:|---:|---:|
| Qasper / 0 | 4009 | 34.48 | 31.25 |
| Qasper / 2 | 4370 | 12.50 | 12.50 |
| HotpotQA / 16 | 6194 | 100.00 | 100.00 |
| HotpotQA / 22 | 5145 | 100.00 | 100.00 |

Mean F1 was 61.75→60.94. Three outputs had identical tokens; one changed wording. This did not demonstrate a quality improvement or full benchmark quality. Prefill first logits were identical on all four prompts. All 32 layers reached the no-new-Beacon decode branch; at layers 0/31 the old rotated-history difference was 3.65625–11.96875, versus 0 after the correction. The cache's last dimension remained 128, confirming reconstructed/full-dimensional storage at this stage. No-new-Beacon prefill tails were covered by CPU tests, not these prompts.

Artifacts: H20 `data/results/longbench_v1_smoke_20261014/`; local `/data/workspace/slimkv_analysis/longbench_v1_smoke_20261014/`. The run exited normally and restored the original resource-holder process. No efficiency conclusion was drawn.

## Latent storage with reconstruction

Ten new real-model CPU tests plus eight RoPE tests passed (18 total) locally and on H20. Coverage: per-layer K/V ranks, global/headwise/grouped up-projection, shared down-projection, bias, eager/SDPA, incomplete-window Beacon states, short prefill/rollover/decode, storage release/reset/generation, checkpoint compatibility, native/full fallback, and bitwise training output/loss/gradient parity with the original Memory.

The same four BF16 H20 samples were run with full and latent caches on one model. All generated tokens and the saved 50×128256 logits matched bitwise (max difference 0); full-cache outputs also matched the previous records. This is a shape/environment-specific result, not a universal bitwise guarantee.

| Task / index | Full persistent K/V MiB | Latent MiB | Byte reduction |
|---|---:|---:|---:|
| Qasper / 0 | 184.000 | 136.750 | 25.68% |
| Qasper / 2 | 104.000 | 52.719 | 49.31% |
| HotpotQA / 16 | 103.625 | 31.062 | 70.02% |
| HotpotQA / 22 | 83.875 | 23.594 | 71.87% |

These are end-of-generation logical K/V bytes, including sink/raw/history/window Beacon, not GPU peak allocation. Beacon features alone shrink exactly fourfold. Unique underlying latent storage equals logical bytes, confirming no full-dimensional backing views remain. Returned caches reference persistent representations only; attention still uses dimension 128. Historical Beacon K stayed unrotated at checked layers 0/31.

Evidence: H20 `data/results/latent_cache_validation_20261014/`, local `/data/workspace/slimkv_analysis/latent_cache_validation_20261014/` (original logits, 18-file SHA receipt and independent audit). Eager GPU was validated; SDPA was CPU-only. LSA and performance benchmarking were outside this stage. The GPU run exited normally and restored occupancy.

## Reconstruction latency, actual memory and eight-sample quality

`tools/benchmark_reconstruction.py` used H20, the same f4 BF16 model, eager attention and token ratio 8. Reset each call; warm all three modes, then two symmetric rounds: full → full_compact → latent → latent → full_compact → full. Four trials per mode, median reported. Generate 64 tokens including the prefill first token and 63 decode steps, without early EOS for timing. Tokenization, scoring, file IO and logits saving are excluded.

The ordinary full cache's sink slice retains the complete first-window backing storage. The harness-only compact control clones the sink to release it, preserving values and attention mathematics. All modes generated identical tokens in warmup and timing. Two CPU harness regressions passed (20 total); the same controls passed on H20. No model algorithm change was made for this measurement.

| Input tokens | Compact full prefill ms | Latent prefill ms | Compact full decode ms/token | Latent decode |
|---|---:|---:|---:|---:|
| 4009 | 852.75 | 873.86 | 27.67 | 43.26 |
| 6194 | 1373.87 | 1406.46 | 27.50 | 42.55 |

Reconstruction prefill was 2.48%/2.37% slower, decode 56.30%/54.75% slower. Decode throughput fell from 36.13/36.36 to 23.12/23.50 tok/s. This backend saved storage but did not accelerate decode.

| Tokens | Full / compact / latent persistent storage after prefill MiB | Compact / latent peak allocated GiB |
|---|---|---|
| 4009 | 323.750 / 179.875 / 132.906 | 21.883 / 21.836 |
| 6194 | 247.000 / 103.125 / 30.563 | 21.992 / 21.850 |

Persistent bytes deduplicate actual Memory backing storage. Peak allocation also includes weights/buffers/logits/workspaces. Peak savings versus compact full were only 48.125 MiB (0.215%)/145.496 MiB (0.646%). Fourfold Beacon-feature compression is not fourfold GPU peak-memory reduction; allocator reserved bytes were excluded.

Quality expanded to two examples each from Qasper, HotpotQA, MultiFieldQA-en and LCC: first rows of dataset length 2500–6000; actual prompts 4009–11437 tokens; no truncation. Official task templates/caps, existing Llama-3 template and QA newline stop applied. LCC had no chat template or QA newline EOS. QA used F1 and LCC similarity of the first valid code line, independently checked against official metrics.

| Task (two examples) | Full | Latent |
|---|---:|---:|
| Qasper | 21.88 | 21.88 |
| HotpotQA | 100.00 | 100.00 |
| MultiFieldQA-en | 40.28 | 40.28 |
| LCC | 75.00 | 75.00 |

All eight outputs matched. These are length-selected smoke samples, not full LongBench-v1 scores. Raw timing/memory/predictions/input/source SHA, a 14-file receipt and audit were saved locally at `/data/workspace/slimkv_analysis/longbench_perf_20261014/`, corresponding to H20 `data/results/longbench_perf_20261014/`. Normal exit restored occupancy.

## Initial latent FA2 decode: two iterations

`beacon_decode_backend=fa2` was added without changing the default reconstruct path. Raw/sink use original 128-dimensional FA2. Rotated Queries project through GQA-indexed head slices of U_K; global-latent PV projects through U_V slices. LSE merges both branches into one softmax with scale `1/sqrt(128)`. New Beacon states are down-projected directly into cache; historical/current Beacon K/V are not reconstructed in this decode path. Prefill/training retain the original path.

The checkpoint's global maximum K/V ranks are 304/312. FA2 2.8.3 serves widths ≤256; the local extension builds BF16 SM90 width-320 forward split-KV/combine only from FA2/CUTLASS. K/V temporarily pad to a common compute width; cache is `[B,1,N,R]` at actual rank. This environment used Torch 2.11.0+cu129 and Transformers 4.40.2 with compatible reused FA2 binaries; earlier Torch 2.13 results are separate.

Eligibility at this historical stage: Llama, global Linear up-projection, full-coverage, attend_prev, interleave, window=stride, positive ratio; Beacon K has no RoPE, raw K has RoPE; no chunk-fixed K, K residual or latent position bias. GPU tests used BF16, unpadded B1, one raw token per step (Q2 on Beacon insertion). Raw attends all raw positions; latent uses bottom-right causal masking. Fully masked FA2 rows' +inf LSE is converted to zero probability mass before merging. Training/batches/multiple-raw-token calls used reconstruction at that stage.

Iteration 1 connected both branches; iteration 2 removed per-layer `max().item()` RoPE CUDA synchronization and unnecessary tensor copies. Twenty-five H20 CPU model tests passed, including FP32 global/GQA/bias/shared-down, short prefill, first Beacon, rollover/causality and default training. Width-320 had 148 GPU cases (Q1/Q2, causal/noncausal, B1/B2, GQA, length boundaries, single/auto/16 splits) versus independent FP32, max output error 0.002335; four memcheck cases reported 0 errors. Local Torch was unavailable then, so model tests ran on H20; local work audited source/data/scores.

Timing used one model and full_compact/reconstruct/fa2, each warmed once; two symmetric rounds gave four medians per mode. The full_compact warmup supplied identical fixed decode inputs; first token was prefill, then 63 steps. Diagnostics and IO were outside synchronized timing.

| Tokens | Compact full decode | Reconstruction | FA2 iteration 1 / 2 (ms/token) |
|---|---:|---:|---:|
| 4009 | 26.572 | 41.534 | 32.048 / 29.770 |
| 6194 | 26.405 | 41.356 | 31.929 / 29.426 |

Iteration 2 was 28.32%/28.85% faster than reconstruction but 12.03%/11.44% slower than compact full. Iteration 1 used 31 steps/two trials per mode; iteration 2 used 63/four. Cross-run figures indicate iteration history; formal speedups use same-run controls.

Prefill medians (compact/reconstruct/fa2) were 851.60/869.29/865.48 ms and 1368.01/1401.62/1392.87 ms. Latent modes' persistent storage was 132.906/30.563 MiB, versus compact full 179.875/103.125 MiB. Beacon features remained exactly 4× smaller; total call peaks matched between latent modes and were prefill-dominated.

The same eight quality samples scored Qasper21.88/Hotpot100/MultiField40.28/LCC75 in reconstruction and FA2. Seven token sequences matched; LCC22 diverged from generated token 52 but retained the first code line/score. FA2 iterations matched all eight outputs. Neither full quality nor BF16 bitwise equivalence follows.

Each fixture saved prefill+16 teacher-forced steps, all 128256 logits. Prefill max difference was 0; decode max/mean differences were 0.21875/0.02144 and 0.53125/0.03905, with all 34 argmaxes equal. Each validation recorded 1024 FA2 calls, including 288 width-320 calls across Q1/Q2.

Evidence: H20 `data/results/fa2_decode_20261014/`, immutable source_v1/v1b/v2, build commands/manifests, CPU/GPU/sanitizer logs, predictions, timing/memory and actual logits. Local `/data/workspace/slimkv_analysis/fa2_decode_20261014/` stores receipts/audits; logits stayed on H20 and were actually read by the independent tensor audit. Both GPU suites exited normally and occupancy restored.

## FA2 prefill, Nsys/NCU and fused decoder

Same H20 f4 BF16, Torch2.11+cu129, FA2 2.8.3 and Triton3.6. Standard FA2 prefill uses dimension128; decode is global latent. Eager/FA2 controls share immutable source, prompt and preselected fixed decoder inputs. Warm each mode, then symmetric two rounds/four medians; quality is independent greedy.

| Tokens | Eager prefill ms | FA2 prefill ms | Reduction |
|---|---:|---:|---:|
| 4009 | 867.59 | 736.25 | 15.14% |
| 6194 | 1398.39 | 1168.39 | 16.45% |

Separate Nsys2025.3.1 and NCU2025.2.1 diagnostics are not formal timing. CUDA API correlation and nested NVTX identify all model kernels. Baseline 16-step decode: 41,292 kernels, about172.08ms summed GPU time over757.58ms span. Raw/latent FA2 (including combine) totaled4.26/9.87ms; host launches, positional copies and small operations were additional opportunities. This is not an unprofiled30ms/token breakdown.

Softmax merge launched11 kernels/layer (5632 total). Per-layer GPU-position scalar creation copied8 bytes (512 copies), with636 stream synchronizations across the trace. NCU replayed real layers2/28: width320 Q1 had13 blocks/~11.3µs; Q2 had96 blocks/~58.2–58.8µs. Both used255 registers, shared memory limited residency to1 block/SM, and spilling occurred. Raw128 was~9.28µs. All five replay output/LSE comparisons were exact; durations are diagnostic only.

Two fused operators were implemented in `decode_kernels.py`: Q/raw-K RoPE, and logaddexp/branch weights/output merge. BF16 rounding after each multiplication/addition is preserved; merge uses exact libdevice exp/log1p without FMA. Existing GPU positions eliminate CPU→GPU scalar copies. CUDA BF16/FP16 + Triton uses fusion; CPU, gradients or absent Triton use Torch. Prefill/training/global rank/width320 binary/weights were unchanged.

Final same-model controls switch only the decoder function, using identical weights/FA2 prefill/fixed inputs. Compact full → reconstruct → reference FA2 → fused FA2, then reverse, twice: four trials/mode/fixture,32 total, prefill first token+63 steps.

| Tokens | Compact full FA2 | Latent reconstruction | Reference latent FA2 | Fused FA2 (ms/token) |
|---|---:|---:|---:|---:|
| 4009 | 25.903 | 40.987 | 29.893 | 24.246 |
| 6194 | 25.845 | 40.941 | 29.307 | 23.857 |

Fused decode reduced reference time18.89%/18.60%, compact full6.40%/7.69%. Only these two B1 BF16 fixed-length inputs were measured. Optimized Nsys had25,356 kernels (−38.59%); merge5632→512 (11→1/layer); position copies512→0; stream sync636→124. GPU time139.99ms, span580.19ms; MLP/raw/latent FA2 time largely unchanged. Diagnostic reductions do not establish model-wide gains for every shape.

Twenty-five H20 CPU tests and48 GPU fused cases passed (B1/B2,Q1/Q2,raw lengths,sink0/1,BF16/FP16,strides,fully masked latent). RoPE/merge matched Torch bitwise; four memcheck cases had0 errors. Real17×128256 logits per fixture matched before/after and same-model switching bitwise on actual CPU tensor reads.

All eight FA2 quality outputs were unchanged by fusion. Eager-vs-FA2 prefill matched7/8; LCC22 changed late tokens but not first-line score. FA2-prefill reconstruction Qasper scored23.49; different floating-point paths need not match.

Remaining opportunities at this stage: host launches, dynamic caches, preallocation/Graph, width320Q2 tiling/GQA/shared/spill. Prefill MLP accounted for~57% of summed kernel time; redundant Beacon projection/reconstruction also merited study. These directions were not yet implemented or promised as speedups in this stage.

H20 `data/results/fa2_profile_20261014/` retains raw Nsys/SQLite, five NCU reports/metrics, real kernel inputs/logits, source_baseline/source_optimized/source_abba and predictions. Local `/data/workspace/slimkv_analysis/fa2_profile_20261014/` holds small-file receipts and independent timing/score/tensor/profile audits. Large tensors/reports were initially remote; later Nsys transfers are separately recorded. Suites exited normally and occupancy restored.

## B1 bookkeeping and partial CUDA Graph

Fast decode (`beacon_decode_fast=true`) preserves GPU token histories, uses the known insertion plan for role/raw indices, preallocates positions/role/index buffers and fixed Q2 embedding slices. Token host round-trips/nonzero are removed. Padding checks remain; prefill/training/multiple tokens/attention diagnostics use ordinary paths. KV retains actual rank; dynamic KV cat/allocation is not fully preallocated.

Optional `beacon_decode_graph=true` captures per-layer input norm and post-attention residual/postnorm/MLP/residual as two separate segments for Q1/Q2. Attention/cache update remain eager. CPU/gradient fall back; full hidden/attention diagnostics disable Graph; dtype/device changes clear caches. Graph is off by default and is not the complete generation-loop graph.

H20 f4 BF16/FA2 2.8.3/Torch2.11, standard FA2 prefill, same fixed inputs. Five modes forward/reverse, two rounds, four medians/mode,40 measurements total; each64 tokens=1prefill+63decode. Capture is in warmup, not timing.

| Tokens | Legacy fused FA2 | GPU bookkeeping | Partial Graph (ms/token) | Reduction vs legacy |
|---|---:|---:|---:|---:|
| 4009 | 24.772 | 24.263 | 18.381 | 25.80% |
| 6194 | 24.671 | 24.227 | 18.671 | 24.32% |

Bookkeeping alone saved1.80%–2.06%; Graph accounts for most improvement. Compact full was26.343/26.397ms/token; Graph was30.22%/29.27% faster. The4009 fixture's late-second-round variation remains in raw records; all four predetermined trials contribute without outlier removal. No broader shape/checkpoint guarantee.

First Graph warmup added540804096 live bytes (~515.75MiB) above warmed fast decode; KV bytes unchanged. All formal modes retain the same graphs, so their peaks do not isolate Graph overhead. Startup, shared workspaces and persistent-KV preallocation remain opportunities.

Separate16-step Nsys2025.3.1 trace: cuda,nvtx only, no CPU sampling/context switches, graph node tracing, no additional OSRT/memory trace. Same options in both arms:

- Independent kernel-launch API calls25,356→12,889 plus1024 Graph replays (2 per layer/step); actual GPU kernels25,356→25,177. Submission changed much more than device work.
- Kernel-interval union139.79→135.74ms; kernel span574.10→434.59ms; busy fraction24.35%→31.23%. This is diagnostic GPU busy time, not SM utilization or formal ms/token; host gaps remain.
- Stream sync124→20;8-byteD2H32→0/H2D32→0. Padding check and Graph device copies remain. All kernels correlate to APIs/NVTX. Graph does not reexecute Python MLP wrappers; missing MLP range does not mean missing MLP computation.

Twenty-eight H20 CPU regressions passed. FP32/BF16/FP16 each had42 actual model calls comparing legacy/fast/Graph bitwise; coverage includes insertion/rollover/multi-token resume/reset/diagnostics/dtype clearing/gradient fallback. BF16 quick memcheck had0 errors. Saved17×128256 real logits per fixture matched all FA2 paths bitwise (max0). Eight quality tokens and scores matched previous FA2 records; reconstruct Qasper23.49 retained.

Early source_v1 failed due to an incorrect low-precision LSE validator (FA2 uses FP32); source_v1b launcher referenced nonexistent build_v1b rather than reused build_v1. Both failed normally, preserving logs/source, and are not performance evidence. Final source_v2 exited0. The width320 binary/weights were unchanged and occupancy restored.

Artifacts: H20 `data/results/b1_decode_20261014/` contains reports/SQLite/PT/timing/predictions/source manifests. Local `/data/workspace/slimkv_analysis/b1_decode_20261014/` contains independent audits/receipts and later raw Nsys copies in `nsys_reports/`.

## Outstanding issues and missing coverage at that time

Ordinary Qwen can rotate historical Beacon K when `beacon_size<=0`; the hybrid FA2→eager fallback can lose a causal mask; the prototype benchmark accumulates Memory across repeats. Prototype reconstruct keeps full-cache states rather than paper latent-cache reconstruction. Prototype biased Value up-projection lacks the Beacon probability-mass factor (validated Llama weights have no such bias).

Automatic equal-energy threshold/floor/residual-budget allocation was absent; explicit lists were required. Qwen latent attention/Qwen3 MoE, complete LongBench/NIAH/multiple-model/batch/long-context quality, three-seed judge/character reconstruction/attention similarity/shared-context experiments, paper attention-only/synthetic-Full-KV/Recon/median protocol and SlimKV vLLM/paged/continuous batching were incomplete. Subsequent dense batch work does not eliminate these full-paper gaps.

H20 kept a Llama3.1-8B layer-adaptive f4 `checkpoint-2000` under `model/`:32-layer K/V lists averaging32 per head, original head dimension128, BF16, about621M Beacon parameters. Compatible ordinary inference uses latent storage with original-dimension reconstruction. Weight/config files, manuscript, historical predictions and figures stay local; their presence alone does not prove final-paper provenance.
