# Implementation status

[中文](zh/implementation_status.md)
This is a research implementation with tested subsets of the manuscript design. Existing `beacon_*` configuration and weight keys remain the checkpoint API. The original model/prototype code has not been mathematically rewritten by the repository refresh.

## Execution paths

| Path | Persistent Beacon K/V | Attention computation | Validation / scope |
|---|---|---|---|
| Llama3, compatible `auto`/`latent`, default `reconstruct` backend | Actual low-rank features | Original-dimension prefill and reconstructed decode | CPU global/headwise/grouped/bias/SDPA regressions; local BF16 generation |
| Llama3, `latent` + `fa2` decode | Global low-rank features | Original-dimension prefill; raw/sink FA2 + latent Beacon FA2 with one merged softmax in decode | H20 f4 BF16 smoke/scaling; equal-length unpadded aligned batches |
| Llama3 training, `full`, incompatible `auto`; Qwen2 | Full-dimensional K/V | Standard attention | Low-rank training code and conversion available; no Qwen latent-cache backend |
| `slimkv/latency` | Prototype-specific latent K/V | Independent eager hybrid attention | Separate loader; not the default production/model API |
| `experiments/anchor_prototype` | Historical `anchor_*` representation | Archived monkey-patch implementation | Excluded from package discovery; not compatible with Beacon checkpoints |

`auto` stores latent K/V only for supported inference configurations; `full` forces the full cache; explicit `latent` rejects incompatible configurations. Training retains the existing full-dimensional computation/cache and gradients. Raw-token tail and sink K/V stay full-dimensional in every latent path. Temporary width padding for FA2 is not persistent cache padding.

## Latent cache and FA2 requirements

Latent cache: interleave insertion, window=stride, one-window encoding, positive token ratio, low-rank K and V; no chunk-fixed Beacon K RoPE, K positional residual, or beacon_skip. FA2 additionally requires full-coverage, attend_prev, global Linear up-projection, no Beacon K RoPE, raw K RoPE, and no latent relative-position bias. Inspect `LlamaMemory` and `fa2_decode.py` for eligibility checks.

Eligible FA2 decode accepts one new raw token per sequence, with one additional Beacon query on insertion steps. Equal-length, unpadded dense batches share the insertion schedule but have independent token/KV rows. Multi-raw-token calls use the reconstruction path. Padding is explicitly rejected for latent FA2 decode. This is not variable-length continuous batching.

Ranks are global and coupled across KV heads: cached shapes are `[B,1,N,R]`. The private f4 checkpoint's maximum K/V ranks are 304/312; the custom BF16 SM90 extension pads computation to 320. Raw attention uses head dimension 128 and the scale remains `1/sqrt(128)`. K/V rank lists are independent.

`beacon_decode_fast=True` is the default for eligible FA2 decode and retains GPU token bookkeeping. `beacon_decode_graph=False` is the default; opt-in partial Graph captures input norm and post-attention norm/MLP/residual segments, separately for Q1/Q2 and batch shapes. Attention and dynamic KV updates remain eager. CPU/gradient and diagnostics fall back; dtype/device changes clear Graph caches.

## Evidence and its limits

Existing CPU regressions cover historical Beacon K RoPE, low-rank storage, full/latent parity, training gradients, checkpoint reload, global latent/GQA algebra, causal insertion, rollover, distinct batch rows, and fallback behavior. GPU records cover real H20 BF16 f4 inference, fused operators, Graph/eager parity, width-320 reference tests, and memcheck. [Benchmark guide](benchmark.md) summarizes measured results and protocols; [historical log](archive/validation_20261014.md) preserves stage-specific evidence and counts.

Eight LongBench-v1 examples and two timing fixtures are smoke/scaling evidence. They do not establish full LongBench quality, 128K behavior, other checkpoint/architecture compatibility, cross-batch BF16 bitwise equivalence, or production readiness.

## Known issues retained for follow-up

- Ordinary Qwen decode can apply RoPE to historical Beacon K when no new Beacon is inserted, despite the disabled Beacon K RoPE setting. Llama's corresponding path has a regression-tested fix.
- The independent latency prototype's FA2 fallback can lose a causal mask; use its eager backend.
- The historical latency benchmark does not reset Memory between warmup/repeats. Its `beacon_lowrank_reconstruct` uses full-dimensional caching and is not the paper's latent-cache reconstruction baseline. Use the tools-based benchmark for current comparisons.
- Prototype Value up-projection bias lacks a Beacon probability-mass factor. The validated Llama f4 checkpoint has no such bias.
- FA2 eligibility, temporary padding, dynamic KV concatenation, Graph startup/workspaces and host launch gaps constrain performance and supported configurations.

## Missing paper coverage

| Manuscript component / experiment | Current gap |
|---|---|
| Floor-constrained equal-energy rank allocation and residual budget | Explicit lists only; automatic allocator absent |
| Qwen2.5 and Qwen3 MoE latent inference | Qwen2 full cache only; Qwen3 MoE unsupported |
| Full LongBench at all ratios / backbones | Entrypoint present; final checkpoints, manifests, complete reruns missing |
| Three-seed NIAH with manuscript judge | Current judge/aggregation differ; final protocol not integrated |
| Character reconstruction, attention similarity, shared-context multi-request | Probes/prototypes exist; paper protocol not complete |
| A100 eager synthetic Full KV, K-RoPE reconstruction control, attention-only timings | Not implemented as one audited paper benchmark |
| vLLM / paged KV / continuous batching | Baseline wrappers only; no SlimKV serving integration |

See [reproducibility requirements](reproducibility.md) before treating any command as complete paper reproduction.
