# Backend support

[Chinese](zh/implementation_status.md)
The table below describes the available cache and attention paths. Existing `beacon_*` configuration and weight keys remain the checkpoint API.

## Execution paths

| Path | Persistent Beacon K/V | Attention computation | Validation / scope |
|---|---|---|---|
| Llama3, compatible `auto`/`latent`, default `reconstruct` backend | Actual low-rank features | Original-dimension prefill and reconstructed decode | CPU global/headwise/grouped/bias regressions; local BF16 generation |
| Llama3, `latent` + `fa2` decode | Global low-rank features | Original-dimension prefill; raw/sink FA2 + latent Beacon FA2 with one merged softmax in decode | BF16; equal-length unpadded aligned batches |
| Llama3, latent cache + `latent` decode strategy | Global low-rank features | Original-dimension prefill; direct Beacon decode uses eager or FA2 according to `attn_impl` | Eager works on CPU/GPU, including unequal K/V ranks; same aligned-batch requirements |
| Llama3 training, `full`, incompatible `auto`; Qwen2 | Full-dimensional K/V | Standard attention | Low-rank training code and conversion available; no Qwen latent-cache backend |

`auto` stores latent K/V only for supported inference configurations; `full` forces the full cache; explicit `latent` rejects incompatible configurations. Training retains the existing full-dimensional computation/cache and gradients. Raw-token tail and sink K/V stay full-dimensional in every latent path. Cache views expose actual ranks. Static decode shares their backing storage with padded FA2 buffers, whose padding and reserved capacity count toward physical GPU memory.

## Latent cache and direct decode requirements

Latent cache: interleave insertion, window=stride, one-window encoding, positive token ratio, low-rank K and V; no chunk-fixed Beacon K RoPE, K positional residual, or beacon_skip. Both direct eager and FA2 decode additionally require full-coverage, attend_prev, global Linear up-projection, no Beacon K RoPE, raw K RoPE, and no latent relative-position bias. Inspect `LlamaMemory`, `latent_decode.py` and `fa2_decode.py` for eligibility checks.

Eligible direct latent decode accepts one new raw token per sequence, with one additional Beacon query on insertion steps. Equal-length, unpadded dense batches share the insertion schedule but have independent token/KV rows. Multi-raw-token calls use the reconstruction path. Padding is explicitly rejected for direct latent decode. This is not variable-length continuous batching.

Ranks are global and coupled across KV heads: cached shapes are `[B,1,N,R]`. FA2 ranks above 256 use the custom BF16 SM90 extension, with width-320 FA2 buffers. Eager computes directly at the actual K/V ranks without padding or a width-320 limit. Raw attention uses the original head dimension and scale; for Llama-3.1-8B these are 128 and `1/sqrt(128)`. K/V rank lists are independent.

`beacon_decode_fast=True` retains GPU token bookkeeping for eligible FA2 decode. The Python API defaults to `beacon_decode_graph=False`; CUDA inference and LongBench presets enable whole-model B1 Graphs with append-only KV, including Beacon insertion steps. Real-rank Beacon views share the padded FA2 buffers. Setting `beacon_decode_graph_full=False` keeps partial norm/MLP Graphs. Unsupported calls use the ordinary path, and dtype/device changes clear Graph caches.

Floor-constrained equal-energy rank allocation and residual budget completion are available in `main.conversion.convert_kv_lowrank`. K/V budgets are independent and ranks align to the KV head count. See [automatic initialization](training.md#automatic-equal-energy-allocation).

## Known issues retained for follow-up

- Ordinary Qwen decode can apply RoPE to historical Beacon K when no new Beacon is inserted, despite the disabled Beacon K RoPE setting. Llama's corresponding path has a regression-tested fix.
- The independent latency prototype's FA2 fallback can lose a causal mask; use its eager backend.
- The historical latency benchmark does not reset Memory between warmup/repeats. Its `beacon_lowrank_reconstruct` uses full-dimensional caching and is not the paper's latent-cache reconstruction baseline. Use the tools-based benchmark for current comparisons.
- Prototype Value up-projection bias lacks a Beacon probability-mass factor. Use bias-free Beacon Value up-projections for that prototype.
- FA2 eligibility, temporary padding, dynamic KV concatenation, Graph startup/workspaces and host launch gaps constrain performance and supported configurations.

## Missing paper coverage

| Manuscript component / experiment | Current gap |
|---|---|
| Qwen2.5 and Qwen3 MoE latent inference | Qwen2 full cache only; Qwen3 MoE unsupported |
| Full LongBench at all ratios / backbones | Entrypoint present; final checkpoints, manifests, complete reruns missing |
| Three-seed NIAH with manuscript judge | Current judge/aggregation differ; final protocol not integrated |
| Character reconstruction, attention similarity, shared-context multi-request | Probes/prototypes exist; paper protocol not complete |
| A100 eager synthetic Full KV, K-RoPE reconstruction control, attention-only timings | Not implemented as one audited paper benchmark |
| vLLM / paged KV / continuous batching | Baseline wrappers only; no SlimKV serving integration |

See [reproducibility requirements](reproducibility.md) before treating any command as complete paper reproduction.
