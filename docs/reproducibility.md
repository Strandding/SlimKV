# Reproducing SlimKV

[中文](zh/reproducibility.md)
The repository supplies runnable stages and local validation evidence. Complete manuscript reproduction is not yet possible from public assets alone. Missing assets and implementations are explicit below.

## Manuscript results versus repository measurements

| Evidence | Protocol / coverage | What it establishes |
|---|---|---|
| Manuscript Tables 2/4 | LongBench across stated backbones and nominal ratios | Author-reported quality; not reproduced by the local eight-sample test |
| Manuscript Table 6 / §5.4 | A100 80GB, B1 BF16 eager, 32K–128K, 128 generated tokens, warmup + three-run median, synthetic Full KV | Paper-reported attention/per-token speedups; different from H20 FA2 controls |
| H20 smoke | Eight LongBench-v1 rows, private Llama f4 checkpoint | Small-sample scores and decoder regression checks |
| H20 timing | Two fixtures, synchronized fixed-input decode, four medians per mode | Local latency/cache/memory comparison for those shapes |
| H20 dense scaling | Replicated prompts, B1–32, partial Graph, aligned lengths | Throughput scaling and same-batch parity; not a serving workload |

The manuscript abstract reports up to 3.38× end-to-end decode speedup; Table 6 contains 3.38× for t4f4 and 3.49× for t8f4 at 128K. README deliberately reports the explicitly named **t4f4** row without resolving this manuscript inconsistency. The authors should reconcile it before publication.

## Stage checklist

| Stage | Available entry | Required author assets / remaining work |
|---|---|---|
| Initialization | `main.convert_kv_lowrank`; conversion presets | Exact per-layer K/V rank lists, initialization checkpoint revision; automatic equal-energy/floor/budget allocator absent |
| Pretraining | `main.train`; Llama/Qwen2 presets | Exact 0.5B-token processed RedPajama manifest, schedule, seeds and training environment |
| Finetuning | Same Trainer; finetuning presets | Exact 0.2B-token mixture and generated GPT/needle data, model-only versus resumable state policy |
| LongBench | `main.eval_longbench` | Final weights for each backbone/compression ratio and full outputs/configs/task sets |
| NIAH | `main.eval_needle` | Three-seed construction, manuscript judge and aggregation; current tool is not equivalent |
| Paper efficiency | Existing prototype and current tools | Audited A100 eager synthetic Full KV/Recon/LSA harness, attention-only timings, paper controls and long-context evidence |
| Multi-request/ablations/MoE | Probes and archived prototypes | Final shared-context and ablation protocols; Qwen3 MoE implementation missing |

## Recording a run

`main.train` and `main.eval_longbench` write `run_manifest.json` alongside outputs. It records seed/configuration, package and hardware versions, commit, actual source hashes, and available checkpoint-config/input hashes. Performance tools additionally save raw synchronized timing, input hashes, warmup/order information and cache/allocation measurements. A dirty worktree cannot be identified by commit alone; preserve the source hash manifest.

For another entry point, use the standard-library provenance helper:

```bash
python -m tools.record_run \
  --output data/results/my-run/run_manifest.json \
  --config /path/to/resolved-config.json \
  --inputs /path/to/checkpoint/config.json /path/to/samples.jsonl
```

The helper does not launch work, reset seeds or modify numerical settings. JSON credential fields are redacted; raw command lines and the complete environment are not captured. Weight files are not hashed automatically. Hash/retain all exact checkpoint shards separately for weight identity. Local filesystem paths can remain in private manifests; review them before releasing results.

Keep data selection/order, chat template, token ratio, window/stride, independent K/V rank lists, RoPE policy, dtype, backend, generation/stopping policy and GPU hardware fixed for a controlled comparison. Retain raw trials and failed-run logs separately, without counting failures as successful benchmark measurements.

## Resources needed for a complete reproducibility release

The paper is public as [arXiv:2610.02953](https://arxiv.org/abs/2610.02953), submitted October 2, 2026. Its arXiv record states acceptance to EMNLP 2026 Main Conference (Oral). Proceedings metadata can be added when available. Remaining resources:

1. Final proceedings citation metadata, when available; the public arXiv citation is already recorded.
2. Explicit license for original SlimKV contributions; upstream notices are already retained.
3. Redistributable SlimKV checkpoints/tokenizers and immutable hashes, plus base-model provenance and terms.
4. Exact training-data revisions, processed/generated mixtures, sample/seed manifests and final configuration for each result.
5. Final rank lists and the missing automatic allocation implementation or reproducible derivation.
6. Complete evaluation outputs and the paper-specific efficiency/judge/ablation harnesses.

Figure 3 was extracted from the supplied manuscript as a vector illustration. It documents the paper design; it is not a functionality guarantee. Historical local validation notes remain available [in English](archive/validation_20261014.md) and [Chinese](archive/validation_20261014_zh.md). They contain machine-specific evidence locations and historical support statements; use [implementation status](implementation_status.md) for the current contract.
