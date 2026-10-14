# Reproducing SlimKV

[中文](zh/reproducibility.md)
Use the entry points below to initialize, train and evaluate SlimKV. Complete reproduction also requires the final checkpoints, processed data manifests and per-result configurations.

The paper's efficiency protocol uses Llama-3.1-8B on an A100 80GB with batch size 1, BF16 eager execution, 32K–128K contexts and 128 generated tokens. It reports the median of three runs after warmup and uses synthetic Full KV tensors. The FA2 tools in this repository provide a separate inference benchmark; keep the backend and protocol with each reported result.

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

See the [paper](https://arxiv.org/abs/2610.02953) for experimental settings. The following assets are needed for a complete reproduction release:

1. Explicit license for original SlimKV contributions; upstream notices are already retained.
2. Redistributable SlimKV checkpoints/tokenizers and immutable hashes, plus base-model provenance and terms.
3. Exact training-data revisions, processed/generated mixtures, sample/seed manifests and final configuration for each result.
4. Final rank lists and the missing automatic allocation implementation or reproducible derivation.
5. Complete evaluation outputs and the paper-specific efficiency/judge/ablation harnesses.

See [backend support](implementation_status.md) for cache eligibility and remaining implementation gaps.
