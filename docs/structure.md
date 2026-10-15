# Repository structure

[Chinese](zh/structure.md)

Start with `scripts/` to run an experiment, `main/` to inspect its Python entry
point, and `slimkv/` to inspect the model. Entry points and tools are grouped by
purpose; directory names are stable and contain no experiment timestamps.

```text
slimkv/                 Model and cache implementation
  llama3/               Llama training, prefill, latent and reconstruction decode
  qwen2/                Qwen2/Qwen2.5 model implementation
  vllm_plugin/          Experimental paged cache and continuous scheduling
main/                   Python application entry points
  conversion/           SVD initialization and projection conversion
  training/             Training and pretraining data processing
  inference/            Standalone and vLLM inference
  evaluation/           LongBench, NIAH and other accuracy evaluations
  analysis/             Beacon probes, attention heatmaps and plotting
scripts/                Shell launch presets grouped by task
  conversion/
  training/
  evaluation/
  benchmark/
  slurm/                Scheduler templates
  vllm/                 Separate-environment baseline utilities
tools/                 Development and measurement tools
  benchmark/            Decode timing, batch scaling and FA2 profiling
  data/                 LongBench download and deterministic fixture preparation
  validation/           GPU numerical and communication checks
config/
  data/                 Training data mixtures
  accelerate/           FSDP and inference launcher configurations
  deepspeed/            DeepSpeed training configurations
  environments/         CPU dependency constraints
kernels/fa2_decode/     Optional SM90 width-320 FA2 extension
integrations/vllm/      Separate vLLM installation profile
tests/                  Automated regression tests
docs/                   English guides, zh/ translations and license texts
```

## Common entry points

| Task | Launcher or module |
|---|---|
| Initialize low-rank Beacon weights | `scripts/conversion/convert_kv_lowrank.sh` |
| Train / finetune Llama | `scripts/training/train_llama_8B.sh` / `finetune_llama_8B.sh` |
| Run inference | `python -m main.inference.infer` |
| Run vLLM inference | `python -m main.inference.infer_vllm` |
| Evaluate LongBench-v1 | `scripts/evaluation/eval_longbench_slimkv.sh` |
| Inspect LongBench preprocessing and scoring | `main/evaluation/eval_longbench.py`, `longbench_utils.py` |
| Compare Full KV, reconstruction and latent at 32K–128K | `scripts/benchmark/benchmark_context_decode.sh` |
| Inspect the fixed-context benchmark | `tools/benchmark/benchmark_context_decode.py` |
| Profile FA2 | `python -m tools.benchmark.profile_fa2` |
| Build the optional FA2 extension | `python -m tools.build_fa2_decode` |
| Check repository syntax, imports and links | `python tools/check_repository.py` |
| Record run configuration and source hashes | `python -m tools.record_run` |

The shell launchers keep model/data/output paths configurable. They call the
corresponding Python modules rather than implementing model logic.

## Model internals

`slimkv/__init__.py` exports the loader and selects Llama3 or Qwen2. Shared
arguments, data processing, Trainer integration and Beacon window planning live
in `args.py`, `data.py`, `trainer.py` and `memory.py`.

In `slimkv/llama3/`, `modeling_llama.py` contains the projections and model
forward, and `memory.py` manages compressed storage and reconstruction views.
`latent_decode.py` and `fa2_decode.py` implement direct latent attention;
`decode_ops.py` / `decode_kernels.py`, `static_decode.py`, `decode_graph.py` and
`compile_decode.py` provide optional decode acceleration. `prefill_ops.py` and
`prefill_kernels.py` implement the original-dimensional prefill fast path.

## Generated files

Checkpoints, datasets, predictions, profiler traces and build products are
excluded by `.gitignore`. Local settings and notes belong in `.local/`, with
personal exclusions in `.git/info/exclude`.
