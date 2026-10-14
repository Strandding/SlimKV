# Source and tool map

[中文](zh/structure.md)
Paths are retained to preserve imports, checkpoint handling and historical experiment launchers.

| Location | Responsibility |
|---|---|
| `slimkv/__init__.py` | `get_model_and_tokenizer`, architecture dispatch and public exports |
| `slimkv/args.py`, `slimkv/data.py`, `slimkv/trainer.py` | CLI dataclasses, dataset preprocessing and Trainer integration |
| `slimkv/memory.py` | Shared Beacon window/insertion and cache management |
| `slimkv/llama3/memory.py` | Low-rank storage, reconstruction views, cache eligibility and decode bookkeeping |
| `slimkv/llama3/modeling_llama.py` | Llama attention/projections/model forward; original-dimension prefill and training |
| `slimkv/llama3/fa2_decode.py`, `decode_ops.py`, `decode_kernels.py`, `decode_graph.py` | Optional global-latent decode, fused operators and partial Graph |
| `slimkv/qwen2`, `slimkv/llama`, `slimkv/mistral` | Other existing model paths; see support restrictions |
| `slimkv/latency` | Independent latency research prototype; separate loader and cache semantics |
| `main` | Installation-visible command modules; `infer`, `train`, conversion and evaluation |
| `scripts/conversion`, `training`, `evaluation`, `benchmark` | Environment-configurable launch presets; source `scripts/common.sh` |
| `scripts/slurm` | Cluster-specific scheduler templates; check partition/GPU/module settings |
| `scripts/vllm` | Separate-environment Full KV baseline setup/evaluation utilities |
| `config/data`, `config/deepspeed` | Data mixtures and distributed-training settings |
| `experiments/anchor_prototype` | Preserved historical implementation; not installed |
| `tests` | CPU regression and CLI/provenance checks; no download required |
| `kernels/fa2_decode` | BF16 SM90 forward-only width-320 extension and upstream license |

## Tools by purpose

**General maintenance:** `check_repository.py` (syntax, local imports, shell paths, Markdown links, tracked artifacts); `record_run.py` (configuration/source/package/hardware provenance).

**Controlled evaluation:** `prepare_longbench.py` produces benchmark JSONL fixtures from official local task files; selection is explicit and deterministic, with hashes and source indices.

**Current performance:** `benchmark_reconstruction.py`, `benchmark_fa2_decode.py` (including `--compare_b1` and `--reference_decode_source` controls), and `benchmark_batch_decode.py`. Each is a specialized experiment harness, not a unified serving benchmark.

**GPU development:** `build_fa2_decode.py`, `validate_fa2_decode.py`, `validate_decode_ops.py`, `validate_b1_decode.py`, `profile_fa2.py`; some need the width-320 extension or a real checkpoint. GPU validation is outside public CPU CI.

**Hardware helpers:** GPU/communication/watch/occupancy tools are opt-in diagnostics, not inference entry points. Post-job occupancy is disabled unless explicitly requested through launcher environment flags.

## Local artifacts

`model/`, `data/datasets/`, `data/outputs/`, `data/results/`, `data/fig/`, build products and profiler traces are ignored. `data/toy/infbench.json` is the tracked example. Do not commit weights, private data, credentials, or profiler databases. Historical documentation is preserved in `docs/archive/`; it describes the environment at the time, not a current support guarantee.
