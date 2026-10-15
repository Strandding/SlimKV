# Source and tool map

[中文](zh/structure.md)
Model implementations live in `slimkv`, command modules in `main`, and launch presets in `scripts`. Configuration and environment constraints are grouped under `config`.

| Location | Responsibility |
|---|---|
| `slimkv/__init__.py` | `get_model_and_tokenizer`, architecture dispatch and public exports |
| `slimkv/args.py`, `slimkv/data.py`, `slimkv/trainer.py` | CLI dataclasses, dataset preprocessing and Trainer integration |
| `slimkv/memory.py` | Shared Beacon window/insertion and cache management |
| `slimkv/llama3/memory.py` | Low-rank storage, reconstruction views, cache eligibility and decode bookkeeping |
| `slimkv/llama3/modeling_llama.py` | Llama attention/projections/model forward; original-dimension prefill and training |
| `slimkv/llama3/fa2_decode.py`, `decode_ops.py`, `decode_kernels.py`, `decode_graph.py` | Optional global-latent decode, fused operators and partial Graph |
| `slimkv/qwen2` | Qwen2/Qwen2.5 training and full-cache inference |
| `slimkv/vllm_plugin` | Experimental Llama paged cache and continuous scheduling integration |
| `main` | Installation-visible command modules; `infer`, `train`, conversion and evaluation |
| `scripts/conversion`, `training`, `evaluation`, `benchmark` | Environment-configurable launch presets; source `scripts/common.sh`. Qwen and Llama LongBench presets share `scripts/evaluation/longbench_common.sh` |
| `scripts/slurm` | Cluster-specific scheduler templates; check partition/GPU/module settings |
| `scripts/vllm` | Separate-environment Full KV baseline setup/evaluation utilities |
| `config/data`, `config/deepspeed` | Data mixtures and distributed-training settings |
| `config/environments` | Dependency constraints for CPU validation |
| `docs/zh`, `docs/licenses` | Chinese README/guides and third-party license texts |
| `tests` | CPU regression and CLI/provenance checks; no download required |
| `kernels/fa2_decode` | BF16 SM90 forward-only width-320 extension and upstream license |

## Tools by purpose

**General maintenance:** `check_repository.py` (syntax, local imports, shell paths, Markdown links, tracked artifacts); `record_run.py` (configuration/source/package/hardware provenance).

**Controlled evaluation:** `prepare_longbench.py` produces benchmark JSONL fixtures from official local task files; selection is explicit and deterministic, with hashes and source indices.

**Current performance:** `benchmark_reconstruction.py`, `benchmark_fa2_decode.py` (including `--compare_b1` and `--reference_decode_source` controls), and `benchmark_batch_decode.py`. Each is a specialized experiment harness, not a unified serving benchmark.

**GPU development:** `build_fa2_decode.py`, `validate_fa2_decode.py`, `validate_decode_ops.py`, `validate_b1_decode.py`, `profile_fa2.py`; some need the width-320 extension or a real checkpoint. GPU validation is outside public CPU CI.

**Communication check:** `nccl_comm_test.py` checks distributed GPU communication.

## Generated files

Checkpoints, datasets, predictions, profiler traces and build products are excluded by `.gitignore`. Store machine-specific notes and settings in `.local/`, or add personal exclusions to `.git/info/exclude`.
