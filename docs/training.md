# Initialization and training

[中文](zh/training.md)
Training has three stages: offline SVD initialization, low-rank pretraining, and long-context finetuning. By default the backbone is frozen and only Beacon parameters are updated (`only_train_beacon`). Existing `beacon_*` fields and weight names remain compatible.

## Inputs and resources

Install [training dependencies](installation.md#training). Set paths in the shell; `.env.example` is not loaded automatically:

```bash
export SLIMKV_DATA_ROOT=/path/to/long-llm
export SLIMKV_BASE_MODEL_DIR=/path/to/base-models
export SLIMKV_OUTPUT_DIR=/path/to/slimkv-outputs
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NPROC_PER_NODE=4
# Optional: use an already activated CUDA environment instead of a cluster module.
export SLIMKV_CUDA_MODULE=''
```

Presets assume multiple CUDA GPUs, BF16, FlashAttention, gradient checkpointing and DeepSpeed ZeRO-2. The required GPU memory depends on model, lengths and distributed settings; a minimum successful training configuration has not been measured in this refresh. Do not interpret the single-H20 inference test as evidence that paper training fits one GPU.

Expected local data for Llama presets:

| Stage | Files under `SLIMKV_DATA_ROOT` |
|---|---|
| Pretraining | `redpajama/train.json` |
| Finetuning | `gpt/one_detail_book.train.16K.json`, `gpt/one_detail_paper.train.16K.json`, `longalpaca/train.json`, `booksum/train.16K.json`, `needle/train.16K.json`, and `redpajama/train.json` |

The `[5000]` suffix used by finetuning is a loader subset selector, not part of the filename. Read `slimkv/data.py` and `config/data/` for data schema and preprocessing. The `long-llm:` prefix resolves against `--data_root`. Public source datasets do not by themselves reproduce the private processed mixture: author-provided preprocessing versions, sample manifests and GPT/needle data are still required.

## Low-rank initialization

```bash
MODEL_NAME_OR_PATH=/path/to/Llama-3.1-8B-Instruct \
OUTPUT_DIR=/path/to/initialized-lowrank \
SVD_DEVICE=cpu \
bash scripts/conversion/convert_llama31_8B_layerwise_lowrank.sh
```

This preset performs global SVD with separate explicit K/V rank lists. **Its default lists are historical presets, not the paper's automatic allocator or the private f4 checkpoint's ranks.** For an intended layer-adaptive experiment, supply both lists:

```bash
export BEACON_K_LATENT_HEAD_DIM_LIST='...'
export BEACON_V_LATENT_HEAD_DIM_LIST='...'
# Replace ... with one positive integer per layer before running the conversion.
```

List entries are dimensions averaged over KV heads; actual global rank is `num_key_value_heads * latent_head_dim`. Lists must match the layer count; K and V can differ. Use the actual checkpoint's lists when continuing that experiment. Do not replace a layer-adaptive list with a uniform compression ratio.

The general entry is `python -m main.convert_kv_lowrank --help`. Global SVD supports Llama3 and Qwen2; per-head SVD is a Llama3 option. The converter saves weights, config, tokenizer, optional spectra plot and singular-value `.pt` records. CPU SVD avoids a GPU allocation but can be expensive for full-sized checkpoints.

## Pretraining and finetuning

The following illustrates uniform f4 training; it is **not** an exact adaptive-rank paper recipe:

```bash
MODEL_NAME_OR_PATH=/path/to/initialized-lowrank \
BEACON_KV_COMPRESSION=lowrank_svd \
BEACON_KV_LATENT_COMPRESS_RATIO=4 \
BEACON_SKIP_ROPE_FOR_BEACON_K=True \
BEACON_CHUNK_FIXED_ROPE_FOR_BEACON_K=False \
bash scripts/training/train_llama_8B.sh

MODEL_NAME_OR_PATH=/path/to/pretrained/checkpoint-6000 \
BEACON_KV_COMPRESSION=lowrank_svd \
BEACON_KV_LATENT_COMPRESS_RATIO=4 \
BEACON_SKIP_ROPE_FOR_BEACON_K=True \
BEACON_CHUNK_FIXED_ROPE_FOR_BEACON_K=False \
bash scripts/training/finetune_llama_8B.sh
```

For adaptive ranks, pass both `BEACON_*_LATENT_HEAD_DIM_LIST` variables in **both stages**, matching initialization. Check launcher summaries before training: several presets have historical r2, chunk-fixed RoPE and mixture defaults. Llama finetuning's fixed token-ratio list is 4/8/16/32; it is not controlled by all of the pretraining preset's variables. Existing numerical defaults are retained.

Llama presets target `main.train`; Qwen2.5 presets are `train_qwen25_14B.sh` / `finetune_qwen25_14B.sh`. Generic `train.sh` / `finetune.sh` are historical Qwen presets, not architecture-neutral wrappers. Slurm templates require a cluster-specific partition/environment review before submission.

## Output, configuration and resume

Outputs and timestamped logs are under `SLIMKV_OUTPUT_DIR` (default `data/outputs/`). `main.train` records `run_manifest.json` on rank 0: model/training arguments including seed, checkpoint config hash, source commit and actual source hashes, package versions and visible GPU properties. Credentials are redacted in this manifest. It does not hash large weight shards or every dataset automatically; retain exact weights/data, or hash them explicitly with `tools.record_run`.

Presets support `RESUME_FROM_CHECKPOINT`; some accept `auto`/`latest`. **Most presets pass `--save_only_model`**, so their checkpoints omit optimizer/scheduler state and cannot promise an exact numerical training continuation. For a resumable training experiment, call `main.train` with `--save_only_model False` and the required distributed settings. Copy the previous run manifest before resuming into the same output directory.

The paper describes 0.5B RedPajama tokens, 0.2B long-instruction tokens, global batch 8, learning rates 5e-5/1e-5, linear decay and no warmup. Existing preset sample budgets do not establish those token budgets. Exact final schedules, seeds, data mixtures and rank lists are still needed from the authors; see [reproducibility](reproducibility.md).
