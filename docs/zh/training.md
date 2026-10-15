# 初始化与训练

[English](../training.md)

三个阶段为离线 SVD、低秩预训练、长上下文微调。默认冻结 backbone，仅训练 Beacon 参数；`beacon_*` 格式保持兼容。先安装训练依赖并设置：

```bash
export SLIMKV_DATA_ROOT=/path/to/long-llm
export SLIMKV_BASE_MODEL_DIR=/path/to/base-models
export SLIMKV_OUTPUT_DIR=/path/to/slimkv-outputs
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NPROC_PER_NODE=4
export SLIMKV_CUDA_MODULE=''
```

`.env.example` 不自动加载。预设使用多 GPU、BF16、FA2、gradient checkpointing 和 ZeRO-2；所需显存取决于模型、上下文长度与分布式配置。

预训练需要 `redpajama/train.json`。微调需要 `gpt/one_detail_book.train.16K.json`、`gpt/one_detail_paper.train.16K.json`、`longalpaca/train.json`、`booksum/train.16K.json`、`needle/train.16K.json` 及 RedPajama。`[5000]` 是 loader 的子集选择语法；`long-llm:` 由 data_root 解析。公开原始数据不等同于论文处理后的混合数据。

## SVD

```bash
MODEL_NAME_OR_PATH=/path/to/Llama-3.1-8B-Instruct \
OUTPUT_DIR=/path/to/initialized-lowrank \
SVD_DEVICE=cpu \
bash scripts/conversion/convert_llama31_8B_layerwise_lowrank.sh
```

此预设的默认 rank 列表是显式配置。逐层实验必须显式传 `BEACON_K_LATENT_HEAD_DIM_LIST` 和 `BEACON_V_LATENT_HEAD_DIM_LIST`；每层各一个正整数，列表长度匹配层数。元素为平均每个 KV head 的维度参数，全局 rank 为 KV head 数乘该值，K/V 可不同。初始化、预训练、微调都必须用同一列表。

通用入口 `python -m main.conversion.convert_kv_lowrank --help` 支持 Llama3/Qwen2 global SVD，Llama3 还有 per-head 选项。转换保存模型、配置、tokenizer，可选 spectra 图和奇异值 tensor。

### 自动等能量秩分配

`--rank_allocation equal_energy` 根据预训练 raw-token K/V 投影的奇异值谱，按论文 Eq. 5–9 分别分配 K 和 V 的秩预算。`--feature_compression_ratio` 控制特征压缩，与 token 压缩独立。论文的秩下限系数为 `--rank_floor_ratio 0.5`。以下为 f4 初始化，将 `4` 改为 `2` 或 `8` 即可选择其他特征预算。

```bash
python -m main.conversion.convert_kv_lowrank \
  --model_name_or_path /path/to/Llama-3.1-8B-Instruct \
  --output_dir /path/to/initialized-lowrank-f4 \
  --beacon_param q k v \
  --beacon_skip_rope_for_beacon_k True \
  --beacon_chunk_fixed_rope_for_beacon_k False \
  --rank_allocation equal_energy \
  --feature_compression_ratio 4 \
  --rank_floor_ratio 0.5 \
  --svd_device cpu \
  --save_dtype bf16
```

算法求出预算内最大的共同能量阈值，施加秩下限，再按归一化边际能量增益分配剩余容量。全局秩向上对齐到 KV head 数的倍数，同时不超出各自的预算。不能整除该粒度的预算余量会记录在报告中。自动模式要求 global、独立 K/V 和非 headwise 投影。

自动模式从 **raw-token** K/V 的截断 SVD 初始化 Beacon K/V，会替换已有的 Beacon K/V 权重，基座不变。手工转换仍为默认模式，保留原有行为。自动模式不能同时传入显式 latent 维度或逐层秩列表。

输出 config 保存 K/V 秩列表和 `beacon_rank_allocation`；`rank_allocation.json` 记录阈值、请求与实际预算、秩下限和保留能量。初始化复用已经计算的 SVD 分解，不重复计算。预训练和微调须沿用输出的秩列表。

## 训练示例

以下只是均匀 f4 示例，不是逐层论文精确配置：

```bash
MODEL_NAME_OR_PATH=/path/to/initialized-lowrank \
BEACON_KV_COMPRESSION=lowrank_svd BEACON_KV_LATENT_COMPRESS_RATIO=4 \
BEACON_SKIP_ROPE_FOR_BEACON_K=True BEACON_CHUNK_FIXED_ROPE_FOR_BEACON_K=False \
bash scripts/training/train_llama_8B.sh

MODEL_NAME_OR_PATH=/path/to/pretrained/checkpoint-6000 \
BEACON_KV_COMPRESSION=lowrank_svd BEACON_KV_LATENT_COMPRESS_RATIO=4 \
BEACON_SKIP_ROPE_FOR_BEACON_K=True BEACON_CHUNK_FIXED_ROPE_FOR_BEACON_K=False \
bash scripts/training/finetune_llama_8B.sh
```

自适应实验在两阶段显式传对应 K/V 列表。部分默认仍是 r2、chunk-fixed 或历史 mixture，应检查脚本摘要。Llama 微调的 token ratio 固定列表为 4/8/16/32，不能假定所有预训练变量都覆盖微调。默认数值行为没有更改。

Qwen2.5 使用 `train_qwen25_14B.sh` / `finetune_qwen25_14B.sh`；通用名字 `train.sh` / `finetune.sh` 实际是历史 Qwen 预设。Slurm 提交时指定集群分区与 GPU 资源，例如 `sbatch --partition=<partition> --gres=gpu:4 scripts/slurm/train_llama31_8b_lowrank_r2_kchunkfixedrope.slurm`。通过 `SLIMKV_CUDA_MODULE` 配置集群 module，设为空字符串则使用已激活的 CUDA 环境。

## 输出与恢复

输出默认 `data/outputs/`，日志有时间戳。`main.training.train` 的 rank 0 写 `run_manifest.json`，记录配置/seed、checkpoint config SHA、源码 commit 与实际 SHA、包和 GPU 信息，凭据字段脱敏。不会自动 hash 大权重或全部数据；要保留精确输入。恢复进同输出目录前归档旧 manifest。

`RESUME_FROM_CHECKPOINT` 支持显式路径，部分脚本有 auto/latest。**多数预设传 save_only_model**，不保存 optimizer/scheduler，不能保证精确训练续跑；需精确恢复时直接调用 `main.training.train --save_only_model False` 并保存分布式状态。

论文的 0.5B RedPajama + 0.2B 指令 token、global batch 8、5e-5/1e-5、linear/no warmup，不等于脚本 sample budget。完整数据、最终配置/seed/rank 仍需作者提供。
