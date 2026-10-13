# SlimKV 训练

训练包含离线 SVD 初始化、低秩预训练和长上下文微调。Backbone 默认冻结，仅更新 Beacon 分支参数；`beacon_*` 字段保留为现有 checkpoint 的兼容接口。

## 数据与环境

先按根目录 README 安装训练依赖，设置 `SLIMKV_DATA_ROOT`、`SLIMKV_BASE_MODEL_DIR` 和 `SLIMKV_OUTPUT_DIR`。数据集应包含 RedPajama，以及微调使用的 LongAlpaca、BookSum、GPT QA 和 needle 数据。`long-llm:` 前缀由 `--data_root` 解析。

## 初始化

从原始 Llama 模型初始化低秩投影：

```bash
MODEL_NAME_OR_PATH=/path/to/Llama-3.1-8B-Instruct \
OUTPUT_DIR=/path/to/initialized-lowrank \
SVD_DEVICE=cpu \
bash scripts/conversion/convert_llama31_8B_layerwise_lowrank.sh
```

逐层 rank 通过 `BEACON_K_LATENT_HEAD_DIM_LIST`、`BEACON_V_LATENT_HEAD_DIM_LIST` 显式传入，列表元素为平均到每个 KV head 的维度参数。实际全局 latent rank 为 `num_key_value_heads * latent_head_dim`。该转换脚本默认列表是实验预设，不能视为论文自动 rank 分配算法。

通用入口为 `python -m main.convert_kv_lowrank --help`；它支持 Llama3、Qwen2 的全局 SVD，以及 Llama3 的 per-head SVD。

## 预训练与微调

```bash
MODEL_NAME_OR_PATH=/path/to/initialized-lowrank \
SLIMKV_DATA_ROOT=/path/to/long-llm \
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 \
BEACON_KV_COMPRESSION=lowrank_svd \
BEACON_KV_LATENT_COMPRESS_RATIO=4 \
BEACON_SKIP_ROPE_FOR_BEACON_K=True \
BEACON_CHUNK_FIXED_ROPE_FOR_BEACON_K=False \
bash scripts/training/train_llama_8B.sh

MODEL_NAME_OR_PATH=/path/to/pretrained/checkpoint-6000 \
SLIMKV_DATA_ROOT=/path/to/long-llm \
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 \
BEACON_KV_COMPRESSION=lowrank_svd \
BEACON_KV_LATENT_COMPRESS_RATIO=4 \
BEACON_SKIP_ROPE_FOR_BEACON_K=True \
BEACON_CHUNK_FIXED_ROPE_FOR_BEACON_K=False \
bash scripts/training/finetune_llama_8B.sh
```

使用逐层 rank 时，在两个阶段都显式传入对应 K/V 列表；不传列表且传入压缩比例会选择统一 rank，不能替代逐层 checkpoint 的配置。运行前检查脚本打印的实际参数和输出路径。

Qwen2.5 对应 `train_qwen25_14B.sh`、`finetune_qwen25_14B.sh`。配置文件位于 `config/deepspeed/` 和 `config/data/`。Slurm 模板位于 `scripts/slurm/`；从仓库根目录提交，并按实际集群修改分区、节点和 GPU 配置。旧任务 dependency 已移除。

## 恢复与输出

训练脚本支持 `RESUME_FROM_CHECKPOINT=/path/to/checkpoint`，部分脚本也支持 `auto`/`latest`。新 checkpoint、日志输出到 `SLIMKV_OUTPUT_DIR`，默认 `data/outputs/`，不会进入 Git。已有 checkpoint 目录无需改名，内部参数键也不改名。

训练数值语义和已知边界问题见 [实现状态](implementation_status.md)。
