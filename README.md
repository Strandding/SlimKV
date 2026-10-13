# SlimKV

SlimKV 通过 Beacon 汇聚历史上下文，训练逐层低秩 K/V 投影。Llama3 的常规推理可将 Beacon K/V 保存为低维表示，在 prefill 和 decode 的注意力计算前按层重建到原始维度；raw token 和 sink 保持完整维度，训练沿用完整维度缓存。

本仓库对应论文 **SlimKV: Joint Token-Feature KV Cache Compression with Reconstruction-Free Beacon Attention**。当前主实现支持 Llama3 与 Qwen2 的低秩训练；论文中的 LSA 尚未实现为正式推理后端。`slimkv/latency` 保留独立实验代码，不作为当前推理和质量评测入口。Llama3 的 `beacon_cache_mode=auto` 会为兼容的低秩配置启用低维存储与重建；`full` 保留完整缓存作对照，`latent` 显式要求低维缓存。Qwen2 当前仍使用完整维度缓存。具体支持范围和已知问题见 [实现状态](docs/implementation_status.md)。

## 目录

```text
SlimKV/
├── slimkv/                  # 模型、配置、Memory、训练与数据工具
│   ├── llama3/              # Llama3 低秩模型
│   ├── qwen2/               # Qwen2/Qwen2.5 低秩模型
│   ├── llama/              # 旧 Llama 基础支持
│   ├── mistral/            # Mistral 基础支持
│   └── latency/            # Llama latent cache 与 hybrid attention 原型
├── main/                    # python -m main.<command> 命令入口
├── scripts/                 # training、conversion、evaluation、benchmark、slurm、vllm
├── config/                  # DeepSpeed、数据与分布式配置
├── tools/                   # 仓库检查、GPU 与通信工具
├── docs/                    # 训练、评测、实现状态与同步说明
├── experiments/anchor_prototype/  # GitHub 早期 anchor 原型归档
├── data/                    # 示例；结果、图、数据集与输出为本地产物
└── model/                   # 本地权重，不进入 Git
```

## 安装

在独立环境中先安装适合机器的 PyTorch，再安装项目。主实现使用 checkpoint 保存时的 Transformers 4.40 系列接口；依赖范围在 `pyproject.toml` 中记录。

```bash
git clone https://github.com/Strandding/SlimKV.git
cd SlimKV
python -m pip install -e .
# 训练时安装可选依赖
python -m pip install -e '.[training]'
# CUDA、PyTorch 兼容后，再按需安装 FlashAttention2
python -m pip install flash-attn --no-build-isolation
```

vLLM 基线使用独立环境，见 `scripts/vllm/`，避免其 Transformers 依赖改变主实现环境。

## 路径配置

启动脚本会自动定位项目根目录，无需修改代码中的机器路径。变量示例见 [.env.example](.env.example)；文件不会自动加载，运行前在 shell 中 export。

```bash
export SLIMKV_DATA_ROOT=/path/to/long-llm
export SLIMKV_BASE_MODEL_DIR=/path/to/base-models
export SLIMKV_OUTPUT_DIR=/path/to/slimkv-outputs
export MODEL_NAME_OR_PATH=/path/to/checkpoint-2000
export CUDA_VISIBLE_DEVICES=0
export NPROC_PER_NODE=1
```

训练、评测完成后默认退出。GPU 占用功能只有显式设置 `OCCUPY_AFTER_TRAIN=True`、`OCCUPY_AFTER_FINETUNE=True` 或 `OCCUPY_AFTER_EVAL=True` 才启用。

## 使用

```bash
python -m main.train --help
python -m main.convert_kv_lowrank --help
python -m main.eval_longbench --help
python -m main.benchmark_latency --help

bash scripts/evaluation/eval_longbench_llama31_8B.sh
python tools/check_repository.py
```

详见 [训练](docs/training.md)、[评测](docs/evaluation.md) 和 [本机与 H20 同步](docs/synchronization.md)。已有权重的 `beacon_*` 配置字段和 tensor 名称继续兼容；项目包名与新输出名称统一为 SlimKV。

## 来源

Beacon 分块压缩基础实现来自 [Activation Beacon](https://arxiv.org/abs/2401.03462)，模型实现保留原有作者和许可声明。仓库中的早期 anchor 原型独立归档，不参与主实现的安装。

```bibtex
@misc{zhang2024soaring,
  title={Soaring from 4K to 400K: Extending LLM's Context with Activation Beacon},
  author={Peitian Zhang and Zheng Liu and Shitao Xiao and Ninglu Shao and Qiwei Ye and Zhicheng Dou},
  year={2024},
  eprint={2401.03462},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}
```
