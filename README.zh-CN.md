# SlimKV：联合 token 与特征维度 KV Cache 压缩及无需重建的 Beacon 注意力

[English](README.md) · [快速开始](docs/zh/quickstart.md) · [论文复现](docs/zh/reproducibility.md) · [实现状态](docs/zh/implementation_status.md)

SlimKV 通过训练紧凑的 Beacon 记忆，同时减少长上下文 KV Cache 的 token 数量和特征维度。

**发布状态：研究实现。** 尚未确认公开论文地址、模型下载地址和发表会议。下面的论文结果来自作者提供的稿件，不代表当前仓库已完成全部复现。

## 方法概览

- **联合 token–feature 压缩**：Beacon 总结已完成的上下文窗口，低秩 K/V 投影继续压缩每个 Beacon 的特征。
- **逐层自适应低秩表示**：不同层的 K/V rank 分配容量。当前代码接受显式 rank 列表并支持 SVD 初始化，尚未实现论文中的自动预算分配算法。
- **无需重建的 Beacon 注意力**：Beacon K 不使用 K-side RoPE；raw K 和所有 Q 保留 RoPE，使 latent attention 可以通过 query/output 投影完成。

训练和 prefill 的注意力仍在原始维度计算。兼容配置下的 Llama3 推理存储低维 Beacon K/V；默认 decode 先重建，再计算标准注意力。可选 FA2 decode 直接计算全局 latent Beacon 注意力，并与完整维度的 raw/sink 分支合并。Qwen2 支持低秩训练，但常规推理仍保存完整维度缓存。

## 方法图

![论文 Figure 3](docs/assets/method.svg)

该图复用作者提供稿件的 Figure 3，展示论文设计。当前重建与 latent decode 路径的区别见 [英文 README 示意图](README.md#method-overview)；方法图不意味着全部论文实验已复现。

## 主要结果

**论文报告数据**，来源为 Table 2、Table 6 和 §5.4：

| 指标 | Llama-3.1-8B | Qwen2.5-14B |
|---|---:|---:|
| Full KV 的 LongBench 平均分 | 49.50 | 51.52 |
| SlimKV-f4，名义 16×（t4f4） | 46.36 | 46.23 |
| SlimKV-f4，名义 32×（t8f4） | 44.75 | 45.09 |
| 128K t4f4 attention 加速 | 7.34× | — |
| 128K t4f4 每 token decode 加速 | 3.38× | — |
| 11.5K t4f4 / t8f4 峰值显存减少倍数 | 2.71× / 2.82× | — |

论文效率协议使用 A100 80GB、B1、BF16、eager，以及长上下文 synthetic Full KV。这些不是 H20 FA2 后端的实测数据。Token × feature 压缩是名义比例；raw tail/sink 使实际缓存比例降低。

**仓库局部验证**：H20、私有 f4 BF16 checkpoint，8 条 LongBench-v1 样本及 2 条计时输入。标准 FA2 prefill 加局部 Graph 的 decode，在 4009 / 6194 token 输入上为 18.38 / 18.67 ms/token；同轮 compact full 对照为 26.34 / 26.40 ms/token。B1 Graph 额外持有约 515.75 MiB。它们是 smoke/scaling 数据，不是完整 LongBench 结果。[协议、范围和证据](docs/zh/benchmark.md)。

## 安装

使用独立 Python 3.11 环境，先安装匹配硬件的 PyTorch，再执行：

```bash
git clone https://github.com/Strandding/SlimKV.git
cd SlimKV
python -m pip install -e .
python -m pip check
```

模型实现要求 Transformers 4.40.x，已验证 4.40.2。最小 eager 推理不需要 FlashAttention；训练和 kernel 环境要求见 [安装说明](docs/zh/installation.md)。

## 快速开始

需要**训练后的 SlimKV checkpoint**，包含 config、完整权重及 tokenizer。单独的 Llama 基座不包含训练后的 SlimKV 投影。目前无公开权重下载地址，请从作者处获取或按训练说明构建。

```bash
python -m main.infer \
  --model_name_or_path /path/to/slimkv-checkpoint \
  --prompt 'Explain why a KV cache helps language model inference.' \
  --max_new_tokens 32
```

默认 BF16、cuda:0、eager、greedy、重建后端，并读取 checkpoint 的 rank 和 Beacon 配置。长上下文使用 `--prompt_file`；短 prompt 可能未完成一个 Beacon 窗口。[Python API 与 FA2](docs/zh/quickstart.md)。

## 复现实验入口

| 阶段 | 入口 | 文档 |
|---|---|---|
| SVD 初始化 | `python -m main.convert_kv_lowrank --help` | [训练](docs/zh/training.md) |
| 预训练 | `scripts/training/train_llama_8B.sh` | [训练](docs/zh/training.md) |
| 微调 | `scripts/training/finetune_llama_8B.sh` | [训练](docs/zh/training.md) |
| LongBench | `python -m main.eval_longbench --help` | [评测](docs/zh/evaluation.md) |
| NIAH / accuracy | `python -m main.eval_needle --help` | [评测](docs/zh/evaluation.md) |
| 耗时与显存 | `python -m tools.benchmark_fa2_decode --help` | [benchmark](docs/zh/benchmark.md) |

从仓库根目录启动，路径用环境变量覆盖。[.env.example](.env.example) 不会自动加载。完整论文复现还缺资源和协议，见 [复现说明](docs/zh/reproducibility.md)。

## 目录

```text
slimkv/                     模型、配置、Memory、训练工具
  llama3/                   Llama3 低秩缓存与可选 latent FA2
  qwen2/                    Qwen2 低秩训练，完整维度缓存
  latency/                  独立研究原型
main/                       Python 命令入口
scripts/                    conversion / training / evaluation / benchmark / slurm / vllm
config/                     数据和分布式配置
kernels/fa2_decode/          BF16 SM90 width-320 forward
tools/                     评测、验证、provenance 与硬件工具
tests/                     CPU 回归；GPU 验证入口在 tools
docs/                      英文指南、zh 中文指南及 archive 历史记录
experiments/anchor_prototype/  不参与主安装的早期原型
requirements/              CPU CI 约束
data/toy/                  小样例；其他数据、结果和权重保留在本地
```

[模块与工具对应表](docs/zh/structure.md)。CPU 检查：`python tools/check_repository.py` 和 `python -m unittest discover -s tests -v`。

## 引用

作者列表来自稿件，未杜撰年份、会议、DOI 或 arXiv ID。

```bibtex
@unpublished{teng_slimkv,
  title = {SlimKV: Joint Token-Feature KV Cache Compression with Reconstruction-Free Beacon Attention},
  author = {Teng, Zihan and Zhao, Jiayu and Ren, Wentao and Fan, Minhao and Ma, Tianrui and Chen, Song and Liu, Weichen},
  note = {Unpublished manuscript}
}
```

[CITATION.cff](CITATION.cff) 提供机器可读信息。

## 致谢和许可

SlimKV 基于 [Activation Beacon](https://github.com/FlagOpen/FlagEmbedding/tree/master/research/Long_LLM/activation_beacon)、Transformers 和 FlashAttention。保留上游版权、许可及归属，[第三方声明](THIRD_PARTY_NOTICES.md) 附对应文本。

**SlimKV 原创贡献的许可证尚待作者决定**，本次没有代选。上游组件、模型和数据各自的许可继续适用。
