<h1 align="center"><img src="../assets/slimkv-logo.svg" width="370" alt="SlimKV"></h1>

<p align="center"><strong>SlimKV：联合 Token–Feature KV Cache 压缩与无需重建的 Beacon 注意力</strong></p>
<p align="center"><em>EMNLP 2026 Main Conference（Oral）</em></p>

<p align="center">
  <a href="https://arxiv.org/abs/2610.02953"><img src="https://img.shields.io/badge/EMNLP%202026-Oral-8a2be2.svg" alt="EMNLP 2026 Oral"></a>
  <a href="https://arxiv.org/abs/2610.02953"><img src="https://img.shields.io/badge/arXiv-2610.02953-b31b1b.svg" alt="arXiv:2610.02953"></a>
  <a href="https://huggingface.co/collections/Zihant0513/slimkv"><img src="https://img.shields.io/badge/Hugging%20Face-Models-FFD21E?logo=huggingface" alt="Model checkpoints on Hugging Face"></a>
  <a href="https://github.com/Strandding/SlimKV/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/Strandding/SlimKV/ci.yml?branch=master&amp;label=CI" alt="GitHub Actions CI"></a>
  <a href="../../LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="Apache License 2.0"></a>
</p>

<p align="center">
  <a href="#动态">动态</a> ·
  <a href="#概览">概览</a> ·
  <a href="#方法">方法</a> ·
  <a href="#模型">模型</a> ·
  <a href="#快速开始">快速开始</a> ·
  <a href="#longbench-v1">LongBench-v1</a> ·
  <a href="#decode-测速">测速</a> ·
  <a href="#引用">引用</a>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2610.02953">论文</a> ·
  <a href="https://github.com/Strandding/SlimKV">代码</a> ·
  <a href="https://huggingface.co/collections/Zihant0513/slimkv">模型</a> ·
  <a href="../../README.md">English</a>
</p>

## 动态

- **2026-10-15:** Llama-3.1-8B 与 Qwen2.5-14B 的2×、4×、8×特征压缩权重已发布至 [Hugging Face collection](https://huggingface.co/collections/Zihant0513/slimkv)。
- **2026-10-02:** 论文公开于 [arXiv](https://arxiv.org/abs/2610.02953)。

## 概览

SlimKV 同时压缩 KV Cache 的 token 数量和特征维度，用于长上下文大模型推理。它通过训练紧凑的 Beacon 记忆和逐层自适应低秩 K/V 表示实现联合压缩。

## 方法

传统特征压缩需要先重建原始维度的 Key，再应用 RoPE。Beacon Keys 对去除 key-side RoPE 的敏感度低于 raw-token Keys。SlimKV 利用这种位置不对称性，在关闭 Beacon K-RoPE 的条件下训练低秩 Beacon 投影，同时保留 raw-token Keys 和全部 Queries 的 RoPE。

<p align="center">
  <img src="../assets/method.svg" width="1000" alt="SlimKV 方法概览">
</p>

Prefill 在原始特征维度计算 attention，将完成的窗口存为压缩 Beacon K/V。Decode 可选择先重建再做 attention，或直接进行 latent-space attention，避免重建历史 Beacon K/V。Raw tail 和 sink 保留原始维度的 K/V。实验与结果见[论文](https://arxiv.org/abs/2610.02953)。

## 模型

| 基座 | 2× 特征压缩 | 4× 特征压缩 | 8× 特征压缩 |
| --- | --- | --- | --- |
| Llama-3.1-8B | [r2](https://huggingface.co/Zihant0513/SlimKV-Llama3.1-8B-r2) | [r4](https://huggingface.co/Zihant0513/SlimKV-Llama3.1-8B-r4) | [r8](https://huggingface.co/Zihant0513/SlimKV-Llama3.1-8B-r8) |
| Qwen2.5-14B | [r2](https://huggingface.co/Zihant0513/SlimKV-Qwen2.5-14B-r2) | [r4](https://huggingface.co/Zihant0513/SlimKV-Qwen2.5-14B-r4) | [r8](https://huggingface.co/Zihant0513/SlimKV-Qwen2.5-14B-r8) |

`r2`、`r4`、`r8` 表示特征压缩倍数，token 压缩比例在推理时选择。例如 r4 权重配合 token ratio 4 为 t4f4，名义联合压缩16×；配合 token ratio 8 为 t8f4，名义联合压缩32×。这些倍数描述 KV 压缩，不等于整个模型的显存减少倍数。

以下流程使用 **Llama-3.1-8B-r4**。下载内容包含基座、训练后的 Beacon 投影、配置和 tokenizer，无需另行下载或合并基座权重。Qwen 使用重建路径，详见[后端支持](implementation_status.md)。

## 快速开始

以下命令在同一个已激活环境的终端中，从仓库根目录执行。FA2 安装示例面向单张 **H20/H100 类 SM90 GPU**，使用 Python 3.11 或3.12、兼容的 NVIDIA 驱动及 **CUDA 12.9 开发工具链（`nvcc`）**。r4 的 latent FA2 路径需要仓库中的320维扩展。其他硬件及 eager 用法见[安装指南](installation.md)。

### 1. 安装

```bash
git clone https://github.com/Strandding/SlimKV.git
cd SlimKV
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu129
python -m pip install -e . "huggingface_hub[cli]>=0.34,<1" ninja packaging
MAX_JOBS=2 python -m pip install flash-attn==2.8.3 --no-build-isolation

git clone --depth 1 --branch v2.8.3 \
  https://github.com/Dao-AILab/flash-attention.git .cache/flash-attention-v2.8.3
git -C .cache/flash-attention-v2.8.3 submodule update --init csrc/cutlass
python -m tools.build_fa2_decode \
  --fa2_source .cache/flash-attention-v2.8.3 \
  --output_dir .cache/fa2_decode_320 --jobs 2
export SLIMKV_FA2_DECODE_EXTENSION_DIR="$PWD/.cache/fa2_decode_320"
```

该扩展用于 SM90 BF16 推理。独立推理和评测脚本使用 Transformers 4.40.2，vLLM 使用单独环境。

### 2. 下载权重

```bash
export MODEL_NAME_OR_PATH="$PWD/models/SlimKV-Llama3.1-8B-r4"
hf download Zihant0513/SlimKV-Llama3.1-8B-r4 \
  --local-dir "$MODEL_NAME_OR_PATH"
```

保留下载目录中的全部权重分片、分片索引、配置及 tokenizer。重复运行 `hf download` 会复用已有文件。新开终端后，重新激活 `.venv`，并再次设置 `MODEL_NAME_OR_PATH` 和 `SLIMKV_FA2_DECODE_EXTENSION_DIR`。

### 3. 生成一条回答

```bash
python -m main.inference.infer \
  --model_name_or_path "$MODEL_NAME_OR_PATH" \
  --prompt 'Explain why a KV cache helps language model inference.' \
  --max_new_tokens 32
```

长上下文通过 `--prompt_file /path/to/context.txt` 传入 UTF-8 文件。CUDA 默认使用 BF16、原始维度 FA2 prefill、免重建 latent decode，并为满足条件的 B1 调用启用 Graph 和可选 decoder-tail 编译。Python API、eager 和重建路径见[推理指南](quickstart.md)。

## LongBench-v1

### 下载数据

```bash
export EVAL_DATA="$PWD/data/datasets/longbench"
python -m tools.data.download_longbench --output_dir "$EVAL_DATA"
```

下载器获取官方 v1 压缩包并提取所选子集的 JSONL 文件，仓库不包含数据文件。

### 先用四条样本检查流程

```bash
TASKS="qasper hotpotqa" MAX_SAMPLES_PER_TASK=2 BATCH_SIZE=1 \
RESULT_DIR=quickstart-t4f4 \
  bash scripts/evaluation/eval_longbench_slimkv.sh
```

该命令评测每个所选子集的前两条样本。流程跑通后，再执行下面的完整英文/代码子集评测。

### 运行 t4f4 或 t8f4

```bash
BEACON_RATIO=4 RESULT_DIR=longbench-t4f4 \
  bash scripts/evaluation/eval_longbench_slimkv.sh

BEACON_RATIO=8 RESULT_DIR=longbench-t8f4 \
  bash scripts/evaluation/eval_longbench_slimkv.sh
```

默认评测14个英文/代码子集，使用 BF16 greedy decode，**最终聊天模板处理后的输入上限为31,500 token**，超长输入保留首尾。训练语料只有中文，因此过滤中文子集；另行排除 `passage_count` 和 `passage_retrieval_en`。Batch 上限32，仅将等长输入组成 batch，不做 padding，可根据显存调整 `BATCH_SIZE` 和 `MAX_BATCH_TOKENS`。

输出位于 `data/results/longbench/<RESULT_DIR>/`：`scores.json` 保存得分，`task_timings.csv` 保存每个子集的耗时，`progress.json` 保存进度。保持配置一致并设置 `LOAD_RESULT=True` 可以复用已完成子集。任务选择、输出定义和后端选项见[评测指南](evaluation.md)。

## Decode 测速

测速无需下载 LongBench，脚本会构造并保存合成输入。使用前面下载的权重和已导出的扩展路径。

先测试一组32K输入、t4f4：

```bash
CONTEXT_LENGTHS=32768 BEACON_RATIOS=4 REPETITIONS=1 \
OUTPUT_DIR=data/results/speed-smoke \
  bash scripts/benchmark/benchmark_context_decode.sh
```

再测试默认的 **32K、64K、96K、128K** 输入及 **t4f4、t8f4** 两种配置：

```bash
OUTPUT_DIR=data/results/speed-32k-128k \
  bash scripts/benchmark/benchmark_context_decode.sh
```

每个输入长度都对比 Full KV、重建和直接 latent attention，统一 **B1、BF16、FA2**，采用相同的预分配、整模型 Graph 和可选编译策略。每轮在 prefill 后测量 **128次 decode forward**，各方法使用相同输入及后续 token。完整预设先预热一次，再测四次取中位数，decode 计时排除 prefill、编译和 Graph 构建。测速前会对照各自参考路径检查优化后的 logits 和最终 KV。

分别保存**attention GPU kernel 耗时**和**端到端 decode 耗时**，重建路径的前者包含每一步 KV 重建及 RoPE。`results.json` 包含时间、tokens/s 与数值检查，`summary.csv` 包含各模式的中位数。每次测速使用新的 `OUTPUT_DIR`。Full KV、参考快照及 Graph 工作区都需要额外显存，显存不足时减少 `CONTEXT_LENGTHS`。指标边界、运行开关及 profiling 见[测速指南](benchmark.md)。

生成的输出保存在已忽略的 `data/results/` 下。仓库提供实验代码和操作说明，实验结果统一见[论文](https://arxiv.org/abs/2610.02953)。

## 训练与其他工具

| 任务 | 入口 | 指南 |
| --- | --- | --- |
| 低秩初始化 | `python -m main.conversion.convert_kv_lowrank --help` | [训练](training.md) |
| 预训练 | `bash scripts/training/train_llama_8B.sh` | [训练](training.md) |
| 微调 | `bash scripts/training/finetune_llama_8B.sh` | [训练](training.md) |
| Needle-in-a-Haystack | `python -m main.evaluation.eval_needle --help` | [评测](evaluation.md) |
| vLLM 推理 | `python -m main.inference.infer_vllm --help` | [vLLM](vllm.md) |

其他细节见[复现说明](reproducibility.md)、[源码导航](structure.md)和[后端支持](implementation_status.md)。

## 引用

ACL Anthology 的正式论文记录尚未公开，以下使用临时会议引用并链接 arXiv 版本。

```bibtex
@inproceedings{teng2026slimkv,
  title = {SlimKV: Joint Token-Feature KV Cache Compression with Reconstruction-Free Beacon Attention},
  author = {Teng, Zihan and Zhao, Jiayu and Ren, Wentao and Fan, Minhao and Ma, Tianrui and Chen, Song and Liu, Weichen},
  booktitle = {2026 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year = {2026},
  note = {Accepted to the Main Conference (Oral). Provisional conference citation.},
  eprint = {2610.02953},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG},
  url = {https://arxiv.org/abs/2610.02953}
}
```

## 致谢与许可

SlimKV 基于 [Activation Beacon](https://github.com/FlagOpen/FlagEmbedding/tree/master/research/Long_LLM/activation_beacon)、Hugging Face Transformers 和 [FlashAttention](https://github.com/Dao-AILab/flash-attention)。归属和许可见[第三方声明](../../THIRD_PARTY_NOTICES.md)。SlimKV 原创代码采用 [Apache License 2.0](../../LICENSE)。第三方组件保留各自的版权与许可，模型权重和数据集遵循各自的授权条款。
