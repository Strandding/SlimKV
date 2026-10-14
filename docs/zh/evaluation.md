# LongBench-v1 评测

[English](../evaluation.md)

Llama3 预设评测 14 个英文及代码子集。训练语料只有中文，因此排除中文评测子集。Synthetic 子集 `passage_count` 和 `passage_retrieval_en` 单独排除，保留 `lcc` 和 `repobench-p`。设置 `TASKS=all` 可评测 LongBench-v1 全部子集。

## 准备数据

下载官方 [LongBench-v1 压缩包](https://huggingface.co/datasets/zai-org/LongBench/tree/main)，提取所选子集：

```bash
python -m tools.download_longbench --output_dir data/datasets/longbench
```

工具固定数据 revision `5e628be450b7e67fb7ae6e201bd6d8f7056f7672`，将压缩包和各任务文件的 SHA256 保存到 `dataset_manifest.json`。已有 `data.zip` 时，添加 `--archive /path/to/data.zip`，无需下载。不会覆盖内容不同的已有任务文件。使用 v1 JSONL，不混用 LongBench-E 或 LongBench-v2。

## 运行 SlimKV

使用包含权重、config 和 tokenizer 的 Llama3 SlimKV checkpoint。特征 rank 从 checkpoint 读取。f4 权重搭配 `BEACON_RATIO=4` 即 t4f4，标称 token-feature 压缩比为 16×。

```bash
MODEL_NAME_OR_PATH=/path/to/slimkv-f4-checkpoint \
EVAL_DATA=/path/to/longbench \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/evaluation/eval_longbench_slimkv.sh
```

默认采用 BF16 greedy 推理、token ratio 4、latent KV 存储、原始维度 FA2 prefill、latent-space FA2 Beacon decode、CUDA Graph 和 `num_logits_to_keep=1`。Beacon K 不加 RoPE，各层 rank 和投影结构从 checkpoint 读取。

### 选择 attention 和 decode 路径

`ATTN_IMPL` 选择原始维度 attention 的实现，`DECODE_BACKEND` 选择 decode 时如何使用压缩后的 Beacon。下列组合都保留 latent KV 存储。

| 环境变量 | Prefill | Beacon decode | 默认 Graph 设置 |
| --- | --- | --- | --- |
| `ATTN_IMPL=eager` | Eager | 重建后使用 eager attention | 关闭 |
| `ATTN_IMPL=eager DECODE_BACKEND=latent` | Eager | 免重建 latent eager | 关闭 |
| `ATTN_IMPL=flash_attention_2 DECODE_BACKEND=reconstruct` | FA2 | 重建后使用 FA2 | 关闭 |
| `ATTN_IMPL=flash_attention_2` | FA2 | 免重建 latent FA2 | 开启 |

`DECODE_BACKEND=reconstruct` 在 attention 前重建 Beacon K/V；`DECODE_BACKEND=latent` 直接在压缩后的 Beacon K/V 上计算，使用所选 `ATTN_IMPL`。Eager 的两条路径都无需安装 FlashAttention 或构建320维扩展。Latent eager 使用 FP32 attention scores 和加权求和，支持 K/V rank 不同，并保留原始 head 的缩放系数。

旧的 `DECODE_BACKEND=fa2` 保留为显式 latent FA2 decode，不受 prefill 选择影响，兼容原有命令。`DECODE_GRAPH=False` 可单独关闭 Graph。

默认结果目录包含两项后端名称，各路径分别保存。直接使用 `python -m main.eval_longbench` 时，应显式选择 `--attn_impl` 和 `--beacon_decode_backend`；脚本自动配对只作用于其环境变量。

选择任一 FlashAttention 路径都需安装 FA2。全局 latent rank 不超过 256 时使用标准 FA2；257–320 需 BF16 SM90 扩展，并设置 `SLIMKV_FA2_DECODE_EXTENSION_DIR`，详见[英文构建说明](../benchmark.md#building-the-width-320-extension)。Latent eager 没有320维上限。两条直接 latent 路径都要求 global Linear up 投影及[对应缓存/RoPE 配置](implementation_status.md)。其他投影结构使用重建路径。当前直接 latent decode 支持 Llama3，尚不支持 Qwen2。

**最终包含 chat template 的输入**上限为 31500 tokens，超过时保留前 15750 和后 15750，不做 decode/re-encode。Few-shot 和代码任务沿用原有模板规则。直接使用 Python 入口时，默认保留历史的 chat template 之前截断规则，需显式增加 `--truncate_after_chat True` 才采用新协议。

Batch 上限为 32。仅将最终 token 长度完全相同的输入合批，不 padding、不缩短额外输入、不丢样本。长度不同的样本通常以 batch 1 运行。`BATCH_SIZE` 和 `MAX_BATCH_TOKENS` 控制显存，默认 token 预算为 `BATCH_SIZE × MAX_LENGTH`，应按 GPU 和 checkpoint 调整。

## 小样本测试与续跑

评测两个子集各自的前两条：

```bash
MODEL_NAME_OR_PATH=/path/to/slimkv-f4-checkpoint \
EVAL_DATA=/path/to/longbench \
TASKS="qasper hotpotqa" MAX_SAMPLES_PER_TASK=2 RESULT_DIR=smoke \
bash scripts/evaluation/eval_longbench_slimkv.sh
```

t8f4 可设置 `BEACON_RATIO=8`，并使用独立结果目录。脚本名之后可追加评测入口参数。`bash scripts/evaluation/eval_longbench_slimkv.sh --help` 列出环境变量选项。

设置 `LOAD_RESULT=True` 可复用已完成子集及原计时记录，除 `load_result` 外，保存的配置必须一致。未完成的子集从头重跑。续跑时保持权重及数据文件不变，改变协议时使用新的 `RESULT_DIR`。旧预测文件没有计时记录时，不会补造耗时。

## 结果与耗时

默认保存到 `data/results/longbench/slimkv-t4-flash_attention_2-latent/`：

| 文件 | 内容 |
| --- | --- |
| `<task>.json` | 首行得分，后续为预测及原始样本索引 |
| `<task>.timing.json` | 推理耗时、token 数、实际 batch 分布及 GPU 峰值分配显存 |
| `task_timings.csv` | 每个已完成且有计时记录的子集一行，完成后立即更新 |
| `scores.json` | 各子集得分及不加权平均分 |
| `progress.json` | 每批更新的快照，包含当前样本进度和已完成子集 |
| `config.json`、`run_manifest.json` | 参数、源码 SHA、checkpoint config/数据 SHA、软件及硬件 |

`time_sec` 是生成阶段的墙钟时间，包括 prefill、decode、传输、输出文本解码和进度写入。模型加载与数据预处理单独记录，不属于 attention kernel 耗时。预设使用单进程、单 GPU；进度文件是保存的快照，不能单凭它判断进程仍存活。Config SHA 不等于权重 SHA，应保留每次运行使用的完整 checkpoint。

平均分只覆盖所选子集，英文预设不代表完整双语 LongBench 得分。`BEACON_RATIO` 控制 token 压缩，特征压缩取决于 checkpoint rank，两者乘积不等于 GPU 总显存缩减比例。

独立 batch 会 reset `model.memory`。QA 使用既有换行停止规则，各子集保留生成长度上限与指标。批量计算的浮点差异可能导致生成结果与 batch 1 不同。

## 其他入口

原 `eval_longbench_llama31_8B.sh` 和 Qwen 预设保留原默认配置。按长度选取的延迟测试样本使用 [fixture 工具](benchmark.md)。

| 评测 | 入口 | 输入 |
| --- | --- | --- |
| NIAH | `python -m main.eval_needle --help` | 本地 haystack、可选 judge |
| Synthetic passkey retrieval | `python -m main.eval_passkey --help` | Seed、长度和深度 |
| LM perplexity | `python -m main.eval_lm --help` | 本地数据及对应 tokenization/stride |
| 诊断 | `main.eval_longbench_beacon_probe`、`main.eval_longbench_chunk2_heatmap` | Probe 和注意力分析 |
