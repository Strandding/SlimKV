# SlimKV 评测

常规生成评测采用标准注意力。Llama3 可以保存低维 Beacon K/V，在 prefill 和 decode 的注意力计算前按层重建；Qwen2 目前仍保存完整维度 K/V。独立的 `slimkv/latency` 实验路径不作为已交付的论文 LSA 后端。

## LongBench

```bash
MODEL_NAME_OR_PATH=/path/to/checkpoint-2000 \
EVAL_DATA=/path/to/long-llm/longbench \
CUDA_VISIBLE_DEVICES=0 NPROC_PER_NODE=1 \
BEACON_RATIO=8 \
bash scripts/evaluation/eval_longbench_llama31_8B.sh
```

通用入口为 `python -m main.eval_longbench --help`。独立请求之间必须调用 `model.memory.reset()`；评测入口会为每个 batch 重置。输出保存在 `data/results/`，每个任务文件包含首行分数和后续预测记录。

Llama3 常规入口新增 `--beacon_cache_mode`：默认 `auto` 在兼容配置的推理中保存低维 Beacon K/V；`full` 用于完整缓存对照；`latent` 显式要求低维缓存，配置不兼容时会报错。当前低维存储支持 interleave、window=stride、单窗口编码、正 token 压缩比、K/V 均为低秩，且不启用 chunk-fixed K RoPE、K positional residual 或 beacon_skip。训练始终保留完整缓存；切换训练状态或模式前先重置 Memory。验证使用 greedy 生成。

同一权重的对照可在调用 `main.eval_longbench` 时分别传入 `--beacon_cache_mode full` 和 `--beacon_cache_mode latent`，保持其他参数一致。小规模评测可以设置 `--tasks qasper hotpotqa --max_samples_per_task 2`。当前 H20 验证显式使用 `--attn_impl eager`；SDPA 已做 CPU 回归，FA2 尚未做此次 GPU 验收。

`BEACON_RATIO` 是 token 压缩比例，不是低秩特征压缩比例。平均 f4 checkpoint 配合 token ratio 4/8 分别对应名义 t4f4/t8f4；raw tail 和 sink 会使实际压缩比低于名义值。

## 重建推理的耗时、显存与小样本 LongBench-v1

```bash
python -m tools.benchmark_reconstruction \
  --model_name_or_path /path/to/checkpoint-2000 \
  --samples_jsonl /path/to/samples.jsonl \
  --output_dir data/results/reconstruction-benchmark \
  --attn_impl eager --beacon_ratio 8 \
  --decode_tokens 64 --warmup 1 --rounds 2 \
  --perf_sample_indices 0 2
```

输入 JSONL 使用 LongBench-v1 原始字段，另加 `task`（任务名）和 `source_index`（原始行号）。工具采用现有 LongBench 模板与 Llama-3 chat template，代码补全不加 chat template。质量评测使用任务默认生成上限、greedy 和现有 QA 换行停止规则；输出包含每条预测、生成 token 和各任务评分。

耗时测量固定生成长度，不按 EOS 提前停止，首 token 来自 prefill，其余 token 是 decode；这部分不用于质量评分。每次调用前重置 Memory，预热后以 full → full_compact → latent → latent → full_compact → full 的对称顺序测量，每轮每种模式各两次。使用同步后的 host wall time，报告模型 prefill、首 token 延迟及 decode 每 token 耗时；tokenization、评分、文件 IO 和 logits 保存不计入计时。

full_compact 是对照工具中释放 sink 切片多余 backing storage 的完整缓存模式，值和注意力数学不变；标准 full 模式仍按仓库原路径运行。工具同时记录 CUDA live allocated/peak allocated、Memory 逻辑 K/V 字节和去重后的底层 storage 字节，不能把缓存字节节省当作 GPU 总显存节省。allocator reserved 字节不参与显存对比。

`--disable_optional_flash_import` 只用于 eager/SDPA 验证环境中存在 FlashAttention 元信息但缺少扩展的情况；它在工具内关闭未使用的可选导入，不修改模型源码。

## NIAH 与其他任务

```bash
python -m main.eval_needle --help
python -m main.eval_passkey --help
python -m main.eval_infbench --help
python -m main.eval_lm --help
```

NIAH 默认使用 ROUGE，`--gpt_eval` 需要可选 judge 依赖与环境中的 API key。当前 judge 和随机种子流程不是论文最终实验协议，见 [实现状态](implementation_status.md)。

## Latent attention 原型

```bash
python -m main.benchmark_latency \
  --lowrank_model_or_ckpt /path/to/checkpoint-2000 \
  --mode beacon_lowrank_hybrid_latent \
  --attn_impl eager --beacon_ratio 8 \
  --prompt_len 8192 --gen_len 128 --repeat 1 --warmup 0 \
  --output_json /tmp/slimkv-latency.json
```

该入口目前用于诊断原型。已有 benchmark 存在重复试验 memory 累积问题，且 Recon 基线与论文的 latent-cache 重建基线不同；上述单次调用不能替代论文速度表。Hybrid 必须使用 eager，FA2 fallback 的 mask 问题尚未修复。

热力图与 probe 入口位于 `main/eval_longbench_chunk2_heatmap.py` 和 `main/eval_longbench_beacon_probe.py`；对应启动脚本位于 `scripts/evaluation/`。图和历史预测是本地产物，不进入新提交。

## vLLM 基线

`scripts/vllm/` 用于原始模型基线，模型配置在 `config/vllm_models.sh`。它没有 SlimKV 的 paged latent cache 或服务端模型实现，不能作为 SlimKV 已接入 vLLM 的证据。
