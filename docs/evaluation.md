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
