# SlimKV 评测

常规生成评测和 latent attention benchmark 使用不同模型路径。常规路径可以验证低秩权重的生成效果，但缓存已还原为完整维度；只有 `slimkv/latency` 的 hybrid 路径保存 latent Beacon K/V。

## LongBench

```bash
MODEL_NAME_OR_PATH=/path/to/checkpoint-2000 \
EVAL_DATA=/path/to/long-llm/longbench \
CUDA_VISIBLE_DEVICES=0 NPROC_PER_NODE=1 \
BEACON_RATIO=8 \
bash scripts/evaluation/eval_longbench_llama31_8B.sh
```

通用入口为 `python -m main.eval_longbench --help`。独立请求之间必须调用 `model.memory.reset()`；评测入口会为每个 batch 重置。输出保存在 `data/results/`，每个任务文件包含首行分数和后续预测记录。

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
