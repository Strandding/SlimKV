# 论文复现

[English](../reproducibility.md)

下面列出 SlimKV 初始化、训练与评测入口。完整复现还需要最终 checkpoint、处理后的数据 manifests 和各项结果的实验配置。

论文效率协议使用 A100 80GB、Llama-3.1-8B、B1/BF16/eager、32K–128K 上下文和128个生成 token，预热后取三轮耗时中位数，Full KV 使用合成缓存。仓库 FA2 工具提供另一种推理性能对照，报告结果时需记录相应后端与协议。

| 阶段 | 现有入口 | 缺口 |
|---|---|---|
| SVD/rank |main.conversion.convert_kv_lowrank|精确 rank 列表和初始化 revision；自动等能量、秩下限与预算分配已实现|
| 预训练 |main.training.train及presets|0.5B处理后数据manifest、schedule/seed/environment|
| 微调 |同Trainer|0.2B混合/GPT/needle数据、恢复状态策略|
| LongBench |main.evaluation.eval_longbench|各模型比例最终权重和全量结果/config/task set|
| NIAH |main.evaluation.eval_needle|论文三seed/judge/aggregation协议|
| 效率 |prototype和tools|A100 eager synthetic Full KV/Recon/LSA统一harness、attention-only计时|
| 多请求/ablation/MoE |probes/archives|最终protocol；Qwen3 MoE代码|

Train和LongBench写run_manifest.json；性能工具另存rawtiming、inputs、warmup/order和显存。记录commit还不够，dirty源码必须保留SHA。通用helper：

```bash
python -m tools.record_run \
  --output data/results/my-run/run_manifest.json \
  --config /path/to/resolved-config.json \
  --inputs /path/to/checkpoint/config.json /path/to/samples.jsonl
```

工具不启动实验或改变seed。Credential字段脱敏，不记录raw argv/全环境；不自动hash大权重，须另存exact shards/SHA。私有manifest可能带本地路径，发布前检查。

保持data选择顺序、模板、ratio、window/stride、K/V rank、RoPE、dtype/backend、EOS和GPU一致。失败日志保留，但不计为成功benchmark；不要删慢trial美化结果。

完整实验设置见[论文](https://arxiv.org/abs/2610.02953)。完整复现发布还需要：可分发权重/tokenizer/hash与基座来源、处理后的训练数据和seed manifests、最终 rank 列表、分配参数及基座 revision、完整评测与efficiency/judge/ablation资产。当前缓存条件与实现缺口见[后端支持](implementation_status.md)。
