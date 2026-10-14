# 论文复现

[English](../reproducibility.md)

仓库提供可运行阶段及局部证据，**尚不能只靠公开资产完整复现论文**。

论文LongBench表来自作者稿件；A100效率协议为B1/BF16/eager、32K–128K、128生成token、warmup加三次median、synthetic Full KV。H20局部FA2数据是另一协议：8样本quality、2fixture fixed-input对照，以及B1–32复制prompt的dense scaling。

稿件abstract提到最高3.38× E2E，但Table6在128K列含t4f4的3.38×和t8f4的3.49×。README明确引用t4f4，未替作者修改论文；发表前应核对。

| 阶段 | 现有入口 | 缺口 |
|---|---|---|
| SVD/rank |main.convert_kv_lowrank|精确rank列表和初始化revision；自动分配未实现|
| 预训练 |main.train及presets|0.5B处理后数据manifest、schedule/seed/environment|
| 微调 |同Trainer|0.2B混合/GPT/needle数据、恢复状态策略|
| LongBench |main.eval_longbench|各模型比例最终权重和全量结果/config/task set|
| NIAH |main.eval_needle|论文三seed/judge/aggregation协议|
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

作者仍需提供公开论文链接/发表信息、原创许可证、可分发权重/tokenizer/hash和基座来源、处理数据与seed manifests、最终rank及自动算法、完整评测/efficiency/judge/ablation资产。Figure3已从稿件提取，但不是功能完成声明。历史记录有[中文](../archive/validation_20261014_zh.md)与[英文](../archive/validation_20261014.md)，当前契约见[状态](implementation_status.md)。
