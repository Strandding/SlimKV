# 实现状态

[English](../implementation_status.md)

当前是研究实现，只验证论文设计的一部分。配置及权重的`beacon_*`接口保持兼容，本轮不改模型数学。

| 路径 | Beacon持久缓存 | 注意力计算 |
|---|---|---|
| 兼容Llama3 auto/latent，默认reconstruct |真实低秩维度|prefill原始维度，decode重建|
| Llama3 latent + fa2 |全局低秩|prefill原始维度，decode raw/sink+latent两支LSE合并|
| Llama训练/full/不兼容auto；Qwen2 |完整维度|标准注意力|
| slimkv/latency |原型自己的latent格式|独立eager hybrid|
| anchor_prototype |历史anchor格式|不参与主安装，不兼容Beacon checkpoint|

Auto仅在支持的推理配置启用latent，不支持时回full；显式latent报错。训练计算和梯度保持原路径。Raw tail和sink始终完整维度。

Latent条件：interleave、window=stride、单窗口编码、正ratio、K/V低秩，排除chunk-fixed K RoPE、K residual、beacon_skip。FA2额外要求full-coverage、attend_prev、global Linear up、Beacon K无RoPE、raw K有RoPE、无latent relative-position bias。

符合条件decode每序列一个新raw token；插Beacon时多一个query。支持等长无padding、同步插入计划的dense batch，各行KV独立。多raw token调用回重建；latent FA2 padding明确拒绝，不是连续批处理。

Global coupled rank缓存形状[B,1,N,R]，私有f4最大K/V rank304/312；临时补到320计算，缓存不padding；raw head dim128，scale1/sqrt128。K/V列表独立。

Fast默认True，仅eligible FA2；Graph默认False，只捕获norm及attention之后MLP/residual，Q1/Q2和B分开缓存。Attention/KV更新仍eager；CPU/梯度/诊断fallback，dtype/device变化清图。

已做CPU历史RoPE、cache/full-latent、训练loss/梯度、checkpoint、causal/rollover/distinct-row等回归；H20 f4 BF16小样本、融合/Graph/320kernel/memcheck有记录。8条LongBench和2条计时不证明全量质量、128K、多模型或生产可用。[证据与历史日志](../archive/validation_20261014_zh.md)。

## 已知问题和未完成项

- Qwen无新Beacon时历史K的RoPE问题仍未修；Llama对应路径已修并有回归。
- 独立latency原型的FA2 fallback可能丢causal mask、Value bias缺概率质量系数；验证Llama f4无该bias。
- 历史benchmark warmup/repeat未reset，reconstruct标签实际为完整cache，不等于论文latent-cache Recon。
- 自动equal-energy+rank-floor+budget分配缺失；Qwen latent cache和Qwen3 MoE不支持。
- 全量LongBench、多seed NIAH judge、随机字符/attention统计、shared-context多请求和论文A100完整效率协议未贯通。
- vLLM仅基线wrapper，paged/continuous batching未实现；Graph工作区、动态KV拼接和host gaps仍需优化。

[复现缺口](reproducibility.md)区分缺实现、缺数据和缺公开资源。
