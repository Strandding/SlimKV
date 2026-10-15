# 后端支持范围

[English](../implementation_status.md)

下表说明当前缓存与注意力计算路径。配置及权重的 `beacon_*` 接口保持兼容。

| 路径 | Beacon持久缓存 | 注意力计算 |
|---|---|---|
| 兼容Llama3 auto/latent，默认reconstruct |真实低秩维度|prefill原始维度，decode重建|
| Llama3 latent + fa2 |全局低秩|prefill原始维度，decode raw/sink+latent两支LSE合并|
| Llama3 latent缓存 + latent decode策略 |全局低秩|prefill原始维度，直接Beacon decode按attn_impl选择eager或FA2|
| Llama训练/full/不兼容auto；Qwen2 |完整维度|标准注意力|

Auto仅在支持的推理配置启用latent，不支持时回full；显式latent报错。训练计算和梯度保持原路径。Raw tail和sink始终完整维度。

Latent条件：interleave、window=stride、单窗口编码、正ratio、K/V低秩，排除chunk-fixed K RoPE、K residual、beacon_skip。直接eager和FA2 decode都额外要求full-coverage、attend_prev、global Linear up、Beacon K无RoPE、raw K有RoPE、无latent relative-position bias。

符合条件decode每序列一个新raw token；插Beacon时多一个query。支持等长无padding、同步插入计划的dense batch，各行KV独立。多raw token调用回重建；直接latent decode明确拒绝padding，不是连续批处理。

Global coupled rank缓存形状[B,1,N,R]。FA2在rank超过256时使用BF16 SM90/SM120扩展，使用320维 FA2 缓冲区，导出的缓存视图保留真实 rank。静态解码的缓存视图与补齐后的 FA2 缓冲区共用存储，padding 和预留容量计入实际显存。Eager直接按真实K/V rank计算，不补齐维度，也没有320维上限。Raw attention保留原始head维度和缩放系数，Llama-3.1-8B为128和1/sqrt128。K/V列表独立。

Fast 默认 True，在符合条件的 FA2 解码中使用 GPU token bookkeeping。Python API 的 Graph 默认 False，CUDA 推理和 LongBench 预设则开启整模型 B1 Graph，支持原位追加 KV 和插入 Beacon 的步骤。真实 rank 的 Beacon 视图与补齐后的 FA2 缓冲区共用存储。设置 `beacon_decode_graph_full=False` 可保留局部 Norm/MLP Graph。不支持的调用使用普通路径，dtype/device 变化时清除 Graph 缓存。

`main.conversion.convert_kv_lowrank` 已实现等能量阈值、秩下限和剩余预算分配，K/V 预算独立，秩按 KV head 粒度对齐。用法见[自动初始化](training.md#自动等能量秩分配)。

## 已知问题和未完成项

- Qwen无新Beacon时历史K的RoPE问题仍未修；Llama对应路径已修并有回归。
- 独立 latency 原型的 FA2 fallback 可能丢 causal mask，Value bias 缺概率质量系数；该原型应使用不带 bias 的 Value up-projection。
- 历史benchmark warmup/repeat未reset，reconstruct标签实际为完整cache，不等于论文latent-cache Recon。
- Qwen latent cache 和 Qwen3 MoE 仍不支持。
- 全量LongBench、多seed NIAH judge、随机字符/attention统计、shared-context多请求和论文A100完整效率协议未贯通。
- vLLM仅基线wrapper，paged/continuous batching未实现；Graph工作区、动态KV拼接和host gaps仍需优化。

[复现缺口](reproducibility.md)区分缺实现、缺数据和缺公开资源。
