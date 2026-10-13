# SlimKV 实现状态

主实现提供 Beacon 分块压缩、Llama3/Qwen2 低秩 K/V 投影、逐层 rank 配置、SVD 初始化、冻结 Backbone 的训练和常规长上下文评测。Llama 的 latent cache 与 hybrid attention 是独立 eager 原型，尚未接入所有评测入口或 vLLM。

## 计算路径

| 路径 | Beacon 缓存 | 注意力 | 支持模型 |
|---|---|---|---|
| `slimkv/llama3`、`slimkv/qwen2` | 完整维度 K/V | 标准 eager/SDPA/FA2 | Llama3、Qwen2 |
| `slimkv/latency` | 全局 latent K/V | raw/Beacon logits 统一 softmax | Llama eager 原型 |

`beacon_*` 字段和权重键保留，兼容已有 checkpoint。项目名称和 Python 包名为 SlimKV；Beacon 是算法术语。Headwise/grouped up projection 可用于部分常规实验，但不能直接映射到现有 hybrid loader。

## 已知问题

以下问题来自源码检查与小型 FP32 CPU 输入验证，尚未在本次目录整理中改变算法行为：

- 常规 Llama/Qwen 路径在本步 `beacon_size<=0` 时会给历史 Beacon K 加 RoPE，与关闭 Beacon K RoPE 的配置不一致。
- Hybrid 的 FA2 fallback 回到 eager 时，Memory 可能已经移除了因果 mask。请显式使用 eager。
- Latency benchmark 的 warmup/repeat 之间没有重置 Memory，连续调用会累积上下文。
- `beacon_lowrank_reconstruct` 使用完整维度缓存，不是论文中每步从 latent cache 重建历史 K/V 的基线。
- Hybrid Value up-projection 带 bias 时，bias 缺少 Beacon attention 概率质量系数。当前无 bias 的 Llama checkpoint 不受此项直接影响。

## 尚未贯通的功能

- 论文的等能量阈值、rank floor 和剩余预算自动分配；目前需要外部提供 rank 列表。
- Qwen latent-cache attention、Qwen3 MoE 支持。
- 统一 latent backend 的 LongBench/NIAH 与完整模型等价性验收。
- 最终三 seed NIAH judge、随机字符重建、attention 相似性统计和多问题共享上下文实验。
- 论文完整效率协议：attention 单独计时、synthetic Full KV、latent-cache Recon、独立重复和 median。
- SlimKV 的 vLLM 模型、paged cache 和连续批处理。

## 远端已有权重

H20 的 `model/` 下保留 Llama-3.1-8B 的逐层 f4 `checkpoint-2000`。32 层 K/V 列表各自平均为 32，原始 head dim 为 128；权重为 BF16，Beacon 参数约 621M。常规加载不会自动变成 latent-cache 推理。

该权重、论文、历史预测和图作为本地产物保留。历史文件不能自动代表论文最终实验结果，应通过 checkpoint、有效配置、样本集和实际代码版本追溯。
