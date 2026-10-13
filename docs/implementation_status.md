# SlimKV 实现状态

主实现提供 Beacon 分块压缩、Llama3/Qwen2 低秩 K/V 投影、逐层 rank 配置、SVD 初始化、冻结 Backbone 的训练和常规长上下文评测。按作者确认的实际状态，论文中的 LSA 尚未实现为正式推理后端；当前推理使用低秩 K/V 重建回原始维度的常规路径。`slimkv/latency` 是独立实验代码，不作为已交付或已验收的 LSA 实现。

## 计算路径

| 路径 | Beacon 缓存 | 注意力 | 支持模型 |
|---|---|---|---|
| `slimkv/llama3`、`slimkv/qwen2` | 完整维度 K/V | 标准 eager/SDPA/FA2 | Llama3、Qwen2 |
| `slimkv/latency` | 全局 latent K/V | raw/Beacon logits 统一 softmax | Llama eager 原型 |

`beacon_*` 字段和权重键保留，兼容已有 checkpoint。项目名称和 Python 包名为 SlimKV；Beacon 是算法术语。Headwise/grouped up projection 可用于部分常规实验，但不能直接映射到现有 hybrid loader。

当前常规路径在缓存前完成重建，因此 Beacon K/V 缓存的最后一维仍为原始 head dim。特征压缩四倍描述低秩投影维度，不代表此路径的持久 K/V 缓存节省四倍；token 维度的 Beacon 压缩仍然生效。

## 已知问题

Llama 常规路径已修复无新 Beacon 时的历史 K RoPE 处理。CPU 回归验证覆盖纯 raw、包含新 Beacon、训练最后窗口、prefill 尾块、decode 和 chunk-fixed 历史缓存；已有 BF16 权重的小规模生成验证已完成，完整质量评测仍待开展。

## 实际权重的小规模验证（2026-10-14）

在 H20 上使用下述 f4 `checkpoint-2000`，比较修复前 `eeaca5e` 和修复后 `c8c8935` 两份独立源码。推理采用常规完整维度重建路径、BF16、eager、Llama-3 chat template、greedy；token 压缩比固定为 8，window/stride 保持 1024，每条样本前重置 Memory。逐层特征维度和权重不变。

从 LongBench-v1 官方 `THUDM/LongBench` 的 Qasper 和 HotpotQA 中各取两条：按原始顺序选择数据集 `length` 在 2500–6000 的前两条，不截断上下文。QA 使用各任务默认生成上限（128/32）和现有评测的换行停止规则。

| 任务 / 原始索引 | 实际输入 token | 修复前 QA F1 | 修复后 QA F1 |
|---|---:|---:|---:|
| Qasper / 0 | 4009 | 34.48 | 31.25 |
| Qasper / 2 | 4370 | 12.50 | 12.50 |
| HotpotQA / 16 | 6194 | 100.00 | 100.00 |
| HotpotQA / 22 | 5145 | 100.00 | 100.00 |

四条 QA F1 均值为 61.75 → 60.94；三条生成 token 完全一致，一条措辞改变。本次没有观察到质量提升，四条按长度选取的样本也不能代表完整 LongBench 分数。

四条输入在修复前后的 prefill 首步 logits 完全一致。全部 32 层均实际经过无新 Beacon 的 decode 分支；抽查第 0、31 层时，旧代码返回的历史 Beacon K 相对未旋转 K 的最大绝对差为 3.65625–11.96875，新代码均为 0。全部 32 层 Beacon K/V 缓存最后一维仍为 128，确认实际采用重建后的完整维度缓存。这些样本没有覆盖无新 Beacon 的 prefill 尾块，相关验证来自上述 CPU 回归。

H20 原始样本、参数、输出、诊断和日志保存在本地忽略目录 `data/results/longbench_v1_smoke_20261014/`；本机对应记录和独立评分审核保存在 `/data/workspace/slimkv_analysis/longbench_v1_smoke_20261014/`。运行正常退出，原占卡程序已恢复。本次记录不用于效率结论。

## 待处理问题

以下问题仍待处理：

- 常规 Qwen 路径在本步 `beacon_size<=0` 时会给历史 Beacon K 加 RoPE，与关闭 Beacon K RoPE 的配置不一致。
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
