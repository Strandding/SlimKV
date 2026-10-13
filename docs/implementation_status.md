# SlimKV 实现状态

主实现提供 Beacon 分块压缩、Llama3/Qwen2 低秩 K/V 投影、逐层 rank 配置、SVD 初始化、冻结 Backbone 的训练和常规长上下文评测。按作者确认的实际状态，论文中的 LSA 尚未实现为正式推理后端；当前推理使用低秩 K/V 重建回原始维度的常规路径。`slimkv/latency` 是独立实验代码，不作为已交付或已验收的 LSA 实现。

## 计算路径

| 路径 | Beacon 缓存 | 注意力 | 支持模型 |
|---|---|---|---|
| `slimkv/llama3` 兼容的低秩推理配置 | 低维 Beacon K/V，raw/sink 完整维度 | 重建后标准注意力 | Llama3 |
| `slimkv/llama3` 训练、full 模式、不兼容配置；`slimkv/qwen2` | 完整维度 K/V | 标准注意力 | Llama3、Qwen2 |
| `slimkv/latency` | 全局 latent K/V | raw/Beacon logits 统一 softmax | Llama eager 原型 |

`beacon_*` 字段和权重键保留，兼容已有 checkpoint。项目名称和 Python 包名为 SlimKV；Beacon 是算法术语。Headwise/grouped up projection 可用于部分常规实验，但不能直接映射到现有 hybrid loader。

Llama3 新增 `beacon_cache_mode=auto/full/latent`。auto 是默认模式，在支持的推理配置中缓存低维 Beacon K/V；full 保留此前完整缓存；latent 显式要求低维存储。训练保持此前数学计算、完整缓存和梯度路径。历史 Beacon 与未完成窗口中的 Beacon 均保存低维表示，读取时逐层重建；生成接口返回的缓存引用也不保留完整重建结果。

当前支持 interleave、window=stride、单窗口编码、正 token 压缩比、K/V 均为低秩，并排除 chunk-fixed K RoPE、K positional residual 和 beacon_skip。auto 对不支持的配置保留完整缓存；显式 latent 会报错。Qwen2 仍保留完整缓存。

f4 权重的 Beacon K/V 特征维度平均缩小四倍，但 raw/sink、重建工作区和其他模型开销仍存在，整体显存不会直接缩小四倍。注意力仍在原始 head dim 上执行，属于重建推理，LSA 尚未交付。

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

## 低维缓存与重建验证（2026-10-14）

新增 10 项实际模型 CPU 回归，加上原有 8 项 RoPE 回归，本机和 H20 的 18 项均通过。覆盖不同逐层 K/V rank、global/headwise/grouped up projection、共享 down projection、带偏置投影、eager/SDPA、当前窗口 Beacon、短 prefill 和跨窗口 decode、缓存释放、重置与生成、checkpoint 权重兼容、native/full fallback，以及训练输出、loss 和梯度与原始 Memory 逐位一致。

在 H20 上复用上一节相同的四条 LongBench-v1 样本、BF16 权重与生成设置，用同一个模型对象分别运行 full 和 latent 缓存模式。两种模式的全部生成 token 一致；实际保存的 50 个生成步骤、每步 128256 个 logits 在本机逐位核对全部一致，最大绝对差为 0。full 模式的生成和 prefill 首步 logits 也与此前提交的记录一致。该结果只覆盖这些输入和当前 eager 环境，不是所有硬件及形状的逐位一致保证。

| 任务 / 原始索引 | full 持久 K/V（MiB） | latent 持久 K/V（MiB） | 字节减少 |
|---|---:|---:|---:|
| Qasper / 0 | 184.000 | 136.750 | 25.68% |
| Qasper / 2 | 104.000 | 52.719 | 49.31% |
| HotpotQA / 16 | 103.625 | 31.062 | 70.02% |
| HotpotQA / 22 | 83.875 | 23.594 | 71.87% |

表中是生成结束时 Memory 中的逻辑 K/V 字节数，包括 sink、raw、历史 Beacon 和当前窗口 Beacon，不是 GPU 总显存或峰值。四条样本的 Beacon K/V 部分均精确缩小四倍；raw/sink 维度不变。latent 模式的实际底层 K/V storage 字节数等于表中逻辑字节数，确认没有视图意外保留完整投影的 backing storage；生成返回缓存仅引用这些持久表示。两种模式注意力计算的 head dim 始终为 128，抽查第 0、31 层的无新 Beacon decode，历史 K 变化量保持为 0。

H20 证据位于忽略目录 `data/results/latent_cache_validation_20261014/`；本机对应目录 `/data/workspace/slimkv_analysis/latent_cache_validation_20261014/` 保存原始 logits tensor、18 个文件的 SHA receipt 和 `independent_audit.json`。GPU 运行正常退出，原占卡程序已恢复。FA2 已接入读取与重建逻辑，但本次只做 eager GPU 验证，SDPA 为 CPU 验证；LSA 与性能 benchmark 不在此次验收范围内。

## 重建推理耗时、实际显存与 LongBench-v1 扩展样本（2026-10-14）

`tools/benchmark_reconstruction.py` 在同一 H20、同一 f4 BF16 模型、eager、token ratio 8 下测量两条输入。每次重置 Memory，三种模式各预热一次，再做两轮对称 full → full_compact → latent → latent → full_compact → full；每种模式共四次计时，表中为中位数。性能阶段固定生成 64 token，首 token 来自 prefill，63 次 decode 不按 EOS 提前停止；tokenization、评分、文件 IO 和 logits 保存均在计时外。

标准 full 的 sink 切片会保留首窗口完整 K/V backing storage；full_compact 仅在工具中克隆首个 sink，消除这部分额外占用。三种模式在全部预热和计时调用中生成 token 相同。新增两项 CPU 控制回归通过，总计 20 项 CPU 回归通过，H20 同样通过两项新增控制回归。主模型算法未因本次测量改变。

| 实际输入 token | full_compact prefill（ms） | latent prefill（ms） | full_compact decode（ms/token） | latent decode（ms/token） |
|---|---:|---:|---:|---:|
| 4009 | 852.75 | 873.86 | 27.67 | 43.26 |
| 6194 | 1373.87 | 1406.46 | 27.50 | 42.55 |

重建模式的 prefill 耗时增加 2.48% / 2.37%，decode 每 token 耗时增加 56.30% / 54.75%；decode 吞吐从 36.13 / 36.36 降到 23.12 / 23.50 token/s。该后端当前提供空间节省，并未带来解码加速。

| 实际输入 token | 标准 full / full_compact / latent 的 prefill 后持久 K/V storage（MiB） | full_compact / latent 的峰值 CUDA allocated（GiB） |
|---|---|---|
| 4009 | 323.750 / 179.875 / 132.906 | 21.883 / 21.836 |
| 6194 | 247.000 / 103.125 / 30.563 | 21.992 / 21.850 |

持久 K/V 为 Memory 中去重后的实际底层 storage，峰值显存还包含模型、buffers、logits 和注意力工作区。相对 full_compact，峰值只减少 48.125 MiB（0.215%）/ 145.496 MiB（0.646%）；不能把 Beacon 维度的四倍节省解释成 GPU 总显存四倍节省。CUDA allocator reserved 字节未作为存活显存计入比较。

质量部分扩展为 Qasper、HotpotQA、MultiFieldQA-en、LCC 各两条，共八条 LongBench-v1 官方数据；选取原始 `length` 在 2500–6000 的前两条，实际输入 4009–11437 token，未截断。使用官方任务模板和生成上限、现有 Llama-3 模板及 QA 换行停止规则，LCC 不加 chat template、不使用 QA 换行 EOS。QA 用 F1，LCC 用首个有效代码行相似度，独立使用官方 metrics 复核。

| 任务（各两条） | full | latent |
|---|---:|---:|
| Qasper | 21.88 | 21.88 |
| HotpotQA | 100.00 | 100.00 |
| MultiFieldQA-en | 40.28 | 40.28 |
| LCC | 75.00 | 75.00 |

八条生成 token 全部一致；这是按长度选取的四个任务小样本，不代表完整 LongBench-v1 得分。完整 raw timing、显存、预测、输入 SHA、源码版本、14 文件 SHA receipt 和独立审核在本机 `/data/workspace/slimkv_analysis/longbench_perf_20261014/`，H20 对应 `data/results/longbench_perf_20261014/`。运行正常退出并恢复原占卡。

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

H20 的 `model/` 下保留 Llama-3.1-8B 的逐层 f4 `checkpoint-2000`。32 层 K/V 列表各自平均为 32，原始 head dim 为 128；权重为 BF16，Beacon 参数约 621M。该配置的常规 Llama3 推理现在默认启用低维 Beacon 缓存与原始维度重建；原始权重文件和配置文件保持不变。

该权重、论文、历史预测和图作为本地产物保留。历史文件不能自动代表论文最终实验结果，应通过 checkpoint、有效配置、样本集和实际代码版本追溯。
