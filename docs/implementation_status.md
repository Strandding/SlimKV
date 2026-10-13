# SlimKV 实现状态

主实现提供 Beacon 分块压缩、Llama3/Qwen2 低秩 K/V 投影、逐层 rank 配置、SVD 初始化、冻结 Backbone 的训练和常规长上下文评测。Llama3 默认推理使用低秩 K/V 重建回原始维度；新增可选 FA2 latent decode 后端，已做 H20 f4 权重的小规模验证。完整论文 LSA 的多模型、质量与效率协议仍待完成。`slimkv/latency` 保留独立实验代码。

## 计算路径

| 路径 | Beacon 缓存 | 注意力 | 支持模型 |
|---|---|---|---|
| `slimkv/llama3` 兼容的低秩推理配置 | 低维 Beacon K/V，raw/sink 完整维度 | 重建后标准注意力 | Llama3 |
| `slimkv/llama3` + `beacon_decode_backend=fa2` | 全局 latent Beacon K/V，raw/sink 完整维度 | prefill 重建；decode 用 raw/latent 两支 FA2 与 LSE 合并 | Llama3，H20 BF16 已验证 |
| `slimkv/llama3` 训练、full 模式、不兼容配置；`slimkv/qwen2` | 完整维度 K/V | 标准注意力 | Llama3、Qwen2 |
| `slimkv/latency` | 全局 latent K/V | raw/Beacon logits 统一 softmax | Llama eager 原型 |

`beacon_*` 字段和权重键保留，兼容已有 checkpoint。项目名称和 Python 包名为 SlimKV；Beacon 是算法术语。Headwise/grouped up projection 可用于部分常规实验，但不能直接映射到现有 hybrid loader。

Llama3 新增 `beacon_cache_mode=auto/full/latent`。auto 是默认模式，在支持的推理配置中缓存低维 Beacon K/V；full 保留此前完整缓存；latent 显式要求低维存储。训练保持此前数学计算、完整缓存和梯度路径。历史 Beacon 与未完成窗口中的 Beacon 均保存低维表示，读取时逐层重建；生成接口返回的缓存引用也不保留完整重建结果。

当前支持 interleave、window=stride、单窗口编码、正 token 压缩比、K/V 均为低秩，并排除 chunk-fixed K RoPE、K positional residual 和 beacon_skip。auto 对不支持的配置保留完整缓存；显式 latent 会报错。Qwen2 仍保留完整缓存。

f4 权重的 Beacon K/V 特征维度平均缩小四倍，但 raw/sink、工作区和其他模型开销仍存在，整体显存不会直接缩小四倍。默认重建路径的注意力在原始 head dim 上执行；可选 FA2 后端在 decode 中使用全局 latent 维度。

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


## 可选 FA2 latent decode：两轮实现与验证（2026-10-14）

新增 `beacon_decode_backend=fa2`，默认保持 reconstruct。实际 decode 时，raw/sink 使用原始 128 维 FA2；Beacon 使用全头耦合的 global latent attention。按 GQA 分组把旋转后的 Q 投入 U_K 的各 head 切片；latent PV 再经 U_V 切片投回输出维度。两支的 LSE 合并成同一个 softmax，scale 保持 1/sqrt(128)。新 Beacon 直接 down-project 后加入 cache，不重建历史或当前 Beacon K/V。prefill 和训练保留原路径。

f4 checkpoint 每层 K/V rank 不同，全局最大为 304/312。预编译 FA2 2.8.3 处理不超过 256 维的层；`kernels/fa2_decode` 基于官方 FA2 v2.8.3/CUTLASS，仅补编译 BF16、SM90、320 维的 forward split-KV 与 combine kernels。临时计算补零到同一宽度，持久缓存保存实际 rank，形状为 `[B,1,N,R]`。H20 独立环境为 Torch 2.11.0+cu129、Transformers 4.40.2，复用已有 FA2 二进制；与此前 Torch 2.13 的性能记录分开比较。

支持范围：Llama、global Linear up projection、full-coverage、attend_prev、interleave、window=stride、正 token ratio；Beacon K 不加 RoPE，raw K 加 RoPE，无 chunk-fixed/K residual/latent relative-position bias。GPU 验证为 BF16、无 padding、batch=1、每步一个原始 token，新 Beacon 步为两个 query。其 raw 分支可见全部 raw，latent 分支使用 bottom-right causal mask。遇到 FA2 整行 mask 的 +inf LSE 约定时，合并前转换为零概率质量。训练、批量和多 raw token 调用保留重建路径。

第一轮先贯通两支 attention；第二轮省掉逐层 CUDA `max().item()` RoPE 同步和无需 padding 的张量拷贝。25 项实际模型 CPU 测试在 H20 通过，包含 FP32 global/GQA/bias/shared down、短 prefill、首次 Beacon、跨窗口、新 Beacon 因果关系和默认训练回归。320 维 kernel 的 148 个 GPU cases 覆盖 query=1/2、因果/非因果、batch=1/2、GQA、边界长度、单/自动/16 splits；对独立 FP32 reference 最大输出绝对误差为 0.002335，四个实际 compute-sanitizer memcheck cases 为 0 errors。本机当前临时 Python 环境缺 Torch，CPU 模型测试在 H20 执行；本机完成源码检查和独立数据/评分审核。

性能采用同一个模型，full_compact/reconstruct/fa2 各一次预热；两轮对称 full_compact → reconstruct → fa2 → fa2 → reconstruct → full_compact，每模式四次取中位。首 token 由 prefill 产生，63 次 decode 固定输入为 full_compact 预热产生的 token；诊断、tokenization、评分与 IO 均在同步计时外。

| 实际输入 token | full_compact decode（ms/token） | reconstruct decode（ms/token） | FA2 第一轮 / 第二轮（ms/token） |
|---|---:|---:|---:|
| 4009 | 26.572 | 41.534 | 32.048 / 29.770 |
| 6194 | 26.405 | 41.356 | 31.929 / 29.426 |

第二轮 FA2 相对重建路径减少 decode 耗时 28.32% / 28.85%，但仍比 full_compact 慢 12.03% / 11.44%；当前没有达到比完整缓存更快的结果。第一轮使用 31 次 decode、每模式两次，第二轮使用 63 次、每模式四次；两轮的跨运行数字仅展示迭代趋势，正式收益来自第二轮同运行对照。

prefill 的 full_compact/reconstruct/fa2 中位数为 851.60/869.29/865.48 ms（4009 token）与 1368.01/1401.62/1392.87 ms（6194 token）。FA2 和 reconstruct 的 prefill 后持久 K/V storage 相同，为 132.906/30.563 MiB；full_compact 为 179.875/103.125 MiB。Beacon K/V 部分仍精确压缩四倍。两种低维模式的整次调用峰值相同，受原始维度 prefill 支配；不能把 decode 算法变化解释为整体显存进一步四倍下降。

LongBench-v1 复用此前八条原始样本和同一生成配置，两个版本均得到：

| 任务（各两条） | reconstruct | FA2 |
|---|---:|---:|
| Qasper | 21.88 | 21.88 |
| HotpotQA | 100.00 | 100.00 |
| MultiFieldQA-en | 40.28 | 40.28 |
| LCC | 75.00 | 75.00 |

七条生成 token 完全一致；LCC 原始索引 22 在第 52 个生成 token 起不同，首个有效代码行及该任务评分保持一致。FA2 两轮的八条生成 token 逐条一致。只覆盖这些小样本，不能据此宣称完整 LongBench 质量或 BF16 逐位等价。

两条输入各保存 prefill 加 16 次 teacher-forced decode 的完整 128256 logits。prefill 最大差为 0，decode 最大/平均绝对差分别为 0.21875/0.02144 与 0.53125/0.03905，34 个 argmax 全部相同。实际每条验证记录 1024 次 FA2 调用，其中 288 次使用新 320 维扩展，覆盖两种 query 长度。

原始结果、不可变 source_v1/v1b/v2、编译命令和 SHA manifest、CPU/GPU/sanitizer 日志、预测、计时与显存记录在 H20 `data/results/fa2_decode_20261014/`；本机对应 `/data/workspace/slimkv_analysis/fa2_decode_20261014/` 保存 SHA receipt 和独立审核。完整 logits tensor 保留在 H20；独立 tensor 审核读取实际文件。两轮 GPU suite 均正常退出，原占卡程序已恢复。启用及构建命令见 [评测说明](evaluation.md)。

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
- FA2 latent 后端的完整 LongBench/NIAH、多模型、批处理与长上下文质量验收；目前仅 Llama 小样本。
- 最终三 seed NIAH judge、随机字符重建、attention 相似性统计和多问题共享上下文实验。
- 论文完整效率协议：attention 单独计时、synthetic Full KV、latent-cache Recon、独立重复和 median。
- SlimKV 的 vLLM 模型、paged cache 和连续批处理。

## 远端已有权重

H20 的 `model/` 下保留 Llama-3.1-8B 的逐层 f4 `checkpoint-2000`。32 层 K/V 列表各自平均为 32，原始 head dim 为 128；权重为 BF16，Beacon 参数约 621M。该配置的常规 Llama3 推理现在默认启用低维 Beacon 缓存与原始维度重建；原始权重文件和配置文件保持不变。

该权重、论文、历史预测和图作为本地产物保留。历史文件不能自动代表论文最终实验结果，应通过 checkpoint、有效配置、样本集和实际代码版本追溯。
