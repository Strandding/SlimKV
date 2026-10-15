# 耗时与显存评测

[English](../benchmark.md)

正式对照使用下列 tools 工具和当前模型实现，从仓库根目录运行命令。

## 样本和运行

```bash
python -m tools.data.prepare_longbench \
  --dataset_dir /path/to/longbench --output data/results/fixtures/samples.jsonl \
  --tasks qasper hotpotqa multifieldqa_en lcc \
  --max_samples_per_task 2 --min_length 2500 --max_length 6000
```

按原始顺序选择符合 dataset.length 的前两行，保留官方字段并增加 task/source_index，保存 SHA manifest；不截断。工具当前需要 Llama3 checkpoint、prompt 超过两个窗口且不超过31550 token，不是任意短 prompt 测试。

`tools.benchmark.benchmark_reconstruction` 比较 full/full_compact/latent，用 eager、beacon_ratio8、decode_tokens64、warmup1、rounds2、perf_sample_indices0 2。Full_compact 仅在 harness 克隆首次 sink view 释放多余 backing storage，不改注意力数学。

[扩展构建命令](../benchmark.md#building-the-width-320-extension) 基于 FA2 v2.8.3/CUTLASS，只编 BF16 SM90 forward。静态解码支持 raw KV 的128维和 latent KV 的200–256/320维；普通解码在≤256维使用标准 FA2。较大 rank 临时补到320，持久缓存保留真实 rank。

## Prefill 索引与 RoPE 融合

Llama 默认开启 `--beacon_prefill_fast True`。推理时，对使用 latent 缓存、无 padding 的 interleave FA2 窗口，在层间共享 token 索引，并跳过完整窗口结束时不再保留的 raw KV 拷贝。BF16 Q/K 的 RoPE 操作合并到一个 Triton kernel，Beacon K 保持不旋转。每次乘法仍先舍入为 BF16 再做加法，与原操作一致。投影矩阵形状、权重、秩、注意力 scale 和 mask 保持不变，prefill 注意力仍在原始特征维度计算。

训练、带 padding 的输入和不支持的配置使用原路径。不支持的 RoPE kernel 也回退到原 RoPE 操作。在 `main.evaluation.eval_longbench` 中传入 `--beacon_prefill_fast False`，或设置 `model.config.beacon_prefill_fast = False`，可关闭上述优化进行对照。首次符合条件的调用会编译 Triton kernel。

H20 上使用 BF16 Llama-3.1-8B f4 权重和 t4f4，预热一次后交替测量四次，得到以下 prefill 中位数。计时使用同步 forward 耗时，包含全部 prompt 窗口和最后一个 token 的 logits，不含 decode。

| 输入 token 数 | 原实现（秒） | 优化后（秒） | 加速比 |
| ---: | ---: | ---: | ---: |
| 4,009 | 0.796 | 0.695 | 1.14× |
| 6,194 | 1.283 | 1.112 | 1.15× |
| 31,500 | 7.427 | 6.585 | 1.13× |

以上输入、窗口边界用例，以及不同真实 prompt 组成的 B4/B32 × 31,500-token 批次，logits 和各层持久 KV 均逐位一致。单次配对验证中，B4 耗时从26.28降至24.54秒，B32从196.98降至187.12秒；这两组是单次观测，并非重复测量的中位数。大 batch 的耗时更集中于矩阵乘法，而本轮保持矩阵乘法不变。

两条输入各接续128步固定 decode，在跨 Beacon 插入和窗口切换后，logits 和最终 KV 仍逐位一致。八条固定 LongBench-v1 样本的贪心生成 token 和官方评分一致。`tests/test_llama_prefill.py` 中的 CUDA 测试覆盖 BF16 RoPE 舍入、张量 stride、sink 和历史 Beacon，在 CPU CI 中跳过。

## B1 解码 Graph

重新构建扩展后，`--beacon_decode_graph True` 会对符合条件的 B1 单 token latent FA2 解码启用整模型 Graph，包括插入 Beacon 的步骤。KV 使用固定地址的缓冲区原位追加。较大 batch 和诊断调用使用普通路径。设置 `--beacon_decode_graph_full False` 可保留原来的局部层 Graph。

插入 Beacon 时，将 raw token 与 Beacon 的两个 query 表示为共享 KV 存储的两条单 query 序列。两者的有效 KV 长度相差一个位置，raw query 仍不可见新插入的 Beacon。FA2 可以分别打包 GQA heads，提高 KV 复用率。split 数与原双 query 调用一致，以保留 softmax 的归约顺序。持久缓存的秩和内容不变。

H20 上使用 BF16 Llama-3.1-8B f4 权重和 t8f4，同样的两条输入及63步固定解码输入，预热一次后测量四次，得到以下中位数。计时包含 forward 和 argmax。

| 输入 token 数 | 局部层 Graph（tokens/s） | 整模型 Graph（tokens/s） | Prefill（秒） | 分配显存峰值（GiB） |
| ---: | ---: | ---: | ---: | ---: |
| 4,009 | 54.13 | 82.38 | 0.76 | 22.66 |
| 6,194 | 55.08 | 82.09 | 1.21 | 22.72 |

显存峰值包含模型、缓存、FA 工作区和保留的 Graph。这组对照的64步 logits 与原实现逐位一致。来自 Qasper、HotpotQA、MultiFieldQA-en 和 LCC 的八条固定 LongBench-v1 样本，贪心生成内容和官方评分也完全一致。

```bash
python -m tools.benchmark.benchmark_fa2_decode \
  --model_name_or_path /path/to/slimkv-checkpoint \
  --samples_jsonl data/results/fixtures/samples.jsonl \
  --output_dir data/results/fa2-benchmark \
  --attn_impl flash_attention_2 --beacon_ratio 8 \
  --decode_tokens 64 --warmup 1 --rounds 2 \
  --perf_sample_indices 0 2 --compare_b1
```

`--compare_b1` 在同模型比较 compact full、重建、legacy FA2、bookkeeping、Graph 解码；不传则三路径对照。另有 prior decoder 对照，不能与 compare_b1 混用。

每次 reset Memory，预热涵盖 JIT/capture；计时为同步 host wall time，含 forward/argmax/同步边界，不含 tokenization/评分/IO/诊断/保存 logits。64生成含 prefill 首 token和63 decode，性能阶段无EOS提前停止，模式之间用相同固定输入，quality另用正常greedy。每模式四次对称测量取median，保留全部raw，不删慢样本。

KV物理字节、live/peak CUDA allocated、allocator reserved 是不同概念。所有对照保留预热 Graph，模式peak差不能证明Graph额外显存为0。

## 可选 decoder-tail 编译

`--beacon_decode_compile True` 在 Torch 2.11 下为 B1 BF16 FA2 Graph decode 启用已验证的编译路径，保留原始 FP32 方差计算和 BF16 舍入边界，继续复用现有 CUDA Graph。其他 Torch 版本和不支持的形状使用原路径。首次符合条件的调用会在 Graph capture 前编译。

CUDA 推理 CLI 和 `eval_longbench_slimkv.sh` 默认打开此选项。`main.inference.infer` 使用 `--no-beacon_decode_compile` 关闭，shell 预设使用 `DECODE_COMPILE=False`，`main.evaluation.eval_longbench` 使用 `--beacon_decode_compile False`。Python API 和对照测试工具保留原默认设置，训练不变。

H20 上同一 f4 权重、t8f4、B1 和63步固定 decode，预热后交替测量四次：4,009-token 输入从87.76提升到89.48 tokens/s，6,194-token 输入从87.62提升到89.39 tokens/s。编译不计入耗时。全部64步 logits 和最终持久 KV 逐位一致，八条固定 LongBench-v1 样本的 greedy token 和官方评分一致。这里采用当前源码的同期对照，历史82 tokens/s 是另一组基线。

## 多 batch

`tools.benchmark.benchmark_batch_decode` 需要 checkpoint、samples_jsonl、forced_tokens_json、output_dir。固定输入 JSON 为样本索引到 token ID 列表（默认每条至少63个）；从reference/warmup保存的`logits_<index>.pt`中提取`forced_token_ids`，不能拿任意token冒充same-input对照。

复制同一 prompt 来测量等长 dense batch 的吞吐量，每行维护独立 KV。通过 `--rounds` 指定每组测量轮数。工具预热并保留各 batch 的 Graph 工作区，比较时分别记录每步耗时、总 tokens/s 和显存；这组负载不代表不同长度独立请求的服务吞吐量。

## Profiling

[英文命令](../benchmark.md#profiling-and-gpu-validation) 用cuda,nvtx、sample=none、cpuctxsw=none、node Graph trace及ProfilerAPI capture。需要GPU支持和采样权限时再加metrics参数；均值限定真实kernel时间段。NCU用真实保存Q/K/V replay诊断spill/occupancy。Profiler duration不代替无profile正式median。

独立GPU验证入口：validate_decode_ops、validate_b1_decode、validate_fa2_decode，部分支持batch_sizes；记录dtype/shape/GPU/extension SHA/memcheck。CPU CI不证明CUDA二进制正确，也不会启动长训练。

## 固定上下文 decode 对照

使用同一 checkpoint，统一 BF16、B1 和 FlashAttention 2，对比 Full KV、重建和直接 latent attention：

```bash
MODEL_NAME_OR_PATH=/path/to/slimkv-checkpoint \
  bash scripts/benchmark/benchmark_context_decode.sh
```

启动脚本默认 `EXECUTION_MODE=optimized`：三种方法均使用预分配缓存、整模型 CUDA Graph 和相同的可选精确 decoder-tail 编译。编译需要 Torch 2.11，`DECODE_COMPILE=False` 对所有方法关闭编译。预热、编译和 Graph 构建不计入时间，计时区间出现 Graph 构建会报错。

默认覆盖 32K、64K、96K、128K 输入、t4f4 与 t8f4，每项在 prefill 后执行完整的 128 次单 token decode forward。K = 1024。各方法使用保存的相同合成输入和 Full KV 生成的相同后续输入，EOS 不提前结束测试，prefill 不计入 decode 时间。

优化 runner 面向完整 Beacon 窗口结束后的固定上下文，后续长度不超过一个窗口。各方法共享原始模型权重和执行优化策略。测速前，各方法在未捕获执行和 Graph replay 下的逐步 logits、最终持久 KV 都必须与对应原实现逐位一致。数值检查不能替代 LongBench 精度评估。此 runner 用于固定上下文性能测试，不是服务调度器。

Full KV 不插入 Beacon、不丢弃原始 token，每个新 Key 仅做一次原始 RoPE。重建路径每一步重新构造历史 Beacon K/V，原始维度 workspace 属于临时 attention 存储，不计作压缩后的持久 KV。各路径 prefill 均在原始维度使用 FA2。128K 输入后的 decode 额外补充 128 个相同频率的 RoPE 位置，已有表不变。

分别报告两个指标：

- **Attention GPU kernel 耗时**：attention 内实际 GPU kernel 耗时之和，包含 QKV/输出投影、KV 重建、RoPE、缓存操作和 latent 映射。重建与 RoPE 单列明细，已包含在总值内，排除 Norm、MLP 和 LM head。独立 profiler pass 在未捕获模式执行相同的优化操作并检查预测一致。GPU annotation 只用于定位 kernel，不把 annotation 跨度当成 kernel 时间，并检查 FA2 分项不超过 attention 总值。
- **端到端 decode 耗时**：128 次模型 forward 与 argmax 的同步墙钟时间，排除缓存恢复、分词、prefill、编译、Graph 构建和 profiler 开销。预热一次后交替测试顺序，重复四次，使用中位数。

H20 上使用 BF16 Llama-3.1-8B f4 权重，优化 runner 得到以下结果。各方法统一使用 FA2、预分配缓存、整模型 Graph 和精确 decoder-tail 编译。端到端耗时为预热后四次测量的中位数，每次执行 128 步 decode。kernel 耗时来自独立的未捕获 profiling pass。

**端到端 decode 耗时（ms/token）**

| 输入 | Full KV | Recon t4f4 | Latent t4f4 | Recon t8f4 | Latent t8f4 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32K | 10.81 | 15.32 | 9.86 | 12.11 | 8.63 |
| 64K | 14.40 | 21.95 | 11.58 | 15.03 | 9.18 |
| 96K | 17.97 | 28.61 | 13.17 | 18.22 | 9.81 |
| 128K | 21.53 | 35.55 | 14.71 | 21.54 | 10.35 |

**Attention GPU kernel 耗时（ms/token）**

| 输入 | Full KV | Recon t4f4 | Latent t4f4 | Recon t8f4 | Latent t8f4 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32K | 5.83 | 10.38 | 4.84 | 6.95 | 3.61 |
| 64K | 9.44 | 17.24 | 6.57 | 10.10 | 4.17 |
| 96K | 12.97 | 24.10 | 8.14 | 13.44 | 4.80 |
| 128K | 16.50 | 31.19 | 9.69 | 16.84 | 5.32 |

重建路径的 kernel 耗时包含每一步的历史 KV 重建及 RoPE。四个输入长度均通过与各自原模型路径的 logits 和持久 KV 逐位一致检查。这里使用合成输入测速，LongBench 精度评估单独进行。

如需比较原先关闭 Graph 的实现路径，使用：

```bash
EXECUTION_MODE=matched MODEL_NAME_OR_PATH=/path/to/slimkv-checkpoint \
  bash scripts/benchmark/benchmark_context_decode.sh
```

`matched` 对所有方法关闭 Graph 和编译。此时 Full KV 使用 append cache，SlimKV 使用原有非 Graph 缓存路径，两者分配和发射开销不同，因此这一模式不单独衡量 attention 算法差异。

输出默认 `data/results/context_decode_<execution_mode>`，必须使用新目录。文件包含 `results.json`、`summary.csv`、实际输入与后续 token、数值验证、源码和配置哈希、逐次耗时。设置 `REFERENCE_DIR` 可以复用并核对上一轮的实际输入和后续 token。可以通过 `OUTPUT_DIR`、`CONTEXT_LENGTHS`、`BEACON_RATIOS`、`REPETITIONS`、`PROMPT_FILE` 调整实验。

实现位于 `tools/benchmark/benchmark_context_decode.py` 和 `context_decode_runtime.py`，使用 Transformers 4.40 环境及兼容的 FA2 decode 扩展。直接运行 Python 模块默认 `matched`，优化模式需显式传入 `--execution_mode optimized`。
