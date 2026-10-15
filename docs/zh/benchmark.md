# 耗时与显存评测

[English](../benchmark.md)

正式对照使用下列 tools 工具和当前模型实现，从仓库根目录运行命令。

## 样本和运行

```bash
python -m tools.prepare_longbench \
  --dataset_dir /path/to/longbench --output data/results/fixtures/samples.jsonl \
  --tasks qasper hotpotqa multifieldqa_en lcc \
  --max_samples_per_task 2 --min_length 2500 --max_length 6000
```

按原始顺序选择符合 dataset.length 的前两行，保留官方字段并增加 task/source_index，保存 SHA manifest；不截断。工具当前需要 Llama3 checkpoint、prompt 超过两个窗口且不超过31550 token，不是任意短 prompt 测试。

`tools.benchmark_reconstruction` 比较 full/full_compact/latent，用 eager、beacon_ratio8、decode_tokens64、warmup1、rounds2、perf_sample_indices0 2。Full_compact 仅在 harness 克隆首次 sink view 释放多余 backing storage，不改注意力数学。

[扩展构建命令](../benchmark.md#building-the-width-320-extension) 基于 FA2 v2.8.3/CUTLASS，只编 BF16 SM90 forward。静态解码支持 raw KV 的128维和 latent KV 的200–256/320维；普通解码在≤256维使用标准 FA2。较大 rank 临时补到320，持久缓存保留真实 rank。

## Prefill 索引与 RoPE 融合

Llama 默认开启 `--beacon_prefill_fast True`。推理时，对使用 latent 缓存、无 padding 的 interleave FA2 窗口，在层间共享 token 索引，并跳过完整窗口结束时不再保留的 raw KV 拷贝。BF16 Q/K 的 RoPE 操作合并到一个 Triton kernel，Beacon K 保持不旋转。每次乘法仍先舍入为 BF16 再做加法，与原操作一致。投影矩阵形状、权重、秩、注意力 scale 和 mask 保持不变，prefill 注意力仍在原始特征维度计算。

训练、带 padding 的输入和不支持的配置使用原路径。不支持的 RoPE kernel 也回退到原 RoPE 操作。在 `main.eval_longbench` 中传入 `--beacon_prefill_fast False`，或设置 `model.config.beacon_prefill_fast = False`，可关闭上述优化进行对照。首次符合条件的调用会编译 Triton kernel。

H20 上使用 BF16 Llama-3.1-8B f4 权重和 t4f4，预热一次后交替测量四次，得到以下 prefill 中位数。计时使用同步 forward 耗时，包含全部 prompt 窗口和最后一个 token 的 logits，不含 decode。

| 输入 token 数 | 原实现（秒） | 优化后（秒） | 加速比 |
| ---: | ---: | ---: | ---: |
| 4,009 | 0.796 | 0.695 | 1.14× |
| 6,194 | 1.283 | 1.112 | 1.15× |
| 31,500 | 7.427 | 6.585 | 1.13× |

以上输入、窗口边界用例，以及不同真实 prompt 组成的 B4/B32 × 31,500-token 批次，logits 和各层持久 KV 均逐位一致。单次配对验证中，B4 耗时从26.28降至24.54秒，B32从196.98降至187.12秒；这两组是单次观测，并非重复测量的中位数。大 batch 的耗时更集中于矩阵乘法，而本轮保持矩阵乘法不变。

两条输入各接续128步固定 decode，在跨 Beacon 插入和窗口切换后，logits 和最终 KV 仍逐位一致。八条固定 LongBench-v1 样本的贪心生成 token 和官方评分一致。`tests/test_llama_prefill.py` 中的 CUDA 测试覆盖 BF16 RoPE 舍入、张量 stride、sink 和历史 Beacon，在 CPU CI 中跳过。

## B1 解码 Graph

重新构建扩展后，`--beacon_decode_graph True` 会对符合条件的 B1 单 token latent FA2 解码启用整模型 Graph。KV 使用固定地址的缓冲区原位追加。插入 Beacon、较大 batch 和诊断调用使用普通路径。设置 `--beacon_decode_graph_full False` 可保留原来的局部层 Graph。

H20 上使用 BF16 Llama-3.1-8B f4 权重和 t8f4，同样的两条输入及63步固定解码输入，预热一次后测量四次，得到以下中位数。计时包含 forward 和 argmax。

| 输入 token 数 | 局部层 Graph（tokens/s） | 整模型 Graph（tokens/s） | Prefill（秒） | 分配显存峰值（GiB） |
| ---: | ---: | ---: | ---: | ---: |
| 4,009 | 54.13 | 82.38 | 0.76 | 22.66 |
| 6,194 | 55.08 | 82.09 | 1.21 | 22.72 |

显存峰值包含模型、缓存、FA 工作区和保留的 Graph。这组对照的64步 logits 与原实现逐位一致。来自 Qasper、HotpotQA、MultiFieldQA-en 和 LCC 的八条固定 LongBench-v1 样本，贪心生成内容和官方评分也完全一致。

```bash
python -m tools.benchmark_fa2_decode \
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

CUDA 推理 CLI 和 `eval_longbench_slimkv.sh` 默认打开此选项。`main.infer` 使用 `--no-beacon_decode_compile` 关闭，shell 预设使用 `DECODE_COMPILE=False`，`main.eval_longbench` 使用 `--beacon_decode_compile False`。Python API 和对照测试工具保留原默认设置，训练不变。

H20 上同一 f4 权重、t8f4、B1 和63步固定 decode，预热后交替测量四次：4,009-token 输入从87.76提升到89.48 tokens/s，6,194-token 输入从87.62提升到89.39 tokens/s。编译不计入耗时。全部64步 logits 和最终持久 KV 逐位一致，八条固定 LongBench-v1 样本的 greedy token 和官方评分一致。这里采用当前源码的同期对照，历史82 tokens/s 是另一组基线。

## 多 batch

`tools.benchmark_batch_decode` 需要 checkpoint、samples_jsonl、forced_tokens_json、output_dir。固定输入 JSON 为样本索引到 token ID 列表（默认每条至少63个）；从reference/warmup保存的`logits_<index>.pt`中提取`forced_token_ids`，不能拿任意token冒充same-input对照。

复制同一 prompt 来测量等长 dense batch 的吞吐量，每行维护独立 KV。通过 `--rounds` 指定每组测量轮数。工具预热并保留各 batch 的 Graph 工作区，比较时分别记录每步耗时、总 tokens/s 和显存；这组负载不代表不同长度独立请求的服务吞吐量。

## Profiling

[英文命令](../benchmark.md#profiling-and-gpu-validation) 用cuda,nvtx、sample=none、cpuctxsw=none、node Graph trace及ProfilerAPI capture。需要GPU支持和采样权限时再加metrics参数；均值限定真实kernel时间段。NCU用真实保存Q/K/V replay诊断spill/occupancy。Profiler duration不代替无profile正式median。

独立GPU验证入口：validate_decode_ops、validate_b1_decode、validate_fa2_decode，部分支持batch_sizes；记录dtype/shape/GPU/extension SHA/memcheck。CPU CI不证明CUDA二进制正确，也不会启动长训练。
