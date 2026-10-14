# 耗时与显存评测

[English](../benchmark.md)

正式对照使用 tools 中的当前模型工具。`main.benchmark_latency` 是独立历史原型，warmup/repeat 不 reset Memory，且所谓 reconstruct 的缓存语义不是论文 latent-cache Recon，不能用作论文复现结论。

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

## 多 batch

`tools.benchmark_batch_decode` 需要 checkpoint、samples_jsonl、forced_tokens_json、output_dir。固定输入 JSON 为样本索引到 token ID 列表（默认每条至少63个）；从reference/warmup保存的`logits_<index>.pt`中提取`forced_token_ids`，不能拿任意token冒充same-input对照。

复制同一 prompt 来测量等长 dense batch 的吞吐量，每行维护独立 KV。通过 `--rounds` 指定每组测量轮数。工具预热并保留各 batch 的 Graph 工作区，比较时分别记录每步耗时、总 tokens/s 和显存；这组负载不代表不同长度独立请求的服务吞吐量。

## Profiling

[英文命令](../benchmark.md#profiling-and-gpu-validation) 用cuda,nvtx、sample=none、cpuctxsw=none、node Graph trace及ProfilerAPI capture。需要GPU支持和采样权限时再加metrics参数；均值限定真实kernel时间段。NCU用真实保存Q/K/V replay诊断spill/occupancy。Profiler duration不代替无profile正式median。

独立GPU验证入口：validate_decode_ops、validate_b1_decode、validate_fa2_decode，部分支持batch_sizes；记录dtype/shape/GPU/extension SHA/memcheck。CPU CI不证明CUDA二进制正确，也不会启动长训练。
