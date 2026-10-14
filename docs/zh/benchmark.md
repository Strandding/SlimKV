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

[width-320 构建命令](../benchmark.md#building-the-width-320-extension) 基于 FA2 v2.8.3/CUTLASS，只编 BF16 SM90 forward。标准 FA2 用于≤256；较大 rank 临时补到320，持久缓存保留真实 rank。扩展不是 A100 或训练 backward kernel。

```bash
python -m tools.benchmark_fa2_decode \
  --model_name_or_path /path/to/slimkv-checkpoint \
  --samples_jsonl data/results/fixtures/samples.jsonl \
  --output_dir data/results/fa2-benchmark \
  --attn_impl flash_attention_2 --beacon_ratio 8 \
  --decode_tokens 64 --warmup 1 --rounds 2 \
  --perf_sample_indices 0 2 --compare_b1
```

`--compare_b1` 在同模型比较 compact full、重建、legacy FA2、bookkeeping、局部 Graph；不传则三路径对照。另有 prior decoder 对照，不能与 compare_b1 混用。

每次 reset Memory，预热涵盖 JIT/capture；计时为同步 host wall time，含 forward/argmax/同步边界，不含 tokenization/评分/IO/诊断/保存 logits。64生成含 prefill 首 token和63 decode，性能阶段无EOS提前停止，模式之间用相同固定输入，quality另用正常greedy。每模式四次对称测量取median，保留全部raw，不删慢样本。

KV物理字节、live/peak CUDA allocated、allocator reserved 是不同概念。所有对照保留预热 Graph，模式peak差不能证明Graph额外显存为0。

## 已核实的 H20 局部结果

BF16 f4 Llama3.1-8B、Torch2.11+cu129、TF4.40.2、FA2 2.8.3、Triton3.6、ratio8、window=stride1024。

| 输入 | compact full | legacy latent FA2 | bookkeeping | partial Graph |
|---|---:|---:|---:|---:|
| 4009 token | 26.343 ms/token | 24.772 | 24.263 | 18.381 |
| 6194 token | 26.397 ms/token | 24.671 | 24.227 | 18.671 |

Prefill eager→FA2为867.59→736.25ms /1398.39→1168.39ms。Prefill后latent持久KV为132.906/30.563MiB，compact full为179.875/103.125MiB。B1 Graph增加540804096live bytes（515.75MiB），KV不变。不是完整benchmark或128K验证。[摘要和原始产物hash](../results/validation_20261014.json)。原始大文件尚未公开发布。

## 多 batch

`tools.benchmark_batch_decode` 需要 checkpoint、samples_jsonl、forced_tokens_json、output_dir。固定输入 JSON 为样本索引到 token ID 列表（默认每条至少63个）；从reference/warmup保存的`logits_<index>.pt`中提取`forced_token_ids`，不能拿任意token冒充same-input对照。

| B | 4009: step ms /总tok/s | 6194: step ms /总tok/s |
|---|---:|---:|
| 1 |18.183/55.00|17.939/55.74|
| 2 |18.646/107.26|18.506/108.08|
| 4 |18.763/213.19|18.477/216.49|
| 8 |20.139/397.23|19.092/419.03|
| 16 |25.827/619.50|20.387/784.81|
| 32 |40.917/782.08|24.714/1294.82|

复制同prompt，每行计算独立cache；六种B预热图都保留。共48次正式运行，same-batch Graph/eager全row logits一致，不保证跨B BF16逐位相等，也不是32独立ragged服务请求吞吐。

6194输入的独立16decode Nsys：B1/B8/B32平均SMs Active 19.21/30.69/54.92%，kernel区间并集忙碌比例31.86/41.94/66.40%，两种指标不同，均不是持续100%。独立进程peak allocated22.38/34.49/75.99GiB；计时进程保留多B图不能当单B部署内存。没有测B64。

## Profiling

[英文命令](../benchmark.md#profiling-and-gpu-validation) 用cuda,nvtx、sample=none、cpuctxsw=none、node Graph trace及ProfilerAPI capture。需要GPU支持和采样权限时再加metrics参数；均值限定真实kernel时间段。NCU用真实保存Q/K/V replay诊断spill/occupancy。Profiler duration不代替无profile正式median。

独立GPU验证入口：validate_decode_ops、validate_b1_decode、validate_fa2_decode，部分支持batch_sizes；记录dtype/shape/GPU/extension SHA/memcheck。CPU CI不证明CUDA二进制正确，也不会启动长训练。
