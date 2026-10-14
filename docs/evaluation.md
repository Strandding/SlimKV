# SlimKV 评测

默认生成评测采用标准注意力。Llama3 可以保存低维 Beacon K/V，在计算前按层重建；可选 FA2 后端在 decode 中直接计算 latent attention。Qwen2 目前仍保存完整维度 K/V。独立的 `slimkv/latency` 实验入口保留用于诊断。

## LongBench

```bash
MODEL_NAME_OR_PATH=/path/to/checkpoint-2000 \
EVAL_DATA=/path/to/long-llm/longbench \
CUDA_VISIBLE_DEVICES=0 NPROC_PER_NODE=1 \
BEACON_RATIO=8 \
bash scripts/evaluation/eval_longbench_llama31_8B.sh
```

通用入口为 `python -m main.eval_longbench --help`。独立请求之间必须调用 `model.memory.reset()`；评测入口会为每个 batch 重置。输出保存在 `data/results/`，每个任务文件包含首行分数和后续预测记录。

Llama3 常规入口新增 `--beacon_cache_mode`：默认 `auto` 在兼容配置的推理中保存低维 Beacon K/V；`full` 用于完整缓存对照；`latent` 显式要求低维缓存，配置不兼容时会报错。当前低维存储支持 interleave、window=stride、单窗口编码、正 token 压缩比、K/V 均为低秩，且不启用 chunk-fixed K RoPE、K positional residual 或 beacon_skip。训练始终保留完整缓存；切换训练状态或模式前先重置 Memory。验证使用 greedy 生成。

同一权重的对照可在调用 `main.eval_longbench` 时分别传入 `--beacon_cache_mode full` 和 `--beacon_cache_mode latent`，保持其他参数一致。小规模评测可以设置 `--tasks qasper hotpotqa --max_samples_per_task 2`。当前 H20 验证显式使用 `--attn_impl eager`；SDPA 已做 CPU 回归，标准 FA2 prefill 与 latent FA2 decode 的 H20 GPU 验证见下节。

`BEACON_RATIO` 是 token 压缩比例，不是低秩特征压缩比例。平均 f4 checkpoint 配合 token ratio 4/8 分别对应名义 t4f4/t8f4；raw tail 和 sink 会使实际压缩比低于名义值。

## 重建推理的耗时、显存与小样本 LongBench-v1

```bash
python -m tools.benchmark_reconstruction \
  --model_name_or_path /path/to/checkpoint-2000 \
  --samples_jsonl /path/to/samples.jsonl \
  --output_dir data/results/reconstruction-benchmark \
  --attn_impl eager --beacon_ratio 8 \
  --decode_tokens 64 --warmup 1 --rounds 2 \
  --perf_sample_indices 0 2
```

输入 JSONL 使用 LongBench-v1 原始字段，另加 `task`（任务名）和 `source_index`（原始行号）。工具采用现有 LongBench 模板与 Llama-3 chat template，代码补全不加 chat template。质量评测使用任务默认生成上限、greedy 和现有 QA 换行停止规则；输出包含每条预测、生成 token 和各任务评分。

耗时测量固定生成长度，不按 EOS 提前停止，首 token 来自 prefill，其余 token 是 decode；这部分不用于质量评分。每次调用前重置 Memory，预热后以 full → full_compact → latent → latent → full_compact → full 的对称顺序测量，每轮每种模式各两次。使用同步后的 host wall time，报告模型 prefill、首 token 延迟及 decode 每 token 耗时；tokenization、评分、文件 IO 和 logits 保存不计入计时。

full_compact 是对照工具中释放 sink 切片多余 backing storage 的完整缓存模式，值和注意力数学不变；标准 full 模式仍按仓库原路径运行。工具同时记录 CUDA live allocated/peak allocated、Memory 逻辑 K/V 字节和去重后的底层 storage 字节，不能把缓存字节节省当作 GPU 总显存节省。allocator reserved 字节不参与显存对比。

`--disable_optional_flash_import` 只用于 eager/SDPA 验证环境中存在 FlashAttention 元信息但缺少扩展的情况；它在工具内关闭未使用的可选导入，不修改模型源码。

## 可选 FA2 latent decode

本后端使用 `beacon_decode_backend=fa2`，默认仍为 `reconstruct`。训练与 prefill 在原始 head dim 上计算；decode 的历史和新 Beacon K/V 直接参与 latent attention，不重建历史 K/V。raw/sink 分支在原始维度计算，两支用 LSE 合并为同一个 softmax。新插入 Beacon 的 decode 包含 raw、Beacon 两个 query，因果关系已单独验证。

当前支持 Llama 的低维缓存兼容配置，另要求 full-coverage、attend_prev、global Linear up projection、关闭 Beacon K RoPE、开启 raw K RoPE，以及关闭 latent relative-position bias。H20 验证使用 BF16、未 padding 的 batch=1、每步一个原始 token。其他批量或多 token 调用保留重建路径。cache mode 与 decode backend 的切换都需要先 `model.memory.reset()`。

现成 FA2 2.8.3 扩展用于计算维度不超过 256 的层；逐层不等 K/V rank 临时补零至相同维度，超过 256 的层补到 320，并调用本仓库补编译的 forward kernel。缓存保存实际 rank。f4 checkpoint 的全局 K/V rank 最大为 304/312；softmax scale 始终为 `1/sqrt(128)`。

先准备能够导入 FA2 2.8.3 的环境，Torch、CUDA、Python 和 C++ ABI 必须与现成二进制匹配。H20 实测环境为独立 venv 的 Torch 2.11.0+cu129、Transformers 4.40.2；复用了已有 FA2 `.so`，仅编译 BF16、SM90、320 维 forward。构建不包含 backward 或其他 GPU 架构。

```bash
git clone --depth 1 --branch v2.8.3 \
  https://github.com/Dao-AILab/flash-attention.git .cache/flash-attention-v2.8.3
git -C .cache/flash-attention-v2.8.3 submodule update --init csrc/cutlass
python -m tools.build_fa2_decode \
  --fa2_source .cache/flash-attention-v2.8.3 \
  --output_dir .cache/fa2_decode_320 --jobs 2
export SLIMKV_FA2_DECODE_EXTENSION_DIR="$PWD/.cache/fa2_decode_320"
python -m tools.validate_fa2_decode \
  --extension_dir "$SLIMKV_FA2_DECODE_EXTENSION_DIR" \
  --output data/results/fa2-kernel-validation.json
```

模型入口通过 `--beacon_cache_mode latent --beacon_decode_backend fa2` 启用。初次验证使用 eager prefill；当前可设置 `--attn_impl flash_attention_2`，让 prefill 使用标准 128 维 FA2，decode 使用独立的 latent FA2 后端。CUDA BF16/FP16 且 Triton 可用时，decode 自动融合 RoPE 和 softmax 合并；CPU、梯度计算或缺少 Triton 时保留 Torch 路径。320 维扩展仍只支持 BF16、SM90。首次 Triton JIT 在预热阶段完成。

```bash
python -m tools.benchmark_fa2_decode \
  --model_name_or_path /path/to/checkpoint-2000 \
  --samples_jsonl /path/to/samples.jsonl \
  --output_dir data/results/fa2-benchmark \
  --attn_impl flash_attention_2 --beacon_ratio 8 \
  --decode_tokens 64 --warmup 1 --rounds 2 \
  --perf_sample_indices 0 2
```

性能阶段在同一模型、同一环境下交替 full_compact → reconstruct → fa2 → fa2 → reconstruct → full_compact，每轮各两次。decode 输入固定为 full_compact 预热产生的 token，使各模式处理同一输入；计时包含实际 model forward、argmax 和 CUDA 同步边界，tokenization、诊断和 IO 在计时外。质量阶段单独使用正常 greedy 生成。工具另外保存两条输入的 teacher-forced logits 和 kernel 调用覆盖；BF16 运算顺序不同，输出不保证逐位相等。具体结果见 [实现状态](implementation_status.md)。

性能对照还可传入 `--forced_tokens_json /path/to/tokens.json`，文件为样本索引到 decode 输入 token 列表的 JSON 对象；它让 eager/FA2 prefill 和优化前后处理完全相同的 decode 输入。`--reference_decode_source /path/to/previous/fa2_decode.py` 加载兼容的旧 decoder，只切换计算函数，复用同一个模型和参数；性能顺序扩展为四种模式再反序，各模式每轮两次。`--skip_quality` 可用于只复测性能。默认另外保存 16 步 teacher-forced logits；旧 decoder 对照同时报告优化后是否逐位相等。

profiling 与无 profiling 的计时分开运行：

```bash
nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  -o data/results/slimkv-decode \
  python -m tools.profile_fa2 --mode model --phase decode \
  --model_name_or_path /path/to/checkpoint-2000 \
  --samples_jsonl /path/to/samples.jsonl --sample_index 2 \
  --output_dir data/results/slimkv-decode-data
```

`--phase prefill` 采集 prefill；`--mode capture` 保存第 0、2、28 层真实的 raw/latent Q/K/V、LSE 和输出。对其中一个 case 使用 NCU：

```bash
ncu --nvtx --nvtx-include 'kernel_probe/' \
  --kernel-name-base demangled --kernel-name 'regex:.*flash_fwd.*' \
  --launch-count 1 --replay-mode kernel \
  --section LaunchStats --section SpeedOfLight \
  --section MemoryWorkloadAnalysis --section SchedulerStats \
  --section WarpStateStats --section SourceCounters --import-source yes \
  --export data/results/slimkv-kernel \
  python -m tools.profile_fa2 --mode kernel \
  --case_file /path/to/captured-case.pt \
  --output_dir data/results/slimkv-kernel-data
```

GPU 融合算子的独立验证入口为 `python -m tools.validate_decode_ops --output data/results/decode-ops.json`。Nsys/NCU 会增加运行开销；它们的 duration 用于定位问题，正式性能使用前述独立 benchmark 中位数。

## NIAH 与其他任务

```bash
python -m main.eval_needle --help
python -m main.eval_passkey --help
python -m main.eval_infbench --help
python -m main.eval_lm --help
```

NIAH 默认使用 ROUGE，`--gpt_eval` 需要可选 judge 依赖与环境中的 API key。当前 judge 和随机种子流程不是论文最终实验协议，见 [实现状态](implementation_status.md)。

## Latent attention 原型

```bash
python -m main.benchmark_latency \
  --lowrank_model_or_ckpt /path/to/checkpoint-2000 \
  --mode beacon_lowrank_hybrid_latent \
  --attn_impl eager --beacon_ratio 8 \
  --prompt_len 8192 --gen_len 128 --repeat 1 --warmup 0 \
  --output_json /tmp/slimkv-latency.json
```

该入口目前用于诊断原型。已有 benchmark 存在重复试验 memory 累积问题，且 Recon 基线与论文的 latent-cache 重建基线不同；上述单次调用不能替代论文速度表。Hybrid 必须使用 eager，FA2 fallback 的 mask 问题尚未修复。

热力图与 probe 入口位于 `main/eval_longbench_chunk2_heatmap.py` 和 `main/eval_longbench_beacon_probe.py`；对应启动脚本位于 `scripts/evaluation/`。图和历史预测是本地产物，不进入新提交。

## vLLM 基线

`scripts/vllm/` 用于原始模型基线，模型配置在 `config/vllm_models.sh`。它没有 SlimKV 的 paged latent cache 或服务端模型实现，不能作为 SlimKV 已接入 vLLM 的证据。
