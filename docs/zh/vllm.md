# vLLM 接入

[English](../vllm.md)

SlimKV 可以在单张 GPU 上，使用 vLLM 的调度器与 KV 分配器运行已转换的 BF16 Llama 权重。Prefill 仍在原始维度计算，decode 使用已有的免重建 FA2 实现，各层 K/V 秩直接读取 checkpoint。

## 安装

请使用独立于训练和原有推理环境的 Python 环境。已验证的版本为 Python 3.12、PyTorch 2.11.0+cu129、Transformers 5.8.0、FlashAttention 2.8.3，以及 vLLM `0.24.1.dev0+gee0da84ab.d20260731.cu129`。接入使用该版本的 V1 内部接口，其他版本尚未验证。

在已安装上述 vLLM 和 CUDA 依赖的环境中，从仓库根目录执行：

```bash
uv pip install -e ./integrations/vllm
```

这个安装配置使用同一份 SlimKV 源码并注册 vLLM 插件。仓库主安装配置继续保留训练与原有推理所用的 Transformers 4.40.x 依赖。每个环境只安装一种配置。

如果 checkpoint 的秩需要 width-320 扩展，请按[构建说明](../benchmark.md#building-the-width-320-extension)编译，并设置其中的 `SLIMKV_FA2_DECODE_EXTENSION_DIR`。使用 checkpoint 原有的权重、配置和 tokenizer。

## 推理

```bash
python -m main.inference.infer_vllm \
  --model /path/to/slimkv-checkpoint \
  --token-ratio 8 --max-num-seqs 4 \
  --prompt 'Explain why a KV cache helps language model inference.' \
  --prompt 'What does feature compression change in a KV cache?'
```

长输入可以通过 `--prompts-json /path/to/prompts.json` 提供，文件内容为字符串组成的 JSON 列表，每个字符串对应一个独立的聊天请求。入口不会自动截断输入，`--max-model-len` 应符合 checkpoint 和可用显存的限制。[英文说明](../vllm.md#inference)提供完整 Python API 示例。

`enforce_eager=True` 关闭 vLLM 外层的 Graph 捕获，SlimKV 自身已验证的 decode Graph 仍会使用。

## 缓存与调度

物理页、block table 和请求生命周期均由 vLLM 管理。每层分别存储当前窗口的原始 K/V 与历史压缩 Beacon K/V。原始页对应不重叠的注意力窗口，Beacon 页保留各层真实 K/V 秩，第一张页还存储原始 sink token。物理页的 padding 不改变 rank 或注意力计算。

每个请求执行时，按自己的 block table 取回临时工作区，运行原有 B1 注意力计算，写回新 K/V，再释放请求 KV 视图。请求结束后，vLLM 回收物理页。运行中的 batch 可以加入新请求、移除已完成请求，已有请求的 decode 也可以与新请求的 prefill 在同一调度步执行。目前同一步内仍逐请求计算，尚未实现合并 batch 计算或直接读取分页 KV 的 attention kernel。

已验证的配置为单卡 Llama、BF16、交错插入的 full-coverage Beacon、不重叠窗口，以及最多一个 sink token。请关闭 prefix caching、推测解码、量化和 chunked prefill。此入口不支持 prompt log-probability 输出、张量/流水线并行或 Qwen。
