# 快速开始

[English](../quickstart.md)

先按[安装指南](installation.md)准备环境和所需 FA2 扩展，然后从仓库根目录下载完整权重：

```bash
export MODEL_NAME_OR_PATH="$PWD/models/SlimKV-Llama3.1-8B-r4"
hf download Zihant0513/SlimKV-Llama3.1-8B-r4 \
  --local-dir "$MODEL_NAME_OR_PATH"
```

下载内容包含基座、训练后的 Beacon 投影、配置及 tokenizer，无需合并基座。其他权重见 [Hugging Face collection](https://huggingface.co/collections/Zihant0513/slimkv)。使用同一个已激活环境的终端运行：

```bash
python -m main.inference.infer \
  --model_name_or_path "$MODEL_NAME_OR_PATH" \
  --prompt 'Explain why a KV cache helps language model inference.' \
  --max_new_tokens 32
```

CUDA 默认使用 BF16、FA2 prefill、latent 缓存和免重建 decode、prefill 优化、CUDA Graph 及可选的 decoder-tail 编译，需先安装 FA2 和所需[扩展](../benchmark.md#building-the-width-320-extension)。采用 greedy 生成，只计算所需的末尾 token logits，不覆盖 rank/ratio/window/RoPE。`--device cpu --dtype fp32` 自动使用 eager 重建；Qwen2 使用 `--chat_template qwen --beacon_cache_mode auto --beacon_decode_backend reconstruct`，缓存限制见[实现状态](implementation_status.md)。长 UTF-8 上下文使用 `--prompt_file`。短输入可能不触发完整 Beacon 窗口。

[英文指南的 Python 示例](../quickstart.md#python-api) 与 CLI 使用同一真实 API：`ModelArgs`、`get_model_and_tokenizer`、`apply_chat_template`、`model.generate`。各独立请求之间、切换后端之前调用 `model.memory.reset()`。reset 不一定释放 Graph 工作区。

免重建 eager 直接使用压缩后的 Beacon K/V，不需要安装 FlashAttention：

```bash
python -m main.inference.infer \
  --model_name_or_path "$MODEL_NAME_OR_PATH" \
  --attn_impl eager --beacon_cache_mode latent --beacon_decode_backend latent \
  --prompt_file /path/to/context.txt
```

使用 FlashAttention 时，安装 FA2 和所需扩展后执行：

```bash
export SLIMKV_FA2_DECODE_EXTENSION_DIR="$PWD/.cache/fa2_decode_320"
python -m main.inference.infer \
  --model_name_or_path "$MODEL_NAME_OR_PATH" \
  --attn_impl flash_attention_2 \
  --beacon_cache_mode latent --beacon_decode_backend latent \
  --beacon_decode_graph --prompt_file /path/to/context.txt
```

CUDA CLI 默认开启 Graph 和编译。`--no-beacon_decode_graph` 关闭 Graph，`--no-beacon_decode_graph_full` 只保留局部层 Graph，`--no-beacon_decode_compile` 关闭编译，`--no-beacon_prefill_fast` 关闭 prefill 优化。编译目前仅用于 Torch 2.11 下启用 Graph 的 B1 BF16 FA2 decode；训练、更大 batch 和不支持的配置使用原路径。首次编译有额外启动耗时。Python API 保持原默认设置，使用时按[英文示例](../quickstart.md#python-api)显式选择优化。两条路径的 prefill 都在原始维度计算，符合条件的 decode 直接进行 latent Beacon attention。`beacon_decode_backend=latent` 使用所选 `attn_impl`；`reconstruct` 先重建 Beacon K/V，旧值 `fa2` 显式使用 latent FA2 decode。支持等长、无 padding、插入计划同步的 dense batch，不同长度请求的服务另见实验性 [vLLM 接入](vllm.md)，Qwen latent decode 暂不可用。CLI 演示单请求。

无需 checkpoint 的 smoke：`python -m unittest discover -s tests -v`。其中真实 CLI 测试会生成临时小 checkpoint/tokenizer 并离线加载，不证明训练权重的质量。

## Decode token 压缩

`beacon_decode_compress` 默认 `True`：prefill 和 decode 都按 `beacon_ratio` 插入 Beacon。设置 `False` 后，prefill 仍压缩输入，decode 新增 token 保留原始维度 raw KV：

```bash
python -m main.inference.infer \
  --model_name_or_path "$MODEL_NAME_OR_PATH" \
  --no-beacon_decode_compress --prompt_file /path/to/context.txt
```

Python API 使用 `ModelArgs(beacon_decode_compress=False)`。保留已有 prompt Beacon 和未完成的 prompt 窗口，后续 raw KV 随生成长度增长。历史 Beacon 继续使用所选重建或 latent attention 后端。训练忽略这个推理选项。decode 开始后切换设置，需要先 reset Memory。不同压缩策略的输出可能不同，应单独比较精度。
