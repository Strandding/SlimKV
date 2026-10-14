# 快速开始

[English](../quickstart.md)

需要本地训练后的 SlimKV checkpoint：`config.json`、完整 safetensors/PyTorch shards 和 index、tokenizer。`beacon_*` 字段与权重键保留；已有目录名不用改。本仓库不提供模型权重，请使用已有 SlimKV checkpoint 或按[训练说明](training.md)自行训练。

```bash
python -m main.infer \
  --model_name_or_path /path/to/slimkv-checkpoint \
  --prompt 'Explain why a KV cache helps language model inference.' \
  --max_new_tokens 32
```

默认 cuda:0、BF16、eager、greedy、auto cache 与重建 decode，不覆盖 rank/ratio/window/RoPE。CPU 可设 `--device cpu --dtype fp32`；Qwen2 模板为 `--chat_template qwen`，缓存限制见 [实现状态](implementation_status.md)。长 UTF-8 上下文使用 `--prompt_file`。短输入可能不触发完整 Beacon 窗口。

[英文指南的 Python 示例](../quickstart.md#python-api) 与 CLI 使用同一真实 API：`ModelArgs`、`get_model_and_tokenizer`、`apply_chat_template`、`model.generate`。各独立请求之间、切换后端之前调用 `model.memory.reset()`。reset 不一定释放 Graph 工作区。

安装 FA2 和所需扩展后可执行：

```bash
export SLIMKV_FA2_DECODE_EXTENSION_DIR="$PWD/.cache/fa2_decode_320"
python -m main.infer \
  --model_name_or_path /path/to/slimkv-checkpoint \
  --attn_impl flash_attention_2 \
  --beacon_cache_mode latent --beacon_decode_backend fa2 \
  --beacon_decode_graph --prompt_file /path/to/context.txt
```

Graph 默认关闭；启用后只是局部 norm/MLP 图。Prefill 仍在原始维度计算；符合条件的 decode 才直接进行 latent Beacon attention。支持等长、无 padding、插入计划同步的 dense batch，不支持 ragged/paged serving 或 Qwen latent decode。CLI 演示单请求。

无需 checkpoint 的 smoke：`python -m unittest discover -s tests -v`。其中真实 CLI 测试会生成临时小 checkpoint/tokenizer 并离线加载，不证明训练权重的质量。
