# 仓库结构

[English](../structure.md)

运行实验从 `scripts/` 找启动脚本，查看对应 Python 入口到 `main/`，查看模型实现到 `slimkv/`。入口和工具按用途分组，目录名称固定，不包含实验时间戳。

```text
slimkv/                 模型与缓存实现
  llama3/               Llama 训练、prefill、latent 与重建 decode
  qwen2/                Qwen2/Qwen2.5 模型实现
  vllm_plugin/          实验性分页缓存与连续调度
main/                   Python 应用入口
  conversion/           SVD 初始化和投影转换
  training/             训练与预训练数据处理
  inference/            独立推理与 vLLM 推理
  evaluation/           LongBench、NIAH 等精度评估
  analysis/             Beacon 探针、注意力热图与绘图
scripts/                按任务分类的 shell 启动预设
  conversion/
  training/
  evaluation/
  benchmark/
  slurm/                调度器模板
  vllm/                 独立环境的基线工具
tools/                 开发与测量工具
  benchmark/            Decode 测速、batch 扩展与 FA2 profiling
  data/                 LongBench 下载和确定性样本准备
  validation/           GPU 数值与通信检查
config/
  data/                 训练数据混合配置
  accelerate/           FSDP 与推理启动配置
  deepspeed/            DeepSpeed 训练配置
  environments/         CPU 依赖约束
kernels/fa2_decode/     可选 SM90 width-320 FA2 扩展
integrations/vllm/      独立 vLLM 安装配置
tests/                  自动回归测试
docs/                   英文指南、zh/ 中文翻译和许可文本
```

## 常用入口

| 任务 | 启动脚本或模块 |
|---|---|
| 初始化低秩 Beacon 权重 | `scripts/conversion/convert_kv_lowrank.sh` |
| Llama 训练 / 微调 | `scripts/training/train_llama_8B.sh` / `finetune_llama_8B.sh` |
| 独立推理 | `python -m main.inference.infer` |
| vLLM 推理 | `python -m main.inference.infer_vllm` |
| LongBench-v1 精度评估 | `scripts/evaluation/eval_longbench_slimkv.sh` |
| 查看 LongBench 预处理和评分 | `main/evaluation/eval_longbench.py`、`longbench_utils.py` |
| 32K–128K Full KV、重建与 latent 测速 | `scripts/benchmark/benchmark_context_decode.sh` |
| 查看固定上下文测速实现 | `tools/benchmark/benchmark_context_decode.py` |
| FA2 profiling | `python -m tools.benchmark.profile_fa2` |
| 编译可选 FA2 扩展 | `python -m tools.build_fa2_decode` |
| 检查仓库语法、导入和链接 | `python tools/check_repository.py` |
| 保存运行配置和源码哈希 | `python -m tools.record_run` |

启动脚本提供模型、数据和输出路径配置，调用对应 Python 模块，不承载模型算法。

## 模型内部

`slimkv/__init__.py` 导出 loader 并分发到 Llama3 或 Qwen2。共享参数、数据处理、Trainer 和 Beacon 窗口规划分别在 `args.py`、`data.py`、`trainer.py` 和 `memory.py`。

`slimkv/llama3/modeling_llama.py` 包含投影与模型 forward，`memory.py` 管理压缩存储和重建视图。`latent_decode.py`、`fa2_decode.py` 实现直接 latent attention；`decode_ops.py` / `decode_kernels.py`、`static_decode.py`、`decode_graph.py`、`compile_decode.py` 提供可选 decode 加速。`prefill_ops.py`、`prefill_kernels.py` 实现原始维度的 prefill 快路径。

## 生成文件

权重、数据集、预测、profiler trace 和编译产物由 `.gitignore` 排除。本地配置与笔记放入 `.local/`，个人忽略项可写入 `.git/info/exclude`。
