# 模块和工具

[English](../structure.md)

模型与命令模块保留现有路径，兼容已有导入和 checkpoint。重复的评测启动逻辑已合并，文档、示例和依赖约束按用途归拢。

- `slimkv/__init__.py`：公共API、checkpoint loader、架构分发。
- `slimkv/args.py/data.py/trainer.py`：参数、数据、Trainer。
- `slimkv/memory.py`：窗口与Beacon计划；`llama3/memory.py`：latent存储/重建视图/eligibility/bookkeeping。
- `llama3/modeling_llama.py`：投影、attention、forward；fa2_decode/decode_ops/decode_kernels/decode_graph提供可选backend。
- `qwen2/llama/mistral`：其他既有路径，支持范围见状态文档。
- `latency`：独立研究原型；`experiments/anchor_prototype`：历史实现，不参加安装。
- `main`：infer/train/conversion/eval命令；`scripts`按任务分组，共用common.sh。Qwen 和 Llama 的 LongBench 预设共用 `scripts/evaluation/longbench_common.sh`。
- `config`：data/deepspeed；`config/environments`：CPU 验证依赖约束；`kernels/fa2_decode`：BF16 SM90 width320 forward。
- `tests`：CPU回归；`docs`：英文主文档、zh 中文主页和指南、licenses 许可文本、examples 格式样例、archive 双语历史记录。

通用维护：check_repository（语法/导入/链接/注释/产物）、record_run（provenance）。Fixture准备：prepare_longbench。当前性能：benchmark_reconstruction、benchmark_fa2_decode及compare_b1/reference控件、benchmark_batch_decode。GPU开发：build/validate_fa2_decode、validate_decode_ops、validate_b1_decode、profile_fa2。GPU occupancy/通信监测工具是显式诊断，不是推理入口。

Slurm模板需集群适配。vLLM目录只提供独立环境的完整cache基线，没有SlimKV paged cache。模型、数据、results/outputs/fig、build/profile产物都不进入Git；仅docs/examples/infbench.json 为受版本管理的格式样例。
