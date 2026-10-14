# 模块和工具

[English](../structure.md)

模型与命令模块保留现有路径，兼容已有导入和 checkpoint。重复的评测启动逻辑已合并，文档、示例和依赖约束按用途归拢。

- `slimkv/__init__.py`：公共API、checkpoint loader、架构分发。
- `slimkv/args.py/data.py/trainer.py`：参数、数据、Trainer。
- `slimkv/memory.py`：窗口与Beacon计划；`llama3/memory.py`：latent存储/重建视图/eligibility/bookkeeping。
- `llama3/modeling_llama.py`：投影、attention、forward；fa2_decode/decode_ops/decode_kernels/decode_graph提供可选backend。
- `qwen2/llama/mistral`：其他既有路径，支持范围见状态文档。
- `latency`：独立注意力研究原型，使用单独的 loader 与缓存约定。
- `main`：infer/train/conversion/eval命令；`scripts`按任务分组，共用common.sh。Qwen 和 Llama 的 LongBench 预设共用 `scripts/evaluation/longbench_common.sh`。
- `config`：data/deepspeed；`config/environments`：CPU 验证依赖约束；`kernels/fa2_decode`：BF16 SM90 width320 forward。
- `tests`：CPU回归；`docs`：英文主文档、zh 中文主页和指南、licenses 许可文本。

通用维护：check_repository（语法/导入/链接/注释/产物）、record_run（provenance）。Fixture准备：prepare_longbench。当前性能：benchmark_reconstruction、benchmark_fa2_decode及compare_b1/reference控件、benchmark_batch_decode。GPU开发：build/validate_fa2_decode、validate_decode_ops、validate_b1_decode、profile_fa2。NCCL 与任务结束后的 occupancy 工具保留，供已有启动脚本调用。

Slurm 模板需适配集群。vLLM 目录提供独立环境的 Full KV 基线，没有 SlimKV paged cache。

## 公开与本地文件

公开保留模型和 kernel、命令入口、启动预设、配置模板、CPU 测试/CI、用户指南、方法图和许可文本。Benchmark 构建与验证工具也保留，方便使用者检查自己的环境。

机器相关的开发记录、验收 receipts、源码同步说明、早期 Anchor 原型、未使用的 GPU 监测脚本和旧 InfBench 样例通过 `.gitignore` 留在本地。权重、数据集、预测输出、profiler traces 和编译产物同样不公开跟踪。仓库检查只覆盖公开文件。
