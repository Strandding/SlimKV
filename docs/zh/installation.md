# 安装

[English](../installation.md)

推荐独立 Python 3.11 环境，包要求最低 3.10。模型和配置接口基于 Transformers 4.40.x，已验证 4.40.2；不要单独升级 Transformers。

## Eager 推理和 CPU 验证

先安装适合硬件的 PyTorch CUDA wheel 和驱动，再从根目录安装：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m pip check
python -m main.infer --help
```

上述命令假定已在环境中安装匹配硬件的 Torch。CPU CI 使用：

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -c config/environments/cpu.txt -e .
python tools/check_repository.py
python -m unittest discover -s tests -v
```

测试用随机小模型，不下载权重和数据。CPU 推理可设 `--device cpu --dtype fp32`，但 8B 权重仍需足够内存。

## 训练

`python -m pip install -e '.[training]'` 安装 DeepSpeed 和可选 PEFT 0.10.x。CUDA toolkit/nvcc 要匹配 PyTorch；`CUDA_HOME` 指向 toolkit 根目录。CPU CI 不验证 DeepSpeed，尚未提供完整论文训练锁定环境。已有脚本的 CUDA module 可用 `SLIMKV_CUDA_MODULE` 覆盖，设为空则使用当前环境。

## 可选 FA2

一组已验证的 FA2 运行配置为 Torch 2.11.0+cu129、Transformers 4.40.2、FlashAttention 2.8.3、Triton 3.6.0、CUDA toolkit 12.9。请选择匹配 GPU 和驱动的 Torch/CUDA 构建。

```bash
python -m pip install ninja packaging
python -m pip install flash-attn==2.8.3 --no-build-isolation
python -c 'import torch, flash_attn; print(torch.__version__, torch.version.cuda, flash_attn.__version__)'
```

二进制需匹配 Torch、Python ABI、C++ ABI 和 CUDA。Eager 不需 FA2。Checkpoint 的 rank 超过 256 时，latent FA2 还需要 [width-320 扩展](../benchmark.md#building-the-width-320-extension)，仅支持 BF16、SM90 forward，不是 A100 或训练 kernel。

vLLM 基线使用单独环境，尚无 SlimKV paged cache 集成。NIAH judge 的可选依赖为 `.[needle-judge]`，其当前协议不同于论文。
