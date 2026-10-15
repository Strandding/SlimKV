# 安装

[English](../installation.md)

推荐独立 Python 3.11 环境，包要求最低 3.10。模型和配置接口基于 Transformers 4.40.x，已验证 4.40.2；不要单独升级 Transformers。

## Eager 推理和 CPU 验证

先安装适合硬件的 PyTorch CUDA wheel 和驱动，再从根目录安装：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e . "huggingface_hub[cli]>=0.34,<1"
python -m pip check
python -m main.inference.infer --help
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

二进制需匹配 Torch、Python ABI、C++ ABI 和 CUDA。Eager 不需 FA2。Checkpoint 的 rank 超过 256 时，latent FA2 还需要 [width-320 扩展](../benchmark.md#building-the-width-320-extension)，支持 BF16、SM90 和 SM120 forward，不是 A100 或训练 kernel，SM120 安装步骤见[下文](#rtx-pro-5000sm120)。

vLLM 基线使用单独环境，尚无 SlimKV paged cache 集成。NIAH judge 的可选依赖为 `.[needle-judge]`，其当前协议不同于论文。

## RTX PRO 5000（SM120）

RTX PRO 5000 Blackwell 需要 **CUDA 12.8 或更新的开发工具链**。本次独立推理环境使用 Python **3.12**、Torch **2.11.0+cu130**、Transformers **4.40.2**、FlashAttention **2.8.3**、Triton **3.6.0** 和 CUDA toolkit **13.3**。训练及 vLLM 继续使用各自的独立环境。

下列命令假定使用 CUDA 13.x 工具链。若使用其他版本，先选择与工具链匹配的 PyTorch CUDA wheel 再编译。`--cuda_arch` 中的 `12.0` 是 GPU 计算能力，不是工具链版本。

在新 clone 的仓库根目录安装 CUDA 13 版本的 PyTorch，并为 SM120 编译 FA2：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -e . "huggingface_hub[cli]>=0.34,<1" ninja packaging
FLASH_ATTENTION_FORCE_BUILD=TRUE FLASH_ATTN_CUDA_ARCHS=120 MAX_JOBS=4 \
  python -m pip install flash-attn==2.8.3 --no-build-isolation

git clone --depth 1 --branch v2.8.3 \
  https://github.com/Dao-AILab/flash-attention.git .cache/flash-attention-v2.8.3
git -C .cache/flash-attention-v2.8.3 submodule update --init csrc/cutlass
python -m tools.build_fa2_decode \
  --fa2_source .cache/flash-attention-v2.8.3 \
  --output_dir .cache/fa2_decode_sm120 --cuda_arch 12.0 --jobs 2
export SLIMKV_FA2_DECODE_EXTENSION_DIR="$PWD/.cache/fa2_decode_sm120"
python -m pip check
```

找不到 `nvcc` 时，将 `CUDA_HOME` 设为工具链根目录。主机内存较小时降低 `MAX_JOBS`。编译只需在首次安装时执行，后续复用环境和构建目录。接着执行[下载权重](README.md#2-下载权重)，LongBench 和测速沿用同样的入口。

扩展在 SM90 上使用原来的 120 KiB width-320 内核，在单 block 共享内存上限为 99 KiB 的 SM120 上使用 90 KiB 的 query tile。两者均保留训练秩、BF16 输入、原 attention scale、64-key tile 和 split-combine 归约方式。

未指定 `--cuda_arch` 或 `TORCH_CUDA_ARCH_LIST` 时，构建脚本自动识别当前 GPU。工具链兼容时，可用 `--cuda_arch '9.0;12.0'` 同时编译两种架构。目标架构列表不能消除 PyTorch、Python 或 CUDA ABI 差异，不同环境仍需分别编译。

公开模型下载无需 Hugging Face token。如果保存的过期 token 导致 HTTP 401，可在下载命令前添加 `HF_HUB_DISABLE_IMPLICIT_TOKEN=1` 后重试。
