# Installation

[Chinese](zh/installation.md)
Use an isolated Python 3.11 environment (package minimum: 3.10). Model/config code is based on Transformers 4.40.x; use **4.40.2**. Do not upgrade Transformers independently of checkpoint and model-code validation.

## Eager inference and CPU checks

Install PyTorch for your hardware before installing the project. For GPU inference, use the appropriate official PyTorch CUDA wheel and a compatible NVIDIA driver. Then:

```bash
python -m venv .venv
source .venv/bin/activate
# Install a hardware-compatible torch wheel here.
python -m pip install -e . "huggingface_hub[cli]>=0.34,<1"
python -m pip check
python -m main.inference.infer --help
```

A CPU environment used for CI and local regression is reproducible with:

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -c config/environments/cpu.txt -e .
python tools/check_repository.py
python -m unittest discover -s tests -v
```

CPU tests instantiate small random models and require no weight or dataset download. CPU eager inference can use `--device cpu --dtype fp32`; an 8B checkpoint is still large and requires sufficient RAM.

## Training

```bash
python -m pip install -e '.[training]'
```

Training uses DeepSpeed and optional PEFT (0.10.x). Match the CUDA toolkit/nvcc to PyTorch, and set `CUDA_HOME` to the toolkit root. DeepSpeed compatibility must be checked in your chosen training environment; the CPU CI does not test it. No single validated training lockfile or complete paper training environment has been released. Activate the desired CUDA environment first. Override `SLIMKV_CUDA_MODULE`, or set it to an empty string to skip module loading; see [training](training.md).

## Optional FA2 inference

One tested FA2 runtime uses Torch **2.11.0+cu129**, Transformers **4.40.2**, FlashAttention **2.8.3**, Triton **3.6.0**, and CUDA toolkit **12.9**. Match the Torch/CUDA build to your GPU and driver.

Install `flash-attn==2.8.3` after installing a compatible Torch/CUDA toolchain. Source installation needs `ninja`, `packaging`, CUDA development headers and substantial RAM:

```bash
python -m pip install ninja packaging wheel setuptools
python -m pip install flash-attn==2.8.3 --no-build-isolation
python -c 'import torch, flash_attn; print(torch.__version__, torch.version.cuda, flash_attn.__version__)'
```

PyTorch version, Python ABI, C++ ABI, and CUDA compatibility all matter for binary reuse. Eager inference needs neither FA2 nor the custom extension. For checkpoint ranks above 256, latent FA2 decode also requires the [width-320 build](benchmark.md#building-the-width-320-extension). That extension supports BF16 forward on SM90 and SM120; it is not a training/backward kernel or an A100 build. See the [RTX PRO 5000 setup](#rtx-pro-5000-sm120).

## Other environments

Keep vLLM in a separate environment: `scripts/vllm/setup_longbench_vllm_env.sh` and `scripts/vllm/eval_longbench_vllm.sh` are baseline utilities, not a SlimKV paged-cache integration. Needle judging has optional dependencies via `.[needle-judge]`; its current protocol differs from the manuscript.

## RTX PRO 5000 (SM120)

For RTX PRO 5000 Blackwell, use a CUDA development toolkit **12.8 or newer**. The isolated inference environment tested here uses Python **3.12**, Torch **2.11.0+cu130**, Transformers **4.40.2**, FlashAttention **2.8.3**, Triton **3.6.0** and CUDA toolkit **13.3**. Keep existing training or vLLM environments separate.

The commands below assume a CUDA 13.x toolkit. For another toolkit version, choose the matching PyTorch CUDA wheel before compiling. `12.0` in `--cuda_arch` is the GPU compute capability, not the toolkit version.

From a fresh clone, install the CUDA 13 PyTorch wheel and compile FA2 for SM120:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -e . "huggingface_hub[cli]>=0.34,<1" ninja packaging wheel setuptools
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

Set `CUDA_HOME` to your toolkit root if `nvcc` is not found. Reduce `MAX_JOBS` on hosts with limited RAM. Compilation is a one-time setup step; reuse the environment and build directory for later runs. Continue with [checkpoint download](../README.md#2-download-the-checkpoint), then the same LongBench and speed commands.

The extension selects the original 120 KiB width-320 kernel on SM90 and a 90 KiB query tile on SM120, whose per-block limit is 99 KiB. Both use the trained ranks, BF16 inputs, original attention scale, 64-key tiles and the same split-combine reductions.

The builder detects the current GPU unless `--cuda_arch` or `TORCH_CUDA_ARCH_LIST` is set. Use `--cuda_arch '9.0;12.0'` to compile both targets with a compatible toolkit. A CUDA target list does not make a binary portable across different PyTorch/Python/CUDA ABIs; rebuild in each environment.

Public model downloads do not need a Hugging Face token. If an expired saved token causes HTTP 401, retry the download with `HF_HUB_DISABLE_IMPLICIT_TOKEN=1`.
