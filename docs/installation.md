# Installation

[中文](zh/installation.md)
Use an isolated Python 3.11 environment (package minimum: 3.10). Model/config code is based on Transformers 4.40.x; install **4.40.2** for the recorded environments. Do not upgrade Transformers independently of checkpoint and model-code validation.

## Eager inference and CPU checks

Install PyTorch for your hardware before installing the project. For GPU inference, use the appropriate official PyTorch CUDA wheel and a compatible NVIDIA driver. Then:

```bash
python -m venv .venv
source .venv/bin/activate
# Install a hardware-compatible torch wheel here.
python -m pip install -e .
python -m pip check
python -m main.infer --help
```

A CPU environment used for CI and local regression is reproducible with:

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -c requirements/cpu.txt -e .
python tools/check_repository.py
python -m unittest discover -s tests -v
```

CPU tests instantiate small random models and require no weight or dataset download. CPU eager inference can use `--device cpu --dtype fp32`; an 8B checkpoint is still large and requires sufficient RAM.

## Training

```bash
python -m pip install -e '.[training]'
```

Training uses DeepSpeed and optional PEFT (0.10.x). Match the CUDA toolkit/nvcc to PyTorch, and set `CUDA_HOME` to the toolkit root. DeepSpeed compatibility must be checked in your chosen training environment; the CPU CI does not test it. No single validated training lockfile or complete paper training environment has been released. Presets contain cluster CUDA module names; activate the desired environment first, or override `SLIMKV_CUDA_MODULE` as described in [training](training.md).

## Optional FA2 inference

The recorded H20 GPU environment uses Torch **2.11.0+cu129**, Transformers **4.40.2**, FlashAttention **2.8.3**, Triton **3.6.0**, and CUDA toolkit **12.9**. A compatible prebuilt FA2 binary was reused; this is a tested runtime configuration, not a universal installation guarantee.

Install `flash-attn==2.8.3` after installing a compatible Torch/CUDA toolchain. Source installation needs `ninja`, `packaging`, CUDA development headers and substantial RAM:

```bash
python -m pip install ninja packaging
python -m pip install flash-attn==2.8.3 --no-build-isolation
python -c 'import torch, flash_attn; print(torch.__version__, torch.version.cuda, flash_attn.__version__)'
```

PyTorch version, Python ABI, C++ ABI, and CUDA compatibility all matter for binary reuse. Eager inference needs neither FA2 nor the custom extension. The f4 checkpoint has ranks above 256, so latent FA2 decode also requires the [width-320 build](benchmark.md#building-the-width-320-extension). That extension supports BF16 forward on SM90 only; it is not a training/backward kernel or an A100 build.

## Other environments

Keep vLLM in a separate environment: `scripts/vllm/setup_longbench_vllm_env.sh` and `scripts/vllm/eval_longbench_vllm.sh` are baseline utilities, not a SlimKV paged-cache integration. Needle judging has optional dependencies via `.[needle-judge]`; its current protocol differs from the manuscript.
