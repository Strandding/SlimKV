#!/usr/bin/env bash

set -euo pipefail

ENV_NAME="${ENV_NAME:-beacon-vllm}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
PYTORCH_CUDA_CHANNEL="${PYTORCH_CUDA_CHANNEL:-cu121}"
PIP_TMP_DIR="${PIP_TMP_DIR:-/export/home2/zihan/pip_tmp}"

mkdir -p "${PIP_TMP_DIR}"
export TMPDIR="${PIP_TMP_DIR}"
export TMP="${PIP_TMP_DIR}"
export TEMP="${PIP_TMP_DIR}"
export PIP_CACHE_DIR="${PIP_TMP_DIR}"

echo "[setup] pip temp/cache dir: ${PIP_TMP_DIR}"

echo "[setup] creating conda env: ${ENV_NAME} (python=${PYTHON_VERSION})"
conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"

echo "[setup] activating env: ${ENV_NAME}"
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

python -m pip install --upgrade pip setuptools wheel

echo "[setup] installing pytorch (${PYTORCH_CUDA_CHANNEL})"
python -m pip install \
  torch torchvision torchaudio \
  --index-url "https://download.pytorch.org/whl/${PYTORCH_CUDA_CHANNEL}"

echo "[setup] installing core dependencies"
python -m pip install \
  "vllm>=0.6.0" \
  "transformers>=4.45.0" \
  "accelerate>=1.0.0" \
  "datasets>=2.20.0" \
  "tqdm" \
  "sentencepiece" \
  "protobuf"

echo "[setup] installing project utilities"
python -m pip install \
  pandas seaborn rouge fuzzywuzzy jieba python-Levenshtein

echo "[setup] done."
echo "[setup] verify: conda activate ${ENV_NAME} && python -c 'import torch, vllm; print(torch.__version__)'"
