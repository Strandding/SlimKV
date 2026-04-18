#!/usr/bin/env bash
set -euo pipefail

# 在计算节点上运行此脚本。它只做离线 checkpoint 转换，不启动训练。

PYTHON_BIN="${PYTHON_BIN:-/export/home2/zihan/miniconda3/envs/beacon/bin/python}"
default_dense_dir="/dataset/common/tzh/.model/Qwen2-7B-Instruct"

BEACON_K_LATENT_HEAD_DIM="${BEACON_K_LATENT_HEAD_DIM:-64}"
BEACON_V_LATENT_HEAD_DIM="${BEACON_V_LATENT_HEAD_DIM:-64}"
BEACON_K_LATENT_HEAD_DIM_LIST="${BEACON_K_LATENT_HEAD_DIM_LIST:-}"
BEACON_V_LATENT_HEAD_DIM_LIST="${BEACON_V_LATENT_HEAD_DIM_LIST:-}"
DTYPE="${DTYPE:-fp32}"
SAFE_SERIALIZATION="${SAFE_SERIALIZATION:-true}"
SVD_DEVICE="${SVD_DEVICE:-cpu}"
SAVE_SVD_SPECTRA="${SAVE_SVD_SPECTRA:-true}"
SAVE_SVD_SPECTRA_DATA="${SAVE_SVD_SPECTRA_DATA:-true}"

if [[ -z "${MODEL_NAME_OR_PATH:-}" ]]; then
  if compgen -G "${default_dense_dir}/checkpoint-*" > /dev/null; then
    MODEL_NAME_OR_PATH="$(printf '%s\n' ${default_dense_dir}/checkpoint-* | sort -V | tail -n 1)"
  elif [[ -d "${default_dense_dir}" ]]; then
    MODEL_NAME_OR_PATH="${default_dense_dir}"
  else
    echo "Cannot find dense beacon checkpoint under ${default_dense_dir}. Set MODEL_NAME_OR_PATH manually." >&2
    exit 1
  fi
fi

OUTPUT_NAME="${OUTPUT_NAME:-beacon-qwen2-pretrain-lowrank-kv${BEACON_K_LATENT_HEAD_DIM}}"
OUTPUT_DIR="${OUTPUT_DIR:-data/outputs/${OUTPUT_NAME}}"

# 如果源 checkpoint 已经保存了 beacon 配置，这些参数通常不会改变行为；
# 这里显式传入是为了确保一定走本地 beacon 模型类。
beacon_args=(
  --enable_beacon
  --beacon_param q k v
)

echo "[convert_beacon_kv_lowrank] MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH}"
echo "[convert_beacon_kv_lowrank] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[convert_beacon_kv_lowrank] BEACON_K_LATENT_HEAD_DIM=${BEACON_K_LATENT_HEAD_DIM}"
echo "[convert_beacon_kv_lowrank] BEACON_V_LATENT_HEAD_DIM=${BEACON_V_LATENT_HEAD_DIM}"
echo "[convert_beacon_kv_lowrank] SVD_DEVICE=${SVD_DEVICE}"
echo "[convert_beacon_kv_lowrank] SAVE_SVD_SPECTRA=${SAVE_SVD_SPECTRA}"
echo "[convert_beacon_kv_lowrank] SAVE_SVD_SPECTRA_DATA=${SAVE_SVD_SPECTRA_DATA}"

latent_dim_args=()
if [[ -n "${BEACON_K_LATENT_HEAD_DIM_LIST}" || -n "${BEACON_V_LATENT_HEAD_DIM_LIST}" ]]; then
  if [[ -z "${BEACON_K_LATENT_HEAD_DIM_LIST}" || -z "${BEACON_V_LATENT_HEAD_DIM_LIST}" ]]; then
    echo "Invalid config: set both BEACON_K_LATENT_HEAD_DIM_LIST and BEACON_V_LATENT_HEAD_DIM_LIST, or neither." >&2
    exit 1
  fi
  read -r -a K_LAYERWISE_DIMS <<< "${BEACON_K_LATENT_HEAD_DIM_LIST}"
  read -r -a V_LAYERWISE_DIMS <<< "${BEACON_V_LATENT_HEAD_DIM_LIST}"
  if (( ${#K_LAYERWISE_DIMS[@]} == 0 || ${#V_LAYERWISE_DIMS[@]} == 0 )); then
    echo "Invalid BEACON_K_LATENT_HEAD_DIM_LIST / BEACON_V_LATENT_HEAD_DIM_LIST: empty after parsing." >&2
    exit 1
  fi
  if (( ${#K_LAYERWISE_DIMS[@]} != ${#V_LAYERWISE_DIMS[@]} )); then
    echo "Invalid config: K/V layerwise list lengths differ (${#K_LAYERWISE_DIMS[@]} vs ${#V_LAYERWISE_DIMS[@]})." >&2
    exit 1
  fi
  for dim in "${K_LAYERWISE_DIMS[@]}"; do
    if ! [[ "${dim}" =~ ^[1-9][0-9]*$ ]]; then
      echo "Invalid per-layer latent dim in BEACON_K_LATENT_HEAD_DIM_LIST: ${dim}" >&2
      exit 1
    fi
  done
  for dim in "${V_LAYERWISE_DIMS[@]}"; do
    if ! [[ "${dim}" =~ ^[1-9][0-9]*$ ]]; then
      echo "Invalid per-layer latent dim in BEACON_V_LATENT_HEAD_DIM_LIST: ${dim}" >&2
      exit 1
    fi
  done
  latent_dim_args+=(--beacon_k_latent_head_dim_list "${K_LAYERWISE_DIMS[@]}")
  latent_dim_args+=(--beacon_v_latent_head_dim_list "${V_LAYERWISE_DIMS[@]}")
  echo "[convert_beacon_kv_lowrank] BEACON_K_LATENT_HEAD_DIM_LIST=${BEACON_K_LATENT_HEAD_DIM_LIST}"
  echo "[convert_beacon_kv_lowrank] BEACON_V_LATENT_HEAD_DIM_LIST=${BEACON_V_LATENT_HEAD_DIM_LIST}"
else
  latent_dim_args+=(--beacon_k_latent_head_dim "${BEACON_K_LATENT_HEAD_DIM}")
  latent_dim_args+=(--beacon_v_latent_head_dim "${BEACON_V_LATENT_HEAD_DIM}")
fi

"${PYTHON_BIN}" -m main.convert_beacon_kv_lowrank \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --dtype "${DTYPE}" \
  "${latent_dim_args[@]}" \
  --svd_device "${SVD_DEVICE}" \
  --save_svd_spectra "${SAVE_SVD_SPECTRA}" \
  --save_svd_spectra_data "${SAVE_SVD_SPECTRA_DATA}" \
  --safe_serialization "${SAFE_SERIALIZATION}" \
  "${beacon_args[@]}"
