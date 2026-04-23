#!/usr/bin/env bash

if type module &>/dev/null; then
  module load cuda12.8/toolkit/12.8.1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}"
MODEL_OR_CKPT="${MODEL_OR_CKPT:-}"
DENSE_MODEL_OR_CKPT="${DENSE_MODEL_OR_CKPT:-/export/home2/zihan/activation_beacon/data/outputs/beacon-llama31-8b-finetune-dense/checkpoint-2000}"
LOWRANK_MODEL_OR_CKPT="${LOWRANK_MODEL_OR_CKPT:-/export/home2/zihan/activation_beacon/data/outputs/beacon-llama31-8b-finetune-lowrank-r2-nokropek-noshareddown/checkpoint-2000}"
MODE="${MODE:-all}"   # all | raw | beacon_dense | beacon_lowrank_reconstruct | beacon_lowrank_hybrid_latent
DTYPE="${DTYPE:-bf16}"
ATTN_IMPL="${ATTN_IMPL:-eager}"
PROMPT_LEN="${PROMPT_LEN:-8192}"
GEN_LEN="${GEN_LEN:-128}"
WARMUP="${WARMUP:-1}"
REPEAT="${REPEAT:-3}"
BEACON_KV_LATENT_COMPRESS_RATIO="${BEACON_KV_LATENT_COMPRESS_RATIO:-2}"
BEACON_RATIO="${BEACON_RATIO:-8}"

RUN_NAME="${RUN_NAME:-latency-llama31-8b-${MODE}-p${PROMPT_LEN}-g${GEN_LEN}-r${REPEAT}}"
OUTPUT_DIR="data/results/latency/${RUN_NAME}"
mkdir -p "${OUTPUT_DIR}"
LOG_FILE="${OUTPUT_DIR}/benchmark.log"
JSON_FILE="${OUTPUT_DIR}/benchmark.json"

if [[ -z "${DENSE_MODEL_OR_CKPT}" && -z "${LOWRANK_MODEL_OR_CKPT}" ]]; then
  echo "ERROR: Please set DENSE_MODEL_OR_CKPT and/or LOWRANK_MODEL_OR_CKPT (or MODEL_OR_CKPT as fallback)." >&2
  exit 1
fi

echo "========== Latency Benchmark ==========" | tee "${LOG_FILE}"
echo "dense_model_or_ckpt: ${DENSE_MODEL_OR_CKPT}" | tee -a "${LOG_FILE}"
echo "lowrank_model_or_ckpt: ${LOWRANK_MODEL_OR_CKPT}" | tee -a "${LOG_FILE}"
echo "mode: ${MODE}" | tee -a "${LOG_FILE}"
echo "dtype: ${DTYPE}" | tee -a "${LOG_FILE}"
echo "attn_impl: ${ATTN_IMPL}" | tee -a "${LOG_FILE}"
echo "prompt_len: ${PROMPT_LEN}" | tee -a "${LOG_FILE}"
echo "gen_len: ${GEN_LEN}" | tee -a "${LOG_FILE}"
echo "repeat: ${REPEAT}, warmup: ${WARMUP}" | tee -a "${LOG_FILE}"
echo "beacon_ratio: ${BEACON_RATIO}" | tee -a "${LOG_FILE}"
echo "latent_compress_ratio: ${BEACON_KV_LATENT_COMPRESS_RATIO}" | tee -a "${LOG_FILE}"
echo "output_json: ${JSON_FILE}" | tee -a "${LOG_FILE}"

bench_args=(
  --dense_model_or_ckpt "${DENSE_MODEL_OR_CKPT}"
  --lowrank_model_or_ckpt "${LOWRANK_MODEL_OR_CKPT}"
  --mode "${MODE}"
  --dtype "${DTYPE}"
  --attn_impl "${ATTN_IMPL}"
  --prompt_len "${PROMPT_LEN}"
  --gen_len "${GEN_LEN}"
  --warmup "${WARMUP}"
  --repeat "${REPEAT}"
  --beacon_ratio "${BEACON_RATIO}"
  --beacon_kv_latent_compress_ratio "${BEACON_KV_LATENT_COMPRESS_RATIO}"
  --output_json "${JSON_FILE}"
)
if [[ -n "${MODEL_OR_CKPT}" ]]; then
  bench_args+=(--model_or_ckpt "${MODEL_OR_CKPT}")
fi

python -m main.benchmark_latency "${bench_args[@]}" 2>&1 | tee -a "${LOG_FILE}"
