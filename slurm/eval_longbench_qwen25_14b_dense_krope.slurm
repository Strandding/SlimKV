#!/bin/bash
#SBATCH -p NH100q
#SBATCH -w node07
#SBATCH -J eval_qwen25_14b_dense_krope
#SBATCH -c 16
#SBATCH -o /export/home2/zihan/activation_beacon/data/outputs/logs/eval_qwen25_14b_dense_krope.%j.out
#SBATCH -e /export/home2/zihan/activation_beacon/data/outputs/logs/eval_qwen25_14b_dense_krope.%j.err
#SBATCH --time=UNLIMITED
#SBATCH --dependency=afterany:435560

set -euo pipefail

echo "Job started on $(hostname) at $(date)"

cd /export/home2/zihan/activation_beacon

source /export/home2/zihan/miniconda3/etc/profile.d/conda.sh
conda activate beacon

CUDA_VISIBLE_DEVICES=4,5,6,7 \
NPROC_PER_NODE=4 \
MASTER_PORT=29552 \
MODEL_NAME_OR_PATH=/export/home2/zihan/activation_beacon/data/outputs/beacon-qwen25-14b-finetune-dense-krope-481632/checkpoint-2000 \
EVAL_MODE=beacon8 \
BEACON_RATIO=16 \
BEACON_RATIO_MIX=sequence \
BEACON_SKIP_ROPE_FOR_BEACON_K=False \
CHAT_TEMPLATE=qwen \
ATTN_IMPL=flash_attention_2 \
DTYPE=bf16 \
BATCH_SIZE=1 \
MAX_LENGTH=31500 \
OUTPUT_NAME=beacon-qwen25-14b-tokenr16-dense-krope \
OCCUPY_AFTER_EVAL=False \
bash /export/home2/zihan/activation_beacon/eval_longbench.sh

echo "Job finished at $(date)"
