#!/bin/bash
#SBATCH -p HPCAIq
#SBATCH -w node14
#SBATCH -J eval_qwen25_14b_lowrank_r2_nokrope
#SBATCH -c 16
#SBATCH -o /export/home2/zihan/activation_beacon/data/outputs/logs/eval_qwen25_14b_lowrank_r2_nokrope.%j.out
#SBATCH -e /export/home2/zihan/activation_beacon/data/outputs/logs/eval_qwen25_14b_lowrank_r2_nokrope.%j.err
#SBATCH --time=UNLIMITED
#SBATCH --dependency=afterany:433629

set -euo pipefail

echo "Job started on $(hostname) at $(date)"

cd /export/home2/zihan/activation_beacon

source /export/home2/zihan/miniconda3/etc/profile.d/conda.sh
conda activate beacon

CUDA_VISIBLE_DEVICES=4,6,7 \
NPROC_PER_NODE=3 \
MASTER_PORT=29551 \
MODEL_NAME_OR_PATH=/export/home2/zihan/activation_beacon/data/outputs/beacon-qwen25-14b-finetune-lowrank-r2-nokrope/checkpoint-2000 \
EVAL_MODE=beacon8 \
BEACON_RATIO=8 \
BEACON_RATIO_MIX=sequence \
CHAT_TEMPLATE=qwen \
ATTN_IMPL=flash_attention_2 \
DTYPE=bf16 \
BATCH_SIZE=1 \
MAX_LENGTH=31500 \
OUTPUT_NAME=beacon-qwen25-14b-finetune-lowrank-r2-nokrope \
OCCUPY_AFTER_EVAL=False \
bash /export/home2/zihan/activation_beacon/eval_longbench.sh

echo "Job finished at $(date)"
