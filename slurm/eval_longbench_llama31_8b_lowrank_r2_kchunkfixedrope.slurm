#!/bin/bash
#SBATCH -p NH100q
#SBATCH -w node07
#SBATCH -J eval_llama31_8b_r2_kchunkfixed
#SBATCH -c 16
#SBATCH -o /export/home2/zihan/activation_beacon/data/outputs/logs/eval_llama31_8b_r2_kchunkfixed.%j.out
#SBATCH -e /export/home2/zihan/activation_beacon/data/outputs/logs/eval_llama31_8b_r2_kchunkfixed.%j.err
#SBATCH --time=UNLIMITED
#SBATCH --dependency=afterany:435978

set -euo pipefail

echo "Job started on $(hostname) at $(date)"

cd /export/home2/zihan/activation_beacon
source /export/home2/zihan/miniconda3/etc/profile.d/conda.sh
conda activate beacon

CUDA_VISIBLE_DEVICES=4,5,6,7 \
NPROC_PER_NODE=4 \
MASTER_PORT=29561 \
MODEL_NAME_OR_PATH=/export/home2/zihan/activation_beacon/data/outputs/beacon-llama31-8b-finetune-lowrank-r2-kchunkfixedrope/checkpoint-2000 \
EVAL_MODE=beacon8 \
BEACON_RATIO=8 \
BEACON_RATIO_MIX=sequence \
BEACON_SKIP_ROPE_FOR_BEACON_K=False \
BEACON_CHUNK_FIXED_ROPE_FOR_BEACON_K=True \
BEACON_ENHANCE_K_POS_RESIDUAL=False \
BEACON_SHARED_KV_DOWN_PROJ=False \
BEACON_HEADWISE_UP_PROJ=False \
BEACON_UP_PROJ_GROUP_SIZE=1 \
CHAT_TEMPLATE=llama-3 \
ATTN_IMPL=flash_attention_2 \
DTYPE=bf16 \
BATCH_SIZE=1 \
MAX_LENGTH=31500 \
OUTPUT_NAME=beacon-llama31-8b-finetune-lowrank-r2-kchunkfixedrope \
OCCUPY_AFTER_EVAL=False \
bash /export/home2/zihan/activation_beacon/eval_longbench_llama31_8B.sh

echo "Job finished at $(date)"
