#!/bin/bash
#SBATCH -p NH100q
#SBATCH -w node07
#SBATCH -J train_llama31_8b_r2_kchunkfixed
#SBATCH -c 16
#SBATCH -o /export/home2/zihan/activation_beacon/data/outputs/logs/train_llama31_8b_r2_kchunkfixed.%j.out
#SBATCH -e /export/home2/zihan/activation_beacon/data/outputs/logs/train_llama31_8b_r2_kchunkfixed.%j.err
#SBATCH --time=UNLIMITED
#SBATCH --dependency=afterany:435968

set -euo pipefail

echo "Job started on $(hostname) at $(date)"

cd /export/home2/zihan/activation_beacon
source /export/home2/zihan/miniconda3/etc/profile.d/conda.sh
conda activate beacon

CUDA_VISIBLE_DEVICES=4,5,6,7 \
NPROC_PER_NODE=4 \
PER_DEVICE_TRAIN_BATCH_SIZE=1 \
GRADIENT_ACCUMULATION_STEPS=2 \
MODEL_NAME_OR_PATH=/export/home2/zihan/activation_beacon/data/outputs/beacon-llama31-8b-lowrank-r2-nokropek \
BEACON_KV_COMPRESSION=lowrank_svd \
BEACON_KV_LATENT_COMPRESS_RATIO=2 \
BEACON_SKIP_ROPE_FOR_BEACON_K=False \
BEACON_CHUNK_FIXED_ROPE_FOR_BEACON_K=True \
BEACON_ENHANCE_K_POS_RESIDUAL=False \
BEACON_SHARED_KV_DOWN_PROJ=False \
BEACON_HEADWISE_UP_PROJ=False \
BEACON_UP_PROJ_GROUP_SIZE=1 \
TOTAL_TRAIN_SAMPLES=24000 \
OUTPUT_NAME=beacon-llama31-8b-pretrain-lowrank-r2-kchunkfixedrope \
OCCUPY_AFTER_TRAIN=False \
bash /export/home2/zihan/activation_beacon/train_llama_8B.sh

echo "Job finished at $(date)"
