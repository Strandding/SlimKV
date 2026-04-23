#!/bin/bash
#SBATCH -p NH100q
#SBATCH -w node07
#SBATCH -J train_qwen25_14b_dense_krope
#SBATCH -c 16
#SBATCH -o /export/home2/zihan/activation_beacon/data/outputs/train_qwen25_14b_dense_krope.%j.out
#SBATCH -e /export/home2/zihan/activation_beacon/data/outputs/train_qwen25_14b_dense_krope.%j.err
#SBATCH --time=UNLIMITED

set -e

echo "Job started on $(hostname) at $(date)"

cd /export/home2/zihan/activation_beacon

source /export/home2/zihan/miniconda3/etc/profile.d/conda.sh
conda activate beacon

# Fixed GPU ids requested by user.
export CUDA_VISIBLE_DEVICES=4,5,6,7

NPROC_PER_NODE=4 \
PER_DEVICE_TRAIN_BATCH_SIZE=1 \
GRADIENT_ACCUMULATION_STEPS=2 \
MODEL_NAME_OR_PATH=/dataset/common/tzh/.model/Qwen2.5-14B-Instruct \
BEACON_KV_COMPRESSION=none \
BEACON_SKIP_ROPE_FOR_BEACON_K=False \
BEACON_SHARED_KV_DOWN_PROJ=False \
BEACON_WINDOW=2048 \
BEACON_STRIDE=2048 \
OUTPUT_NAME=beacon-qwen25-14b-pretrain-dense-krope \
OCCUPY_AFTER_TRAIN=False \
bash /export/home2/zihan/activation_beacon/train_qwen25_14B.sh

echo "Job finished at $(date)"
