#!/bin/bash
#SBATCH -p HPCAIq
#SBATCH -w node14
#SBATCH -J eval_llama31_8b_raw_nokrope
#SBATCH -c 16
#SBATCH -o /export/home2/zihan/activation_beacon/data/outputs/logs/eval_llama31_8b_raw_nokrope.%j.out
#SBATCH -e /export/home2/zihan/activation_beacon/data/outputs/logs/eval_llama31_8b_raw_nokrope.%j.err
#SBATCH --time=UNLIMITED

set -euo pipefail

echo "Job started on $(hostname) at $(date)"

cd /export/home2/zihan/activation_beacon

source /export/home2/zihan/miniconda3/etc/profile.d/conda.sh
conda activate beacon

CUDA_VISIBLE_DEVICES=4,6,7 \
NPROC_PER_NODE=3 \
MASTER_PORT=29541 \
MODEL_NAME_OR_PATH=/dataset/common/tzh/.model/Llama-3.1-8B-Instruct \
EVAL_MODE=no_beacon \
RAW_NOKROPE=True \
OUTPUT_NAME=llama31-8b-raw-nokrope \
OCCUPY_AFTER_EVAL=False \
bash /export/home2/zihan/activation_beacon/eval_longbench_llama31_8B.sh

echo "Job finished at $(date)"
