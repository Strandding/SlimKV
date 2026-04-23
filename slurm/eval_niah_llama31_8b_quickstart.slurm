#!/bin/bash
#SBATCH -p HPCAIq
#SBATCH -w node14
#SBATCH -J niah_llama31_8b_dense_qs
#SBATCH -c 16
#SBATCH -o /export/home2/zihan/activation_beacon/data/outputs/logs/niah_llama31_8b_dense_qs.%j.out
#SBATCH -e /export/home2/zihan/activation_beacon/data/outputs/logs/niah_llama31_8b_dense_qs.%j.err
#SBATCH --time=UNLIMITED
#SBATCH --dependency=afterany:434531

set -euo pipefail

echo "Job started on $(hostname) at $(date)"

cd /export/home2/zihan/activation_beacon

source /export/home2/zihan/miniconda3/etc/profile.d/conda.sh
conda activate beacon

# Pass OPENAI_API_KEY from submit environment.
# Example: OPENAI_API_KEY=sk-... sbatch /export/home2/zihan/activation_beacon/slurm/eval_niah_llama31_8b_quickstart.slurm
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "ERROR: OPENAI_API_KEY is not set. Please export it before sbatch." >&2
  exit 1
fi

# Quick Start: one length x one depth (fast smoke test)
CUDA_VISIBLE_DEVICES=4,5,6,7 \
MASTER_PORT=29563 \
torchrun --nproc_per_node=4 -m main.eval_needle \
  --model_name_or_path /export/home2/zihan/activation_beacon/data/outputs/beacon-llama31-8b-finetune-dense/checkpoint-2000 \
  --enable_beacon \
  --beacon_window 1024 \
  --beacon_stride 1024 \
  --beacon_ratio 16 \
  --beacon_ratio_mix sequence \
  --beacon_skip_rope_for_beacon_k False \
  --beacon_shared_kv_down_proj False \
  --attn_impl flash_attention_2 \
  --dtype bf16 \
  --batch_size 1 \
  --haystack_path /dataset/common/tzh/.dataset/long-llm/needle/PaulGrahamEssays \
  --result_dir llama31_8b_beacon_dense_r16_qs_65k \
  --min_length 4096 \
  --max_length 65536 \
  --num_length_interval 10 \
  --min_depth 0 \
  --max_depth 100 \
  --num_depth_interval 10 \
  --max_new_tokens 50 \
  --gpt_eval

echo "Job finished at $(date)"
