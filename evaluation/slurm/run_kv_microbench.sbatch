#!/bin/bash
#SBATCH --job-name=kv_microbench_16k
#SBATCH --output=/scratch/cc9171/results/%x_%j.out
#SBATCH --error=/scratch/cc9171/results/%x_%j.err
#SBATCH --account=ece_gy_9143-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --requeue

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-8B}"
METHODS="${METHODS:-no_press,kvzip,kvzap_prune_mlp}"
COMPRESSION_RATIOS="${COMPRESSION_RATIOS:-0.25,0.50,0.75}"
MAX_CONTEXT_LENGTH="${MAX_CONTEXT_LENGTH:-16384}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
OUTPUT_CSV="${OUTPUT_CSV:-/scratch/cc9171/results/kv_microbenchmark_16k.csv}"
PROJECT_DIR="${PROJECT_DIR:-/scratch/cc9171/kvpress}"
HF_HOME="${HF_HOME:-/scratch/cc9171/hf_cache}"

: "${HF_TOKEN:?Set HF_TOKEN in your environment before submitting this job}"

singularity exec --nv \
  --overlay /scratch/cc9171/my_env/my_overlay.ext3:ro \
  /share/apps/images/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
  bash << INNER
set -eo pipefail

source /ext3/miniconda3/bin/activate
conda activate kvpress

export PYTHONPATH=/scratch/cc9171/python_packages:\$PYTHONPATH
export HF_HOME="${HF_HOME}"
export HF_TOKEN="${HF_TOKEN}"

cd "${PROJECT_DIR}/evaluation"

python benchmark_kv_methods.py \
  --model "${MODEL}" \
  --methods "${METHODS}" \
  --compression_ratios "${COMPRESSION_RATIOS}" \
  --max_context_length "${MAX_CONTEXT_LENGTH}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --output_csv "${OUTPUT_CSV}"
INNER
