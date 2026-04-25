#!/bin/bash
#SBATCH --job-name=kv_eval
#SBATCH --output=/scratch/cc9171/results/%x_%j.out
#SBATCH --error=/scratch/cc9171/results/%x_%j.err
#SBATCH --account=ece_gy_9143-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --requeue

set -euo pipefail

# Override these with sbatch --export=ALL,VAR=value.
MODEL="${MODEL:-Qwen/Qwen3-8B}"
PRESS_NAME="${PRESS_NAME:-kvzap_prune_mlp}"
COMPRESSION_RATIO="${COMPRESSION_RATIO:-0.5}"
DATASET="${DATASET:-ruler}"
DATA_DIR="${DATA_DIR:-16384}"
FRACTION="${FRACTION:-0.1}"
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/cc9171/results}"
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

python evaluate.py \
  --model "${MODEL}" \
  --press_name "${PRESS_NAME}" \
  --compression_ratio "${COMPRESSION_RATIO}" \
  --dataset "${DATASET}" \
  --data_dir "${DATA_DIR}" \
  --fraction "${FRACTION}" \
  --output_dir "${OUTPUT_DIR}"
INNER
