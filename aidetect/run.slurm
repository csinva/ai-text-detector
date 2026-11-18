#!/usr/bin/env bash
# One-GPU job runner for the AI-text detector:
#   CMD="aidetect/03_train.py --run x ..." sbatch --export=ALL --job-name=04-<name> aidetect/run.slurm
#SBATCH --partition=P4
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --time=06:00:00
#SBATCH --exclude=voyage-gpu-[1-2],voyage-gpu-4,voyage-gpu-16,voyage-gpu-31,voyage-gpu-33,voyage-gpu-37,voyage-gpu-41
#SBATCH --output=reports/slurm_scripts/slurm_logs/%x_%j.out
set -euo pipefail
export V45_DATA_ROOT=/data18/voyage_4_5_experiments
export HF_HOME=$V45_DATA_ROOT/data/aidetect/cache/hf TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_${USER}_${SLURM_JOB_ID}
PY=/data/chandan.singh/voyage-4.5/.venv-aid/bin/python
nvidia-smi --query-gpu=name,memory.used --format=csv,noheader
eval "$PY -I $CMD"
