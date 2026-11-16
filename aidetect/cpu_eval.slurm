#!/usr/bin/env bash
# CPU-only eval of a checkpoint (int8 dynamic quantization by default) on a subsample of every set.
#   MODEL=<model.pt> OUT=<json> [QUANT=1] [MAXDOCS=400] sbatch --export=ALL --job-name=04-cpueval aidetect/cpu_eval.slurm
#SBATCH --partition=P4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --exclude=voyage-gpu-[1-2],voyage-gpu-4,voyage-gpu-16,voyage-gpu-31,voyage-gpu-33,voyage-gpu-37,voyage-gpu-41
#SBATCH --output=slurm_scripts/slurm_logs/%x_%j.out
set -uo pipefail
export V45_DATA_ROOT=/data18/voyage_4_5_experiments PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=
/data/chandan.singh/voyage-4.5/.venv-aid/bin/python -I aidetect/04_eval.py --model $MODEL --out $OUT --device cpu --quant ${QUANT:-2} --max_docs ${MAXDOCS:-400} ${SETS:+--sets $SETS}
