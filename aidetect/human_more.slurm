#!/usr/bin/env bash
# Round 4 human-text extraction, one source per CPU job:  SRC=pes2o sbatch --export=ALL --job-name=17-hm-pes2o aidetect/human_more.slurm
#SBATCH --partition=P3
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_scripts/slurm_logs/%x_%j.out
export V45_DATA_ROOT=/data18/voyage_4_5_experiments HF_HOME=/data18/voyage_4_5_experiments/data/aidetect/cache/hf PYTHONUNBUFFERED=1
/data/chandan.singh/voyage-4.5/.venv-aid/bin/python -I aidetect/17_human_more.py $SRC 1
