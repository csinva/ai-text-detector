#!/usr/bin/env bash
# Humanizer eval for one run:  NAME=lic3_nano_r10 sbatch --export=ALL --job-name=16-hum aidetect/hum_eval.slurm
#SBATCH --partition=P3
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=00:30:00
#SBATCH --exclude=voyage-gpu-[1-2],voyage-gpu-4,voyage-gpu-16,voyage-gpu-31,voyage-gpu-33,voyage-gpu-37,voyage-gpu-41
#SBATCH --output=reports/slurm_scripts/slurm_logs/%x_%j.out
export V45_DATA_ROOT=/data18/voyage_4_5_experiments TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
R=$V45_DATA_ROOT/data/aidetect/runs
/data/chandan.singh/voyage-4.5/.venv-aid/bin/python -I aidetect/16_eval_humanizers.py --model $R/$NAME/model.pt --out $R/$NAME/humanizers.json ${EXTRA:-} > $R/$NAME.hum.log 2>&1
