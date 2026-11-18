#!/usr/bin/env bash
# Train one detector on one GPU, then run the full eval. Safe to resubmit after a preemption: 03_train.py resumes from ckpt.pt.
#   NAME=final [EVALX="--n_layers 28"] EXTRA="--tok tok_full --lr 1.5e-4" sbatch --export=ALL --job-name=04-final aidetect/train_eval.slurm
#SBATCH --partition=P4
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --time=03:00:00
#SBATCH --exclude=voyage-gpu-[1-2],voyage-gpu-4,voyage-gpu-16,voyage-gpu-31,voyage-gpu-33,voyage-gpu-37,voyage-gpu-41
#SBATCH --output=reports/slurm_scripts/slurm_logs/%x_%j.out
set -uo pipefail
export V45_DATA_ROOT=/data18/voyage_4_5_experiments
export HF_HOME=$V45_DATA_ROOT/data/aidetect/cache/hf TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_${USER}_${SLURM_JOB_ID}
PY=/data/chandan.singh/voyage-4.5/.venv-aid/bin/python
R=$V45_DATA_ROOT/data/aidetect/runs
$PY -I aidetect/03_train.py --run $NAME --total_tokens ${TOKENS:-1e9} --eval_every 500 --save_every 250 $EXTRA >> $R/$NAME.log 2>&1
$PY -I aidetect/04_eval.py --model $R/$NAME/model.pt --out $R/$NAME/eval.json ${EVALX:-} > $R/$NAME.eval.log 2>&1
