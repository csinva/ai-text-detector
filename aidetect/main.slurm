#!/usr/bin/env bash
# Main runs on one GPU, back to back: final model (full mix), then the OOD-study model (RAID generators/domains/attacks held out); 1B tokens each, then the full eval.
#   sbatch --export=ALL --job-name=04-main aidetect/main.slurm
#SBATCH --partition=P4
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --time=04:30:00
#SBATCH --exclude=voyage-gpu-[1-2],voyage-gpu-4,voyage-gpu-16,voyage-gpu-31,voyage-gpu-33,voyage-gpu-37,voyage-gpu-41
#SBATCH --output=slurm_scripts/slurm_logs/%x_%j.out
set -uo pipefail
export V45_DATA_ROOT=/data18/voyage_4_5_experiments
export HF_HOME=$V45_DATA_ROOT/data/aidetect/cache/hf TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_${USER}_${SLURM_JOB_ID}
PY=/data/chandan.singh/voyage-4.5/.venv-aid/bin/python
R=$V45_DATA_ROOT/data/aidetect/runs
run() {  # name, extra train args
  name=$1; shift
  $PY -I aidetect/03_train.py --run $name --total_tokens 1e9 --eval_every 500 --save_every 500 "$@" > $R/$name.log 2>&1
  $PY -I aidetect/04_eval.py --model $R/$name/model.pt --out $R/$name/eval.json $EVAL_ARGS $([[ $name == *causal* ]] && echo "--causal 1") > $R/$name.eval.log 2>&1
}
EVAL_ARGS=""
run final --tok tok_full --lr 1.5e-4
run final_ood --tok tok --lr 1.5e-4
