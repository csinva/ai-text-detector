#!/usr/bin/env bash
# MFU variants of the training micro-batch (fwd+bwd): attention kernel, micro-batch size, torch.compile mode.
#SBATCH --partition=P4
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --time=00:40:00
#SBATCH --exclude=voyage-gpu-[1-2],voyage-gpu-4,voyage-gpu-16,voyage-gpu-31,voyage-gpu-33,voyage-gpu-37,voyage-gpu-41
#SBATCH --output=reports/slurm_scripts/slurm_logs/%x_%j.out
set -uo pipefail
export V45_DATA_ROOT=/data18/voyage_4_5_experiments PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_${USER}_${SLURM_JOB_ID}
PY=/data/chandan.singh/voyage-4.5/.venv-aid/bin/python
for cfg in "fa3 65536 default" "fa3 131072 default" "fa3 32768 default" "fa3 65536 max-autotune-no-cudagraphs" "flash 65536 default"; do
  set -- $cfg
  echo "=== attn=$1 budget=$2 compile=$3"
  ATTN=$1 $PY -I aidetect/prof.py $2 $3 2>&1 | grep -E "^budget|^\{'other|Error|error" | head -4
done
