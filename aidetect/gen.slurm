#!/usr/bin/env bash
# AI text generation with vLLM, one permissively licensed model per job step.
#   MODELS="Qwen/Qwen3-8B:qwen3_8b Qwen/Qwen3-4B-Instruct-2507:qwen3_4b" [LIMIT=50] sbatch --export=ALL --job-name=00-gen aidetect/gen.slurm
#SBATCH --partition=P3
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --exclude=voyage-gpu-[1-2],voyage-gpu-4,voyage-gpu-16,voyage-gpu-31,voyage-gpu-33,voyage-gpu-37,voyage-gpu-41
#SBATCH --output=reports/slurm_scripts/slurm_logs/%x_%j.out
set -uo pipefail
export V45_DATA_ROOT=/data18/voyage_4_5_experiments
export HF_HOME=$V45_DATA_ROOT/data/aidetect/cache/hf TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 VLLM_LOGGING_LEVEL=WARNING VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN
export XDG_CACHE_HOME=$V45_DATA_ROOT/data/aidetect/cache/xdg VLLM_CACHE_ROOT=$V45_DATA_ROOT/data/aidetect/cache/vllm TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_${USER}_${SLURM_JOB_ID}
PY=$V45_DATA_ROOT/data/aidetect/cache/vllmvenv/bin/python
nvidia-smi --query-gpu=name,memory.used --format=csv,noheader
for mt in $MODELS; do
  m=${mt%%:*}; t=${mt##*:}
  echo "=== $m"
  $PY aidetect/14_generate.py --model $m --tag $t ${LIMIT:+--limit $LIMIT}
done
