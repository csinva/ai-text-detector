#!/usr/bin/env bash
# Zero-shot baselines (Binoculars, Fast-DetectGPT, DetectGPT) on one GPU or CPU:
#   METHOD=binoculars OUT=<json> [ARGS="--sets a,b --cap 40"] sbatch --export=ALL --job-name=18-zs-binoculars aidetect/zs.slurm       (CPU timing: add --cpu-bench and sbatch --gres=none --cpus-per-task=16)
#SBATCH --partition=P3
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=05:00:00
#SBATCH --exclude=voyage-gpu-[1-2],voyage-gpu-4,voyage-gpu-16,voyage-gpu-31,voyage-gpu-33,voyage-gpu-37,voyage-gpu-41
#SBATCH --output=reports/slurm_scripts/slurm_logs/%x_%j.out
export V45_DATA_ROOT=/data18/voyage_4_5_experiments HF_HOME=/data18/voyage_4_5_experiments/data/aidetect/cache/hf TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
case " $ARGS " in *cpu-bench*) export CUDA_VISIBLE_DEVICES=;; esac
nvidia-smi --query-gpu=name,memory.used --format=csv,noheader 2>/dev/null
/data/chandan.singh/voyage-4.5/.venv-aid/bin/python -I aidetect/18_zeroshot_baselines.py --method $METHOD --out $OUT $ARGS
