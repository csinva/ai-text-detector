#!/usr/bin/env bash
# CPU-only job (no GPU): latency benchmark.  MODEL=<model.pt> OUT=<json> [NL=12] sbatch --export=ALL --job-name=04-cpu aidetect/cpu.slurm
#SBATCH --partition=P4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --exclude=voyage-gpu-[1-2],voyage-gpu-4,voyage-gpu-16,voyage-gpu-31,voyage-gpu-33,voyage-gpu-37,voyage-gpu-41
#SBATCH --output=slurm_scripts/slurm_logs/%x_%j.out
set -uo pipefail
export V45_DATA_ROOT=/data18/voyage_4_5_experiments PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=
lscpu | grep -E "Model name|^CPU\(s\)|Thread|amx|avx512_bf16" | cut -c1-100
/data/chandan.singh/voyage-4.5/.venv-aid/bin/python -I aidetect/06_cpu_bench.py --model $MODEL --out $OUT --n_layers ${NL:-12} --threads 1,4,8,16
