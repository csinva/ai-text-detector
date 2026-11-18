#!/usr/bin/env bash
# CPU latency of the baselines:  sbatch --export=ALL --job-name=18-cpubase aidetect/cpu_base.slurm
#SBATCH --partition=P3
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=slurm_scripts/slurm_logs/%x_%j.out
export V45_DATA_ROOT=/data18/voyage_4_5_experiments PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES= HF_HOME=/data18/voyage_4_5_experiments/data/aidetect/cache/hf
lscpu | grep -E "Model name|^CPU\(s\)" | cut -c1-100
/data/chandan.singh/voyage-4.5/.venv-aid/bin/python -I aidetect/17_cpu_bench_baselines.py --out $V45_DATA_ROOT/data/aidetect/runs/cpu_bench_baselines.json
