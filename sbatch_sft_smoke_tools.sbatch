#!/bin/bash
#SBATCH --job-name=sft_smoke_tools
#SBATCH --partition=general
#SBATCH --gres=gpu:a6000:1
#SBATCH --exclude=gpu28
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=/zfsauton/scratch/cpulling/CORA/Verlog/logs/sft_smoke/%x_%j.out
#SBATCH --error=/zfsauton/scratch/cpulling/CORA/Verlog/logs/sft_smoke/%x_%j.err

# SFT smoke on the baseline-generated tools corpus. See run_sft_smoke_tools.sh
# for the launcher body; this sbatch just requests one A6000 for 30 min.

mkdir -p /zfsauton/scratch/cpulling/CORA/Verlog/logs/sft_smoke

cd /zfsauton/scratch/cpulling/CORA/Verlog
bash run_sft_smoke_tools.sh
