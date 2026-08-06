#!/bin/bash
#SBATCH --job-name=sft_4b_tools_v2
#SBATCH --partition=general
#SBATCH --gres=gpu:a6000:2
#SBATCH --exclude=gpu28
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/zfsauton/scratch/cpulling/CORA/Verlog/logs/sft/%x_%j.out
#SBATCH --error=/zfsauton/scratch/cpulling/CORA/Verlog/logs/sft/%x_%j.err

mkdir -p /zfsauton/scratch/cpulling/CORA/Verlog/logs/sft

cd /zfsauton/scratch/cpulling/CORA/Verlog
bash run_sft_4b_tools_v2.sh
