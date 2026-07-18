#!/bin/bash
# 1×A6000 smoke test: Qwen3-4B-Instruct-2507 on the CORA ARC disaster-response env.
# Wraps train_arc_game_dryrun.sh with the model override + slurm resources.
#
# Submit:
#   sbatch train_arc_game_qwen3_dryrun.sbatch

#SBATCH --job-name=cora_qwen3_4b_smoke
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --mem=80G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=general
#SBATCH --qos=qos_general
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=02:00:00

set -euo pipefail

REPO_DIR=/zfsauton/scratch/cpulling/CORA/Verlog
ARC_GAME_PATH=/zfsauton/scratch/cpulling/CORA/ARC_Game/ARC_Game_New
ARC_GAME_BUILD=${ARC_GAME_PATH}/Build/Headless/Linux/ARC_Headless.x86_64

export ARC_GAME_PATH
export ARC_GAME_BUILD

export MODEL=Qwen/Qwen3-4B-Instruct-2507
export EXPERIMENT_NAME=qwen3_4b_smoke_$(date +%m%d_%H%M)
export NUM_ENVS=2
export TOTAL_EPOCHS=1
export STEP_GAMMA=1.0
export RESTART_UNITY_EACH_EPISODE=False

export WANDB_API_KEY=${WANDB_API_KEY:?WANDB_API_KEY must be set in env}
export WANDB_ENTITY=${WANDB_ENTITY:-cpulling}
export WANDB_PROJECT=${WANDB_PROJECT:-CORA_RL}

cd "$REPO_DIR"

bash train_arc_game_dryrun.sh "$@"
