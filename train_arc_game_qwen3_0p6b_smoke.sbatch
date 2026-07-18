#!/bin/bash
# 1×A6000 smoke: Qwen3-0.6B on ARC disaster-response. Small enough to fit
# comfortably on one A6000 without FSDP offload; exercises Qwen3's chat
# template + tokenizer against the ARC env wrapper (the real risk before
# scaling up to Qwen3-4B). Dedicated slurm job so we own the memory
# cgroup — piggy-backing on a sibling job caused runtime_env_agent OOM.
#
# Submit:
#   sbatch train_arc_game_qwen3_0p6b_smoke.sbatch

#SBATCH --job-name=cora_qwen3_0p6b_smoke
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --mem=80G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --account=users
#SBATCH --partition=general
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=02:00:00

set -euo pipefail

REPO_DIR=/zfsauton/scratch/cpulling/CORA/Verlog
ARC_GAME_PATH=/zfsauton/scratch/cpulling/CORA/ARC_Game/ARC_Game_New
ARC_GAME_BUILD=${ARC_GAME_PATH}/Build/Headless/Linux/ARC_Headless.x86_64

export ARC_GAME_PATH
export ARC_GAME_BUILD

export TAG=${TAG:-$(date +%m%d_%H%M)}

# Delegate to launch_qwen3_slim.sh — the single source of truth for the 0.6B
# ARC training knobs (STEP_GAMMA, CRITIC_WARMUP, FORMAT_PENALTY, penalties,
# checkpoint dir). Sbatch just provides the slurm cgroup + $TAG.
bash "$REPO_DIR/launch_qwen3_slim.sh" "$@"
