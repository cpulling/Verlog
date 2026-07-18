#!/bin/bash
# Verlog PPO on ARC disaster-response, Qwen3-4B + LoRA, 4×A6000.
# LoRA (PEFT) on actor+critic (rank 32, all-linear) sharded across 4 GPUs.
# Single-GPU LoRA was attempted (17610/17698/17885/17899) and always OOMed —
# transient peak of actor+critic unshard exceeds 48 GiB no matter how we tune
# gpu_memory_utilization / sleep_level / layered_summon. First 4-GPU attempts
# (17965/18003) hung because launch_qwen3_4b_lora.sh routed through
# train_arc_game_dryrun.sh (1-GPU test launcher). Launcher now invokes
# verl.trainer.main_ppo directly, mirroring the working full-FT sbatch spine.
#
# Submit:  sbatch train_arc_game_qwen3_4B_lora.sbatch
# Override knobs via env, e.g.:
#   LORA_RANK=16 BATCH_SIZE=256 sbatch train_arc_game_qwen3_4B_lora.sbatch

#SBATCH --job-name=cora_train
#SBATCH --output=/zfsauton/scratch/cpulling/CORA/Verlog/logs/slurm-%j.out
#SBATCH --error=/zfsauton/scratch/cpulling/CORA/Verlog/logs/slurm-%j.err
# 17614 SIGKILL'd 4 min in (exit 137) with MaxRSS 79 GB. That's per-process,
# not per-cgroup: FSDP CPU offload for actor+critic+ref + /dev/shm cached
# weights + Ray object store + vLLM CPU pages easily sum >160 GB. Bumped to
# 240 GB to give the cgroup real headroom.
#SBATCH --mem=240G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=general
#SBATCH --qos=qos_general
#SBATCH --gres=gpu:a6000:4
#SBATCH --time=23:59:59

set -euo pipefail

REPO_DIR=/zfsauton/scratch/cpulling/CORA/Verlog
cd "$REPO_DIR"

# Thin wrapper: all Hydra overrides live in launch_qwen3_4b_lora.sh, which
# now invokes verl.trainer.main_ppo directly. Pass through only the knobs a
# user might want to override at sbatch-submit time.
TAG=${TAG:-$(date +%m%d_%H%M)} \
N_GPUS=4 \
NUM_ENVS=${NUM_ENVS:-4} \
MAX_EPISODE_STEPS=${MAX_EPISODE_STEPS:-32} \
LORA_RANK=${LORA_RANK:-32} \
LORA_ALPHA=${LORA_ALPHA:-64} \
PPO_EPOCHS=${PPO_EPOCHS:-4} \
CRITIC_WARMUP=${CRITIC_WARMUP:-10} \
FORMAT_PENALTY=${FORMAT_PENALTY:-0.02} \
SEMANTIC_PENALTY=${SEMANTIC_PENALTY:-0.02} \
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.40} \
SAVE_FREQ=${SAVE_FREQ:-5} \
BATCH_SIZE="${BATCH_SIZE-}" \
GUIDED_REGEX="${GUIDED_REGEX-}" \
bash launch_qwen3_4b_lora.sh
