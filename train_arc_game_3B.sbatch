#!/bin/bash
# Verlog PPO on ARC disaster-response, Qwen2.5-3B-Instruct, 4xA6000.
# Adapted for Auton from train_arc_game.sbatch (Delta) and train.sbatch (3B babyai).
#
# Submit:
#   sbatch train_arc_game_3B.sbatch
# Override knobs via env, e.g.:
#   NUM_ENVS=16 BATCH_SIZE=128 sbatch train_arc_game_3B.sbatch

#SBATCH --job-name=cora_qwen3B
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --mem=200G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=general
#SBATCH --qos=qos_general
#SBATCH --gres=gpu:a6000:4
#SBATCH --time=23:59:59

set -euo pipefail

REPO_DIR=/zfsauton/scratch/cpulling/CORA/Verlog
ARC_GAME_PATH=/zfsauton/scratch/cpulling/CORA/ARC_Game/ARC_Game_New
ARC_GAME_BUILD=${ARC_GAME_PATH}/Build/Headless/Linux/ARC_Headless.x86_64

# Path-style activation of the verlog conda env
export PATH=/zfsauton/scratch/cpulling/conda_envs/verlog/bin:$PATH

cd "$REPO_DIR"
mkdir -p logs/arc_game

ulimit -n 65535 || true

NUM_GPUS_PER_NODE=${SLURM_GPUS_ON_NODE:-4}
unset ROCR_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS_PER_NODE-1)))
export VLLM_USE_V1=1
export PYTHONUNBUFFERED=1

# WandB
export WANDB_API_KEY=${WANDB_API_KEY:?WANDB_API_KEY must be set in env}
export WANDB_ENTITY=${WANDB_ENTITY:-cpulling}
export WANDB_PROJECT=${WANDB_PROJECT:-CORA_RL}

# ── Scaling knobs (4-GPU 3B defaults; matches recipe in train.sbatch) ────────
NUM_ENVS=${NUM_ENVS:-16}
MAX_EPISODE_STEPS=${MAX_EPISODE_STEPS:-32}
# train_batch_size must be ≥ NUM_ENVS × MAX_EPISODE_STEPS so the per-step
# turn counter has room for every env to finish an episode.
BATCH_SIZE=${BATCH_SIZE:-$((NUM_ENVS * MAX_EPISODE_STEPS))}
MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-$((BATCH_SIZE / 8))}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-2}
FORWARD_BATCH_SIZE=${FORWARD_BATCH_SIZE:-$((4 * MICRO_BATCH_SIZE))}
PPO_EPOCHS=${PPO_EPOCHS:-1}
BASE_PORT=${BASE_PORT:-9876}
MAX_PROMPT=${MAX_PROMPT:-12288}
MAX_RESP=${MAX_RESP:-384}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}

MODEL=${MODEL:-Qwen/Qwen2.5-3B-Instruct}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-bugfix_qwen3B_dryrun}

# Reset per-run Unity-port counter
rm -f "/tmp/arc_game_port_counter_${BASE_PORT}_${NUM_ENVS}.txt"

# Unity processes are spawned by Ray workers via auto_start_unity=True
export ARC_GAME_UNITY_LOG_DIR="${REPO_DIR}/logs/arc_game"
export ARC_GAME_PATH
export ARC_GAME_BUILD

cleanup() {
  echo "[arc_game] sweeping leftover headless processes"
  pkill -f "$ARC_GAME_BUILD" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

CONFIG_PATH="$(pwd)/examples/sglang_multiturn/config"
DATA_DIR=${DATA_DIR:-/zfsauton/scratch/cpulling/data/gsm8k}

python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='gsm8k_multiturn_grpo' \
    algorithm.adv_estimator=gae \
    data.train_files=${DATA_DIR}/train.parquet \
    data.val_files=${DATA_DIR}/test.parquet \
    data.train_batch_size=${BATCH_SIZE} \
    data.max_prompt_length=${MAX_PROMPT} \
    data.max_response_length=${MAX_RESP} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.model.path=${MODEL} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE} \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.ppo_epochs=${PPO_EPOCHS} \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${FORWARD_BATCH_SIZE} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.agent.num_workers=${NUM_ENVS} \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${FORWARD_BATCH_SIZE} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=True \
    trainer.balance_batch=False \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='CORA_RL' \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node=${NUM_GPUS_PER_NODE} \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.val_before_train=False \
    envs.num_envs=${NUM_ENVS} \
    envs.env_name=arc_game \
    envs.task='ARC disaster response' \
    envs.arc_game_kwargs.unity_port=${BASE_PORT} \
    envs.arc_game_kwargs.max_episode_steps=${MAX_EPISODE_STEPS} \
    envs.arc_game_kwargs.auto_start_unity=True \
    envs.arc_game_kwargs.unity_exe_path=${ARC_GAME_BUILD} \
    +envs.arc_game_kwargs.restart_unity_each_episode=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=10240 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=10240 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=10240 \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.path=${MODEL} \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_epochs=${PPO_EPOCHS} \
    critic.ppo_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE} \
    critic.ppo_mini_batch_size=${MINI_BATCH_SIZE} \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    critic.ppo_max_token_len_per_gpu=10240 \
    critic.forward_max_token_len_per_gpu=10240 \
    critic.forward_micro_batch_size_per_gpu=${FORWARD_BATCH_SIZE} \
    "$@"
