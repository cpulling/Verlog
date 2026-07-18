#!/bin/bash
# Single-GPU dryrun: train Qwen2.5-0.5B-Instruct as a monolithic director on the
# ARC disaster-response env. Launches NUM_ENVS headless Unity processes locally
# (no slurm), waits for their TCP ports, then runs verl PPO.
#
# Usage:
#   ARC_GAME_BUILD=/path/to/ARC_Headless.x86_64 \
#   ARC_GAME_PATH=/zfsauton/scratch/cpulling/CORA/ARC_Game/ARC_Game_New \
#   bash train_arc_game_dryrun.sh
#
# Environment knobs (with defaults tuned for one 48GB A6000):
#   NUM_ENVS=2     # parallel headless games
#   BATCH_SIZE=8   # data.train_batch_size
#   BASE_PORT=9876
#   MODEL=Qwen/Qwen2.5-0.5B-Instruct

set -euo pipefail

REPO_DIR=${REPO_DIR:-/zfsauton/scratch/cpulling/CORA/Verlog}
ARC_GAME_PATH=${ARC_GAME_PATH:?ARC_GAME_PATH must point at ARC_Game_New}
ARC_GAME_BUILD=${ARC_GAME_BUILD:?ARC_GAME_BUILD must point at ARC_Headless.x86_64}

# Activate the verlog venv (path-style activation, not conda)
export PATH="/zfsauton/scratch/cpulling/conda_envs/verlog/bin:$PATH"

cd "$REPO_DIR"
ulimit -n 65535 || true

export VLLM_USE_V1=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONUNBUFFERED=1

NUM_ENVS=${NUM_ENVS:-2}
# A full ARC game is 8 days × 4 segments = 32 segments. To let Verlog's
# global per-PPO-step turn-counter (`Counter` in agent_loop.py) leave
# room for EVERY env to finish a full episode, the train_batch_size
# must be ≥ NUM_ENVS × MAX_EPISODE_STEPS. Hence the 2*32 = 64 default.
MAX_EPISODE_STEPS=${MAX_EPISODE_STEPS:-32}
BATCH_SIZE=${BATCH_SIZE:-$((NUM_ENVS * MAX_EPISODE_STEPS))}
MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-$((BATCH_SIZE / 8))}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-2}
FORWARD_BATCH_SIZE=${FORWARD_BATCH_SIZE:-$((4 * MICRO_BATCH_SIZE))}
PPO_EPOCHS=${PPO_EPOCHS:-1}
BASE_PORT=${BASE_PORT:-9876}
MODEL=${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}
MAX_PROMPT=${MAX_PROMPT:-12288}
MAX_RESP=${MAX_RESP:-384}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}

# WandB: set WANDB_API_KEY in env to enable. Logger picks up "wandb" automatically.
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  LOGGER='["console","wandb"]'
  export WANDB_PROJECT=${WANDB_PROJECT:-CORA_RL}
  export WANDB_ENTITY=${WANDB_ENTITY:-cpulling}
else
  LOGGER='["console"]'
fi

LOG_DIR="$REPO_DIR/logs/arc_game"
mkdir -p "$LOG_DIR"

# Reset the per-run Unity-port assignment counter so the next NUM_ENVS workers
# claim ports BASE_PORT, BASE_PORT+1, ... in arrival order.
rm -f "/tmp/arc_game_port_counter_${BASE_PORT}_${NUM_ENVS}.txt"

# Each Ray worker owns its Unity process now (auto_start_unity=True is
# passed below). The shell doesn't pre-launch Unity any more — Unity will
# be spawned + respawned by Python so reset() yields a genuinely fresh
# game state per episode.
export ARC_GAME_UNITY_LOG_DIR="$LOG_DIR"

cleanup() {
  echo "[arc_game] sweeping leftover headless processes"
  pkill -f "$ARC_GAME_BUILD" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

CONFIG_PATH="$(pwd)/examples/sglang_multiturn/config"
DATA_DIR=${DATA_DIR:-/zfsauton/scratch/cpulling/data/gsm8k}

echo "[dryrun DEBUG] PPO_EPOCHS=$PPO_EPOCHS BATCH_SIZE=$BATCH_SIZE MINI_BATCH_SIZE=$MINI_BATCH_SIZE MICRO_BATCH_SIZE=$MICRO_BATCH_SIZE" >&2
echo "[dryrun DEBUG] positional args ($#): $@" >&2

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
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${FORWARD_BATCH_SIZE} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
    actor_rollout_ref.rollout.agent.num_workers=${NUM_ENVS} \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${FORWARD_BATCH_SIZE} \
    algorithm.use_kl_in_reward=True \
    trainer.balance_batch=False \
    trainer.logger="${LOGGER}" \
    trainer.project_name='CORA_RL' \
    trainer.experiment_name="${EXPERIMENT_NAME:-bugfix_qwen0.5B_dryrun}" \
    trainer.n_gpus_per_node=1 \
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
    +envs.arc_game_kwargs.restart_unity_each_episode=${RESTART_UNITY_EACH_EPISODE:-False} \
    algorithm.step_gamma=${STEP_GAMMA:-0.99} \
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
    critic.ppo_max_token_len_per_gpu=10240 \
    critic.forward_max_token_len_per_gpu=10240 \
    critic.forward_micro_batch_size_per_gpu=${FORWARD_BATCH_SIZE} \
    +ray_kwargs.ray_init.include_dashboard=False \
    ++ray_kwargs.ray_init.num_cpus=4 \
    "$@"
