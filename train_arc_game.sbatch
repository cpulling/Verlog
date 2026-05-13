#!/bin/bash
# Cluster training entrypoint for Verlog + ARCGame env.
#
# Targets Delta (NCSA) by default. For Auton or Babel: change SBATCH partition,
# account, and module loads — everything below the SBATCH block is portable
# Linux+CUDA bash.
#
# Prereqs (one-time per cluster):
#   1. Build the Linux headless ARCGame binary (BuildLinux in HeadlessBuildScript.cs)
#      and place ARC_Headless.x86_64 under $ARC_GAME_BUILD.
#   2. Install Verlog per its README (conda env "verlog", balrog, vllm/sglang/mcore).
#   3. Export ARC_GAME_PATH to the directory holding arc_game_gym_env_tcp.py.
#
# The script launches one headless game per parallel env on a unique TCP port
# (base 9876 + worker_id), then runs verl's PPO trainer with envs.env_name=arc_game.

#SBATCH --job-name=arc_game_rl
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --mem=200G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpuA40x4        # Delta-specific; change for Auton/Babel
#SBATCH --account=bfoz-delta-gpu    # Delta-specific; change for Auton/Babel
#SBATCH --time=23:59:59
#SBATCH --gpus-per-node=4

set -euo pipefail

# ── Activate the verlog env (adjust path on your cluster) ───────────────────
source /u/${USER}/anaconda3/bin/activate
conda activate verlog
cd /u/${USER}/Verlog

ulimit -n 65535

# ── Cluster + scaling knobs ─────────────────────────────────────────────────
NUM_GPUS_PER_NODE=${SLURM_GPUS_ON_NODE:-4}
unset ROCR_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS_PER_NODE-1)))
export VLLM_USE_V1=1

NUM_ENVS=${NUM_ENVS:-8}
BATCH_SIZE=${BATCH_SIZE:-64}
MINI_BATCH_SIZE=$((BATCH_SIZE / 2))
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-4}
FORWARD_BATCH_SIZE=$((4 * MICRO_BATCH_SIZE))
PPO_EPOCHS=${PPO_EPOCHS:-1}
BASE_PORT=${BASE_PORT:-9876}

ARC_GAME_PATH=${ARC_GAME_PATH:?ARC_GAME_PATH must point at ARC_Game_New}
ARC_GAME_BUILD=${ARC_GAME_BUILD:?ARC_GAME_BUILD must point at ARC_Headless.x86_64}

# ── Spin up one headless game per env ───────────────────────────────────────
mkdir -p logs/arc_game
GAME_PIDS=()
for ((i=0; i<NUM_ENVS; i++)); do
  PORT=$((BASE_PORT + i))
  LOG=logs/arc_game/headless_${PORT}.log
  echo "[arc_game] launching headless on :${PORT} → ${LOG}"
  "$ARC_GAME_BUILD" -batchmode -nographics \
                   -gym-server -gym-port "$PORT" \
                   -logFile "$LOG" &
  GAME_PIDS+=($!)
done

cleanup() {
  echo "[arc_game] cleaning up headless servers"
  for pid in "${GAME_PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

# Wait until each port is accepting connections
for ((i=0; i<NUM_ENVS; i++)); do
  PORT=$((BASE_PORT + i))
  for _ in $(seq 1 60); do
    if (echo > /dev/tcp/127.0.0.1/$PORT) 2>/dev/null; then break; fi
    sleep 1
  done
done

# ── Verlog trainer ──────────────────────────────────────────────────────────
# Smoke-test config: Qwen2.5-0.5B-Instruct, small batches, short rollouts.
# Bump model size + batch knobs once you've confirmed the loop is healthy.
CONFIG_PATH="$(pwd)/examples/sglang_multiturn/config"

python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='gsm8k_multiturn_grpo' \
    algorithm.adv_estimator=gae \
    data.train_batch_size=${BATCH_SIZE} \
    data.max_prompt_length=2048 \
    data.max_response_length=256 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
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
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.agent.num_workers=${NUM_ENVS} \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${FORWARD_BATCH_SIZE} \
    algorithm.use_kl_in_reward=True \
    trainer.balance_batch=False \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='arc_game_rl' \
    trainer.experiment_name='qwen2.5-0.5b-smoke' \
    trainer.n_gpus_per_node=${NUM_GPUS_PER_NODE} \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=30 \
    trainer.total_epochs=5 \
    trainer.val_before_train=False \
    envs.num_envs=${NUM_ENVS} \
    envs.env_name=arc_game \
    envs.task='ARC disaster response' \
    envs.arc_game_kwargs.unity_port=${BASE_PORT} \
    envs.arc_game_kwargs.max_episode_steps=50 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=8192 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=8192 \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.path=Qwen/Qwen2.5-0.5B-Instruct \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_epochs=${PPO_EPOCHS} \
    critic.ppo_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE} \
    critic.ppo_mini_batch_size=${MINI_BATCH_SIZE} \
    critic.ppo_max_token_len_per_gpu=8192 \
    critic.forward_max_token_len_per_gpu=8192 \
    critic.forward_micro_batch_size_per_gpu=${FORWARD_BATCH_SIZE} \
    "$@"
