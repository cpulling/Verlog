#!/bin/bash
# Verlog PPO on ARC disaster-response, Qwen3-4B full-FT, 4×A6000 FSDP.
# Baseline (no LoRA) for the CORA experiments — matches the 0.6B run's
# reward-shaping, penalties, and metric plumbing but with a 4B model
# sharded across 4 GPUs. Meant as the head-to-head control against later
# Qwen3-4B + LoRA / OLoRA-tail runs.
#
# Submit:  sbatch train_arc_game_qwen3_4B.sbatch
# Override knobs via env, e.g.:
#   BATCH_SIZE=256 NUM_ENVS=8 sbatch train_arc_game_qwen3_4B.sbatch

#SBATCH --job-name=cora_qwen3_4B
#SBATCH --output=/zfsauton/scratch/cpulling/CORA/Verlog/logs/slurm-%j.out
#SBATCH --error=/zfsauton/scratch/cpulling/CORA/Verlog/logs/slurm-%j.err
#SBATCH --mem=240G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=general
#SBATCH --qos=qos_general
#SBATCH --gres=gpu:a6000:4
#SBATCH --time=23:59:59
# NUMA hints: on multi-user nodes, Slurm was scattering our 32 CPUs across BOTH
# sockets → FSDP all-gathers hit UPI on every step → 2× slower training (measured
# 63 vs 142 tokens/s on gpu26 vs a clean node). `--gpu-bind=closest` schedules
# CPUs near their allocated GPUs, `--hint=nomultithread` avoids SMT thread noise.
#SBATCH --gpu-bind=closest
#SBATCH --hint=nomultithread

set -euo pipefail

REPO_DIR=/zfsauton/scratch/cpulling/CORA/Verlog
ARC_GAME_PATH=/zfsauton/scratch/cpulling/CORA/ARC_Game/ARC_Game_New
ARC_GAME_BUILD=${ARC_GAME_PATH}/Build/Headless/Linux/ARC_Headless.x86_64

# Path-style activation of the verlog conda env
export PATH=/zfsauton/scratch/cpulling/conda_envs/verlog/bin:$PATH

# Point HF at the local cache — Qwen3-4B already present (7.5 GB, verified)
export HF_HOME=${HF_HOME:-/zfsauton/scratch/cpulling/.cache/huggingface}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}

cd "$REPO_DIR"
mkdir -p logs/arc_game logs/qwen3_4b

ulimit -n 65535 || true

NUM_GPUS_PER_NODE=${SLURM_GPUS_ON_NODE:-4}
unset ROCR_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS_PER_NODE-1)))
export VLLM_USE_V1=1
export PYTHONUNBUFFERED=1
# Distributed init timeout (default 30 min blows up at 4B on 4 GPUs when workers
# are still loading shards). 2h leaves margin without hiding real hangs.
export TORCH_DIST_TIMEOUT_SEC=${TORCH_DIST_TIMEOUT_SEC:-7200}

# Ray tmp — per-job so parallel runs don't collide on socket paths
export RAY_TMPDIR=/tmp/ray_cora_qwen3_4b_${SLURM_JOB_ID:-local}
mkdir -p "$RAY_TMPDIR"
export RAY_OBJECT_STORE_MEMORY=${RAY_OBJECT_STORE_MEMORY:-8000000000}
export RAY_memory_monitor_refresh_ms=0

# WandB
export WANDB_API_KEY=${WANDB_API_KEY:?WANDB_API_KEY must be set in env}
export WANDB_ENTITY=${WANDB_ENTITY:-cpulling}
export WANDB_PROJECT=${WANDB_PROJECT:-CORA_RL}

# ── Scaling knobs ────────────────────────────────────────────────────────────
# NUM_ENVS=4 keeps the batch shape identical to the 0.6B BATCH=128 run for
# a clean head-to-head. Bump NUM_ENVS to 8/16 later once the 4B run is stable
# to fully use the 4-GPU rollout throughput.
NUM_ENVS=${NUM_ENVS:-4}
MAX_EPISODE_STEPS=${MAX_EPISODE_STEPS:-32}
BATCH_SIZE=${BATCH_SIZE:-$((NUM_ENVS * MAX_EPISODE_STEPS))}
MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-$BATCH_SIZE}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-2}
FORWARD_BATCH_SIZE=${FORWARD_BATCH_SIZE:-4}
PPO_EPOCHS=${PPO_EPOCHS:-2}
CRITIC_WARMUP=${CRITIC_WARMUP:-20}
ENTROPY_COEFF=${ENTROPY_COEFF:-0.005}
# DAPO-style decoupled clip: keep lower at 0.2, loosen upper to 0.28 so PPO_EPOCHS=2
# doesn't crush low-prob exploration tokens (Yu et al. 2025).
CLIP_RATIO_LOW=${CLIP_RATIO_LOW:-0.2}
CLIP_RATIO_HIGH=${CLIP_RATIO_HIGH:-0.28}
# Tag count capped at {1,3} to prevent multi-tag bursts (18394 collapsed at {1,15}).
# Site names enum-locked to the 28-name vocabulary observed in 18394 trajectory
# (Motel/Community[01-03]/Shelter[_0-4]/Shelters/Kitchen[_0-4]/Kitchens/Casework/
# CaseworkSite[_0-4]). ACTION: prefix required. Task IDs and choice indices remain
# unbound (state-dependent, would require per-turn dynamic grammar to enum-lock).
GUIDED_REGEX=${GUIDED_REGEX-'REASONING:[^<]{0,4000}\s*ACTION:\s*(\s*(<build>(kitchen|shelter|casework),[0-9]{1,3}</build>|<hire>(untrained|trained),[0-9]{1,3}</hire>|<staff>(Motel|Community0?[1-3]?|Shelter(_[0-4])?|Shelters|Kitchen(_[0-4])?|Kitchens|Casework|CaseworkSite(_[0-4])?),[0-9]{1,3}</staff>|<deconstruct>(Motel|Community0?[1-3]?|Shelter(_[0-4])?|Shelters|Kitchen(_[0-4])?|Kitchens|Casework|CaseworkSite(_[0-4])?)</deconstruct>|<task>[0-9]{1,4},[0-9]{1,3}</task>|<train>[0-9]{1,3}</train>|<transfer>[a-z]+,[A-Za-z0-9_]+,[A-Za-z0-9_]+,[0-9]{1,4}</transfer>)){1,3}\s*'}
export GUIDED_REGEX

# Stable, title-based task IDs (BUDGET_DAILY, FOOD_C01, ...) instead of Unity's
# drifting integer taskIds. Fixes the top-1 documented policy blocker (52.7% of
# <task> commands referenced hallucinated IDs in prior runs). Default ON; set
# ARC_STABLE_TASK_TOKENS=0 to disable (baseline / benchmark parity).
export ARC_STABLE_TASK_TOKENS=${ARC_STABLE_TASK_TOKENS:-1}
BASE_PORT=${BASE_PORT:-19876}   # different from 0.6B run (9876) — no port clash if both run
MAX_PROMPT=${MAX_PROMPT:-12288}
MAX_RESP=${MAX_RESP:-768}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
STEP_GAMMA=${STEP_GAMMA:-1.0}

MODEL=${MODEL:-Qwen/Qwen3-4B}

# Reward-shaping penalties (match 0.6B setup)
FORMAT_PENALTY=${FORMAT_PENALTY:-0.05}
SEMANTIC_PENALTY=${SEMANTIC_PENALTY:-0.05}
# Flat holistic-correctness bonus. Cap on total format-negative = -FORMAT_BONUS,
# so best/worst format outcomes are symmetric ±R. Set to 0 to disable bonus.
FORMAT_BONUS=${FORMAT_BONUS:-0.10}

TAG=${TAG:-$(date +%m%d_%H%M)}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_4b_full_${TAG}}

# Checkpointing
CKPT_DIR="$REPO_DIR/checkpoints/CORA_RL/${EXPERIMENT_NAME}"
mkdir -p "$CKPT_DIR"
SAVE_FREQ=${SAVE_FREQ:-5}
MAX_CKPT_TO_KEEP=${MAX_CKPT_TO_KEEP:-2}

# Trajectory logging
export ARC_TRAJECTORY_LOG="$REPO_DIR/logs/qwen3_4b/${EXPERIMENT_NAME}.trajectory.jsonl"
: > "$ARC_TRAJECTORY_LOG"

# Reset per-run Unity-port counter
rm -f "/tmp/arc_game_port_counter_${BASE_PORT}_${NUM_ENVS}.txt"

# Unity spawn / logging
export ARC_GAME_UNITY_LOG_DIR="${REPO_DIR}/logs/qwen3_4b"
export ARC_GAME_PATH
export ARC_GAME_BUILD

cleanup() {
  echo "[arc_game] sweeping leftover headless processes"
  pkill -f "$ARC_GAME_BUILD" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

CONFIG_PATH="$(pwd)/examples/sglang_multiturn/config"
DATA_DIR=${DATA_DIR:-/zfsauton/scratch/cpulling/data/gsm8k}

LOGF="$REPO_DIR/logs/qwen3_4b/${EXPERIMENT_NAME}.log"
echo "=== $EXPERIMENT_NAME job $SLURM_JOB_ID on $SLURMD_NODENAME @ $(date) ==="
echo "MODEL=$MODEL NUM_ENVS=$NUM_ENVS BATCH=$BATCH_SIZE MINI=$MINI_BATCH_SIZE MICRO=$MICRO_BATCH_SIZE PPO_EPOCHS=$PPO_EPOCHS"
echo "checkpoint dir: $CKPT_DIR   save_freq: $SAVE_FREQ"
echo "trajectory log: $ARC_TRAJECTORY_LOG"
echo "stdout log:     $LOGF"
nvidia-smi -L || true

numactl --interleave=all python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='gsm8k_multiturn_grpo' \
    algorithm.adv_estimator=gae \
    algorithm.use_kl_in_reward=False \
    algorithm.step_gamma=${STEP_GAMMA} \
    data.train_files=${DATA_DIR}/train.parquet \
    data.val_files=${DATA_DIR}/test.parquet \
    data.train_batch_size=${BATCH_SIZE} \
    data.max_prompt_length=${MAX_PROMPT} \
    data.max_response_length=${MAX_RESP} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    +data.apply_chat_template_kwargs.enable_thinking=false \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.model.path=${MODEL} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE} \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.ppo_epochs=${PPO_EPOCHS} \
    actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF} \
    actor_rollout_ref.actor.clip_ratio_low=${CLIP_RATIO_LOW} \
    actor_rollout_ref.actor.clip_ratio_high=${CLIP_RATIO_HIGH} \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${FORWARD_BATCH_SIZE} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.40 \
    actor_rollout_ref.rollout.agent.num_workers=${NUM_ENVS} \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${FORWARD_BATCH_SIZE} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
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
    trainer.balance_batch=False \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='CORA_RL' \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node=${NUM_GPUS_PER_NODE} \
    trainer.nnodes=1 \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.critic_warmup=${CRITIC_WARMUP} \
    trainer.max_actor_ckpt_to_keep=${MAX_CKPT_TO_KEEP:-2} \
    trainer.max_critic_ckpt_to_keep=${MAX_CKPT_TO_KEEP:-2} \
    +trainer.best_ckpt_metric=${BEST_CKPT_METRIC:-arc/snap/score} \
    trainer.test_freq=-1 \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    ${TOTAL_TRAINING_STEPS:++trainer.total_training_steps=${TOTAL_TRAINING_STEPS}} \
    trainer.val_before_train=False \
    trainer.default_local_dir="${CKPT_DIR}" \
    trainer.max_actor_ckpt_to_keep=${MAX_CKPT_TO_KEEP} \
    envs.num_envs=${NUM_ENVS} \
    envs.env_name=arc_game \
    envs.task='ARC disaster response' \
    envs.arc_game_kwargs.unity_port=${BASE_PORT} \
    envs.arc_game_kwargs.max_episode_steps=${MAX_EPISODE_STEPS} \
    envs.arc_game_kwargs.auto_start_unity=True \
    envs.arc_game_kwargs.unity_exe_path=${ARC_GAME_BUILD} \
    +envs.arc_game_kwargs.restart_unity_each_episode=False \
    +envs.arc_game_kwargs.format_penalty=$FORMAT_PENALTY \
    +envs.arc_game_kwargs.semantic_penalty=$SEMANTIC_PENALTY \
    +envs.arc_game_kwargs.format_bonus=$FORMAT_BONUS \
    +ray_kwargs.ray_init._temp_dir="$RAY_TMPDIR" \
    +ray_kwargs.ray_init.object_store_memory=$RAY_OBJECT_STORE_MEMORY \
    +ray_kwargs.ray_init.include_dashboard=False \
    ++ray_kwargs.ray_init.num_cpus=${RAY_NUM_CPUS:-24} \
    "$@" 2>&1 | tee "$LOGF"
