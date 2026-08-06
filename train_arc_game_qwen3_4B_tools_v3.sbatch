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

#SBATCH --job-name=cora_qwen3_4B_tools
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

# WandB — accept key from env or ~/.netrc (never echo). If the resolved key
# fails authentication at runtime, override PPO_LOGGER='[console]' from the
# submitting env to skip wandb entirely rather than crash the run.
if [ -z "${WANDB_API_KEY:-}" ] && [ -r "$HOME/.netrc" ]; then
  _wandb_key=$(awk '/machine api.wandb.ai/{f=1;next} f && /password/{print $2; exit}' "$HOME/.netrc")
  [ -n "$_wandb_key" ] && export WANDB_API_KEY="$_wandb_key"
  unset _wandb_key
fi
: "${WANDB_API_KEY:?WANDB_API_KEY must be in env or ~/.netrc}"
export WANDB_ENTITY=${WANDB_ENTITY:-cpulling}
export WANDB_PROJECT=${WANDB_PROJECT:-CORA_RL}
PPO_LOGGER=${PPO_LOGGER:-'["console","wandb"]'}

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

# v3 additions (see analysis/exploration_vs_exploitation.md for rationale):
# - ACTOR_LR: 3e-6 (was hardcoded 1e-6) — compensates for grad_norm 3.6x smaller
#   on SFT-warm actor vs raw-PPO. Restores per-step actor movement.
# - USE_KL_LOSS + KL_LOSS_COEF + REF_MODEL: anchor KL to BASE Qwen3-4B, not to SFT.
#   SFT reference would re-freeze the actor at the failure basin (Echo Collapse
#   per RAGEN / StarPO). Base ref only fights out-of-distribution drift.
ACTOR_LR=${ACTOR_LR:-3e-6}
CRITIC_LR=${CRITIC_LR:-1e-5}
USE_KL_LOSS=${USE_KL_LOSS:-True}
KL_LOSS_COEF=${KL_LOSS_COEF:-0.05}
REF_MODEL=${REF_MODEL:-Qwen/Qwen3-4B}
# Tag count capped at {1,3} to prevent multi-tag bursts (18394 collapsed at {1,15}).
# Site names enum-locked to the 28-name vocabulary observed in 18394 trajectory
# (Motel/Community[01-03]/Shelter[_0-4]/Shelters/Kitchen[_0-4]/Kitchens/Casework/
# CaseworkSite[_0-4]). ACTION: prefix required. Task IDs and choice indices remain
# unbound (state-dependent, would require per-turn dynamic grammar to enum-lock).
# TOOL-CALL ARM: GCD is off (would collide with Qwen3's Hermes <tool_call>
# JSON output). Also unnecessary here — the schema/enum validation happens
# server-side via HermesToolParser and env-side via _synthesize_tags_from_tool_calls,
# not via a decoding-time regex. Kept as explicit empty export so any inherited
# env value is neutralized.
GUIDED_REGEX=""
export GUIDED_REGEX

# TOOL-CALL ARM: route obs_encoder + get_instruction_prompt to the tool-call
# system prompt (no XML grammar section). tool_agent_loop reads tool_schemas
# from multi_turn.tool_config_path and routes assistant output through
# HermesToolParser; llm_agents_wrapper.extract_action synthesizes the
# equivalent XML tags from the parsed calls and reuses parse_commands so the
# behavior/* metrics are directly comparable to the text-mode baseline.
export ARC_ACTION_MODE=${ARC_ACTION_MODE:-tool_calls}
TOOL_CONFIG_PATH=${TOOL_CONFIG_PATH:-verl/envs/environments/arc_game/tool_config/arc_tools.yaml}
MULTI_TURN_FORMAT=${MULTI_TURN_FORMAT:-hermes}

# Stable, title-based task IDs (BUDGET_DAILY, FOOD_C01, ...) instead of Unity's
# drifting integer taskIds. Fixes the top-1 documented policy blocker (52.7% of
# <task> commands referenced hallucinated IDs in prior runs). Default ON; set
# ARC_STABLE_TASK_TOKENS=0 to disable (baseline / benchmark parity).
export ARC_STABLE_TASK_TOKENS=${ARC_STABLE_TASK_TOKENS:-1}
# Per-job Unity port (see launch_qwen3_slim_tools.sh for rationale — orphaned
# Unity from a killed job keeps its port bound; the wrapper's already-alive
# probe attaches to the dead server and Unity rejects the reconnect as a
# second client). NUM_ENVS=4 reserves 4 consecutive ports starting from
# BASE_PORT, so pick a base that leaves headroom before the next job's base.
if [ -n "${SLURM_JOB_ID:-}" ]; then
  BASE_PORT=${BASE_PORT:-$(( 40000 + (SLURM_JOB_ID * 10) % 10000 ))}
else
  BASE_PORT=${BASE_PORT:-40000}
fi
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
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_4b_tools_${TAG}}

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
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE} \
    actor_rollout_ref.actor.use_kl_loss=${USE_KL_LOSS} \
    actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF} \
    +actor_rollout_ref.ref.model.path=${REF_MODEL} \
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
    actor_rollout_ref.rollout.multi_turn.tool_config_path=${TOOL_CONFIG_PATH} \
    actor_rollout_ref.rollout.multi_turn.format=${MULTI_TURN_FORMAT} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${FORWARD_BATCH_SIZE} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=10240 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=10240 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=10240 \
    critic.optim.lr=${CRITIC_LR} \
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
    trainer.logger="${PPO_LOGGER}" \
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
    +trainer.rollout_data_dir="${ROLLOUT_DUMP_DIR}" \
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
