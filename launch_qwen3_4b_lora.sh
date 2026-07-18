#!/bin/bash
# Qwen3-4B + LoRA on ARC — 4×A6000 verlog PPO.
#
# Directly invokes verl.trainer.main_ppo (mirrors the working full-FT sbatch
# spine at train_arc_game_qwen3_4B.sbatch), adding only LoRA-specific
# overrides on top: lora_rank/alpha, target_modules=all-linear on both actor
# and critic, load_format=safetensors, layered_summon, LoRA-friendly FSDP
# CPU offload. Previous versions routed through train_arc_game_dryrun.sh,
# which is a 1-GPU test launcher (hardcodes CUDA_VISIBLE_DEVICES=0,
# num_cpus=4, n_gpus=1) — that path repeatedly hung on multi-GPU inits.
#
# Env knobs (with defaults):
#   N_GPUS=4
#   NUM_ENVS=4
#   BATCH_SIZE=$((NUM_ENVS*MAX_EPISODE_STEPS))
#   MAX_EPISODE_STEPS=32
#   LORA_RANK=32
#   LORA_ALPHA=64
#   GPU_MEM_UTIL=0.40
#   CRITIC_WARMUP=10
#   FORMAT_PENALTY=0.02
#   SEMANTIC_PENALTY=0.02
#   GUIDED_REGEX=<ARC action grammar> (set to "" to disable)

set -euo pipefail

REPO_DIR=/zfsauton/scratch/cpulling/CORA/Verlog
ARC_GAME_PATH=/zfsauton/scratch/cpulling/CORA/ARC_Game/ARC_Game_New
ARC_GAME_BUILD=${ARC_GAME_PATH}/Build/Headless/Linux/ARC_Headless.x86_64
export ARC_GAME_PATH ARC_GAME_BUILD

export PATH="/zfsauton/scratch/cpulling/conda_envs/verlog/bin:$PATH"
export HF_HOME=${HF_HOME:-/zfsauton/scratch/cpulling/.cache/huggingface}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}

N_GPUS=${N_GPUS:-4}
unset ROCR_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((N_GPUS-1)))
export VLLM_USE_V1=1
export PYTHONUNBUFFERED=1
export TORCH_DIST_TIMEOUT_SEC=${TORCH_DIST_TIMEOUT_SEC:-7200}

cd "$REPO_DIR"
ulimit -n 65535 || true

TAG=${TAG:-$(date +%m%d_%H%M)}
LOG_DIR="$REPO_DIR/logs/qwen3_4b_lora"
mkdir -p "$LOG_DIR"

MODEL=${MODEL:-Qwen/Qwen3-4B}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_4b_lora_${TAG}}

# NUM_ENVS matches the full-FT baseline (4 parallel rollout envs across
# 4 GPUs). BATCH_SIZE = NUM_ENVS*MAX_EPISODE_STEPS guarantees each PPO step
# has room for one full 32-turn ARC episode per env.
NUM_ENVS=${NUM_ENVS:-4}
MAX_EPISODE_STEPS=${MAX_EPISODE_STEPS:-32}
BATCH_SIZE=${BATCH_SIZE:-$((NUM_ENVS * MAX_EPISODE_STEPS))}
MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-$BATCH_SIZE}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
FORWARD_BATCH_SIZE=${FORWARD_BATCH_SIZE:-4}
PPO_EPOCHS=${PPO_EPOCHS:-4}
MAX_PROMPT=${MAX_PROMPT:-12288}
MAX_RESP=${MAX_RESP:-768}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
STEP_GAMMA=${STEP_GAMMA:-1.0}
BASE_PORT=${BASE_PORT:-29876}   # separate from 0.6B (9876) and full-FT 4B (19876)

# Warm-start the critic before joint PPO. LoRA critic has near-zero output
# at init → advantages are noise until it fits reward shape.
CRITIC_WARMUP=${CRITIC_WARMUP:-10}

# Reward-shaping penalties. 0.02 (down from 0.05 baseline) because at 4B the
# base policy has enough capacity that we want game reward, not format
# shaping, to dominate.
FORMAT_PENALTY=${FORMAT_PENALTY:-0.02}
SEMANTIC_PENALTY=${SEMANTIC_PENALTY:-0.02}

# LoRA hyperparams — actor + critic both LoRA-adapted, rank 32, all-linear.
LORA_RANK=${LORA_RANK:-32}
LORA_ALPHA=${LORA_ALPHA:-64}

# vLLM KV-cache reservation. 40% is the same value the working full-FT
# 4-GPU sbatch uses.
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.40}

# Grammar-constrained decoding (vLLM V1 guided_regex). Locks the model into
# the ARC command grammar at sampling time: build TYPE ∈ {kitchen,shelter,
# casework}, hire KIND ∈ {untrained,trained}. Site IDs / building names are
# still \d+ / [A-Za-z0-9_]+ because they're episode-state-dependent (a static
# whitelist would over-constrain). Structure: REASONING: <free text without
# '<'> then 1..15 action tags with any whitespace between.
# Set GUIDED_REGEX="" to disable (compare unconstrained baseline).
export GUIDED_REGEX=${GUIDED_REGEX-'REASONING:[^<]{0,4000}(\s*(<build>(kitchen|shelter|casework),[0-9]{1,3}</build>|<hire>(untrained|trained),[0-9]{1,3}</hire>|<staff>[A-Za-z0-9_]{1,32},[0-9]{1,3}</staff>|<deconstruct>[A-Za-z0-9_]{1,32}</deconstruct>|<task>[0-9]{1,4},[0-9]{1,3}</task>|<train>[0-9]{1,3}</train>|<transfer>[a-z]+,[A-Za-z0-9_]+,[A-Za-z0-9_]+,[0-9]{1,4}</transfer>)){1,15}\s*'}

export RAY_TMPDIR=/tmp/ray_cora_${SLURM_JOB_ID:-local}
mkdir -p "$RAY_TMPDIR"
export RAY_OBJECT_STORE_MEMORY=${RAY_OBJECT_STORE_MEMORY:-8000000000}
export RAY_memory_monitor_refresh_ms=0

export ARC_TRAJECTORY_LOG="$LOG_DIR/${EXPERIMENT_NAME}.trajectory.jsonl"
: > "$ARC_TRAJECTORY_LOG"
export ARC_GAME_UNITY_LOG_DIR="$LOG_DIR"

# Reset per-run Unity-port counter
rm -f "/tmp/arc_game_port_counter_${BASE_PORT}_${NUM_ENVS}.txt"

cleanup() {
  echo "[arc_game] sweeping leftover headless processes"
  pkill -f "$ARC_GAME_BUILD" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

export WANDB_API_KEY=${WANDB_API_KEY:?WANDB_API_KEY must be set in env}
export WANDB_ENTITY=${WANDB_ENTITY:-cpulling}
export WANDB_PROJECT=${WANDB_PROJECT:-CORA_RL}

CKPT_DIR="$REPO_DIR/checkpoints/CORA_RL/${EXPERIMENT_NAME}"
mkdir -p "$CKPT_DIR"
SAVE_FREQ=${SAVE_FREQ:-5}
MAX_CKPT_TO_KEEP=${MAX_CKPT_TO_KEEP:-5}

CONFIG_PATH="$REPO_DIR/examples/sglang_multiturn/config"
DATA_DIR=${DATA_DIR:-/zfsauton/scratch/cpulling/data/gsm8k}

LOGF="$LOG_DIR/${EXPERIMENT_NAME}.log"
echo "=== $EXPERIMENT_NAME job ${SLURM_JOB_ID:-local} on ${SLURMD_NODENAME:-$(hostname)} @ $(date) ==="
echo "MODEL=$MODEL N_GPUS=$N_GPUS NUM_ENVS=$NUM_ENVS BATCH=$BATCH_SIZE LORA=r${LORA_RANK}a${LORA_ALPHA}"
echo "trajectory log: $ARC_TRAJECTORY_LOG"
echo "stdout log:     $LOGF"
echo "checkpoint dir: $CKPT_DIR"
nvidia-smi -L || true

python3 -m verl.trainer.main_ppo \
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
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${FORWARD_BATCH_SIZE} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEM_UTIL} \
    actor_rollout_ref.rollout.agent.num_workers=${NUM_ENVS} \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${FORWARD_BATCH_SIZE} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=10240 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=10240 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=10240 \
    actor_rollout_ref.model.lora_rank=${LORA_RANK} \
    actor_rollout_ref.model.lora_alpha=${LORA_ALPHA} \
    actor_rollout_ref.model.target_modules=all-linear \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.path=${MODEL} \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_epochs=${PPO_EPOCHS} \
    critic.ppo_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE} \
    critic.ppo_mini_batch_size=${MINI_BATCH_SIZE} \
    critic.model.fsdp_config.param_offload=True \
    critic.model.fsdp_config.optimizer_offload=True \
    critic.ppo_max_token_len_per_gpu=10240 \
    critic.forward_max_token_len_per_gpu=10240 \
    critic.forward_micro_batch_size_per_gpu=${FORWARD_BATCH_SIZE} \
    critic.model.lora_rank=${LORA_RANK} \
    critic.model.lora_alpha=${LORA_ALPHA} \
    critic.model.target_modules=all-linear \
    trainer.balance_batch=False \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='CORA_RL' \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node=${N_GPUS} \
    trainer.nnodes=1 \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=-1 \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.val_before_train=False \
    trainer.critic_warmup=${CRITIC_WARMUP} \
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
    +envs.arc_game_kwargs.format_penalty=${FORMAT_PENALTY} \
    +envs.arc_game_kwargs.semantic_penalty=${SEMANTIC_PENALTY} \
    +ray_kwargs.ray_init._temp_dir="$RAY_TMPDIR" \
    +ray_kwargs.ray_init.object_store_memory=${RAY_OBJECT_STORE_MEMORY} \
    +ray_kwargs.ray_init.include_dashboard=False \
    ++ray_kwargs.ray_init.num_cpus=8 \
    "$@" 2>&1 | tee "$LOGF"
