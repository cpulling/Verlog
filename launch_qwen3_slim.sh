#!/bin/bash
# Slim Qwen3-0.6B run on ARC — minimal_cmd_v3 wrapper + no-think + PPO batching.
# Enables ARC_TRAJECTORY_LOG so each turn's raw model output + parsed action
# + reward + game state is appended as JSONL for later inspection.

set -euo pipefail

export PATH="/zfsauton/scratch/cpulling/conda_envs/verlog/bin:$PATH"
export VLLM_USE_V1=1
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

REPO_DIR=/zfsauton/scratch/cpulling/CORA/Verlog
ARC_GAME_PATH=/zfsauton/scratch/cpulling/CORA/ARC_Game/ARC_Game_New
ARC_GAME_BUILD=${ARC_GAME_PATH}/Build/Headless/Linux/ARC_Headless.x86_64
export ARC_GAME_PATH ARC_GAME_BUILD

TAG=${TAG:-$(date +%m%d_%H%M)}
LOG_DIR="$REPO_DIR/logs/qwen3_local"
mkdir -p "$LOG_DIR"

export MODEL=Qwen/Qwen3-0.6B
export EXPERIMENT_NAME=qwen3_0p6b_slim_long_${TAG}
export NUM_ENVS=1
# A full ARC game is 8 days × 4 segments = 32 turns. Let episodes play out to
# their natural end rather than truncating at 8 (which was only there to fit
# the minimum-batch cgroup constraint during earlier vLLM-init debugging).
export MAX_EPISODE_STEPS=32
# 2 full episodes per PPO step for a cleaner gradient signal. verl's per-step
# turn counter requires train_batch_size ≥ NUM_ENVS × MAX_EPISODE_STEPS.
export BATCH_SIZE=128
export MINI_BATCH_SIZE=128     # whole batch is one minibatch → 1 grad step / epoch
export MICRO_BATCH_SIZE=1      # grad-accum 128 micros → mini
export FORWARD_BATCH_SIZE=4
# 4 PPO epochs: epoch 1 mirrors REINFORCE (ratio ≡ 1); epochs 2-4 actually
# use the frozen old_logprob so the importance ratio, clipping, and KL do
# their PPO thing. With epochs=1 the whole PPO ratio machinery is a no-op.
export PPO_EPOCHS=4
echo "[launch] just after PPO_EPOCHS export: PPO_EPOCHS=$PPO_EPOCHS" >&2
export MAX_PROMPT=12288
export MAX_RESP=768
export TOTAL_EPOCHS=1
# Discount factor. Was 1.0 (undiscounted) — theoretically right target but
# return variance over 32-turn episodes blew up and the critic couldn't
# converge. 0.95 shortens the effective horizon (~20 turns of meaningful
# lookahead) so the critic sees stabler targets and can actually learn to
# value early "invest now, pay off later" states like a fresh build.
export STEP_GAMMA=0.95
# Warm-start the critic for CRITIC_WARMUP iterations before the joint PPO
# loop starts. The actor stays frozen during warmup; rollouts feed only the
# critic. Same idea as verl's async_ticker_admissions setup — without this,
# the actor gets random advantages from a random critic in the first few
# steps and burns useful gradient signal on noise. 10 is a reasonable
# default; bump higher if reward variance stays large.
export CRITIC_WARMUP=${CRITIC_WARMUP:-10}
export RESTART_UNITY_EACH_EPISODE=False

# Reward-shaping penalties. Post-mortem on the 0710_0110 0.6B run showed raw
# game reward is ~zero-mean noise across valid moves (mean +0.0005 over 145
# valid turns), and 0.05-magnitude flat penalties swamped the signal, so the
# actor gradient collapsed to ~1e-4 by step 5. Drop to 0.01 so raw game reward
# isn't drowned. Delayed-reward credit assignment (build costs now, pays off
# later) is left to the critic — see STEP_GAMMA and CRITIC_WARMUP below.
# - format_penalty: flat per-turn when extract_action rejected the LLM output.
# - semantic_penalty: per Unity-rejected command (exec_attempted - exec_success).
export FORMAT_PENALTY=${FORMAT_PENALTY:-0.01}
export SEMANTIC_PENALTY=${SEMANTIC_PENALTY:-0.01}

export RAY_TMPDIR=/tmp/ray_cora_${SLURM_JOB_ID:-local}
mkdir -p "$RAY_TMPDIR"

# Ray defaults object_store_memory to ~30% of node RAM (~150 GB on this box).
# That plus vLLM CPU-side buffers blows past the 64 GB slurm cgroup ceiling
# and a Ray worker gets OOM-killed → "Actor unavailable: Socket closed".
export RAY_OBJECT_STORE_MEMORY=${RAY_OBJECT_STORE_MEMORY:-4000000000}   # 4 GB
export RAY_memory_monitor_refresh_ms=0

export ARC_TRAJECTORY_LOG="$LOG_DIR/qwen3_0p6b_slim_long_${TAG}.trajectory.jsonl"
: > "$ARC_TRAJECTORY_LOG"

export WANDB_API_KEY=${WANDB_API_KEY:?WANDB_API_KEY must be set in env}
export WANDB_ENTITY=${WANDB_ENTITY:-cpulling}
export WANDB_PROJECT=${WANDB_PROJECT:-CORA_RL}

cd "$REPO_DIR"

LOGF="$LOG_DIR/qwen3_0p6b_slim_long_${TAG}.log"
echo "trajectory log: $ARC_TRAJECTORY_LOG"
echo "stdout log:     $LOGF"
echo "[launch] just before dryrun: PPO_EPOCHS=$PPO_EPOCHS" >&2
echo "DEBUG: PPO_EPOCHS=$PPO_EPOCHS BATCH_SIZE=$BATCH_SIZE MINI_BATCH_SIZE=$MINI_BATCH_SIZE MICRO_BATCH_SIZE=$MICRO_BATCH_SIZE" | tee -a "$LOGF"

DEBUG_FILE="$LOG_DIR/qwen3_0p6b_slim_long_${TAG}.debug.txt"
{
  echo "=== launch script debug ==="
  echo "PPO_EPOCHS=$PPO_EPOCHS"
  echo "BATCH_SIZE=$BATCH_SIZE"
  echo "MINI_BATCH_SIZE=$MINI_BATCH_SIZE"
  echo "MICRO_BATCH_SIZE=$MICRO_BATCH_SIZE"
  echo "arg 1: actor_rollout_ref.actor.ppo_epochs=$PPO_EPOCHS"
  echo "arg 2: critic.ppo_epochs=$PPO_EPOCHS"
} > "$DEBUG_FILE"

# Checkpoint every SAVE_FREQ PPO steps to survive sbatch time-limit or session
# death. resume_mode=auto (verl default) re-launches from the newest checkpoint
# in default_local_dir; keep TAG stable across relaunches to reuse the dir.
CKPT_DIR="$REPO_DIR/checkpoints/CORA_RL/${EXPERIMENT_NAME}"
mkdir -p "$CKPT_DIR"
echo "checkpoint dir: $CKPT_DIR"

bash train_arc_game_dryrun.sh \
    algorithm.use_kl_in_reward=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.actor.ppo_epochs=$PPO_EPOCHS \
    critic.ppo_epochs=$PPO_EPOCHS \
    +data.apply_chat_template_kwargs.enable_thinking=false \
    +ray_kwargs.ray_init._temp_dir="$RAY_TMPDIR" \
    +ray_kwargs.ray_init.object_store_memory=$RAY_OBJECT_STORE_MEMORY \
    +envs.arc_game_kwargs.format_penalty=$FORMAT_PENALTY \
    +envs.arc_game_kwargs.semantic_penalty=$SEMANTIC_PENALTY \
    trainer.critic_warmup=$CRITIC_WARMUP \
    trainer.save_freq=${SAVE_FREQ:-5} \
    trainer.default_local_dir="$CKPT_DIR" \
    trainer.max_actor_ckpt_to_keep=${MAX_CKPT_TO_KEEP:-5} \
    "$@" 2>&1 | tee "$LOGF"
