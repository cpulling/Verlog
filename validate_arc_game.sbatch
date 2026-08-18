#!/bin/bash
# Validation-ONLY rollouts of an arbitrary HF model on the ARCGame env, through the
# same veRL + vLLM + tool-calling path the RL runs use. No training, no optimizer, no
# checkpoints -- trainer.val_only=True makes main_ppo do a single validation pass and exit.
#
# WHY THIS EXISTS
# The Ollama benchmark and the RL wing were measuring different things through different
# interfaces. This runs a candidate model through the RL wing's own rollout code, so a
# benchmark number and an RL baseline are produced by the same machinery: same
# arc_tools.yaml (schema-identical to cora_tools, verified 0 diff), same env, same
# obs_encoder rendering.
#
# STATUS: AUTHORED BUT NEVER EXECUTED -- no cluster access at time of writing. Treat the
# first run as a debug run. Most likely friction points are called out inline.
#
# USAGE
#   MODEL=Qwen/Qwen3-4B sbatch validate_arc_game.sbatch
#   MODEL=Qwen/Qwen3.5-4B NUM_ENVS=8 VAL_EPISODES=10 sbatch validate_arc_game.sbatch
#
#SBATCH --job-name=arc_validate
#SBATCH --partition=general
#SBATCH --gres=gpu:a6000:1
#SBATCH --exclude=gpu28
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH --output=/zfsauton/scratch/cpulling/CORA/Verlog/logs/validate/%x_%j.out
#SBATCH --error=/zfsauton/scratch/cpulling/CORA/Verlog/logs/validate/%x_%j.err

set -uo pipefail

REPO_DIR=/zfsauton/scratch/cpulling/CORA/Verlog
mkdir -p "$REPO_DIR/logs/validate"
export PATH="/zfsauton/scratch/cpulling/conda_envs/verlog/bin:$PATH"
export PYTHONUNBUFFERED=1
export VLLM_USE_V1=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# The env lives in the GAME repo; verl imports it via ARC_GAME_PATH.
export ARC_GAME_PATH=/zfsauton/scratch/cpulling/CORA/ARC_Game/ARC_Game_New
export ARC_GAME_BUILD=$ARC_GAME_PATH/Build/Headless/Linux/ARC_Headless.x86_64
export ARC_STABLE_TASK_TOKENS=${ARC_STABLE_TASK_TOKENS:-1}
# WITHOUT THIS the env falls back to "text" mode -- the XML <build>...</build> grammar --
# even though tool_config_path is passed, because get_instruction_prompt() selects the
# HOW-TO-ACT paragraph off ARC_ACTION_MODE, not off the tool config. The training launchers
# all set it; omitting it here would have silently benchmarked the wrong action interface.
export ARC_ACTION_MODE=${ARC_ACTION_MODE:-tool_calls}

cd "$REPO_DIR"

MODEL=${MODEL:-Qwen/Qwen3-4B}
NUM_ENVS=${NUM_ENVS:-4}
MAX_EPISODE_STEPS=${MAX_EPISODE_STEPS:-32}
VAL_EPISODES=${VAL_EPISODES:-8}
BATCH_SIZE=${BATCH_SIZE:-$((NUM_ENVS * MAX_EPISODE_STEPS))}

MAX_PROMPT=${MAX_PROMPT:-12288}
# ------------------------------------------------------------------------------------
# 6000, NOT the training script's 768. Measured on qwen3:4b: with thinking nominally
# OFF it still emits ~14k chars (~3.5k tokens) of chain-of-thought into the CONTENT
# channel before its tool call, because the Qwen chat template's enable_thinking=false
# prefills an empty <think></think> and the model reopens it -- so only the CLOSING tag
# appears and the reasoning parser never splits it out. At a 2000-token cap that
# produced NO action in 88% of rounds. 768 would be far worse. This is a chat-template
# mechanism, not an Ollama quirk, so it is expected to reproduce here.
#
# ACTION ITEM for the training runs: MAX_RESP=768 in train_arc_game_qwen3_4B_tools*.sbatch
# is very likely truncating the policy mid-thought. Worth re-checking those runs.
# ------------------------------------------------------------------------------------
MAX_RESP=${MAX_RESP:-6000}

TOOL_CONFIG_PATH=${TOOL_CONFIG_PATH:-verl/envs/environments/arc_game/tool_config/arc_tools.yaml}
MULTI_TURN_FORMAT=${MULTI_TURN_FORMAT:-hermes}
CONFIG_PATH="$(pwd)/examples/sglang_multiturn/config"
DATA_DIR=${DATA_DIR:-/zfsauton/scratch/cpulling/data/gsm8k}

# Per-job Unity ports; an orphaned headless from a killed job keeps its port bound and
# the next job attaches to the dead server (Unity then rejects it as a second client).
if [ -n "${SLURM_JOB_ID:-}" ]; then
  BASE_PORT=${BASE_PORT:-$(( 40000 + (SLURM_JOB_ID * 10) % 10000 ))}
else
  BASE_PORT=${BASE_PORT:-41000}
fi
rm -f "/tmp/arc_game_port_counter_${BASE_PORT}_${NUM_ENVS}.txt"

TAG=${TAG:-$(date +%m%d_%H%M)}
SAFE_MODEL=$(echo "$MODEL" | tr '/:.' '___')
EXPERIMENT_NAME=${EXPERIMENT_NAME:-validate_${SAFE_MODEL}_${TAG}}

export ARC_GAME_UNITY_LOG_DIR="${REPO_DIR}/logs/validate"
export ARC_TRAJECTORY_LOG="$REPO_DIR/logs/validate/${EXPERIMENT_NAME}.trajectory.jsonl"
: > "$ARC_TRAJECTORY_LOG"

cleanup() {
  echo "[arc_game] sweeping leftover headless processes"
  pkill -f "$ARC_GAME_BUILD" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "=== $EXPERIMENT_NAME job ${SLURM_JOB_ID:-local} on ${SLURMD_NODENAME:-$(hostname)} @ $(date) ==="
echo "MODEL=$MODEL NUM_ENVS=$NUM_ENVS MAX_RESP=$MAX_RESP VAL_EPISODES=$VAL_EPISODES"
echo "trajectory log: $ARC_TRAJECTORY_LOG"
nvidia-smi -L || true

# Per-model thinking overlay. Qwen3/Qwen3.5 templates honor enable_thinking; gpt-oss has
# no "none" level (its minimum is "low") and ignores enable_thinking; gemma3 has neither
# and no tool support at all. Verify a model's tokenizer_config.json actually contains the
# enable_thinking branch before trusting the flag -- vLLM silently discards chat template
# kwargs a template does not declare.
THINK_ARGS=()
case "$MODEL" in
  *[Qq]wen3*) THINK_ARGS+=("+data.apply_chat_template_kwargs.enable_thinking=false") ;;
  *gpt-oss*)  THINK_ARGS+=("+actor_rollout_ref.rollout.reasoning_effort=low") ;;
esac

numactl --interleave=all python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='gsm8k_multiturn_grpo' \
    algorithm.adv_estimator=gae \
    data.train_files=${DATA_DIR}/train.parquet \
    data.val_files=${DATA_DIR}/test.parquet \
    data.train_batch_size=${BATCH_SIZE} \
    data.val_batch_size=${BATCH_SIZE} \
    data.max_prompt_length=${MAX_PROMPT} \
    data.max_response_length=${MAX_RESP} \
    "${THINK_ARGS[@]}" \
    envs.env_name=arc_game \
    envs.max_episode_steps=${MAX_EPISODE_STEPS} \
    actor_rollout_ref.model.path=${MODEL} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.agent.num_workers=${NUM_ENVS} \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=${TOOL_CONFIG_PATH} \
    actor_rollout_ref.rollout.multi_turn.format=${MULTI_TURN_FORMAT} \
    critic.model.path=${MODEL} \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.val_before_train=True \
    trainer.val_only=True \
    trainer.total_epochs=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=1 \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    2>&1 | tee "$REPO_DIR/logs/validate/${EXPERIMENT_NAME}.log"

echo "=== done @ $(date) ==="
echo "trajectories: $ARC_TRAJECTORY_LOG"
