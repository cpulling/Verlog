#!/bin/bash
# RL test: Qwen3-1.7B PPO on the CORA surrogate (ARC_BACKEND=sim), 2×A6000, through the same
# trainer body as the 4B run (train_arc_game_qwen3_4B_tools_v3.sbatch) with test-sized defaults.
#
# Against collapse into invalid tool calls:
#  - a <tool_call> block the parser drops (unreadable JSON, cut off at the length cap) is a malformed
#    turn (cora_adapter), so FORMAT_PENALTY sees it instead of it passing as a silent no-op;
#  - no format bonus: +0.10 x 29 turns would outweigh a whole game's score (~1.4) and train "every
#    call executes" over play; passing a decision costs nothing (EMPTY_PENALTY=0), it is a choice;
#  - no entropy bonus (it pays for probability on non-JSON tokens); KL to the base model; low lr;
#  - preflight: the sim smoke must reproduce the rules baseline exactly, and one rules game runs on
#    the Unity headless build in ARC_GAME_PATH (reported, not a gate);
#  - watchdog: stops the job when malformed turns > max(0.5, 3x the first rollout's rate), or
#    empty turns > 0.9, for 3 consecutive actor steps (checkpoints every SAVE_FREQ steps stay).
#
# Submit from rhea (export knobs in the shell; --export=VAR breaks jobs on Auton):
#   sbatch train_arc_game_qwen3_1p7b_sim_2gpu.sbatch

#SBATCH --job-name=cora_rl_1p7b_sim
#SBATCH --output=/zfsauton/scratch/cpulling/CORA/Verlog/logs/slurm-%j.out
#SBATCH --error=/zfsauton/scratch/cpulling/CORA/Verlog/logs/slurm-%j.err
#SBATCH --mem=200G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=general
#SBATCH --qos=qos_general
#SBATCH --gres=gpu:a6000:2
#SBATCH --time=30:00:00
#SBATCH --gpu-bind=closest
#SBATCH --hint=nomultithread

set -euo pipefail

export REPO_DIR=${REPO_DIR:-/zfsauton/scratch/cpulling/CORA/Verlog_rl1102}
export ARC_GAME_PATH=${ARC_GAME_PATH:-/zfsauton/scratch/cpulling/CORA/release/arc_69f39dc0d_wt1102}
export ARC_BACKEND=${ARC_BACKEND:-sim}
# The human study's GUI rules, as the benchmark plays them (ARC_Game oracle/sim/env.py).
export CORA_STUDY_RULES=${CORA_STUDY_RULES:-1}

export MODEL=${MODEL:-Qwen/Qwen3-1.7B}
export REF_MODEL=${REF_MODEL:-$MODEL}
export NUM_ENVS=${NUM_ENVS:-16}
export CRITIC_WARMUP=${CRITIC_WARMUP:-2}
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-40}
export SAVE_FREQ=${SAVE_FREQ:-5}
export ACTOR_LR=${ACTOR_LR:-2e-6}
export ENTROPY_COEFF=${ENTROPY_COEFF:-0}
export FORMAT_PENALTY=${FORMAT_PENALTY:-0.05}
export EMPTY_PENALTY=${EMPTY_PENALTY:-0}
# No per-refused-call penalty: at 0.02 it was ~-1.0 a game (search agents and the model alike make
# 25-50 refused calls a game) against a ~0.3-1.5 score; a refused call already costs nothing in the game.
export SEMANTIC_PENALTY=${SEMANTIC_PENALTY:-0}
export FORMAT_BONUS=${FORMAT_BONUS:-0}
# 1.7B fits without CPU optimizer offload at micro-batch 4 (job 95597: 54 GB peak, update 2136 s -> 1538 s).
export OPTIM_OFFLOAD=${OPTIM_OFFLOAD:-False}
export MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-4}
# One whole game per env per step (16 x 29 turns), on the benchmark's prompt (tools_v3 resolves it
# from ARC_Game) and reply budget. Thinking stays off (tools_v3: enable_thinking=false).
export FULL_EPISODES=${FULL_EPISODES:-True}
export MAX_RESP=${MAX_RESP:-6000}
export PPO_MAX_TOKEN_LEN=${PPO_MAX_TOKEN_LEN:-18432}
export TAG=${TAG:-$(date +%m%d_%H%M)}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_1p7b_sim_2gpu_${TAG}}

export PATH=/zfsauton/scratch/cpulling/conda_envs/verlog/bin:$PATH
cd "$REPO_DIR"
mkdir -p logs/qwen3_4b
LOGF="$REPO_DIR/logs/qwen3_4b/${EXPERIMENT_NAME}.log"

echo "=== preflight $(date) on ${SLURMD_NODENAME:-?}: ARC $ARC_GAME_PATH  Verlog $(git log --oneline -1) ==="
PYTHONPATH="$REPO_DIR" python smoke_arc_game_env.py --backend sim --policy rules --episodes 1 --seed 7000 \
  | tee /dev/stderr | grep -q "final_score=1.3999 .* OK" \
  || { echo "FATAL: sim preflight did not reproduce the rules baseline (seed 7000, 1.3999)" >&2; exit 1; }
PORT=$(( 30000 + ${SLURM_JOB_ID:-0} % 2000 ))
PYTHONPATH="$REPO_DIR" timeout 900 python smoke_arc_game_env.py --backend unity --policy rules --episodes 1 \
  --seed 7000 --port $PORT 2>&1 | grep -E "\[smoke\]|Error|error" | tail -5 \
  || echo "WARN: Unity headless preflight failed (training uses the surrogate; continuing)"
pkill -f "$ARC_GAME_PATH/Build/Headless/Linux/ARC_Headless.x86_64" 2>/dev/null || true

# The offline critic dataset (generate_critic_dataset.py), built once on this job's CPUs.
if [ -n "${CRITIC_DATASET:-}" ] && [ ! -f "$CRITIC_DATASET" ]; then
  mkdir -p "$(dirname "$CRITIC_DATASET")"
  HF_HOME=${HF_HOME:-/zfsauton/scratch/cpulling/.cache/huggingface} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH="$REPO_DIR" \
    python generate_critic_dataset.py --out "$CRITIC_DATASET" --tokenizer "$MODEL" --prompt-pack "$PROMPT_PACK" --rollouts $CRITIC_ROLLOUTS
fi

# Watchdog over the console metrics ("(TaskRunner pid=..) step:N - key:value - ...").
python - "$LOGF" "$CRITIC_WARMUP" "${SLURM_JOB_ID:-}" <<'EOF' &
import os, re, subprocess, sys, time
logf, warmup, job = sys.argv[1], int(sys.argv[2]), sys.argv[3]
def rate(line, name):
    vals = [float(v) for k, v in re.findall(r"([\w/]*%s[\w/]*):([-\d.eE+]+)" % name, line)
            if "max" not in k and "min" not in k]
    return vals[0] if vals else None
seen, base, bad = set(), None, 0
while True:
    time.sleep(60)
    try:
        lines = [l for l in open(logf, errors="replace") if re.search(r"step:\d+ - ", l)]
    except OSError:
        continue
    for line in lines:
        step = int(re.search(r"step:(\d+) - ", line).group(1))
        if step in seen:
            continue
        seen.add(step)
        mal, emp = rate(line, "malformed_turn"), rate(line, "empty_turn")
        if mal is None:
            continue
        base = mal if base is None else base
        trip = step > warmup and (mal > max(0.5, 3 * base) or (emp or 0) > 0.9)
        bad = bad + 1 if trip else 0
        print(f"[watchdog] step {step}: malformed {mal:.3f} (first {base:.3f}) empty {emp}  strikes {bad}", flush=True)
        if bad >= 3 and job:
            print(f"[watchdog] STOP: invalid-call collapse at step {step}", flush=True)
            subprocess.run(["scancel", job])
EOF

exec bash "$REPO_DIR/train_arc_game_qwen3_4B_tools_v3.sbatch"
