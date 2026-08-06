#!/bin/bash
#SBATCH --job-name=eval_ppo_20737
#SBATCH --partition=general
#SBATCH --gres=gpu:a6000:1
#SBATCH --exclude=gpu28
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/zfsauton/scratch/cpulling/CORA/Verlog/logs/sft_eval/%x_%j.out
#SBATCH --error=/zfsauton/scratch/cpulling/CORA/Verlog/logs/sft_eval/%x_%j.err

set -euo pipefail
mkdir -p /zfsauton/scratch/cpulling/CORA/Verlog/logs/sft_eval

export PATH="/zfsauton/scratch/cpulling/conda_envs/verlog/bin:$PATH"
export PYTHONUNBUFFERED=1
export ARC_GAME_PATH=/zfsauton/scratch/cpulling/CORA/ARC_Game/ARC_Game_New
export ARC_GAME_BUILD=$ARC_GAME_PATH/Build/Headless/Linux/ARC_Headless.x86_64
export ARC_STABLE_TASK_TOKENS=1
export VLLM_USE_V1=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CKPT_ROOT=/zfsauton/scratch/cpulling/CORA/Verlog/checkpoints/CORA_RL/qwen3_4b_sft120_ppo_0722_1333
MERGE_DIR=/zfsauton/scratch/cpulling/CORA/ppo_20737_merged
OUT_DIR=/zfsauton/scratch/cpulling/CORA/sft_eval
mkdir -p "$MERGE_DIR" "$OUT_DIR"

cd /zfsauton/scratch/cpulling/CORA/Verlog

# Merge FSDP shards → HF safetensors.
# step_40 was pruned individually but preserved as best/actor/ (best_iteration=40).
# step_58 is the final checkpoint.
declare -A SOURCES
SOURCES[best_step40]="$CKPT_ROOT/best/actor"
SOURCES[step_58]="$CKPT_ROOT/global_step_58/actor"
for NAME in "${!SOURCES[@]}"; do
    TGT="$MERGE_DIR/$NAME"
    SRC="${SOURCES[$NAME]}"
    if [ -f "$TGT/model.safetensors" ] || [ -f "$TGT/model.safetensors.index.json" ]; then
        echo "already merged: $TGT"
    else
        echo "=== merging $NAME (from $SRC) ==="
        python -m verl.model_merger merge \
            --backend fsdp \
            --local_dir "$SRC" \
            --target_dir "$TGT" \
            --trust-remote-code
    fi
done

cd $ARC_GAME_PATH

# Same 3-episode base ports as SFT step_120 eval (52200) for direct comparison.
# Also run PPO step_40 on 52300 (separate stream) and PPO step_58 on 52400.
# SFT step_120 already has results on 52200 in sft_v2_step120.jsonl.

echo
echo "=== PPO 20737 step_40 (best batch-mean) — 3 ep × 32 rounds ==="
python eval_sft_model.py --model "$MERGE_DIR/best_step40" \
    --episodes 3 --rounds 32 --base-port 52300 \
    --out "$OUT_DIR/ppo_20737_step40.jsonl"

echo
echo "=== PPO 20737 step_58 (final, highest peak) — 3 ep × 32 rounds ==="
python eval_sft_model.py --model "$MERGE_DIR/step_58" \
    --episodes 3 --rounds 32 --base-port 52400 \
    --out "$OUT_DIR/ppo_20737_step58.jsonl"

echo
echo "=== SUMMARY ==="
python <<'EOF'
import json
from pathlib import Path
D = Path("/zfsauton/scratch/cpulling/CORA/sft_eval")
targets = [
    "sft_v2_step120",           # SFT baseline (already run on port 52200)
    "ppo_20737_step40",
    "ppo_20737_step58",
]
print(f"{'condition':<25s} {'reward':>8s} {'valid':>8s} {'tools/turn':>11s} {'empty':>7s}")
for name in targets:
    f = D / f"{name}.jsonl"
    if not f.exists():
        print(f"  {name:<25s}  MISSING")
        continue
    rows = [json.loads(l) for l in open(f)]
    if not rows: continue
    r = sum(x["total_reward"] for x in rows)/len(rows)
    v = sum(x["valid_turn_frac"] for x in rows)/len(rows)
    t = sum(x["mean_tool_calls_per_turn"] for x in rows)/len(rows)
    e = sum(x["empty_call_turns"] for x in rows)
    print(f"  {name:<25s} {r:>8.3f} {v:>8.1%} {t:>11.2f} {e:>7d}")
EOF
