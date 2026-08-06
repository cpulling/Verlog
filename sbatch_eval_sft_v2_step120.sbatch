#!/bin/bash
#SBATCH --job-name=eval_sft_v2_120
#SBATCH --partition=general
#SBATCH --gres=gpu:a6000:1
#SBATCH --exclude=gpu28
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/zfsauton/scratch/cpulling/CORA/Verlog/logs/sft_eval/%x_%j.out
#SBATCH --error=/zfsauton/scratch/cpulling/CORA/Verlog/logs/sft_eval/%x_%j.err

mkdir -p /zfsauton/scratch/cpulling/CORA/Verlog/logs/sft_eval
export PATH="/zfsauton/scratch/cpulling/conda_envs/verlog/bin:$PATH"
export PYTHONUNBUFFERED=1
export ARC_GAME_PATH=/zfsauton/scratch/cpulling/CORA/ARC_Game/ARC_Game_New
export ARC_GAME_BUILD=$ARC_GAME_PATH/Build/Headless/Linux/ARC_Headless.x86_64
export VLLM_USE_V1=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd $ARC_GAME_PATH

OUT_DIR=/zfsauton/scratch/cpulling/CORA/sft_eval
mkdir -p "$OUT_DIR"

V1_40=/zfsauton/scratch/cpulling/CORA/Verlog/checkpoints/sft/qwen3_4b_tools_0721_1219/global_step_40/huggingface
V2_60=/zfsauton/scratch/cpulling/CORA/Verlog/checkpoints/sft/qwen3_4b_tools_v2_0722_1133/global_step_60/huggingface
V2_120=/zfsauton/scratch/cpulling/CORA/Verlog/checkpoints/sft/qwen3_4b_tools_v2_0722_1133/global_step_120/huggingface

# Reuse the earlier step_60 eval? No — earlier eval was on a separate 3-ep
# batch, but Unity task streams are random. Same ports would give same
# streams. Use new ports for fresh streams so all 3 conditions face the
# same distribution of disaster events (base_port 52000 for all).
echo
echo "=== SFT v1 step_40 (5-ep corpus, 40 steps) ==="
python eval_sft_model.py --model "$V1_40" \
    --episodes 3 --rounds 32 --base-port 52000 \
    --out "$OUT_DIR/sft_v1_step40_final.jsonl"

echo
echo "=== SFT v2 step_60 (30-ep corpus, 60 steps) ==="
python eval_sft_model.py --model "$V2_60" \
    --episodes 3 --rounds 32 --base-port 52100 \
    --out "$OUT_DIR/sft_v2_step60_final.jsonl"

echo
echo "=== SFT v2 step_120 (30-ep corpus, 120 steps) ==="
python eval_sft_model.py --model "$V2_120" \
    --episodes 3 --rounds 32 --base-port 52200 \
    --out "$OUT_DIR/sft_v2_step120.jsonl"

echo
echo "=== SUMMARY ==="
python <<'EOF'
import json
from pathlib import Path
D = Path("/zfsauton/scratch/cpulling/CORA/sft_eval")
for f in sorted(D.glob("*.jsonl")):
    rows = [json.loads(l) for l in open(f)]
    if not rows: continue
    r = sum(x["total_reward"] for x in rows)/len(rows)
    v = sum(x["valid_turn_frac"] for x in rows)/len(rows)
    t = sum(x["mean_tool_calls_per_turn"] for x in rows)/len(rows)
    e = sum(x["empty_call_turns"] for x in rows)
    print(f"  {f.stem:30s}  reward={r:6.3f}  valid={v:.1%}  tools/turn={t:.2f}  empty={e}")
EOF
