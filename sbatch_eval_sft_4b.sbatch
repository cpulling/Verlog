#!/bin/bash
#SBATCH --job-name=eval_sft_4b
#SBATCH --partition=general
#SBATCH --gres=gpu:a6000:1
#SBATCH --exclude=gpu28
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
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

SFT_ROOT=/zfsauton/scratch/cpulling/CORA/Verlog/checkpoints/sft/qwen3_4b_tools_0721_1219

# 4 conditions: base + 3 ckpts. 3 episodes each × 32 rounds. Ports don't
# collide because each `python eval_sft_model.py` closes its env before the
# next launches.
declare -a CONDITIONS=(
    "base_qwen3_4b|Qwen/Qwen3-4B|50000"
    "sft_step20|$SFT_ROOT/global_step_20/huggingface|50100"
    "sft_step40|$SFT_ROOT/global_step_40/huggingface|50200"
    "sft_step60|$SFT_ROOT/global_step_60/huggingface|50300"
)

for cond in "${CONDITIONS[@]}"; do
    IFS='|' read -r name model port <<< "$cond"
    echo
    echo "=== $name: $model ==="
    python eval_sft_model.py --model "$model" \
        --episodes 3 --rounds 32 --base-port "$port" \
        --out "$OUT_DIR/${name}.jsonl"
done

echo
echo "=== summary ==="
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
    print(f"  {f.stem:20s}  reward={r:6.3f}  valid={v:.1%}  tools/turn={t:.2f}  empty={e}")
EOF
