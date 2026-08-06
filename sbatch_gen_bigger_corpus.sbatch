#!/bin/bash
#SBATCH --job-name=gen_sft_corpus
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/zfsauton/scratch/cpulling/CORA/Verlog/logs/sft_corpus/%x_%j.out
#SBATCH --error=/zfsauton/scratch/cpulling/CORA/Verlog/logs/sft_corpus/%x_%j.err

# Generate a 30 train + 10 val baseline corpus. Sequential per episode (~40s each),
# so ~30 min total. Runs on CPU partition — no GPU, no A6000 contention with PPO.
# Sliced parquet is written for both train and val, matching RL max_text_history=1.

mkdir -p /zfsauton/scratch/cpulling/CORA/Verlog/logs/sft_corpus

export PATH="/zfsauton/scratch/cpulling/conda_envs/verlog/bin:$PATH"
export PYTHONUNBUFFERED=1
export ARC_GAME_PATH=/zfsauton/scratch/cpulling/CORA/ARC_Game/ARC_Game_New
export ARC_GAME_BUILD=$ARC_GAME_PATH/Build/Headless/Linux/ARC_Headless.x86_64
export ARC_STABLE_TASK_TOKENS=1

cd $ARC_GAME_PATH

# Ports 46000+ don't overlap PPO run's Unity ports (which are in 40000-49999 range
# derived from SLURM_JOB_ID). Different range per split to avoid intra-episode
# collisions.
SFT_DIR=/zfsauton/scratch/cpulling/CORA/sft_corpus_v2
mkdir -p "$SFT_DIR"

echo "=== TRAIN: 30 episodes × 32 rounds, tools mode ==="
python generate_sft_trajectories.py \
    --episodes 30 --rounds 32 --mode tools \
    --slice turn --history 1 \
    --out "$SFT_DIR/train.parquet" --base-port 46000

echo
echo "=== VAL: 10 episodes × 32 rounds, tools mode ==="
python generate_sft_trajectories.py \
    --episodes 10 --rounds 32 --mode tools \
    --slice turn --history 1 \
    --out "$SFT_DIR/val.parquet" --base-port 46100

echo
echo "=== corpus stats ==="
python <<'EOF'
import pandas as pd
for name in ("train", "val"):
    df = pd.read_parquet(f"/zfsauton/scratch/cpulling/CORA/sft_corpus_v2/{name}.parquet")
    print(f"  {name}: {len(df)} sliced rows")
EOF
