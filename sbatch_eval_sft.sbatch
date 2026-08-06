#!/bin/bash
#SBATCH --job-name=eval_sft
#SBATCH --partition=general
#SBATCH --gres=gpu:a6000:1
#SBATCH --exclude=gpu28
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:45:00
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

BASE=Qwen/Qwen3-0.6B
SFT=/zfsauton/scratch/cpulling/CORA/Verlog/checkpoints/sft_smoke/tools_0721_1056/global_step_10/hf_merged

echo "=== BASE MODEL: $BASE ==="
python eval_sft_model.py --model "$BASE" \
    --episodes 3 --rounds 32 --base-port 49000 \
    --out "$OUT_DIR/base_qwen3_0p6b.jsonl"

echo
echo "=== SFT MODEL: $SFT ==="
python eval_sft_model.py --model "$SFT" \
    --episodes 3 --rounds 32 --base-port 49100 \
    --out "$OUT_DIR/sft_smoke_step10.jsonl"
