#!/bin/bash
# SFT smoke on the baseline-generated tools corpus.
#
# Purpose: confirm Verlog's fsdp_sft_trainer accepts our multiturn parquet
# (messages + tools columns), the chat template renders assistant tool_calls
# correctly, loss decreases across a handful of steps, and a checkpoint saves.
#
# 5 train episodes + 2 val, Qwen3-0.6B, single A6000, 2 steps.
# NOT meant to converge — this is a pipeline smoke.

set -euo pipefail

export PATH="/zfsauton/scratch/cpulling/conda_envs/verlog/bin:$PATH"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONUNBUFFERED=1
export WANDB_MODE=${WANDB_MODE:-disabled}   # smoke: no wandb by default
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

REPO_DIR=/zfsauton/scratch/cpulling/CORA/Verlog
SFT_DIR=/zfsauton/scratch/cpulling/CORA/sft_corpus
TAG=${TAG:-$(date +%m%d_%H%M)}
SAVE_DIR="$REPO_DIR/checkpoints/sft_smoke/tools_${TAG}"
mkdir -p "$SAVE_DIR"

cd "$REPO_DIR"

echo "=== SFT smoke: Qwen3-0.6B on baseline tools corpus (5 train / 2 val ep) ==="
echo "    train:   $SFT_DIR/train.parquet"
echo "    val:     $SFT_DIR/val.parquet"
echo "    save:    $SAVE_DIR"
echo

# Ulysses SP=1 (single GPU), FSDP2, small batch. max_length=48000 covers our
# ~35-38k-token rows without truncation (rows were 32 full-episode turns).
torchrun --nnodes=1 --nproc_per_node=1 \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$SFT_DIR/train_sliced.parquet \
    data.val_files=$SFT_DIR/val_sliced.parquet \
    data.multiturn.enable=true \
    data.multiturn.messages_key=messages \
    data.multiturn.tools_key=tools \
    data.train_batch_size=8 \
    data.micro_batch_size_per_gpu=2 \
    data.max_length=8000 \
    data.truncation=error \
    model.partial_pretrain=Qwen/Qwen3-0.6B \
    model.enable_gradient_checkpointing=true \
    model.trust_remote_code=true \
    model.fsdp_config.model_dtype=bf16 \
    optim.lr=1e-5 \
    ulysses_sequence_parallel_size=1 \
    use_remove_padding=true \
    trainer.default_local_dir=$SAVE_DIR \
    trainer.project_name=cora_sft_smoke \
    trainer.experiment_name=tools_${TAG} \
    trainer.total_epochs=1 \
    trainer.total_training_steps=10 \
    trainer.save_freq=10 \
    trainer.test_freq=1 \
    trainer.logger=[console] \
    "$@"
