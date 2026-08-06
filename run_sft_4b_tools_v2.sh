#!/bin/bash
# SFT round-2: Qwen3-4B on the bigger 30+10-episode sliced corpus (960/320 rows).
#
# Old run (160/64 rows, 3 epochs, 60 steps): val floored at 0.09.
# New run: 4 epochs on 960 rows = 480 steps, LR 1e-5 (proven), warmup 5%,
# save every 60 steps so we can eval several checkpoints post-hoc.
# ~17s/step → ~2.3 hours wall clock on 2 A6000s.

set -euo pipefail

export PATH="/zfsauton/scratch/cpulling/conda_envs/verlog/bin:$PATH"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# WANDB: prefer env, fall back to ~/.netrc. Never echo the key.
if [ -z "${WANDB_API_KEY:-}" ] && [ -r "$HOME/.netrc" ]; then
  _wandb_key=$(awk '/machine api.wandb.ai/{f=1;next} f && /password/{print $2; exit}' "$HOME/.netrc")
  [ -n "$_wandb_key" ] && export WANDB_API_KEY="$_wandb_key"
  unset _wandb_key
fi
: "${WANDB_API_KEY:?WANDB_API_KEY must be in env or ~/.netrc}"
export WANDB_ENTITY=${WANDB_ENTITY:-cpulling}
export WANDB_PROJECT=${WANDB_PROJECT:-CORA_RL}

REPO_DIR=/zfsauton/scratch/cpulling/CORA/Verlog
SFT_DIR=/zfsauton/scratch/cpulling/CORA/sft_corpus_v2
TAG=${TAG:-$(date +%m%d_%H%M)}
SAVE_DIR="$REPO_DIR/checkpoints/sft/qwen3_4b_tools_v2_${TAG}"
mkdir -p "$SAVE_DIR"

cd "$REPO_DIR"

echo "=== SFT v2: Qwen3-4B on 960/320-row corpus ==="
echo "    train:   $SFT_DIR/train.parquet"
echo "    val:     $SFT_DIR/val.parquet"
echo "    save:    $SAVE_DIR"
echo "    gpus:    $CUDA_VISIBLE_DEVICES"

NGPU=$(python -c "import os; print(len(os.environ.get('CUDA_VISIBLE_DEVICES','0').split(',')))")

# 960 rows / batch 8 = 120 steps/epoch × 4 epochs = 480 total. Save every
# 60 steps (twice per epoch after epoch 1), so we get 8 ckpts. Warmup ratio
# 5% = 24 steps of LR warmup.
torchrun --nnodes=1 --nproc_per_node=$NGPU \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$SFT_DIR/train.parquet \
    data.val_files=$SFT_DIR/val.parquet \
    data.multiturn.enable=true \
    data.multiturn.messages_key=messages \
    data.multiturn.tools_key=tools \
    data.train_batch_size=8 \
    data.micro_batch_size_per_gpu=1 \
    data.max_length=8000 \
    data.truncation=error \
    model.partial_pretrain=Qwen/Qwen3-4B \
    model.enable_gradient_checkpointing=true \
    model.trust_remote_code=true \
    model.fsdp_config.model_dtype=bf16 \
    model.fsdp_config.cpu_offload=false \
    optim.lr=1e-5 \
    optim.warmup_steps_ratio=0.05 \
    ulysses_sequence_parallel_size=1 \
    use_remove_padding=true \
    trainer.default_local_dir=$SAVE_DIR \
    trainer.project_name=CORA_RL \
    trainer.experiment_name=sft_qwen3_4b_tools_v2_${TAG} \
    trainer.total_epochs=4 \
    trainer.save_freq=60 \
    trainer.test_freq=15 \
    trainer.max_ckpt_to_keep=8 \
    trainer.checkpoint.save_contents='[model,optimizer,extra,hf_model]' \
    trainer.logger=${SFT_LOGGER:-[console]} \
    "$@"

echo
echo "=== SFT v2 done → $SAVE_DIR ==="
ls "$SAVE_DIR"/global_step_*/huggingface/ 2>/dev/null | tail -20
