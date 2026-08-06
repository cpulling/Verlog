#!/bin/bash
# Full-ish SFT run: Qwen3-4B on the baseline tools corpus (5 train + 2 val
# episodes, per-turn sliced to 160 + 64 rows).
#
# 3 epochs at batch 8 → 60 steps. Save best-loss + HF-format checkpoint so
# the resulting model can be handed to launch_qwen3_slim_tools.sh as MODEL=…
# with no extra merge step.

set -euo pipefail

export PATH="/zfsauton/scratch/cpulling/conda_envs/verlog/bin:$PATH"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
# WANDB creds: prefer $WANDB_API_KEY from env, fall back to ~/.netrc
# password line (which the wandb library also reads). We never echo the key.
if [ -z "${WANDB_API_KEY:-}" ] && [ -r "$HOME/.netrc" ]; then
  key=$(awk '/machine api.wandb.ai/{f=1;next} f && /password/{print $2; exit}' "$HOME/.netrc")
  if [ -n "$key" ]; then
    export WANDB_API_KEY="$key"
  fi
fi
: "${WANDB_API_KEY:?WANDB_API_KEY must be in env or ~/.netrc}"
export WANDB_ENTITY=${WANDB_ENTITY:-cpulling}
export WANDB_PROJECT=${WANDB_PROJECT:-CORA_RL}

REPO_DIR=/zfsauton/scratch/cpulling/CORA/Verlog
SFT_DIR=/zfsauton/scratch/cpulling/CORA/sft_corpus
TAG=${TAG:-$(date +%m%d_%H%M)}
SAVE_DIR="$REPO_DIR/checkpoints/sft/qwen3_4b_tools_${TAG}"
mkdir -p "$SAVE_DIR"

cd "$REPO_DIR"

echo "=== SFT: Qwen3-4B on baseline tools corpus (160 train / 64 val) ==="
echo "    train:   $SFT_DIR/train_sliced.parquet"
echo "    val:     $SFT_DIR/val_sliced.parquet"
echo "    save:    $SAVE_DIR"
echo "    gpus:    $CUDA_VISIBLE_DEVICES"
echo

# 2 GPUs → FSDP2 shard the 4B model. batch 8 per step (2 GPUs × 4 micro), 3
# epochs on 160 rows = 60 steps. Sequence-parallel=1 (rows are <=6k tokens,
# no need to split further).
NGPU=$(python -c "import os; print(len(os.environ.get('CUDA_VISIBLE_DEVICES','0').split(',')))")
echo "    nproc_per_node=$NGPU"

torchrun --nnodes=1 --nproc_per_node=$NGPU \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$SFT_DIR/train_sliced.parquet \
    data.val_files=$SFT_DIR/val_sliced.parquet \
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
    optim.warmup_steps_ratio=0.1 \
    ulysses_sequence_parallel_size=1 \
    use_remove_padding=true \
    trainer.default_local_dir=$SAVE_DIR \
    trainer.project_name=CORA_RL \
    trainer.experiment_name=sft_qwen3_4b_tools_${TAG} \
    trainer.total_epochs=3 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.max_ckpt_to_keep=3 \
    trainer.checkpoint.save_contents='[model,optimizer,extra,hf_model]' \
    trainer.logger=${SFT_LOGGER:-[console]} \
    "$@"

echo
echo "=== SFT done → $SAVE_DIR ==="
ls -lah "$SAVE_DIR"/global_step_*/hf_model/ 2>/dev/null | tail -20
