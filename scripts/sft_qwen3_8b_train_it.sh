#!/usr/bin/env bash
# Qwen3-8B FULL fine-tune ("train it") on the DGX Spark (GB10), single node.
#
#   EXP_NAME=sft_qwen3_8b_run1 bash scripts/sft_qwen3_8b_train_it.sh
#   EXP_NAME=smoke bash scripts/sft_qwen3_8b_train_it.sh --max_train_steps 20
#
# Extra arguments are forwarded to finetune.py, so any flag can be overridden
# from the command line without editing this file.
#
# ===========================================================================
# MEMORY GUARDRAIL — START THIS FIRST, IN A SECOND SHELL
# ===========================================================================
# The GB10 shares one LPDDR5X pool between CPU and GPU. Nothing in the OS can
# cap a CUDA allocation: cgroups/systemd MemoryMax, MIG and runai all fail to
# contain it, because driver allocations are not charged to the cgroup. If a
# run overruns, the box swaps itself unreachable for hours instead of dying.
# The only working guardrail is an external poller that kills the process.
#
#   LIMIT_GB=110; while sleep 1; do \
#     used=$(awk '/MemTotal/{t=$2}/MemAvailable/{a=$2}END{print int((t-a)/1048576)}' /proc/meminfo); \
#     if [ "$used" -gt "$LIMIT_GB" ]; then echo "$(date -Is) KILL at ${used}GB"; \
#     pkill -9 -f open_instruct/finetune.py; break; fi; done
#
# 110 GB is deliberate, not a round number. Measured peaks for this config:
#   ~72 GB during training  (base 49.5 GB + 2.46 MB/token is cut to ~0.32
#                            MB/token by liger, so 32k sequences fit)
#   ~84 GB while SAVING     (the checkpoint write gathers the model on top of
#                            the live training state)
# A ceiling below ~90 GB will fire during the final checkpoint save and destroy
# the run at the very end. See docs/spark-memory-guardrail.md.
# ===========================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMAX_DIR="${TMAX_DIR:-$HOME/tmax}"
OI_DIR="$TMAX_DIR/training/open-instruct"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
EXP_NAME="${EXP_NAME:-sft_qwen3_8b_${STAMP}}"
RUN_DIR="${RUN_DIR:-$REPO_ROOT/runs/${EXP_NAME}_${STAMP}}"
RUN_LOG="$RUN_DIR/train.log"

DATASET="${DATASET:-vittorino/tmax-sft-cleaned}"
DATASET_CONFIG="${DATASET_CONFIG:-skill_tax_20260505_2.2k_combined_balanced_thinking_all}"

# --- Guards ---------------------------------------------------------------
# Training never runs on a laptop (AGENTS.md section 6).
if [[ "$(uname -s)" != "Linux" ]] || ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "error: not the Spark (need Linux + nvidia-smi)." >&2
    exit 1
fi
if [[ ! -d "$OI_DIR" ]]; then
    echo "error: open-instruct not found at $OI_DIR" >&2
    echo "       set TMAX_DIR to your clone of https://github.com/hamishivi/tmax" >&2
    exit 1
fi

# finetune.py must carry the Spark patch. Without it --use_8bit_optimizer is
# reachable only under --use_qlora, so it is SILENTLY IGNORED here and the run
# allocates fp32 AdamW state (~66 GB instead of ~17 GB) and blows the pool.
if ! grep -q "args.use_qlora or args.use_8bit_optimizer" "$OI_DIR/open_instruct/finetune.py"; then
    echo "error: $OI_DIR/open_instruct/finetune.py is not patched." >&2
    echo "       git -C \"$TMAX_DIR\" apply $REPO_ROOT/patches/finetune-spark.patch" >&2
    exit 1
fi

# Both are in open-instruct's lockfile but absent from a --no-sync venv.
for mod in bitsandbytes liger_kernel; do
    if ! "$OI_DIR/.venv/bin/python" -c "import $mod" >/dev/null 2>&1; then
        echo "error: '$mod' missing from $OI_DIR/.venv" >&2
        echo "       uv pip install --python $OI_DIR/.venv/bin/python --no-deps <pinned version>" >&2
        exit 1
    fi
done

# --- Provenance (AGENTS.md section 5: an untracked run didn't happen) ------
mkdir -p "$RUN_DIR"
{
    echo "exp_name:   $EXP_NAME"
    echo "started:    $(date -Is)"
    echo "dataset:    $DATASET  [$DATASET_CONFIG]"
    echo "this repo:  $(git -C "$REPO_ROOT" rev-parse HEAD)"
    echo "tmax:       $(git -C "$TMAX_DIR" rev-parse HEAD) (+ patches/finetune-spark.patch)"
    echo "extra args: $*"
} > "$RUN_DIR/provenance.txt"

echo "run dir: $RUN_DIR"
echo "REMINDER: is the 110 GB kill loop running in another shell?"

cd "$OI_DIR"

# No deepspeed: single node makes ZeRO-3 pointless, --use_8bit_optimizer is
# documented incompatible with it, and its CPU offload moves nothing on a
# unified pool while letting the offloaded state spill to swap.
# No --use_flash_attn: no aarch64 wheel; SDPA is detected automatically.
uv run --no-sync accelerate launch \
    --mixed_precision bf16 \
    --num_processes 1 \
    open_instruct/finetune.py \
    --exp_name "$EXP_NAME" \
    --model_name_or_path Qwen/Qwen3-8B \
    --tokenizer_name Qwen/Qwen3-8B \
    --max_seq_length 32768 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0.0 \
    --num_train_epochs 2 \
    --dataset_mixer_list "$DATASET" 1.0 \
    --dataset_mixer_list_config_names "$DATASET_CONFIG" \
    --add_bos \
    --gradient_checkpointing \
    --use_8bit_optimizer \
    --use_liger_kernel \
    --report_to wandb \
    --with_tracking \
    --logging_steps 1 \
    --seed 42 \
    "$@" 2>&1 | tee "$RUN_LOG"
