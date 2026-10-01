#!/usr/bin/env bash
# Qwen3-8B FULL fine-tune ("train it") on the DGX Spark (GB10), single node.
#
#   EXP_NAME=sft_qwen3_8b_run1 bash scripts/sft_qwen3_8b_train_it.sh
#   EXP_NAME=smoke bash scripts/sft_qwen3_8b_train_it.sh --max_train_steps 20
#
# Extra arguments are forwarded to finetune.py, so any flag can be overridden
# from the command line without editing this file.
#
# SPARK_OPT_CONFIG=configs/spark_opt_qwen3_8b.yaml enables spark_opt.py: FP8 +
# torch.compile, fixed Liger FLCE accumulation, pinned FlashAttention-2 and
# adaptive gradient checkpointing. Unset, the run is identical to run1. Check a
# config first with scripts/bench_spark_opt.py (~1 min timed, ~5 min wall).
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
# Training only runs on the Spark.
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
if ! grep -q "args.use_qlora or args.use_8bit_optimizer" "$OI_DIR/open_instruct/finetune.py" \
   || ! grep -q "except ModuleNotFoundError" "$OI_DIR/open_instruct/model_utils.py"; then
    echo "error: the tmax checkout at $TMAX_DIR is not patched." >&2
    echo "       finetune.py   : --use_8bit_optimizer is reachable only under --use_qlora," >&2
    echo "                       so it is silently ignored and fp32 AdamW state is used." >&2
    echo "       model_utils.py: _is_flash_attn_4_available() raises instead of returning" >&2
    echo "                       False when flash_attn is absent, so attention detection dies." >&2
    echo "       git -C \"$TMAX_DIR\" apply $REPO_ROOT/patches/finetune-spark.patch" >&2
    exit 1
fi

# Check the environment `uv run` will actually use. This is NOT necessarily
# $OI_DIR/.venv -- uv may resolve the project to a cache environment under
# ~/.cache/uv/environments-v2/, and checking .venv gives a false pass.
UV_ENV="$(cd "$OI_DIR" && uv run --no-sync python -c "import sys; print(sys.prefix)" 2>/dev/null | tail -1)"
echo "uv environment: ${UV_ENV:-<unresolved>}"
for mod in bitsandbytes liger_kernel olmo_core; do
    if ! (cd "$OI_DIR" && uv run --no-sync python -c "import $mod") >/dev/null 2>&1; then
        echo "error: '$mod' missing from $UV_ENV" >&2
        echo "       uv pip install --python \"$UV_ENV/bin/python\" <pinned version from uv.lock>" >&2
        exit 1
    fi
done

# flash-attn-4 is declared in open-instruct's pyproject.toml gated only on
# Darwin, and ships as a py3-none-any wheel, so `uv pip install -e .` WILL
# install it on aarch64. detect_attn_implementation() then selects flash_4
# (GB10 reports major 12, and the branch tests >= 10), silently replacing the
# SDPA backend every measurement in docs/results.md was made on.
if (cd "$OI_DIR" && uv run --no-sync python -c "import flash_attn") >/dev/null 2>&1; then
    echo "error: flash_attn is installed; attention would silently switch to flash_4." >&2
    echo "       uv pip uninstall --python \"$UV_ENV/bin/python\" flash-attn-4 flash-attn" >&2
    exit 1
fi

# --- Spark throughput optimizations (opt-in) -------------------------------
SPARK_OPT_CONFIG="${SPARK_OPT_CONFIG:-}"
if [[ -n "$SPARK_OPT_CONFIG" ]]; then
    [[ "$SPARK_OPT_CONFIG" = /* ]] || SPARK_OPT_CONFIG="$REPO_ROOT/$SPARK_OPT_CONFIG"
    if [[ ! -f "$SPARK_OPT_CONFIG" ]]; then
        echo "error: SPARK_OPT_CONFIG not found: $SPARK_OPT_CONFIG" >&2
        exit 1
    fi
    # Without the hook finetune.py would ignore the variable and train as run1.
    if ! grep -q "SPARK_OPT_CONFIG" "$OI_DIR/open_instruct/finetune.py"; then
        echo "error: finetune.py lacks the spark_opt hook; apply patches/finetune-spark.patch" >&2
        exit 1
    fi
    if ! (cd "$OI_DIR" && uv run --no-sync python -c "import torchao") >/dev/null 2>&1; then
        echo "error: 'torchao' missing from $UV_ENV (needed for FP8)" >&2
        echo "       uv pip install --python \"$UV_ENV/bin/python\" --no-deps torchao==0.18.0" >&2
        exit 1
    fi
    export SPARK_OPT_CONFIG
    export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
    # Every micro-batch has a different length, which fragments the default
    # allocator: at 32k tokens it reserved 79 GiB for 62 GiB live (86.8 GB system).
    # Expandable segments: 65 GiB reserved, 72.3 GB system, same speed.
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
fi

# --- Provenance: every run records what it was launched from ---------------
mkdir -p "$RUN_DIR"
{
    echo "exp_name:   $EXP_NAME"
    echo "started:    $(date -Is)"
    echo "dataset:    $DATASET  [$DATASET_CONFIG]"
    echo "this repo:  $(git -C "$REPO_ROOT" rev-parse HEAD)"
    echo "tmax:       $(git -C "$TMAX_DIR" rev-parse HEAD) (+ patches/finetune-spark.patch)"
    echo "extra args: $*"
    if [[ -n "$SPARK_OPT_CONFIG" ]]; then
        echo "spark_opt:  $SPARK_OPT_CONFIG (sha256 $(sha256sum "$SPARK_OPT_CONFIG" | cut -c1-16), copied as spark_opt.yaml)"
        echo "            spark_opt.py sha256 $(sha256sum "$REPO_ROOT/spark_opt.py" | cut -c1-16)"
        echo "            torchao $(cd "$OI_DIR" && uv run --no-sync python -c "import torchao; print(torchao.__version__)" 2>/dev/null | tail -1)"
        echo "            PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
    else
        echo "spark_opt:  off"
    fi
    # Uncommitted work is part of the run: diffs and module copies sit next to this file.
    echo "uncommitted: $(git -C "$REPO_ROOT" status --porcelain | wc -l) path(s) in this repo -> repo.diff, spark_opt.py copy"
} > "$RUN_DIR/provenance.txt"
[[ -z "$SPARK_OPT_CONFIG" ]] || cp "$SPARK_OPT_CONFIG" "$RUN_DIR/spark_opt.yaml"
git -C "$REPO_ROOT" diff HEAD > "$RUN_DIR/repo.diff"
git -C "$TMAX_DIR" diff HEAD > "$RUN_DIR/tmax.diff"
cp "$REPO_ROOT"/spark_opt.py "$RUN_DIR"/

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
