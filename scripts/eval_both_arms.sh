#!/usr/bin/env bash
# Evaluate both baseline (official Qwen3-8B) and fine-tuned model on Terminal-Bench 2.0
# with Vanillux2Agent (same harness as training).
#
# RUN ON THE SPARK, in a tmux session. This script:
#   1. Serves the baseline model (Qwen3-8B)
#   2. Runs eval for K attempts
#   3. Stops the server, downloads fine-tuned weights
#   4. Serves the fine-tuned model
#   5. Runs eval for K attempts
#
# Usage (on the Spark, inside tmux):
#   K=5 bash scripts/eval_both_arms.sh                    # default: N_CONCURRENT=32, ~38h total
#   K=5 N_CONCURRENT=16 bash scripts/eval_both_arms.sh    # slower (~70h total) but lower latency
#   K=1 bash scripts/eval_both_arms.sh                    # quick test (~7.6h total)
#
# Default N_CONCURRENT=32 gives 423.4 tok/s aggregated (vs 229.4 with N=16).
# Both evals run against the same endpoint (http://localhost:8888/v1),
# so they must be sequential. Cost scales with K and N_CONCURRENT.

set -euo pipefail

# --- Configuration -------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMAX_DIR="${TMAX_DIR:-$HOME/tmax}"
K="${K:-5}"
N_CONCURRENT="${N_CONCURRENT:-32}"
TIMEOUT_MULT="${TIMEOUT_MULT:-6}"
ENDPOINT="${ENDPOINT:-http://localhost:8888/v1}"
PORT="${PORT:-8888}"

# Fine-tuned model from HuggingFace
FINETUNED_MODEL="vittorino/open_instruct_dev"
FINETUNED_REVISION="sft_qwen3_8b_run1"

# --- Guards ---------------------------------------------------------------
if [[ "$(uname -s)" != "Linux" ]] || ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "error: not the Spark (need Linux + nvidia-smi)." >&2; exit 1
fi

command -v vllm >/dev/null || { echo "error: vllm not on PATH." >&2; exit 1; }
[[ -d "$TMAX_DIR/Vanillux2Agent" ]] || {
    echo "error: Vanillux2Agent not at $TMAX_DIR/Vanillux2Agent" >&2; exit 1; }

echo "=== Evaluate both arms with Vanillux2Agent (same harness as training) ==="
echo "Baseline:    Qwen/Qwen3-8B (official)"
echo "Fine-tuned:  $FINETUNED_MODEL@$FINETUNED_REVISION"
echo "K (attempts per task): $K"
echo "Each arm takes ~7h * K. Be patient; run under tmux."
echo

# --- Helper: wait for endpoint to be ready ------
wait_for_endpoint() {
    local endpoint="$1"
    local max_attempts=60
    local attempt=0
    echo "Waiting for endpoint $endpoint to be ready..."
    while [ $attempt -lt $max_attempts ]; do
        if curl -sf -m 5 "$endpoint/models" >/dev/null 2>&1; then
            echo "✓ Endpoint ready"
            return 0
        fi
        attempt=$((attempt + 1))
        sleep 2
    done
    echo "error: endpoint did not become ready after $((max_attempts * 2))s" >&2
    return 1
}

# --- Helper: kill any running vLLM on the port ------
kill_old_server() {
    local port="$1"
    if lsof -Pi :"$port" -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo "Killing existing process on port $port..."
        lsof -ti:"$port" | xargs kill -9 || true
        sleep 2
    fi
}

# --- Arm 1: Baseline (official Qwen3-8B) ------
echo
echo "======================================================================="
echo "ARM 1: Baseline (official Qwen/Qwen3-8B)"
echo "======================================================================="

kill_old_server "$PORT"

echo "Serving Qwen/Qwen3-8B..."
vllm serve Qwen/Qwen3-8B \
    --served-model-name qwen3-8b \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.5 \
    --port "$PORT" \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml \
    2>&1 | tee "$HOME/tb2-eval/baseline_server.log" &

VLLM_PID=$!
sleep 5
wait_for_endpoint "$ENDPOINT" || { kill $VLLM_PID 2>/dev/null || true; exit 1; }

echo "Running eval for baseline (K=$K, N_CONCURRENT=$N_CONCURRENT, TIMEOUT_MULT=$TIMEOUT_MULT)..."
MODEL_LABEL=base ARM_MODEL=qwen3-8b K="$K" N_CONCURRENT="$N_CONCURRENT" TIMEOUT_MULT="$TIMEOUT_MULT" bash "$REPO_ROOT/scripts/eval_tb2_vanillux.sh"

echo "Killing baseline server..."
kill $VLLM_PID 2>/dev/null || true
sleep 5

# --- Arm 2: Fine-tuned (new weights) ------
echo
echo "======================================================================="
echo "ARM 2: Fine-tuned ($FINETUNED_MODEL@$FINETUNED_REVISION)"
echo "======================================================================="

echo "Downloading fine-tuned model from HuggingFace..."
mkdir -p "$HOME/.cache/huggingface/hub"
FINETUNED_PATH=$(python3 - "$FINETUNED_MODEL" "$FINETUNED_REVISION" <<'PYTHON'
import sys
from huggingface_hub import snapshot_download
model_id, revision = sys.argv[1], sys.argv[2]
path = snapshot_download(repo_id=model_id, revision=revision, repo_type="model")
print(path)
PYTHON
)

echo "Fine-tuned model cached at: $FINETUNED_PATH"

kill_old_server "$PORT"

echo "Serving fine-tuned model..."
vllm serve "$FINETUNED_PATH" \
    --served-model-name qwen3-8b-sft \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.5 \
    --port "$PORT" \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml \
    2>&1 | tee "$HOME/tb2-eval/finetuned_server.log" &

VLLM_PID=$!
sleep 5
wait_for_endpoint "$ENDPOINT" || { kill $VLLM_PID 2>/dev/null || true; exit 1; }

echo "Running eval for fine-tuned (K=$K, N_CONCURRENT=$N_CONCURRENT, TIMEOUT_MULT=$TIMEOUT_MULT)..."
MODEL_LABEL=sft ARM_MODEL=qwen3-8b-sft K="$K" N_CONCURRENT="$N_CONCURRENT" TIMEOUT_MULT="$TIMEOUT_MULT" bash "$REPO_ROOT/scripts/eval_tb2_vanillux.sh"

echo "Killing fine-tuned server..."
kill $VLLM_PID 2>/dev/null || true

echo
echo "======================================================================="
echo "✓ Both evals complete!"
echo "Results are in: $HOME/tb2-eval/jobs/"
echo "======================================================================="
