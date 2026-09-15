#!/usr/bin/env bash
# Launch a LoRA SFT run on the DGX Spark.
#
#   bash scripts/run_sft.sh configs/smoke_qwen3_1_7b_lora.yaml
#   bash scripts/run_sft.sh configs/sft_qwen3_8b_lora_spark.yaml
#
# Snapshots the config, the git commit and the environment into the run
# directory before starting, so the run satisfies the reproducibility rule in
# AGENTS.md §5: "Every run is reproducible from one config file plus one data
# snapshot, both logged in the run directory."
#
# Requires TMAX_DIR to point at a clone of https://github.com/hamishivi/tmax.
# Long runs belong in tmux — see docs/environment.md §6.

set -euo pipefail

CONFIG="${1:-}"
if [[ -z "$CONFIG" ]]; then
    echo "usage: bash scripts/run_sft.sh <config.yaml>" >&2
    exit 2
fi
if [[ ! -f "$CONFIG" ]]; then
    echo "error: config not found: $CONFIG" >&2
    exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMAX_DIR="${TMAX_DIR:-$HOME/tmax}"

# --- Guard: never train on a laptop (AGENTS.md §6) ------------------------
if [[ "$(uname -s)" != "Linux" ]]; then
    echo "error: this is $(uname -s), not the Spark." >&2
    echo "       Training never runs on a laptop (AGENTS.md §6)." >&2
    exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "error: no nvidia-smi — this machine has no GPU." >&2
    exit 1
fi
if [[ ! -d "$TMAX_DIR" ]]; then
    echo "error: TMAX_DIR=$TMAX_DIR does not exist." >&2
    echo "       git clone https://github.com/hamishivi/tmax.git \"$TMAX_DIR\" && cd \"\$_\" && uv sync" >&2
    exit 1
fi

# --- Run directory --------------------------------------------------------
RUN_NAME="$(python3 -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['exp_name'])" "$CONFIG")"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="$REPO_ROOT/runs/${RUN_NAME}_${STAMP}"
mkdir -p "$RUN_DIR"

# Copy the config into the run dir, resolving any relative dataset path to an
# absolute one: we cd into $TMAX_DIR below, so a relative path would otherwise
# resolve against the tmax checkout instead of this repo.
python3 - "$CONFIG" "$RUN_DIR/config.yaml" "$REPO_ROOT" <<'PYEOF'
import os, sys, yaml
src, dst, root = sys.argv[1:4]
cfg = yaml.safe_load(open(src))
mixer = cfg.get("dataset_mixer_list") or []
missing = []
for i in range(0, len(mixer), 2):
    name = str(mixer[i])
    if name.endswith((".parquet", ".jsonl")) and not os.path.isabs(name):
        resolved = os.path.join(root, name)
        mixer[i] = resolved
        if not os.path.exists(resolved):
            missing.append(resolved)
if missing:
    sys.stderr.write("error: dataset snapshot not found:\n")
    for m in missing:
        sys.stderr.write(f"         {m}\n")
    sys.stderr.write("       create it first:  python scripts/prepare_dataset.py --variant all\n")
    sys.exit(1)
yaml.safe_dump(cfg, open(dst, "w"), sort_keys=False, default_flow_style=False)
print(f"resolved dataset -> {mixer[0] if mixer else '(none)'}")
PYEOF

{
    echo "run_name:      $RUN_NAME"
    echo "started_utc:   $STAMP"
    echo "config:        $CONFIG"
    echo "host:          $(hostname)"
    echo "arch:          $(uname -m)"
    echo "repo_commit:   $(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "repo_dirty:    $(git -C "$REPO_ROOT" status --porcelain 2>/dev/null | wc -l | tr -d ' ') file(s)"
    echo "tmax_commit:   $(git -C "$TMAX_DIR" rev-parse HEAD 2>/dev/null || echo unknown)"
    echo
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || true
} > "$RUN_DIR/provenance.txt"

echo "=== run directory: $RUN_DIR"
cat "$RUN_DIR/provenance.txt"
echo

if [[ -z "${TMUX:-}" ]]; then
    echo "WARNING: not inside tmux. A dropped SSH session will kill this run."
    echo "         Ctrl-C now and start again with: tmux new -s sft"
    echo "         Continuing in 10s..."
    sleep 10
fi

# --- Launch ---------------------------------------------------------------
# open_instruct/finetune.py takes the YAML as its first positional argument;
# anything after it overrides a key (ArgumentParserPlus.parse_yaml_and_args).
cd "$TMAX_DIR"
exec 2>&1
uv run accelerate launch \
    --num_processes 1 \
    --mixed_precision bf16 \
    open_instruct/finetune.py \
    "$RUN_DIR/config.yaml" \
    "${@:2}" \
    2>&1 | tee "$RUN_DIR/train.log"
