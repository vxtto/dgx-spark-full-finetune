# Environment — DGX Spark

Everything here runs **on the Spark**, never on a laptop (`AGENTS.md` §6). Commands
that build from source are recorded here as required by §6.

## 0. Hardware facts to confirm before trusting anything below

```bash
uname -m                 # expect: aarch64
nvidia-smi               # expect: GB10, CUDA 13.x
free -h                  # expect: ~128 GB unified
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_capability())"
```

Expected capability is `(12, 1)` — sm_121. PyTorch may warn:

```
Minimum and Maximum cuda capability supported by this version of PyTorch is (8.0) - (12.0)
```

**Safe to ignore.** sm_120 and sm_121 are binary compatible.

## 1. Why this hardware constrains the stack

| Property | Value | Consequence |
|---|---|---|
| Arch | aarch64 (Arm) | many CUDA wheels are x86-only |
| Compute capability | sm_121 (Blackwell) | needs CUDA 13.0 builds; CUDA 12.x will not load |
| Memory | 128 GB **unified** | no quantization needed; but nothing in the OS can cap a CUDA allocation — see `docs/spark-memory-guardrail.md` |
| Bandwidth | ~273 GB/s | the real bottleneck — roughly 10× below an H100 |

The last row is the one that sets expectations: this box has plenty of memory and
modest throughput. It is a machine for one long 8B run, not for fast iteration.

## 2. Known pitfalls (verified against community DGX Spark guides)

- **Do not install `flash-attn`.** Upstream fails with `libcudart.so.12` errors on this
  box, and it is reportedly slower than SDPA on Blackwell regardless. open-instruct
  already excludes it on aarch64 and falls back to SDPA on its own — see §4.
- **`pip install vllm` fails** the same way. It must come from the CUDA 13 nightly
  index: `--extra-index-url https://wheels.vllm.ai/nightly/cu130`.
- **PyTorch must be the cu130 build**: `--index-url https://download.pytorch.org/whl/cu130`.
- `TORCH_CUDA_ARCH_LIST=12.0` is needed at build time for anything compiling CUDA
  kernels from source.

## 3. Setup

```bash
git clone https://github.com/hamishivi/tmax.git
cd tmax
uv sync
```

`uv sync` resolves the CUDA 13.0 index that open-instruct's `pyproject.toml` already
declares. If it tries to build `flash-attn`, stop — the aarch64 marker is not being
applied, and that needs fixing before anything else.

Then, from this repo:

```bash
python scripts/verify_spark_env.py
```

It checks arch, capability, bf16, the resolved attention backend, and DeepSpeed. Do not
start a run until it passes.

## 4. Attention backend — why no patch is needed

`open_instruct/model_utils.py`:

```python
def detect_attn_implementation() -> str:
    if not torch.cuda.is_available():            result = torch
    elif is_flash_attn_4_available() and major >= 10:  result = flash_4
    elif is_flash_attn_3_available() and major >= 9:   result = flash_3
    elif is_flash_attn_2_available():                  result = flash_2
    else:                                              result = torch   # <- SDPA
```

With no flash-attn installed the last branch is taken and the model loads with SDPA.
There is no `--attn_implementation` flag; it is detected, not configured.

**This is a real divergence from the paper** (Table 14 specifies flash attention 3) and
is recorded as such in `docs/decisions.md` D-04.

> Note the second branch: GB10 reports compute major **12**, so `>= 10` is satisfied.
> If a `flash-attn-4` wheel ever does install on this box it will be selected silently.
> If throughput or loss changes unexpectedly between runs, check the
> `Auto-detected attention implementation:` line in the log first.

**This is not hypothetical — confirmed 2026-09-20.** `flash-attn` and `flash-attn-3`
are correctly gated in open-instruct's `pyproject.toml` (`platform_machine != 'aarch64'`
and x86_64-only respectively), but **`flash-attn-4` is gated only on Darwin** and ships
as a `py3-none-any` wheel, so it installs here. A plain `uv pip install -e .` in the
open-instruct checkout resolves 197 packages including `flash-attn-4==4.0.0b5` and
`vllm==0.26.0`, and would silently move attention from SDPA to `flash_4` — away from
the backend every number in `docs/experiments.md` was measured on. Install open-instruct's
dependencies selectively, not with `-e .`.
`scripts/sft_qwen3_8b_train_it.sh` refuses to start if `flash_attn` is importable.

### `_is_flash_attn_4_available()` needs a patch

With no `flash_attn` installed at all, `importlib.util.find_spec("flash_attn.cute")`
**raises `ModuleNotFoundError`** rather than returning `None` — `find_spec` on a dotted
path requires the parent package to be importable. Attention detection therefore dies on
a clean aarch64 box. `patches/finetune-spark.patch` wraps it in `try/except`.

## 5. Dependency status

- [x] `bitsandbytes==0.49.1` — **works on sm_121**, verified by running real
      `AdamW8bit` steps. Ships an aarch64 manylinux wheel; no source build needed.
- [x] `liger-kernel==0.8.0` — **works on sm_121**. Required for the 32k full
      fine-tune (D-08).
- [x] `deepspeed>=0.18.3` — installed, but **not used**. Single node makes ZeRO-3
      pointless and it is incompatible with `--use_8bit_optimizer`; its CPU offload is
      actively harmful on a unified pool (D-08). No longer a project risk.
- [ ] `vllm>=0.19.1` — needed for evaluation/generation, not for the SFT run itself.
      Can be deferred.

**Which environment?** `uv run` in the open-instruct project does **not** necessarily
use `$TMAX_DIR/training/open-instruct/.venv` — it may resolve to a cache environment
under `~/.cache/uv/environments-v2/`. Checking `.venv` gives a false pass. Get the real
one with:

```bash
cd "$TMAX_DIR/training/open-instruct" && uv run --no-sync python -c "import sys; print(sys.prefix)"
```

Beyond torch and the five obvious packages, `finetune.py`'s import chain also needs
these, all pinned from `uv.lock` (verified on a clean clone, 2026-09-20):

`ai2-olmo-core==2.4.0` (git rev `002958a8f15afe5c729affdb12f4c4b12c2f26ad`),
`litellm==1.75.0`, `nltk==3.9.2`, `langdetect==1.0.9`, `immutabledict==1.2.0`,
`antlr4-python3-runtime==4.11.0`, `hf-transfer==0.1.9`, `backoff==2.2.1`,
`absl-py==2.3.1`, `beaker-py==2.5.7`, `ray==2.53.0`.

Both wheels are in open-instruct's `uv.lock` but are absent from a venv created with
`--no-sync`. Install them without disturbing the resolved environment:

```bash
uv pip install --python "$TMAX_DIR/training/open-instruct/.venv/bin/python" \
    --no-deps bitsandbytes==0.49.1 liger-kernel==0.8.0
```

`scripts/sft_qwen3_8b_train_it.sh` refuses to start if either is missing.

## 6. Running long jobs

A dropped SSH session must not kill a multi-day run (§6). **Start the memory
guardrail first** — see `docs/spark-memory-guardrail.md`; without it a memory overrun
makes the box unreachable for hours instead of failing fast.

```bash
# shell 1 — guardrail, leave running
LIMIT_GB=110; while sleep 1; do \
  used=$(awk '/MemTotal/{t=$2}/MemAvailable/{a=$2}END{print int((t-a)/1048576)}' /proc/meminfo); \
  if [ "$used" -gt "$LIMIT_GB" ]; then echo "$(date -Is) KILL at ${used}GB"; \
  pkill -9 -f open_instruct/finetune.py; break; fi; done

# shell 2 — the run
tmux new -s sft
EXP_NAME=sft_qwen3_8b_run1 bash scripts/sft_qwen3_8b_train_it.sh
# detach: Ctrl-b d ; reattach: tmux attach -t sft
```

Both scripts tee to `runs/<exp_name>_<stamp>/train.log` regardless. `systemd-run --user
--unit=<name> --collect` is an alternative to tmux that survives logout.
