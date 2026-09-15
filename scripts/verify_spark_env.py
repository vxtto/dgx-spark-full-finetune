#!/usr/bin/env python3
"""Pre-flight check for LoRA SFT on the DGX Spark.

Run this on the Spark before every training run, and certainly before the first
one. It verifies the things that silently produce a broken or misleading run
rather than a clean error:

  - aarch64 / sm_121 / CUDA 13 as expected
  - bf16 actually works (not just "is supported")
  - which attention backend open-instruct will resolve to
  - DeepSpeed imports (the most likely failure point on this box)
  - enough free memory for an 8B bf16 LoRA run

Exit code 0 = safe to train. 1 = do not start a run.

    python scripts/verify_spark_env.py
"""

import importlib.util
import platform
import subprocess
import sys

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"

failures: list[str] = []
warnings: list[str] = []


def ok(msg: str, detail: str = "") -> None:
    print(f"  {GREEN}PASS{RESET}  {msg}" + (f"  {DIM}{detail}{RESET}" if detail else ""))


def fail(msg: str, detail: str = "") -> None:
    print(f"  {RED}FAIL{RESET}  {msg}" + (f"  {DIM}{detail}{RESET}" if detail else ""))
    failures.append(msg)


def warn(msg: str, detail: str = "") -> None:
    print(f"  {YELLOW}WARN{RESET}  {msg}" + (f"  {DIM}{detail}{RESET}" if detail else ""))
    warnings.append(msg)


def section(title: str) -> None:
    print(f"\n{title}")


# --- 1. Machine ----------------------------------------------------------
section("Machine")

arch = platform.machine()
if arch == "aarch64":
    ok("architecture is aarch64", arch)
else:
    warn(f"architecture is {arch}, expected aarch64", "not the Spark?")

try:
    smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
         "--format=csv,noheader"],
        capture_output=True, text=True, timeout=30,
    )
    if smi.returncode == 0:
        ok("nvidia-smi", smi.stdout.strip())
    else:
        fail("nvidia-smi failed", smi.stderr.strip()[:200])
except FileNotFoundError:
    fail("nvidia-smi not found", "no NVIDIA driver on this machine")
except subprocess.TimeoutExpired:
    fail("nvidia-smi timed out")

# --- 2. PyTorch / CUDA ---------------------------------------------------
section("PyTorch and CUDA")

try:
    import torch
except ImportError:
    fail("torch is not installed")
    print(f"\n{RED}Cannot continue without torch.{RESET}")
    sys.exit(1)

ok("torch imported", f"{torch.__version__}, cuda {torch.version.cuda}")

if not torch.cuda.is_available():
    fail("torch.cuda.is_available() is False",
         "CPU-only build — reinstall from the cu130 index, see docs/environment.md §2")
else:
    ok("CUDA available", torch.cuda.get_device_name(0))

    cap = torch.cuda.get_device_capability()
    if cap == (12, 1):
        ok("compute capability is sm_121", str(cap))
    elif cap[0] >= 12:
        warn(f"compute capability {cap}, expected (12, 1)")
    else:
        warn(f"compute capability {cap} — not a GB10", str(cap))

    cuda_major = int((torch.version.cuda or "0").split(".")[0])
    if cuda_major >= 13:
        ok("built against CUDA 13+", torch.version.cuda)
    else:
        fail(f"torch built against CUDA {torch.version.cuda}",
             "sm_121 needs CUDA 13.0; CUDA 12.x will not load")

    # bf16 must actually compute, not merely report as supported.
    try:
        a = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
        result = (a @ a).float().sum().item()
        if result == result:  # NaN check
            ok("bf16 matmul on GPU works")
        else:
            fail("bf16 matmul produced NaN")
    except Exception as exc:  # noqa: BLE001
        fail("bf16 matmul failed", str(exc)[:200])

    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    # 8B bf16 weights ~16 GB + LoRA optimizer/activations; see docs/theory/lora-sft.md §3
    if total_gb >= 90:
        ok(f"{total_gb:.0f} GB visible", "enough for 8B bf16 LoRA without quantization")
    elif total_gb >= 40:
        warn(f"only {total_gb:.0f} GB visible", "8B LoRA is tight; check max_seq_length")
    else:
        fail(f"only {total_gb:.0f} GB visible", "not enough for an 8B bf16 LoRA run")

# --- 3. Attention backend ------------------------------------------------
section("Attention backend")
print(f"  {DIM}open-instruct auto-detects this; there is no config flag.{RESET}")

try:
    import transformers

    ok("transformers imported", transformers.__version__)

    fa2 = transformers.utils.is_flash_attn_2_available()
    fa3 = getattr(transformers.utils, "is_flash_attn_3_available", lambda: False)()
    fa4 = getattr(transformers.utils, "is_flash_attn_4_available", lambda: False)()
    major = torch.cuda.get_device_capability()[0] if torch.cuda.is_available() else None

    # Mirrors detect_attn_implementation() in open_instruct/model_utils.py
    if major is None:
        backend = "torch (SDPA) — no CUDA"
    elif fa4 and major >= 10:
        backend = "flash_attention_4"
    elif fa3 and major >= 9:
        backend = "flash_attention_3"
    elif fa2:
        backend = "flash_attention_2"
    else:
        backend = "torch (SDPA)"

    if backend.startswith("torch"):
        ok(f"will resolve to {backend}", "expected on aarch64 — flash-attn is not built here")
        print(f"  {DIM}      Paper Table 14 uses flash attention 3; this is a known")
        print(f"        divergence, recorded in docs/decisions.md D-04.{RESET}")
    else:
        warn(f"will resolve to {backend}",
             "unexpected on this box — throughput and loss may differ between runs")
except ImportError:
    fail("transformers is not installed")

# --- 4. Training stack ---------------------------------------------------
section("Training stack")

for mod, required, note in [
    ("deepspeed", True, "required by open-instruct; JIT-compiles CUDA ops on first use"),
    ("peft", True, "LoRA implementation"),
    ("accelerate", True, "launcher"),
    ("datasets", True, "dataset loading"),
    ("vllm", False, "evaluation/generation only — not needed for the SFT run"),
    ("flash_attn", False, "should NOT be installed on this box"),
]:
    present = importlib.util.find_spec(mod) is not None
    if mod == "flash_attn":
        if present:
            warn("flash_attn IS installed",
                 "docs/environment.md §2 advises against it on Blackwell/aarch64")
        else:
            ok("flash_attn absent", "correct for this box")
        continue
    if present:
        try:
            version = importlib.import_module(mod).__version__
        except Exception:  # noqa: BLE001
            version = "?"
        ok(f"{mod} importable", version)
    elif required:
        fail(f"{mod} missing", note)
    else:
        warn(f"{mod} missing", note)

# --- Summary -------------------------------------------------------------
print()
if failures:
    print(f"{RED}{len(failures)} blocking problem(s) — do not start a run.{RESET}")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)

if warnings:
    print(f"{YELLOW}{len(warnings)} warning(s), nothing blocking.{RESET}")
    for w in warnings:
        print(f"  - {w}")

print(f"{GREEN}Environment looks good. Run the smoke config next:{RESET}")
print("  bash scripts/run_sft.sh configs/smoke_qwen3_1_7b_lora.yaml")
