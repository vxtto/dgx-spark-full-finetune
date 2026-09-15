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
| Memory | 128 GB **unified** | no 4-bit quantization needed for an 8B LoRA run |
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

## 5. Still unverified on the box

Neither has been confirmed on aarch64 + CUDA 13; both are required by open-instruct.

- [ ] `deepspeed>=0.18.3` — JIT-compiles CUDA ops on first use. **Most likely failure
      point of the whole setup.** If it needs a source build, record the exact command
      in this file.
- [ ] `vllm>=0.19.1` — needed for evaluation/generation, not for the SFT run itself.
      Can be deferred.

## 6. Running long jobs

A dropped SSH session must not kill a 30+ hour run (§6):

```bash
tmux new -s sft
bash scripts/run_sft.sh configs/sft_qwen3_8b_lora_spark.yaml
# detach: Ctrl-b d ; reattach: tmux attach -t sft
```

`run_sft.sh` tees to `runs/<name>/train.log` regardless.
