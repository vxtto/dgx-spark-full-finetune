# Memory guardrail — DGX Spark (GB10)

Every training run on the Spark needs an external memory guardrail. This is not
defensive habit; without one a memory overrun makes the machine **unreachable for
hours** rather than failing fast. This happened twice while this configuration was
being measured.

## 1. Why the usual mechanisms do not work

The GB10 shares a single LPDDR5X pool between CPU and GPU. There is no separate
VRAM, and **nothing in the OS can cap a CUDA allocation**:

| Mechanism | Result |
|---|---|
| cgroups v2 / `systemd-run -p MemoryMax=` | Does not contain it. Driver allocations are not charged to the cgroup, so the limit bounds host RSS only. |
| MIG | Not available on GB10. |
| runai | Not applicable. |
| `torch.cuda.set_per_process_memory_fraction` | Works, but in-process only: it governs the PyTorch caching allocator and nothing else, and needs a code change. |

Because the limit cannot be enforced *inside* the machine, it has to be enforced
from outside the process: poll memory, kill on breach.

## 2. The guardrail

Run this in a second shell **before** starting a run, and leave it running:

```bash
LIMIT_GB=110; while sleep 1; do \
  used=$(awk '/MemTotal/{t=$2}/MemAvailable/{a=$2}END{print int((t-a)/1048576)}' /proc/meminfo); \
  if [ "$used" -gt "$LIMIT_GB" ]; then echo "$(date -Is) KILL at ${used}GB"; \
  pkill -9 -f open_instruct/finetune.py; break; fi; done
```

`MemTotal - MemAvailable` is the right metric, and this is measured, not assumed:
allocating 20 GB on the GPU moved `MemAvailable` from 121.9 GB to 101.8 GB — 1:1.
`MemAvailable` already discounts reclaimable page cache, so the loop does not fire
on `buff/cache`.

It kills only `finetune.py`, needs no root, and leaves the machine responsive.

## 3. Why 110 GB

Measured peaks for the full fine-tune (`scripts/sft_qwen3_8b_train_it.sh`):

| Phase | Peak |
|---|---|
| Training at `max_seq_length 32768` | **~72 GB** |
| Writing the checkpoint | **~84 GB** |

The save is the higher of the two: it gathers the model on top of the live training
state. A ceiling below ~90 GB will fire **during the final checkpoint write** and
destroy a multi-day run at the last moment. 110 GB clears both peaks and still leaves
~12 GB for the OS and an SSH session.

## 4. The memory model behind those numbers

Base cost, independent of sequence length (measured):

| Component | Size |
|---|---|
| bf16 weights | 15.26 GiB |
| bf16 gradients | 15.26 GiB |
| bnb 8-bit AdamW state | ~15.6 GiB |
| **Base** | **~46 GiB (49.5 GB)** |

Activation cost is **linear** in sequence length — log-log slope 1.00 across
n = 4096 / 8192 / 12288 / 16384, measured with `max_memory_allocated`:

| Config | Activation cost | Peak at 32,768 tokens |
|---|---|---|
| Without liger | 2.46 MB/token | **~130 GB — exceeds the 121.7 GiB pool** |
| With `--use_liger_kernel` | 4.63 GiB + **0.315 MB/token** | **~65 GB** |

Liger's `fused_linear_cross_entropy` never materialises the full logit tensor. With
Qwen3's 151,936-token vocabulary that logits chain is over half the activation cost,
so removing it cuts the marginal rate **7.8×** and is what makes the paper's 32k
sequence length fit at all. It is also ~13% faster, and numerically sound: step-1 loss
1.0517414808273315 with liger against 1.0524177551269531 without — 0.065%, pure bf16
reduction-order noise.

## 5. Two failure modes worth knowing

**`--use_8bit_optimizer` is silently ignored without the patch.** In upstream
`finetune.py` the bnb optimizer is reachable only under `--use_qlora`. A full
fine-tune passing the flag alone gets `torch.optim.AdamW` at 8 bytes/param —
**65.6 GB of optimizer state instead of ~17 GB**. `patches/finetune-spark.patch`
fixes this and `scripts/sft_qwen3_8b_train_it.sh` refuses to start without it.

**DeepSpeed ZeRO-3 CPU offload is harmful here, not neutral.** On a unified pool
"offloading to CPU" moves nothing physically, and with `pin_memory: false` the
offloaded state becomes swappable — which is precisely the mechanism that turns an
overrun into hours of thrashing. Do not use it on a single node.

## 6. If the box becomes unreachable

Symptom: `ssh` fails at *banner exchange* rather than refusing the connection. That
is swap thrash, not a crash — the machine is alive and the kernel OOM killer will
usually reclaim it, in one observed case after about 30 minutes, with no reboot and no
data loss. Retry in a loop and kill the offending process as soon as a session lands;
power-cycle if you need it back sooner.

Two rules learned the hard way:

- **Arm the guardrail before anything memory-heavy, and leave it armed** — including
  one-off probes and experiments, not just training runs. The unreachable incident was
  caused by a diagnostic script run after the kill loop had been stopped.
- Swap makes this worse. 16 GB of swap at `swappiness=60` is what converts a fast OOM
  kill into a multi-hour hang. `swapoff -a` during runs (needs root) would make
  overruns recoverable in seconds, and is worth considering.
