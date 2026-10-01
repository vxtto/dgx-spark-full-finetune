"""~1-minute throughput check of spark_opt.py on the full Qwen3-8B training step.

Loads the model the way patched finetune.py does under SPARK_OPT_CONFIG (Liger
with only FLCE, gradient checkpointing, spark_opt.apply, bnb 8-bit AdamW, bf16
autocast as accelerate applies it) and reports, per optimization:

  FLCE       patched-vs-original loss/grad error, chunks per call
  attention  which attention kernels actually ran (profiled micro-batch)
  FP8        Float8Linear count, GPU time share per kernel class
  compile    graphs, recompiles in the timed window, graph breaks
  ckpt       layers left un-checkpointed per micro-batch, memory peak at the
             longest sequence (--probe-tokens, default 32768 = max_seq_length)

The timed window is optimizer steps 1 and 4 of sft_qwen3_8b_run1 (8 micro-batches,
53,120 tokens, lengths recovered from its nsys profile), ~1 minute. Reference
times for the same micro-batches are in REFERENCES. Load, compile and the memory
probe add ~4 minutes of untimed wall clock.

    cd ~/tmax/training/open-instruct
    PYTHONPATH=<repo root> uv run --no-sync python <repo root>/scripts/bench_spark_opt.py \
        --config <repo root>/configs/spark_opt_qwen3_8b.yaml
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

# Same allocator setting the launcher exports under SPARK_OPT_CONFIG; must be set
# before CUDA initializes.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
import spark_opt  # noqa: E402

GUARD_LIMIT_GB = 110
GRAD_ACCUM = 4
TIMED_STEPS = [[3744, 5152, 7072, 6496], [15840, 8800, 4000, 2016]]
WARMUP_STEP = [3008, 12000, 6001]  # plus the probe; compiles, exercises both checkpoint modes and padding
PROFILE_TOKENS = 9856  # run1's mean micro-batch
REFERENCES = {  # seconds for TIMED_STEPS, 2026-09-28, scripts from ~/bench_fp8
    "bf16 + compile, Liger FLCE": 26.19 + 37.22,
    "fp8 + compile, Liger FLCE": 23.25 + 33.68,
}


class MemoryGuard(threading.Thread):
    """The standing guardrail loop only matches finetune.py, so carry our own."""

    def __init__(self):
        super().__init__(daemon=True)
        self.peak_gb = 0.0

    @staticmethod
    def used_gb():
        with open("/proc/meminfo") as f:
            info = {line.split(":")[0]: int(line.split()[1]) for line in f}
        return (info["MemTotal"] - info["MemAvailable"]) / 1048576

    def reset(self):
        self.peak_gb = self.used_gb()

    def run(self):
        while True:
            used = self.used_gb()
            self.peak_gb = max(self.peak_gb, used)
            if used > GUARD_LIMIT_GB:
                print(f"MEMORY GUARD: {used:.1f} GB used > {GUARD_LIMIT_GB} GB, killing", flush=True)
                os._exit(137)
            time.sleep(0.25)


def make_batch(seq_len, gen):
    ids = torch.randint(0, 151643, (1, seq_len), generator=gen)
    labels = ids.clone()
    labels[:, : seq_len // 2] = -100
    return {"input_ids": ids.cuda(), "attention_mask": torch.ones_like(ids).cuda(), "labels": labels.cuda()}


def micro_step(model, batch):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model(**batch, use_cache=False).loss
    (loss / GRAD_ACCUM).backward()
    return loss.detach()


def check_flce(chunk_tokens):
    """Patched FLCE vs Liger's original on one random chunked problem."""
    from liger_kernel.ops import fused_linear_cross_entropy as flce

    torch.manual_seed(0)
    n, hidden, vocab = 2 * chunk_tokens + 1000, 4096, 151936
    x = (torch.randn(n, hidden, device="cuda") * 0.5).bfloat16().requires_grad_()
    w = (torch.randn(vocab, hidden, device="cuda") * 0.02).bfloat16().requires_grad_()
    y = torch.randint(0, vocab, (n,), device="cuda")
    y[: n // 3] = -100
    new = flce.fused_linear_cross_entropy_forward(x, w, y)
    old = flce.fused_linear_cross_entropy_forward.original(x, w, y, accum_dtype=torch.float32)
    rel = lambda a, b: ((a.float() - b.float()).norm() / b.float().norm()).item()  # noqa: E731
    result = {
        "loss_rel_err": abs(new[0].item() - old[0].item()) / abs(old[0].item()),
        "grad_input_rel_err": rel(new[4], old[4]),
        "grad_weight_rel_err_vs_liger_fp32_accum": rel(new[5], old[5]),
    }
    del new, old, x, w
    torch.cuda.empty_cache()
    return result


def check_padding(model, gen):
    """A 1000-token sequence padded to 1008 by spark_opt vs the same sequence with
    8 tokens appended by hand (labels -100).

    Appending the pad token by hand gives spark_opt's exact input, so the loss
    must be bit-identical. Appending random tokens may not be: tensorwise FP8
    takes one scale per tensor from the max over all tokens, so any appended
    token can nudge every token's rounding. That gap bounds the effect of
    padding on the loss; causal masking itself is exact.
    """
    batch = make_batch(1000, gen)
    pad_id = model.config.pad_token_id if model.config.pad_token_id is not None else model.config.eos_token_id

    def appended(tail):
        return {
            "input_ids": torch.cat([batch["input_ids"], tail], dim=1),
            "attention_mask": torch.ones(1, 1008, dtype=torch.long, device="cuda"),
            "labels": torch.cat([batch["labels"], torch.full((1, 8), -100, device="cuda")], dim=1),
        }

    with torch.autocast("cuda", dtype=torch.bfloat16):
        padded = model(**batch, use_cache=False).loss.item()
        pad_by_hand = model(**appended(torch.full((1, 8), pad_id, device="cuda")), use_cache=False).loss.item()
        junk = model(**appended(torch.randint(0, 151643, (1, 8), device="cuda")), use_cache=False).loss.item()
    return {
        "loss_padded_by_spark_opt": padded,
        "loss_pad_appended_by_hand": pad_by_hand,
        "identical_to_hand_padding": padded == pad_by_hand,
        "loss_junk_appended": junk,
        "fp8_scale_coupling_rel": abs(padded - junk) / abs(padded),
    }


def kernel_breakdown(prof):
    classes = {
        "attention (FlashAttention-2)": lambda n: "flash" in n,
        "attention (other backend!)": lambda n: any(k in n for k in ("cudnn", "fmha", "efficient", "flex")),
        "GEMM": lambda n: any(k in n for k in ("nvjet", "gemm", "cutlass", "xmma", "Kernel2")),
        "inductor fused (norms, rope, swiglu, fp8 casts)": lambda n: n.startswith("triton"),
        "Liger CE kernel": lambda n: "liger_cross_entropy" in n,
        "eager elementwise": lambda n: "elementwise" in n,
        "optimizer": lambda n: "Optimizer" in n,
    }
    totals = {name: 0.0 for name in classes}
    totals["other"] = 0.0
    rows = []
    for evt in prof.key_averages():
        t = evt.self_device_time_total
        if t <= 0:
            continue
        name = evt.key
        rows.append((t, evt.count, name))
        for cls, match in classes.items():
            if match(name):
                totals[cls] += t
                break
        else:
            totals["other"] += t
    total = sum(totals.values())
    rows.sort(reverse=True)
    return {
        "total_ms": total / 1e3,
        "share_pct": {k: round(100 * v / total, 1) for k, v in totals.items() if v},
        "top_kernels": [f"{t / 1e3:8.1f} ms  n={c:4d}  {n[:100]}" for t, c, n in rows[:12]],
        "fp32_add_hotspot_ms": sum(t for t, _, n in rows if "CUDAFunctor_add<float>" in n) / 1e3,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--probe-tokens", type=int, default=32768, help="0 skips the memory probe")
    args = parser.parse_args()

    cfg = spark_opt.load_config(args.config)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = REPO_ROOT / "runs" / f"bench_spark_opt_{stamp}"
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text(Path(args.config).read_text())
    report = {"config": str(Path(args.config).resolve()), "started": stamp}
    report["repo_commit"] = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()

    guard = MemoryGuard()
    guard.start()

    import bitsandbytes as bnb
    from liger_kernel.transformers import AutoLigerKernelForCausalLM

    t0 = time.time()
    model = AutoLigerKernelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
        fused_linear_cross_entropy=True,
        rms_norm=False,
        rope=False,
        swiglu=False,
    ).cuda()
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.train()
    spark_opt.apply(model, cfg)
    optimizer = bnb.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.0, optim_bits=8, is_paged=False)
    report["load_s"] = round(time.time() - t0, 1)
    report["float8_modules"] = sum(type(m).__name__ == "Float8Linear" for m in model.modules())
    print(f"loaded + applied in {report['load_s']}s, Float8Linear: {report['float8_modules']}", flush=True)

    report["flce_check"] = check_flce(cfg["flce"]["chunk_tokens"])
    print(f"FLCE check: {report['flce_check']}", flush=True)

    gen = torch.Generator().manual_seed(0)

    # Warmup: one optimizer step, which also allocates the 8-bit AdamW state.
    t0 = time.time()
    for seq_len in WARMUP_STEP:
        micro_step(model, make_batch(seq_len, gen))
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    report["warmup_s"] = round(time.time() - t0, 1)

    # Memory probe at the longest possible sequence, with weights, gradients and
    # optimizer state all resident as in training. An over-budget checkpoint
    # policy dies here, not hours into a run. Gradients are discarded.
    if args.probe_tokens:
        torch.cuda.reset_peak_memory_stats()
        guard.reset()
        micro_step(model, make_batch(args.probe_tokens, gen))
        torch.cuda.synchronize()
        report["probe"] = {
            "tokens": args.probe_tokens,
            "layers_not_checkpointed": spark_opt.history()[-1]["layers_not_checkpointed"],
            "peak_alloc_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 2),
            "peak_system_used_gb": round(guard.peak_gb, 1),
        }
        optimizer.zero_grad(set_to_none=True)
        print(f"memory probe: {report['probe']}", flush=True)
    report["pad_check"] = check_padding(model, gen)
    print(f"padding check: {report['pad_check']}", flush=True)
    graphs_before = torch._dynamo.utils.counters["stats"]["unique_graphs"]

    # One profiled micro-batch at run1's mean length, gradients discarded.
    from torch.profiler import ProfilerActivity, profile

    batch = make_batch(PROFILE_TOKENS, gen)
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        micro_step(model, batch)
        torch.cuda.synchronize()
    optimizer.zero_grad(set_to_none=True)
    report["kernels"] = kernel_breakdown(prof)
    del prof

    # Timed window.
    spark_opt.log_metrics()
    torch.cuda.reset_peak_memory_stats()
    guard.reset()
    step_s, losses, micro = [], [], []
    for lens in TIMED_STEPS:
        batches = [make_batch(n, gen) for n in lens]
        torch.cuda.synchronize()
        t = time.time()
        for b in batches:
            tm = time.time()
            torch.cuda.reset_peak_memory_stats()
            losses.append(micro_step(model, b))
            torch.cuda.synchronize()
            micro.append({
                **spark_opt.history()[-1],
                "seconds": round(time.time() - tm, 2),
                "peak_alloc_gib": round(torch.cuda.max_memory_allocated() / 2**30, 1),
            })
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        step_s.append(time.time() - t)

    tokens = sum(map(sum, TIMED_STEPS))
    seconds = sum(step_s)
    counters = torch._dynamo.utils.counters
    report.update(
        {
            "timed_tokens": tokens,
            "timed_s": round(seconds, 2),
            "tok_per_s": round(tokens / seconds, 1),
            "step_s": [round(s, 2) for s in step_s],
            "speedup_vs": {k: round(v / seconds, 3) for k, v in REFERENCES.items()},
            "micro_batches": micro,
            "window_metrics": spark_opt.log_metrics(),
            "timed_peak_alloc_gib": max(m["peak_alloc_gib"] for m in micro),
            "timed_peak_system_used_gb": round(guard.peak_gb, 1),
            "losses_finite": bool(all(torch.isfinite(x) for x in losses)),
            "mean_loss": round(float(torch.stack(losses).float().mean()), 4),
            "compile": {
                "unique_graphs": counters["stats"]["unique_graphs"],
                "recompiles_in_timed_window": counters["stats"]["unique_graphs"] - graphs_before,
                "graph_breaks": sum(counters["graph_break"].values()),
            },
        }
    )
    (run_dir / "result.json").write_text(json.dumps(report, indent=2))

    k = report["kernels"]
    print("\n==== spark_opt benchmark ====")
    print(f"timed: {tokens} tok in {seconds:.1f}s = {report['tok_per_s']} tok/s   steps {report['step_s']}")
    for name, ratio in report["speedup_vs"].items():
        print(f"  {ratio:.3f}x vs {name}")
    print("per micro-batch:")
    for m in micro:
        print(f"  {m['tokens']:6d} tok (+{m['pad']:2d} pad)  un-checkpointed layers {m['layers_not_checkpointed']:2d}/36"
              f"  {m['seconds']:5.1f}s  {m['tokens'] / m['seconds']:6.0f} tok/s  peak {m['peak_alloc_gib']} GiB")
    if "probe" in report:
        print(f"memory probe: {report['probe']}")
    print(f"timed window peak: {report['timed_peak_alloc_gib']} GiB allocated, "
          f"{report['timed_peak_system_used_gb']} GB system")
    print(f"FLCE: {report['flce_check']}  window {report['window_metrics']}")
    print(f"padding: {report['pad_check']}")
    print(f"GPU time by class, one {PROFILE_TOKENS}-token micro-batch ({k['total_ms']:.0f} ms):")
    for name, pct in sorted(k["share_pct"].items(), key=lambda kv: -kv[1]):
        print(f"  {pct:5.1f}%  {name}")
    print(f"  old fp32 add hotspot: {k['fp32_add_hotspot_ms']:.1f} ms")
    print("top kernels:")
    for line in k["top_kernels"]:
        print(f"  {line}")
    print(f"compile: {report['compile']}  Float8Linear: {report['float8_modules']}")
    print(f"loss finite: {report['losses_finite']}  mean {report['mean_loss']}")
    print(f"result: {run_dir / 'result.json'}")


if __name__ == "__main__":
    main()
