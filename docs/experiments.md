# Experiments

One row per run. `AGENTS.md` §5: *"An untracked run didn't happen."* Every entry
needs the run directory (which holds `config.yaml`, `provenance.txt`, `train.log`)
so the run can be reconstructed.

Runs are created by `scripts/sft_qwen3_8b_train_it.sh` (full fine-tune) or
`scripts/run_sft.sh` (LoRA) under `runs/<exp_name>_<UTC stamp>/`.

## Engineering runs — memory and throughput

No evaluation attached; these exist to make the full fine-tune fit on the box.
Logs live in the tmax checkout under `training/open-instruct/runs/`, not in this repo.

| # | Date | Log | Config delta | Outcome |
|---|---|---|---|---|
| E-1 | 2026-09-19 | `test1_full_bnb8bit_20260919T093441Z.log` | full FT, bnb 8-bit AdamW, no deepspeed, 32k | **Killed by guardrail at 77 GB**, step 2. Step 1 loss 1.0524177551269531, 730.8 TPS |
| E-2 | 2026-09-19 | `test1_full_bnb8bit_20260919T104739Z.log` | as E-1, ceiling 105 GB, `--max_train_steps 536` | **Killed by guardrail at 106 GB**, step 2. Step 1 loss bit-identical to E-1 |
| E-3 | 2026-09-19 | `test1_full_bnb8bit_20260919T124302Z.log` | as E-1 **+ `--use_liger_kernel`**, `--max_train_steps 20` | **Completed 20/20**, checkpoint written. Peak 72 GB training / 84 GB saving. Step 1 loss 1.0517414808273315, 783.9 tok/s measured. wandb `run-20260919_144313-jokdavh1` |

Both E-1 and E-2 died on the same sequence — step 2 draws a 27,283-token example
(dataset index 9833; the sampler is deterministic at `seed 42`). E-3 passed it at
72 GB. What changed between them is `--use_liger_kernel`; see
`docs/spark-memory-guardrail.md` §4.

## Cost of the full run — measured

From E-3, measured rather than estimated:

- **783.9 tokens/sec** (676,524 real tokens over 863 s of wall clock)
- 95,408,029 tokens/epoch × 2 epochs = **190.8M tokens**
- **≈ 67.6 hours — about 2.8 days** of continuous Spark occupancy

Step time varies **27–53 s** with sequence length, so any ETA extrapolated from a
single step is unreliable; use the token total.

## Main run

| Run dir | Started | tmax commit | Status |
|---|---|---|---|
| `runs/sft_qwen3_8b_run1_20260920T052358Z` | 2026-09-20 07:24 CEST | `6d3d606c` + `patches/finetune-spark.patch` | in progress — 5,362 steps, ~2.8 days |

Launched on a **clean** `~/tmax` clone and a rebuilt environment (transformers 5.17.0
rather than E-3's 5.15.1). Step-1 loss `1.0517414808273315` is **bit-identical to E-3**,
so the stack reproduces despite the version drift.

## Planned order

The ordering matters more than the individual runs — steps 1–2 exist so that
step 4 means something.

1. **Environment check.** `python scripts/verify_spark_env.py` on the Spark.
   Not a run; blocks everything else. Records the resolved attention backend.
2. **Smoke.** Proves the pipeline runs and yields the first **tokens/sec** number.
   Minutes. *Done for the full-FT path by E-3.*
3. **Baseline evaluation.** Qwen3-8B on Terminal-Bench 2.1, untrained, on a pinned
   harness. **Must happen before any training.** Paper Table 5 shows the same model
   scoring 36.0–44.1 across four harnesses — an 8-point spread, larger than the
   effect we are trying to measure. Without this number, no training result is
   interpretable. Blocked on the evaluation-harness milestone (D-07).
4. **Main run.** `scripts/sft_qwen3_8b_train_it.sh`, ~2.8 days. Paper's expectation
   for this configuration: TB 2.1 1.1 → 6.0, TB Lite 7.3 → 11.5.
5. **Ablation: `_all` vs `_only_success`.** The D-06 disagreement — paper §D.2 keeps
   failed rollouts, `docs/theory/lora-sft.md` §5 argues they must be dropped. One
   config key apart.

## Evaluation (D-07: K decision)

**Decision: K=5** (2026-09-24). Matches paper protocol. Cost: ~35h per arm, ~70h total
(~3 days). K=1 would be ~7h per arm but too noisy (95% CI ±9.5% on small effect).

Harness: **Vanillux2Agent** (same as training SFT data generation). Earlier eval
runs (2026-09-17/18) used terminus-2 and are not comparable.

Launch with: `K=5 bash scripts/eval_both_arms.sh` (on Spark, under tmux).

## Evaluated runs

| # | Date | Run dir | Model | Harness | K | TB 2.1 | TB Lite | Notes |
|---|---|---|---|---|---|---|---|---|
| EV-1a | — | `~/tb2-eval/jobs/vanillux2-base-k5-*` | Qwen3-8B baseline | Vanillux2Agent | 5 | — | — | pending |
| EV-1b | — | `~/tb2-eval/jobs/vanillux2-sft-k5-*` | qwen3-8b-sft (run1) | Vanillux2Agent | 5 | — | — | pending |
