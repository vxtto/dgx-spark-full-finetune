# Experiments

One row per run. `AGENTS.md` §5: *"An untracked run didn't happen."* Every entry
needs the run directory (which holds `config.yaml`, `provenance.txt`, `train.log`)
so the run can be reconstructed.

Runs are created by `scripts/run_sft.sh` under `runs/<exp_name>_<UTC stamp>/`.

| # | Date | Run dir | Model | Data | Config delta vs. baseline | TB 2.1 | TB Lite | Notes |
|---|---|---|---|---|---|---|---|---|
| — | — | — | — | — | — | — | — | *no runs yet* |

## Planned order

The ordering matters more than the individual runs — steps 1–2 exist so that
step 4 means something.

1. **Environment check.** `python scripts/verify_spark_env.py` on the Spark.
   Not a run; blocks everything else. Records the resolved attention backend.
2. **Smoke.** `configs/smoke_qwen3_1_7b_lora.yaml`. Proves the pipeline runs and
   yields the first **tokens/sec** number. Minutes.
3. **Baseline evaluation.** Qwen3-8B on Terminal-Bench 2.1, untrained, on a pinned
   harness. **Must happen before any training.** Paper Table 5 shows the same model
   scoring 36.0–44.1 across four harnesses — an 8-point spread, larger than the
   effect we are trying to measure. Without this number, no training result is
   interpretable. Blocked on the evaluation-harness milestone (D-07).
4. **Main run.** `configs/sft_qwen3_8b_lora_spark.yaml`. Paper's expectation for
   this configuration: TB 2.1 1.1 → 6.0, TB Lite 7.3 → 11.5.
5. **Ablation: `_all` vs `_only_success`.** The D-06 disagreement — paper §D.2 keeps
   failed rollouts, `docs/theory/lora-sft.md` §5 argues they must be dropped. One
   config key apart.
6. **Ablation: LoRA rank.** r=16 (D-05) against r=64. Tests the low-rank claim in
   `docs/theory/behaviour-vs-knowledge.md` §5.

## Cost estimate for step 4 — to be replaced with a measurement

Rough, and stated so it can be falsified by step 2:

- ~16.5k trajectories × 2 epochs
- assume ~10k tokens average after truncation at 16384 → **~330M training tokens**
- at an assumed 1.5–3k tokens/sec on this box → **~30–60 hours** of continuous
  Spark occupancy

The throughput figure is the weak link — it is a guess until step 2 measures it.
Record the real tokens/sec here and recompute before committing the Spark for
two days.
