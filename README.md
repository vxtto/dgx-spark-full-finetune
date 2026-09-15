# Training an LLM

## Goal

Improve an open-weight LLM on **Terminal Bench 2.0**, on a single NVIDIA DGX Spark.

## How

LoRA SFT, reproducing the SFT half of **TMAX** ([arXiv:2606.23321](https://arxiv.org/abs/2606.23321),
Ai2 / UW) on `Qwen/Qwen3-8B`, using the authors' own stack
([`hamishivi/tmax`](https://github.com/hamishivi/tmax), a fork of
[open-instruct](https://github.com/allenai/open-instruct)).

Why Qwen3-8B and not Qwen3.5: the paper measures both, and SFT *improves* Qwen3-8B
(TB 2.1: 1.1 → 7.9) while *degrading* Qwen3.5-9B. See `docs/decisions.md` D-02.

## Where things are

| Path | What |
|---|---|
| `docs/decisions.md` | what was decided, what is still open, and why |
| `docs/environment.md` | DGX Spark setup — aarch64 / CUDA 13 / sm_121 pitfalls |
| `docs/experiments.md` | run log and the planned order of runs |
| `docs/theory/` | how LoRA SFT works, and what it can and cannot teach |
| `configs/` | one config per run — no hyperparameters live in scripts |
| `scripts/` | environment check and run launcher |

`AGENTS.md` / `CLAUDE.md` are the agent instructions (two identical copies).

## Getting started, on the Spark

Training never runs on a laptop (`AGENTS.md` §6); `run_sft.sh` enforces this.

```bash
git clone https://github.com/hamishivi/tmax.git ~/tmax && cd ~/tmax && uv sync
```

```bash
python scripts/verify_spark_env.py
```

```bash
tmux new -s sft
bash scripts/run_sft.sh configs/smoke_qwen3_1_7b_lora.yaml
```

The smoke run is the gate: it proves the pipeline works and produces the tokens/sec
number the real run's cost estimate depends on. Only then
`configs/sft_qwen3_8b_lora_spark.yaml`.

## Status

Decisions recorded up to D-07. Not yet done, in order:

1. verify the environment on the Spark — `deepspeed` on aarch64 is the open risk
2. pin an evaluation harness and measure the **untrained** baseline
   (milestone *Epic: configure evaluation harness*)
3. first training run

Step 2 is not optional: the same model scores 36.0–44.1 on TB Lite depending on the
harness, a spread larger than the effect being measured.
