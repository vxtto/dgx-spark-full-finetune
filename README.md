# Full fine-tuning of Qwen3-8B on one NVIDIA DGX Spark

A full fine-tune of an 8 B-parameter model at 32,768-token sequences, on a single
desktop machine with 128 GB of memory shared between CPU and GPU. Two things are
measured here: how to make it fit, and how to make it 40% faster at the same
training loss.

The recipe is the SFT half of TMAX ([arXiv:2606.23321](https://arxiv.org/abs/2606.23321)),
run with the authors' own code ([`hamishivi/tmax`](https://github.com/hamishivi/tmax),
a fork of open-instruct). This repo is an overlay on that code: one patch, one
module, a launcher, and the measurements.

## Results

The paper's launch command, cut down to one GPU, is killed at about 113 GB before
the first optimizer step finishes. Weights, gradients and fp32 AdamW state alone
come to 98 GB. With an 8-bit optimizer and a fused cross-entropy the same run peaks
at 72 GB, and then trains to completion:

| | Run 1: make it fit | Run 2: make it fast |
|---|---|---|
| Throughput | 849 tok/s | 1,189 tok/s (**+40%**) |
| Wall clock, 5,362 steps | 62 h 26 min | 44 h 35 min (**−29%**) |
| Mean training loss | 0.2689 | 0.2696 |

![Steps completed against wall-clock hours for both runs](docs/img/wall_clock.svg)

![Training loss per 100 steps for both runs](docs/img/loss.svg)

Run 2's gain comes from profiling run 1:

1. **A cross-entropy hotspot.** 9% of GPU time went to rebuilding the output
   layer's 151,936 × 4,096 gradient about 30 times per sequence. Larger chunks and
   in-place accumulation make it 2.
2. **Recomputing less.** Gradient checkpointing recomputed every layer on every
   sequence. With memory to spare, a 20 GiB budget lets short sequences skip it.
3. **FP8 matrix multiplies**, with `torch.compile` to fuse the casts. A smaller
   share of the gain, and it shifts the first-step loss by 3%.

On a fixed set of micro-batches the first two give 1.42× and all three give 1.60×.
Details, caveats and the file behind each number: [`docs/results.md`](docs/results.md).

## What this does not show

The paper reports that this fine-tune lifts Qwen3-8B from 1.1% to 6.0% on
Terminal-Bench. Here the untrained model scores 1.1%, run 1 scores 1.5% and run 2
scores 3.4%, with about a third of the fine-tuned model's trials ending in a
timeout. **The model-quality result is not reproduced**, and the evaluation is
still being worked on. The candidate causes and the test for each are in
[section 7 of the results](docs/results.md#7-evaluation-work-in-progress).

This is also one machine. Nothing here is evidence about multi-GPU or multi-node
training.

## What is in the repo

| Path | What |
|---|---|
| `spark_opt.py`, `configs/spark_opt_qwen3_8b.yaml` | the run 2 throughput changes, applied to a loaded model |
| `patches/finetune-spark.patch` | the changes to upstream `finetune.py`: 8-bit optimizer on a full fine-tune, the `spark_opt` hook, an attention-detection fix for ARM |
| `scripts/sft_qwen3_8b_train_it.sh` | launcher: checks the patch and environment, records provenance, runs the fine-tune |
| `scripts/bench_spark_opt.py` | one-minute throughput and correctness check of a `spark_opt` config |
| `scripts/clean_dataset.py` | rebuilds the training dataset from `allenai/tmax-sft` |
| `scripts/plot_runs.py` | draws the two figures above |
| `scripts/eval_*.sh`, `scripts/patch_harbor.py`, `patches/vanillux2-docker-timeout.patch`, `patches/tmax-harness-config/` | Terminal-Bench evaluation on the Spark |
| `scripts/failure_mode_bucketing.py` | groups eval trials by why they ended |
| `docs/results.md` | all measurements |
| `docs/spark-memory-guardrail.md` | the memory model, and the kill loop every run on this machine needs |
| `docs/environment.md` | aarch64 / CUDA 13 setup and pitfalls |
| `evidence/` | per-step training data, benchmark results, kernel profile summary, eval job results |

## Reproducing a run

On the Spark, from a clone of this repo:

```bash
git clone https://github.com/hamishivi/tmax.git ~/tmax
git -C ~/tmax checkout 6d3d606c
git -C ~/tmax apply "$PWD/patches/finetune-spark.patch"
```

Install the environment as described in [`docs/environment.md`](docs/environment.md),
start the memory guardrail from [`docs/spark-memory-guardrail.md`](docs/spark-memory-guardrail.md)
in a second shell, then:

```bash
# run 1
EXP_NAME=sft_qwen3_8b_run1 bash scripts/sft_qwen3_8b_train_it.sh

# run 2
SPARK_OPT_CONFIG=configs/spark_opt_qwen3_8b.yaml \
  EXP_NAME=sft_qwen3_8b_run2 bash scripts/sft_qwen3_8b_train_it.sh
```

## Credits

A two-person project by Vittorio Rossi and Alfredo Ceci, September 2026, on
Alfredo's DGX Spark. The training-side work in this repo (memory model, patch,
`spark_opt`, benchmarks, the two runs) is Vittorio's. The evaluation harness setup,
the eval launch scripts and the two eval patches are Alfredo's.

Coding agents were used throughout to write and run code. The measurements were
run on the hardware, and the numbers on this page are read from the files in
`evidence/`.

Built on [`hamishivi/tmax`](https://github.com/hamishivi/tmax) and
[open-instruct](https://github.com/allenai/open-instruct) (Apache-2.0),
[Liger Kernel](https://github.com/linkedin/Liger-Kernel),
[torchao](https://github.com/pytorch/ao) and
[bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes).

Licensed under Apache-2.0. See [`NOTICE`](NOTICE) for what derives from upstream.
