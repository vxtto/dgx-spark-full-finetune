# Results

Every number on this page was measured on one NVIDIA DGX Spark (GB10, aarch64,
CUDA 13, 128 GB of memory shared between CPU and GPU). Where the file a number
comes from is in this repo, it is named. Where it is not, section 8 says so.

## 1. What was trained

| | |
|---|---|
| Model | `Qwen/Qwen3-8B`, full fine-tune (all 8.19 B parameters), bf16 |
| Data | [`vittorino/tmax-sft-cleaned`](https://huggingface.co/datasets/vittorino/tmax-sft-cleaned), config `skill_tax_20260505_2.2k_combined_balanced_thinking_all`, 10,724 trajectories |
| Recipe | the SFT recipe of TMAX ([arXiv:2606.23321](https://arxiv.org/abs/2606.23321)), run with the authors' code ([`hamishivi/tmax`](https://github.com/hamishivi/tmax) at `6d3d606c`) plus `patches/finetune-spark.patch` |
| Schedule | 2 epochs, 5,362 optimizer steps, LR 2e-5 linear, warmup 3%, max sequence length 32,768, seed 42 |
| Batch | 1 sequence per micro-batch, gradient accumulation 4 |

The dataset is `allenai/tmax-sft` minus two trajectories (indices 4558 and 4560)
that end in 15 and 25 identical assistant turns in a row. `scripts/clean_dataset.py`
rebuilds it and `--check` confirms the result is identical to the published copy.

**Where this differs from the paper's setup.** The paper trains on 32 GPUs with a
global batch of 128. This run has one device and a global batch of 4, at the same
learning rate, so it takes 32 times as many optimizer steps per epoch. Attention is
PyTorch SDPA (FlashAttention-2 kernel), because no flash-attn wheel exists for
aarch64; the paper uses flash attention 3.

## 2. Memory: making a full fine-tune fit

The paper's launch command, reduced to one process, was killed by the kernel at
about 113 GB on every attempt. The cause is arithmetic, at 8.19 B parameters:

| Per parameter | Bytes | Total |
|---|---|---|
| bf16 weights | 2 | 16.4 GB |
| bf16 gradients | 2 | 16.4 GB |
| fp32 AdamW state (two moments) | 8 | 65.6 GB |
| **Before any activations** | **12** | **98.4 GB** |

Three changes bring it to a 72 GB peak at the full 32,768-token length:

| Change | Effect |
|---|---|
| bitsandbytes 8-bit AdamW (`--use_8bit_optimizer`) | optimizer state 65.6 GB to about 17 GB |
| Liger fused linear cross-entropy (`--use_liger_kernel`) | activation cost 2.46 to 0.315 MB per token |
| no DeepSpeed | ZeRO-3 CPU offload moves nothing on a shared pool and makes the state swappable |

Upstream `finetune.py` only reaches the 8-bit optimizer under `--use_qlora`, so a
full fine-tune that passes `--use_8bit_optimizer` silently gets fp32 AdamW. The
patch fixes that, and the launcher refuses to start on an unpatched checkout.

The memory model, the measured peaks (72 GB training, 84 GB while saving) and the
external kill loop that every run needs on this machine are in
[`spark-memory-guardrail.md`](spark-memory-guardrail.md).

## 3. The two runs

| | Run 1 | Run 2 |
|---|---|---|
| Configuration | bf16, 8-bit AdamW, Liger | run 1 + `SPARK_OPT_CONFIG=configs/spark_opt_qwen3_8b.yaml` |
| Optimizer steps | 5,362 | 5,362 |
| Wall clock, first to last step | 62 h 26 min | 44 h 35 min (**−29%**) |
| Throughput over the run | 849 tok/s | 1,189 tok/s (**+40%**) |
| Mean training loss, all steps | 0.2689 | 0.2696 |
| Mean training loss, last 500 steps | 0.2288 | 0.2302 |
| Largest gap between 100-step loss means | | 0.0056 |
| W&B | [`a3zwdgg5`](https://wandb.ai/vittorio-rossi-epfl/open_instruct_internal/runs/a3zwdgg5) | [`bgewsxq3`](https://wandb.ai/vittorio-rossi-epfl/open_instruct_internal/runs/bgewsxq3) |
| Weights | `vittorino/open_instruct_dev@sft_qwen3_8b_run1` | [`vittorino/qwen3-8b-tmax-sft-run2`](https://huggingface.co/vittorino/qwen3-8b-tmax-sft-run2) |

Both runs use the same data order (seed 42), so their loss curves can be compared
step by step.

Source: `evidence/steps/run1.csv` and `run2.csv` (one row per optimizer step, taken
from each run's training log) and `evidence/provenance/`. `scripts/plot_runs.py`
draws the two figures in the README from those files.

## 4. Where run 1's time went

An `nsys` profile of 5 optimizer steps of the run 1 configuration (197,120 tokens)
gave the kernel breakdown in `evidence/profile/run1_nsys_cuda_gpu_kern_sum.csv`:

- About 77% of GPU time is matrix multiplies and FlashAttention-2. Both are
  compute-bound, so the only way to make them faster is to do less of them or use
  a faster number format.
- About 9% is two memory-bound kernels, a float add and a copy (5.4% and 4.0% in
  the CSV). They come from one place: Liger's fused cross-entropy rebuilds the
  whole `lm_head` gradient, a 151,936 × 4,096 matrix, once per chunk of about 512
  tokens.

## 5. Throughput: what run 2 changes

Everything is in `spark_opt.py`, switched on by one environment variable. Unset,
the launcher runs run 1 exactly.

| Change | Why |
|---|---|
| Cross-entropy: 4,096-token chunks, gradient accumulated in place in fp32 | about 2 chunks per call instead of about 30; removes the 9% hotspot |
| Adaptive gradient checkpointing, 20 GiB budget | recomputing activations was about a quarter of the heavy compute; short sequences now skip it |
| FP8 (torchao, tensorwise) on every decoder linear layer, `lm_head` kept in bf16 | faster matrix multiplies on this GPU |
| `torch.compile` per decoder layer | FP8's casts are memory-bound and need fusing, or they cost more than FP8 gains |
| Pad each micro-batch to a multiple of 16 tokens, labels −100 | required by the FP8 gradient multiply |
| SDPA pinned to the FlashAttention-2 kernel | fastest of the available backends here; pinned so nothing switches silently |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | peak system memory at 32,768 tokens 86.8 to 72.3 GB, same speed |

**Micro-benchmark.** `scripts/bench_spark_opt.py` times a full training step on
the same 8 micro-batches (optimizer steps 1 and 4 of run 1, 53,120 tokens):

| Configuration | Time | tok/s | Relative |
|---|---|---|---|
| bf16 + compile, stock Liger cross-entropy | 63.4 s | 838 | 1.00× |
| FP8 + compile, stock Liger cross-entropy | 56.9 s | 933 | 1.11× |
| `spark_opt` with FP8 off | 44.6 s | 1,190 | 1.42× |
| `spark_opt` (the run 2 config) | 39.6 s | 1,342 | 1.60× |

So the cross-entropy fix and adaptive checkpointing account for most of the gain,
and FP8 adds the step from 1.42× to 1.60× on these micro-batches. The full run
reached 1.40×, lower than the benchmark, because the timed micro-batches stop at
15,840 tokens and longer sequences keep more layers checkpointed. FP8's share of
the full-run gain was not measured separately.

Source: `evidence/bench/` (`...T005729Z` is the run 2 config, `...T011602Z` is FP8
off, `...T005335Z` is the default allocator) and
`evidence/profile/bench_fp8_results.jsonl` for the first two rows.

Checks the benchmark runs on every invocation, from the same result files:

- The patched cross-entropy returns the same loss and input gradient as Liger's
  (relative error 0.0). Its `lm_head` gradient differs by 1.4e-3 from Liger's own
  fp32 accumulation, because Liger rounds each chunk to bf16 first.
- Padding done by `spark_opt` gives the same loss as padding by hand.
- At 32,768 tokens, 6 of 36 layers skip checkpointing; on the timed micro-batches
  30 of 36 do on average.
- No recompiles and no graph breaks inside the timed window.

## 6. What FP8 does to the loss

Tensorwise FP8 changes the forward pass measurably. First three optimizer steps on
the same data:

| Step loss | Run 1 (bf16) | `spark_opt`, FP8 off | FP8 tensorwise | FP8 rowwise |
|---|---|---|---|---|
| 1 (untrained model) | 1.0517 | 1.0508 | 1.0194 (−3.1%) | 1.0608 (+0.9%) |
| 2 | 0.4931 | | 0.4907 | 0.4945 |
| 3 | 0.5743 | | 0.5705 | 0.5822 |

Rowwise is closer to bf16 but ran slower than plain bf16 here (about 600 tok/s
against 1,050 for tensorwise at step 3), so it was not used. Over the full run the difference washes out of the training loss
(section 3). Whether it matters for the trained model's behaviour is not
established: see section 7.

Source: `evidence/smoke/` (the training log of each three-step run) and the first
rows of `evidence/steps/run1.csv`.

## 7. Evaluation: work in progress

Terminal-Bench 2.0, 89 tasks × 3 attempts = 267 trials per arm, with the paper's
own agent harness (Vanillux2Agent) through Harbor, model served locally with vLLM.

| Arm | Solved trials | Mean | Pass@2 | Trials ending in an exception |
|---|---|---|---|---|
| Qwen3-8B, untrained | 3 | 1.1% | 1.9% | 5 |
| Run 1 | 4 | 1.5% | 2.6% | 82 |
| Run 2 | 9 | 3.4% | 5.6% | 89 |

The paper reports 1.1 → 6.0 on Terminal-Bench 2.1 for this model and dataset
(its Table 7). **This run does not reproduce that gain, and these numbers do not
show whether run 2 is better than run 1.** About a third of the fine-tuned model's
trials end in an agent timeout, so the scores partly measure how fast the Spark
serves the model.

Where this eval differs from the paper's:

- Terminal-Bench 2.0, not 2.1.
- Local Docker on an ARM machine, so the benchmark's x86 containers run under
  emulation. Harbor's timeouts are multiplied by 6 to compensate.
- `patches/vanillux2-docker-timeout.patch` makes a command that overruns its
  120 s limit cost the agent one turn, as on the paper's Daytona backend, where
  Harbor's Docker backend would end the trial. `scripts/patch_harbor.py` raises
  the tool-install timeouts for the same reason.
- 3 attempts per task; the paper's protocol uses 5.

How the trials ended, base model against run 1
([`failure-modes-base-vs-run1.md`](failure-modes-base-vs-run1.md), produced by
`scripts/failure_mode_bucketing.py`):

| Primary cause | Base | Run 1 |
|---|---|---|
| Submitted, tests failed | 79% | 5% |
| Stopped early: context window exceeded | 16% | 50% |
| Agent timeout (90 min) | 1% | 29% |
| Hit the 64-step limit | 2% | 13% |

The untrained model gives up early and submits wrong answers. The fine-tuned model
keeps working until it runs out of context or time. The same breakdown has not been
produced for run 2 yet.

Source: `evidence/eval/` (Harbor's `result.json` and `config.json` per job).

The eval harness setup, both launch scripts, the Docker-timeout patch and the
Harbor patch are Alfredo Ceci's work, as are the base and run 1 eval jobs.

### Under investigation

1. **Learning rate against batch size.** Global batch 4 at the paper's LR of 2e-5,
   which was chosen for a global batch of 128. Test: rerun with a scaled-down LR.
2. **Chat template and loss masking.** open-instruct masks the loss by assuming
   that rendering the first N turns gives a prefix of rendering the whole
   conversation. The two trajectories removed from the dataset broke that
   assumption; whether Qwen3's handling of thinking in earlier turns breaks it
   elsewhere has not been checked.
3. **Eval speed.** Timeouts under emulation may dominate the score. Test: the
   failure-mode breakdown for run 2, and a run on native x86 sandboxes.
4. **Sampling settings.** The base model's `generation_config.json` sets
   `top_k=20`; the fine-tuned checkpoints may not carry it. The arms should be
   served with identical, explicit sampling settings.
5. **FP8.** Run 1 and run 2 differ in the whole `spark_opt` bundle, so an eval gap
   between them cannot be attributed to FP8 alone.

## 8. What is not in this repo

- The full training logs, the `nsys` report and its database, and Harbor's
  per-trial directories. `evidence/` holds the per-step extracts, the kernel
  summary and the job-level result files.
- The logs behind section 2's memory figures and the attention-backend comparison.
- Run 2 was launched with a slightly larger patch than the one published here: it
  also had hooks for uploading periodic snapshots to the Hugging Face Hub. They
  were removed from the published patch because they do not affect training.
  `evidence/provenance/sft_qwen3_8b_run2.txt` records the run as launched.
