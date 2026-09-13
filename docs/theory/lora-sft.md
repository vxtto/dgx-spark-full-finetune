# LoRA SFT — how it works

Background notes on the training method under consideration for this project.
Written 2026-09-13.

> **This is theory, not a decision.** Nothing here settles any open item in §4 of
> `AGENTS.md`. It records how the mechanism works so the decision can be made with
> the mechanics understood. The base model, the source of training environments, and
> the final choice of method remain open.

Companion visual for the merge arithmetic (§4 below):
<https://claude.ai/code/artifact/e46d8dd6-3f74-4574-b658-c493d3eda2d7>

Companion note on what this method can and cannot teach: `behaviour-vs-knowledge.md`.

---

## 1. SFT — supervised fine-tuning

**Supervised**, not "soft". Plain supervised learning: input/output pairs,
cross-entropy loss on the target tokens, backpropagation.

### The loss

The loss is **how surprised the model was by the correct answer** — not the difference
between output and input.

At each position the model emits a probability distribution over the whole vocabulary
(~150k tokens). The loss at that position is `-log(p)` where `p` is the probability it
assigned to the token that actually comes next.

| Probability assigned to the correct token | Loss |
|---|---|
| 0.9 | 0.11 |
| 0.3 | 1.20 |
| 0.01 | 4.61 |

Summed over every position in the target. Training pushes up the probability of the
tokens that are actually there.

The **gradient** is not the thing being minimized — it is the compass. For each weight
it answers "which direction, and how hard, does this weight push the loss up?" You step
the opposite way. The gradient tending to zero is a symptom of approaching a minimum,
not the objective.

### Where SFT sits among post-training methods

| Method | Needs | Learns from failure? |
|---|---|---|
| **SFT** | demonstrations | No — it can only imitate |
| **DPO** | pairs (better, worse) | Yes — pushes the worse one down |
| **RL / GRPO** | a score per attempt | Yes — pushes below-average attempts down |

This table is the whole argument for and against the method. See §5.

---

## 2. Why full fine-tuning does not fit the Spark

To run the training loop over all 8B parameters, GPU memory must hold simultaneously:

| Component | Size (8B model) |
|---|---|
| Weights, bf16 | 16 GB |
| One gradient per weight | 16 GB |
| Adam optimizer state (2 fp32 values per weight) | 64 GB |
| fp32 master copy of the weights | 32 GB |
| **Total, before activations** | **~128 GB** |

That is the entire 128 GB of the Spark with nothing left for computation. Full
fine-tuning of an 8B model is therefore borderline; of a 30B model, impossible.

---

## 3. LoRA — Low-Rank Adaptation

### The mechanism

All original weights are **frozen**. New, small parameters are added alongside them.

For a weight matrix `W` of shape 4096×4096, LoRA learns two skinny matrices whose
product has the same shape as `W`:

```
        ΔW      =       B        ×       A
   (4096×4096)     (4096 × r)       (r × 4096)
```

At inference every adapted layer computes `(W + B·A)·x`.

With `r = 16`:

| | Shape | Parameters |
|---|---|---|
| `W` original | 4096 × 4096 | 16,777,216 |
| `B` | 4096 × 16 | 65,536 |
| `A` | 16 × 4096 | 65,536 |
| `B·A` (the delta it generates) | 4096 × 4096 | 16,777,216 |

**131,072 stored numbers generate a 16,777,216-number change.** That ratio is the trick.

Two initialization details:

- `A` is random, `B` is **all zeros**, so `B·A = 0` at step 0 and the model starts
  bit-identical to the base. It cannot be broken before training begins.
- `W` is never written to. Not during training, not at inference.

### What it buys

| | Full FT 8B | LoRA 8B |
|---|---|---|
| Weights in memory | 16 GB | 16 GB (frozen) |
| Gradients | 16 GB | ~0.1 GB |
| Optimizer state | 96 GB | ~0.5 GB |
| **Roughly** | **~128 GB** | **~20 GB** |

Plus: the saved artifact is the adapter alone — ~50M parameters, ~100 MB at r=16 — not
a 16 GB model. Several variants can be kept and hot-swapped against one loaded base.

### The constraint, stated correctly

LoRA does **not** select a subset of weights to train.

- **How many weights change:** nearly all of them. `B·A` is a full-size matrix, so
  100% of the numbers in an adapted layer come out different.
- **How freely they change:** ~50M degrees of freedom across the whole model. Billions
  of values move, but all of that movement is derived from 50M stored numbers, so the
  changes are heavily correlated.

> **LoRA does not limit *how many* weights change. It limits *how freely* they change.**

Analogy: rotating a photo by 3°. Every one of 12 million pixels changes value; the
entire transformation is described by one number.

This is why LoRA is strong for teaching **behavior** (a broad, coordinated tilt) and
weaker for injecting **facts** (which need many independent, unrelated adjustments).

### Which parameters are actually adapted

Controlled by `target_modules`. Embeddings and layer norms are normally left alone.

| Target | Share of an 8B model that changes |
|---|---|
| Attention only (cheap default) | ~20% |
| All linear layers (attention + MLP) | ~87% |

---

## 4. Merging — the arithmetic

Merging is **optional** and happens after training. It does not "activate" anything:
the adapter is live from the first forward pass of training, because otherwise no
gradient would flow into `A` and `B` at all.

```
W_final = W + (α/r) · B · A
```

Plain element-wise matrix addition. **Exact and lossless** — no approximation happens
at merge time. The approximation already happened during training, in the constraint
that the update be low-rank.

### Why the merged model is not bigger

A 100 MB adapter merged into a 16 GB model gives a **16 GB** model.

The adapter is never appended. `B` and `A` are multiplied together *first*, producing a
matrix already the same shape as `W`; that full-size matrix is what gets added.

```
131k stored numbers → multiply → 16.7M numbers → added into W's existing 16.7M slots
```

The layer's parameter count before and after the merge is identical. The adapter is a
compressed recipe for a full-size change, not a pile of extra weights — and once
applied, the recipe is redundant.

### Merged vs unmerged

| | Keep separate | Merge |
|---|---|---|
| Artifact | base 16 GB + adapter ~100 MB | one 16 GB model |
| Swap variants | yes, hot-swap at serving | no, one per merge |
| Runtime cost | small per-layer overhead | none |
| Serving | vLLM loads LoRA adapters directly | an ordinary model |

Outputs are identical either way, modulo float rounding.

**Merging is one-way.** Once folded in, the sum is just weights — the adapter cannot be
extracted or swapped without the pristine base to diff against. Keep base and adapter
separate while iterating; merge only for a final serving artifact.

---

## 5. Where the training data comes from

In general, SFT data comes from outside the model — written by humans or a stronger
model. The plan under discussion for this project uses a specific technique instead:
**rejection sampling** (also called expert iteration, or STaR).

1. The model attempts each task N times. Each attempt is a **trajectory**: a long
   conversation of the model issuing shell commands and reacting to the output.
2. The task's test script grades it pass/fail.
3. **Failed trajectories are discarded.** SFT runs on the survivors.

This works here only because terminal tasks ship with automatic pass/fail tests, giving
a free and objective grader. Most domains have no such thing.

### Why failures are excluded

Because the SFT loss can only say "more of this". Training on text increases the
probability of those exact tokens — there is no mechanism to decrease it. Feeding in a
failed trajectory teaches the model to reproduce the failure. Failures are not neutral
in SFT; they are harmful.

Learning from failure requires an objective that can push probability down: DPO or RL
(see the table in §1). **RL uses the failures, SFT throws them away — and RL costs
roughly 100× more compute.** Rejection-sampling SFT is deliberately giving up half the
signal in exchange for fitting on one Spark.

### Why training on its own successes helps at all

The model solves a given task maybe 1 time in 8; that success is partly luck in
sampling. Training on it does not add a capability the model lacked — it **shifts
probability mass**, so the approach that worked once in eight becomes the default.
Luck converted into habit.

Which also marks the limit: **a task with a 0% pass rate produces no signal at all.**
Filling that gap is what distillation from a stronger teacher would be for.

### Loss masking

Each trajectory becomes a conversation — system prompt, assistant turns, tool outputs.
Compute the loss **only on the assistant's tokens**, never on the shell output the
environment fed back.

Getting this wrong is the most common bug in agentic SFT: train on the tool outputs and
you teach the model to hallucinate terminal output instead of producing commands.

### Failures are still kept

They are excluded from the training set, not deleted. They are the raw material for the
failure taxonomy (is this a model problem or a scaffold problem?), and they are exactly
what a future GRPO run would consume.

---

## 6. Common misconceptions

| Claim | Correct |
|---|---|
| "SFT" means soft fine-tuning | **Supervised** fine-tuning. "Soft prompts" is an unrelated PEFT method. |
| Loss = output − input | Loss = `-log p` of the correct next token. How surprised the model was. |
| Training minimizes the gradient | Training minimizes the **loss**. The gradient is the direction. |
| LoRA trains a random subset of weights | LoRA freezes everything and adds new parameters. Nearly all weights change; the change is rank-constrained. |
| Merging a 100 MB adapter gives a 16.1 GB model | 16.0 GB. `B·A` expands to full size before being added. Addition, not concatenation. |
| The adapter contains the training data | It contains ~50M numbers encoding a behavioral tilt. No data is stored. |

---

## 7. Open questions this does not answer

Per §4 of `AGENTS.md`, still undecided:

- **Base model** — family and size, and whether it serves *and* LoRA-trains in 128 GB.
- **Training environments** — hand-written, adapted from public suites, or both.
  Contamination against the Terminal-Bench 2.0 task set is the binding constraint here.
- **Rank, target modules, learning rate** — downstream of the two above.
- **Whether rejection-sampling SFT is the chosen method at all**, versus distillation
  from a stronger teacher, or a combination.
