# Decisions

Decision record for the open items in `AGENTS.md` §4. One entry per decision:
what was chosen, the alternatives, and the reason. A decision that is not written
here has not been made.

Status values: **proposed** (awaiting the other half of the team) → **accepted** →
**superseded by D-xx**.

---

## D-01 — Reference work: reproduce the TMAX SFT recipe

**Status:** accepted · 2026-09-15 · Refs #5

TMAX, *"A simple recipe for terminal agents"* (Ivison, Yin, Shao, Xiao, Lambert,
Hajishirzi — Ai2 / UW), [arXiv:2606.23321](https://arxiv.org/abs/2606.23321),
code at <https://github.com/hamishivi/tmax>. Local copy:
`docs/documentation/2606.23321v1.pdf`.

We reproduce the paper **up to the SFT portion only**. The paper's headline result is
an RL recipe; a 9B RL run is out of budget (#5).

---

## D-02 — Base model: `Qwen/Qwen3-8B`

**Status:** accepted · 2026-09-15 · Refs #3

**Chosen:** Qwen3-8B (the post-trained instruct release, not `-Base`).

**Why, and this is the load-bearing reason:** the paper's Table 7 measures SFT on both
candidate families, and they behave in opposite directions.

| Model | TB Lite | TB 2.1 |
|---|---|---|
| Qwen3.5-9B | 41.9 | 16.1 |
| + TMAX SFT | 35.5 ↓ | 15.0 ↓ |
| + large SFT | 31.3 ↓ | 16.9 ≈ |
| **Qwen3-8B** | 7.3 | 1.1 |
| + TMAX SFT | 11.5 ↑ | **6.0 ↑** |
| + large SFT | **16.4 ↑** | **7.9 ↑** |

Paper §5.1 is titled *"Qwen 3.5 does not benefit as much from existing SFT datasets"* —
it has already been extensively post-trained, and further SFT degrades it. §3.3 states
the TMAX SFT data is used "for SFT experiments and finetuning Qwen 3 8B".

So Qwen3-8B is the only one of the two where the method under study is known to work,
and it is the configuration whose numbers we can check ourselves against.

**Rejected — Qwen3.5-9B:** higher starting point, but reproducing it means reproducing
a *negative* result. Worth revisiting only once the pipeline is trusted, as a
deliberate replication of §5.1.

**Note:** "Qwen3.5-8B" does not exist. The Qwen3.5 family in the paper's Figure 1 is
2B / 4B / 9B. Qwen3-8B is a separate, earlier family.

---

## D-03 — Training method: LoRA SFT (bf16), not full fine-tuning

**Status:** proposed — Alfredo asked this in #6 and Vittorio has not answered.

**Proposed:** LoRA, bf16, no quantization.

- Full fine-tuning an 8B model needs ~128 GB before activations
  (`docs/theory/lora-sft.md` §2) — the Spark's entire memory, leaving nothing to
  compute with. LoRA brings this to ~20 GB.
- The paper full-fine-tunes across 32 GPUs (see D-05). We have one device.
- `docs/theory/behaviour-vs-knowledge.md` §5 argues Terminal-Bench failures are
  *behavioural*, the regime LoRA is strongest in. **That is a hypothesis, not a
  measurement** — it is what the first run tests.

**Open sub-question for Vittorio (#6):** whether to accept LoRA as the method, or run
one full-FT attempt to measure the gap. The second costs roughly 4× the memory and
rules out the 8B model on this hardware.

---

## D-04 — Stack: `hamishivi/tmax` (fork of `allenai/open-instruct`)

**Status:** accepted · 2026-09-15 · supersedes the `trl` baseline in `requirements.txt`

**Chosen:** the tmax fork — "a fork of open-instruct with fixes for Qwen 3.5 and
terminal-agent training". Training entry point `open_instruct/finetune.py`,
config via YAML (`ArgumentParserPlus.parse_yaml_and_args`).

**Why over HF TRL + PEFT:**
- it is the authors' own code, so D-01 becomes a reproduction rather than a
  re-implementation from a paper description;
- `--use_lora` is supported natively (`lora_rank` / `lora_alpha` / `lora_dropout`);
- Nathan Lambert, a paper co-author, publishes a DGX Spark setup guide that runs
  open-instruct on this exact hardware.

**Why it survives the aarch64 / CUDA 13 constraint** — this was the main risk and it
checks out (see `docs/environment.md`):
- open-instruct's `pyproject.toml` marks `flash-attn` as *not Darwin/aarch64*, so it is
  never built on ARM;
- `detect_attn_implementation()` in `open_instruct/model_utils.py` falls back cleanly to
  `AttentionBackendName.torch` (SDPA) when no flash-attn is present — no patch needed;
- the pyproject already ships a CUDA 13.0 package index.

**Consequence:** we diverge from the paper's Table 14, which specifies flash attention 3.
Attention backend is therefore an uncontrolled difference from the reference run.

**Still to verify on the box:** `deepspeed>=0.18.3` and `vllm>=0.19.1` on aarch64.

**Rejected:** TRL + PEFT (re-implementation, not reproduction); Unsloth, Axolotl,
LLaMA-Factory (same objection, plus unverified sm_121 support).

---

## D-05 — Hyperparameters: Table 14 adapted from 32 GPUs to one Spark

**Status:** proposed — the values are starting points, not measurements.

The paper's SFT recipe is a **32-GPU full fine-tune**: global batch 128 =
per-device 1 × grad-accum 4 × 32 devices. Six of its parameters cannot be carried over
unchanged.

| Parameter | Paper (Table 14) | Ours | Why it changed |
|---|---|---|---|
| Precision | bf16 | bf16 | — |
| Attention | flash attention 3 | SDPA | no flash-attn on aarch64 (D-04) |
| Gradient checkpointing | true | true | — |
| Max sequence length | 32768 (Qwen 3) | 16384 | memory and wall-clock on one device |
| Epochs | 2 | 2 | — |
| Per-device batch size | 1 | 1 | — |
| Grad accumulation | 4 | 32 | compensates for 1 device instead of 32 |
| Global batch | 128 | 32 | full 128 would multiply step time by 4 |
| Learning rate | 2e-5 | 1e-4 | 2e-5 is a full-FT rate; LoRA needs ~5× more |
| LR scheduler | linear | linear | — |
| Warmup ratio | 0.03 | 0.03 | — |
| Weight decay | 0.0 | 0.0 | — |
| Optimizer | AdamW | AdamW | — |

LoRA shape is **not** from the paper (it does not use LoRA). Starting at
`r=16, alpha=32, dropout=0.05` rather than open-instruct's defaults
(`r=64, alpha=16` — a 0.25 scaling factor, which is unusual), following the low-rank
argument in `docs/theory/behaviour-vs-knowledge.md` §5. **Untested hypothesis.**

---

## D-06 — SFT dataset

**Status:** proposed

**Chosen:**
`hamishivi/tmax-sft-skill-tax-20260505-2.2k-combined-balanced-qwen3.6-27b-thinking-no-tool-call`

This matches paper §3.3 exactly: 2.2k environments × 8 trajectories with Qwen 3.6 27B as
teacher ≈ 16.5k trajectories. The **`-no-tool-call`** variant exists because of §3.3
footnote 2 — the traces are stripped of the Qwen 3 XML tool-call format precisely
because it "can confuse small models with different tool call formats like Qwen 3 8B".
Given D-02, that is the variant built for our model.

Two configs ship in the repo, and the choice between them is a real experiment:

| Config | Contents | |
|---|---|---|
| `..._thinking_all` | all ~16.5k trajectories | what the paper uses |
| `..._thinking_only_success` | ~8k successful only | rejection sampling |

**Discrepancy worth resolving.** Paper §D.2: *"we do not filter out
unsuccessful/incomplete rollouts from our dataset"*. Our own
`docs/theory/lora-sft.md` §5 states the opposite — that failed trajectories are
discarded because SFT can only say "more of this". Both cannot be right for this run.
The two configs make it a cheap ablation; it should be run rather than argued.

`allenai/TMax-15K` is the released *environment/task* set, not SFT trajectories — it is
for data generation and RL, not for this pipeline.

**Contamination:** the paper reports 0% 13-gram overlap with Terminal-Bench and TB-Lite
(§3.3, Table 11). We inherit that, and must not weaken it if we extend the data (#4).

---

## D-07 — Evaluation

**Status:** open — nothing decided. Milestone *"Epic: configure evaluation harness"*.

Note from paper Table 5: measured score varies by **8 points** on the same model
depending on harness (Qwen3.5-9B scores 36.0 to 44.1 across four). A baseline number is
meaningless until the harness is pinned. Baseline must be measured before training,
not after.
