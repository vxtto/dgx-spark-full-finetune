# AGENTS.md — TCAD LLM Supervised Fine-Tuning

Canonical guide for any coding agent (Claude Code, Codex, Cursor, …) working in this
repository. Claude Code reads `CLAUDE.md`, which points here; keep this file as the
single source of truth and update it when scope or conventions change.

> ## ⚠️ Keep `AGENTS.md` and `CLAUDE.md` in sync
>
> **These two files change together, always.** If you edit one, you edit the other in
> the same change — never one alone, never "I'll update the other later".
>
> - Content present in both must stay **identical in substance**: same rules, same
>   constraints, same decisions. Don't let the two drift into different wordings of
>   different rules.
> - Content that belongs to only one file still requires **reading the other file**
>   before committing, to confirm nothing there now contradicts the edit.
> - A commit that touches only one of the two is incomplete. Check with
>   `git diff --name-only` before committing.
> - The same applies to any other agent instruction file added later.

---

## Status: early — nothing is decided yet

The goal is set; **everything about how to reach it is still open** (§4). Not the
data, not the base model, not the training method, not the stack.

Do not assume a default and do not let an implementation choice quietly settle one of
these questions. An agent's job here is to **lay out options with trade-offs and
wait**, not to pick.

## 1. Project scope

Build a **domain-specialized LLM for TCAD** (Technology Computer-Aided Design) via
**Supervised Fine-Tuning**, trained on an **NVIDIA DGX Spark** located in the United
States and accessed remotely.

Intended capabilities, in priority order:

1. **Code generation** — produce valid TCAD simulation input from a natural-language
   request.
2. **Domain Q&A** — answer questions about TCAD syntax, physics models, meshing, and
   convergence troubleshooting.
3. **Flow assistance** — explain, repair, and adapt existing simulation setups.

Out of scope for now: anything past SFT, surrogate/physics modelling of device
behaviour, agentic execution of simulations, and serving infrastructure beyond a local
inference smoke test.

**Team:** Alfredo Ceci and Vittorio. Every decision in §4 belongs to them.

## 2. Hardware — NVIDIA DGX Spark (remote, US)

Spec notes (verify on the box with `nvidia-smi`, `uname -m`, `free -h` before relying
on them):

- GB10 Grace Blackwell Superchip: 20-core **aarch64** Arm CPU + Blackwell GPU.
- **128 GB coherent unified LPDDR5X** shared CPU/GPU, **~273 GB/s** bandwidth.
- DGX OS (Ubuntu-based), CUDA 13.x, compute capability **sm_121**.

These are the constraints every open decision has to live inside. State them when
weighing options; don't resolve them unilaterally:

- **Capacity is generous, bandwidth is not.** 128 GB of unified memory means a large
  model *fits*; ~273 GB/s means it trains slowly. "Does it fit" is therefore not the
  deciding question — throughput at the chosen sequence length is.
- **Single node, single GPU.** No multi-node sharding; effective batch size comes from
  gradient accumulation.
- **aarch64 + sm_121 breaks many prebuilt wheels.** Whatever stack is chosen, check
  for an aarch64 + CUDA 13 build *before* designing around it. Prefer NVIDIA's
  `nvcr.io` aarch64 containers; when something must be built from source, record the
  exact command in `docs/environment.md`.
- **Unified memory** means a GPU allocation failure can surface as host OOM. Leave
  headroom; don't run heavy data prep concurrently with training.

**Access model:** the laptop (`darwin`, this working directory) is for authoring code,
configs, and docs. All training, evaluation, and GPU-bound work runs on the Spark over
SSH. Never launch a training job locally. Long jobs run under `tmux`/`nohup` with logs
to a file, so a dropped SSH session doesn't kill them.

## 3. Project tracking — GitHub issues and milestones

Work is tracked with **GitHub issues and milestones** on `aceci0127/training-an-LLM`.
There is no separate tracker; if a piece of work isn't an issue, it isn't planned.

- **Issues** are the unit of work — including each open decision in §4, so the options
  and the rationale are captured where the outcome lands.
- **Milestones** group issues into project phases.
- **Read before writing code:** work to the issue's acceptance criteria instead of
  guessing scope, and check for prior discussion before re-deriving a decision.
- Anything uncovered outside the current task becomes a new issue. Link issues from
  commits and PRs (`Refs #12`, `Closes #12`).

Creating, editing, commenting on, or closing an issue or milestone is an
**outward-facing action**: propose the text and **ask before writing**, unless that was
explicitly requested. Use the `gh` CLI.

## 4. Open decisions

**Nothing below has been decided.** When one comes up, present the realistic options
against the §2 constraints, let the team choose, then record the outcome in
`docs/decisions.md`.

- **Data** — what the training corpus is, where it comes from, how it is built, and
  what licensing allows.
- **Base model** — family, size, license.
- **Fine-tuning technique** — full fine-tuning vs. parameter-efficient methods,
  precision, quantization, and the hyperparameters that follow.
- **Training framework and environment** — subject to the aarch64/CUDA-13 constraint.
- **Evaluation** — what "better" means for this model, and how it is measured.
- **Tooling** — experiment tracking, serving/inference path.

## 5. Repository hygiene

This repo holds **code, configs, and docs**. Never commit: model weights, checkpoints,
datasets over a few MB, simulation outputs, `.env` contents, API keys, or licensed
vendor material. Keep the large artifacts on the Spark or in object storage; if a
dataset must be versioned, use Git LFS and say so in `docs/`.

Two invariants that hold regardless of what §4 decides:

- **Evaluation material never enters training data**, in any form or paraphrase.
- **Every run is reproducible** from one config file plus one data snapshot, both
  logged in the run directory and recorded in `docs/experiments.md`. An untracked run
  didn't happen.

## 6. Working agreements for agents

- Code, identifiers, comments, docstrings, and commit messages: **English**.
  Conversation with the team: **Italian**.
- **`AGENTS.md` and `CLAUDE.md` are edited together** — see the sync rule at the top.
- **Propose, don't decide.** On anything in §4, present options with trade-offs and
  wait. Writing code that only works under one unstated choice counts as deciding.
- Don't run training, large downloads, or GPU work from the laptop. Propose the
  command for the Spark instead.
- When adding a dependency, check it has an aarch64 + CUDA 13 build first.
