# Agent instructions — LLM Training on DGX Spark

Guide for any coding agent (Claude Code, Codex, Cursor, …) working in this repository.
`AGENTS.md` and `CLAUDE.md` are **two identical copies of this same document** — one
source of truth, two file names, because different tools look for different names.

> ## ⚠️ Keep `AGENTS.md` and `CLAUDE.md` identical
>
> **The two files must be byte-for-byte the same.** There is no Claude-specific half
> and no canonical half: every rule below applies to every agent.
>
> - Edit one file, then immediately copy it over the other:
>   `cp AGENTS.md CLAUDE.md` (or the reverse). Never hand-edit both.
> - Verify before committing: `diff AGENTS.md CLAUDE.md` must print nothing, and
>   `git diff --name-only` must list both files.
> - A commit that touches only one of the two is incomplete.
> - The same applies to any other agent instruction file added later.

---

## Status: early — nothing is decided yet

One sentence has been decided: **we want to train an LLM on the DGX Spark.**
Everything else is open (§4) — the domain and use case, the data, the base model, the
training method, the stack, and how success is measured.

Do not assume a default and do not let an implementation choice quietly settle one of
these questions. An agent's job here is to **lay out options with trade-offs and
wait**, not to pick.

## 1. Project scope

**Train an LLM on an NVIDIA DGX Spark** located in the United States and accessed
remotely.

That is the entire scope as it stands today. What the model should be good at, what it
trains on, and how it is trained are open questions (§4), not omissions from this
document — treat them as such.

**Team:** Alfredo Ceci and Vittorio. Every decision in §4 belongs to them.

## 2. Hardware — NVIDIA DGX Spark (remote, US)

GB10 Grace Blackwell Superchip: **aarch64** Arm CPU + Blackwell GPU, **128 GB unified
LPDDR5X** at **~273 GB/s**, DGX OS, CUDA 13.x, compute capability **sm_121**. Verify
on the box (`nvidia-smi`, `uname -m`, `free -h`) before relying on these numbers.

What it means in practice:

- **Big models fit; they don't run fast.** Memory is generous, bandwidth is not, so
  throughput — not "does it fit" — decides what is feasible.
- **Single node, single GPU.** No multi-node sharding.
- **aarch64 + sm_121 breaks many prebuilt wheels.** Check for an aarch64 + CUDA 13
  build before designing around a library; prefer NVIDIA's `nvcr.io` containers.
- **The laptop is for authoring only.** All training and GPU work runs on the Spark
  over SSH, under `tmux`/`nohup` with logs to a file.

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

- **Domain and use case** — what the model is being specialized for, and what it
  should be able to do.
- **Data** — what the training corpus is, where it comes from, how it is built, and
  what licensing allows.
- **Base model** — family, size, license.
- **Training method** — the kind of training itself, then the technique within it
  (full fine-tuning vs. parameter-efficient methods), precision, quantization, and the
  hyperparameters that follow.
- **Training framework and environment** — subject to the aarch64/CUDA-13 constraint.
- **Evaluation** — what "better" means for this model, and how it is measured.
- **Tooling** — experiment tracking, serving/inference path.

## 5. Repository hygiene and guardrails

This repo holds **code, configs, and docs**. Keep large artifacts on the Spark or in
object storage; if a dataset must be versioned, use Git LFS and say so in `docs/`.

- Never commit model weights, checkpoints, or datasets over a few MB.
- Never print, echo, or commit `.env` contents, API keys, or licensed material.
- Ask if a data source's licensing is unclear.
- **Evaluation material never enters training data**, in any form or paraphrase.
- **Every run is reproducible** from one config file plus one data snapshot, both
  logged in the run directory and recorded in `docs/experiments.md`. An untracked run
  didn't happen.

## 6. Execution rules and conventions

- **Never start training, large model downloads, or GPU work on the laptop.** Author
  the code and config locally; hand over (or run over SSH, when configured) the command
  for the Spark.
- Long remote jobs go under `tmux`/`nohup` with logs to a file, so a dropped SSH
  session doesn't kill them.
- Before adding a Python dependency, verify it has an **aarch64 + CUDA 13 (sm_121)**
  build; if it needs building from source, record the command in `docs/environment.md`.
- One config file per run under `configs/`; no hyperparameters hardcoded in scripts.
- Extend an existing pipeline stage rather than creating `*_v2` / `*_final` variants.
- Use a scratch directory for throwaway scripts and exploratory output — not the
  project tree.

## 7. Working agreements

- Code, identifiers, comments, docstrings, file names, and commit messages: **English**.
- **`AGENTS.md` and `CLAUDE.md` are edited together and stay identical** — see the sync
  rule at the top.
- **Propose, don't decide.** On anything in §4, present options with trade-offs and
  wait. Writing code that only works under one unstated choice counts as deciding.
- Be concrete about GPU cost: before proposing a run, say roughly how long it will
  occupy the Spark.
