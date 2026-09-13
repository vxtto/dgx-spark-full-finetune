# CLAUDE.md

Project-specific instructions for Claude Code.

> ## ⚠️ Keep `CLAUDE.md` and `AGENTS.md` in sync
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

**Read `AGENTS.md` first — it is the canonical guide.** This file only adds what is
specific to working through Claude Code; when the two disagree, `AGENTS.md` wins and
should be corrected.

@AGENTS.md

## Context

We are building a domain-specialized LLM for **TCAD** via **supervised fine-tuning**,
trained on an **NVIDIA DGX Spark located in the US** and reached over SSH. The repo
holds code, configs, and docs — never weights, checkpoints, or large datasets. The
project is run by Alfredo and Vittorio.

## The project is early — propose, don't decide

**Nothing is decided yet**: not the data, not the base model, not the fine-tuning
technique, not the framework, not the evaluation (`AGENTS.md` §4).

- Never fill these in with a default, and never write code that only works under one
  unstated choice — that is deciding by implementation.
- When a decision is needed to move forward, lay out the realistic options against the
  DGX Spark constraints (`AGENTS.md` §2), give the trade-offs, and **stop and ask**.
- Once something is decided, record it in `docs/decisions.md`.

## Communication

- Reply to the user in **Italian**. Code, identifiers, comments, docstrings, file
  names, and commit messages stay in **English**.
- Be concrete about GPU cost: before proposing a run, say roughly how long it will
  occupy the Spark.

## Ticketing — GitHub issues and milestones

Work is tracked with **GitHub issues and milestones** on `aceci0127/training-an-LLM`
(see `AGENTS.md` §3). There is no other tracker.

- Look up the issue behind the request and work to its acceptance criteria rather than
  inferring scope.
- Each open decision should have its own issue — that is where the options, the
  trade-offs, and the final rationale belong.
- Anything uncovered outside the current task becomes a new issue rather than silent
  scope creep. Reference issues from commits and PRs (`Refs #12`, `Closes #12`).

Use the `gh` CLI. Creating, editing, commenting on, or closing an issue or milestone
is outward-facing: **draft the text and ask first**, unless that was explicitly
requested.

## Execution rules

- **Never start training, large model downloads, or GPU work on this laptop.** Author
  the code and config here; hand over (or run over SSH, when configured) the command
  for the Spark.
- Long remote jobs go under `tmux`/`nohup` with logs to a file, so a dropped SSH
  session doesn't kill them.
- Use the session scratchpad for throwaway scripts and exploratory output — not the
  project tree.
- Before adding a Python dependency, verify it has an **aarch64 + CUDA 13 (sm_121)**
  build; if it needs building from source, record the command in `docs/environment.md`.

## Guardrails

- Never print, echo, or commit `.env`, API keys, or licensed vendor material.
- Never commit weights, checkpoints, simulation outputs, or datasets over a few MB.
- Never mix evaluation material into training data.
- Ask if a data source's licensing is unclear.

## Conventions

- One config file per run under `configs/`; no hyperparameters hardcoded in scripts.
- Extend an existing pipeline stage rather than creating `*_v2` / `*_final` variants.
- Log each run in `docs/experiments.md` — config, data snapshot, wall time, metrics,
  and what it was testing.
