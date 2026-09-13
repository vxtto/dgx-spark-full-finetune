# CLAUDE.md

Project-specific instructions for Claude Code.

**Read `AGENTS.md` first — it is the canonical guide** to this project's scope,
hardware, data pipeline, and conventions. This file only adds what is specific to
working through Claude Code; when the two disagree, `AGENTS.md` wins and should be
corrected.

@AGENTS.md

## One-paragraph context

We are building a domain-specialized LLM for **TCAD** (Synopsys Sentaurus flow) via
**supervised fine-tuning**, trained on an **NVIDIA DGX Spark located in the US** and
reached over SSH. The corpus is TCAD command files (`sde_dvs.cmd`, `*_des.cmd`,
`sdevice.par`) and TCAD documentation, turned into instruction data drawing on the
approaches in `wddddds1/TcadGPT` and seeded with the 22 validated baseline decks in
`guangxifan/llm4tcad_flow`. This repo holds code, data pipelines, and configs — **not**
weights, checkpoints, or large datasets. The project is run by Alfredo and Vittorio.

## The project is early — propose, don't decide

**No technical choice has been made yet**: not the base model, not the fine-tuning
technique, not the framework, not the data-synthesis model (`AGENTS.md` §9).

- Never fill these in with a default, and never write code that only works under one
  unstated choice — that is deciding by implementation.
- When a decision is needed to move forward, lay out the realistic options against the
  DGX Spark constraints (`AGENTS.md` §3), give the trade-offs, and **stop and ask**.
- Start with the work that is genuinely upstream of these decisions: corpus ingest,
  deck splitting, eval sets, corpus measurements (token-length distribution, sample
  counts, duplication). Those numbers are what makes the decisions answerable.
- Once something is decided, record it in `docs/decisions.md` and move it out of §9.

## Communication

- Reply to the user in **Italian**. Code, identifiers, comments, docstrings, file
  names, and commit messages stay in **English**.
- Be concrete about GPU cost: before proposing a run, say roughly how long it will
  occupy the Spark.

## Execution rules

- **Never start training, large model downloads, or GPU work on this laptop.** Author
  the code and config here; hand over (or run over SSH, when configured) the command
  for the Spark.
- Long remote jobs go under `tmux`/`nohup` with logs to a file, so a dropped SSH
  session doesn't kill them.
- Use the session scratchpad for throwaway scripts, intermediate dumps, and
  exploratory output — not the project tree.
- Before adding a Python dependency, verify it has an **aarch64 + CUDA 13 (sm_121)**
  build; if it needs building from source, record the command in `docs/environment.md`.

## Guardrails

- Never print, echo, or commit `.env`, API keys, or Synopsys license material.
- Never commit weights, checkpoints, simulation outputs, or datasets over a few MB.
- Never mix evaluation material into training data — dedup across the split boundary
  and fail loudly on a collision.
- Don't ingest proprietary Sentaurus documentation into the repo; ask if a source's
  licensing is unclear.

## Conventions

- One config file per run under `configs/`; no hyperparameters hardcoded in scripts.
- Keep `src/data/` independent of any specific model or tokenizer, apart from a final
  swappable formatting step.
- Extend an existing pipeline stage rather than creating `*_v2` / `*_final` variants.
- Every generated data record carries `source`, `pipeline`, and `granularity`.
- Log each run in `docs/experiments.md` — config, data snapshot, wall time, metrics,
  and what it was testing.
