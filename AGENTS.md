# AGENTS.md — TCAD LLM Supervised Fine-Tuning

Canonical guide for any coding agent (Claude Code, Codex, Cursor, …) working in this
repository. Claude Code reads `CLAUDE.md`, which points here; keep this file as the
single source of truth and update it when scope, stack, or conventions change.

**Status: early. Nothing about the model or the training method has been chosen yet.**
See §9 — do not assume defaults, do not write code that silently commits the project
to one of the open decisions.

---

## 1. Project scope

Build a **domain-specialized LLM for TCAD** (Technology Computer-Aided Design,
primarily the Synopsys Sentaurus flow) via **Supervised Fine-Tuning (SFT)**, trained
on an **NVIDIA DGX Spark** located in the United States and accessed remotely.

Target capabilities of the fine-tuned model, in priority order:

1. **Code generation** — turn a natural-language device/simulation request into valid
   Sentaurus command files: `sde_dvs.cmd` (structure + meshing, Scheme/TCL-like SDE
   syntax), `sdevice.cmd` / `*_des.cmd` (physics, solve blocks, Id-Vg / Id-Vd / C-V
   sweeps), `sdevice.par` (parameter files), and the `run.sh` driver.
2. **Domain Q&A** — answer single-fact and explanatory questions about TCAD syntax,
   physics models, meshing strategy, and convergence troubleshooting.
3. **Flow assistance** — explain, repair, and adapt existing decks (change device
   architecture, doping profile, gate length, bias sweep) with multi-turn edits.

Non-goals for v1: preference-optimization stages after SFT (leave data hooks for it,
build nothing), surrogate/physics modelling of device characteristics, agentic
execution of simulations, serving infrastructure beyond a local inference smoke test.

**Team:** Alfredo Ceci and Vittorio. Every design decision listed in §9 belongs to
them — an agent proposes options and trade-offs, it does not pick.

## 2. Reference work

Two public repositories define the shape of the data and the evaluation. We
reimplement the ideas; we do **not** vendor their code wholesale.

**`wddddds1/TcadGPT`** — pipelines and eval sets for a TCAD/Elmer domain model:
- `tcad/scripts/` — QA synthesis from documentation: keyword extraction
  (`kaywords_gen_V6.py`) then Alpaca-format QA generation from those keywords
  (`data_gen_from_keywords_*.py`, `data_gen_parallel_v6-general.py`).
- `tcad/IR_DPO/tcad_coder/` — code-side pipeline: split decks into lines/blocks
  (`0-code_split*.py`), synthesize instructions at **line → block → full-cmd** level
  (`1-`, `2-`, `3-`), augment, adjust code, add comments, and build
  **multi-turn long-code** samples (`6-`).
- `tcad/QA_test/` — 100-question manually written single-fact QA test set plus
  baseline outputs (GPT-4o, DeepSeek V3/R1, llama3.1:8b, ±RAG) for comparison.
- `tcad/code_test/` — instruction→code pairs as `N.txt` (instruction) + `N.cmd`
  (reference deck).

Ideas worth borrowing (still to be evaluated, not adopted by decree): the
**three-granularity instruction synthesis** (line, block, whole deck), the
**IR (intermediate representation) → instruction** round-trip with a consistency
check, and a **frozen QA test set kept separate from training data**.

**`guangxifan/llm4tcad_flow`** — 22 validated Sentaurus baseline decks (2D planar
n/p MOSFET ± LDD ± Gaussian doping; 3D FinFET, SiGe FinFET, InGaAs FinFET,
junction-less FinFET, tunnel FinFET, rectangular/cylindrical nanowire and nanosheet
GAAFET, FinFET/nanosheet arrays), each with `sde_dvs.cmd`, `IdVg_des.cmd`,
`IdVd_des.cmd`, `CV_des.cmd`, `sdevice.par`, `run.sh`, `readme.txt` and
`readme4llm.txt`. These are the most likely **seed corpus** for code-side SFT: real,
convergent, syntactically diverse decks with an LLM-oriented description attached.

Note the mesh caveat in that repo: meshes are tuned for short-channel devices; do not
present them as general-purpose for long-channel geometries in generated content.

## 3. Hardware — NVIDIA DGX Spark (remote, US)

Spec notes (verify on the box with `nvidia-smi`, `uname -m`, `free -h` before relying
on them):

- GB10 Grace Blackwell Superchip: 20-core **aarch64** Arm CPU (10× Cortex-X925 +
  10× Cortex-A725) + Blackwell GPU, 5th-gen Tensor Cores, FP4 support.
- **128 GB coherent unified LPDDR5X** shared CPU/GPU, **~273 GB/s** bandwidth.
- ~1 PFLOP sparse FP4; NVMe local storage; ConnectX-7 200 GbE for pairing two units.
- DGX OS (Ubuntu-based), CUDA 13.x, compute capability **sm_121**.

These are the constraints any model/method choice has to live inside — state them
when the team weighs options, don't resolve them unilaterally:

- **Capacity is generous, bandwidth is not.** 128 GB of unified memory means a large
  model *fits*; ~273 GB/s means it trains slowly. Memory footprint and throughput
  point at different answers here, so "does it fit" is not the deciding question —
  tokens/second at the chosen sequence length is.
- **It is a single-node, single-GPU machine.** No multi-node sharding; effective batch
  size comes from gradient accumulation. Unless a second unit is paired over
  ConnectX-7, scale-out is not an option.
- **aarch64 + sm_121 breaks many prebuilt wheels** (bitsandbytes, flash-attn,
  xformers, vllm). Whatever stack is chosen, check for an aarch64 + CUDA 13 build
  *before* designing around it. Prefer NVIDIA's `nvcr.io` aarch64 containers and the
  DGX Spark playbooks over `pip install` from PyPI; when something must be built from
  source, record the exact command in `docs/environment.md`.
- **Unified memory** means a GPU allocation failure can surface as host OOM. Leave
  headroom; don't run heavy data prep concurrently with training.

**Access model:** the laptop (`darwin`, this working directory) is for authoring code,
data pipelines, configs, and docs. All training, evaluation, and GPU-bound work runs
on the Spark over SSH. Never launch a training job locally. Long jobs run under
`tmux`/`nohup` with logs written to a file, so a dropped SSH session doesn't kill them.

## 4. Stack and tools

**Undecided.** Base model, fine-tuning technique, training framework, data-synthesis
model, serving path, and experiment tracking are all open (§9). Nothing in this repo
should hardcode one of them, and no agent should introduce one by default.

What *is* fixed:

- The project tree holds **code, data pipelines, configs, and docs** — never weights,
  checkpoints, or large datasets.
- Data on disk is **JSONL with provenance fields** (§6). The prompt/chat format the
  trainer consumes depends on the base model's template and is decided with it.
- The laptop side uses standard Python tooling; the Spark side is expected to run in a
  container for the aarch64/CUDA-13 reasons above.

When a decision from §9 is made, record it here with a one-line rationale and move it
out of §9.

### Project tracking — GitHub issues and milestones

Work is tracked with **GitHub issues and milestones** on the project repository
(`aceci0127/training-an-LLM`). There is no separate tracker; if a piece of work isn't
an issue, it isn't planned.

- **Issues** are the unit of work: a pipeline stage, an experiment to run, an eval to
  build, an environment blocker, or one of the open decisions in §9. Each open
  decision should exist as its own issue so the options and the rationale are captured
  where the outcome lands.
- **Milestones** group issues into the phases of the project (for example: corpus and
  data pipeline, evaluation harness, model/method decision, first training runs). Use
  them to see what the current phase actually depends on.
- **Read before writing code.** Look up the issue behind a request and work to its
  acceptance criteria instead of guessing scope; check for prior discussion before
  re-deriving a decision.
- **Close the loop.** When work uncovers something outside the current task (a broken
  deck, a missing eval, a dependency that won't build on aarch64), it becomes a new
  issue. Run outcomes are reported on the issue that requested them, and link the
  relevant issue from commits and PRs (`Refs #12`, `Closes #12`).

Creating, editing, commenting on, or closing an issue or milestone is an
**outward-facing action**: propose the text and **ask before writing**, unless that was
explicitly requested in the current task. Use the `gh` CLI for this.

## 5. Repository layout (target)

```
configs/            # one config per run: data mix, method settings, schedule, eval
data/
  raw/              # source decks, docs, scraped material — never edited in place
  interim/          # splits, IR, keyword extractions, generation intermediates
  processed/        # final train/val JSONL, the only thing the trainer reads
  eval/             # frozen test sets — QA and instruction→code; never trained on
src/
  data/             # corpus ingest, deck splitting, IR round-trip, synthesis, dedup
  train/            # training entrypoint
  eval/             # QA scoring, code pass@1, syntax linting
  serve/            # inference smoke tests
scripts/            # thin CLI wrappers + remote launch helpers for the Spark
docs/               # environment.md, data-card.md, decisions.md, experiments.md
```

Not in git: model weights, checkpoints, datasets over a few MB, `.env`, simulation
outputs. Keep them on the Spark or in object storage; if a dataset must be versioned,
use Git LFS and say so in `docs/data-card.md`.

The data pipeline is deliberately **upstream of the model choice** — building the
corpus and the eval sets does not require knowing which base model or technique wins,
so that work can start first. Keep it that way: no tokenizer- or template-specific
logic in `src/data/` beyond a final, swappable formatting step.

## 6. Data conventions

- **Provenance is mandatory.** Every generated record carries `source` (file or doc it
  derives from), `pipeline` (which generator and version), and `granularity`
  (`line` | `block` | `deck` | `qa` | `multi_turn`). Records without provenance don't
  ship.
- **Train/eval separation is absolute.** Decks or documentation sections that feed the
  eval sets never appear in training data, in any granularity or paraphrase. Dedup
  across the split boundary (exact + near-duplicate, e.g. MinHash) and fail the build
  if a collision is found.
- **Licensing.** The Sentaurus toolchain and its documentation are proprietary
  Synopsys material. Public example decks and our own authored content go in the repo;
  vendor manuals, licensed docs, and anything under NDA do not. If unsure about a
  source, ask before ingesting it.
- Prefer real, executable decks over synthetic ones. Synthetic augmentation must be
  validated at least syntactically before it enters `data/processed/`.
- `readme4llm.txt` files from the baseline repo are the natural instruction seeds for
  whole-deck samples — use them rather than re-deriving descriptions.
- **Measure the corpus before choosing anything.** Token-length distribution of the
  decks, number of usable samples per granularity, and duplication rate are inputs to
  the model and method decision. Produce those numbers early.

## 7. Training conventions

Method-agnostic rules that hold whatever the team picks:

- Every run is reproducible from **one config file + one data snapshot hash**. Log both
  into the run directory.
- **Loss on completions only** — mask the prompt. A model that learns to echo
  instructions is a silent failure mode here.
- **Check the truncation rate** against the corpus length distribution before
  committing to a sequence length. TCAD decks are long; heavy truncation invalidates
  deck-level samples regardless of the technique used.
- Always run a **tiny overfit sanity check** (a handful of samples, loss → ~0) before
  spending real GPU hours.
- Record every run in `docs/experiments.md`: config, data version, wall time, final
  metrics, and what it was testing. One line per run is enough; an untracked run
  didn't happen.
- The first runs exist to **inform the open decisions**, not to produce a final model.
  Design them as comparisons (same data, same eval, one variable) rather than as
  attempts at a best result.

## 8. Evaluation

Three layers, in increasing cost. Build these early — they are what makes the §9
decisions answerable with evidence instead of opinion.

1. **Syntax/static** — generated decks must parse: balanced parentheses in SDE
   Scheme-like blocks, required sections present (`File`, `Electrode`, `Physics`,
   `Solve`), referenced files and contacts consistent. Runs on the laptop, no license
   needed. This is the fast gate.
2. **QA accuracy** — the frozen single-fact QA set, scored by an LLM judge with a
   fixed rubric and a fixed judge model. Compare against meaningful baselines (the
   untuned base model, base + RAG, a frontier model) so the numbers mean something.
3. **Execution pass@1** — run the generated deck through Sentaurus and check
   convergence and output curves. Requires a licensed toolchain; gate it behind a flag
   and don't assume it's available.

Before judging, strip Markdown fences and prose from model output — raw generations
need cleaning to be executable.

## 9. Open decisions

**Nothing below has been decided.** Do not assume a default, do not let a code choice
quietly settle one. When one of these comes up, lay out the options and the trade-offs
under the §3 constraints and let the team choose; then record the outcome in
`docs/decisions.md` and update §4.

- **Base model** — family, size, license, whether a code-specialized base beats a
  general one for this corpus.
- **Fine-tuning technique** — full fine-tuning vs. parameter-efficient methods vs.
  anything else; precision and quantization; and the hyperparameters that follow.
- **Training framework and environment** — subject to the aarch64/CUDA-13 constraint.
- **Data synthesis** — whether instruction data is generated by an API model, a local
  model, or written by hand; which model; what budget.
- **Sentaurus licensing** — whether a licensed toolchain is reachable from the Spark.
  This decides whether execution-based evaluation (§8.3) is possible at all, so it is
  worth resolving early.
- **Experiment tracking**, **serving/inference path**, and whether a preference-
  optimization stage follows SFT at all.

## 10. Working agreements for agents

- Code, identifiers, comments, docstrings, and commit messages: **English**.
  Conversation with the team: **Italian**.
- **Propose, don't decide.** On anything in §9, present options with trade-offs and
  wait. Writing code that works only under one unstated choice counts as deciding.
- Don't run training, large downloads, or GPU work from the laptop. Propose the
  command for the Spark instead.
- Don't print or commit `.env` contents, API keys, or license files.
- Prefer editing an existing pipeline stage over adding a parallel one; this repo
  should not accumulate `*_v2` / `*_final` script variants the way the reference repos
  did.
- When adding a dependency, check it has an aarch64 + CUDA 13 build before writing
  code against it.
