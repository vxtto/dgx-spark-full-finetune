# Behaviour vs. knowledge — what post-training can actually teach

Background notes on what an SFT run is capable of changing, and what it is the wrong
tool for. Companion to `lora-sft.md`. Written 2026-09-13.

> **This is theory, not a decision.** Nothing here settles any open item in §4 of
> `AGENTS.md`. It bears directly on the open **data** and **evaluation** decisions —
> it argues for a diagnostic step (§3) before any dataset is built — but the choice of
> whether to take that step belongs to the team.

---

## 1. Mechanically there is no difference

Both are gradient descent on `-log p(next token)`. There is no knowledge module and no
behaviour module in the network. Every SFT example does the same thing: make the tokens
in the target more probable in that context.

The distinction is not in the machinery. It is in **what the update has to accomplish**.

---

## 2. The dividing line

One question about anything you want to teach:

> **Was it already in the weights before training?**

**Yes → behaviour.** You are re-weighting a choice among options the model already has.
`ls -la` is this. The model saw `-la` a million times in pretraining, knows what `-l`
and `-a` do, and can explain them on request. What it lacks is the *policy* of reaching
for them. Nothing is being added; probability mass is moving between candidates that
already exist.

**No → knowledge.** The association exists nowhere in the network — an internal tool's
flag, an API renamed after the cutoff. There is no prior to lean on. You are carving a
new association into weights that were never shaped by it.

So `ls -la` is knowledge *and* behaviour — but the knowledge half is already present.
The gap is retrieval and deployment, not storage.

---

## 3. The diagnostic

Before building any dataset, probe the base model **outside** the agentic context:

```
"What does the -la flag do for ls?"
"How do you list hidden files including permissions?"
```

Answers correctly → the failure observed in the trajectory was **not** a knowledge
failure. It was a policy failure.

This changes what training is supposed to fix, and whether training is the right tool at
all. Worth running against a sample of failed trajectories before committing to a data
pipeline.

---

## 4. Why the two behave differently

| | Behaviour (re-weighting) | Knowledge (new association) |
|---|---|---|
| Data needed | tens to a few hundred examples | many paraphrases, repeated; still unreliable |
| LoRA rank needed | low — r=8–16 is plenty | high, and still a poor fit |
| Generalizes? | yes, along "situation" — teaching `ls -la` may also produce `df -h`, because a disposition toward informative output was reinforced | no — fact A says nothing about fact B |
| Half-learned looks like | does it sometimes; degrades gracefully | **confident wrong answer** — interpolates to the nearest known neighbour |
| Right tool | SFT / LoRA | RAG, or context |

The fourth row is the important one. A half-learned behaviour is a model that is
inconsistent. A half-learned *fact* is a model that hallucinates a plausible neighbour
with full confidence — worse than never having trained.

There is published work pointing this way (Gekhman et al., roughly *"Does fine-tuning
LLMs on new knowledge encourage hallucinations?"*) reporting that SFT on facts the model
did not already know measurably increases hallucination rate. **Recalled, not verified —
check the citation before relying on it.**

### Why the asymmetry exists

**Knowledge has a truth value; behaviour has a fitness value.** A flag either is or is
not the real flag — the world decides. Preferring `-la` is not true or false, only
better or worse for the goal. You cannot be *wrong* about a preference, so a partially
learned preference stays harmless. A partially learned fact is simply an error.

---

## 5. What this implies for Terminal-Bench

Most expected Terminal-Bench failures are **behaviour** failures:

- does not read the error message before retrying
- gives up after two attempts instead of eight
- does not verify that the file it wrote exists
- forgets to `cd` back
- writes a command that would work, then does not check the exit code

The model knows `tar` syntax and knows what exit code 1 means. What it lacks is the
disposition to check, persist, and verify — a policy over knowledge it already holds.

This is precisely the regime where LoRA SFT on self-generated successes is strong, and
precisely why rejection sampling fits: **successful trajectories are demonstrations of
good policy, not carriers of new facts.**

It also bears on rank. If the target is disposition rather than content, little capacity
is needed. Low rank is not a compromise here — it is the right size for the job. This is
a hypothesis to test, not a settled parameter.

---

## 6. The spectrum

It is not binary.

| Example | Where it sits |
|---|---|
| "Emit tool calls in this JSON shape" | pure behaviour — format only |
| "Prefer `ls -la`" | behaviour — knowledge exists, policy does not |
| "Check the exit code before continuing" | behaviour — a disposition, generalizes widely |
| "In this repo run `make test`, not `pytest`" | **procedural knowledge** — a genuine fact, deployed as policy |
| "Library X v3 renamed `foo()` to `bar()`" | pure knowledge — post-cutoff, no prior |

The fourth row is the awkward one, and it is where most agent training actually lives.
It is a fact, so it does not generalize; it is deployed as policy, so SFT teaches it
adequately *within that environment*. It will not transfer to a different repo.

---

## 7. Open questions this does not answer

- **Whether Terminal-Bench failures really are mostly behavioural.** §3 describes how to
  find out. Untested.
- **What proportion of the task set needs environment-specific procedural knowledge**
  (row 4 of §6) — that portion will not transfer and inflates apparent gains.
- **Whether low rank is in fact sufficient.** The argument in §5 is a hypothesis.
- Everything in §4 of `AGENTS.md` remains open.
