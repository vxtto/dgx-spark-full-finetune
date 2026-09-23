# tmax legacy SFT harness config

Two files that `hamishivi/tmax` looks for and does not ship. Copy them into the
tmax checkout before evaluating:

    mkdir -p "$TMAX_DIR/sft/preprocessing/config"
    cp patches/tmax-harness-config/tool_schemas.json \
       patches/tmax-harness-config/system_prompt.txt \
       "$TMAX_DIR/sft/preprocessing/config/"

`scripts/eval_tb2_vanillux.sh` refuses to start without them.

## Why

`rl_data/generator/sample_solutions.py` resolves

    HARNESS_CONFIG_DIR = <tmax root>/sft/preprocessing/config

and `_load_harness_config()` — *"Load legacy SFT harness config, falling back to
the checked-in tmax defaults"* — needs **both** `system_prompt.txt` and
`tool_schemas.json` to be present, or it silently returns
`_DEFAULT_TOOL_SCHEMAS`. That directory is never created: `git ls-files` under
`sft/` returns 0 entries at tmax `6d3d606c`. It is a dangling reference to an
Ai2-internal path.

The fallback and the released SFT data disagree about the one thing that
matters most to the agent's behaviour:

| | `bash` tool description |
|---|---|
| `_DEFAULT_TOOL_SCHEMAS` (what you get) | Execute a bash command. **Each command runs in a new subshell.** |
| the SFT trajectories (what the model learnt) | Execute a bash command **in a persistent shell. Working directory and environment variables are preserved between calls.** |

Those are opposite claims about whether `cd` and `export` survive between
turns. The fallback also contradicts `Vanillux2Agent` itself, which defaults to
`persistent_bash=True` and builds a persistent shell in `setup()`, and it
contradicts the instance template two lines above it in the same system
message: *"Directory and environment-variable changes ARE persistent across
calls in this harness [...] treat the shell as a long-running login shell."*

Evaluating on the fallback measures how the model copes with a self-
contradictory prompt, not what the SFT taught it.

## Provenance

- `tool_schemas.json` — extracted from the `<tools>` block rendered inside the
  training data itself (`runs/sft_qwen3_8b_run1_20260920T052358Z/train.log`),
  not from tmax's source. Verified against 300 random rows of the 10,724 in
  `vittorino/tmax-sft-cleaned`: 300/300 carry the persistent-shell wording, 0
  carry the subshell wording.
- `system_prompt.txt` — `system_template` from
  `rl_data/generator/vanillux_prompts.yaml`, verified byte-identical to the
  system prompt in the training data. `Vanillux2Agent` reads the template from
  that yaml and never uses this file; it exists only because
  `_load_harness_config()` requires both files before it will read either.
