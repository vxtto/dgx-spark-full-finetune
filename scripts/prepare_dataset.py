#!/usr/bin/env python3
"""Materialise the TMAX SFT trajectories to a local parquet snapshot.

Why this exists: open-instruct cannot select a config of a multi-config HF
dataset. `DatasetConfig.__post_init__` in open_instruct/dataset_transformation.py
calls

    load_dataset(self.dataset_name, split=..., revision=..., num_proc=...)

with no `name=` argument. `allenai/tmax-sft` ships two configs and no default, so
pointing the mixer at it directly raises "Config name is missing".

The same function does accept a local `.parquet` path, so we snapshot the config
we want to disk and point the run at that. This also satisfies AGENTS.md §5 —
"Every run is reproducible from one config file plus one data snapshot".

    python scripts/prepare_dataset.py                    # the paper's recipe (_all)
    python scripts/prepare_dataset.py --variant only_success
    python scripts/prepare_dataset.py --stats            # add token statistics

Writes data/processed/<name>.parquet plus a .json sidecar with provenance.
"""

import argparse
import hashlib
import json
import os
import pathlib
import platform
import sys
from collections import Counter

REPO = "allenai/tmax-sft"
PREFIX = "skill_tax_20260505_2.2k_combined_balanced"
VARIANTS = {
    # what the paper uses: §D.2 "we do not filter out unsuccessful/incomplete rollouts"
    "all": f"{PREFIX}_thinking_all",
    # rejection sampling, already applied by the authors
    "only_success": f"{PREFIX}_thinking_only_success",
}
# Large artifacts live on the Spark, never in the repo tree (AGENTS.md §5).
# Override with LLM_DATA_DIR; run_sft.sh expands the same variable.
DEFAULT_DATA_DIR = pathlib.Path(
    os.environ.get("LLM_DATA_DIR", pathlib.Path.home() / "llm-training-data")
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=sorted(VARIANTS), default="all",
                    help="which config to snapshot (default: all — the paper's recipe)")
    ap.add_argument("--stats", action="store_true",
                    help="tokenize with the Qwen3 tokenizer and report length stats "
                         "(slow; turns the cost estimate into a measurement)")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-8B")
    ap.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_DATA_DIR,
                    help=f"where to write the snapshot (default: $LLM_DATA_DIR, "
                         f"currently {DEFAULT_DATA_DIR})")
    ap.add_argument("--allow-laptop", action="store_true",
                    help="download anyway on a non-Linux machine (AGENTS.md §6 says "
                         "large artifacts belong on the Spark)")
    args = ap.parse_args()

    # This pulls several hundred MB. AGENTS.md §6: not on the laptop.
    if platform.system() != "Linux" and not args.allow_laptop:
        print(f"error: this is {platform.system()}, not the Spark.", file=sys.stderr)
        print("       The snapshot is 139-319 MB and belongs on the Spark "
              "(AGENTS.md §5, §6).", file=sys.stderr)
        print("       Run this there, or pass --allow-laptop for local inspection.",
              file=sys.stderr)
        return 1

    try:
        from datasets import load_dataset
    except ImportError:
        print("error: `datasets` is not installed.", file=sys.stderr)
        return 1

    config = VARIANTS[args.variant]
    print(f"loading {REPO} :: {config}")
    ds = load_dataset(REPO, config, split="train")
    print(f"  {len(ds):,} rows, columns: {ds.column_names}")

    # --- Sanity: this must be SFT data, not the RL task set ------------------
    # allenai/tmax-15k-open-instruct looks superficially similar but has only
    # [system, user] messages and no assistant turns — there is nothing for the
    # SFT loss to attach to. Fail loudly rather than train on it.
    roles = [m["role"] for m in ds[0]["messages"]]
    n_assistant = roles.count("assistant")
    if n_assistant == 0:
        print(f"\nerror: no assistant turns in row 0 (roles: {roles}).", file=sys.stderr)
        print("       This looks like an RL task set, not SFT trajectories.", file=sys.stderr)
        return 1
    print(f"  row 0: {len(roles)} messages, {n_assistant} assistant turns  [OK]")

    # --- Per-task pass rate --------------------------------------------------
    # metadata carries `task` and `episode`, so the difficulty distribution is
    # derivable here rather than guessed. See docs/decisions.md D-06.
    try:
        per_task = Counter(m["task"] for m in ds["metadata"])
        dist = Counter(per_task.values())
        print(f"\n  {len(per_task):,} distinct tasks; episodes retained per task:")
        for k in sorted(dist):
            bar = "#" * min(40, dist[k] * 40 // max(dist.values()))
            print(f"    {k:>2} episode(s): {dist[k]:>5} tasks  {bar}")
        if args.variant == "only_success":
            print("    (for only_success this IS the per-task pass count out of 8)")
    except (KeyError, TypeError) as exc:
        print(f"  note: could not derive per-task counts ({exc})")

    # --- Optional token statistics ------------------------------------------
    if args.stats:
        try:
            from transformers import AutoTokenizer
        except ImportError:
            print("error: --stats needs `transformers`.", file=sys.stderr)
            return 1
        print(f"\ntokenizing with {args.tokenizer} (this takes a while)...")
        tok = AutoTokenizer.from_pretrained(args.tokenizer)
        lengths = []
        for i, row in enumerate(ds):
            lengths.append(len(tok.apply_chat_template(row["messages"], tokenize=True)))
            if (i + 1) % 1000 == 0:
                print(f"  {i + 1:,}/{len(ds):,}", end="\r", flush=True)
        lengths.sort()
        total = sum(lengths)
        pct = lambda p: lengths[int(len(lengths) * p / 100)]  # noqa: E731
        print(f"\n  total tokens : {total:,}")
        print(f"  mean         : {total // len(lengths):,}")
        print(f"  p50/p90/p99  : {pct(50):,} / {pct(90):,} / {pct(99):,}")
        print(f"  max          : {lengths[-1]:,}")
        for limit in (8192, 16384, 32768):
            over = sum(1 for n in lengths if n > limit)
            print(f"  > {limit:>5} : {over:>6,} rows ({100 * over / len(lengths):.1f}%) truncated")
        print(f"\n  2 epochs = {2 * total:,} training tokens")
        print("  divide by measured tokens/sec from the smoke run for a real ETA")

    # --- Write snapshot ------------------------------------------------------
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"tmax_sft_{args.variant}.parquet"
    ds.to_parquet(out)
    size_mb = out.stat().st_size / 1e6

    digest = hashlib.sha256()
    with open(out, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    sha = digest.hexdigest()

    (out.with_suffix(".json")).write_text(json.dumps({
        "source_repo": REPO,
        "source_config": config,
        "variant": args.variant,
        "num_rows": len(ds),
        "size_mb": round(size_mb, 1),
        "sha256": sha,
        "columns": ds.column_names,
    }, indent=2) + "\n")

    print(f"\nwrote {out}  ({size_mb:.0f} MB, sha256 {sha[:16]}...)")
    print(f"      {out.with_suffix('.json')}")
    print(f"\nThe run configs reference this as ${{LLM_DATA_DIR}}/{out.name};")
    print("run_sft.sh expands that at launch. Export it if you used a custom path:")
    print(f"  export LLM_DATA_DIR={out.parent}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
