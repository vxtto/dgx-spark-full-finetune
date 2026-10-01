#!/usr/bin/env python3
"""Rebuild vittorino/tmax-sft-cleaned from allenai/tmax-sft.

Both training runs used the cleaned copy. It differs from the source by two
examples (indices 4558 and 4560 of 10,726): trajectories that end in 15 and 25
identical assistant turns in a row, with no tool result between them. Two
adjacent assistant turns break the assumption open-instruct's loss masking
makes about the chat template, so those two examples are dropped.

    python scripts/clean_dataset.py                    # count what would be dropped
    python scripts/clean_dataset.py --check            # compare with the published copy
    python scripts/clean_dataset.py --push <user/name> # upload the result

Needs `datasets`; run it inside the open-instruct environment.
"""

import argparse
import hashlib
import json

from datasets import load_dataset

SOURCE = "allenai/tmax-sft"
PUBLISHED = "vittorino/tmax-sft-cleaned"
CONFIG = "skill_tax_20260505_2.2k_combined_balanced_thinking_all"


def has_adjacent_assistant_turns(example):
    roles = [m["role"] for m in example["messages"]]
    return any(a == b == "assistant" for a, b in zip(roles, roles[1:]))


def digest(example):
    return hashlib.sha256(json.dumps(example, sort_keys=True, default=str).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--check", action="store_true", help=f"compare the result with {PUBLISHED}")
    parser.add_argument("--push", metavar="REPO", help="upload the cleaned dataset to this Hub repo")
    args = parser.parse_args()

    source = load_dataset(SOURCE, args.config, split="train")
    dropped = [i for i, ex in enumerate(source) if has_adjacent_assistant_turns(ex)]
    cleaned = source.select([i for i in range(len(source)) if i not in set(dropped)])
    print(f"{SOURCE}: {len(source)} examples, dropped {len(dropped)} at {dropped}, kept {len(cleaned)}")

    if args.check:
        published = load_dataset(PUBLISHED, args.config, split="train")
        same = len(published) == len(cleaned) and all(
            digest(a) == digest(b) for a, b in zip(cleaned, published)
        )
        print(f"{PUBLISHED}: {len(published)} examples, identical in order and content: {same}")
        if not same:
            raise SystemExit(1)

    if args.push:
        cleaned.push_to_hub(args.push, args.config, split="train")


if __name__ == "__main__":
    main()
