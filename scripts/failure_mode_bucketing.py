#!/usr/bin/env python3
"""Group harbor Vanillux2Agent trials by failure cause, side by side per job.

    python3 scripts/failure_mode_bucketing.py \
        --job base=~/tb2-eval/compare/vanillux2-base-k3-20260926T140340Z \
        --job sft-run1=~/tb2-eval/compare/vanillux2-sft-k3-20260926T230224Z \
        --out report.md

Reads only what harbor wrote per trial: result.json, agent/trajectory.json,
agent/timing.json, verifier/test-stdout.txt. Standard library only.
"""

import argparse
import collections
import glob
import json
import os
import random
import re
import urllib.parse

SUBMIT_MARKER = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
MAX_STEPS = 64
MAX_FORMAT_ERRORS = 64
RUNAWAY_CHARS = 45_000  # ~16k tokens, the agent's max_tokens
REPETITIVE_UNIQUE_RATIO = 0.5
SNIPPETS_PER_GROUP = 2

PRIMARY_RULES = [
    ("Solved", "reward == 1"),
    ("Verifier error/timeout (infra)", "no verifier reward, or exception VerifierTimeoutError"),
    ("Agent timeout (90 min)", "exception AgentTimeoutError"),
    ("Other agent exception", "any other exception_type"),
    ("Submitted, tests failed", f"agent ran a command containing {SUBMIT_MARKER}, reward 0"),
    ("Hit 64-step limit", f"{MAX_STEPS} recorded steps, never submitted"),
    ("Hit format-error limit", f"{MAX_FORMAT_ERRORS} format-error turns"),
    ("Stopped early: context window exceeded",
     "none of the above: the agent loop only exits this way on ContextWindowExceededError"),
]

FORMAT_ERROR_RULES = [
    ("Truncated: <tool_call> never closed", "text opens <tool_call> but has no </tool_call>"),
    ("Invalid JSON: raw control char at end of command",
     "json error 'Invalid control character' within the last 10 chars of the call"),
    ("Invalid JSON: raw control char mid-command", "'Invalid control character' elsewhere"),
    ("Invalid JSON: invalid \\escape", "json error 'Invalid \\escape'"),
    ("Invalid JSON: other", "any other json.JSONDecodeError"),
    ("Closed call with valid JSON", "server still returned no tool_calls (e.g. wrong tool name)"),
    ("No <tool_call> in text", "prose/thinking only, or a call the server parsed but the agent rejected"),
]


def unique_line_ratio(text):
    lines = [l.strip() for l in text.replace("\\n", "\n").split("\n") if len(l.strip()) > 25]
    if not lines:
        return 1.0, None
    top, n = collections.Counter(lines).most_common(1)[0]
    return len(set(lines)) / len(lines), (top, n)


def classify_format_error(text):
    if "<tool_call>" not in text:
        return "No <tool_call> in text", None
    m = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.S)
    if not m:
        return "Truncated: <tool_call> never closed", None
    body = m.group(1)
    try:
        json.loads(body)
        return "Closed call with valid JSON", None
    except json.JSONDecodeError as e:
        where = body[max(0, e.pos - 120): e.pos + 40]
        if e.msg.startswith("Invalid control character"):
            kind = ("Invalid JSON: raw control char at end of command" if e.pos >= len(body) - 10
                    else "Invalid JSON: raw control char mid-command")
        elif e.msg.startswith("Invalid \\escape"):
            kind = "Invalid JSON: invalid \\escape"
        else:
            kind = "Invalid JSON: other"
        return kind, f"{e.msg} at char {e.pos} of {len(body)}:\n...{where}"


def load_trial(tdir):
    result = json.load(open(os.path.join(tdir, "result.json")))
    traj = json.load(open(os.path.join(tdir, "agent", "trajectory.json")))
    timing = json.load(open(os.path.join(tdir, "agent", "timing.json")))
    try:
        test_out = open(os.path.join(tdir, "verifier", "test-stdout.txt"), errors="replace").read()
    except OSError:
        test_out = ""

    fmt_errors = []
    for i, msg in enumerate(traj):
        if (msg.get("role") == "assistant" and not msg.get("tool_calls") and i + 1 < len(traj)
                and "Format error" in str(traj[i + 1].get("content", ""))):
            text = msg.get("content") or ""
            kind, detail = classify_format_error(text)
            ratio, top = unique_line_ratio(text)
            fmt_errors.append(dict(kind=kind, detail=detail, chars=len(text), ratio=ratio, top=top,
                                   tail=text[-300:], msg_index=i))

    submitted = any(
        SUBMIT_MARKER in json.dumps(tc.get("function", {}).get("arguments", ""))
        for m in traj if m.get("role") == "assistant" for tc in (m.get("tool_calls") or []))

    exc = (result.get("exception_info") or {}).get("exception_type")
    vr = result.get("verifier_result")
    reward = (vr or {}).get("rewards", {}).get("reward") if vr else None
    steps = len(timing)

    if reward == 1:
        primary = "Solved"
    elif vr is None or exc == "VerifierTimeoutError":
        primary = "Verifier error/timeout (infra)"
    elif exc == "AgentTimeoutError":
        primary = "Agent timeout (90 min)"
    elif exc:
        primary = "Other agent exception"
    elif submitted:
        primary = "Submitted, tests failed"
    elif steps >= MAX_STEPS:
        primary = "Hit 64-step limit"
    elif len(fmt_errors) >= MAX_FORMAT_ERRORS:
        primary = "Hit format-error limit"
    else:
        primary = "Stopped early: context window exceeded"

    summary = [l for l in test_out.splitlines() if re.match(r"^(PASSED|FAILED|ERROR) ", l)]
    return dict(
        trial=os.path.basename(tdir.rstrip("/")), task=result["task_name"], exc=exc,
        model=((result.get("agent_info") or {}).get("model_info") or {}).get("name", ""),
        reward=reward, primary=primary, steps=steps, fmt_errors=fmt_errors,
        cmd_timeouts=sum(1 for s in timing if s.get("return_code") == 124),
        verifier_timeouts=test_out.count("TimeoutExpired"),
        fmt_llm_s=sum(s.get("llm_s") or 0 for s in timing if s.get("format_error")),
        llm_s=sum(s.get("llm_s") or 0 for s in timing),
        timing_tail=timing[-3:], test_summary=summary,
    )


def viewer_link(base, job, t):
    task = urllib.parse.quote(t["task"].split("/")[-1])
    return f"{base}/jobs/{job}/tasks/terminal-bench-2.0/vanillux2-agent/openai/{t['model']}/terminal-bench%2F{task}"


def fence(text):
    return "```\n" + text.replace("```", "ʼʼʼ").rstrip() + "\n```"


def pct(n, d):
    return f"{n} ({100 * n / d:.0f}%)" if d else "0"


def pick(items, k):
    items = sorted(items, key=lambda x: x[-1]["trial"] if isinstance(x, tuple) else x["trial"])
    return random.Random(0).sample(items, min(k, len(items)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", action="append", required=True, help="label=path/to/job_dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--viewer", default="http://localhost:8080")
    args = ap.parse_args()

    jobs = {}
    for spec in args.job:
        label, path = spec.split("=", 1)
        path = os.path.expanduser(path.rstrip("/"))
        trials = [load_trial(d) for d in sorted(glob.glob(os.path.join(path, "*__*/")))
                  if os.path.exists(os.path.join(d, "result.json"))]
        jobs[label] = (os.path.basename(path), trials)
    labels = list(jobs)

    out = ["# Failure report: " + " vs ".join(labels), ""]
    for label, (job, trials) in jobs.items():
        out.append(f"- **{label}**: `{job}`, {len(trials)} finished trials")
    out += ["", "Every number below is computed from harbor's per-trial files with the fixed rules at the end of this page. "
            "No model or human judgement is involved.", ""]

    out += ["## 1. Why each trial ended (one primary cause per trial)", "",
            "| primary cause | " + " | ".join(labels) + " |", "|---|" + "---|" * len(labels)]
    for name, _ in PRIMARY_RULES:
        row = [pct(sum(t["primary"] == name for t in tr), len(tr)) for _, tr in jobs.values()]
        out.append(f"| {name} | " + " | ".join(row) + " |")
    out.append("| **trials with ≥1 format-error turn** | " + " | ".join(
        pct(sum(bool(t["fmt_errors"]) for t in tr), len(tr)) for _, tr in jobs.values()) + " |")
    out.append("")

    out += ["## 2. Everything that went wrong, counted per occurrence", "",
            "| event | " + " | ".join(labels) + " |", "|---|" + "---|" * len(labels)]
    out.append("| agent turns (total) | " + " | ".join(
        str(sum(t["steps"] for t in tr)) for _, tr in jobs.values()) + " |")
    for name, _ in FORMAT_ERROR_RULES:
        out.append(f"| format error: {name} | " + " | ".join(
            str(sum(e["kind"] == name for t in tr for e in t["fmt_errors"])) for _, tr in jobs.values()) + " |")
    out.append(f"| format-error turn ≥{RUNAWAY_CHARS:,} chars (≈16k-token cap) | " + " | ".join(
        str(sum(e["chars"] >= RUNAWAY_CHARS for t in tr for e in t["fmt_errors"])) for _, tr in jobs.values()) + " |")
    out.append(f"| format-error turn with unique-line ratio < {REPETITIVE_UNIQUE_RATIO} (repetition loop) | " + " | ".join(
        str(sum(e["ratio"] < REPETITIVE_UNIQUE_RATIO for t in tr for e in t["fmt_errors"])) for _, tr in jobs.values()) + " |")
    out.append("| LLM time spent on turns that became format errors | " + " | ".join(
        f"{sum(t['fmt_llm_s'] for t in tr) / 3600:.1f} h of {sum(t['llm_s'] for t in tr) / 3600:.1f} h"
        for _, tr in jobs.values()) + " |")
    out.append("| agent commands killed at the 120 s limit (exit 124) | " + " | ".join(
        str(sum(t["cmd_timeouts"] for t in tr)) for _, tr in jobs.values()) + " |")
    out.append("| `TimeoutExpired` inside the verifier's tests | " + " | ".join(
        str(sum(t["verifier_timeouts"] for t in tr)) for _, tr in jobs.values()) + " |")
    out.append("")

    out += ["## 3. Primary causes: top tasks and examples", ""]
    for name, rule in PRIMARY_RULES:
        if not any(t["primary"] == name for _, tr in jobs.values() for t in tr):
            continue
        out += [f"### {name}", f"_Rule: {rule}_", ""]
        for label, (job, trials) in jobs.items():
            hit = [t for t in trials if t["primary"] == name]
            if not hit:
                continue
            top = collections.Counter(t["task"].split("/")[-1] for t in hit).most_common(5)
            out.append(f"**{label}** ({len(hit)}), top tasks: " + ", ".join(f"{k} ×{v}" for k, v in top))
            out.append("")
            for t in pick(hit, SNIPPETS_PER_GROUP):
                out.append(f"- `{t['trial']}`: {t['steps']} steps, {len(t['fmt_errors'])} format errors, "
                           f"{t['fmt_llm_s'] / 60:.0f} of {t['llm_s'] / 60:.0f} LLM-min in format-error turns. "
                           f"[viewer]({viewer_link(args.viewer, job, t)})")
                body = "last steps:\n" + "\n".join(
                    f"  step {s['step']}: llm {s.get('llm_s')}s  "
                    + ("FORMAT ERROR" if s.get("format_error") else f"rc={s.get('return_code')}  {str(s.get('cmd'))[:100]!r}")
                    for s in t["timing_tail"])
                if t["test_summary"]:
                    body += "\nverifier:\n" + "\n".join("  " + l[:140] for l in t["test_summary"][:6])
                out.append(fence(body))
            out.append("")

    out += ["## 4. Format-error kinds: examples", ""]
    for name, rule in FORMAT_ERROR_RULES:
        events = [(label, job, e, t) for label, (job, tr) in jobs.items() for t in tr for e in t["fmt_errors"]
                  if e["kind"] == name]
        if not events:
            continue
        out += [f"### {name}", f"_Rule: {rule}_", ""]
        for label in labels:
            ev = [x for x in events if x[0] == label]
            for _, job, e, t in pick(ev, SNIPPETS_PER_GROUP):
                rep = f", most repeated line ×{e['top'][1]}: `{e['top'][0][:80]}`" if e["top"] and e["top"][1] > 3 else ""
                out.append(f"- **{label}** `{t['trial']}` message {e['msg_index']}: {e['chars']:,} chars, "
                           f"unique-line ratio {e['ratio']:.2f}{rep}. [viewer]({viewer_link(args.viewer, job, t)})")
                out.append(fence(e["detail"] or ("...last 300 chars:\n" + e["tail"])))
        out.append("")

    out += ["## Rules", "", "Primary cause, first match wins:", ""]
    out += [f"{i}. **{n}**: {r}" for i, (n, r) in enumerate(PRIMARY_RULES, 1)]
    out += ["", "A format error is an assistant turn with no tool call that the agent answered with its "
            "'Format error' message. Its kind, first match wins:", ""]
    out += [f"{i}. **{n}**: {r}" for i, (n, r) in enumerate(FORMAT_ERROR_RULES, 1)]
    out.append("")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    open(args.out, "w").write("\n".join(out))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
