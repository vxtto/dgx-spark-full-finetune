#!/usr/bin/env python3
"""Draw the README figures.

    python scripts/plot_runs.py

Writes docs/img/{timeline,memory,speedup,wall_clock,loss}.svg. Standard library
only. loss and wall_clock are drawn from evidence/steps/run{1,2}.csv (the
`Step: N, LR: ..., Loss: ..., TPS: ...` lines of each run's train.log, one row
per optimizer step). speedup reads evidence/bench/ and evidence/profile/. The
memory figures are the measured values in docs/spark-memory-guardrail.md, and
the timeline dates come from the run and eval provenance files.
"""

import csv
import datetime as dt
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = [
    ("run1", "Run 1: BF16, 8-bit AdamW, Liger", "#2a78d6"),
    ("run2", "Run 2: run 1 + spark_opt", "#eb6834"),
]
SURFACE, GRID, TEXT, MUTED = "#fcfcfb", "#e6e5e1", "#0b0b0b", "#52514e"
FONT = "system-ui, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
W, H = 760, 410
LEFT, RIGHT, TOP, BOTTOM = 64, 150, 96, 48
LEGEND_Y = 70
LOSS_WINDOW = 100


def load(name):
    with open(ROOT / "evidence" / "steps" / f"{name}.csv") as f:
        rows = list(csv.DictReader(f))
    start = dt.datetime.fromisoformat(rows[0]["timestamp"])
    return [
        {
            "step": int(r["step"]),
            "hours": (dt.datetime.fromisoformat(r["timestamp"]) - start).total_seconds() / 3600,
            "loss": float(r["loss"]),
            "tok_per_s": float(r["cumulative_tok_per_s"]),
        }
        for r in rows
    ]


class Plot:
    def __init__(self, title, subtitle, x_range, y_range, x_label):
        self.x0, self.x1 = x_range
        self.y0, self.y1 = y_range
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
            f'font-family="{FONT}" role="img" aria-label="{title}">',
            f'<rect width="{W}" height="{H}" rx="8" fill="{SURFACE}"/>',
            f'<text x="{LEFT}" y="28" font-size="15" font-weight="600" fill="{TEXT}">{title}</text>',
            f'<text x="{LEFT}" y="46" font-size="12" fill="{MUTED}">{subtitle}</text>',
            f'<text x="{(LEFT + W - RIGHT) / 2}" y="{H - 10}" font-size="12" fill="{MUTED}" '
            f'text-anchor="middle">{x_label}</text>',
        ]

    def x(self, v):
        return LEFT + (v - self.x0) / (self.x1 - self.x0) * (W - LEFT - RIGHT)

    def y(self, v):
        return H - BOTTOM - (v - self.y0) / (self.y1 - self.y0) * (H - TOP - BOTTOM)

    def axes(self, x_ticks, y_ticks, y_format):
        for t in y_ticks:
            y = self.y(t)
            self.parts.append(f'<line x1="{LEFT}" x2="{W - RIGHT}" y1="{y:.1f}" y2="{y:.1f}" stroke="{GRID}"/>')
            self.parts.append(
                f'<text x="{LEFT - 8}" y="{y + 4:.1f}" font-size="11" fill="{MUTED}" '
                f'text-anchor="end">{y_format(t)}</text>'
            )
        for t in x_ticks:
            self.parts.append(
                f'<text x="{self.x(t):.1f}" y="{H - BOTTOM + 18}" font-size="11" fill="{MUTED}" '
                f'text-anchor="middle">{t:,}</text>'
            )

    def line(self, points, color):
        path = " ".join(f"{self.x(px):.1f},{self.y(py):.1f}" for px, py in points)
        self.parts.append(
            f'<polyline points="{path}" fill="none" stroke="{color}" stroke-width="2" '
            f'stroke-linejoin="round" stroke-linecap="round"/>'
        )

    def end_label(self, point, color, lines, dy=0):
        px, py = self.x(point[0]), self.y(point[1])
        self.parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4" fill="{color}" stroke="{SURFACE}" stroke-width="2"/>')
        for i, text in enumerate(lines):
            weight = "600" if i == 0 else "400"
            fill = TEXT if i == 0 else MUTED
            self.parts.append(
                f'<text x="{px + 10:.1f}" y="{py + 4 + dy + i * 15:.1f}" font-size="12" '
                f'font-weight="{weight}" fill="{fill}">{text}</text>'
            )

    def legend(self, entries):
        x = LEFT
        for label, color in entries:
            self.parts.append(f'<line x1="{x}" x2="{x + 18}" y1="{LEGEND_Y - 4}" y2="{LEGEND_Y - 4}" stroke="{color}" stroke-width="2" stroke-linecap="round"/>')
            self.parts.append(f'<text x="{x + 24}" y="{LEGEND_Y}" font-size="12" fill="{MUTED}">{label}</text>')
            x += 24 + 7 * len(label) + 24

    def save(self, name):
        out = ROOT / "docs" / "img" / name
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(self.parts + ["</svg>"]) + "\n")
        print(f"wrote {out.relative_to(ROOT)}")


def windowed_loss(run):
    return [
        (run[i + LOSS_WINDOW - 1]["step"], statistics.mean(r["loss"] for r in run[i : i + LOSS_WINDOW]))
        for i in range(0, len(run) - LOSS_WINDOW + 1, LOSS_WINDOW)
    ]


NEUTRAL = "#7d7c77"
RUN1, RUN2 = RUNS[0][2], RUNS[1][2]


def svg_open(height, label):
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {height}" width="{W}" height="{height}" '
        f'font-family="{FONT}" role="img" aria-label="{label}">',
        f'<rect width="{W}" height="{height}" rx="8" fill="{SURFACE}"/>',
    ]


def text(x, y, body, size=12, fill=MUTED, weight="400", anchor="start"):
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" '
        f'fill="{fill}" text-anchor="{anchor}">{body}</text>'
    )


def bar(x, y, width, height, fill, round_end=True):
    """A horizontal bar: square at the baseline (left), 4px rounded at the data end."""
    if not round_end or width < 8:
        return f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height}" fill="{fill}"/>'
    r = 4
    return (
        f'<path d="M{x:.1f},{y:.1f} h{width - r:.1f} a{r},{r} 0 0 1 {r},{r} v{height - 2 * r} '
        f'a{r},{r} 0 0 1 -{r},{r} h-{width - r:.1f} z" fill="{fill}"/>'
    )


def save(parts, name):
    out = ROOT / "docs" / "img" / name
    out.write_text("\n".join(parts + ["</svg>"]) + "\n")
    print(f"wrote {out.relative_to(ROOT)}")


def draw_timeline():
    """Project phases on a date axis, 13 Sep to 2 Oct 2026."""
    day0 = dt.datetime(2026, 9, 13)
    days = 19
    phases = [  # label, start, end, colour, note
        ("Setup, first attempts", "2026-09-13T00:00", "2026-09-17T00:00", NEUTRAL, "13 Sep: project start"),
        ("Make it fit", "2026-09-17T00:00", "2026-09-20T00:00", NEUTRAL, "killed at 113 GB, then 72 GB"),
        ("Run 1", "2026-09-20T07:24", "2026-09-22T21:52", RUN1, "62 h 26 min"),
        ("Profile and benchmark", "2026-09-26T08:32", "2026-09-28T03:20", NEUTRAL, "nsys, micro-benchmarks"),
        ("Run 2", "2026-09-28T03:31", "2026-09-30T00:08", RUN2, "44 h 35 min"),
        ("Evaluation", "2026-09-26T16:03", "2026-10-01T02:04", NEUTRAL, "3 arms, 267 trials each"),
    ]
    left, right, top, row = 170, 24, 62, 30
    height = top + row * len(phases) + 44
    span = W - left - right
    x = lambda stamp: left + (dt.datetime.fromisoformat(stamp) - day0).total_seconds() / 86400 / days * span
    parts = svg_open(height, "Project timeline, 13 September to 1 October 2026")
    parts.append(text(24, 28, "Three weeks, September 2026", 15, TEXT, "600"))
    parts.append(text(24, 46, "Dates from the run and eval provenance files"))
    axis_y = top + row * len(phases) + 6
    for d in range(0, days + 1, 2):
        gx = left + d / days * span
        parts.append(f'<line x1="{gx:.1f}" x2="{gx:.1f}" y1="{top - 6}" y2="{axis_y}" stroke="{GRID}"/>')
        date = day0 + dt.timedelta(days=d)
        parts.append(text(gx, axis_y + 16, f"{date.day} {date:%b}", 11, MUTED, anchor="middle"))
    for i, (label, start, end, colour, note) in enumerate(phases):
        y = top + i * row
        parts.append(text(left - 12, y + 16, label, 12, TEXT, "600" if colour != NEUTRAL else "400", "end"))
        x0, x1 = x(start), x(end)
        parts.append(f'<rect x="{x0:.1f}" y="{y + 4}" width="{x1 - x0:.1f}" height="16" rx="4" fill="{colour}"/>')
        if x1 + 8 + 7 * len(note) < W - right:
            parts.append(text(x1 + 8, y + 16, note, 11))
        else:
            parts.append(text(x0 - 8, y + 16, note, 11, anchor="end"))
    save(parts, "timeline.svg")


def draw_memory():
    """Where the memory goes: the paper's recipe on one GPU against run 1."""
    colours = {"weights": "#1baf7a", "gradients": "#eda100", "optimizer": "#e87ba4", "activations": "#4a3aa7"}
    names = {"weights": "bf16 weights", "gradients": "bf16 gradients", "optimizer": "AdamW state", "activations": "activations at 32,768 tokens"}
    rows = [
        ("Paper's recipe, one GPU", "fp32 AdamW", [("weights", 16.4), ("gradients", 16.4), ("optimizer", 65.6)], 113, "killed at about 113 GB, before step 1 finishes"),
        ("This repo", "8-bit AdamW + fused cross-entropy", [("weights", 16.4), ("gradients", 16.4), ("optimizer", 16.7), ("activations", 22.5)], None, "72 GB peak: trains to completion"),
    ]
    left, right, top, row, thick = 24, 24, 96, 86, 24
    height = top + row * len(rows) + 4
    span, scale_max = W - left - right, 120
    x = lambda gb: left + gb / scale_max * span
    parts = svg_open(height, "Memory needed for a full fine-tune of Qwen3-8B, before and after")
    parts.append(text(24, 28, "Making it fit: 98 GB before a single activation, or 72 GB in total", 15, TEXT, "600"))
    parts.append(text(24, 46, "Memory by component for an 8.19 B-parameter full fine-tune, in GB"))
    lx = left
    for key, colour in colours.items():
        parts.append(f'<rect x="{lx}" y="60" width="12" height="12" rx="3" fill="{colour}"/>')
        parts.append(text(lx + 18, 71, names[key]))
        lx += 18 + 6.6 * len(names[key]) + 22
    for i, (label, sub, segments, killed_at, verdict) in enumerate(rows):
        y = top + i * row
        parts.append(text(left, y + 12, label, 13, TEXT, "600"))
        parts.append(text(left + 8 + 7.6 * len(label), y + 12, sub))
        total = 0.0
        for j, (key, gb) in enumerate(segments):
            last = j == len(segments) - 1
            x0 = x(total) + (2 if j else 0)
            parts.append(bar(x0, y + 22, x(total + gb) - x0, thick, colours[key], round_end=last))
            parts.append(text((x(total) + x(total + gb)) / 2, y + 22 + thick + 15, f"{gb:g}", 11, MUTED, anchor="middle"))
            total += gb
        end = total
        if killed_at:
            parts.append(
                f'<rect x="{x(total) + 2:.1f}" y="{y + 22.5}" width="{x(killed_at) - x(total) - 2:.1f}" height="{thick - 1}" '
                f'fill="none" stroke="{colours["activations"]}" stroke-dasharray="4 3"/>'
            )
            end = killed_at
        parts.append(text(x(end) + 10, y + 22 + thick / 2 + 4, verdict if not killed_at else "", 12, TEXT, "600"))
        if killed_at:
            parts.append(text(x(end), y + 12, verdict, 12, TEXT, "600", "end"))
    save(parts, "memory.svg")


def draw_speedup():
    """The micro-benchmark: tokens per second on the same 8 micro-batches."""
    timed_tokens = 53120
    reference = {}
    with open(ROOT / "evidence" / "profile" / "bench_fp8_results.jsonl") as f:
        for line in f:
            row = json.loads(line)
            reference[row["mode"]] = timed_tokens / (row["step_seconds"][0] + row["step_seconds"][3])
    bench = lambda stamp: json.loads((ROOT / "evidence" / "bench" / f"bench_spark_opt_{stamp}" / "result.json").read_text())["tok_per_s"]
    rows = [
        ("bf16, stock cross-entropy", reference["bf16"], NEUTRAL),
        ("+ FP8 only", reference["fp8"], NEUTRAL),
        ("cross-entropy fix + adaptive checkpointing", bench("20260928T011602Z"), NEUTRAL),
        ("all three: the run 2 config", bench("20260928T005729Z"), RUN2),
    ]
    base = rows[0][1]
    left, right, top, row, thick = 300, 130, 66, 34, 20
    height = top + row * len(rows) + 16
    span, scale_max = W - left - right, 1400
    parts = svg_open(height, "Throughput of each configuration on the same micro-batches")
    parts.append(text(24, 28, "Where the speed comes from", 15, TEXT, "600"))
    parts.append(text(24, 46, "Tokens per second on the same 8 micro-batches of run 1 (53,120 tokens), all with torch.compile"))
    parts.append(f'<line x1="{left}" x2="{left}" y1="{top - 4}" y2="{top + row * len(rows) - 6}" stroke="{GRID}"/>')
    for i, (label, tok_per_s, colour) in enumerate(rows):
        y = top + i * row
        emphasis = colour != NEUTRAL
        parts.append(text(left - 12, y + 14, label, 12, TEXT, "600" if emphasis else "400", "end"))
        width = tok_per_s / scale_max * span
        parts.append(bar(left, y, width, thick, colour))
        parts.append(text(left + width + 8, y + 14, f"{tok_per_s:,.0f} tok/s", 12, TEXT, "600"))
        parts.append(text(left + width + 8 + 78, y + 14, f"{tok_per_s / base:.2f}×", 12))
    save(parts, "speedup.svg")


def main():
    runs = {name: load(name) for name, _, _ in RUNS}
    steps = runs["run1"][-1]["step"]
    legend = [(label, color) for _, label, color in RUNS]

    loss = {name: windowed_loss(run) for name, run in runs.items()}
    gap = max(abs(a[1] - b[1]) for a, b in zip(loss["run1"], loss["run2"]))
    plot = Plot(
        "Training loss is the same in both runs",
        f"Mean loss per {LOSS_WINDOW} optimizer steps. Largest gap between the two lines: {gap:.4f}",
        (0, steps), (0.15, 0.65), "optimizer step",
    )
    plot.axes(range(0, steps + 1, 1000), [0.15, 0.25, 0.35, 0.45, 0.55, 0.65], lambda v: f"{v:.2f}")
    plot.legend(legend)
    for name, _, color in RUNS:
        plot.line(loss[name], color)
    plot.end_label(loss["run2"][-1], RUNS[1][2], [f"{loss['run2'][-1][1]:.3f}", "run 2"], dy=-22)
    plot.end_label(loss["run1"][-1], RUNS[0][2], [f"{loss['run1'][-1][1]:.3f}", "run 1"], dy=14)
    plot.save("loss.svg")

    longest = max(run[-1]["hours"] for run in runs.values())
    plot = Plot(
        "Same 5,362 steps, 29% less wall clock",
        "Optimizer steps completed against hours since the first step, from each run's training log",
        (0, 70), (0, 6000), "hours",
    )
    plot.axes(range(0, 71, 10), range(0, 6001, 1000), lambda v: f"{v:,}")
    plot.legend(legend)
    for name, _, color in RUNS:
        run = runs[name]
        plot.line([(r["hours"], r["step"]) for r in run[::20]] + [(run[-1]["hours"], run[-1]["step"])], color)
    for (name, _, color), dy in zip(RUNS, (14, -22)):
        last = runs[name][-1]
        hours, minutes = divmod(round(last["hours"] * 60), 60)
        plot.end_label((last["hours"], last["step"]), color, [f"{hours} h {minutes:02d} min", f"{last['tok_per_s']:,.0f} tok/s"], dy=dy)
    assert longest < 70
    plot.save("wall_clock.svg")

    draw_timeline()
    draw_memory()
    draw_speedup()


if __name__ == "__main__":
    main()
