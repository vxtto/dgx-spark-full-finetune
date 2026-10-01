#!/usr/bin/env python3
"""Draw the two README figures from evidence/steps/run{1,2}.csv.

    python scripts/plot_runs.py

Writes docs/img/loss.svg and docs/img/wall_clock.svg. Standard library only.
The CSVs are the `Step: N, LR: ..., Loss: ..., TPS: ...` lines of each run's
train.log, one row per optimizer step.
"""

import csv
import datetime as dt
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


if __name__ == "__main__":
    main()
