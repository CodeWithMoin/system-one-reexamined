"""Figures for the README, drawn from cached predictions and results/latency*.json.

Colour does one job: identity of the two System One models (Laya, Jev). Every older method is
drawn in neutral grey and labelled directly, so no chart relies on colour alone.
Palette: the dataviz reference categorical palette, first three slots (validated all-pairs).

  uv run --no-sync python scripts/make_figures.py [--latency results/latency_m4.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from s1x import cache, metrics  # noqa: E402

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"   # Laya, Jev, SetFit
GREY, GREY_DARK = "#8a8a86", "#5c5c58"
INK, INK_2, GRID, SURFACE = "#1f1f1d", "#5c5c58", "#e6e5e1", "#fcfcfb"
MIN_BIN = 20
TASKS = {"ag_news": "AG News (4 topics)", "emotion": "Emotion (6)", "banking77": "Banking77 (77 intents)",
         "sst5": "SST-5 (1–5 rating)", "sms_spam": "SMS spam (yes/no)"}

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK_2,
    "xtick.color": INK_2, "ytick.color": INK_2, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlecolor": INK, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "lines.linewidth": 2,
})


def acc(task: str, method: str, shot: str | None = None) -> float:
    p, y, _ = cache.read(task, method, shot)["test"]
    return metrics.accuracy(p, y)


def kshot(task: str, method: str, k: int) -> tuple[float, float] | None:
    vals = [acc(task, method, f"k={k},seed={s}") for s in range(5) if cache.exists(task, method, f"k={k},seed={s}")]
    return (float(np.mean(vals)), float(np.std(vals))) if vals else None


def fig_speed(latency_path: Path) -> Path:
    lat = json.loads(latency_path.read_text())
    bench = lat.get("benchmark_v2", lat)
    points = [  # (label, method key in latency, accuracy method, k, style)
        ("Laya (PyTorch)", "laya-torch", "laya", None, "laya"),
        ("Laya (MLX runtime)", "laya", "laya", None, "laya-mlx"),
        ("NLI-large", "nli", "nli", None, "old"),
        ("NLI-base", "nli-base", "nli-base", None, "old"),
        ("Bi-encoder ZS", "embed-zs", "embed-zs", None, "old"),
        ("SetFit, 16 labels", "setfit", "setfit", 16, "old"),
        ("Emb+LR, 16 labels", "embed-lr", "embed-lr", 16, "old"),
    ]
    # Hand-placed label offsets (points), so no two labels collide.
    offsets = {
        "ag_news": {"Laya (PyTorch)": (8, 6), "Laya (MLX runtime)": (-8, 8, "right"), "NLI-large": (8, -12),
                    "NLI-base": (-8, -14, "right"), "Bi-encoder ZS": (8, 4), "SetFit, 16 labels": (8, -12),
                    "Emb+LR, 16 labels": (-8, 8, "right"), "Jev": (-8, -14, "right")},
        "banking77": {"Laya (PyTorch)": (8, -12), "Laya (MLX runtime)": (-8, 8, "right"), "NLI-large": (8, 4),
                      "NLI-base": (-8, -14, "right"), "Bi-encoder ZS": (8, 4), "SetFit, 16 labels": (8, -12),
                      "Emb+LR, 16 labels": (8, 6), "Jev": (-8, 8, "right")},
    }
    ylims = {"ag_news": (0.70, 0.95), "banking77": (0.30, 0.95)}

    def place(ax, task, label, key, x, y):
        o = offsets[task][key]
        ax.annotate(label, (x, y), xytext=o[:2], textcoords="offset points", fontsize=8.5, color=INK_2,
                    ha=o[2] if len(o) > 2 else "left")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    for ax, task in zip(axes, ["ag_news", "banking77"]):
        for label, lkey, amethod, k, style in points:
            entry = bench[task].get(f"{lkey}_single")
            if not entry:
                continue
            x = entry["p50_ms"]
            y = acc(task, amethod) if k is None else (kshot(task, amethod, k) or (np.nan,))[0]
            if style == "laya":
                ax.scatter(x, y, s=90, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=4)
            elif style == "laya-mlx":
                ax.scatter(x, y, s=90, facecolor=SURFACE, edgecolor=BLUE, linewidth=2, zorder=4)
            else:
                ax.scatter(x, y, s=60, color=GREY, edgecolor=SURFACE, linewidth=2, zorder=3)
            place(ax, task, label, label, x, y)
        jev = cache.read(task, "jev")["test"]
        jx, jy = float(np.nanmedian(jev[2])), metrics.accuracy(jev[0], jev[1])
        ax.scatter(jx, jy, s=90, facecolor=SURFACE, edgecolor=ORANGE, linewidth=2, zorder=4)
        place(ax, task, "Jev (API, incl. network)", "Jev", jx, jy)
        ax.set_xscale("log")
        ax.set_xlim(5, 5000)
        ax.set_ylim(*ylims[task])
        ax.set_title(TASKS[task], loc="left")
        ax.set_xlabel("Median latency per decision (ms, log scale) — lower is faster")
    axes[0].set_ylabel("Accuracy (1,000 test examples)")
    fig.suptitle("Accuracy vs speed — up and to the left is better", x=0.01, ha="left", fontsize=12.5,
                 fontweight="bold", color=INK)
    fig.text(0.01, -0.02, f"Local methods timed one decision at a time on one machine ({latency_path.name}). "
             "Filled blue: Laya on PyTorch, same framework as the grey methods. Hollow markers: Laya on MLX "
             "and Jev over its API — not like-for-like with the rest.", fontsize=8, color=INK_2, ha="left")
    fig.tight_layout()
    out = ROOT / "figures" / "accuracy_vs_speed.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_labels() -> Path:
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.9), sharey=False)
    ks = [8, 16, 64]
    for ax, (task, title) in zip(axes, TASKS.items()):
        for method, color, label in [("setfit", AQUA, "SetFit"), ("embed-lr", GREY, "Emb+LR")]:
            vals = [kshot(task, method, k) for k in ks]
            m = np.array([v[0] for v in vals]); s = np.array([v[1] for v in vals])
            ax.plot(ks, m, color=color, marker="o", markersize=5, zorder=3)
            ax.fill_between(ks, m - s, m + s, color=color, alpha=0.12, linewidth=0)
            dy = {("banking77", "setfit"): -8, ("banking77", "embed-lr"): 6}.get((task, method), 0)
            ax.annotate(label, (ks[-1], m[-1]), xytext=(5, dy), textcoords="offset points", va="center",
                        fontsize=8.5, color=INK_2)
        for method, color, ls, label in [("laya", BLUE, "-", "Laya"), ("jev", ORANGE, "-", "Jev"),
                                         ("nli-base", GREY_DARK, (0, (4, 3)), "NLI-base")]:
            y = acc(task, method)
            ax.axhline(y, color=color, linestyle=ls, linewidth=1.6, zorder=2)
            below = (task, method) in {("emotion", "laya"), ("ag_news", "jev"), ("banking77", "jev")}
            ax.annotate(f"{label} {y:.2f}", (6.2, y), xytext=(0, -3 if below else 3), textcoords="offset points",
                        fontsize=8, color=INK_2, va="top" if below else "bottom")
        ax.set_xscale("log", base=2)
        ax.set_xticks(ks, [str(k) for k in ks])
        ax.set_xlim(6, 110)
        ax.set_title(title, loc="left")
        ax.set_xlabel("Labelled examples per class")
    axes[0].set_ylabel("Accuracy")
    fig.suptitle("How many labels does it take to catch the zero-shot models?", x=0.01, ha="left",
                 fontsize=12.5, fontweight="bold", color=INK)
    fig.text(0.01, -0.04, "Lines: few-label methods, mean over 5 random draws (band = ±1 std). Horizontal lines: "
             "zero-shot models, no labels. Laya blue, Jev orange, NLI-base dashed grey.", fontsize=8,
             color=INK_2, ha="left")
    fig.tight_layout()
    out = ROOT / "figures" / "accuracy_vs_labels.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_reliability() -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    models = [("laya", BLUE, "Laya"), ("jev", ORANGE, "Jev"), ("nli-base", GREY_DARK, "NLI-base")]
    for ax, task in zip(axes, ["emotion", "sst5", "ag_news"]):
        ax.plot([0, 1], [0, 1], color=GRID, linewidth=1.5, zorder=1)
        for method, color, label in models:
            p, y, _ = cache.read(task, method)["test"]
            # Bins with fewer than MIN_BIN examples are dropped: a 5-example bin is noise, not calibration.
            bins = [b for b in metrics.reliability(p, y, bins=10) if b["n"] >= MIN_BIN]
            x = [b["confidence"] for b in bins]; a = [b["accuracy"] for b in bins]
            ax.plot(x, a, color=color, marker="o", markersize=5, zorder=3, label=f"{label}  ECE {metrics.ece(p, y):.2f}")
        ax.legend(loc="upper left", fontsize=8.5, handlelength=1.4)
        ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02); ax.set_aspect("equal")
        ax.set_title(TASKS[task], loc="left")
        ax.set_xlabel("Stated confidence")
    axes[0].set_ylabel("Actual accuracy")
    fig.suptitle("Does “90% sure” mean right 90% of the time? (raw, before any fix)", x=0.01, ha="left",
                 fontsize=12.5, fontweight="bold", color=INK)
    fig.text(0.01, -0.06, "Points below the diagonal are over-confident. 10 equal-width confidence bins, bins with "
             f"fewer than {MIN_BIN} examples omitted; ECE = expected calibration error over all examples "
             "(lower is better).", fontsize=8, color=INK_2, ha="left")
    fig.tight_layout()
    out = ROOT / "figures" / "reliability.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--latency", default=str(ROOT / "results" / "latency.json"))
    a = ap.parse_args()
    for f in (fig_speed(Path(a.latency)), fig_labels(), fig_reliability()):
        print("wrote", f.relative_to(ROOT))
