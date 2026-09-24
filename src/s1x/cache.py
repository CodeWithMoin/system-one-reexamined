"""Per-example predictions on disk, so a model is run once and every metric reads the file.

results/preds/<task>/<method>[__k=K,seed=S].jsonl — one line per example:
{"split": "test"|"calib", "i": row, "y": gold, "p": [probabilities in label order], "ms": latency}
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

PREDS = Path(__file__).resolve().parents[2] / "results" / "preds"


def path(task: str, method: str, shot: str | None = None) -> Path:
    name = method if shot is None else f"{method}__{shot}"
    return PREDS / task / f"{name}.jsonl"


def write(task: str, method: str, rows: list[dict], shot: str | None = None) -> Path:
    out = path(task, method, shot)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return out


def read(task: str, method: str, shot: str | None = None) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """split -> (probabilities (n, C), labels (n,), latency ms (n,))."""
    rows = [json.loads(line) for line in path(task, method, shot).read_text().splitlines()]
    out = {}
    for split in ("test", "calib"):
        sel = [r for r in rows if r["split"] == split]
        if sel:
            out[split] = (np.array([r["p"] for r in sel]), np.array([r["y"] for r in sel]),
                          np.array([r.get("ms", np.nan) for r in sel], dtype=float))
    return out


def exists(task: str, method: str, shot: str | None = None) -> bool:
    return path(task, method, shot).exists()
