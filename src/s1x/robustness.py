"""Option-order robustness: does the answer change when the options are listed in another order?

On the first 200 test examples of AG News and Emotion, each method is re-run with the option
list in 3 fixed permutations (the same ones for every method). Flip rate = share of examples
whose argmax label differs from the original order. NLI scores each option in its own
(text, hypothesis) pair, so it is order-invariant by construction up to numerical noise; Laya
reads all options in one sequence, so order can matter.
"""
from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np

from s1x.tasks import load_task

RESULTS = Path(__file__).resolve().parents[2] / "results"
N = 200
TASKS = ("ag_news", "emotion")


def permutations(c: int) -> list[list[int]]:
    rng = np.random.default_rng(20260924)
    return [list(range(c))[::-1], list(range(1, c)) + [0], rng.permutation(c).tolist()]


def run(methods=("laya", "nli")) -> dict:
    out: dict = {"n": N, "note": "flip = argmax label differs from the original option order"}
    for method in methods:
        if method == "laya":
            from s1x.runners.laya import LayaRunner
            runner = LayaRunner("laya")
        else:
            from s1x.runners.nli import NLIRunner
            runner = NLIRunner()
        for task in TASKS:
            data = load_task(task)
            texts, labels = data.test.texts[:N], data.labels
            base, _, _ = runner.predict(data.task, labels, texts)
            flips, any_flip = [], np.zeros(N, dtype=bool)
            for perm in permutations(len(labels)):
                p, _, _ = runner.predict(data.task, tuple(labels[i] for i in perm), texts)
                back = np.empty_like(p)
                back[:, perm] = p  # column j of p is label perm[j]
                changed = back.argmax(1) != base.argmax(1)
                flips.append(float(changed.mean()))
                any_flip |= changed
            out[f"{method}/{task}"] = {"permutations": permutations(len(labels)), "flip_rate": flips,
                                      "mean_flip_rate": float(np.mean(flips)), "any_flip_rate": float(any_flip.mean())}
            print(method, task, out[f"{method}/{task}"])
        del runner
        gc.collect()
    (RESULTS / "option_order.json").write_text(json.dumps(out, indent=1))
    return out
