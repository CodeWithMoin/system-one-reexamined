"""Contamination probe: accuracy on fresh (post-2026-09-16) news vs the original AG News test set.

Every model sees the same AG News question and labels. Fresh text is "title. description", the same
shape as AG News. Predictions are cached under results/preds/ag_news_fresh/.
  python scripts/fresh_eval.py run <method>        # laya | jev | nli-c | nli-base-c | nli | embed-lr | setfit
  python scripts/fresh_eval.py report
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from s1x import cache, metrics  # noqa: E402
from s1x.tasks import load_task  # noqa: E402

FRESH = ROOT / "data" / "fresh" / "ag_news_fresh.jsonl"
TASK = "ag_news_fresh"


def fresh(label_key: str = "label"):
    rows = [json.loads(line) for line in FRESH.read_text().splitlines()]
    rows = [r for r in rows if r["verified"] != "dropped"]
    data = load_task("ag_news")
    idx = {name: i for i, name in enumerate(data.labels)}
    texts = [f"{r['title']}. {r['description']}".strip() for r in rows]
    return data, texts, np.array([idx[r[label_key]] for r in rows]), np.array([idx[r["section_label"]] for r in rows])


def run(method: str) -> None:
    data, texts, y, _ = fresh()
    if method in ("embed-lr", "setfit"):
        preds = []
        for seed in range(5):
            shot = f"k=64,seed={seed}"
            if method == "embed-lr":
                from s1x.runners.fewshot import EmbedLR
                r = EmbedLR()
                ids = data.shot_ids[shot]
                from sklearn.linear_model import LogisticRegression
                clf = LogisticRegression(max_iter=5000).fit(r.encode([data.pool.texts[i] for i in ids]), data.pool.labels[ids])
                preds.append(clf.predict_proba(r.encode(texts)))
            else:
                from s1x.runners.setfit import SetFitRunner
                import copy
                d2 = copy.copy(data); d2.test = type(data.test)(texts, y)
                out = SetFitRunner().fit_predict(d2, shot, seed)
                preds.append(out["test"])
        p = np.mean(preds, axis=0)
    else:
        sys.argv = [sys.argv[0]]
        from s1x.cli import _runner
        p, _, _ = _runner(method).predict(data.task, data.labels, texts)
    rows = [{"split": "test", "i": i, "y": int(t), "p": [round(float(v), 6) for v in row]} for i, (t, row) in enumerate(zip(y, p))]
    cache.write(TASK, method, rows)
    print(f"{method}: fresh acc {metrics.accuracy(p, y):.3f} on {len(y)}")


def report() -> None:
    data, texts, y, y_section = fresh()
    rng = np.random.default_rng(0)
    out = []
    for method in ("laya", "jev", "nli-c", "nli-base-c", "nli", "embed-lr", "setfit"):
        if not cache.exists(TASK, method):
            continue
        pf = np.array(cache.read(TASK, method)["test"][0])
        if method in ("embed-lr", "setfit"):
            orig = np.mean([metrics.accuracy(*cache.read("ag_news", method, f"k=64,seed={s}")["test"][:2]) for s in range(5)])
        else:
            orig = metrics.accuracy(*cache.read("ag_news", method)["test"][:2])
        cf = (pf.argmax(1) == y).astype(float)
        boot = [cf[rng.integers(0, len(cf), len(cf))].mean() for _ in range(10000)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        out.append({"method": method, "ag_news_test": orig, "fresh": cf.mean(), "fresh_ci": [lo, hi],
                    "drop": orig - cf.mean(), "fresh_section_labels": float((pf.argmax(1) == y_section).mean())})
    print(f"{'method':12}{'AG News test':>13}{'fresh':>8}{'95% CI':>16}{'drop':>8}{'fresh (section labels)':>24}")
    for r in out:
        print(f"{r['method']:12}{r['ag_news_test']:13.3f}{r['fresh']:8.3f}   [{r['fresh_ci'][0]:.3f},{r['fresh_ci'][1]:.3f}]{r['drop']:8.3f}{r['fresh_section_labels']:24.3f}")
    (ROOT / "results" / "fresh_probe.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    {"run": lambda: run(sys.argv[2]), "report": report}[sys.argv[1]]()
