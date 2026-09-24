"""s1x splits <task> | s1x run <method> <task> [--limit N] | s1x report [tasks...]"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from s1x import cache, metrics
from s1x.tasks import TASKS, build_splits, load_task

RESULTS = Path(__file__).resolve().parents[2] / "results"
ZERO_SHOT = ("majority", "laya", "laya-ml", "nli")


def _runner(method: str):
    if method in ("laya", "laya-ml"):
        from s1x.runners.laya import LayaRunner
        return LayaRunner(method)
    if method == "nli":
        from s1x.runners.nli import NLIRunner
        return NLIRunner()
    raise SystemExit(f"unknown method {method}")


def run(method: str, task_name: str, limit: int | None) -> None:
    data = load_task(task_name)
    rows: list[dict] = []
    if method == "majority":
        prior = np.bincount(data.pool.labels, minlength=len(data.labels)) / len(data.pool.labels)
        for split_name, split in (("test", data.test), ("calib", data.calib)):
            for i, y in enumerate(split.labels[:limit]):
                rows.append({"split": split_name, "i": i, "y": int(y), "p": prior.tolist(), "ms": 0.0})
    else:
        runner = _runner(method)
        for split_name, split in (("test", data.test), ("calib", data.calib)):
            texts, ys = split.texts[:limit], split.labels[:limit]
            p, conf, ms = runner.predict(data.task, data.labels, texts)
            for i, (y, row, c, t) in enumerate(zip(ys, p, conf, ms)):
                rows.append({"split": split_name, "i": i, "y": int(y), "p": [round(float(v), 6) for v in row],
                             "conf": None if np.isnan(c) else float(c), "ms": round(float(t), 2)})
            print(f"  {method} {task_name} {split_name}: {len(texts)} done, "
                  f"acc {metrics.accuracy(p, ys):.3f}, median {np.median(ms):.1f} ms")
    out = cache.write(task_name, method, rows)
    print(f"wrote {out}")


def report(task_names: list[str]) -> None:
    table = []
    for name in task_names:
        ordinal = TASKS[name].kind == "score"
        for method in ZERO_SHOT:
            if not cache.exists(name, method):
                continue
            d = cache.read(name, method)
            p, y, ms = d["test"]
            pc, yc, _ = d.get("calib", (None, None, None))
            s = metrics.summary(p, y, p_calib=pc, y_calib=yc, ordinal_task=ordinal)
            s |= {"task": name, "method": method, "p50_ms": float(np.nanmedian(ms)), "p95_ms": float(np.nanpercentile(ms, 95))}
            table.append(s)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "zero_shot.json").write_text(json.dumps(table, indent=1))
    cols = ["task", "method", "n", "accuracy", "macro_f1", "ece", "ece_ts", "sel_acc@80", "p50_ms"]
    print(" | ".join(f"{c:>10}" for c in cols))
    for s in table:
        print(" | ".join(f"{s.get(c, ''):>10.3f}" if isinstance(s.get(c), float) else f"{str(s.get(c, '')):>10}" for c in cols))


def main() -> None:
    ap = argparse.ArgumentParser(prog="s1x")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("splits"); sp.add_argument("tasks", nargs="+")
    rp = sub.add_parser("run"); rp.add_argument("method"); rp.add_argument("task"); rp.add_argument("--limit", type=int)
    pp = sub.add_parser("report"); pp.add_argument("tasks", nargs="*", default=list(TASKS))
    a = ap.parse_args()
    if a.cmd == "splits":
        for t in a.tasks:
            r = build_splits(t)
            print(f"{t}: test {len(r['test_ids'])}, calib {len(r['calib_ids'])}, pool {len(r['pool_ids'])}, labels {len(r['labels'])}")
    elif a.cmd == "run":
        run(a.method, a.task, a.limit)
    else:
        report(a.tasks)
