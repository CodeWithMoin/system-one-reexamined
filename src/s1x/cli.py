"""s1x splits <task> | s1x run <method> <task> [--limit N] | s1x report [tasks...]"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from s1x import cache, metrics
from s1x.tasks import TASKS, build_splits, load_task

RESULTS = Path(__file__).resolve().parents[2] / "results"
ZERO_SHOT = ("majority", "laya", "laya-ml", "nli", "nli-base", "embed-zs", "embed-zs-base")


def _runner(method: str, batch: int = 1):
    if method in ("laya", "laya-ml"):
        from s1x.runners.laya import LayaRunner
        return LayaRunner(method, batch_states=batch)
    if method in ("nli", "nli-base"):
        from s1x.runners.nli import MODELS, NLIRunner
        return NLIRunner(batch_examples=batch, model=MODELS[method])
    if method in ("embed-zs", "embed-zs-base"):
        from s1x.runners.embed_zs import EmbedZSRunner
        return EmbedZSRunner(method, batch_examples=batch)
    if method == "laya-torch":
        from s1x.runners.laya_torch import LayaTorchRunner
        return LayaTorchRunner(batch_states=batch)
    raise SystemExit(f"unknown method {method}")


def run(method: str, task_name: str, limit: int | None, batch: int = 1) -> None:
    data = load_task(task_name)
    rows: list[dict] = []
    if method == "majority":
        prior = np.bincount(data.pool.labels, minlength=len(data.labels)) / len(data.pool.labels)
        for split_name, split in (("test", data.test), ("calib", data.calib)):
            for i, y in enumerate(split.labels[:limit]):
                rows.append({"split": split_name, "i": i, "y": int(y), "p": prior.tolist(), "ms": 0.0})
    else:
        runner = _runner(method, batch)
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


def fewshot(method: str, task_names: list[str], ks: list[int], seeds: list[int], force: bool = False,
            max_steps: int | None = None, no_cache: bool = False) -> None:
    """Train on each committed k-shot draw, predict test and calib, cache per (method, k, seed)."""
    import time

    if method == "embed-lr":
        from s1x.runners.fewshot import EmbedLR
        runner = EmbedLR()
    elif method == "setfit":
        from s1x.runners.setfit import SetFitRunner
        runner = SetFitRunner(max_steps=max_steps)
    else:
        raise SystemExit(f"unknown few-shot method {method}")
    for name in task_names:
        data = load_task(name)
        for k in ks:
            for seed in seeds:
                shot = f"k={k},seed={seed}"
                if cache.exists(name, method, shot) and not (force or no_cache):
                    print(f"  skip {method} {name} {shot} (cached)")
                    continue
                start = time.perf_counter()
                out = runner.fit_predict(data, shot) if method == "embed-lr" else runner.fit_predict(data, shot, seed)
                rows = []
                for split_name, split in (("test", data.test), ("calib", data.calib)):
                    ms = out["ms"] if split_name == "test" else []
                    for i, (y, row) in enumerate(zip(split.labels, out[split_name])):
                        t = float(ms[i]) if i < len(ms) else None
                        rows.append({"split": split_name, "i": i, "y": int(y),
                                     "p": [round(float(v), 6) for v in row],
                                     "ms": None if t is None else round(t, 2)})
                if not no_cache:
                    cache.write(name, method, rows, shot=shot)
                print(f"  {method} {name} {shot}: acc {metrics.accuracy(out['test'], data.test.labels):.3f} "
                      f"({time.perf_counter() - start:.0f}s)", flush=True)


def overflow(task_names: list[str]) -> None:
    """Count test inputs that do not fit Laya's 512-token context (reported, not cut by us)."""
    from s1x.runners.laya import LayaRunner

    runner = LayaRunner("laya")
    out = {}
    for name in task_names:
        data = load_task(name)
        out[name] = runner.overflow(data.task, data.labels, data.test.texts)
        print(name, out[name])
    path = RESULTS / "laya_overflow.json"
    existing = json.loads(path.read_text()) if path.exists() else {}
    path.write_text(json.dumps(existing | out, indent=1))
    print(f"wrote {path}")


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
    from s1x import report as main_report
    rows = main_report.write(task_names)
    print(f"main table: {len(rows)} rows -> {RESULTS / 'main_table.json'}, {RESULTS / 'RESULTS.md'}")


def main() -> None:
    ap = argparse.ArgumentParser(prog="s1x")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("splits"); sp.add_argument("tasks", nargs="+")
    rp = sub.add_parser("run"); rp.add_argument("method"); rp.add_argument("task"); rp.add_argument("--limit", type=int)
    rp.add_argument("--batch", type=int, default=1, help="examples per forward pass (latency then = pass time / batch)")
    sub.add_parser("latency")
    sub.add_parser("order")
    fp = sub.add_parser("fewshot"); fp.add_argument("method"); fp.add_argument("tasks", nargs="+")
    fp.add_argument("--k", type=int, nargs="+", default=[8, 16, 64]); fp.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    fp.add_argument("--force", action="store_true")
    fp.add_argument("--max-steps", type=int, help="setfit: override the contrastive step cap (smoke tests)")
    fp.add_argument("--no-cache", action="store_true", help="train and score but do not write predictions")
    op = sub.add_parser("overflow"); op.add_argument("tasks", nargs="*", default=list(TASKS))
    pp = sub.add_parser("report"); pp.add_argument("tasks", nargs="*", default=list(TASKS))
    a = ap.parse_args()
    if a.cmd == "splits":
        for t in a.tasks:
            r = build_splits(t)
            print(f"{t}: test {len(r['test_ids'])}, calib {len(r['calib_ids'])}, pool {len(r['pool_ids'])}, labels {len(r['labels'])}")
    elif a.cmd == "run":
        run(a.method, a.task, a.limit, a.batch)
    elif a.cmd == "latency":
        from s1x import latency
        result = latency.run()
        print(json.dumps(result, indent=1))
        print(f"wrote {latency.write(result)}")
    elif a.cmd == "fewshot":
        fewshot(a.method, a.tasks, a.k, a.seeds, a.force, a.max_steps, a.no_cache)
    elif a.cmd == "order":
        from s1x import robustness
        robustness.run()
    elif a.cmd == "overflow":
        overflow(a.tasks)
    else:
        report(a.tasks)
