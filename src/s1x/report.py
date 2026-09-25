"""Main table: zero-shot runs plus k-shot runs aggregated over seeds (mean ± std).

Writes results/main_table.json and results/RESULTS.md. Temperatures for "ECE after
temperature" are fit on each run's calib split (metrics.summary), never on test.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from s1x import cache, metrics
from s1x.tasks import TASKS

RESULTS = Path(__file__).resolve().parents[2] / "results"
ZERO_SHOT = ("majority", "laya", "laya-ml", "nli", "nli-base", "embed-zs", "embed-zs-base", "jev")
FEW_SHOT = ("embed-lr", "setfit", "ft-ce")
PARITY_METHODS = ("setfit", "embed-lr")
METRICS = ("accuracy", "macro_f1", "ece", "ece_ts", "brier", "sel_acc@50", "sel_acc@80", "sel_acc@95",
           "p50_ms", "p95_ms", "mae", "qwk", "temperature")
TASK_NOTES = {
    "banking77": "Laya note: a choice question's options share `head_max_len` = 192 tokens, so with 77 labels "
                 "each option is cut to [MASK] + 3 tokens (30 of 77 label names are trimmed; 7 labels collapse "
                 "into 3 identical trimmed forms, e.g. \"top up by …\"), and the instruction is cut to "
                 "\"What is the customer asking\". laya-mlx rejected no options (77 markers kept). The vendor calls "
                 "0.425 an architectural ceiling. laya-mlx also clamps the shipped `choice:11+` temperature "
                 "(0.1006, a 10x sharpening) to 0.5. Data: mteb/banking77 (parquet copy of PolyAI/banking77).",
    "sms_spam": "Caveat: Laya's BENCHMARKS.md says email spam and phishing were in its training mix, so this is "
                "possibly in Laya's training distribution — not a clean zero-shot test. Laya answers a `noul` "
                "question and returns P(true) for \"Is this text message spam?\".",
    "sst5": "SST-5 is Laya's `score` type (ordinal, 5 levels); MAE and QWK are on the argmax level.",
}
SHOT_RE = re.compile(r"^(?P<method>.+)__k=(?P<k>\d+),seed=(?P<seed>\d+)\.jsonl$")


def _one(task: str, method: str, shot: str | None) -> dict:
    d = cache.read(task, method, shot)
    p, y, ms = d["test"]
    pc, yc, _ = d.get("calib", (None, None, None))
    s = metrics.summary(p, y, p_calib=pc, y_calib=yc, ordinal_task=TASKS[task].kind == "score")
    ok = ms[~np.isnan(ms)]
    s |= {"p50_ms": float(np.median(ok)) if len(ok) else None, "p95_ms": float(np.percentile(ok, 95)) if len(ok) else None,
          "_correct": (p.argmax(1) == y).astype(float)}
    return s


def _few_shot_runs(task: str) -> dict[tuple[str, int], dict[int, str]]:
    runs: dict[tuple[str, int], dict[int, str]] = {}
    folder = cache.PREDS / task
    if not folder.exists():
        return runs
    for f in sorted(folder.iterdir()):
        m = SHOT_RE.match(f.name)
        if m and m["method"] in FEW_SHOT:
            runs.setdefault((m["method"], int(m["k"])), {})[int(m["seed"])] = f"k={m['k']},seed={m['seed']}"
    return runs


def build(task_names: list[str]) -> list[dict]:
    rows = []
    for task in task_names:
        for method in ZERO_SHOT:
            if cache.exists(task, method):
                s = _one(task, method, None)
                rows.append({"task": task, "method": method, "k": 0, "seeds": 1, "n": s["n"], "_correct": s["_correct"],
                             **{m: {"mean": s[m], "std": 0.0} for m in METRICS if s.get(m) is not None}})
        runs = _few_shot_runs(task)
        for (method, k), shots in sorted(runs.items(), key=lambda kv: (FEW_SHOT.index(kv[0][0]), kv[0][1])):
            per_seed = {seed: _one(task, method, shot) for seed, shot in sorted(shots.items())}
            row = {"task": task, "method": method, "k": k, "seeds": len(per_seed), "seed_ids": sorted(per_seed),
                   "n": next(iter(per_seed.values()))["n"],
                   # per example: share of seeds that got it right (the bootstrap resamples examples)
                   "_correct": np.mean([s["_correct"] for s in per_seed.values()], axis=0)}
            for m in METRICS:
                vals = [s[m] for s in per_seed.values() if s.get(m) is not None]
                if vals:
                    row[m] = {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0}
            rows.append(row)
    for task in task_names:
        laya = next((r for r in rows if r["task"] == task and r["method"] == "laya"), None)
        if laya is None:
            continue
        for r in rows:
            if r["task"] == task and r is not laya:
                r["vs_laya"] = metrics.paired_bootstrap(r["_correct"], laya["_correct"])
    return rows


def parity(rows: list[dict], task: str) -> dict:
    laya = next((r for r in rows if r["task"] == task and r["method"] == "laya"), None)
    if laya is None:
        return {"laya_accuracy": None}
    target = laya["accuracy"]["mean"]
    out = {"laya_accuracy": target}
    for method in PARITY_METHODS:
        cand = sorted((r for r in rows if r["task"] == task and r["method"] == method), key=lambda r: r["k"])
        hit = next((r["k"] for r in cand if r["accuracy"]["mean"] >= target), None)
        out[method] = {"k": hit, "tried": [r["k"] for r in cand]}
    return out


def _fmt(cell: dict | None, digits: int = 3) -> str:
    if cell is None:
        return "–"
    if cell.get("std"):
        return f"{cell['mean']:.{digits}f} ± {cell['std']:.{digits}f}"
    return f"{cell['mean']:.{digits}f}"


def _delta(b: dict | None) -> str:
    if b is None:
        return "–"
    return f"{b['diff']:+.3f} [{b['lo']:+.3f}, {b['hi']:+.3f}]{' *' if b['excludes_zero'] else ''}"


def _ms(cell: dict | None) -> str:
    return "–" if cell is None else f"{cell['mean']:.1f}"


def markdown(rows: list[dict], task_names: list[str]) -> str:
    out = ["# Results", "",
           "Generated by `s1x report` from the cached predictions in `results/preds/`. Test = 1,000 fixed "
           "examples per task. k-shot rows are mean ± sample std over the seeds listed (k examples per class, "
           "draws committed in `data/splits/`). ECE uses 15 equal-width bins; \"ECE (TS)\" is after one "
           "temperature fit on the 500-example calib split. Sel@80 = accuracy on the 80% most-confident "
           "test examples. The p50 ms column in the per-task tables is each run's own per-example timing on the M1 "
           "(16 GB) and is NOT the headline speed number: those runs were taken while the benchmark itself was "
           "leaking GPU memory into swap (see Notes). Headline speed = the \"Latency\" sections below, measured on an "
           "M4 Pro (24 GB) with each method in its own process and swap flat throughout "
           "(`results/latency_m4.json`, memory log in `results/latency_m4_memory.log`). Δ acc vs Laya: paired bootstrap over the 1,000 test examples "
           "(10,000 resamples, fixed seed), percentile 95% CI; * = CI excludes 0. For k-shot rows each example's "
           "score is the share of the seeds that got it right (mean over seeds per example), so the CI covers "
           "test-set sampling, not seed-to-seed variation (that is the ± std).", ""]
    for task in task_names:
        trs = [r for r in rows if r["task"] == task]
        if not trs:
            continue
        ordinal = TASKS[task].kind == "score"
        head = ["Method", "k", "seeds", "Accuracy", "Δ acc vs Laya [95% CI]", "Macro-F1", "ECE", "ECE (TS)", "Sel@80", "p50 ms"]
        if ordinal:
            head += ["MAE", "QWK"]
        out += [f"## {task}", "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        for r in trs:
            cells = [r["method"], str(r["k"]), str(r["seeds"]), _fmt(r.get("accuracy")), _delta(r.get("vs_laya")),
                     _fmt(r.get("macro_f1")),
                     _fmt(r.get("ece")), _fmt(r.get("ece_ts")), _fmt(r.get("sel_acc@80")), _ms(r.get("p50_ms"))]
            if ordinal:
                cells += [_fmt(r.get("mae")), _fmt(r.get("qwk"))]
            out.append("| " + " | ".join(cells) + " |")
        if task in TASK_NOTES:
            out += ["", TASK_NOTES[task]]
        par = parity(rows, task)
        if par["laya_accuracy"] is not None:
            bits = []
            for method in PARITY_METHODS:
                info = par[method]
                if not info["tried"]:
                    bits.append(f"{method}: not run")
                elif info["k"] is not None:
                    bits.append(f"{method}: k={info['k']}")
                else:
                    bits.append(f"{method}: not reached by {max(info['tried'])}")
            hits = [par[m]["k"] for m in PARITY_METHODS if par[m]["tried"] and par[m]["k"] is not None]
            best = f"k={min(hits)} per class" if hits else (
                "not reached by 64" if any(par[m]["tried"] and max(par[m]["tried"]) >= 64 for m in PARITY_METHODS)
                else "not reached by the k values run so far")
            out += ["", f"**Labels to parity** (mean accuracy ≥ Laya zero-shot {par['laya_accuracy']:.3f}): "
                        f"{best} ({'; '.join(bits)})."]
        out.append("")

    # Headline latency: the M4 Pro run (one process per method, flat swap). Falls back to the M1 run.
    lat = RESULTS / "latency_m4.json"
    machine = "M4 Pro 24 GB, one process per method, swap flat"
    if not lat.exists():
        lat, machine = RESULTS / "latency.json", "M1 16 GB (taken under swap; provisional)"
    bench = json.loads(lat.read_text()).get("benchmark_v2", {}) if lat.exists() else {}
    torch_methods = [("laya-torch", "Laya (upstream PyTorch)"), ("nli", "NLI DeBERTa-v3-large"),
                     ("nli-base", "NLI DeBERTa-v3-base"), ("embed-zs", "embed-zs (bge-small)"),
                     ("embed-zs-base", "embed-zs-base (bge-base)"), ("embed-lr", "embed-lr (bge-small + LR)"),
                     ("setfit", "SetFit (mpnet-base + LR)")]
    for task in ("ag_news", "banking77"):
        b_ = bench.get(task)
        if not b_:
            continue
        out += [f"## Latency on {task} ({b_['options']} options; same {b_['n']} test examples, after warm-up, {machine})", "",
                "Primary comparison: everything on the same runtime (PyTorch, MPS). Single = one input per forward "
                "pass (NLI: all its hypotheses in that one pass); batched = several inputs per pass, per-example "
                "time = pass wall time / batch size. Few-shot rows time inference only.", "",
                "| Method | single p50 ms | single p95 ms | batched ms/example | batched examples/s | batch |",
                "|---|---|---|---|---|---|"]

        def row(label, key):
            s1, s2 = b_.get(f"{key}_single"), b_.get(f"{key}_batched")
            if not s1:
                return None
            bb = (s2 or {}).get("batch") or (s2 or {}).get("batch_states") or (s2 or {}).get("batch_examples") or "–"
            return (f"| {label} | {s1['p50_ms']:.1f} | {s1['p95_ms']:.1f} | "
                    f"{(s2['wall_s'] * 1000 / s2['n']) if s2 else float('nan'):.1f} | "
                    f"{s2['examples_per_s'] if s2 else float('nan'):.1f} | {bb} |")

        for key, label in torch_methods:
            r_ = row(label, key)
            if r_:
                out.append(r_)
        out += ["", "Other runtimes (not part of the same-framework comparison):", "",
                "| Method | single p50 ms | single p95 ms | batched ms/example | batched examples/s | batch |",
                "|---|---|---|---|---|---|"]
        for key, label in [("laya", "Laya on MLX (laya-mlx, optimised runtime)")]:
            r_ = row(label, key)
            if r_:
                out.append(r_)
        if "nli_pipeline_per_pair" in b_:
            s1 = b_["nli_pipeline_per_pair"]
            out.append(f"| NLI large via HF pipeline (1 pass per hypothesis, old runner) | {s1['p50_ms']:.1f} | "
                       f"{s1['p95_ms']:.1f} | – | {s1['examples_per_s']:.1f} | – |")
        agree = b_.get("laya-torch_single", {}).get("argmax_agree_vs_laya_mlx")
        if agree is not None:
            out += ["", f"laya-torch vs laya-mlx argmax agreement on these {b_['n']} examples: {agree:.3f}."]
        out.append("")

    out += ["## Accuracy vs latency", "",
            "Per task: test accuracy (few-shot: mean over seeds) against single-example p50 latency. Latency "
            "comes from the benchmark above where it covers the task, otherwise from the run's own per-example timings "
            "(for NLI on emotion that is the old per-pair HF pipeline, slower than the batched runner). "
            "★ = Pareto-optimal (no other point is both at least as accurate and at least as fast, and strictly better in one). "
            "Laya appears twice where both runtimes were timed: MLX (its optimised runtime) and PyTorch MPS (same "
            "runtime as every other method). Few-shot latency: encode + head.", ""]
    for task in task_names:
        pts = []
        for r in rows:
            if r["task"] != task or r["method"] == "majority" or "accuracy" not in r:
                continue
            key = f"{r['method']}_single"
            if task in bench and key in bench[task]:
                ms, src = bench[task][key]["p50_ms"], "benchmark"
            elif r.get("p50_ms"):
                ms, src = r["p50_ms"]["mean"], "run"
                if r["method"] == "nli" and task in ("ag_news", "emotion"):
                    src = "run (old per-pair HF pipeline)"
            else:
                continue
            name = r["method"] + (f" k={r['k']}" if r["k"] else "")
            if r["method"] == "laya":
                name = "laya (MLX)"
                torch_ = bench.get(task, {}).get("laya-torch_single")
                if torch_:
                    pts.append(("laya (PyTorch MPS)", r["accuracy"]["mean"], torch_["p50_ms"], "benchmark"))
            pts.append((name, r["accuracy"]["mean"], ms, src))
        if not pts:
            continue
        out += [f"**{task}**", "", "| Method | Accuracy | p50 ms | latency source | Pareto |", "|---|---|---|---|---|"]
        for name, acc, ms, src in sorted(pts, key=lambda x: x[2]):
            dominated = any((a2 >= acc and m2 <= ms) and (a2 > acc or m2 < ms) for n2, a2, m2, _ in pts if n2 != name)
            out.append(f"| {name} | {acc:.3f} | {ms:.1f} | {src} | {'' if dominated else '★'} |")
        out.append("")

    ov = RESULTS / "laya_overflow.json"
    if ov.exists():
        o = json.loads(ov.read_text())
        out += ["## Laya context (512 tokens): test inputs that do not fit", "",
                "| Task | text > 512 tokens | text > room left after question+options | room (tokens) | longest text (tokens) |",
                "|---|---|---|---|---|"]
        for task in task_names:
            if task in o:
                s = o[task]
                out.append(f"| {task} | {s['over_512']} | {s['state_truncated']} | {s['state_room']} | {s['max_state_tokens']} |")
        out.append("")
    oo = RESULTS / "option_order.json"
    if oo.exists():
        o = json.loads(oo.read_text())
        out += ["## Option-order robustness (first 200 test examples, 3 fixed permutations)", "",
                "Flip rate = share of examples whose predicted label changes versus the original option order.", "",
                "| Method / task | flip rate per permutation | mean | any of the 3 |", "|---|---|---|---|"]
        for key, s_ in o.items():
            if isinstance(s_, dict):
                out.append(f"| {key} | {', '.join(f'{v:.3f}' for v in s_['flip_rate'])} | {s_['mean_flip_rate']:.3f} | {s_['any_flip_rate']:.3f} |")
        out.append("")

    td = RESULTS / "typed_decisions.json"
    if td.exists():
        out += ["## On Laya's own benchmark", "", json.loads(td.read_text()).get("markdown", ""), ""]

    out += ["## Notes", "",
            "- Laya's `confidence` field is 1 − normalised entropy, not a probability of being right; every "
            "calibration number here uses Laya's per-option probabilities instead.",
            "- embed-zs / embed-zs-base: probabilities are softmax(cosine / τ) with τ = 0.05 fixed before any "
            "result was seen; their raw ECE reflects that arbitrary τ, and \"ECE (TS)\" (temperature fit on calib) "
            "is the comparable number, exactly as for every other method.",
            "- embed-lr's high raw ECE is under-confidence: an L2-regularised logistic regression on unit-norm "
            "embeddings from a handful of examples gives flat probabilities; temperature scaling on calib fixes it.",
            "- nli-xsmall: skipped, no `deberta-v3-xsmall-zeroshot-v2.0` checkpoint exists on the Hub.",
            "- Jev: TypeSafe API (`jev-latest`, reported `jev-1.13.0`), zero-shot, same instruction and label wording as "
            "Laya; its latency is an API round trip including the network (~830 ms p50 from India) and is never mixed "
            "into the local speed tables. Jev returns exact 0/1 probabilities on many items (62% of AG News "
            "predictions); where it said 1.0 it was right 77% of the time on Emotion and 79% on SST-5.",
            "- Latency history: the first benchmark loaded every model into one process; MPS/MLX memory is not freed "
            "by `del` + `gc.collect()`, so it grew to a 26 GB footprint on a 24 GB M4 and ~11 GB of swap. The M1's "
            "\"memory pressure\" was almost certainly the same leak. Fixed by running each (task, method) in its own "
            "process; the M1 numbers in `results/latency.json` are kept for the record only.",
            "- Related work: nibzard/decision-model-benchmark compares Jev with 8 LLMs and trivial baselines "
            "(no trained classifiers); this repo adds the classifiers we already had.", ""]
    return "\n".join(out)


def write(task_names: list[str]) -> list[dict]:
    rows = build(task_names)
    pars = {t: parity(rows, t) for t in task_names}
    RESULTS.mkdir(exist_ok=True)
    clean = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
    (RESULTS / "main_table.json").write_text(json.dumps({"rows": clean, "labels_to_parity": pars}, indent=1))
    (RESULTS / "RESULTS.md").write_text(markdown(rows, task_names))
    return rows
