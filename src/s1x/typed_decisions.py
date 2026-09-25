"""Laya's own benchmark: `LocalLLaMA/typed-decisions`, config "all" (train 1,200 cases, test 400).

Each case is one state with five typed questions (choice / score / noul). We evaluate per
*decision* (one question about one state): 2,000 test decisions.

Scoring follows Laya's own eval (research/scripts/bench_local.py, `build_typed_decisions` +
`metrics`): options are indexed as choice -> criteria keys in order, noul -> [false, true],
score -> levels 0..n-1; gold index = the gold `label`; a decision is correct when the argmax of the
predicted distribution is the gold index; ECE is 15 equal-width bins on the top probability.
Their fine-tuning notebook scores noul as p(true) >= 0.5, which differs from argmax only on an
exact 0.5 tie (counted in the report).

Splits: test = the dataset's test split. calib = a fixed 20% of TRAIN cases (stratified by
workflow, seed 20260924) used only to fit one temperature per method; never test.

Predictions: results/preds/typed_decisions/<method>.jsonl, one line per decision:
{"split", "i" (case row), "case", "workflow", "qid", "qtype", "y", "p" (options in index order)}.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

from s1x import cache
from s1x.tasks import DATA, SPLIT_SEED

TASK = "typed_decisions"
HF_ID, CONFIG = "LocalLLaMA/typed-decisions", "all"
RESULTS = Path(__file__).resolve().parents[2] / "results"
CALIB_FRACTION = 0.2
METHODS = ("majority", "laya", "laya-td", "nli-base", "embed-lr", "embed-lr-slot", "jev")

PUBLISHED = {  # NandhaKishorM/laya BENCHMARKS.md, "typed-decisions — 400 cases, 2,000 decisions"
    "laya-td": {"accuracy": 0.766, "ece": 0.213, "soft_accuracy": 0.471, "score_mae": 0.242,
                "by_workflow": {"agent_trace_observability": 0.730, "customer_service": 0.764,
                                "invoice_processing": 0.804, "security_incidents": 0.766}},
    "laya": {"accuracy": 0.361, "ece": 0.175, "soft_accuracy": 0.332, "score_mae": 0.694},
    "laya-ml": {"accuracy": 0.352, "ece": 0.314, "soft_accuracy": 0.328, "score_mae": 0.760},
    "jev": {"accuracy": 0.727, "ece": 0.144, "soft_accuracy": 0.580, "score_mae": 0.391},
    "majority": {"accuracy": 0.461},
    "random": {"accuracy": 0.318},
}


@dataclass
class Decision:
    split: str
    i: int  # case row in its HF split
    case: str
    workflow: str
    qid: str
    qtype: str
    state: str  # raw JSON string as stored
    qdef: dict
    options: list[str]  # option keys in index order
    y: int
    soft: np.ndarray  # gold probabilities in index order
    gold_score: float | None


def _options(qdef: dict) -> list[str]:
    t, crit = qdef["type"], qdef.get("criteria")
    if t == "choice":
        return list(crit.keys()) if isinstance(crit, dict) else list(crit)
    if t == "noul":
        return ["false", "true"]
    return [str(i) for i in range(len(crit))]


def _decisions(ds, split: str, rows: list[int]) -> list[Decision]:
    out = []
    for i in rows:
        r = ds[i]
        qs, gold = json.loads(r["questions"]), json.loads(r["gold"])
        for qid, qdef in qs.items():
            g, opts = gold[qid], _options(qdef)
            label = str(g["label"]).lower() if qdef["type"] == "noul" else str(g["label"])
            probs = g.get("probabilities", {})
            soft = np.array([float(probs.get(o, 0.0)) for o in opts])
            out.append(Decision(split, i, r["id"], r["workflow"], qid, qdef["type"], r["state"], qdef, opts,
                                opts.index(label), soft / max(soft.sum(), 1e-12),
                                float(g.get("score", g["label"])) if qdef["type"] == "score" else None))
    return out


def _splits(train) -> dict:
    path = DATA / "splits" / f"{TASK}.json"
    if path.exists():
        return json.loads(path.read_text())
    rng = np.random.default_rng(SPLIT_SEED)
    wf = np.array(train["workflow"])
    calib = []
    for w in sorted(set(wf)):
        members = np.flatnonzero(wf == w)
        calib.extend(rng.choice(members, size=int(round(len(members) * CALIB_FRACTION)), replace=False).tolist())
    calib = sorted(int(i) for i in calib)
    record = {"task": TASK, "hf_id": HF_ID, "config": CONFIG, "test_source": "test (all 400 cases)",
              "calib_ids": calib, "fit_ids": [i for i in range(len(train)) if i not in set(calib)]}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record))
    return record


def load() -> dict[str, list[Decision]]:
    """split -> decisions. Splits: test, calib (20% of train cases), fit (other 80%), train (all)."""
    from datasets import load_dataset

    ds = load_dataset(HF_ID, CONFIG)
    rec = _splits(ds["train"])
    return {"test": _decisions(ds["test"], "test", list(range(len(ds["test"])))),
            "calib": _decisions(ds["train"], "calib", rec["calib_ids"]),
            "fit": _decisions(ds["train"], "fit", rec["fit_ids"]),
            "train": _decisions(ds["train"], "train", list(range(len(ds["train"]))))}


def _row(d: Decision, p) -> dict:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, None)
    p = p / p.sum()
    return {"split": d.split, "i": d.i, "case": d.case, "workflow": d.workflow, "qid": d.qid, "qtype": d.qtype,
            "y": d.y, "p": [round(float(v), 6) for v in p]}


def _by_case(decisions: list[Decision]) -> dict[str, list[Decision]]:
    out: dict[str, list[Decision]] = {}
    for d in decisions:
        out.setdefault(d.case, []).append(d)
    return out


# ------------------------------------------------------------------ contenders

def run_majority(data) -> list[dict]:
    """Per (workflow, question) prior of the hard gold labels on the full train split; argmax = the
    most frequent label. Ignores the state."""
    counts: dict = {}
    for d in data["train"]:
        counts.setdefault((d.workflow, d.qid), np.zeros(len(d.options)))[d.y] += 1
    return [_row(d, counts[(d.workflow, d.qid)] / counts[(d.workflow, d.qid)].sum())
            for split in ("test", "calib") for d in data[split]]


def run_laya(data, method: str) -> list[dict]:
    """laya-mlx, one `predict(state, questions)` call per case (all five questions in one call), as
    Laya's fine-tuning notebook does. State is passed as the parsed dict, as their eval does."""
    import warnings

    import laya_mlx

    repo = {"laya": "aac6fef/laya-mlx", "laya-td": "aac6fef/laya-typed-decisions-mlx"}[method]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        agent = laya_mlx.load(repo)
    rows = []
    for split in ("test", "calib"):
        cases = _by_case(data[split])
        for n, ds in enumerate(cases.values()):
            answers = agent.predict(json.loads(ds[0].state), {d.qid: d.qdef for d in ds})["answers"]
            for d in ds:
                a = answers[d.qid]
                if d.qtype == "noul":
                    p = [1.0 - float(a["noul"]), float(a["noul"])]
                else:
                    p = [float(a["probabilities"][o]) for o in d.options]
                rows.append(_row(d, p))
            if n % 50 == 0:
                print(f"  {method} {split}: {n}/{len(cases)} cases", flush=True)
    return rows


def run_jev(data) -> list[dict]:
    """Jev via the TypeSafe API: one call per case with all five questions, state as the parsed dict."""
    from concurrent.futures import ThreadPoolExecutor

    from s1x.runners.jev import MODEL, WORKERS, JevRunner

    runner = JevRunner()
    rows, versions = [], set()
    for split in ("test", "calib"):
        cases = list(_by_case(data[split]).values())

        def call(ds):
            body = {"state": json.loads(ds[0].state), "model": MODEL, "questions": {d.qid: d.qdef for d in ds}}
            return _post(runner, body)

        with ThreadPoolExecutor(WORKERS) as pool:
            outs = list(pool.map(call, cases))
        for ds, out in zip(cases, outs):
            versions.add(out.get("model", "?"))
            for d in ds:
                a = out["answers"][d.qid]
                p = [1.0 - float(a["noul"]), float(a["noul"])] if d.qtype == "noul" else \
                    [float(a["probabilities"].get(o, 0.0)) for o in d.options]
                rows.append(_row(d, p) | {"model": out.get("model", "?")})
        print(f"  jev {split}: {len(cases)} cases, versions {sorted(versions)}", flush=True)
    return rows


def _post(runner, body: dict) -> dict:
    """JevRunner._call, but with an arbitrary questions dict and a structured state."""
    import time
    import urllib.error
    import urllib.request

    from s1x.runners.jev import URL

    data, delay = json.dumps(body).encode(), 1.0
    for attempt in range(8):
        req = urllib.request.Request(URL, data=data, headers={"Authorization": f"Bearer {runner.key}",
                                                              "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in (429, 529, 500, 502, 503, 504) and attempt < 7:
                time.sleep(delay); delay = min(delay * 2, 30); continue
            raise RuntimeError(f"Jev HTTP {e.code}: {e.read().decode()[:200]}") from None
        except (urllib.error.URLError, TimeoutError):
            if attempt < 7:
                time.sleep(delay); delay = min(delay * 2, 30); continue
            raise
    raise RuntimeError("unreachable")


def _lower_first(s: str) -> str:
    return s[:1].lower() + s[1:]


def nli_hypotheses(d: Decision) -> list[str]:
    """Written once, before any result: choice/score -> "<instructions> Answer: <option>", with the
    option rendered as Laya renders it (choice "key: description", score the level text); noul ->
    "It is false that <statement>" / "It is true that <statement>"."""
    q, crit = d.qdef["instructions"], d.qdef.get("criteria")
    if d.qtype == "noul":
        return [f"It is false that {_lower_first(q)}", f"It is true that {_lower_first(q)}"]
    if d.qtype == "score":
        return [f"{q} Answer: {c}" for c in crit]
    return [f"{q} Answer: {k}" + (f": {v}" if v else "") for k, v in crit.items()]


def run_nli(data, method: str = "nli-base") -> list[dict]:
    """Zero-shot NLI (existing runner's model and scoring): premise = state, one hypothesis per option,
    entailment logits softmaxed across the options of that decision."""
    from s1x.runners.nli import MODELS, NLIRunner

    runner = NLIRunner(model=MODELS[method])
    rows = []
    for split in ("test", "calib"):
        for n, d in enumerate(data[split]):
            z = runner._score([d.state], nli_hypotheses(d))[0]
            p = np.exp(z - z.max())
            rows.append(_row(d, p / p.sum()))
            if n % 500 == 0:
                print(f"  {method} {split}: {n}/{len(data[split])}", flush=True)
    return rows


def pair_text(d: Decision, k: int) -> str:
    """Question and option first, so the 512-token encoder never truncates them away."""
    q, crit = d.qdef["instructions"], d.qdef.get("criteria")
    if d.qtype == "choice":
        v = crit[d.options[k]]
        opt = d.options[k] + (f": {v}" if v else "")
    elif d.qtype == "score":
        opt = f"level {k}: {crit[k]}"
    else:
        opt = ["false", "true"][k] + (f": {crit[['false', 'true'][k]]}" if crit else "")
    return f"Question: {q}\nAnswer: {opt}\nState: {d.state}"


def run_embed_lr(data) -> list[dict]:
    """Supervised cross-feature baseline: bge-small embeds each (state + question + option) text; one
    binary logistic regression "is this the gold option" is trained over every option of every train
    decision; per decision the option logits are softmaxed and the argmax is the prediction.
    Test predictions come from the model trained on the full train split; calib predictions (for the
    temperature only) come from a second model trained on the other 80% of train, so calib is held out."""
    from sklearn.linear_model import LogisticRegression

    from s1x.runners.fewshot import EmbedLR

    enc = EmbedLR()

    def embed(decisions):
        texts = [pair_text(d, k) for d in decisions for k in range(len(d.options))]
        x = enc.model.encode(texts, batch_size=32, normalize_embeddings=True, convert_to_numpy=True,
                             show_progress_bar=False)
        spans, s = [], 0
        for d in decisions:
            spans.append((s, s + len(d.options))); s += len(d.options)
        return x, spans

    emb = {}
    for split in ("train", "test"):
        emb[split] = embed(data[split])
        print(f"  embed-lr: encoded {split} ({len(emb[split][0])} option texts)", flush=True)
    calib_cases = {d.case for d in data["calib"]}

    def fit(keep):
        x, spans = emb["train"]
        xs, ys = [], []
        for d, (a, b) in zip(data["train"], spans):
            if keep(d):
                xs.append(x[a:b]); ys.extend(int(k == d.y) for k in range(b - a))
        return LogisticRegression(max_iter=5000).fit(np.concatenate(xs), np.array(ys))

    def predict(clf, x, spans, decisions):
        z = clf.decision_function(x)
        out = []
        for d, (a, b) in zip(decisions, spans):
            p = np.exp(z[a:b] - z[a:b].max())
            out.append(_row(d, p / p.sum()))
        return out

    full = fit(lambda d: True)
    rows = predict(full, *emb["test"], data["test"])
    held = fit(lambda d: d.case not in calib_cases)
    x, spans = emb["train"]
    calib_pairs = [(d, s) for d, s in zip(data["train"], spans) if d.case in calib_cases]
    calib_x = np.concatenate([x[a:b] for _, (a, b) in calib_pairs])
    calib_spans, s = [], 0
    for d, (a, b) in calib_pairs:
        calib_spans.append((s, s + b - a)); s += b - a
    calib_decisions = [Decision(**{**d.__dict__, "split": "calib"}) for d, _ in calib_pairs]
    rows += predict(held, calib_x, calib_spans, calib_decisions)
    return rows


def run_embed_lr_slot(data) -> list[dict]:
    """Per-slot classifier: the option set of every (workflow, question) slot is fixed in this dataset
    (checked: all 1,600 cases use identical question definitions), so the ordinary classifier recipe
    applies directly: bge-small embeds the state once, and one multinomial logistic regression per slot
    (20 heads) is trained on the train decisions of that slot. Test from full-train heads; calib from
    heads trained without the calib cases."""
    from sklearn.linear_model import LogisticRegression

    from s1x.runners.fewshot import EmbedLR, _full_proba

    enc = EmbedLR()
    states = sorted({d.state for split in ("train", "test") for d in data[split]})
    x = dict(zip(states, enc.encode(states)))
    calib_cases = {d.case for d in data["calib"]}
    rows = []
    for keep, target in ((lambda d: True, "test"), (lambda d: d.case not in calib_cases, "calib")):
        heads = {}
        for slot in sorted({(d.workflow, d.qid) for d in data["train"]}):
            ds = [d for d in data["train"] if (d.workflow, d.qid) == slot and keep(d)]
            heads[slot] = (LogisticRegression(max_iter=5000).fit(np.stack([x[d.state] for d in ds]),
                                                                  [d.y for d in ds]), len(ds[0].options))
        for d in data[target]:
            clf, k = heads[(d.workflow, d.qid)]
            rows.append(_row(d, _full_proba(clf, x[d.state][None], k)[0]))
    return rows


def run(method: str) -> None:
    data = load()
    if method == "majority":
        rows = run_majority(data)
    elif method in ("laya", "laya-td"):
        rows = run_laya(data, method)
    elif method == "nli-base":
        rows = run_nli(data, method)
    elif method == "embed-lr":
        rows = run_embed_lr(data)
    elif method == "embed-lr-slot":
        rows = run_embed_lr_slot(data)
    elif method == "jev":
        rows = run_jev(data)
    else:
        raise SystemExit(f"unknown method {method}")
    out = cache.write(TASK, method, rows)
    test = [r for r in rows if r["split"] == "test"]
    print(f"wrote {out}: test acc {np.mean([np.argmax(r['p']) == r['y'] for r in test]):.4f} on {len(test)}")


# ------------------------------------------------------------------ metrics

def _read(method: str) -> dict[str, list[dict]]:
    rows = [json.loads(line) for line in cache.path(TASK, method).read_text().splitlines()]
    return {s: [r for r in rows if r["split"] == s] for s in ("test", "calib")}


def _fit_temperature(rows: list[dict]) -> float:
    logs = [np.log(np.clip(np.array(r["p"]), 1e-12, 1.0)) for r in rows]
    ys = [r["y"] for r in rows]

    def nll(t):
        tot = 0.0
        for z, y in zip(logs, ys):
            z = z / t
            tot -= z[y] - (z.max() + np.log(np.exp(z - z.max()).sum()))
        return tot / len(ys)

    return float(minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded").x)


def _scale(p: list[float], t: float) -> np.ndarray:
    z = np.log(np.clip(np.array(p), 1e-12, 1.0)) / t
    e = np.exp(z - z.max())
    return e / e.sum()


def _ece(conf: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    """Same binning as metrics.ece and Laya's ece_score, from top-1 confidence and correctness."""
    edges, total = np.linspace(0, 1, bins + 1), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def _stats(rows: list[dict], soft: dict, gold_score: dict, t: float | None) -> dict:
    ps = [np.array(r["p"]) for r in rows]
    y = np.array([r["y"] for r in rows])
    pred = np.array([int(p.argmax()) for p in ps])
    correct = (pred == y).astype(float)
    conf = np.array([p.max() for p in ps])
    key = lambda r: (r["case"], r["qid"])  # noqa: E731
    out = {"n": len(rows), "accuracy": float(correct.mean()), "ece": _ece(conf, correct),
           "soft_accuracy": float(np.mean([(p * soft[key(r)]).sum() for p, r in zip(ps, rows)])),
           "mean_confidence": float(conf.mean())}
    scores = [(float((np.arange(len(p)) * p).sum()), gold_score[key(r)]) for p, r in zip(ps, rows)
              if r["qtype"] == "score"]
    if scores:
        out["score_mae"] = float(np.mean([abs(a - b) for a, b in scores]))
    if t is not None:
        pt = [_scale(r["p"], t) for r in rows]
        out["ece_ts"] = _ece(np.array([p.max() for p in pt]), correct)
    return out


def evaluate(method: str, data) -> dict:
    d = _read(method)
    soft = {(x.case, x.qid): x.soft for x in data["test"]}
    gold_score = {(x.case, x.qid): x.gold_score for x in data["test"]}
    t = _fit_temperature(d["calib"]) if d["calib"] else None
    test = d["test"]
    out = _stats(test, soft, gold_score, t) | {"temperature": t}
    out["by_workflow"] = {w: _stats([r for r in test if r["workflow"] == w], soft, gold_score, t)["accuracy"]
                          for w in sorted({r["workflow"] for r in test})}
    out["by_qtype"] = {q: _stats([r for r in test if r["qtype"] == q], soft, gold_score, t)["accuracy"]
                       for q in ("choice", "score", "noul")}
    out["by_question"] = {f"{w}/{q}": float(np.mean([np.argmax(r["p"]) == r["y"] for r in test
                                                      if r["workflow"] == w and r["qid"] == q]))
                          for w, q in sorted({(r["workflow"], r["qid"]) for r in test})}
    versions = sorted({r["model"] for r in test if "model" in r})
    if versions:
        out["version"] = ", ".join(versions)
    out["accuracy_noul_ge_0.5"] = float(np.mean([  # their notebook's noul rule: true iff p(true) >= 0.5
        (int(r["p"][1] >= 0.5) if r["qtype"] == "noul" else int(np.argmax(r["p"]))) == r["y"] for r in test]))
    out["noul_exact_ties"] = sum(1 for r in test if r["qtype"] == "noul" and r["p"][0] == r["p"][1])
    return out


LABELS = {"majority": "majority (per workflow×question, train)", "random": "random (expected, 1/#options)",
          "laya": "laya (base, zero-shot)", "laya-td": "laya-typed-decisions (fine-tuned on train)",
          "nli-base": "nli-base (DeBERTa-v3-base zero-shot NLI)",
          "embed-lr": "embed-lr (bge-small pair + one binary LR, full train)",
          "embed-lr-slot": "embed-lr-slot (bge-small state + one LR per slot, full train)", "jev": "jev (API, zero-shot)"}


def report() -> dict:
    data = load()
    res: dict = {"dataset": f"{HF_ID} ({CONFIG})", "n_test_decisions": len(data["test"]),
                 "n_calib_decisions": len(data["calib"]), "published": PUBLISHED, "methods": {}}
    res["methods"]["random"] = {"accuracy": float(np.mean([1 / len(d.options) for d in data["test"]]))}
    for m in METHODS:
        if cache.exists(TASK, m):
            res["methods"][m] = evaluate(m, data)
    res["laya_state_truncated"] = overflow()
    res["markdown"] = markdown(res)
    (RESULTS / "typed_decisions.json").write_text(json.dumps(res, indent=1))
    return res


def _f(x, nd=3):
    return "—" if x is None else f"{x:.{nd}f}"


def markdown(res: dict) -> str:
    M, P = res["methods"], PUBLISHED
    order = ["random", "majority", "laya", "laya-td", "nli-base", "embed-lr", "embed-lr-slot", "jev"]
    wfs = ["agent_trace_observability", "customer_service", "invoice_processing", "security_incidents"]
    lines = [
        f"`{HF_ID}`, config `{CONFIG}`: test = all 400 cases / {res['n_test_decisions']:,} decisions (5 typed "
        "questions per case, 4 workflows × 5 question slots). Scored per decision exactly as Laya's own eval "
        "(`research/scripts/bench_local.py`): options indexed choice → criteria keys in order, noul → [false, true], "
        "score → levels 0..n−1; correct when argmax = gold `label`; ECE = 15 equal-width bins on the top probability. "
        f"Temperatures are fit on a fixed 20% of *train* cases ({res['n_calib_decisions']:,} decisions), never on test.",
        "",
        "**Summary.** Laya's own numbers reproduce: base `laya` 0.361 (published 0.361) and `laya-typed-decisions` "
        "0.7665 (published 0.766, ECE 0.214 vs 0.213, and per workflow to within 0.002). Base Laya zero-shot is below the "
        "majority baseline and below a 2020-style zero-shot NLI model (`nli-base`, 0.485). The 0.766 belongs to a "
        "checkpoint fine-tuned on this benchmark's train split. A plain classifier trained on the same split "
        "(`embed-lr-slot`, 0.564) is 0.20 below it, and zero-shot Jev (0.736) is 0.03 below it. Among zero-shot models, "
        "only Jev clears the majority baseline by a wide margin.",
        "",
        "| method | labels used | accuracy (ours) | published | ECE (ours) | published ECE | ECE after TS | soft acc | score MAE |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    used = {"random": "0", "majority": "train prior", "laya": "0", "laya-td": "6,000 train decisions (their fine-tune)",
            "nli-base": "0", "embed-lr": "6,000 train decisions", "embed-lr-slot": "6,000 train decisions", "jev": "0"}
    for m in order:
        s = M.get(m)
        p = P.get(m, {})
        if s is None:
            lines.append(f"| {LABELS[m]} | {used[m]} | not run | {_f(p.get('accuracy'))} | | {_f(p.get('ece'))} | | | |")
            continue
        lines.append(f"| {LABELS[m]} | {used[m]} | **{_f(s['accuracy'])}** | {_f(p.get('accuracy'))} | {_f(s.get('ece'))} | "
                     f"{_f(p.get('ece'))} | {_f(s.get('ece_ts'))} | {_f(s.get('soft_accuracy'))} | {_f(s.get('score_mae'))} |")
    lines.append(f"| laya-multilingual | 0 | not run | {_f(P['laya-ml']['accuracy'])} | | {_f(P['laya-ml']['ece'])} | | | |")
    lines += ["", "Per workflow and per question type (accuracy):", "",
              "| method | " + " | ".join(w.replace("_", " ") for w in wfs) + " | choice | score | noul |",
              "|---|" + "---|" * (len(wfs) + 3)]
    for m in order:
        s = M.get(m)
        if s is None or "by_workflow" not in s:
            continue
        lines.append(f"| {m} | " + " | ".join(_f(s["by_workflow"].get(w)) for w in wfs) + " | "
                     + " | ".join(_f(s["by_qtype"][q]) for q in ("choice", "score", "noul")) + " |")
    lines.append("| laya-td, published | " + " | ".join(_f(P["laya-td"]["by_workflow"][w]) for w in wfs) + " | | | |")
    maj = M.get("majority", {}).get("accuracy")
    lines += [
        "",
        "**Majority baseline — our definition differs from theirs.** Ours: for each (workflow, question) slot, the most "
        "frequent gold `label` on the full train split, predicted for every test case (probabilities = the train label "
        f"frequencies of that slot). It scores **{_f(maj, 4)}** against the **0.461** Laya publishes. Laya's scripts "
        "hard-code 0.4610 as `per_question_majority_class` without code or a definition, so we could not reproduce it. "
        "We tried about ten definitions (per question id across workflows 0.4115, per question type 0.3105, prob-weighted "
        "priors 0.4785, scored against `label_agreement.argmax_majority` 0.483–0.4855, fit on test itself 0.5225); none "
        "gives 0.461. The dataset card itself gives two further, different floors: \"Prior\" 0.470 (per-question train "
        "frequencies) and \"Majority baseline\" 0.520 (measured over all 1,600 cases). The gap to 0.461 is 0.02 points "
        "and changes no conclusion below; we report our own number and state the definition.",
        "",
        "**Random**: expected accuracy of a uniform guess, mean of 1/#options over the test decisions — reproduces "
        "their 0.318 (0.3175).",
        "",
        "Other definition notes:",
        "- `laya-typed-decisions` was fine-tuned by Convai on this benchmark's train split, so it is a *specialist*; the "
        "dataset card warns that specialist and zero-shot (generalist) numbers are not comparable. Its temperature for "
        "\"ECE after TS\" is fit on train cases it was trained on, so that one TS number is optimistic about its held-out "
        "calibration. We ran the MLX FP16 conversion (`aac6fef/laya-typed-decisions-mlx`, source revision f9ab0b2); "
        "base `laya` is `aac6fef/laya-mlx`. Both via `agent.predict(state, questions)`, one call per case, state as the "
        "parsed JSON dict — the same call their notebook makes.",
        "- Laya's fine-tuning notebook scores noul as p(true) ≥ 0.5 rather than argmax; the two differ only on an exact "
        "0.5 tie (count per method in `typed_decisions.json`, `noul_exact_ties`).",
        "- Laya's published ECE is on the shipped temperatures; our \"ECE (ours)\" is the same (raw model output). "
        "\"ECE after TS\" is one temperature per method, fit on the 20% train slice.",
        "- `nli-base`: premise = the state JSON (DeBERTa truncates the premise at 512 tokens); hypotheses "
        "\"<question> Answer: <option>\" for choice/score (option rendered as Laya renders it), \"It is false/true "
        "that <statement>\" for noul; entailment logits softmaxed across the decision's options. Written once, not tuned.",
        "- `embed-lr` is a simple supervised cross-feature baseline: bge-small embeds each \"question + option + state\" "
        "text; one binary logistic regression (\"is this the gold option?\") is trained over every option of all 6,000 "
        "train decisions; per test decision the option logits are softmaxed and the argmax wins. Its calib predictions "
        "come from a twin model trained without the calib cases.",
        "- `embed-lr-slot` (added because the premise for the pair design did not hold): every (workflow, question) "
        "slot has a fixed option set in this dataset — all 1,600 cases use identical question definitions — so the "
        "ordinary classifier recipe works: embed the state once with bge-small, one multinomial logistic regression per "
        "slot (20 heads). This is the like-for-like counterpart of the dataset card's MiniLM-L6 specialist (0.587) and of "
        "`laya-typed-decisions`, which is also trained on these 20 slots.",
        "- `jev`: TypeSafe API, `model: jev-latest` (reported version in `typed_decisions.json`), one call per case "
        "with all five questions, state as the parsed JSON object.",
        f"- Context: base `laya` (512 tokens) cuts the state on {res['laya_state_truncated']['laya']['state_truncated']} of "
        f"{res['n_test_decisions']:,} test decisions; `laya-typed-decisions` (1,024) on "
        f"{res['laya_state_truncated']['laya-td']['state_truncated']}. Counted, not changed.",
        "- Soft accuracy = Σ p·gold distribution; score MAE = |expected level − gold `score`|, as in Laya's eval.",
    ]
    if "jev" in M:
        j = M["jev"]
        lines.append(
            f"- Jev reported itself as {j.get('version', '?')} on every call (the published 0.727 is the same version, "
            "measured by the dataset card's author on 2026-09-18 and copied into Laya's table). Ours: "
            f"{_f(j['accuracy'], 4)} ({_f(j['accuracy_noul_ge_0.5'], 4)} with the notebook's noul ≥ 0.5 rule; "
            f"{j['noul_exact_ties']} exact 0.5 noul answers). The +0.009 accuracy gap is within what an API model "
            "re-run on another day, possibly with the state sent as a string rather than an object, can move. "
            f"The ECE gap is not: we get {_f(j['ece'])} against the published 0.144, with mean top probability "
            f"{_f(j['mean_confidence'])} vs accuracy {_f(j['accuracy'])} (over-confidence "
            f"{j['mean_confidence'] - j['accuracy']:+.3f}; the card itself states +0.023, so the confidence level agrees). "
            "Neither the card nor Laya's table defines that ECE, so we cannot tell whether the gap is a different "
            "definition or a different run. We report 15-bin top-label ECE for every model: the definition in Laya's own "
            "script, under which we reproduce Laya's published 0.213 and 0.175.")
    if "laya-td" in M:
        t = M["laya-td"]
        lines.append(
            f"- `laya-typed-decisions` is *under*-confident here (mean top probability {_f(t['mean_confidence'])} vs "
            f"accuracy {_f(t['accuracy'])}); the fitted temperature is {t['temperature']:.2f} (< 1 sharpens). Base `laya` "
            f"is over-confident ({_f(M['laya']['mean_confidence'])} vs {_f(M['laya']['accuracy'])}; temperature "
            f"{M['laya']['temperature']:.2f}).")
    return "\n".join(lines)


def overflow() -> dict:
    """How many test decisions have their state cut by Laya's context (tokenizer only, no model).
    Base laya: 512 tokens total, 192 for question + options; laya-typed-decisions: 1024 / 256."""
    from huggingface_hub import snapshot_download
    from laya_mlx.agent import Agent
    from laya_mlx.common import build_prefix, serialize_state
    from laya_mlx.tokenizer import Tokenizer

    data, out = load(), {}
    for method, repo, max_len, head in (("laya", "aac6fef/laya-mlx", 512, 192),
                                        ("laya-td", "aac6fef/laya-typed-decisions-mlx", 1024, 256)):
        tok = Tokenizer(Path(snapshot_download(repo, allow_patterns=["tokenizer/*"])) / "tokenizer")
        cut = 0
        for d in data["test"]:
            prefix, _ = build_prefix(tok, Agent._to_internal(d.qdef), head)
            room = max(0, max_len - len(prefix) - 1)
            n = len(tok(serialize_state(json.loads(d.state)).replace(tok.mask_token, " "),
                        add_special_tokens=False)["input_ids"])
            cut += n > room
        out[method] = {"max_len": max_len, "state_truncated": int(cut), "n": len(data["test"])}
    return out
