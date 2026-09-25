"""Latency benchmark: the same 200 test examples for every method, after a warm-up, model load
excluded, on one machine.

Single-example mode is the fair comparison (one input per forward pass; for Laya and NLI all its
options in that pass). Batched mode packs several inputs into one pass and reports throughput.
On AG News the per-pair HF pipeline is measured too, as the old reference. Banking77 (77 options)
shows how cost scales with option count: NLI runs one pair per option, Laya reads all options in
one sequence, embed-lr and SetFit read only the text.

Framework: laya runs on MLX (laya-mlx); laya-torch is upstream Laya on PyTorch MPS, the same
runtime as nli, nli-base, embed-zs(-base), embed-lr and SetFit.

Few-shot methods are timed on inference only (encode + logistic-regression head). Their models
are trained here on the k=8, seed=0 draw purely for timing and never written to the cache.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from s1x.tasks import load_task

RESULTS = Path(__file__).resolve().parents[2] / "results"
N = 200
WARMUP = 16
LAYA_BATCH, NLI_BATCH, ENCODER_BATCH = 16, 8, 64


def _stats(ms: np.ndarray, wall_s: float, n: int) -> dict:
    return {"n": n, "p50_ms": float(np.median(ms)), "p95_ms": float(np.percentile(ms, 95)),
            "mean_ms": float(ms.mean()), "wall_s": wall_s, "examples_per_s": n / wall_s}


def _bench(runner, data, texts, warm) -> tuple[dict, np.ndarray]:
    runner.predict(data.task, data.labels, warm)
    start = time.perf_counter()
    p, _, ms = runner.predict(data.task, data.labels, texts)
    wall = time.perf_counter() - start
    return _stats(ms, wall, len(texts)), p


def _bench_encoder(encode, head, texts, warm) -> dict:
    """Single-example: encode + head one text at a time. Batched: encode in batches of 64."""
    for t in warm:
        head(encode([t]))
    ms = []
    start = time.perf_counter()
    for t in texts:
        s = time.perf_counter()
        head(encode([t]))
        ms.append((time.perf_counter() - s) * 1000)
    single = _stats(np.array(ms), time.perf_counter() - start, len(texts))
    head(encode(warm))
    start = time.perf_counter()
    for s in range(0, len(texts), ENCODER_BATCH):
        head(encode(texts[s:s + ENCODER_BATCH]))
    wall = time.perf_counter() - start
    per = wall / len(texts) * 1000
    batched = {"n": len(texts), "batch": ENCODER_BATCH, "wall_s": wall, "examples_per_s": len(texts) / wall,
               "p50_ms": per, "p95_ms": per}
    return {"single": single, "batched": batched}


METHODS = ("laya", "laya-torch", "nli", "nli-base", "embed-zs", "embed-zs-base", "nli-pipeline", "embed-lr", "setfit")


def bench_one(task: str, method: str, pred_dir: Path) -> dict:
    """Time ONE method on ONE task. Meant to run in its own process (see `run`), so no model's
    device memory — MPS or MLX — is still allocated when the next one is measured. Single-mode
    probabilities are saved to pred_dir for the cross-method agreement checks."""
    data = load_task(task)
    texts = data.test.texts[:N]
    warm = data.calib.texts[:WARMUP]  # warm-up on calib, not on the measured examples
    out: dict = {}

    def zero_shot(runner, batch_attr, batch):
        out[f"{method}_single"], p1 = _bench(runner, data, texts, warm)
        setattr(runner, batch_attr, batch)
        out[f"{method}_batched"], p2 = _bench(runner, data, texts, warm)
        out[f"{method}_batched"] |= {"batch": batch, "max_abs_diff_vs_single": float(np.abs(p1 - p2).max()),
                                     "argmax_agree_vs_single": float((p1.argmax(1) == p2.argmax(1)).mean())}
        np.save(pred_dir / f"{task}__{method}.npy", p1)

    if method == "laya":
        from s1x.runners.laya import LayaRunner
        zero_shot(LayaRunner("laya"), "batch_states", LAYA_BATCH)
    elif method == "laya-torch":
        from s1x.runners.laya_torch import LayaTorchRunner
        zero_shot(LayaTorchRunner(), "batch_states", LAYA_BATCH)
    elif method in ("nli", "nli-base"):
        from s1x.runners.nli import MODELS, NLIRunner
        zero_shot(NLIRunner(model=MODELS[method]), "batch_examples", NLI_BATCH)
    elif method in ("embed-zs", "embed-zs-base"):
        from s1x.runners.embed_zs import EmbedZSRunner
        zero_shot(EmbedZSRunner(method), "batch_examples", ENCODER_BATCH)
    elif method == "nli-pipeline":
        from s1x.runners.nli import NLIPipelineRunner
        out["nli_pipeline_per_pair"], q = _bench(NLIPipelineRunner(), data, texts, warm)
        np.save(pred_dir / f"{task}__nli-pipeline.npy", q)
    elif method == "embed-lr":
        from s1x.runners.fewshot import EmbedLR
        emb = EmbedLR()
        clf = _fit_embed_lr(emb, data)
        r = _bench_encoder(emb.encode, clf.predict_proba, texts, warm)
        out["embed-lr_single"], out["embed-lr_batched"] = r["single"], r["batched"]
    elif method == "setfit":
        model = _fit_setfit(data)
        body, head = model.model_body, model.model_head

        def encode(ts):
            return body.encode(ts, batch_size=ENCODER_BATCH, normalize_embeddings=model.normalize_embeddings,
                               convert_to_numpy=True, show_progress_bar=False)

        r = _bench_encoder(encode, head.predict_proba, texts, warm)
        out["setfit_single"], out["setfit_batched"] = r["single"], r["batched"]
    else:
        raise SystemExit(f"unknown method {method}")
    return out


def run_task(task: str, with_pipeline: bool, pred_dir: Path) -> dict:
    """Every method on one task, each in a fresh child process.

    Running all nine models in one process leaked device memory (MPS/MLX allocations are not freed
    by `del` + gc.collect()): on a 24 GB M4 Pro the process reached a 26 GB footprint and ~11 GB of
    swap, so later methods were timed under swap. A fresh process per method returns everything to
    the OS between measurements and starts each timing from the same clean state."""
    out: dict = {"task": task, "n": N, "warmup": WARMUP, "options": len(load_task(task).labels),
                 "isolation": "one process per method"}
    for method in METHODS:
        if method == "nli-pipeline" and not with_pipeline:
            continue
        result_file = pred_dir / f"{task}__{method}.json"
        subprocess.run([sys.executable, "-m", "s1x.latency", "--one", task, method, "--pred-dir", str(pred_dir),
                        "--result", str(result_file)], check=True)
        out |= json.loads(result_file.read_text())
        print(f"  {task} {method}: done", flush=True)
    # Cross-method agreement, from the saved single-mode predictions.
    def load(m):
        f = pred_dir / f"{task}__{m}.npy"
        return np.load(f) if f.exists() else None
    laya, laya_t = load("laya"), load("laya-torch")
    if laya is not None and laya_t is not None:
        out["laya-torch_single"]["argmax_agree_vs_laya_mlx"] = float((laya.argmax(1) == laya_t.argmax(1)).mean())
    nli, pipe = load("nli"), load("nli-pipeline")
    if nli is not None and pipe is not None:
        out["nli_single"] |= {"max_abs_diff_vs_pipeline": float(np.abs(nli - pipe).max()),
                              "argmax_agree_vs_pipeline": float((nli.argmax(1) == pipe.argmax(1)).mean())}
    return out


def _fit_embed_lr(emb, data):
    from sklearn.linear_model import LogisticRegression

    ids = data.shot_ids["k=8,seed=0"]
    x = emb.encode([data.pool.texts[i] for i in ids])
    return LogisticRegression(max_iter=5000).fit(x, data.pool.labels[ids])


def _fit_setfit(data):
    """Throwaway SetFit model for timing only: 10 contrastive steps; inference cost does not
    depend on how long the body was trained."""
    from datasets import Dataset
    from setfit import SetFitModel, Trainer, TrainingArguments

    from s1x.runners.fewshot import _device
    from s1x.runners.setfit import SETFIT_ARGS, SETFIT_MODEL

    ids = data.shot_ids["k=8,seed=0"]
    train = Dataset.from_dict({"text": [data.pool.texts[i] for i in ids],
                               "label": [int(v) for v in data.pool.labels[ids]]})
    model = SetFitModel.from_pretrained(SETFIT_MODEL, device=_device())
    args = TrainingArguments(**(SETFIT_ARGS | {"max_steps": 10}), seed=0, report_to="none", save_strategy="no",
                             show_progress_bar=False)
    Trainer(model=model, args=args, train_dataset=train).train()
    return model


def run(tasks=("ag_news", "banking77")) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        return {"machine": f"{platform.machine()} {platform.platform()}",
                "note": "per-example latency in batched mode = batch wall time / batch size; few-shot = inference only",
                **{task: run_task(task, with_pipeline=(task == "ag_news"), pred_dir=Path(tmp)) for task in tasks}}


def write(result: dict, path: Path | None = None) -> Path:
    path = path or RESULTS / "latency.json"
    existing = json.loads(path.read_text()) if path.exists() else {}
    if "benchmark" in existing:  # first run (AG News, Laya + NLI only), kept for the record
        existing["benchmark_v1_ag_news"] = existing.pop("benchmark")
    existing |= {"benchmark_v2": result}
    path.write_text(json.dumps(existing, indent=1))
    return path


if __name__ == "__main__":
    # `python -m s1x.latency` runs the whole benchmark; `--one task method` is the per-process worker.
    ap = argparse.ArgumentParser()
    ap.add_argument("--one", nargs=2, metavar=("TASK", "METHOD"))
    ap.add_argument("--pred-dir")
    ap.add_argument("--result")
    ap.add_argument("--out", help="where to write the full result (default results/latency.json)")
    a = ap.parse_args()
    if a.one:
        Path(a.result).write_text(json.dumps(bench_one(a.one[0], a.one[1], Path(a.pred_dir))))
    else:
        print(f"wrote {write(run(), Path(a.out) if a.out else None)}")
