"""Latency benchmark: same 200 AG News test examples, after a warm-up, model load excluded.

Single-example mode is the fair comparison (one input, all its options in one forward pass,
for both Laya and NLI). Batched mode packs several inputs into one pass and reports
throughput. The per-pair HF pipeline is measured too, as the old reference.
"""
from __future__ import annotations

import gc
import json
import platform
import time
from pathlib import Path

import numpy as np

from s1x.tasks import load_task

RESULTS = Path(__file__).resolve().parents[2] / "results"
N = 200
WARMUP = 16


def _stats(ms: np.ndarray, wall_s: float, n: int) -> dict:
    return {"n": n, "p50_ms": float(np.median(ms)), "p95_ms": float(np.percentile(ms, 95)),
            "mean_ms": float(ms.mean()), "wall_s": wall_s, "examples_per_s": n / wall_s}


def _bench(runner, data, texts, warm) -> tuple[dict, np.ndarray]:
    runner.predict(data.task, data.labels, warm)
    start = time.perf_counter()
    p, _, ms = runner.predict(data.task, data.labels, texts)
    wall = time.perf_counter() - start
    return _stats(ms, wall, len(texts)), p


def run(task: str = "ag_news", laya_batch: int = 16, nli_batch: int = 8) -> dict:
    data = load_task(task)
    texts = data.test.texts[:N]
    warm = data.calib.texts[:WARMUP]  # warm-up on calib, not on the measured examples
    out: dict = {"task": task, "n": N, "warmup": WARMUP, "machine": f"{platform.machine()} {platform.platform()}",
                 "note": "per-example latency in batched mode = batch wall time / batch size"}

    from s1x.runners.laya import LayaRunner
    laya = LayaRunner("laya")
    out["laya_single"], p_single = _bench(laya, data, texts, warm)
    laya.batch_states = laya_batch
    out["laya_batched"], p_batched = _bench(laya, data, texts, warm)
    out["laya_batched"]["batch_states"] = laya_batch
    out["laya_batched"]["max_abs_diff_vs_single"] = float(np.abs(p_single - p_batched).max())
    out["laya_batched"]["argmax_agree_vs_single"] = float((p_single.argmax(1) == p_batched.argmax(1)).mean())
    del laya
    gc.collect()

    from s1x.runners.nli import NLIPipelineRunner, NLIRunner
    nli = NLIRunner()
    out["nli_single"], q_single = _bench(nli, data, texts, warm)
    nli.batch_examples = nli_batch
    out["nli_batched"], q_batched = _bench(nli, data, texts, warm)
    out["nli_batched"]["batch_examples"] = nli_batch
    out["nli_batched"]["max_abs_diff_vs_single"] = float(np.abs(q_single - q_batched).max())
    del nli
    gc.collect()

    pipe = NLIPipelineRunner()
    out["nli_pipeline_per_pair"], q_pipe = _bench(pipe, data, texts, warm)
    out["nli_single"]["max_abs_diff_vs_pipeline"] = float(np.abs(q_single - q_pipe).max())
    out["nli_single"]["argmax_agree_vs_pipeline"] = float((q_single.argmax(1) == q_pipe.argmax(1)).mean())
    return out


def write(result: dict) -> Path:
    path = RESULTS / "latency.json"
    existing = json.loads(path.read_text()) if path.exists() else {}
    existing |= {"benchmark": result}
    path.write_text(json.dumps(existing, indent=1))
    return path
