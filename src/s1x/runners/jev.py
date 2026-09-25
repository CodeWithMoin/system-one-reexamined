"""Jev (TypeSafe AI) through its HTTP API, zero-shot, with the same questions as Laya.

Endpoint and shapes per https://docs.typesafe.ai/api.md. The key is read from .env.local
(gitignored) and never logged. Latency is the API round trip and includes the network, so it
is recorded but never compared with the local, same-machine timings.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from s1x.tasks import Task

URL = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-latest"
WORKERS = 4
ROOT = Path(__file__).resolve().parents[3]


def _key() -> str:
    if os.environ.get("TYPESAFE_API_KEY"):
        return os.environ["TYPESAFE_API_KEY"]
    match = re.search(r"^TYPESAFE_API_KEY=(\S+)", (ROOT / ".env.local").read_text(), re.M)
    if not match:
        raise SystemExit("TYPESAFE_API_KEY not found in .env.local")
    return match.group(1)


class JevRunner:
    def __init__(self):
        self.key = _key()
        self.versions: set[str] = set()

    def question(self, task: Task, labels: tuple[str, ...]) -> dict:
        # Identical instruction and label wording to the Laya runner.
        if task.kind == "noul":
            return {"type": "noul", "instructions": task.instruction}
        if task.kind == "score":
            return {"type": "score", "instructions": task.instruction, "criteria": list(labels)}
        return {"type": "choice", "instructions": task.instruction, "criteria": {label: None for label in labels}}

    def _call(self, text: str, q: dict) -> tuple[dict, float]:
        body = json.dumps({"state": text, "model": MODEL, "questions": {"q": q}}).encode()
        delay = 1.0
        for attempt in range(8):
            req = urllib.request.Request(URL, data=body, headers={"Authorization": f"Bearer {self.key}",
                                                                  "Content-Type": "application/json"})
            start = time.perf_counter()
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    out = json.load(r)
                return out, (time.perf_counter() - start) * 1000
            except urllib.error.HTTPError as e:
                if e.code in (429, 529, 500, 502, 503, 504) and attempt < 7:
                    time.sleep(delay); delay = min(delay * 2, 30); continue
                raise RuntimeError(f"Jev HTTP {e.code}: {e.read().decode()[:200]}") from None
            except (urllib.error.URLError, TimeoutError):
                if attempt < 7:
                    time.sleep(delay); delay = min(delay * 2, 30); continue
                raise
        raise RuntimeError("unreachable")

    def predict(self, task: Task, labels: tuple[str, ...], texts: list[str]):
        q = self.question(task, labels)
        with ThreadPoolExecutor(WORKERS) as pool:
            results = list(pool.map(lambda t: self._call(t, q), texts))
        probs, conf, ms = [], [], []
        for out, t in results:
            self.versions.add(out.get("model", "?"))
            a = out["answers"]["q"]
            if task.kind == "noul":
                row = [1.0 - float(a["noul"]), float(a["noul"])]
            elif task.kind == "score":
                row = [float(a["probabilities"][str(i)]) for i in range(len(labels))]
            else:
                row = [float(a["probabilities"][label]) for label in labels]
            row = np.clip(np.array(row), 1e-6, None)  # exact 0.0/1.0 come back; zeros break log-loss
            probs.append(row / row.sum())
            conf.append(float(a.get("confidence", np.nan)))
            ms.append(t)
        return np.array(probs), np.array(conf), np.array(ms)
