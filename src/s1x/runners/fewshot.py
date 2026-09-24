"""Few-label contenders, trained on the committed k-shot draws (k per class, 5 seeds).

- `embed-lr`: frozen sentence embeddings (BAAI/bge-small-en-v1.5) + sklearn LogisticRegression.
- `setfit`: SetFit, in runners/setfit.py.

Nothing is tuned: hyper-parameters are library defaults or fixed below, written once. Test and
calib are only ever predicted. Latency is per example, single-example mode (one text, encode +
head), measured on test after a warm-up.
"""
from __future__ import annotations

import time

import numpy as np

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
LATENCY_N = 200  # single-example timing sample per (method, task); the rest is encoded in batches


def _device() -> str:
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


def _single_example_ms(encode, head, texts: list[str]) -> np.ndarray:
    """Per-example wall time for encode + head on one text at a time, after a warm-up."""
    for t in texts[:8]:
        head(encode([t]))
    ms = []
    for t in texts[:LATENCY_N]:
        start = time.perf_counter()
        head(encode([t]))
        ms.append((time.perf_counter() - start) * 1000)
    return np.array(ms)


def _full_proba(clf, x: np.ndarray, n_classes: int) -> np.ndarray:
    """predict_proba in label order, with zero columns for classes absent from training."""
    p = np.zeros((len(x), n_classes))
    p[:, clf.classes_] = clf.predict_proba(x)
    p = np.clip(p, 1e-6, None)
    return p / p.sum(1, keepdims=True)


class EmbedLR:
    """Embeddings are computed once per task (they do not depend on k or seed)."""

    def __init__(self):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(EMBED_MODEL, device=_device())
        self._cache: dict = {}

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, batch_size=64, normalize_embeddings=True, convert_to_numpy=True,
                                 show_progress_bar=False)

    def prepare(self, data) -> None:
        name = data.task.name
        if name in self._cache:
            return
        needed = sorted({i for ids in data.shot_ids.values() for i in ids})
        pool = dict(zip(needed, self.encode([data.pool.texts[i] for i in needed])))
        self._cache[name] = {"test": self.encode(data.test.texts), "calib": self.encode(data.calib.texts),
                             "pool": pool, "ms": None}

    def fit_predict(self, data, shot: str) -> dict:
        from sklearn.linear_model import LogisticRegression

        self.prepare(data)
        c = self._cache[data.task.name]
        ids = data.shot_ids[shot]
        x = np.stack([c["pool"][i] for i in ids])
        y = data.pool.labels[ids]
        clf = LogisticRegression(max_iter=5000).fit(x, y)
        n_classes = len(data.labels)
        if c["ms"] is None:  # one timing per task: encoder cost dominates, head cost is ~constant
            c["ms"] = _single_example_ms(self.encode, lambda e: clf.predict_proba(e), data.test.texts)
        return {"test": _full_proba(clf, c["test"], n_classes), "calib": _full_proba(clf, c["calib"], n_classes),
                "ms": c["ms"]}
