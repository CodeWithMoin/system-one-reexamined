"""Bi-encoder zero-shot: embed the text once and each label once, score = cosine similarity.

Labels become short sentences with the same per-task template as NLI ("This news article is
about Business."). Probabilities = softmax(cos / TAU) with TAU = 0.05 fixed before seeing any
result; temperature scaling on the calib split then fixes calibration, as for every method.
bge conventions: the text is the query (with bge's retrieval instruction), label sentences are
passages. Label embeddings are computed once per task and cached, so a decision costs one text
encode plus a dot product; latency is measured that way, one text at a time.
"""
from __future__ import annotations

import time

import numpy as np

from s1x.runners.nli import TEMPLATES
from s1x.tasks import Task

MODELS = {"embed-zs": "BAAI/bge-small-en-v1.5", "embed-zs-base": "BAAI/bge-base-en-v1.5"}
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
TAU = 0.05


class EmbedZSRunner:
    def __init__(self, method: str = "embed-zs", batch_examples: int = 1):
        import torch
        from sentence_transformers import SentenceTransformer

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = SentenceTransformer(MODELS[method], device=device)
        self.batch_examples = batch_examples
        self._labels: dict = {}

    def _encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, batch_size=64, normalize_embeddings=True, convert_to_numpy=True,
                                 show_progress_bar=False)

    def label_matrix(self, task: Task, labels: tuple[str, ...]) -> np.ndarray:
        key = (task.name, labels)
        if key not in self._labels:
            self._labels[key] = self._encode([TEMPLATES[task.name].format(label) for label in labels])
        return self._labels[key]

    def predict(self, task: Task, labels: tuple[str, ...], texts: list[str]):
        lab = self.label_matrix(task, labels)
        per = max(1, self.batch_examples)
        probs, ms = [], []
        for s in range(0, len(texts), per):
            chunk = texts[s:s + per]
            start = time.perf_counter()
            z = self._encode([QUERY_INSTRUCTION + t for t in chunk]) @ lab.T / TAU
            z = z - z.max(1, keepdims=True)
            p = np.exp(z) / np.exp(z).sum(1, keepdims=True)
            elapsed = (time.perf_counter() - start) * 1000
            probs.extend(p)
            ms.extend([elapsed / len(chunk)] * len(chunk))
        return np.array(probs), np.full(len(texts), np.nan), np.array(ms)
