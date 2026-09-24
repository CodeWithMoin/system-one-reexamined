"""Zero-shot NLI classification — the 2019-20 way of doing the same thing.

Each option becomes a hypothesis; entailment scores are softmaxed across options
(single-label), which is the standard zero-shot-classification recipe.
"""
from __future__ import annotations

import time

import numpy as np

from s1x.tasks import Task

MODEL = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"

# One template per task, written once and never tuned on test.
TEMPLATES = {
    "ag_news": "This news article is about {}.",
    "emotion": "This message expresses {}.",
    "banking77": "The customer is asking about {}.",
    "sst5": "The sentiment of this review is {}.",
    "sms_spam": "This text message is {}.",
}


class NLIRunner:
    def __init__(self):
        import torch
        from transformers import pipeline

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.pipe = pipeline("zero-shot-classification", model=MODEL, device=device)

    def predict(self, task: Task, labels: tuple[str, ...], texts: list[str]):
        template = TEMPLATES[task.name]
        probs, ms = [], []
        for text in texts:
            start = time.perf_counter()
            out = self.pipe(text, candidate_labels=list(labels), hypothesis_template=template, multi_label=False)
            ms.append((time.perf_counter() - start) * 1000)
            score = dict(zip(out["labels"], out["scores"]))
            probs.append([score[label] for label in labels])
        return np.array(probs), np.full(len(texts), np.nan), np.array(ms)
