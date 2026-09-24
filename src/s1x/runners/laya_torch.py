"""Upstream Laya on PyTorch (pip `laya`, checkpoint convaiinnovations/laya), for same-framework
latency: nli, embed-lr and SetFit run on PyTorch MPS, so this puts Laya on the same runtime.

Used as shipped: on MPS upstream runs fp32 for a single row and enables fp16 autocast only once
a pass has >= 5 rows (LAYA_MPS_AMP_MIN_ROWS). Single-example mode = `agent.predict(state, q)`;
batched mode = upstream's own `predict_batch(..., sort_by_length=True)`.
"""
from __future__ import annotations

import time
import warnings

import numpy as np

from s1x.runners.laya import LayaRunner
from s1x.tasks import Task

CHECKPOINT = "convaiinnovations/laya"


class LayaTorchRunner:
    def __init__(self, batch_states: int = 1):
        import torch
        from laya import Agent

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.agent = Agent(CHECKPOINT, device=device)
        self.batch_states = batch_states

    def predict(self, task: Task, labels: tuple[str, ...], texts: list[str]):
        q = {"q": LayaRunner.question(None, task, labels)}
        probs, conf, ms = [], [], []
        if self.batch_states > 1:
            for s in range(0, len(texts), 256):
                chunk = texts[s:s + 256]
                start = time.perf_counter()
                outs = self.agent.predict_batch(chunk, q, batch_size=self.batch_states, sort_by_length=True)
                per = (time.perf_counter() - start) * 1000 / len(chunk)
                for out in outs:
                    answer = out["answers"]["q"]
                    probs.append(LayaRunner._row(task, labels, answer))
                    conf.append(float(answer.get("confidence", np.nan)))
                    ms.append(per)
            return np.array(probs), np.array(conf), np.array(ms)
        for text in texts:
            start = time.perf_counter()
            answer = self.agent.predict(text, q)["answers"]["q"]
            ms.append((time.perf_counter() - start) * 1000)
            probs.append(LayaRunner._row(task, labels, answer))
            conf.append(float(answer.get("confidence", np.nan)))
        return np.array(probs), np.array(conf), np.array(ms)
