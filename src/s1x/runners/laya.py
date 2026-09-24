"""Laya via laya-mlx, zero-shot, exactly as advertised: state + typed question -> probabilities."""
from __future__ import annotations

import time
import warnings

import numpy as np

from s1x.tasks import Task

CHECKPOINTS = {"laya": "aac6fef/laya-mlx", "laya-ml": "aac6fef/laya-multilingual-mlx"}


class LayaRunner:
    def __init__(self, method: str = "laya"):
        import laya_mlx

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.agent = laya_mlx.load(CHECKPOINTS[method])

    def question(self, task: Task, labels: tuple[str, ...]) -> dict:
        if task.kind == "noul":
            return {"type": "noul", "instructions": task.instruction}
        return {"type": task.kind, "instructions": task.instruction, "criteria": list(labels)}

    def predict(self, task: Task, labels: tuple[str, ...], texts: list[str]):
        """Returns probabilities (n, C), Laya's own `confidence` field (n,), and latency ms (n,)."""
        q = {"q": self.question(task, labels)}
        probs, conf, ms = [], [], []
        for text in texts:
            start = time.perf_counter()
            answer = self.agent.predict(text, q)["answers"]["q"]
            ms.append((time.perf_counter() - start) * 1000)
            if task.kind == "noul":
                p_true = float(answer["noul"])
                row = [1.0 - p_true, p_true]
            elif task.kind == "score":
                row = [float(answer["probabilities"][str(i)]) for i in range(len(labels))]
            else:
                row = [float(answer["probabilities"][label]) for label in labels]
            row = np.clip(np.array(row), 1e-6, None)  # upstream rounds to 4 d.p.; zeros break log-loss
            probs.append(row / row.sum())
            conf.append(float(answer.get("confidence", np.nan)))
        return np.array(probs), np.array(conf), np.array(ms)
