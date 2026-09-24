"""Laya via laya-mlx, zero-shot, exactly as advertised: state + typed question -> probabilities.

Single-example mode calls `agent.predict(state, questions)` — one forward pass per input, all
options scored at their [MASK] tokens. `Agent.predict` batches several *questions* about one
state, not several states, so the throughput mode (`batch_states > 1`) uses the agent's own
building blocks (`prepare` -> `collate_items` -> `forward`) to put several states in one pass,
then applies the same temperature and softmax `system_one` does.
"""
from __future__ import annotations

import time
import warnings

import numpy as np

from s1x.tasks import Task

CHECKPOINTS = {"laya": "aac6fef/laya-mlx", "laya-ml": "aac6fef/laya-multilingual-mlx"}


class LayaRunner:
    def __init__(self, method: str = "laya", batch_states: int = 1):
        import laya_mlx

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.agent = laya_mlx.load(CHECKPOINTS[method])
        self.batch_states = batch_states

    def question(self, task: Task, labels: tuple[str, ...]) -> dict:
        if task.kind == "noul":
            return {"type": "noul", "instructions": task.instruction}
        return {"type": task.kind, "instructions": task.instruction, "criteria": list(labels)}

    @staticmethod
    def _row(task: Task, labels, answer) -> np.ndarray:
        if task.kind == "noul":
            p_true = float(answer["noul"])
            row = [1.0 - p_true, p_true]
        elif task.kind == "score":
            row = [float(answer["probabilities"][str(i)]) for i in range(len(labels))]
        else:
            row = [float(answer["probabilities"][label]) for label in labels]
        row = np.clip(np.array(row), 1e-6, None)  # upstream rounds to 4 d.p.; zeros break log-loss
        return row / row.sum()

    def predict(self, task: Task, labels: tuple[str, ...], texts: list[str]):
        """Returns probabilities (n, C), Laya's own `confidence` field (n,), and latency ms (n,)."""
        if self.batch_states > 1:
            return self._predict_batched(task, labels, texts)
        q = {"q": self.question(task, labels)}
        probs, conf, ms = [], [], []
        for text in texts:
            start = time.perf_counter()
            answer = self.agent.predict(text, q)["answers"]["q"]
            ms.append((time.perf_counter() - start) * 1000)
            probs.append(self._row(task, labels, answer))
            conf.append(float(answer.get("confidence", np.nan)))
        return np.array(probs), np.array(conf), np.array(ms)

    def _predict_batched(self, task: Task, labels: tuple[str, ...], texts: list[str]):
        from laya_mlx.agent import collate_items
        from laya_mlx.common import confidence_from_probs, temp_bucket

        agent = self.agent
        q = {"q": self.question(task, labels)}
        max_len = agent.cfg.get("max_len", 512)
        probs, conf, ms = [], [], []
        for s in range(0, len(texts), self.batch_states):
            chunk = texts[s:s + self.batch_states]
            start = time.perf_counter()
            items = [agent.prepare(text, q)[0][0] for text in chunk]
            batch = collate_items(items, agent.tok.pad_token_id, pad_to_multiple=agent.pad_to_multiple,
                                  max_length=max_len)
            logits, _act = agent.forward(batch)
            logits = np.asarray(logits)
            rows, confs = [], []
            for r, item in enumerate(items):
                k, qt = len(item["markers"]), item["qtype"]
                scale = agent.temperature_by_options.get(temp_bucket(qt, k), agent.temperature[qt])
                z = logits[r, :k] / scale
                p = np.exp(z - z.max())
                p /= p.sum()
                # Same 4 d.p. rounding the public API applies, so both modes give the same numbers.
                if task.kind == "noul":
                    answer = {"noul": round(float(p[1]), 4)}
                    c = max(float(p[1]), 1.0 - float(p[1]))
                elif task.kind == "score":
                    answer = {"probabilities": {str(i): round(float(v), 4) for i, v in enumerate(p)}}
                    c = confidence_from_probs(p, k)
                else:
                    answer = {"probabilities": {lab: round(float(v), 4) for lab, v in zip(labels, p)}}
                    c = confidence_from_probs(p, k)
                rows.append(self._row(task, labels, answer))
                confs.append(round(c, 4))
            elapsed = (time.perf_counter() - start) * 1000
            probs.extend(rows)
            conf.extend(confs)
            ms.extend([elapsed / len(chunk)] * len(chunk))
        return np.array(probs), np.array(conf), np.array(ms)

    def overflow(self, task: Task, labels: tuple[str, ...], texts: list[str]) -> dict:
        """How many inputs do not fit Laya's context. `over_512`: the text alone is longer than
        512 tokens. `state_truncated`: the text is longer than the room left after the question
        and options (laya-mlx then keeps the first `room` tokens of the state)."""
        from laya_mlx.common import build_prefix

        agent = self.agent
        qi = agent._to_internal(self.question(task, labels))
        max_len = agent.cfg.get("max_len", 512)
        prefix, _ = build_prefix(agent.tok, qi, agent.cfg.get("head_max_len", 192))
        room = max(0, max_len - len(prefix) - 1)
        lengths = np.array([len(agent.tok(t.replace(agent.tok.mask_token, " "))["input_ids"]) for t in texts])
        return {"n": int(len(texts)), "max_len": int(max_len), "prefix_tokens": int(len(prefix)),
                "state_room": int(room), "over_512": int((lengths > max_len).sum()),
                "state_truncated": int((lengths > room).sum()), "max_state_tokens": int(lengths.max()),
                "p99_state_tokens": float(np.percentile(lengths, 99))}
