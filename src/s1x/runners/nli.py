"""Zero-shot NLI classification — the 2019-20 way of doing the same thing.

Each option becomes a hypothesis; entailment scores are softmaxed across options
(single-label), which is the standard zero-shot-classification recipe.

The HF zero-shot pipeline runs one forward pass per (text, label) pair. Laya scores every
option of one input in a single pass, so for a like-for-like latency comparison this runner
builds all (text, hypothesis) pairs for one example itself and scores them as one padded
batch in one forward pass. The maths is the pipeline's single-label mode exactly: premise =
text, hypothesis = template.format(label), truncation "only_first", entailment logit per
pair, softmax across options (tests/test_nli_batched.py checks it against the pipeline).

`batch_examples > 1` additionally packs several examples into one forward pass
(throughput mode); latency per example is then the batch wall time divided by its size.
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
    def __init__(self, batch_examples: int = 1, max_pairs: int = 96, dtype: str = "auto", device: str | None = None):
        """batch_examples: examples per forward pass (1 = single-example mode, the fair
        latency comparison). max_pairs caps pairs per pass so 77-label tasks fit in memory;
        a pass never holds more than that many pairs."""
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(MODEL)
        # dtype="auto" is the load the pipeline does (checkpoint dtype, fp16). In fp16 the padded
        # batch differs from per-pair passes by fp16 kernel noise (up to ~2e-3 in probability);
        # in fp32 the two agree to ~1e-6.
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL, dtype=dtype).to(self.device).eval()
        self.entail = next(i for label, i in self.model.config.label2id.items() if label.lower().startswith("entail"))
        self.batch_examples = batch_examples
        self.max_pairs = max_pairs

    def _entail_logits(self, premises: list[str], hypotheses: list[str]) -> np.ndarray:
        enc = self.tok(premises, hypotheses, padding=True, truncation="only_first", return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with self.torch.inference_mode():
            logits = self.model(**enc).logits
        return logits[:, self.entail].float().cpu().numpy()

    def _score(self, texts: list[str], hyps: list[str]) -> np.ndarray:
        """(len(texts), len(hyps)) entailment logits; pairs from all texts share passes."""
        premises = [t for t in texts for _ in hyps]
        hypotheses = hyps * len(texts)
        out = np.concatenate([
            self._entail_logits(premises[s:s + self.max_pairs], hypotheses[s:s + self.max_pairs])
            for s in range(0, len(premises), self.max_pairs)
        ])
        return out.reshape(len(texts), len(hyps))

    def predict(self, task: Task, labels: tuple[str, ...], texts: list[str]):
        template = TEMPLATES[task.name]
        hyps = [template.format(label) for label in labels]
        per_pass = max(1, self.batch_examples)
        probs, ms = [], []
        for s in range(0, len(texts), per_pass):
            chunk = texts[s:s + per_pass]
            start = time.perf_counter()
            z = self._score(chunk, hyps)
            elapsed = (time.perf_counter() - start) * 1000
            z = z - z.max(1, keepdims=True)
            p = np.exp(z) / np.exp(z).sum(1, keepdims=True)
            probs.extend(p)
            ms.extend([elapsed / len(chunk)] * len(chunk))
        return np.array(probs), np.full(len(texts), np.nan), np.array(ms)


class NLIPipelineRunner:
    """The original per-pair HF pipeline, kept as the reference for equivalence checks."""

    def __init__(self, dtype: str = "auto", device: str | None = None):
        import torch
        from transformers import pipeline

        device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.pipe = pipeline("zero-shot-classification", model=MODEL, device=device, dtype=dtype)

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
