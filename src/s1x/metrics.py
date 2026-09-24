"""Metrics. Everything takes a (n, C) probability matrix and integer labels.

Calibration is reported before and after temperature scaling, with the temperature fit
on the calibration split only — never on test.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import cohen_kappa_score, f1_score


def accuracy(p: np.ndarray, y: np.ndarray) -> float:
    return float((p.argmax(1) == y).mean())


def macro_f1(p: np.ndarray, y: np.ndarray) -> float:
    return float(f1_score(y, p.argmax(1), average="macro", labels=np.arange(p.shape[1]), zero_division=0))


def ece(p: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    """Expected calibration error of the top-1 prediction, equal-width bins."""
    conf = p.max(1)
    correct = (p.argmax(1) == y).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if mask.any():
            total += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(total)


def brier(p: np.ndarray, y: np.ndarray) -> float:
    onehot = np.eye(p.shape[1])[y]
    return float(((p - onehot) ** 2).sum(1).mean())


def reliability(p: np.ndarray, y: np.ndarray, bins: int = 15) -> list[dict]:
    """Per-bin confidence vs accuracy, for the reliability diagram."""
    conf = p.max(1)
    correct = (p.argmax(1) == y).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if mask.any():
            out.append({"lo": float(lo), "hi": float(hi), "n": int(mask.sum()),
                        "confidence": float(conf[mask].mean()), "accuracy": float(correct[mask].mean())})
    return out


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def fit_temperature(p_calib: np.ndarray, y_calib: np.ndarray) -> float:
    """One temperature minimising NLL on the calibration split. Works from probabilities."""
    logits = np.log(np.clip(p_calib, 1e-12, 1.0))

    def nll(t: float) -> float:
        q = _softmax(logits / t)
        return float(-np.log(np.clip(q[np.arange(len(y_calib)), y_calib], 1e-12, 1.0)).mean())

    return float(minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded").x)


def apply_temperature(p: np.ndarray, t: float) -> np.ndarray:
    return _softmax(np.log(np.clip(p, 1e-12, 1.0)) / t)


def selective_accuracy(p: np.ndarray, y: np.ndarray, coverage: float) -> float:
    """Accuracy on the most-confident `coverage` fraction — is the confidence good for routing?"""
    n = max(1, int(round(len(y) * coverage)))
    order = np.argsort(-p.max(1), kind="stable")[:n]
    return float((p[order].argmax(1) == y[order]).mean())


def ordinal(p: np.ndarray, y: np.ndarray) -> dict:
    pred = p.argmax(1)
    return {
        "mae": float(np.abs(pred - y).mean()),
        "qwk": float(cohen_kappa_score(y, pred, weights="quadratic")),
    }


def summary(p: np.ndarray, y: np.ndarray, *, p_calib: np.ndarray | None = None, y_calib: np.ndarray | None = None,
            ordinal_task: bool = False) -> dict:
    out = {
        "n": int(len(y)),
        "accuracy": accuracy(p, y),
        "macro_f1": macro_f1(p, y),
        "ece": ece(p, y),
        "brier": brier(p, y),
        "sel_acc@50": selective_accuracy(p, y, 0.5),
        "sel_acc@80": selective_accuracy(p, y, 0.8),
        "sel_acc@95": selective_accuracy(p, y, 0.95),
    }
    if p_calib is not None and y_calib is not None:
        t = fit_temperature(p_calib, y_calib)
        pt = apply_temperature(p, t)
        out |= {"temperature": t, "ece_ts": ece(pt, y), "brier_ts": brier(pt, y)}
    if ordinal_task:
        out |= ordinal(p, y)
    return out


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_resamples: int = 10_000, seed: int = 20260924) -> dict:
    """Paired bootstrap of mean(a) - mean(b) over test examples.

    a, b: per-example scores on the same examples (1/0 correctness, or the share of seeds
    correct for a k-shot method). Resamples examples with replacement; returns the observed
    difference and a percentile 95% CI.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_resamples, len(d)))
    boots = d[idx].mean(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"diff": float(d.mean()), "lo": float(lo), "hi": float(hi), "excludes_zero": bool(lo > 0 or hi < 0)}
