"""Tasks and their fixed splits.

Every method sees the same test examples, the same calibration examples, and the same
k-shot draws, identified by row id and written to data/splits/<task>.json. Splits are
created once, from a fixed seed, and committed — every later number is reproducible.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parents[2] / "data"
TEST_SIZE = 1000
CALIB_SIZE = 500
SPLIT_SEED = 20260924
SHOT_SEEDS = (0, 1, 2, 3, 4)
SHOTS = (8, 16, 64)


@dataclass(frozen=True)
class Task:
    name: str
    hf_id: str
    kind: str  # "choice" | "score" | "noul"
    text_field: str
    label_field: str
    labels: tuple[str, ...]  # human-readable names, index = label id
    instruction: str  # the one question every zero-shot method is asked
    has_test_split: bool = True


TASKS: dict[str, Task] = {
    t.name: t
    for t in [
        Task(
            "ag_news", "fancyzhx/ag_news", "choice", "text", "label",
            ("World", "Sports", "Business", "Sci/Tech"),
            "What is this news article about?",
        ),
        Task(
            "emotion", "dair-ai/emotion", "choice", "text", "label",
            ("sadness", "joy", "love", "anger", "fear", "surprise"),
            "Which emotion does this message express?",
        ),
        Task(
            "banking77", "PolyAI/banking77", "choice", "text", "label",
            (),  # filled from the dataset's own label names
            "What is the customer asking the bank about?",
        ),
        Task(
            "sst5", "SetFit/sst5", "score", "text", "label",
            ("very negative", "negative", "neutral", "positive", "very positive"),
            "How positive is this review?",
        ),
        Task(
            "sms_spam", "ucirvine/sms_spam", "noul", "sms", "label",
            ("not spam", "spam"),
            "Is this text message spam?",
            has_test_split=False,
        ),
    ]
}


@dataclass
class Split:
    texts: list[str]
    labels: np.ndarray


@dataclass
class TaskData:
    task: Task
    labels: tuple[str, ...]
    test: Split
    calib: Split
    pool: Split  # k-shot draws come from here; disjoint from test and calib
    shot_ids: dict[str, list[int]]  # "k=16,seed=3" -> row indices into pool


def _load(task: Task):
    from datasets import load_dataset

    ds = load_dataset(task.hf_id)
    labels = task.labels or tuple(ds["train"].features[task.label_field].names)
    train = ds["train"]
    test = ds["test"] if task.has_test_split else None
    return train, test, labels


def _stratified(y: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Indices for a sample of size n that keeps class proportions."""
    if n >= len(y):
        return np.arange(len(y))
    picked: list[int] = []
    classes, counts = np.unique(y, return_counts=True)
    quotas = np.floor(counts / counts.sum() * n).astype(int)
    for cls, quota in zip(classes, quotas):
        members = np.flatnonzero(y == cls)
        picked.extend(rng.choice(members, size=min(quota, len(members)), replace=False))
    rest = np.setdiff1d(np.arange(len(y)), picked)
    picked.extend(rng.choice(rest, size=n - len(picked), replace=False))
    return np.sort(np.array(picked))


def build_splits(name: str) -> dict:
    task = TASKS[name]
    train, test, labels = _load(task)
    rng = np.random.default_rng(SPLIT_SEED)
    y_train = np.array(train[task.label_field])

    if test is not None:
        y_test = np.array(test[task.label_field])
        test_ids = _stratified(y_test, TEST_SIZE, rng).tolist()
        remaining = np.arange(len(y_train))
        source_test = "test"
    else:
        # One split only: carve test out of it first, then everything else from the rest.
        test_ids = _stratified(y_train, TEST_SIZE, rng).tolist()
        remaining = np.setdiff1d(np.arange(len(y_train)), test_ids)
        source_test = "train"

    calib_local = _stratified(y_train[remaining], CALIB_SIZE, rng)
    calib_ids = remaining[calib_local].tolist()
    pool_ids = np.setdiff1d(remaining, calib_ids)

    shots: dict[str, list[int]] = {}
    y_pool = y_train[pool_ids]
    for k in SHOTS:
        for seed in SHOT_SEEDS:
            draw_rng = np.random.default_rng([SPLIT_SEED, k, seed])
            chosen: list[int] = []
            for cls in range(len(labels)):
                members = np.flatnonzero(y_pool == cls)
                chosen.extend(draw_rng.choice(members, size=min(k, len(members)), replace=False).tolist())
            shots[f"k={k},seed={seed}"] = sorted(int(i) for i in chosen)

    record = {
        "task": name,
        "hf_id": task.hf_id,
        "labels": list(labels),
        "test_source": source_test,
        "test_ids": [int(i) for i in test_ids],
        "calib_ids": [int(i) for i in calib_ids],
        "pool_ids": [int(i) for i in pool_ids],
        "shots": shots,
    }
    out = DATA / "splits" / f"{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record))
    return record


def load_task(name: str) -> TaskData:
    task = TASKS[name]
    path = DATA / "splits" / f"{name}.json"
    record = json.loads(path.read_text()) if path.exists() else build_splits(name)
    train, test, _ = _load(task)
    source = test if record["test_source"] == "test" else train

    def take(ds, ids) -> Split:
        rows = ds.select(ids)
        return Split(list(rows[task.text_field]), np.array(rows[task.label_field]))

    return TaskData(
        task=task,
        labels=tuple(record["labels"]),
        test=take(source, record["test_ids"]),
        calib=take(train, record["calib_ids"]),
        pool=take(train, record["pool_ids"]),
        shot_ids=record["shots"],
    )
