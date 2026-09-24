"""SetFit on the committed k-shot draws: contrastive fine-tune of a sentence-transformer body,
then a logistic-regression head (the library defaults), trained once per (task, k, seed).

Training follows the SetFit paper's default recipe: 20 pair-iterations per example (each yields
one positive and one negative pair) and ONE epoch over those pairs. A 1,000-step cap bounds only
the largest cells (banking77 at k=16/64, and the k=64 cells of the other tasks).

Why max_steps is computed rather than fixed: in SetFit (as in HF TrainingArguments) a positive
max_steps overrides num_epochs, so a fixed max_steps=1000 forced exactly 1,000 steps in every
cell — 8-12 epochs on small k, under one epoch on large k. Found by profiling on the M4 run;
every cell trained before this fix was re-run.
Run one cell with `s1x fewshot setfit <task> --k 8 --seeds 0`.
"""
from __future__ import annotations

from s1x.runners.fewshot import _device, _full_proba, _single_example_ms

SETFIT_MODEL = "sentence-transformers/paraphrase-mpnet-base-v2"  # SetFit's reference backbone
LATENCY_N = 50  # single-example timings per (task, k), seed 0 only
SETFIT_ARGS = dict(batch_size=16, num_epochs=1, num_iterations=20)
STEP_CAP = 1000


def epoch_steps(n_examples: int, num_iterations: int = SETFIT_ARGS["num_iterations"],
                batch_size: int = SETFIT_ARGS["batch_size"]) -> int:
    """Contrastive steps in one epoch: 2 pairs per example per iteration, in batches."""
    return -(-2 * num_iterations * n_examples // batch_size)


class SetFitRunner:
    def __init__(self, max_steps: int | None = None):
        self.override = max_steps  # smoke tests only

    def fit_predict(self, data, shot: str, seed: int) -> dict:
        from datasets import Dataset
        from setfit import SetFitModel, Trainer, TrainingArguments

        ids = data.shot_ids[shot]
        train = Dataset.from_dict({"text": [data.pool.texts[i] for i in ids],
                                   "label": [int(v) for v in data.pool.labels[ids]]})
        model = SetFitModel.from_pretrained(SETFIT_MODEL, device=_device())
        steps = self.override or min(epoch_steps(len(ids)), STEP_CAP)
        args = TrainingArguments(**SETFIT_ARGS, max_steps=steps, seed=seed, report_to="none", save_strategy="no",
                                 show_progress_bar=False)
        Trainer(model=model, args=args, train_dataset=train).train()
        n_classes = len(data.labels)
        clf = model.model_head

        def encode(texts):
            return model.model_body.encode(texts, batch_size=64, normalize_embeddings=model.normalize_embeddings,
                                           convert_to_numpy=True, show_progress_bar=False)

        # Latency depends on the backbone and the text, not on which examples it was trained
        # on, so it is timed once per (task, k) — seed 0, 50 examples — and skipped for the
        # other seeds. One-at-a-time MPS calls dominated each run's wall time when every
        # seed timed 200. The cross-method speed comparison is measured separately on one
        # machine (`s1x latency`), so this is a per-run sanity figure, not the headline.
        ms = (_single_example_ms(encode, lambda e: clf.predict_proba(e), data.test.texts, n=LATENCY_N)
              if seed == 0 else [])
        return {"test": _full_proba(clf, encode(data.test.texts), n_classes),
                "calib": _full_proba(clf, encode(data.calib.texts), n_classes),
                "ms": ms}
