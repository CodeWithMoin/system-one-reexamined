"""SetFit on the committed k-shot draws: contrastive fine-tune of a sentence-transformer body,
then a logistic-regression head (the library defaults), trained once per (task, k, seed).

Budget kept modest so the grid finishes in hours on a laptop: 20 pair-iterations per example,
one epoch, and at most 1,000 contrastive steps (binding only for banking77 at k=64).
Run one cell with `s1x fewshot setfit <task> --k 8 --seeds 0`.
"""
from __future__ import annotations

from s1x.runners.fewshot import _device, _full_proba, _single_example_ms

SETFIT_MODEL = "sentence-transformers/paraphrase-mpnet-base-v2"  # SetFit's reference backbone
SETFIT_ARGS = dict(batch_size=16, num_epochs=1, num_iterations=20, max_steps=1000)


class SetFitRunner:
    def __init__(self, max_steps: int | None = None):
        self.args = SETFIT_ARGS | ({"max_steps": max_steps} if max_steps else {})

    def fit_predict(self, data, shot: str, seed: int) -> dict:
        from datasets import Dataset
        from setfit import SetFitModel, Trainer, TrainingArguments

        ids = data.shot_ids[shot]
        train = Dataset.from_dict({"text": [data.pool.texts[i] for i in ids],
                                   "label": [int(v) for v in data.pool.labels[ids]]})
        model = SetFitModel.from_pretrained(SETFIT_MODEL, device=_device())
        args = TrainingArguments(**self.args, seed=seed, report_to="none", save_strategy="no",
                                 show_progress_bar=False)
        Trainer(model=model, args=args, train_dataset=train).train()
        n_classes = len(data.labels)
        clf = model.model_head

        def encode(texts):
            return model.model_body.encode(texts, batch_size=64, normalize_embeddings=model.normalize_embeddings,
                                           convert_to_numpy=True, show_progress_bar=False)

        return {"test": _full_proba(clf, encode(data.test.texts), n_classes),
                "calib": _full_proba(clf, encode(data.calib.texts), n_classes),
                "ms": _single_example_ms(encode, lambda e: clf.predict_proba(e), data.test.texts)}
