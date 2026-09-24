"""The batched NLI runner must give the HF zero-shot pipeline's single-label probabilities.

Runs in fp32 on CPU (in fp16 the padded batch differs from per-pair passes by kernel noise
of ~1e-3). Skipped when transformers/torch or the model weights are not available locally.
"""
import os

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from s1x.runners import nli  # noqa: E402
from s1x.tasks import TASKS  # noqa: E402


def _model_cached() -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache

        return isinstance(try_to_load_from_cache(nli.MODEL, "config.json"), str)
    except Exception:
        return False


@pytest.mark.skipif(not _model_cached(), reason="NLI model weights not in the local HF cache")
def test_batched_matches_pipeline_single_label():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    task = TASKS["emotion"]
    labels = ("sadness", "joy", "love", "anger", "fear", "surprise")
    texts = [
        "i feel so alone since my best friend moved away",
        "i am thrilled that we finally won the championship after all these years of trying",
        "why would anyone do that to me",
    ]
    ref, _, _ = nli.NLIPipelineRunner(dtype="float32", device="cpu").predict(task, labels, texts)
    runner = nli.NLIRunner(dtype="float32", device="cpu")
    single, _, _ = runner.predict(task, labels, texts)
    runner.batch_examples = 3
    multi, _, _ = runner.predict(task, labels, texts)
    assert np.abs(single - ref).max() < 1e-4
    assert np.abs(multi - ref).max() < 1e-4
    assert np.allclose(single.sum(1), 1.0)
