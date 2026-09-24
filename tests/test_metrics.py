import numpy as np

from s1x import metrics


def test_perfectly_calibrated_has_zero_ece():
    # 100 examples at confidence 0.8, exactly 80 right.
    p = np.tile([0.8, 0.2], (100, 1))
    y = np.array([0] * 80 + [1] * 20)
    assert metrics.ece(p, y) < 1e-9
    assert metrics.accuracy(p, y) == 0.8


def test_overconfident_model_has_ece_equal_to_gap():
    p = np.tile([0.99, 0.01], (100, 1))
    y = np.array([0] * 60 + [1] * 40)
    assert abs(metrics.ece(p, y) - 0.39) < 1e-9


def test_temperature_scaling_fixes_overconfidence():
    rng = np.random.default_rng(0)
    n = 4000
    y = rng.integers(0, 3, n)
    logits = rng.normal(0, 1, (n, 3))
    logits[np.arange(n), y] += 1.0          # a genuinely informative model...
    p = metrics._softmax(logits * 4.0)       # ...made over-confident
    t = metrics.fit_temperature(p[:2000], y[:2000])
    assert t > 2.0
    after = metrics.ece(metrics.apply_temperature(p[2000:], t), y[2000:])
    before = metrics.ece(p[2000:], y[2000:])
    assert after < before / 2
    assert metrics.accuracy(metrics.apply_temperature(p, t), y) == metrics.accuracy(p, y)  # argmax unchanged


def test_selective_accuracy_rewards_useful_confidence():
    p = np.array([[0.95, 0.05], [0.9, 0.1], [0.55, 0.45], [0.51, 0.49]])
    y = np.array([0, 0, 1, 1])  # the confident ones are right, the unsure ones wrong
    assert metrics.selective_accuracy(p, y, 0.5) == 1.0
    assert metrics.selective_accuracy(p, y, 1.0) == 0.5


def test_brier_and_ordinal():
    p = np.eye(5)[[0, 1, 2, 3, 4]]
    y = np.array([0, 1, 2, 3, 4])
    assert metrics.brier(p, y) == 0.0
    assert metrics.ordinal(p, y) == {"mae": 0.0, "qwk": 1.0}


def test_paired_bootstrap():
    rng = np.random.default_rng(1)
    base = rng.random(1000) < 0.7
    same = metrics.paired_bootstrap(base, base)
    assert same["diff"] == 0.0 and same["lo"] == 0.0 and same["hi"] == 0.0 and not same["excludes_zero"]
    better = base | (rng.random(1000) < 0.3)  # strictly more correct
    r = metrics.paired_bootstrap(better, base)
    assert r["diff"] > 0 and r["lo"] > 0 and r["excludes_zero"]
    assert r["lo"] <= r["diff"] <= r["hi"]
    assert metrics.paired_bootstrap(better, base) == r  # fixed seed -> reproducible
    noise = metrics.paired_bootstrap(rng.random(1000) < 0.5, rng.random(1000) < 0.5)
    assert noise["lo"] < noise["diff"] < noise["hi"]


def test_setfit_trains_one_epoch_capped():
    from s1x.runners.setfit import STEP_CAP, epoch_steps
    assert epoch_steps(48) == 120            # emotion k=8: 1,920 pairs / 16, as logged on the M4
    assert epoch_steps(32) == 80             # ag_news k=8
    assert min(epoch_steps(77 * 64), STEP_CAP) == STEP_CAP  # banking77 k=64 hits the cap
