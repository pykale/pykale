import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from kale.evaluate.bootstrap import bootstrap_ci
from kale.evaluate.classification_metrics import MetricError, multiclass_roc_auc, roc_auc
from kale.evaluate.regression_metrics import r2_score


@pytest.fixture(scope="module")
def binary_scores():
    generator = np.random.default_rng(0)
    labels = generator.integers(0, 2, size=60)
    return labels, generator.normal(size=60) + labels


def test_bootstrap_ci_contains_point_and_is_reproducible(binary_scores):
    labels, scores = binary_scores
    first = bootstrap_ci(roc_auc, labels, scores, n_resamples=300, seed=7)
    assert first == bootstrap_ci(roc_auc, labels, scores, n_resamples=300, seed=7)
    assert first["low"] <= first["point"] <= first["high"]
    assert first["point"] == pytest.approx(roc_auc(labels, scores))
    assert first["confidence"] == 0.95


def test_bootstrap_ci_wider_confidence_gives_wider_interval(binary_scores):
    labels, scores = binary_scores
    narrow = bootstrap_ci(roc_auc, labels, scores, n_resamples=300, confidence=0.5)
    wide = bootstrap_ci(roc_auc, labels, scores, n_resamples=300, confidence=0.99)
    assert wide["high"] - wide["low"] > narrow["high"] - narrow["low"]


def test_bootstrap_ci_resamples_every_array_with_the_same_indices():
    # Predictions equal to targets give R-squared of one only if each resample keeps the pairs aligned.
    targets = np.arange(20.0)
    interval = bootstrap_ci(r2_score, targets, targets.copy(), n_resamples=50)
    assert interval["low"] == interval["high"] == pytest.approx(1.0)


def test_bootstrap_ci_paired_difference_resamples_both_models_together(binary_scores):
    labels, scores = binary_scores
    baseline = scores + np.random.default_rng(3).normal(size=len(scores))
    interval = bootstrap_ci(
        lambda y, s, b: roc_auc(y, s) - roc_auc(y, b), labels, scores, baseline, n_resamples=300, seed=5
    )
    assert interval["point"] == pytest.approx(roc_auc(labels, scores) - roc_auc(labels, baseline))
    # Scoring both models on one shared draw per resample reproduces the interval exactly.
    generator = np.random.default_rng(5)
    differences = []
    for _ in range(300):
        draw = generator.integers(0, len(labels), size=len(labels))
        differences.append(roc_auc(labels[draw], scores[draw]) - roc_auc(labels[draw], baseline[draw]))
    assert (interval["low"], interval["high"]) == pytest.approx(tuple(np.percentile(differences, [2.5, 97.5])))


def test_bootstrap_ci_accepts_multiclass_scores():
    labels = np.arange(30) % 3
    interval = bootstrap_ci(multiclass_roc_auc, labels, np.eye(3)[labels], n_resamples=50)
    assert interval["point"] == pytest.approx(1.0)


def test_bootstrap_ci_skips_unscorable_resamples():
    # One positive in twenty: about a third of resamples miss it and cannot be scored.
    labels = np.array([0] * 19 + [1])
    scores = np.concatenate([np.random.default_rng(0).normal(size=19), [3.0]])
    assert 0 < bootstrap_ci(roc_auc, labels, scores, n_resamples=200)["n_used"] < 200


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_bootstrap_ci_skips_non_finite_resamples(binary_scores, bad_value):
    labels, scores = binary_scores
    calls = []

    def auc_with_one_bad_resample(y, s):
        # Call 1 is the full data; call 5 is the fourth resample.
        calls.append(None)
        return bad_value if len(calls) == 5 else roc_auc(y, s)

    interval = bootstrap_ci(auc_with_one_bad_resample, labels, scores, n_resamples=200, seed=3)
    generator = np.random.default_rng(3)
    draws = [generator.integers(0, len(labels), size=len(labels)) for _ in range(200)]
    kept = [roc_auc(labels[draw], scores[draw]) for index, draw in enumerate(draws) if index != 3]
    assert interval["n_used"] == 199
    assert interval["low"] == pytest.approx(np.percentile(kept, 2.5))
    assert interval["high"] == pytest.approx(np.percentile(kept, 97.5))


def test_bootstrap_ci_rejects_when_no_resample_is_finite():
    def nan_on_resamples(values):
        return 0.0 if np.unique(values).size == values.size else np.nan

    with pytest.raises(MetricError, match="no bootstrap resample could be scored to a finite value"):
        bootstrap_ci(nan_on_resamples, np.arange(10), n_resamples=20)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf])
def test_bootstrap_ci_rejects_non_finite_estimate(bad_value):
    calls = []

    def bad_metric(values):
        calls.append(None)
        return bad_value

    with pytest.raises(MetricError, match="not finite on all samples"):
        bootstrap_ci(bad_metric, np.arange(10), n_resamples=20)
    assert len(calls) == 1


def test_bootstrap_ci_rejects_when_no_resample_scores():
    def distinct_only(values):
        # Defined on the original samples, which are distinct, but almost never on a resample, which repeats some.
        if np.unique(values).size < values.size:
            raise MetricError("repeated sample")
        return 0.0

    with pytest.raises(MetricError, match="no bootstrap resample could be scored"):
        bootstrap_ci(distinct_only, np.arange(10), n_resamples=20)


def test_bootstrap_ci_fails_on_the_original_samples_before_resampling():
    calls = []

    def single_class_auc(labels, scores):
        calls.append(len(labels))
        return roc_auc(labels, scores)

    with pytest.raises(MetricError, match="both classes"):
        bootstrap_ci(single_class_auc, np.zeros(20), np.zeros(20), n_resamples=20)
    assert calls == [20]


@pytest.mark.parametrize("error", [RuntimeError, ValueError])
def test_bootstrap_ci_propagates_errors_that_do_not_mark_undefined(error):
    def broken_on_resamples(values):
        # Scores the original samples, so the failure can only come from inside the resampling loop.
        if np.unique(values).size < values.size:
            raise error("bug in metric")
        return 0.0

    with pytest.raises(error, match="bug in metric"):
        bootstrap_ci(broken_on_resamples, np.arange(10), n_resamples=5)


def test_bootstrap_ci_skips_plain_value_errors_only_when_asked():
    def strict_auc(labels, scores):
        # Like an external metric that raises a plain ValueError, not MetricError, on a resample without a positive.
        if np.unique(labels).size < 2:
            raise ValueError("only one class present")
        return roc_auc_score(labels, scores)

    labels = np.array([0] * 19 + [1])
    scores = np.linspace(0.0, 1.0, 20)
    with pytest.raises(ValueError, match="only one class present"):
        bootstrap_ci(strict_auc, labels, scores, n_resamples=200)
    interval = bootstrap_ci(strict_auc, labels, scores, n_resamples=200, undefined_errors=(ValueError,))
    assert 0 < interval["n_used"] < 200


@pytest.mark.parametrize(
    "arrays, kwargs, error, match",
    [
        ((), {}, ValueError, "at least one array"),
        ((np.zeros(4), np.zeros(5)), {}, MetricError, r"same number of samples, got \[4, 5\]"),
        ((np.array([]), np.array([])), {}, MetricError, "at least one sample"),
        ((np.zeros(4), np.zeros(4)), {"confidence": 1.5}, ValueError, "confidence must lie"),
        ((np.zeros(4), np.zeros(4)), {"n_resamples": 0}, ValueError, "n_resamples must be positive"),
    ],
)
def test_bootstrap_ci_rejects_invalid_inputs(arrays, kwargs, error, match):
    with pytest.raises(error, match=match):
        bootstrap_ci(roc_auc, *arrays, **kwargs)
