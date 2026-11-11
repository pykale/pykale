import json

import numpy as np
import pytest
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from kale.evaluate.classification_metrics import MetricError
from kale.evaluate.regression_metrics import regression_metrics

TARGETS = np.array([1.0, 2.0, 3.0, 4.0])


@pytest.mark.parametrize(
    "predictions, expected", [(TARGETS, 1.0), (np.full(4, TARGETS.mean()), 0.0), (TARGETS[::-1], -3.0)]
)
def test_regression_metrics_r2_known_values(predictions, expected):
    assert regression_metrics(TARGETS, predictions)["r2"] == pytest.approx(expected)


def test_regression_metrics_matches_sklearn():
    generator = np.random.default_rng(0)
    targets = generator.normal(size=50)
    predictions = targets + generator.normal(size=50) * 0.5
    summary = regression_metrics(targets, predictions)
    assert summary["r2"] == pytest.approx(r2_score(targets, predictions))
    assert summary["rmse"] == pytest.approx(np.sqrt(mean_squared_error(targets, predictions)))
    assert summary["mae"] == pytest.approx(mean_absolute_error(targets, predictions))


@pytest.mark.parametrize(
    "targets, predictions, match",
    [
        (np.zeros(4), np.zeros(5), "4 targets and 5 predictions"),
        (np.array([]), np.array([]), "at least one sample"),
    ],
)
def test_regression_metrics_rejects_invalid_inputs(targets, predictions, match):
    with pytest.raises(MetricError, match=match):
        regression_metrics(targets, predictions)


def test_regression_metrics_known_errors():
    targets = np.array([10.0, 20.0, 30.0])
    summary = regression_metrics(targets, targets + np.array([1.0, -2.0, 4.0]), unit="mmHg")
    assert set(summary) == {"n", "rmse", "mae", "bias", "unit", "r2"}
    assert summary["n"] == 3
    assert summary["rmse"] == pytest.approx(np.sqrt(7.0))
    assert summary["mae"] == pytest.approx(7.0 / 3.0)
    assert summary["bias"] == pytest.approx(1.0)
    assert summary["unit"] == "mmHg"
    json.dumps(summary)


@pytest.mark.parametrize("offset", [5.0, -5.0])
def test_regression_metrics_bias_is_signed(offset):
    targets = np.array([10.0, 20.0, 30.0])
    assert regression_metrics(targets, targets + offset)["bias"] == pytest.approx(offset)


def test_regression_metrics_unit_is_a_label_only():
    targets = np.array([10.0, 20.0])
    assert regression_metrics(targets, targets + 1)["rmse"] == regression_metrics(targets, targets + 1, "mmHg")["rmse"]


# A constant 0.1 leaves a tiny positive total sum of squares after rounding, so it must be detected directly.
@pytest.mark.parametrize("value", [3.0, 0.1])
def test_regression_metrics_constant_target_has_no_r2(value):
    summary = regression_metrics(np.full(3, value), np.full(3, value + 1.0))
    assert summary["r2"] is None
    assert summary["rmse"] == pytest.approx(1.0)


def test_regression_metrics_rejects_mismatched_lengths():
    with pytest.raises(MetricError, match="targets and"):
        regression_metrics(np.zeros(3), np.zeros(2))
