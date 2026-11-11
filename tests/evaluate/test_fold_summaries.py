import numpy as np
import pytest

from kale.evaluate.classification_metrics import binary_metrics, MetricError, multiclass_roc_auc
from kale.evaluate.cross_validation import pool_folds, summarize_folds


@pytest.fixture(scope="module")
def binary_scores():
    generator = np.random.default_rng(0)
    labels = generator.integers(0, 2, size=60)
    return labels, generator.normal(size=60) + labels


@pytest.fixture(scope="module")
def fold_metrics(binary_scores):
    labels, scores = binary_scores
    generator = np.random.default_rng(1)
    return [binary_metrics(labels, scores + generator.normal(size=len(scores)) * 0.3) for _ in range(5)]


def test_summarize_folds_reports_mean_std_and_values(fold_metrics):
    summary = summarize_folds(fold_metrics)
    values = [fold["roc_auc"] for fold in fold_metrics]
    assert summary["n_folds"] == 5
    assert summary["roc_auc"]["mean"] == pytest.approx(np.mean(values))
    assert summary["roc_auc"]["std"] == pytest.approx(np.std(values))
    assert summary["roc_auc"]["values"] == pytest.approx(values)


def test_summarize_folds_sample_standard_deviation(fold_metrics):
    values = [fold["roc_auc"] for fold in fold_metrics]
    assert summarize_folds(fold_metrics, ddof=1)["roc_auc"]["std"] == pytest.approx(np.std(values, ddof=1))


def test_summarize_folds_single_fold_has_zero_sample_standard_deviation():
    assert summarize_folds([{"roc_auc": 0.7}], ddof=1)["roc_auc"]["std"] == 0.0


def test_summarize_folds_keeps_nan_from_a_failed_fold():
    summary = summarize_folds([{"roc_auc": 0.7}, {"roc_auc": float("nan")}, {"roc_auc": 0.8}], ddof=1)
    assert np.isnan(summary["roc_auc"]["mean"])


def test_summarize_folds_skips_non_numeric_values(fold_metrics):
    summary = summarize_folds(fold_metrics)
    assert "confusion" not in summary
    assert {"roc_auc", "threshold", "n"} <= set(summary)


def test_summarize_folds_accepts_numpy_scalars_and_skips_bools():
    folds = [{"auc": np.float32(0.5), "count": np.int64(3), "flag": True}, {"auc": 0.7, "count": 5, "flag": False}]
    summary = summarize_folds(folds)
    assert set(summary) == {"n_folds", "auc", "count"}
    assert summary["count"]["mean"] == pytest.approx(4.0)


def test_summarize_folds_honors_named_keys():
    summary = summarize_folds([{"roc_auc": 0.7, "f1": 0.5}, {"roc_auc": 0.8}], keys=["roc_auc", "f1", "absent"])
    assert set(summary) == {"n_folds", "roc_auc", "f1"}
    assert summary["roc_auc"]["mean"] == pytest.approx(0.75)
    assert summary["f1"]["values"] == [0.5]


def test_summarize_folds_rejects_no_fold():
    with pytest.raises(MetricError, match="at least one fold"):
        summarize_folds([])


def test_pool_folds_concatenates_in_order():
    folds = [(np.array([0, 1]), np.array([0.1, 0.9])), (np.array([1]), np.array([0.6]))]
    labels, scores = pool_folds(iter(folds))
    np.testing.assert_array_equal(labels, [0, 1, 1])
    np.testing.assert_array_equal(scores, [0.1, 0.9, 0.6])


def test_pool_folds_keeps_multiclass_score_columns():
    folds = [(np.array([0, 2]), np.eye(3)[[0, 2]]), (np.array([1]), np.eye(3)[[1]])]
    labels, scores = pool_folds(folds)
    assert scores.shape == (3, 3)
    assert multiclass_roc_auc(labels, scores) == pytest.approx(1.0)


def test_pool_folds_rejects_no_fold():
    with pytest.raises(MetricError, match="at least one fold"):
        pool_folds([])


def test_pool_folds_rejects_misaligned_fold():
    # Lengths that cancel out across folds would still concatenate to equal lengths, so each fold is checked.
    folds = [(np.array([0, 1]), np.array([0.1])), (np.array([1]), np.array([0.6, 0.7]))]
    with pytest.raises(MetricError, match="fold 0 has 2 labels but 1 scores"):
        pool_folds(folds)
