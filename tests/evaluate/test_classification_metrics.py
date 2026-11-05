import json

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from kale.evaluate.classification_metrics import (
    binary_metrics,
    mean_roc_curve,
    MetricError,
    multiclass_metrics,
    multiclass_roc_auc,
    roc_auc,
    sensitivity_specificity,
)

BINARY_KEYS = {
    "n",
    "n_positive",
    "positive_rate",
    "threshold",
    "roc_auc",
    "average_precision",
    "accuracy",
    "balanced_accuracy",
    "sensitivity",
    "specificity",
    "f1",
    "mcc",
    "confusion",
}


@pytest.fixture(scope="module")
def binary_scores():
    generator = np.random.default_rng(0)
    labels = generator.integers(0, 2, size=60)
    return labels, generator.normal(size=60) + labels


@pytest.fixture(scope="module")
def three_class():
    generator = np.random.default_rng(0)
    labels = generator.integers(0, 3, size=60)
    probabilities = generator.dirichlet(np.ones(3), size=60)
    probabilities[np.arange(60), labels] += 1.0
    return labels, probabilities / probabilities.sum(axis=1, keepdims=True)


@pytest.mark.parametrize(
    "scores, expected", [([0.1, 0.2, 0.8, 0.9], 1.0), ([0.9, 0.8, 0.2, 0.1], 0.0), ([0.1, 0.4, 0.35, 0.8], 0.75)]
)
def test_roc_auc_known_rankings(scores, expected):
    assert roc_auc([0, 0, 1, 1], scores) == pytest.approx(expected)


def test_roc_auc_is_invariant_to_sigmoid():
    labels = np.array([0, 1, 0, 1, 1])
    logits = np.array([-2.0, 1.0, -0.5, 0.3, 2.0])
    assert roc_auc(labels, logits) == pytest.approx(roc_auc(labels, 1.0 / (1.0 + np.exp(-logits))))


def test_roc_auc_rejects_one_class():
    with pytest.raises(MetricError, match="needs both classes"):
        roc_auc(np.zeros(10), np.random.default_rng(0).normal(size=10))


@pytest.mark.parametrize(
    "labels, invalid",
    [
        ([0.2, 1.2, 0.2, 1.2], r"\[0.2, 1.2\]"),
        ([0, 2, 0, 2], r"\[2\]"),
        ([-1, 1, -1, 1], r"\[-1\]"),
        ([0.0, np.nan, 1.0, 1.0], r"\[nan\]"),
    ],
)
@pytest.mark.parametrize("metric", [roc_auc, sensitivity_specificity, binary_metrics])
def test_binary_metrics_reject_labels_other_than_zero_and_one(metric, labels, invalid):
    with pytest.raises(MetricError, match=r"class indices 0 to 1, got " + invalid):
        metric(labels, [0.1, 0.9, 0.2, 0.8])


def test_binary_metrics_accept_float_and_boolean_labels():
    scores = [0.1, 0.9, 0.4, 0.6]
    expected = binary_metrics([0, 1, 0, 1], scores)
    assert binary_metrics([0.0, 1.0, 0.0, 1.0], scores) == expected
    assert binary_metrics([False, True, False, True], scores) == expected


def test_metric_error_is_a_value_error():
    assert issubclass(MetricError, ValueError)


def test_sensitivity_specificity_known_confusion():
    labels = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    probabilities = np.array([0.9, 0.9, 0.9, 0.1, 0.9, 0.1, 0.1, 0.1])
    assert sensitivity_specificity(labels, probabilities) == pytest.approx((0.75, 0.75))


def test_sensitivity_specificity_threshold_moves_operating_point():
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.2, 0.6, 0.4, 0.9])
    lenient = sensitivity_specificity(labels, probabilities, threshold=0.3)
    strict = sensitivity_specificity(labels, probabilities, threshold=0.8)
    assert lenient[0] > strict[0]
    assert lenient[1] < strict[1]


def test_threshold_is_inclusive():
    assert sensitivity_specificity([0, 1], [0.2, 0.5], threshold=0.5) == (1.0, 1.0)


def test_binary_metrics_keys_and_counts(binary_scores):
    labels, scores = binary_scores
    summary = binary_metrics(labels, scores)
    assert set(summary) == BINARY_KEYS
    assert summary["n"] == len(labels)
    assert summary["n_positive"] == int(labels.sum())
    assert sum(summary["confusion"].values()) == len(labels)
    json.dumps(summary)


def test_binary_metrics_matches_sensitivity_specificity(binary_scores):
    labels, scores = binary_scores
    summary = binary_metrics(labels, scores, threshold=0.4)
    assert (summary["sensitivity"], summary["specificity"]) == sensitivity_specificity(labels, scores, 0.4)


def test_binary_metrics_accepts_logits_and_probabilities(binary_scores):
    labels, scores = binary_scores
    from_logits = binary_metrics(labels, scores)
    from_probabilities = binary_metrics(labels, 1.0 / (1.0 + np.exp(-scores)))
    for key in ("roc_auc", "accuracy", "sensitivity", "specificity", "confusion"):
        assert from_logits[key] == pytest.approx(from_probabilities[key])


def test_score_type_resolves_logits_that_look_like_probabilities():
    # Logits that all fall in [0, 1] are ambiguous: "auto" reads them as probabilities, "logit" applies the sigmoid,
    # which puts every one above one half.
    labels, logits = np.array([0, 0, 1, 1]), np.array([0.2, 0.4, 0.6, 0.9])
    assert sensitivity_specificity(labels, logits) == (1.0, 1.0)
    assert sensitivity_specificity(labels, logits, score_type="probability") == (1.0, 1.0)
    assert sensitivity_specificity(labels, logits, score_type="logit") == (1.0, 0.0)
    assert binary_metrics(labels, logits, score_type="logit")["confusion"] == {"tn": 0, "fp": 2, "fn": 0, "tp": 2}


def test_score_type_logit_matches_auto_outside_unit_interval(binary_scores):
    labels, scores = binary_scores
    assert binary_metrics(labels, scores, score_type="logit") == binary_metrics(labels, scores)


@pytest.mark.parametrize(
    "scores, score_type, error, match",
    [
        ([-1.0, 0.2, 0.6, 0.9], "probability", MetricError, r"needs every score in \[0, 1\]"),
        ([0.1, 0.2, 0.6, 0.9], "logits", ValueError, "score_type must be one of"),
    ],
)
def test_score_type_rejects_invalid_binary_inputs(scores, score_type, error, match):
    with pytest.raises(error, match=match):
        binary_metrics([0, 0, 1, 1], scores, score_type=score_type)


@pytest.mark.parametrize("metric", [sensitivity_specificity, binary_metrics])
@pytest.mark.parametrize("score_type", ["auto", "logit"])
@pytest.mark.parametrize("bad_score", [np.nan, np.inf, -np.inf])
def test_binary_threshold_metrics_reject_nonfinite_scores(metric, score_type, bad_score):
    with pytest.raises(MetricError, match="scores must be finite"):
        metric([0, 1], [bad_score, 0.9], score_type=score_type)


@pytest.mark.parametrize("metric", [sensitivity_specificity, binary_metrics])
@pytest.mark.parametrize("threshold", [np.nan, np.inf, -np.inf])
def test_binary_threshold_metrics_reject_nonfinite_threshold(metric, threshold):
    with pytest.raises(MetricError, match="threshold must be finite"):
        metric([0, 1], [0.1, 0.9], threshold=threshold)


def test_multiclass_score_type_resolves_logit_rows_that_look_like_distributions():
    # Each row sums to one, so "auto" reads it as probabilities, while a softmax reorders some samples within a class.
    labels = np.array([0, 1, 2, 0, 1, 2])
    rows = np.array([[0.3, 0, 0.7], [0.2, 0.2, 0.6], [0.4, 0.2, 0.4], [0.2, 0, 0.8], [0.3, 0.4, 0.3], [0, 0.5, 0.5]])
    probabilities = np.exp(rows) / np.exp(rows).sum(axis=1, keepdims=True)
    assert multiclass_roc_auc(labels, rows) == pytest.approx(multiclass_roc_auc(labels, rows, score_type="probability"))
    assert multiclass_roc_auc(labels, rows, score_type="logit") == pytest.approx(
        roc_auc_score(labels, probabilities, multi_class="ovr", average="macro")
    )
    assert multiclass_roc_auc(labels, rows, score_type="logit") != pytest.approx(multiclass_roc_auc(labels, rows))


def test_multiclass_metrics_scores_two_probability_columns(binary_scores):
    # Two columns are what a softmax head over two classes returns; scikit-learn alone rejects them.
    labels, scores = binary_scores
    probabilities = 1.0 / (1.0 + np.exp(-scores))
    two_columns = np.c_[1.0 - probabilities, probabilities]
    assert multiclass_roc_auc(labels, two_columns) == pytest.approx(roc_auc(labels, probabilities))
    assert multiclass_metrics(labels, two_columns)["roc_auc_ovr"] == pytest.approx(roc_auc(labels, probabilities))


def test_binary_threshold_and_two_column_argmax_differ_only_at_one_half():
    labels = np.array([0, 1, 0, 1, 1, 0])
    probabilities = np.array([0.2, 0.5, 0.5, 0.8, 0.4, 0.6])
    two_columns = np.c_[1.0 - probabilities, probabilities]
    from_threshold = binary_metrics(labels, probabilities)["confusion"]
    from_argmax = np.array(multiclass_metrics(labels, two_columns)["confusion"])
    # The threshold calls both 0.5 samples positive; the argmax gives them to class 0.
    assert from_threshold == {"tn": 1, "fp": 2, "fn": 1, "tp": 2}
    np.testing.assert_array_equal(from_argmax, [[2, 1], [2, 1]])
    off_tie = probabilities != 0.5
    assert binary_metrics(labels[off_tie], probabilities[off_tie])["balanced_accuracy"] == pytest.approx(
        multiclass_metrics(labels[off_tie], two_columns[off_tie])["balanced_accuracy"]
    )


def test_multiclass_score_type_probability_rejects_non_distributions(three_class):
    labels, probabilities = three_class
    with pytest.raises(MetricError, match="every row to be a distribution"):
        multiclass_metrics(labels, probabilities * 2.0, score_type="probability")


@pytest.mark.parametrize("metric", [multiclass_roc_auc, multiclass_metrics])
@pytest.mark.parametrize("score_type", ["auto", "logit"])
@pytest.mark.parametrize("bad_score", [np.nan, np.inf, -np.inf])
def test_multiclass_metrics_reject_nonfinite_scores(metric, score_type, bad_score):
    scores = np.array([[bad_score, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]])
    with pytest.raises(MetricError, match="scores must be finite"):
        metric([0, 1, 2], scores, score_type=score_type)


def test_multiclass_rows_summing_to_one_with_negative_entries_are_not_distributions():
    labels = np.array([0, 1, 2, 0, 1, 2])
    rows = np.array([[1.5, -0.5, 0.0], [-0.2, 0.9, 0.3], [0.1, -0.1, 1.0]] * 2)
    with pytest.raises(MetricError, match="every row to be a distribution"):
        multiclass_roc_auc(labels, rows, score_type="probability")


def test_binary_metrics_all_negative_has_no_sensitivity():
    summary = binary_metrics(np.array([0] * 8 + [1] * 2), np.full(10, -10.0))
    assert summary["accuracy"] == pytest.approx(0.8)
    assert summary["sensitivity"] == pytest.approx(0.0)
    assert summary["specificity"] == pytest.approx(1.0)


def test_binary_metrics_perfect_classifier():
    summary = binary_metrics(np.array([0, 0, 1, 1]), np.array([0.01, 0.02, 0.98, 0.99]))
    for key in ("roc_auc", "accuracy", "balanced_accuracy", "sensitivity", "specificity", "f1", "mcc"):
        assert summary[key] == pytest.approx(1.0)


def test_binary_metrics_rejects_one_class():
    with pytest.raises(MetricError, match="needs both classes"):
        binary_metrics(np.ones(5), np.linspace(0, 1, 5))


def test_multiclass_metrics_summary(three_class):
    labels, probabilities = three_class
    summary = multiclass_metrics(labels, probabilities, class_names=["a", "b", "c"])
    assert summary["n"] == 60
    assert np.array(summary["confusion"]).shape == (3, 3)
    assert sum(map(sum, summary["confusion"])) == 60
    assert summary["classes"] == ["a", "b", "c"]
    assert "roc_auc_ovr" in summary
    json.dumps(summary)


def test_multiclass_metrics_default_class_names(three_class):
    labels, probabilities = three_class
    assert multiclass_metrics(labels, probabilities)["classes"] == ["0", "1", "2"]


@pytest.mark.parametrize("class_names", [[], ["only one"], ["a", "b", "c", "d"]])
def test_multiclass_metrics_rejects_wrong_number_of_class_names(three_class, class_names):
    labels, probabilities = three_class
    with pytest.raises(MetricError, match=f"expected 3 class names, got {len(class_names)}"):
        multiclass_metrics(labels, probabilities, class_names=class_names)


def test_multiclass_metrics_checks_class_names_when_a_class_is_absent():
    labels = [0, 0, 1, 1]
    scores = [[0.7, 0.2, 0.1], [0.6, 0.3, 0.1], [0.2, 0.7, 0.1], [0.1, 0.8, 0.1]]
    with pytest.raises(MetricError, match="expected 3 class names, got 2"):
        multiclass_metrics(labels, scores, class_names=["healthy", "disease"])


def test_multiclass_metrics_omits_auc_for_absent_class():
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([[0.7, 0.2, 0.1], [0.6, 0.3, 0.1], [0.2, 0.7, 0.1], [0.1, 0.8, 0.1]])
    assert "roc_auc_ovr" not in multiclass_metrics(labels, probabilities)


def test_multiclass_metrics_rejects_one_class():
    with pytest.raises(MetricError, match="at least two classes"):
        multiclass_metrics(np.zeros(4), np.full((4, 3), 1 / 3))


def test_multiclass_roc_auc_matches_sklearn_macro_ovr():
    generator = np.random.default_rng(1)
    labels = generator.integers(0, 4, size=80)
    probabilities = generator.dirichlet(np.ones(4), size=80)
    expected = roc_auc_score(labels, probabilities, multi_class="ovr", average="macro")
    assert multiclass_roc_auc(labels, probabilities) == pytest.approx(expected)


def test_multiclass_roc_auc_weights_rare_class_equally():
    labels = np.array([0] * 9 + [1] * 9 + [2] * 2)
    probabilities = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.5, 0.5, 0.0]])[labels]
    assert multiclass_roc_auc(labels, probabilities) == pytest.approx(2 / 3)


def test_multiclass_roc_auc_reads_logits_through_softmax():
    # Subject 0 has the lower class-0 logit but the higher class-0 probability, so skipping the softmax would
    # change the ranking.
    logits = np.array([[1.0, -5.0, -5.0], [2.0, 9.0, 0.0], [0.0, 0.0, 3.0], [-1.0, 4.0, 0.0]])
    labels = np.array([0, 1, 2, 1])
    probabilities = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    assert multiclass_roc_auc(labels, logits) == pytest.approx(multiclass_roc_auc(labels, probabilities))
    assert multiclass_roc_auc(labels, logits) == pytest.approx(1.0)


def test_multiclass_roc_auc_accepts_whole_number_float_labels():
    assert multiclass_roc_auc(np.array([0.0, 1.0, 2.0]), np.eye(3)) == pytest.approx(1.0)


@pytest.mark.parametrize("metric", [binary_metrics, roc_auc, multiclass_roc_auc, multiclass_metrics])
def test_classification_metrics_reject_complex_labels(metric):
    labels = np.array([0 + 1j, 1 + 0j])
    scores = np.eye(2) if metric in (multiclass_roc_auc, multiclass_metrics) else [0.1, 0.9]
    with pytest.raises(MetricError, match="real class indices"):
        metric(labels, scores)


@pytest.mark.parametrize(
    "labels, scores, match",
    [
        (np.array([0, 1, 0, 1]), np.full((4, 3), 1 / 3), r"\[2\] are absent"),
        (np.array([0, 1, 3]), np.full((3, 3), 1 / 3), r"class indices 0 to 2, got \[3\]"),
        (np.array([0.0, 0.5, 2.0]), np.full((3, 3), 1 / 3), r"got \[0.5\]"),
        (np.array([0.0, 0.999999, 2.0]), np.full((3, 3), 1 / 3), r"got \[0.999999\]"),
        (np.array([0.0, np.nan, 2.0]), np.full((3, 3), 1 / 3), r"got \[nan\]"),
        (np.array([0, 1, 2]), np.full(3, 1 / 3), "expected"),
        (np.array([0, 1, 1]), np.array([[0.2], [0.7], [0.9]]), "expected"),
        (np.array([0, 1]), np.zeros((3, 3)), "expected"),
    ],
)
def test_multiclass_roc_auc_rejects_invalid_inputs(labels, scores, match):
    with pytest.raises(MetricError, match=match):
        multiclass_roc_auc(labels, scores)


@pytest.fixture(scope="module")
def runs():
    generator = np.random.default_rng(0)
    return [(generator.integers(0, 2, 40), generator.normal(size=40)) for _ in range(5)]


def test_mean_roc_curve_shapes_and_endpoints(runs):
    summary = mean_roc_curve([labels for labels, _ in runs], [scores for _, scores in runs], grid_size=50)
    assert summary["fpr"].shape == summary["tpr_mean"].shape == summary["tpr_std"].shape == (50,)
    assert summary["tpr_runs"].shape == (5, 50)
    assert summary["tpr_mean"][0] == pytest.approx(0.0)
    assert summary["tpr_mean"][-1] == pytest.approx(1.0)
    assert np.all(np.diff(summary["tpr_mean"]) >= 0)


@pytest.mark.parametrize("grid_size", [0, 1, 2.0, True])
def test_mean_roc_curve_rejects_invalid_grid_size(runs, grid_size):
    with pytest.raises(ValueError, match="grid_size must be an integer of at least two"):
        mean_roc_curve([runs[0][0]], [runs[0][1]], grid_size=grid_size)


def test_mean_roc_curve_auc_matches_runs(runs):
    summary = mean_roc_curve([labels for labels, _ in runs], [scores for _, scores in runs])
    assert summary["auc_runs"] == pytest.approx([roc_auc(labels, scores) for labels, scores in runs])
    assert summary["auc_mean"] == pytest.approx(float(np.mean(summary["auc_runs"])))


def test_mean_roc_curve_starts_at_zero_when_top_scores_are_positive():
    # The ROC curve rises vertically at a false-positive rate of zero, so interpolation alone would start at one.
    summary = mean_roc_curve([np.array([0, 0, 1, 1])], [np.array([0.1, 0.2, 0.8, 0.9])])
    assert summary["tpr_runs"][0, 0] == 0.0
    assert summary["tpr_runs"][0, 1] == 1.0


@pytest.mark.parametrize(
    "labels_per_run, scores_per_run, match",
    [([], [], "at least one run"), ([np.array([0, 1])], [], "exactly one of each")],
)
def test_mean_roc_curve_rejects_invalid_runs(labels_per_run, scores_per_run, match):
    with pytest.raises(MetricError, match=match):
        mean_roc_curve(labels_per_run, scores_per_run)
