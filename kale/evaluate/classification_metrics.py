"""Summarize binary and multiclass predictions with ranking and threshold metrics.

``roc_auc`` and ``multiclass_roc_auc`` rank scores. ``sensitivity_specificity`` and ``binary_metrics`` add metrics at an
operating point, and ``multiclass_metrics`` summarizes a multiclass prediction. ``mean_roc_curve`` averages ROC curves
from several runs or folds on a shared false-positive-rate grid. Metrics that are undefined for the values given, such
as an area for a split holding one class, raise ``MetricError``.

Scores may be logits or probabilities, as ``score_type`` states. Binary ranking metrics are invariant to the sigmoid,
so only threshold metrics convert binary scores. A one-versus-rest ranking is not invariant to the softmax, so
multiclass logits pass through a softmax first. The default, ``score_type="auto"``, infers the type from the values:
binary scores that all lie in ``[0, 1]``, and multiclass rows that are all distributions, are read as probabilities.
Logits that happen to fall in that range are then misread, so pass ``"logit"`` or ``"probability"`` when the type is
known.

Adapted from KaleCardiac (https://github.com/pykale/cardiac), MIT License,
Copyright (c) 2026 PyKale.
"""

from collections.abc import Sequence
from contextlib import suppress

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    roc_curve,
)


class MetricError(ValueError):
    """Raised when a metric cannot be computed from the values given."""


_SCORE_TYPES = ("auto", "probability", "logit")


def _check_score_type(score_type: str) -> None:
    if score_type not in _SCORE_TYPES:
        raise ValueError(f"score_type must be one of {_SCORE_TYPES}, got {score_type!r}")


def _reads_as_probabilities(score_type: str, looks_like_probabilities: bool, mismatch: str) -> bool:
    """Decide whether scores are probabilities, raising ``MetricError(mismatch)`` if ``"probability"`` is stated for
    scores that are not."""
    _check_score_type(score_type)
    if score_type == "probability" and not looks_like_probabilities:
        raise MetricError(mismatch)
    return {"probability": True, "logit": False, "auto": looks_like_probabilities}[score_type]


def _in_unit_interval(scores: np.ndarray) -> bool:
    return bool(scores.size) and bool(np.all((scores >= 0.0) & (scores <= 1.0)))


def _class_indices(labels: ArrayLike, num_classes: int) -> np.ndarray:
    """Return labels as integer class indices, rejecting any value that is not exactly one rather than casting it."""
    labels = np.asarray(labels).reshape(-1)
    if np.iscomplexobj(labels):
        raise MetricError("labels must be real class indices")
    invalid = labels[~np.isin(labels.astype(np.float64), np.arange(num_classes))]
    if invalid.size:
        raise MetricError(f"labels must be class indices 0 to {num_classes - 1}, got {np.unique(invalid).tolist()}")
    return labels.astype(int)


def _binary_labels(labels: ArrayLike) -> np.ndarray:
    labels = _class_indices(labels, 2)
    present = np.unique(labels)
    if present.size < 2:
        raise MetricError(
            f"a binary metric needs both classes, but every sample has label {present.tolist()}; this usually means "
            f"the split is too small or a filter removed one class"
        )
    return labels


def _binary_probabilities(scores: ArrayLike, score_type: str = "auto") -> np.ndarray:
    """Read scores as probabilities, applying a sigmoid to logits, or under ``"auto"`` unless all lie in ``[0, 1]``."""
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if not np.isfinite(scores).all():
        raise MetricError("scores must be finite")
    mismatch = "score_type='probability' needs every score in [0, 1]; pass 'logit' for raw model outputs"
    if _reads_as_probabilities(score_type, _in_unit_interval(scores), mismatch):
        return scores
    return 1.0 / (1.0 + np.exp(-scores))


def _binary_confusion(
    labels: np.ndarray, scores: ArrayLike, threshold: float, score_type: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return the predicted labels and the ``(tn, fp, fn, tp)`` counts at a probability threshold."""
    if not np.isfinite(threshold):
        raise MetricError("threshold must be finite")
    predicted = (_binary_probabilities(scores, score_type) >= threshold).astype(int)
    return predicted, confusion_matrix(labels, predicted, labels=[0, 1]).ravel()


def _rates(true_negative: int, false_positive: int, false_negative: int, true_positive: int) -> tuple[float, float]:
    sensitivity = true_positive / max(true_positive + false_negative, 1)
    specificity = true_negative / max(true_negative + false_positive, 1)
    return float(sensitivity), float(specificity)


def roc_auc(labels: ArrayLike, scores: ArrayLike) -> float:
    """Compute the area under the binary ROC curve.

    Args:
        labels (ArrayLike): Targets of shape ``(N,)``, 1 for positive and 0 for negative.
        scores (ArrayLike): Scores of shape ``(N,)``, higher meaning more likely positive.

    Returns:
        float: The area, 0.5 for a random ranking and 1.0 for a perfect one.

    Raises:
        MetricError: If a label is not 0 or 1, or only one class is present.

    Example:
        .. code-block:: pycon

            >>> roc_auc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
            0.75
    """
    return float(roc_auc_score(_binary_labels(labels), np.asarray(scores).reshape(-1)))


def sensitivity_specificity(
    labels: ArrayLike, scores: ArrayLike, threshold: float = 0.5, score_type: str = "auto"
) -> tuple[float, float]:
    """Compute sensitivity and specificity at an operating point.

    Args:
        labels (ArrayLike): Targets of shape ``(N,)``, 1 for positive and 0 for negative.
        scores (ArrayLike): Logits or probabilities of shape ``(N,)``.
        threshold (float, optional): Probability at or above which a sample is called positive. Defaults to 0.5.
        score_type (str, optional): ``"logit"``, ``"probability"`` or ``"auto"``, which reads scores that all lie in
            ``[0, 1]`` as probabilities. Defaults to ``"auto"``.

    Returns:
        tuple[float, float]: ``(sensitivity, specificity)``, each in ``[0, 1]``.

    Raises:
        MetricError: If a label is not 0 or 1, only one class is present, a score or threshold is not finite, or
            ``score_type`` is ``"probability"`` and a score lies outside ``[0, 1]``.
        ValueError: If ``score_type`` is not recognized.
    """
    labels = _binary_labels(labels)
    _, counts = _binary_confusion(labels, scores, threshold, score_type)
    return _rates(*counts)


def binary_metrics(labels: ArrayLike, scores: ArrayLike, threshold: float = 0.5, score_type: str = "auto") -> dict:
    """Summarize a binary prediction with ranking and threshold metrics.

    ``roc_auc`` and ``average_precision`` rank the scores and ignore ``threshold``. The other metrics call a sample
    positive when its probability is at least ``threshold``, so a probability of exactly 0.5 counts as positive. An
    argmax over two probability columns, as in :func:`multiclass_metrics` and scikit-learn's ``predict``, breaks that
    tie toward the first class instead; the two agree on every other probability.

    Args:
        labels (ArrayLike): Targets of shape ``(N,)``, 1 for positive and 0 for negative.
        scores (ArrayLike): Logits or probabilities of shape ``(N,)``.
        threshold (float, optional): Probability at or above which a sample is called positive. Defaults to 0.5.
        score_type (str, optional): ``"logit"``, ``"probability"`` or ``"auto"``, which reads scores that all lie in
            ``[0, 1]`` as probabilities. Defaults to ``"auto"``.

    Returns:
        dict: JSON-serializable ``n``, ``n_positive``, ``positive_rate``, ``threshold``, ``roc_auc``,
        ``average_precision``, ``accuracy``, ``balanced_accuracy``, ``sensitivity``, ``specificity``, ``f1``, ``mcc``
        and ``confusion``, a mapping of ``tn``, ``fp``, ``fn`` and ``tp`` counts.

    Raises:
        MetricError: If a label is not 0 or 1, only one class is present, a score or threshold is not finite, or
            ``score_type`` is ``"probability"`` and a score lies outside ``[0, 1]``.
        ValueError: If ``score_type`` is not recognized.

    Example:
        .. code-block:: pycon

            >>> summary = binary_metrics([0, 0, 1, 1], [-2.0, 0.5, 1.0, 3.0], score_type="logit")
            >>> summary["sensitivity"], summary["specificity"]
            (1.0, 0.5)
    """
    labels = _binary_labels(labels)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    predicted, counts = _binary_confusion(labels, scores, threshold, score_type)
    true_negative, false_positive, false_negative, true_positive = counts
    sensitivity, specificity = _rates(*counts)

    return {
        "n": int(labels.size),
        "n_positive": int(labels.sum()),
        "positive_rate": float(labels.mean()),
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "accuracy": float(accuracy_score(labels, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, predicted)),
        "confusion": {
            "tn": int(true_negative),
            "fp": int(false_positive),
            "fn": int(false_negative),
            "tp": int(true_positive),
        },
    }


def _class_probabilities(scores: np.ndarray, score_type: str) -> np.ndarray:
    """Read ``(N, K)`` scores as probabilities, applying a softmax to logits, or under ``"auto"`` unless every row is
    already a distribution."""
    distributions = _in_unit_interval(scores) and np.allclose(scores.sum(axis=1), 1.0, atol=1e-4)
    mismatch = "score_type='probability' needs every row to be a distribution; pass 'logit' for raw outputs"
    if _reads_as_probabilities(score_type, distributions, mismatch):
        # Renormalize because a float32 softmax can sum to one less tightly than scikit-learn requires.
        return scores / scores.sum(axis=1, keepdims=True)
    exponentials = np.exp(scores - scores.max(axis=1, keepdims=True))
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def _score_matrix(scores: ArrayLike, num_samples: int) -> np.ndarray:
    """Return ``(num_samples, num_classes)`` scores with at least two classes."""
    scores = np.asarray(scores, dtype=np.float64)
    expected = f"expected ({num_samples}, num_classes) scores, got {scores.shape}"
    if scores.ndim != 2:
        raise MetricError(expected)
    if scores.shape[0] != num_samples or scores.shape[1] < 2:
        raise MetricError(expected)
    if not np.isfinite(scores).all():
        raise MetricError("scores must be finite")
    return scores


def _multiclass_inputs(labels: ArrayLike, scores: ArrayLike, score_type: str) -> tuple[np.ndarray, np.ndarray]:
    """Validate a multiclass prediction and return ``(labels, probabilities)``."""
    labels = np.asarray(labels).reshape(-1)
    probabilities = _class_probabilities(_score_matrix(scores, labels.size), score_type)
    return _class_indices(labels, probabilities.shape[1]), probabilities


def multiclass_roc_auc(labels: ArrayLike, scores: ArrayLike, score_type: str = "auto") -> float:
    """Compute the macro one-versus-rest area under the ROC curve.

    Each class is ranked against all the others and the areas are averaged with equal weight, so a rare class counts
    as much as a common one.

    Args:
        labels (ArrayLike): Class indices of shape ``(N,)``. Whole-number floats are accepted.
        scores (ArrayLike): Probabilities or logits of shape ``(N, num_classes)``.
        score_type (str, optional): ``"logit"``, ``"probability"`` or ``"auto"``, which reads rows that are all
            distributions over the classes as probabilities. Defaults to ``"auto"``.

    Returns:
        float: The macro area, 0.5 for a random ranking and 1.0 for a perfect one.

    Raises:
        MetricError: If the shapes disagree, a label is not a valid class index, a score is not finite, a class is
            absent, which leaves its one-versus-rest curve undefined, or ``score_type`` is ``"probability"`` and a
            row is not a distribution.
        ValueError: If ``score_type`` is not recognized.
    """
    labels, probabilities = _multiclass_inputs(labels, scores, score_type)
    num_classes = probabilities.shape[1]
    absent = sorted(set(range(num_classes)) - set(np.unique(labels).tolist()))
    if absent:
        raise MetricError(
            f"class(es) {absent} are absent from these labels, so their one-versus-rest curve is undefined; the split "
            f"is too small or too imbalanced to score every class"
        )
    if num_classes == 2:
        # scikit-learn reads two classes as a binary problem and rejects two score columns. Each class's
        # one-versus-rest area then equals the binary area of the second column, so their mean does too.
        return roc_auc(labels, probabilities[:, 1])
    return float(roc_auc_score(labels, probabilities, multi_class="ovr", average="macro"))


def multiclass_metrics(
    labels: ArrayLike, scores: ArrayLike, class_names: Sequence[str] | None = None, score_type: str = "auto"
) -> dict:
    """Summarize a multiclass prediction.

    Args:
        labels (ArrayLike): Class indices of shape ``(N,)``. Whole-number floats are accepted.
        scores (ArrayLike): Probabilities or logits of shape ``(N, num_classes)``.
        class_names (Sequence[str], optional): One name per score column for the confusion-matrix axes. Defaults to the
            class indices.
        score_type (str, optional): ``"logit"``, ``"probability"`` or ``"auto"``, which reads rows that are all
            distributions over the classes as probabilities. Defaults to ``"auto"``.

    Returns:
        dict: JSON-serializable ``n``, ``accuracy``, ``balanced_accuracy``, ``f1_macro``, ``confusion`` as a nested
        list, ``classes``, and ``roc_auc_ovr`` when every class is present.

    Raises:
        MetricError: If the shapes disagree, a label is not a valid class index, a score is not finite, fewer than
            two classes are present, the number of class names differs from the score columns, or ``score_type`` is
            ``"probability"`` and a row is not a distribution.
        ValueError: If ``score_type`` is not recognized.
    """
    labels, probabilities = _multiclass_inputs(labels, scores, score_type)
    if np.unique(labels).size < 2:
        raise MetricError("a multiclass metric needs at least two classes present")

    num_classes = probabilities.shape[1]
    names = [str(index) for index in range(num_classes)] if class_names is None else list(class_names)
    if len(names) != num_classes:
        raise MetricError(f"expected {num_classes} class names, got {len(names)}")
    predicted = probabilities.argmax(axis=1)
    summary = {
        "n": int(labels.size),
        "accuracy": float(accuracy_score(labels, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "f1_macro": float(f1_score(labels, predicted, average="macro", zero_division=0)),
        "confusion": confusion_matrix(labels, predicted, labels=list(range(num_classes))).tolist(),
        "classes": names,
    }
    with suppress(MetricError):
        summary["roc_auc_ovr"] = multiclass_roc_auc(labels, probabilities, score_type="probability")
    return summary


def mean_roc_curve(
    labels_per_run: Sequence[ArrayLike], scores_per_run: Sequence[ArrayLike], grid_size: int = 100
) -> dict:
    """Average the ROC curves of several runs on a shared false-positive-rate grid.

    A run is one evaluation of the same model setup, such as a cross-validation fold or a repeat with a different
    seed; it is not a training epoch. Each run's ROC curve is interpolated onto the grid, and the true-positive rate is
    averaged at each grid point (vertical averaging), so every run counts equally regardless of its size. This differs
    from pooling all predictions into one curve, which weights samples equally.

    Args:
        labels_per_run (Sequence[ArrayLike]): One ``(N,)`` label array per run.
        scores_per_run (Sequence[ArrayLike]): One ``(N,)`` score array per run, aligned with the labels.
        grid_size (int, optional): Number of false-positive-rate points, at least two. Defaults to 100.

    Returns:
        dict: ``fpr`` (the grid), ``tpr_mean`` and ``tpr_std`` (mean and standard deviation of the true-positive rate
        at each grid point), ``tpr_runs`` (each run's interpolated curve), and ``auc_runs``, ``auc_mean`` and
        ``auc_std`` (each run's AUROC and their mean and standard deviation; ``auc_mean`` is not the area under the
        mean curve).

    Raises:
        MetricError: If no run is given, the two sequences differ in length, or a run holds one class.
        ValueError: If ``grid_size`` is not an integer of at least two.
    """
    if not labels_per_run:
        raise MetricError("mean_roc_curve needs at least one run")
    if len(labels_per_run) != len(scores_per_run):
        raise MetricError(
            f"got {len(labels_per_run)} label arrays but {len(scores_per_run)} score arrays; each run contributes "
            f"exactly one of each"
        )

    if isinstance(grid_size, (bool, np.bool_)) or not isinstance(grid_size, (int, np.integer)) or grid_size < 2:
        raise ValueError("grid_size must be an integer of at least two")
    grid = np.linspace(0.0, 1.0, grid_size)
    curves, areas = [], []
    for labels, scores in zip(labels_per_run, scores_per_run):
        labels = _binary_labels(labels)
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        false_positive, true_positive, _ = roc_curve(labels, scores)
        interpolated = np.interp(grid, false_positive, true_positive)
        # Interpolation does not guarantee the (0, 0) endpoint that every ROC curve has.
        interpolated[0] = 0.0
        curves.append(interpolated)
        areas.append(float(roc_auc_score(labels, scores)))

    stacked = np.vstack(curves)
    mean = stacked.mean(axis=0)
    mean[-1] = 1.0

    return {
        "fpr": grid,
        "tpr_mean": mean,
        "tpr_std": stacked.std(axis=0),
        "tpr_runs": stacked,
        "auc_mean": float(np.mean(areas)),
        "auc_std": float(np.std(areas)),
        "auc_runs": areas,
    }
