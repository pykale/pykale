"""Compute a percentile bootstrap interval for a metric by resampling samples.

``bootstrap_ci`` resamples one dataset with replacement and recomputes a metric, so it applies to any metric in
:mod:`kale.evaluate` and to paired comparisons of two models scored on the same samples. Adapted from
https://github.com/pykale/cardiac.
"""

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike

from kale.evaluate.classification_metrics import MetricError


def bootstrap_ci(
    metric: Callable[..., float],
    *arrays: ArrayLike,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
    undefined_errors: tuple[type[Exception], ...] = (MetricError,),
) -> dict:
    """Compute a percentile bootstrap interval for a metric by resampling samples.

    Each resample draws samples with replacement, applies the same indices to every array, and recomputes the metric.
    The interval describes how the metric would vary on another dataset of the same size; it does not account for
    refitting the model.

    The metric is first computed on all samples, so an error there, such as a mismatched input or a bug in the metric,
    propagates before any resampling. A resample on which the metric then raises one of ``undefined_errors`` is
    treated as undefined, for example when it holds one class, and is skipped rather than counted; ``n_used`` reports
    how many were kept. Any other error propagates. Metrics in :mod:`kale.evaluate` raise ``MetricError`` when
    undefined. A scikit-learn metric raises a plain ``ValueError`` instead, so pass ``undefined_errors=(ValueError,)``
    to use one directly.

    To compare two models, pass both models' scores and a metric of their difference, such as
    ``lambda labels, scores, baseline: roc_auc(labels, scores) - roc_auc(labels, baseline)``. Both are then scored on
    the same resampled samples, so the interval is of the paired difference. Two separate intervals can overlap even
    when one model is consistently better.

    Args:
        metric (Callable[..., float]): Takes the resampled arrays in order and returns a float, such as
            :func:`kale.evaluate.classification_metrics.roc_auc`.
        *arrays (ArrayLike): Arrays indexed by sample along the first axis, such as labels and scores.
        n_resamples (int, optional): Number of resamples to draw. Defaults to 1000.
        confidence (float, optional): Interval coverage in ``(0, 1)``. Defaults to 0.95.
        seed (int, optional): Seed for the resampling. Defaults to 0.
        undefined_errors (tuple[type[Exception], ...], optional): Exceptions that mark a resample as undefined.
            Defaults to ``(MetricError,)``.

    Returns:
        dict: ``point`` (the metric on all samples), ``low``, ``high``, ``confidence`` and ``n_used``.

    Raises:
        MetricError: If the arrays differ in length, hold no sample, or no resample could be scored.
        ValueError: If no array is given, ``n_resamples`` is not positive, or ``confidence`` is outside ``(0, 1)``.

    Example:
        .. code-block:: pycon

            >>> from kale.evaluate.classification_metrics import roc_auc
            >>> labels = [0, 0, 0, 0, 1, 1, 1, 1]
            >>> scores = [0.1, 0.3, 0.2, 0.6, 0.4, 0.7, 0.8, 0.9]
            >>> interval = bootstrap_ci(roc_auc, labels, scores, n_resamples=200)
            >>> interval["point"]
            0.9375
    """
    if not arrays:
        raise ValueError("bootstrap_ci needs at least one array to resample")
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be positive, got {n_resamples}")
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must lie in (0, 1), got {confidence}")
    arrays = tuple(np.asarray(array) for array in arrays)
    lengths = [len(array) for array in arrays]
    if len(set(lengths)) != 1:
        raise MetricError(f"every array must have the same number of samples, got {lengths}")
    if lengths[0] == 0:
        raise MetricError("bootstrap_ci needs at least one sample")

    point = float(metric(*arrays))
    generator = np.random.default_rng(seed)
    values = []
    for _ in range(n_resamples):
        draw = generator.integers(0, lengths[0], size=lengths[0])
        try:
            values.append(float(metric(*(array[draw] for array in arrays))))
        except undefined_errors:
            continue

    if not values:
        raise MetricError(
            "no bootstrap resample could be scored; with this few samples, or this severe an imbalance, almost every "
            "resample holds one class"
        )

    tail = (1.0 - confidence) / 2.0
    return {
        "point": point,
        "low": float(np.percentile(values, 100 * tail)),
        "high": float(np.percentile(values, 100 * (1.0 - tail))),
        "confidence": float(confidence),
        "n_used": len(values),
    }
