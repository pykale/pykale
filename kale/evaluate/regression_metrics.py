"""Summarize continuous predictions in the target's units.

``regression_metrics`` reports root mean squared error, mean absolute error, signed bias and the coefficient of
determination together. ``r2_score`` raises ``MetricError`` when the targets have no variation to explain.

Adapted from https://github.com/pykale/cardiac.
"""

import numpy as np
from numpy.typing import ArrayLike

from kale.evaluate.classification_metrics import MetricError


def _paired(targets: ArrayLike, predictions: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    targets = np.asarray(targets, dtype=np.float64).reshape(-1)
    predictions = np.asarray(predictions, dtype=np.float64).reshape(-1)
    if targets.size != predictions.size:
        raise MetricError(f"got {targets.size} targets and {predictions.size} predictions")
    if targets.size == 0:
        raise MetricError("a regression metric needs at least one sample")
    return targets, predictions


def r2_score(targets: ArrayLike, predictions: ArrayLike) -> float:
    """Compute the coefficient of determination.

    A value of 0 means no better than predicting the mean target, and a negative value means worse than that.

    Args:
        targets (ArrayLike): Measured values of shape ``(N,)``.
        predictions (ArrayLike): Predicted values of shape ``(N,)``.

    Returns:
        float: One minus the ratio of the residual to the total sum of squares.

    Raises:
        MetricError: If the arrays differ in length, are empty, or every target has the same value.
    """
    targets, predictions = _paired(targets, predictions)
    # Test for equal targets directly: rounding in the mean leaves a tiny positive total for a constant such as 0.1.
    if np.ptp(targets) == 0.0:
        raise MetricError(
            "every target has the same value, so there is no variation for R-squared to explain; report the root "
            "mean squared error instead"
        )
    total = float(((targets - targets.mean()) ** 2).sum())
    residual = float(((targets - predictions) ** 2).sum())
    return 1.0 - residual / total


def regression_metrics(targets: ArrayLike, predictions: ArrayLike, unit: str | None = None) -> dict:
    """Summarize a continuous prediction.

    Args:
        targets (ArrayLike): Measured values of shape ``(N,)``.
        predictions (ArrayLike): Predicted values of shape ``(N,)``.
        unit (str, optional): A display label for the target's unit, such as ``"mmHg"``. Nothing is converted.
            Defaults to None.

    Returns:
        dict: JSON-serializable ``n``, ``rmse``, ``mae``, ``bias`` (mean of prediction minus target), ``unit`` and
        ``r2``, which is None when every target has the same value.

    Raises:
        MetricError: If the arrays differ in length or are empty.

    Example:
        .. code-block:: pycon

            >>> summary = regression_metrics([10.0, 20.0, 30.0], [12.0, 22.0, 32.0], unit="mmHg")
            >>> summary["bias"], summary["r2"]
            (2.0, 0.94)
    """
    targets, predictions = _paired(targets, predictions)
    residual = predictions - targets

    summary = {
        "n": int(targets.size),
        "rmse": float(np.sqrt((residual**2).mean())),
        "mae": float(np.abs(residual).mean()),
        "bias": float(residual.mean()),
        "unit": unit,
    }
    try:
        summary["r2"] = r2_score(targets, predictions)
    except MetricError:
        summary["r2"] = None
    return summary
