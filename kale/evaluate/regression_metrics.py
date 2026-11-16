"""Summarize continuous predictions in the target's units.

``regression_metrics`` reports root mean squared error, mean absolute error, signed bias and the coefficient of
determination together. The errors and R-squared are scikit-learn's; R-squared is reported as None, rather than a
number, when the targets have no variation to explain.

Adapted from KaleCardiac (https://github.com/pykale/cardiac), MIT License,
Copyright (c) 2026 PyKale.
"""

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from kale.evaluate.classification_metrics import MetricError


def _paired(targets: ArrayLike, predictions: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    targets = np.asarray(targets, dtype=np.float64).reshape(-1)
    predictions = np.asarray(predictions, dtype=np.float64).reshape(-1)
    if targets.size != predictions.size:
        raise MetricError(f"got {targets.size} targets and {predictions.size} predictions")
    if targets.size == 0:
        raise MetricError("a regression metric needs at least one sample")
    return targets, predictions


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
    return {
        "n": int(targets.size),
        "rmse": float(np.sqrt(mean_squared_error(targets, predictions))),
        "mae": float(mean_absolute_error(targets, predictions)),
        "bias": float(np.mean(predictions - targets)),
        "unit": unit,
        # Undefined when every target is the same value. Test that directly: scikit-learn reports 0 or 1 there, and
        # rounding in the mean can leave a tiny positive total sum of squares for a constant such as 0.1.
        "r2": float(r2_score(targets, predictions)) if np.ptp(targets) > 0 else None,
    }
