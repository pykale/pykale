"""
Authors: Lawrence Schobs, lawrenceschobs@gmail.com

Functions related to similarity metrics including similarity measures and correlations.
"""

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from kale.interpret.uncertainty_utils import analyze_and_plot_uncertainty_correlation
from kale.prepdata.tabular_transform import apply_confidence_inversion


@dataclass
class CorrelationConfig:
    """
    Configuration for :func:`evaluate_correlations`.

    Groups the analysis, plotting and output settings so that callers pass one object instead of a long
    parameter list, and so that the binning and file-naming constants are named rather than embedded in
    the analysis code.
    """

    # Analysis settings
    num_bins: int = 10  # Number of quantile bins used to derive the uncertainty thresholds
    num_folds: int = 8  # Number of testing folds to include
    error_scaling_factor: float = 1.0  # Multiplicative factor for error values (e.g. pixel to mm conversion)
    combine_middle_bins: bool = False  # If True, keep only the outer thresholds so middle bins merge into one

    # Plotting and output
    colormap: str = "Set1"  # Matplotlib colormap name for the correlation plots
    save_path: Optional[str] = None  # Directory to save plots into; if None, no figure is written
    to_log: bool = False  # If True, use logarithmic scaling on the plot axes
    file_name_template: str = "{model}_{uncertainty}_correlation_pwr_all_targets.pdf"  # Output file name pattern


def _fold_rows(data_structs: pd.DataFrame, num_folds: int) -> pd.DataFrame:
    """Select the rows belonging to the first ``num_folds`` testing folds.

    Args:
        data_structs (pd.DataFrame): Predictions for a single model.
        num_folds (int): Number of testing folds to include.

    Returns:
        pd.DataFrame: The rows for those folds.
    """
    return data_structs[data_structs["Testing Fold"].isin(np.arange(num_folds))]


def _fold_errors_and_uncertainties(
    data_structs: pd.DataFrame,
    uncertainty_pair: Tuple[str, str, str],
    invert_uncertainty: bool,
    num_folds: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract the errors and uncertainties for one uncertainty type across the testing folds.

    Args:
        data_structs (pd.DataFrame): Predictions for a single model.
        uncertainty_pair (tuple): The (name, error key, uncertainty key) triple being analysed.
        invert_uncertainty (bool): Whether the uncertainty values should be inverted.
        num_folds (int): Number of testing folds to include.

    Returns:
        tuple: Flattened error values and (optionally inverted) uncertainty values.
    """
    uncertainty_type = uncertainty_pair[0]
    rows = _fold_rows(data_structs, num_folds)

    errors = np.array(rows[[uncertainty_type + " Error"]].values.tolist()).flatten()

    uncertainties = rows[[uncertainty_type + " Uncertainty"]]
    if invert_uncertainty:
        uncertainties = apply_confidence_inversion(uncertainties, uncertainty_pair[2])

    return errors, np.array(uncertainties.values.tolist()).flatten()


def _quantile_thresholds(uncertainty_values: np.ndarray, num_bins: int, combine_middle_bins: bool) -> List[float]:
    """Compute the uncertainty values that separate the quantile bins.

    Args:
        uncertainty_values (np.ndarray): Uncertainty values across all folds.
        num_bins (int): Number of quantile bins.
        combine_middle_bins (bool): If True, keep only the outer thresholds, so the middle bins merge
            into a single bin and only the two edge bins remain distinct.

    Returns:
        list: The threshold values, ordered from lowest to highest uncertainty.
    """
    sorted_uncertainties = np.sort(uncertainty_values)

    quantiles = np.arange(1 / num_bins, 1, 1 / num_bins)[: num_bins - 1]
    thresholds = [np.quantile(sorted_uncertainties, quantile) for quantile in quantiles]

    if combine_middle_bins:
        thresholds = [thresholds[0], thresholds[-1]]

    return thresholds


def evaluate_correlations(
    bin_predictions: Dict[str, pd.DataFrame],
    uncertainty_error_pairs: List[Tuple[str, str, str]],
    confidence_invert_tuples: List[Tuple[str, bool]],
    config: Optional[CorrelationConfig] = None,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Calculates the correlation between error and uncertainty for each bin and for each target,
    using a piece-wise linear regression model.

    Designed for use in Quantile Binning (/pykale/examples/landmark_uncertainty/main.py).

    Args:
        bin_predictions: A dictionary of Pandas DataFrames containing model predictions for each testing fold.
        uncertainty_error_pairs: A list of tuples specifying the names of the uncertainty,
            error, and uncertainty inversion keys for each pair.
        confidence_invert_tuples: A list of tuples specifying whether to invert the uncertainty values for each method.
                          First element is a string specifying the uncertainty method name and the second element is
                          a boolean whether to invert e.g. [["E-MHA", True], ["E-CPV", False]]
        config: Analysis, plotting and output settings. Defaults to :class:`CorrelationConfig`.

    Returns:
        A dictionary containing the correlation statistics for each model and uncertainty method.
        The dictionary has the following structure:
        {
            <model_name>: {
                <uncertainty_name>: {
                    "all_folds": {
                        "r": <correlation coefficient>,
                        "p": <p-value>,
                        "fit_params": <regression line parameters>,
                        "ci": <confidence intervals for the regression line parameters>
                    },
                    "quantiles": {
                        <quantile_index>: {
                            "r": <correlation coefficient>,
                            "p": <p-value>,
                            "fit_params": <regression line parameters>,
                            "ci": <confidence intervals for the regression line parameters>
                        }
                    }
                }
            }
        }
        The "all_folds" key contains the correlation statistics for all testing folds combined.
        The "quantiles" key contains the correlation statistics for each quantile bin separately.
    """
    config = config or CorrelationConfig()

    logger = logging.getLogger("qbin")
    # define dict to save correlations to.
    correlation_dict: Dict[str, Dict[str, Dict[str, Any]]] = {}

    invert_by_type = dict(confidence_invert_tuples)

    # Loop over models (model) and uncertainty methods (uncert_pair)
    for model, data_structs in bin_predictions.items():
        correlation_dict[model] = {}
        for uncert_pair in uncertainty_error_pairs:  # uncert_pair = [pair name, error name , uncertainty name]
            uncertainty_type = uncert_pair[0]
            logger.info("All folds correlation for Model: %s, Uncertainty type %s :", uncertainty_type, model)

            fold_errors, fold_uncertainty_values = _fold_errors_and_uncertainties(
                data_structs, uncert_pair, invert_by_type[uncertainty_type], config.num_folds
            )

            # Get the quantiles of the data w.r.t to the uncertainty values.
            quantile_thresholds = _quantile_thresholds(
                fold_uncertainty_values, config.num_bins, config.combine_middle_bins
            )

            # fit a piece-wise linear regression line between quantiles, and return correlation values.
            save_path_fig = (
                os.path.join(
                    config.save_path,
                    config.file_name_template.format(model=model, uncertainty=uncertainty_type),
                )
                if config.save_path
                else None
            )
            corr_dict = analyze_and_plot_uncertainty_correlation(
                fold_errors,
                fold_uncertainty_values,
                quantile_thresholds,
                colormap=config.colormap,
                error_scaling_factor=config.error_scaling_factor,
                save_path=save_path_fig,
                to_log=config.to_log,
            )

            correlation_dict[model][uncertainty_type] = corr_dict

    return correlation_dict
