"""
Authors: Lawrence Schobs, lawrenceschobs@gmail.com

Functions related to similarity metrics including similarity measures and correlations.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from kale.prepdata.tabular_transform import apply_confidence_inversion


def jaccard_similarity(list1: list, list2: list) -> float:
    """
    Calculates the Jaccard Index (JI) between two lists.
    Args:
        list1 (list): List of elements in set A.
        list2 (list): List of elements in set B.
    Returns:
        float: JI between list1 and list2.
    Raises:
        None.
    Example:
        >>> jaccard_similarity([1,2,3], [2,3,4])
        0.5
    """

    if len(list1) == 0 or len(list2) == 0:
        return 0
    else:
        intersection = len(set(list1).intersection(list2))  # no need to call list here
        union = len(list1 + list2) - intersection  # you only need to call len once here
        return intersection / union  # also no need to cast to float as this will be done for you


def evaluate_correlations(
    bin_predictions: Dict[str, pd.DataFrame],
    uncertainty_error_pairs: List[Tuple[str, str, str]],
    cmaps: List[Dict[Any, Any]],
    num_bins: int,
    confidence_invert_tuples: List[Tuple[str, bool]],
    num_folds: int = 8,
    error_scaling_factor: float = 1,
    combine_middle_bins: bool = False,
    save_path: Optional[str] = None,
    to_log: bool = False,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Calculates the correlation between error and uncertainty for each bin and for each target,
    using a piece-wise linear regression model.

    Designed for use in Quantile Binning (/pykale/examples/landmark_uncertainty/main.py).

    Args:
        bin_predictions: A dictionary of Pandas DataFrames containing model predictions for each testing fold.
        uncertainty_error_pairs: A list of tuples specifying the names of the uncertainty,
            error, and uncertainty inversion keys for each pair.
        cmaps: A dictionary of colour maps to use for plotting the results.
        num_bins: The number of quantile bins to divide the data into.
        confidence_invert_tuples: A list of tuples specifying whether to invert the uncertainty values for each method..
                          First element is a string specifying the uncertainty method name and the second element is a boolean
                          whether to invert e.g. [["E-MHA", True], ["E-CPV", False]]
        num_folds: The number of folds to use for cross-validation (default: 8).
        error_scaling_factor: The scale factor to transform error by (default: 1).
        combine_middle_bins: Whether to combine the middle bins into one bin (default: False).
        save_path: The path to save the correlation plots (default: None).
        to_log: Whether to use logarithmic scaling for the x and y axes of the plots (default: False).
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
    from kale.interpret.uncertainty_quantiles import fit_line_with_ci

    logger = logging.getLogger("qbin")
    # define dict to save correlations to.
    correlation_dict: Dict[str, Dict[str, Dict[str, Any]]] = {}

    num_bins_quantiles = num_bins
    # If we are combining the middle bins, we only have the 2 edge bins and the middle bins are combined into 1 bin.
    if combine_middle_bins:
        num_bins = 3

    # Loop over models (model) and uncertainty methods (uncert_pair)
    for _, (model, data_structs) in enumerate(bin_predictions.items()):
        correlation_dict[model] = {}
        for uncert_pair in uncertainty_error_pairs:  # uncert_pair = [pair name, error name , uncertainty name]
            uncertainty_type = uncert_pair[0]
            logger.info("All folds correlation for Model: %s, Uncertainty type %s :", uncertainty_type, model)
            fold_errors = np.array(
                data_structs[(data_structs["Testing Fold"].isin(np.arange(num_folds)))][
                    [uncertainty_type + " Error"]
                ].values.tolist()
            ).flatten()

            fold_uncertainty_values = data_structs[(data_structs["Testing Fold"].isin(np.arange(num_folds)))][
                [uncertainty_type + " Uncertainty"]
            ]

            invert_uncert_bool = [x[1] for x in confidence_invert_tuples if x[0] == uncertainty_type][0]
            if invert_uncert_bool:
                fold_uncertainty_values = apply_confidence_inversion(fold_uncertainty_values, uncert_pair[2])

            fold_uncertainty_values = np.array(fold_uncertainty_values.values.tolist()).flatten()

            # Get the quantiles of the data w.r.t to the uncertainty values.
            # Get the true uncertianty quantiles
            sorted_uncertainties = np.sort(fold_uncertainty_values)

            quantiles = np.arange(1 / num_bins_quantiles, 1, 1 / num_bins_quantiles)[: num_bins_quantiles - 1]
            quantile_thresholds = [np.quantile(sorted_uncertainties, q) for q in quantiles]

            # If we are combining the middle bins, combine the middle lists into 1 list.
            if combine_middle_bins:
                quantile_thresholds = [quantile_thresholds[0], quantile_thresholds[-1]]

            # fit a piece-wise linear regression line between quantiles, and return correlation values.
            save_path_fig = (
                os.path.join(save_path, model + "_" + uncertainty_type + "_correlation_pwr_all_targets.pdf")
                if save_path
                else None
            )
            corr_dict = fit_line_with_ci(
                fold_errors,
                fold_uncertainty_values,
                quantile_thresholds,
                cmaps,
                error_scaling_factor=error_scaling_factor,
                save_path=save_path_fig,
                to_log=to_log,
            )

            correlation_dict[model][uncertainty_type] = corr_dict

    return correlation_dict
