"""
Authors: Lawrence Schobs, lawrenceschobs@gmail.com
Module from the implementation of L. A. Schobs, A. J. Swift and H. Lu, "Uncertainty Estimation for Heatmap-Based Landmark Localization,"
in IEEE Transactions on Medical Imaging, vol. 42, no. 4, pp. 1021-1034, April 2023, doi: 10.1109/TMI.2022.3222730.

Functions to predict uncertainty quantiles from the quantile binning method. Includes:
   A) Binning Predictions: quantile_binning_predictions
   """
from typing import Dict, List, Optional, Union

import numpy as np


def quantile_binning_predictions(
    uncertainties_test: Dict[str, Union[int, float]],
    uncert_thresh: List[List[float]],
    save_pred_path: Optional[str] = None,
) -> Dict[str, int]:
    """
    Bin predictions based on quantile thresholds.

    Args:
        uncertainties_test (Dict): A dictionary of uncertainties with string ids and float/int uncertainty values.
        uncert_thresh (List[List[float]]): A list of quantile thresholds to determine binning.
        save_pred_path (str, optional): A path preamble to save predicted bins to.

    Returns:
        Dict: A dictionary of predicted quantile bins with string ids as keys and integer bin values as values.
    """
    # Test if dictionary correct structure
    if not isinstance(uncertainties_test, dict):
        raise ValueError("uncertainties_test must be of type dict")
    else:
        for i, (key, fc) in enumerate(uncertainties_test.items()):
            if not isinstance(key, str) or (not isinstance(fc, float) and not isinstance(fc, int)):
                raise ValueError(
                    r"Dict uncertainties_test should be of structure {string_id1: float_uncertainty1/int_uncertainty1, string_id2: float_uncertainty2/int_uncertainty2 } "
                )

    if np.array(uncert_thresh).shape != (len(uncert_thresh), 1):
        raise ValueError("uncert_thresh list should be 2D e.g. [[0.1], [0.2], [0.3]] ")

    all_binned_errors = {}

    for i, (key, fc) in enumerate(uncertainties_test.items()):
        for q in range(len(uncert_thresh) + 1):
            if q == 0:
                lower_c_bound = uncert_thresh[q][0]

                if fc <= lower_c_bound:
                    all_binned_errors[key] = q

            elif q < len(uncert_thresh):
                lower_c_bound = uncert_thresh[q - 1][0]
                upper_c_bound = uncert_thresh[q][0]

                if fc <= upper_c_bound:
                    if fc > lower_c_bound:
                        all_binned_errors[key] = q

            # Finally do the last bin
            else:
                lower_c_bound = uncert_thresh[q - 1][0]

                if fc > lower_c_bound:
                    all_binned_errors[key] = q

    return all_binned_errors
