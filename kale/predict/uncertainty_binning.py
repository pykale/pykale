# =============================================================================
# Author: Lawrence Schobs, laschobs1@sheffield.ac.uk
# =============================================================================

import numpy as np


def quantile_binning_predictions(uncertainties_test, uncert_thresh, save_pred_path=None):
    """Bin predictions based on quantile thresholds.

    Args:
        uncertainties_test (Dict): Dict of uncertainties like {id: x, id: x}
        uncert_thresh (list): quantile thresholds to determine binning,
        save_pred_path (str): path preamble to save predicted bins to,


    Returns:
        Dict: Dictionary of predicted quantile bins.
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
