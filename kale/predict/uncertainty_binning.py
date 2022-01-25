# =============================================================================
# Author: Lawrence Schobs, laschobs1@sheffield.ac.uk
# =============================================================================


def quantile_binning_predictions(uncertainties_test, uncert_thresh, save_pred_path=None):
    """Bin predictions based on quantile thresholds.

    Args:
        uncertainties_test (list): list of uncertainties,
        uncert_thresh (list): quantile thresholds to determine binning,
        save_pred_path (str): path preamble to save predicted bins to,


    Returns:
        Dict: Dictionary of predicted quantile bins.
    """

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
