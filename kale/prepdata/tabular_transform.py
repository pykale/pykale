"""
Functions for manipulating tabular data
"""

import os
from typing import Any, Dict, List, Tuple

import pandas as pd


def apply_confidence_inversion(data: pd.DataFrame, uncertainty_measure: str) -> Dict[str, Any]:
    """Invert a list of numbers, add a small number to avoid division by zero.

    Args:
        data (Dict): Dictionary of data to invert.
        uncertainty_measure (str): Key of dict to invert.

    Returns:
        Dict: Dictionary with inverted data.
    """

    if uncertainty_measure not in data:
        raise KeyError("The key %s not in the dictionary provided" % uncertainty_measure)

    # Make sure no value is less than zero.
    min_not_zero = min(i for i in data[uncertainty_measure] if i > 0)
    data.loc[data[uncertainty_measure] < 0, uncertainty_measure] = min_not_zero
    data[uncertainty_measure] = 1 / data[uncertainty_measure] + 0.0000000000001
    return data


def get_data_struct(
    models_to_compare: List[str], landmarks: List[int], saved_bins_path_pre: str, dataset: str
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """Returns dictionaries of pandas dataframes for:
        a) all error and prediction info
        b) landmark separated error and prediction info
        c) all estimated error bounds
        d) landmark separated estimated error bounds.

    Args:
        models_to_compare: List of set models to add to data struct.
        landmarks: List of landmarks to add to data struct.
        saved_bins_path_pre: Preamble to path of where the predicted quantile bins are saved.
        dataset: String of what dataset you're measuring.

    Returns:
        A tuple containing the following dictionaries of pandas dataframes:
            - all error and prediction info
            - landmark separated error and prediction info
            - all estimated error bounds
            - landmark separated estimated error bounds.
    """

    data_structs = {}
    data_struct_sep = {}  #

    data_struct_bounds = {}
    data_struct_bounds_sep = {}

    for model in models_to_compare:
        all_landmarks = []
        all_err_bounds = []

        for lm in landmarks:
            bin_pred_path = os.path.join(saved_bins_path_pre, model, dataset, "res_predicted_bins_l" + str(lm))
            bin_preds = pd.read_csv(bin_pred_path + ".csv", header=0)
            bin_preds["landmark"] = lm

            error_bounds_path = os.path.join(saved_bins_path_pre, model, dataset, "estimated_error_bounds_l" + str(lm))
            error_bounds_pred = pd.read_csv(error_bounds_path + ".csv", header=0)
            error_bounds_pred["landmark"] = lm

            all_landmarks.append(bin_preds)
            all_err_bounds.append(error_bounds_pred)
            data_struct_sep[model + " L" + str(lm)] = bin_preds
            data_struct_bounds_sep[model + "Error Bounds L" + str(lm)] = error_bounds_pred

        data_structs[model] = pd.concat(all_landmarks, axis=0, ignore_index=True)
        data_struct_bounds[model + " Error Bounds"] = pd.concat(all_err_bounds, axis=0, ignore_index=True)

    return data_structs, data_struct_sep, data_struct_bounds, data_struct_bounds_sep
