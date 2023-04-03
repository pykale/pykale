"""
Functions for manipulating tabular data
"""

import os

import pandas as pd


def apply_confidence_inversion(data, uncertainty_measure):
    """ Inverses a list of numbers, adds a small number to avoid 1/0.
    Args:
        data (Dict): dictionary of data to invert
        uncertainty_measure (string): key of dict to invert

    Returns:
        Dict: dict with inverted data.

    """

    if uncertainty_measure not in data:
        raise KeyError("The key %s not in the dictionary provided" % uncertainty_measure)

    # data[uncertainty_measure]  = data[data[uncertainty_measure]<0] = 0.1

    # Make sure no value is less than zero.
    min_not_zero = min(i for i in data[uncertainty_measure] if i > 0)
    data.loc[data[uncertainty_measure] < 0, uncertainty_measure] = min_not_zero
    data[uncertainty_measure] = 1 / data[uncertainty_measure] + 0.0000000000001
    return data


def get_data_struct(models_to_compare, landmarks, saved_bins_path_pre, dataset):
    """ Makes a dict of pandas dataframe used to evaluate uncertainty measures
        Args:
            models_to_compare (list): list of set models to add to datastruct,
            landmarks (list): list of landmarks to add to datastruct.
            saved_bins_path_pre (string): preamble to path of where the predicted quantile bins are saved.
            dataset (string): string of what dataset you're measuring.

        Returns:
            [Dict, Dict, Dict, Dict]: dictionaries of pandas dataframes for: a) all error & pred info, b) landmark
            seperated error & pred info c) all estimated error bound d) landmark seperated estimated error bounds.
    """

    data_structs = {}
    data_struct_sep = {}  #

    data_struct_bounds = {}
    data_struct_bounds_sep = {}

    for model in models_to_compare:
        all_landmarks = []
        all_err_bounds = []

        for lm in landmarks:
            bin_pred_path = os.path.join(saved_bins_path_pre, model, "res_predicted_bins_l" + str(lm))
            bin_preds = pd.read_csv(bin_pred_path + ".csv", header=0)
            bin_preds["landmark"] = lm

            error_bounds_path = os.path.join(saved_bins_path_pre, model, "estimated_error_bounds_l" + str(lm))
            error_bounds_pred = pd.read_csv(error_bounds_path + ".csv", header=0)
            error_bounds_pred["landmark"] = lm

            all_landmarks.append(bin_preds)
            all_err_bounds.append(error_bounds_pred)
            data_struct_sep[model + " L" + str(lm)] = bin_preds
            data_struct_bounds_sep[model + "Error Bounds L" + str(lm)] = error_bounds_pred

        data_structs[model] = pd.concat(all_landmarks, axis=0, ignore_index=True)
        data_struct_bounds[model + " Error Bounds"] = pd.concat(all_err_bounds, axis=0, ignore_index=True)

    return data_structs, data_struct_sep, data_struct_bounds, data_struct_bounds_sep
