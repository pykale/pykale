"""
Authors: Lawrence Schobs, lawrenceschobs@gmail.com

Functions to save results to Excel files.

"""

from typing import Any, Dict

import numpy as np
import pandas as pd


def generate_summary_df(
    results_dictionary: dict, cols_to_save: list, sheet_name: str, save_location: str
) -> pd.DataFrame:
    """
    Generates pandas dataframe with summary statistics.
    Designed for use in Quantile Binning (/pykale/examples/landmark_uncertainty/main.py).
    Args:
        results_dictionary (dict): A dictionary containing results for each quantile bin and uncertainty method.
            The keys are strings indicating the name of each uncertainty method.
            The values are dictionaries containing results for each quantile bin.
        cols_to_save (list): A list of 2-element lists, each containing a string indicating the key in the
            results_dictionary and a string indicating the name to use for that column in the output dataframe.
        sheet_name (str): The name of the sheet to create in the output Excel file.
        save_location (str): The file path and name to use for the output Excel file.
    Returns:
        pd.DataFrame: A dataframe with statistics including mean error, std error of All and individual
        targets. Also includes the Sucess detection rates (SDR).
        The dataframe should have the following structure:
            df = {
                "All um col_save_name Mean": value,
                "All um col_save_name Std": value,
                "B1 um col_save_name Mean": value,
                "B1 um col_save_name Std": value,
                ...
            }
    """

    # Save data to csv files
    summary_dict = {}
    for [col_dict_key, col_save_name] in cols_to_save:
        for um in results_dictionary[col_dict_key].keys():
            col_data = results_dictionary[col_dict_key][um]
            # Remove instances of None (which were added when the list was empty, rather than nan)
            summary_dict["All " + um + " " + col_save_name + " Mean"] = np.mean(
                [x for sublist in col_data for x in sublist if x is not None]
            )
            summary_dict["All " + um + " " + col_save_name + " Std"] = np.std(
                [x for sublist in col_data for x in sublist if x is not None]
            )
            for bin_idx, bin_data in enumerate(col_data):
                filtered_bin_data = [x for x in bin_data if x is not None]
                summ_mean = np.mean(filtered_bin_data) if (len(filtered_bin_data) > 0) else None
                summ_std = np.std(filtered_bin_data) if len(filtered_bin_data) > 0 else None

                summary_dict["B" + str(bin_idx + 1) + " " + um + " " + col_save_name + " Mean"] = summ_mean
                summary_dict["B" + str(bin_idx + 1) + " " + um + " " + col_save_name + " Std"] = summ_std

    save_dict_xlsx(summary_dict, save_location, sheet_name)


def save_dict_xlsx(data_dict: Dict[Any, Any], save_location: str, sheet_name: str) -> None:
    """
    Save a dictionary to an Excel file using the XlsxWriter engine.

    Parameters:
    data_dict (Dict[Any, Any]): The dictionary that needs to be saved to an Excel file. The keys of the dictionary represent the
                                row index and the values represent the data in the row. If a dictionary value is a list or a
                                series, each element in the list/series will be a column in the row.

    save_location (str): The location where the Excel file will be saved. This should include the full path and the filename,
                         for example, "/path/to/save/data.xlsx". Overwrites the file if it already exists.

    sheet_name (str): The name of the sheet where the dictionary will be saved in the Excel file.

    Returns:
    None: This function does not return anything. It saves the dictionary as an Excel file at the specified location.
    """
    pd_df = pd.DataFrame.from_dict(data_dict, orient="index")

    with pd.ExcelWriter(save_location, engine="xlsxwriter") as writer:  # pylint: disable=abstract-class-instantiated
        for n, df in (pd_df).items():
            df.to_excel(writer, sheet_name=sheet_name)
