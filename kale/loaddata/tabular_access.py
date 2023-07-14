"""Authors: Lawrence Schobs, lawrenceschobs@gmail.com

Functions for accessing tabular data.

"""

from typing import List, Union

import numpy as np
import pandas as pd


def load_csv_columns(
    datapath: str, split: str, fold: Union[int, List[int]], cols_to_return: Union[str, List[str]] = "All"
) -> pd.DataFrame:
    """
    Reads a CSV file of data and returns samples where the value of the specified
    `split` column is contained in the `fold` variable. The columns specified in `cols_to_return` are returned.

    Args:
        datapath: The path to the CSV file of data.
        split: The column name for the split (e.g. "Validation", "Testing").
        fold: The fold/s contained in the split column to return. Can be a single integer or a list of integers.
        cols_to_return: Which columns to return. If set to "All", returns all columns.

    Returns:
        A tuple of two pandas DataFrames: the first is the full DataFrame selected, and the second is the DataFrame
        with only the columns specified in `cols_to_return`.
    """
    # Load the uncertainty & error results
    datafame = pd.read_csv(datapath + ".csv", header=0)

    if cols_to_return == "All":
        cols_to_return = datafame.columns
    elif not isinstance(cols_to_return, (list, pd.core.series.Series, np.ndarray)):
        cols_to_return = [cols_to_return]

    # Test if a single fold or list of folds
    if isinstance(fold, (list, pd.core.series.Series, np.ndarray)):
        return_data = (datafame.loc[datafame[split].isin(fold)]).loc[:, cols_to_return]
    else:
        return_data = (datafame.loc[datafame[split] == fold]).loc[:, cols_to_return]

    return return_data
