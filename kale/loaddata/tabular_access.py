import numpy as np
import pandas as pd


def load_csv_columns(datapath, split, fold, cols_to_return="All"):

    """
        Read csv file of data, returns samples where the value of the "split" column
        is contained in the "fold" variable. The columns cols_to_return are returned.

    Args:
        datapath (str): Path to csv file of uncertainty results,
        split (str): column name for split e.g. Validation, testing,
        fold (int or [int]]): fold/s contained in the split column to return,
        cols_to_return ([str]): Which columns to return (default="All").


    Returns:
        [pandas dataframe, pandas dataframe]: dataframe selected
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
