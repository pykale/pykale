import numpy as np
import pandas as pd


def load_uncertainty_pairs_csv(datapath, split, fold, cols_to_return="All"):

    """Read csv file of data, return by split, fold and columns to return.

    Args:
        datapath (str): Path to csv file of uncertainty results,
        split (str): column name for split e.g. Validation, testing,
        split (str): column name for split e.g. Validation, testing,
        cols_to_return ([str]): Which columns to return (default="All").


    Returns:
        [pandas dataframe, pandas dataframe]: dataframe selected
    """

    # Load the uncertainty & error results
    datafame = pd.read_csv(datapath + ".csv", header=0)

    if cols_to_return == "All":
        cols_to_return = datafame.columns

    # Test if a single fold or list of folds
    if isinstance(fold, (list, pd.core.series.Series, np.ndarray)):
        return_data = (datafame.loc[datafame[split].isin(fold)]).loc[:, cols_to_return]
    else:
        return_data = (datafame.loc[datafame[split] == fold]).loc[:, cols_to_return]

    return return_data
