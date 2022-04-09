import numpy as np
import pandas as pd

def read_csv_numeric(csv_path:str, index_col:list) -> np.array:
    """Read numeric data from comma separated text.

    Args:
        csv_path (str): Path to comma separated text file, with one row per example.
        index_col (list): Columns that should be treated as levels of a Pandas
            Dataframe MultiIndex. There should be an level for each subject,
            domain, and example (in that order).

    Returns:
        [array-like]: [description]
    """
    dataframe = pd.read_csv(csv_path, index_col=index_col)
    dataframe.sort_index(levels=index_col, inplace=True)
    value_counts_by_level = [dataframe.index.get_level_values(level).value_counts().to_numpy
                            for level in index_col]

    for value_counts in value_counts_by_level:
        if not all(value_counts[0] == x for x in value_counts[1:]):
            print("This should currently raise an error because we need everything to be balanced.")

    nfeatures = dataframe.shape[1]
    dims = [value_counts[0] for value_counts in value_counts_by_level] + [nfeatures]
    images = np.reshape(dataframe.to_numpy(), dims)

    return images
