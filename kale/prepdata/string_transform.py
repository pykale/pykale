"""Author: Lawrence Schobs, lawrenceschobs@gmail.com
    This file contains functions for string manipulation.
"""


def strip_for_bound(string_: str) -> list:
    """
    Convert a string containing comma-separated floats into a list of floats.
    Args:
        string_ (str): A string containing floats, separated by commas.
    Returns:
        list: A list of floats.
    Example:
        >>> strip_for_bound("[1.0, 2.0], [3.0, 4.0]")
        [[1.0, 2.0], [3.0, 4.0]]
    """
    bounds = []
    for entry in string_:
        entry = entry[1:-1]
        bounds.append([float(i) for i in entry.split(",")])
    return bounds
