"""Author: Lawrence Schobs, lawrenceschobs@gmail.com
    This file contains functions for string manipulation.
"""
import re


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
        bounds.append([convert_to_float(i) for i in entry.split(",")])
    return bounds


def convert_to_float(value: str) -> float:
    """
    Convert a string to a float, handling NumPy float constructors like 'np.float32(...)', 'np.float64(...)', etc.

    Args:
        value (str): The string to convert.

    Returns:
        float: The converted float value.
    """
    # Match patterns like np.float32(0.5), np.float64(1.23e-5), etc.
    match = re.match(r"np\.float\d+\((.+)\)", value)
    if match:
        value = match.group(1)  # Extract the number inside the parentheses
    return float(value)
