import os


def mkdir(path):
    """
    Creates a directory if it does not already exist.

    This function strips any leading or trailing spaces and backslashes from the provided
    path and then checks if the directory exists. If it doesn't exist, the directory is
    created.

    Args:
        path (str): The directory path to be created.
    """
    path = path.strip()
    path = path.rstrip("\\")
    is_exists = os.path.exists(path)
    if not is_exists:
        os.makedirs(path)


def float2str(x):
    """
    Convert a floating-point number to a string with 4 decimal places.

    This function takes a float and returns a string representation of the float, rounded
    to four decimal places.

    Args:
        x (float): The floating-point number to format.
    """
    return "%0.4f" % x
