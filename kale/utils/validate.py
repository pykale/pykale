import inspect


def validate_kwargs(func, kwargs):
    """
    Filters a dictionary to include only the valid kwargs for a given function.

    Parameters:
        func (callable): The function or method to validate against.
        kwargs (dict): The dictionary of arguments to filter.

    Returns:
        dict: A dictionary containing only the valid kwargs.
    """
    # Get the parameter names of the function
    signature = inspect.signature(func)
    valid_params = signature.parameters.keys()

    # Filter kwargs to include only keys that match the function parameters
    valid_kwargs = {key: value for key, value in kwargs.items() if key in valid_params}
    return valid_kwargs
