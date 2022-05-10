"""Screen printing functions, from https://github.com/HaozhiQi/ISONet/blob/master/isonet/utils/misc.py"""


def tprint(*args):
    """Temporarily prints things on the screen so that it won't be flooded"""
    print("\r", end="")  # noqa: T201
    print(*args, end="")  # noqa: T201


def pprint(*args):  # noqa: T204
    """Permanently prints things on the screen to have all info displayed"""
    print("\r", end="")  # noqa: T201
    print(*args)  # noqa: T201


def pprint_without_newline(*args):
    """Permanently prints things on the screen, separated by space rather than newline"""
    print("\r", end="")  # noqa: T201
    print(*args, end=" ")  # noqa: T201
