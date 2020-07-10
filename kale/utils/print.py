# Created by Haiping Lu directly from https://github.com/HaozhiQi/ISONet/blob/master/isonet/utils/misc.py 
# Under the MIT License
def tprint(*args):
    """Temporarily prints things on the screen"""
    print("\r", end="")
    print(*args, end="")


def pprint(*args):
    """Permanently prints things on the screen"""
    print("\r", end="")
    print(*args)


def pprint_without_newline(*args):
    print('\r', end='')
    print(*args, end=' ')
