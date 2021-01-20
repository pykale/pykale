"""Screen printing functions, from https://github.com/HaozhiQi/ISONet/blob/master/isonet/utils/logger.py"""

import datetime
import logging
import os
import shlex
import subprocess


def git_hash():
    """Gets a hash for different runs to have unique logfile names."""
    cmd = 'git log -n 1 --pretty="%h"'
    ret = subprocess.check_output(shlex.split(cmd)).strip()
    if isinstance(ret, bytes):
        ret = ret.decode()
    return ret


def construct_logger(name, save_dir):
    """Constructs a simple txt logger with a specified name at a specified path

    Reference: https://docs.python.org/3/library/logging.html

    Args:
        name (str): the logger name, typically the method name
        save_dir (str): the path to save the log file (.txt)
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    date = str(datetime.datetime.now().strftime("%m%d%H"))
    fh = logging.FileHandler(os.path.join(save_dir, f"log-{date}-{git_hash()}.txt"), encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    os.system(f"git diff HEAD > {save_dir}/gitdiff.patch")

    return logger
