"""Screen printing functions, from https://github.com/HaozhiQi/ISONet/blob/master/isonet/utils/logger.py"""

import datetime
import logging
import os
import uuid


def out_file_core():
    """Creates an output file name concatenating a formatted date and uuid, but without an extension."""
    date = str(datetime.datetime.now().strftime("%Y%d%m_%H%M%S"))
    return f"log-{date}-{str(uuid.uuid4())}"


def construct_logger(name, save_dir):
    """Constructs a simple txt logger with a specified name at a specified path

    Reference: https://docs.python.org/3/library/logging.html

    Args:
        name (str): the logger name, typically the method name
        save_dir (str): the path to save the log file (.txt)
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    file_no_ext = out_file_core()

    fh = logging.FileHandler(os.path.join(save_dir, file_no_ext + ".txt"), encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    gitdiff_patch = os.path.join(save_dir, file_no_ext + ".gitdiff.patch")
    os.system(f"git diff HEAD > {gitdiff_patch}")

    return logger
