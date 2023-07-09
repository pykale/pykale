"""Logging functions, based on https://github.com/HaozhiQi/ISONet/blob/master/isonet/utils/logger.py"""

import datetime
import logging
import os
import uuid


def out_file_core():
    """Creates an output file name concatenating a formatted date and uuid, but without an extension.

    Returns:
        string: A string to be used in a file name.
    """
    date = str(datetime.datetime.now().strftime("%Y%d%m_%H%M%S"))
    return f"log-{date}-{str(uuid.uuid4())}"


def construct_logger(name, save_dir, log_to_terminal=False):
    """Constructs a logger. Saves the output as a text file at a specified path.
    Also saves the output of `git diff HEAD` to the same folder.
    Takes option to log to terminal, which will print logging statements. Default is False.

    The logger is configured to output messages at the DEBUG level, and it saves the output as a text file with a name
    based on the current timestamp and the specified name. It also saves the output of `git diff HEAD` to a file with
    the same name and the extension `.gitdiff.patch`.

    Args:
        name (str): The name of the logger, typically the name of the method being logged.
        save_dir (str): The directory where the log file and git diff file will be saved.
        log_to_terminal (bool, optional): Whether to also log messages to the terminal. Defaults to False.

    Returns:
        logging.Logger: The constructed logger.

    Reference:
        https://docs.python.org/3/library/logging.html

    Raises:
        None.
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

    if log_to_terminal:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger
