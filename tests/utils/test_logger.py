import logging
import os

import pytest

from kale.utils import logger


@pytest.fixture
def testing_logger():
    save_dir = "./"
    testing_logger = logger.construct_logger("test", save_dir)
    yield testing_logger

    # Gather info
    filehandler = testing_logger.handlers[0]
    log_file_name = filehandler.baseFilename

    # Teardown log file
    filehandler.close()
    os.remove(log_file_name)

    # Teardown gitdiff.patch file
    [folder, file] = os.path.split(log_file_name)
    file_core = os.path.splitext(file)[0]
    gitdiff_file_name = os.path.join(folder, file_core + ".gitdiff.patch")
    os.remove(gitdiff_file_name)


def test_out_file_core():
    out_file_eg = logger.out_file_core()
    assert isinstance(out_file_eg, str)


def test_logger_type(testing_logger):
    assert isinstance(testing_logger, logging.Logger)
