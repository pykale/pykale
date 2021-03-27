import logging
import os

import pytest

from kale.utils import logger


@pytest.fixture
def testing_logger():
    save_dir = "./"
    testing_logger = logger.construct_logger("test", save_dir)
    yield testing_logger

    # Teardown
    filehandler = testing_logger.handlers[0]
    log_file_name = filehandler.baseFilename
    gitdiff_file_name = os.path.join(save_dir, "gitdiff.patch")
    filehandler.close()
    os.remove(log_file_name)
    os.remove(gitdiff_file_name)


def test_logger_type(testing_logger):
    assert isinstance(testing_logger, logging.Logger)
