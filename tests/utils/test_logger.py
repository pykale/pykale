import logging

import pytest

from kale.utils import logger


@pytest.fixture
def testing_logger():
    return logger.construct_logger("test", "./")


def test_logger_type(testing_logger):
    assert isinstance(testing_logger, logging.Logger)
