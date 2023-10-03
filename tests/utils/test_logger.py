import logging
import os

import pytest

from kale.utils import logger


@pytest.fixture(scope="module")
def testing_logger():
    save_dir = os.getcwd()
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


def test_construct_logger_terminal(caplog):
    """Test that logger outputs to terminal when log_to_terminal is True."""
    logger_name = "test_logger"
    save_dir = "./outputs"
    os.makedirs(save_dir, exist_ok=True)

    t_logger = logger.construct_logger(logger_name, save_dir, log_to_terminal=True)
    t_logger.debug("This is a debug message")
    t_logger.info("This is an info message")

    # Assert that messages were logged to the terminal
    assert "This is a debug message" in caplog.text
    assert "This is an info message" in caplog.text


@pytest.fixture
def log_file_name(testing_logger):
    filehandler = testing_logger.handlers[0]
    yield filehandler.baseFilename


@pytest.fixture
def gitdiff_file_name(log_file_name):
    [folder, file] = os.path.split(log_file_name)
    file_core = os.path.splitext(file)[0]
    yield os.path.join(folder, file_core + ".gitdiff.patch")


def test_out_file_core():
    out_file_eg = logger.out_file_core()
    assert isinstance(out_file_eg, str)


def test_logger_type(testing_logger):
    assert isinstance(testing_logger, logging.Logger)


def test_log_file_exists(log_file_name):
    assert os.path.isfile(log_file_name)


def test_gitdiff_file_exists(gitdiff_file_name):
    assert os.path.isfile(gitdiff_file_name)
