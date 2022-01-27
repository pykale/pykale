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


def test_create_single_result_dict():
    result_dict = logger.create_single_result_dict([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]], ("a", "b"))
    assert isinstance(result_dict, dict)


def test_create_multi_results_dict():
    result_dict = logger.create_multi_results_dict(
        [[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]], [[0.3, 0.3, 0.3], [0.4, 0.4, 0.4]], ("a", "b")
    )
    assert isinstance(result_dict, dict)


@pytest.mark.parametrize("noun", [True, False])
def test_save_results_to_json(noun):
    y_hat = [[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]]
    y_t_hat = [[0.3, 0.3, 0.3], [0.4, 0.4, 0.4]]
    y_ids = ("a", "b")
    y_t_ids = ("c", "d")
    file_name = "test.json"
    if noun:
        y_hat_noun = [[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]]
        y_t_hat_noun = [[0.3, 0.3, 0.3], [0.4, 0.4, 0.4]]
    else:
        y_hat_noun = None
        y_t_hat_noun = None
    logger.save_results_to_json(
        y_hat, y_t_hat, y_ids, y_t_ids, y_hat_noun, y_t_hat_noun, verb=True, noun=noun, file_name=file_name
    )
    assert os.path.isfile(file_name)
