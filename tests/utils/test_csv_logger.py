import logging
import os

# from kale.utils import logger
from kale.utils.csv_logger import setup_logger


def test_csv_logger(download_path):
    train_params = {"para1": 1, "para2": 2}
    method = "DANN"
    seed = 32  # To define in conftest later?
    testing_logger, results, _, test_csv_file = setup_logger(train_params, download_path, method, seed)
    results.to_csv(test_csv_file)
    assert isinstance(testing_logger, logging.Logger)
    assert os.path.isfile(test_csv_file)

    # Teardown log file
    os.remove(test_csv_file)
    os.remove(os.path.join(download_path, "parameters.json"))
