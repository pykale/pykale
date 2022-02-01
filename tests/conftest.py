# Global settings for tests. Run before any test
import csv
import os

import pytest
from scipy.io import loadmat

from kale.utils.download import download_file_by_url
from tests.data.landmark_uncertainty_data import return_dummy_test_data, return_dummy_valid_data


@pytest.fixture(scope="session")
def download_path():
    path = os.path.join("tests", "test_data")
    os.makedirs(path, exist_ok=True)
    return path


gait_url = "https://github.com/pykale/data/raw/main/videos/gait/gait_gallery_data.mat"


@pytest.fixture(scope="session")
def gait(download_path):
    download_file_by_url(gait_url, download_path, "gait.mat", "mat")
    return loadmat(os.path.join(download_path, "gait.mat"))


@pytest.fixture(scope="session")
def office_path(download_path):
    path_ = os.path.join(download_path, "office")
    os.makedirs(path_, exist_ok=True)
    return path_


landmark_uncertainty_url = (
    "https://github.com/pykale/data/raw/main/tabular/cardiac_landmark_uncertainty/Uncertainty_tuples.zip"
)


@pytest.fixture(scope="session")
def landmark_uncertainty_dl(download_path):
    valid_path_ = os.path.join(download_path, "Uncertainty_tuples/dummy_valid_data")
    os.makedirs(valid_path_, exist_ok=True)

    test_path_ = os.path.join(download_path, "Uncertainty_tuples/dummy_test_data")
    os.makedirs(test_path_, exist_ok=True)

    dummy_validation_data = return_dummy_valid_data()
    dummy_test_data = return_dummy_test_data()

    # save dummy data
    zipped_valid = zip(*dummy_validation_data.values())
    zipped_test = zip(*dummy_test_data.values())

    with open(valid_path_ + ".csv", "w", newline="") as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerow(dummy_validation_data.keys())
        writer.writerows(zipped_valid)

    # save dummy data
    with open(test_path_ + ".csv", "w", newline="") as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerow(dummy_test_data.keys())
        writer.writerows(zipped_test)

    return valid_path_, test_path_
