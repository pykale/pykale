# Global settings for tests. Run before any test
import csv
import os

import pytest
from scipy.io import loadmat

from kale.utils.download import download_file_by_url


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
    path_ = os.path.join(download_path, "Uncertainty_tuples/dummy_data")
    os.makedirs(path_, exist_ok=True)

    dummy_tabular_data = {
        "uid": ["PHD_2154", "PHD_2158", "PHD_217", "PHD_2194"],
        "E-CPV Error": [1.4142135, 3.1622777, 5.0990195, 61.846584],
        "E-CPV Uncertainty": [4.25442667, 4.449976897, 1.912124681, 35.76085777],
        "E-MHA Error": [3.1622777, 3.1622777, 4, 77.00649],
        "E-MHA Uncertainty": [0.331125357, 0.351173535, 1.4142135, 0.142362904],
        "S-MHA Error": [3.1622777, 1.4142135, 5.0990195, 56.32051],
        "S-MHA Uncertainty": [0.500086973, 0.235296882, 1.466040241, 0.123874651],
        "Validation Fold": [1, 1, 1, 1],
        "Testing Fold": [0, 0, 0, 0],
    }

    # save dummy data
    with open(path_ + ".csv", "w") as f:
        w = csv.DictWriter(f, dummy_tabular_data.keys())
        w.writeheader()
        w.writerow(dummy_tabular_data)

    # download_file_by_url(landmark_uncertainty_url, path_, "Uncertainty_tuples.zip", "zip")

    return path_
