# Global settings for tests. Run before any test
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
