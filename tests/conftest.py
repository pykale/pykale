# Global settings for tests. Run before any test
import os

import pytest


@pytest.fixture(scope="session")
def download_path():
    path = os.path.join("tests", "test_data")
    return path
