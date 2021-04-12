# Global settings for tests. Run before any test
import pytest


@pytest.fixture(scope="session")
def download_path():
    path = "tests/test_data/download"
    return path
