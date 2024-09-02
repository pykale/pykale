import os
import shutil

import pytest

from kale.utils.misc_utils import float2str, mkdir


# Test mkdir function
@pytest.fixture(scope="module")
def temp_dir():
    # Create a temporary directory for testing
    dir_path = "temp_test_dir"
    yield dir_path
    # Cleanup after tests
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)


def test_mkdir(temp_dir):
    # Test if mkdir creates a directory when it does not exist
    assert not os.path.exists(temp_dir)
    mkdir(temp_dir)
    assert os.path.exists(temp_dir)

    # Test if mkdir does nothing if the directory already exists
    mkdir(temp_dir)
    assert os.path.exists(temp_dir)


# Test float2str function
@pytest.mark.parametrize(
    "value, expected",
    [
        (3.1415926535, "3.1416"),
        (0.1234, "0.1234"),
        (1.0, "1.0000"),
        (123456.789, "123456.7890"),
    ],
)
def test_float2str(value, expected):
    assert float2str(value) == expected
