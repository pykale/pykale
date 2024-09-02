import os
import pytest
import numpy as np
import shutil
import logging

from kale.utils.misc_utils import mkdir, integer_label_protein, float2str, CHARPROTSET

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

# Test integer_label_protein function
@pytest.mark.parametrize("sequence, expected_length, expected_values", [
    ("ACDEFGHIKLMNPQRSTVWY", 1200, [CHARPROTSET[char] for char in "ACDEFGHIKLMNPQRSTVWY"]),
    ("XYZ", 1200, [24, 24, 24]),  # X, Y, Z are mapped as per CHARPROTSET
    ("", 1200, []),  # Empty sequence should return an array of zeros
])
def test_integer_label_protein(sequence, expected_length, expected_values):
    encoding = integer_label_protein(sequence)
    assert len(encoding) == expected_length
    assert np.all(encoding[:len(expected_values)] == expected_values)  # Check encoded values
    assert np.all(encoding[len(expected_values):] == 0)  # Check padding with zeros

def test_integer_label_protein_unknown_character(caplog):
    # Test with an unknown character in the sequence
    with caplog.at_level(logging.WARNING):
        encoding = integer_label_protein("ACDXYZ*")
        assert "character * does not exists in sequence category encoding" in caplog.text

# Test float2str function
@pytest.mark.parametrize("value, expected", [
    (3.1415926535, "3.1416"),
    (0.1234, "0.1234"),
    (1.0, "1.0000"),
    (123456.789, "123456.7890"),
])
def test_float2str(value, expected):
    assert float2str(value) == expected
