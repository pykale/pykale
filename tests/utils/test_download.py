import os
from pathlib import Path

import pytest

from kale.utils.download import download_file_by_url, download_file_gdrive

output_directory = Path().absolute().parent.joinpath("test_data/download")
PARAM = [
    "https://github.com/pykale/data/raw/main/videos/video_test_data/ADL/annotations/labels_train_test/adl_P_11_train.pkl;a.pkl;pkl",
    "https://github.com/pykale/data/raw/main/videos/video_test_data.zip;video_test_data.zip;zip",
]

GDRIVE_PARAM = [
    "1U4D23R8u8MJX9KVKb92bZZX-tbpKWtga;demo_datasets.zip;zip",
    "1SV7fmAnWj-6AU9X5BGOrvGMoh2Gu9Nih;dummy_data.csv;csv",
]


@pytest.mark.parametrize("param", PARAM)
def test_download_file_by_url(param):
    url, output_file_name, file_format = param.split(";")

    # run twice to test the code when the file exist
    download_file_by_url(url, output_directory, output_file_name, file_format)
    download_file_by_url(url, output_directory, output_file_name, file_format)

    assert os.path.exists(output_directory.joinpath(output_file_name)) is True


@pytest.mark.parametrize("param", GDRIVE_PARAM)
def test_download_file_gdrive(param):
    id, output_file_name, file_format = param.split(";")

    # run twice to test the code when the file exist
    download_file_gdrive(id, output_directory, output_file_name, file_format)
    download_file_gdrive(id, output_directory, output_file_name, file_format)

    assert os.path.exists(output_directory.joinpath(output_file_name)) is True
