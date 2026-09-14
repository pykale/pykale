import os
from pathlib import Path
from unittest.mock import call, MagicMock, patch

import pytest

from kale.utils.download import _retry_download, download_file_by_url, download_file_gdrive

output_directory = Path().absolute().joinpath("tests/test_data/download")
PARAM = [
    "https://github.com/pykale/data/raw/main/videos/video_test_data/ADL/annotations/labels_train_test/adl_P_11_train.pkl;a.pkl;pkl",
    "https://github.com/pykale/data/raw/main/videos/video_test_data.zip;video_test_data.zip;zip",
]

GDRIVE_PARAM = [
    "1U4D23R8u8MJX9KVKb92bZZX-tbpKWtga;demo_datasets.zip;zip",
    "1SV7fmAnWj-6AU9X5BGOrvGMoh2Gu9Nih;dummy_data.csv;csv",
]


def test_retry_download_succeeds_on_first_attempt():
    fn = MagicMock()
    _retry_download(fn, retries=3, backoff=2)
    fn.assert_called_once()


def test_retry_download_retries_on_failure():
    fn = MagicMock(side_effect=[RuntimeError("timeout"), RuntimeError("timeout"), None])
    with patch("kale.utils.download.time.sleep") as mock_sleep:
        _retry_download(fn, retries=3, backoff=2)
    assert fn.call_count == 3
    mock_sleep.assert_has_calls([call(1), call(2)])


def test_retry_download_raises_after_all_retries():
    fn = MagicMock(side_effect=RuntimeError("timeout"))
    with patch("kale.utils.download.time.sleep"):
        with pytest.raises(RuntimeError, match="timeout"):
            _retry_download(fn, retries=3, backoff=2)
    assert fn.call_count == 3


@pytest.mark.parametrize(
    "kwargs",
    [{"retries": 0}, {"retries": -1}, {"backoff": 0}, {"backoff": -1}],
)
def test_retry_download_invalid_args(kwargs):
    with pytest.raises(ValueError):
        _retry_download(MagicMock(), **kwargs)


def test_download_file_by_url_archive_uses_retry(tmp_path):
    with patch("kale.utils.download.download_and_extract_archive") as mock_dl:
        download_file_by_url("http://example.com/data.zip", tmp_path, "data.zip", "zip")
    mock_dl.assert_called_once()


def test_download_file_by_url_plain_uses_retry(tmp_path):
    with patch("kale.utils.download.download_url_to_file") as mock_dl:
        download_file_by_url("http://example.com/data.pkl", tmp_path, "data.pkl", "pkl")
    mock_dl.assert_called_once()


@pytest.mark.parametrize("param", PARAM)
def test_download_file_by_url(param):
    url, output_file_name, file_format = param.split(";")

    # run twice to test the code when the file exist
    download_file_by_url(url, output_directory, output_file_name, file_format)
    download_file_by_url(url, output_directory, output_file_name, file_format)

    assert os.path.exists(output_directory.joinpath(output_file_name)) is True
    assert output_directory.exists()


@pytest.mark.parametrize("param", GDRIVE_PARAM)
def test_download_file_gdrive(param):
    id, output_file_name, file_format = param.split(";")

    # run twice to test the code when the file exist
    download_file_gdrive(id, output_directory, output_file_name, file_format)
    download_file_gdrive(id, output_directory, output_file_name, file_format)

    assert os.path.exists(output_directory.joinpath(output_file_name)) is True
    assert output_directory.exists()
