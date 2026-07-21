import os
from pathlib import Path
from unittest.mock import call, MagicMock, patch

import pytest

from kale.utils.download import _remove_partial_files, _retry_download, download_file_by_url, download_file_gdrive

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


def test_retry_download_does_not_retry_programming_errors():
    # A non-download error (e.g. TypeError) is not one of _DOWNLOAD_ERRORS, so it must
    # propagate immediately without being retried and masked.
    fn = MagicMock(side_effect=TypeError("bad call"))
    with patch("kale.utils.download.time.sleep") as mock_sleep:
        with pytest.raises(TypeError, match="bad call"):
            _retry_download(fn, retries=3, backoff=2)
    fn.assert_called_once()
    mock_sleep.assert_not_called()


def test_retry_download_cleans_partial_file_between_attempts(tmp_path):
    # Simulate a download that writes a partial file and then fails, succeeding on the
    # second attempt. The partial file from the failed attempt must be removed.
    partial = tmp_path / "partial.bin"
    attempts = {"n": 0}

    def flaky():
        attempts["n"] += 1
        if attempts["n"] == 1:
            partial.write_bytes(b"incomplete")
            raise RuntimeError("connection reset")
        # success path leaves the (now complete) file in place
        partial.write_bytes(b"complete")

    seen_after_failure = {}

    original_remove = _remove_partial_files

    def spy(paths):
        paths = list(paths)
        seen_after_failure["existed"] = partial.exists()
        original_remove(paths)
        seen_after_failure["removed"] = not partial.exists()

    with patch("kale.utils.download.time.sleep"):
        with patch("kale.utils.download._remove_partial_files", side_effect=spy):
            _retry_download(flaky, retries=2, backoff=1, cleanup_paths=[partial])

    assert attempts["n"] == 2
    assert seen_after_failure["existed"] is True
    assert seen_after_failure["removed"] is True


def test_retry_download_cleans_partial_file_on_final_failure(tmp_path):
    partial = tmp_path / "partial.bin"

    def always_fails():
        partial.write_bytes(b"incomplete")
        raise RuntimeError("timeout")

    with patch("kale.utils.download.time.sleep"):
        with pytest.raises(RuntimeError, match="timeout"):
            _retry_download(always_fails, retries=2, backoff=1, cleanup_paths=[partial])
    assert not partial.exists()


def test_remove_partial_files_ignores_missing(tmp_path):
    existing = tmp_path / "there.bin"
    existing.write_bytes(b"x")
    missing = tmp_path / "nope.bin"
    # Must not raise on the missing path, and must remove the existing one.
    _remove_partial_files([existing, missing])
    assert not existing.exists()


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
