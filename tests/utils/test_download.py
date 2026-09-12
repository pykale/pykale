import hashlib
import os
from pathlib import Path
from unittest.mock import call, MagicMock, patch

import pytest

from kale.utils.download import (
    _remove_partial_files,
    _retry_download,
    _verify_file,
    download_file_by_url,
    download_file_gdrive,
)

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


def test_remove_partial_files_swallows_unlink_error(tmp_path, caplog):
    # If deleting a partial file fails (e.g. permission error), the error is logged
    # and swallowed rather than propagated, so it cannot mask the original download error.
    path = tmp_path / "locked.bin"
    path.write_bytes(b"x")
    with patch.object(Path, "unlink", side_effect=OSError("permission denied")):
        with caplog.at_level("WARNING"):
            _remove_partial_files([path])
    assert any("Could not remove partial download" in message for message in caplog.messages)


def test_download_file_by_url_archive_uses_retry(tmp_path):
    with patch("kale.utils.download.download_url") as mock_dl:
        with patch("kale.utils.download.extract_archive") as mock_extract:
            download_file_by_url("http://example.com/data.zip", tmp_path, "data.zip", "zip")
    mock_dl.assert_called_once()
    mock_extract.assert_called_once()


def test_download_file_by_url_archive_verifies_before_extract(tmp_path):
    # For archives, a checksum mismatch must be caught BEFORE extraction, so the bad
    # archive is never unpacked and is cleaned up after retries are exhausted.
    def fake_download(url, root, filename, md5=None):
        Path(root).joinpath(filename).write_bytes(b"corrupt-archive")

    with patch("kale.utils.download.time.sleep"):
        with patch("kale.utils.download.download_url", side_effect=fake_download):
            with patch("kale.utils.download.extract_archive") as mock_extract:
                with pytest.raises(RuntimeError, match="md5 mismatch"):
                    download_file_by_url("http://example.com/data.zip", tmp_path, "data.zip", "zip", md5="0" * 32)
    mock_extract.assert_not_called()
    assert not (tmp_path / "data.zip").exists()


def test_download_file_by_url_verifies_cached_file(tmp_path):
    # A cached file that fails verification is re-downloaded rather than silently accepted.
    good = b"the real payload"
    file = tmp_path / "data.pkl"
    file.write_bytes(b"stale-corrupt")  # pre-existing, wrong content

    def fake_download(url, dest):
        Path(dest).write_bytes(good)

    with patch("kale.utils.download.download_url_to_file", side_effect=fake_download) as mock_dl:
        download_file_by_url(
            "http://example.com/data.pkl", tmp_path, "data.pkl", "pkl", md5=hashlib.md5(good).hexdigest()
        )
    mock_dl.assert_called_once()  # the stale cached file triggered a re-download
    assert file.read_bytes() == good


def test_download_file_by_url_skips_valid_cached_file(tmp_path):
    # A cached file that passes verification is reused without downloading.
    content = b"already here"
    file = tmp_path / "data.pkl"
    file.write_bytes(content)

    with patch("kale.utils.download.download_url_to_file") as mock_dl:
        download_file_by_url(
            "http://example.com/data.pkl", tmp_path, "data.pkl", "pkl", md5=hashlib.md5(content).hexdigest()
        )
    mock_dl.assert_not_called()


def test_download_file_by_url_plain_uses_retry(tmp_path):
    with patch("kale.utils.download.download_url_to_file") as mock_dl:
        download_file_by_url("http://example.com/data.pkl", tmp_path, "data.pkl", "pkl")
    mock_dl.assert_called_once()


def test_verify_file_noop_without_expectations(tmp_path):
    # No md5/sha256/size given: nothing is checked, even for a non-existent file.
    _verify_file(tmp_path / "does_not_exist.bin")


def test_verify_file_matches(tmp_path):
    content = b"pykale download payload"
    file = tmp_path / "data.bin"
    file.write_bytes(content)
    _verify_file(
        file,
        md5=hashlib.md5(content).hexdigest(),
        sha256=hashlib.sha256(content).hexdigest(),
        file_size=len(content),
    )


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"md5": "deadbeef"}, "md5 mismatch"),
        ({"sha256": "deadbeef"}, "sha256 mismatch"),
        ({"file_size": 999999}, "Size mismatch"),
    ],
)
def test_verify_file_mismatch_raises(tmp_path, kwargs, match):
    file = tmp_path / "data.bin"
    file.write_bytes(b"pykale download payload")
    with pytest.raises(RuntimeError, match=match):
        _verify_file(file, **kwargs)


def test_verify_file_missing_raises(tmp_path):
    with pytest.raises(RuntimeError, match="does not exist"):
        _verify_file(tmp_path / "missing.bin", md5="deadbeef")


def test_download_file_by_url_verifies_and_passes(tmp_path):
    content = b"hello world"

    def fake_download(url, dest):
        Path(dest).write_bytes(content)

    with patch("kale.utils.download.download_url_to_file", side_effect=fake_download):
        download_file_by_url(
            "http://example.com/data.pkl",
            tmp_path,
            "data.pkl",
            "pkl",
            md5=hashlib.md5(content).hexdigest(),
            file_size=len(content),
        )
    assert (tmp_path / "data.pkl").read_bytes() == content


def test_download_file_by_url_checksum_mismatch_retries_and_cleans(tmp_path):
    # A wrong checksum makes every attempt fail; the corrupt file must not be left behind
    # and the error surfaces after retries are exhausted.
    def fake_download(url, dest):
        Path(dest).write_bytes(b"corrupted")

    with patch("kale.utils.download.time.sleep"):
        with patch("kale.utils.download.download_url_to_file", side_effect=fake_download) as mock_dl:
            with pytest.raises(RuntimeError, match="md5 mismatch"):
                download_file_by_url(
                    "http://example.com/data.pkl",
                    tmp_path,
                    "data.pkl",
                    "pkl",
                    md5="0" * 32,
                )
    assert mock_dl.call_count == 3
    assert not (tmp_path / "data.pkl").exists()


@pytest.mark.parametrize("param", PARAM)
def test_download_file_by_url(param):
    url, output_file_name, file_format = param.split(";")

    # run twice to test the code when the file exist
    download_file_by_url(url, output_directory, output_file_name, file_format)
    download_file_by_url(url, output_directory, output_file_name, file_format)

    assert os.path.exists(output_directory.joinpath(output_file_name)) is True
    assert output_directory.exists()


def test_download_file_gdrive_archive_mocked(tmp_path):
    # Exercise the gdrive download + extract branch without hitting the network.
    def fake_gdrive(id, root, name):
        Path(root).joinpath(name).write_bytes(b"archive-bytes")

    with patch("kale.utils.download.download_file_from_google_drive", side_effect=fake_gdrive) as mock_dl:
        with patch("kale.utils.download.extract_archive") as mock_extract:
            download_file_gdrive("some-id", tmp_path, "data.zip", "zip")
    mock_dl.assert_called_once()
    mock_extract.assert_called_once()
    assert (tmp_path / "data.zip").exists()


def test_download_file_gdrive_plain_mocked(tmp_path):
    # Exercise the gdrive plain (no-extract) branch without hitting the network.
    def fake_gdrive(id, root, name):
        Path(root).joinpath(name).write_bytes(b"plain-bytes")

    with patch("kale.utils.download.download_file_from_google_drive", side_effect=fake_gdrive) as mock_dl:
        with patch("kale.utils.download.extract_archive") as mock_extract:
            download_file_gdrive("some-id", tmp_path, "data.csv", "csv")
    mock_dl.assert_called_once()
    mock_extract.assert_not_called()
    assert (tmp_path / "data.csv").exists()


@pytest.mark.parametrize("param", GDRIVE_PARAM)
def test_download_file_gdrive(param):
    id, output_file_name, file_format = param.split(";")

    # run twice to test the code when the file exist
    download_file_gdrive(id, output_directory, output_file_name, file_format)
    download_file_gdrive(id, output_directory, output_file_name, file_format)

    assert os.path.exists(output_directory.joinpath(output_file_name)) is True
    assert output_directory.exists()
