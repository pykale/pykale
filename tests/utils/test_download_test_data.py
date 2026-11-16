"""Exercise SVHN source selection without external downloads."""

from unittest.mock import Mock

import pytest

from tests import download_test_data as data


def test_svhn_prefers_pinned_mirror_and_disables_torchvision_download(monkeypatch, tmp_path):
    download = Mock()
    dataset = Mock(split_list=data.SVHN.split_list)
    monkeypatch.setattr(data, "download_file_by_url", download)
    monkeypatch.setattr(data, "SVHN", dataset)
    monkeypatch.setattr(data.time, "sleep", Mock())
    data.download_svhn_data(tmp_path)
    assert download.call_count == 2
    for call, split in zip(download.call_args_list, ("train", "test")):
        _, filename, md5 = dataset.split_list[split]
        assert call.args == (data.SVHN_MIRRORS[filename][0], tmp_path, filename, "mat")
        assert data.SVHN_HF_REVISION in call.args[0]
        assert call.kwargs == {"md5": md5}
    assert [call.kwargs["download"] for call in dataset.call_args_list] == [False, False]


def test_svhn_falls_back_after_transport_and_checksum_failures(monkeypatch, tmp_path):
    from kale.utils.download import ChecksumError

    download = Mock(side_effect=[OSError("certificate chain"), ChecksumError("bad bytes"), None])
    monkeypatch.setattr(data, "download_file_by_url", download)
    monkeypatch.setattr(data.time, "sleep", Mock())
    filename = "test_32x32.mat"
    data._download_svhn_file(tmp_path, filename, "expected-md5")
    assert [call.args[0] for call in download.call_args_list] == data.SVHN_MIRRORS[filename]
    assert all(call.kwargs["md5"] == "expected-md5" for call in download.call_args_list)


def test_svhn_reports_exhausted_sources_without_real_delays(monkeypatch, tmp_path):
    monkeypatch.setattr(data, "download_file_by_url", Mock(side_effect=OSError("unavailable")))
    sleep = Mock()
    monkeypatch.setattr(data.time, "sleep", sleep)
    filename = "train_32x32.mat"
    with pytest.raises(RuntimeError, match="Could not download train_32x32.mat") as error:
        data._download_svhn_file(tmp_path, filename, "expected-md5")
    assert data.download_file_by_url.call_count == len(data.SVHN_MIRRORS[filename]) * data.SVHN_MIRROR_ROUNDS
    assert all(url in str(error.value) for url in data.SVHN_MIRRORS[filename])
    assert [call.args[0] for call in sleep.call_args_list] == [0, 60, 60]
