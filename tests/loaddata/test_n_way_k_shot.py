import pytest
import torch
import sys
sys.path.append("/home/wenrui/Projects/pykale/")
from kale.loaddata.n_way_k_shot import NWayKShotDataset
from kale.utils.download import download_file_by_url
from yacs.config import CfgNode as CN
import os
from pathlib import Path
from torchvision import transforms


root_dir = os.path.dirname(os.path.dirname(os.getcwd()))
url = 'https://github.com/pykale/data/raw/main/images/omniglot/demo_data.zip'
mode="train"

@pytest.fixture(scope="module")
def testing_cfg(download_path):
    cfg = CN()
    cfg.DATASET = CN()
    cfg.DATASET.ROOT = root_dir + "/" + download_path + "/omniglot_test_data/"
    yield cfg

@pytest.mark.parametrize('mode', mode)
def test_n_way_k_shot(mode, testing_cfg):
    cfg = testing_cfg
    output_dir = str(Path(cfg.DATASET.ROOT).parent.absolute())
    download_file_by_url(
        url=url,
        output_directory=output_dir,
        filename="demo_data.zip",
        file_format="zip"
    )
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    dataset = NWayKShotDataset(
        path=cfg.DATASET.ROOT,
        mode=mode,
        k_shot=5,
        query_samples=5,
        transform=transform,
    )
    