import pytest
import torch

import kale.predict.isonet as isonet

BATCH_SIZE = 2
INPUT_BATCH = torch.randn(BATCH_SIZE, 3, 224, 224)
TRANS_FUNS = ["basic_transform"]  # "bottleneck_transform"]


@pytest.fixture(scope="module")
def testing_cfg(download_path):
    config_params = {
        "net_params": {
            "use_dirac": True,
            "use_dropout": True,
            "dropout_rate": 0.0,
            "nc": 10,
            "depths": 34,
            "has_bn": True,
            "use_srelu": True,
            "transfun": "basic_transform",
            "has_st": True,
        }
    }
    yield config_params


@pytest.mark.parametrize("transfun", TRANS_FUNS)
def test_isonet(transfun, testing_cfg):
    net_params = testing_cfg["net_params"]
    net_params["transfun"] = transfun
    net = isonet.ISONet(net_params)
    net.eval()
    output_batch = net(INPUT_BATCH)
    assert output_batch.size() == (BATCH_SIZE, net_params["nc"])
