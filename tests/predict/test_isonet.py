import pytest
import torch

import kale.predict.isonet as isonet

TRANS_FUNS = ["basic_transform"]  # "bottleneck_transform"]


@pytest.fixture(scope="module")
def testing_cfg(download_path):
    config_params = {
        "net_params": {
            "use_dirac": True,
            "use_dropout": False,
            "dropout_rate": 0.0,
            "nc": 10,
            "depths": 34,
            "has_bn": False,
            "use_srelu": True,
            "transfun": "basic_transform",
            "has_st": False,
        }
    }
    yield config_params


@pytest.mark.parametrize("transfun", TRANS_FUNS)
def test_isonet(transfun, testing_cfg):

    net_params = testing_cfg["net_params"]
    net_params["transfun"] = transfun
    net = isonet.ISONet(net_params)

    assert isinstance(net, torch.nn.Module)
