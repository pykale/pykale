import pytest
import torch

from kale.predict.isonet import BottleneckTransform, ISONet

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
    net = ISONet(net_params)
    net.eval()
    output_batch = net(INPUT_BATCH)
    assert output_batch.size() == (BATCH_SIZE, net_params["nc"])

    # Ortho penalty
    penalty = net.ortho(device="cpu")
    assert penalty.requires_grad
    assert penalty.item() >= 0


@pytest.mark.parametrize(
    "has_bn,use_srelu,stride,num_gs",
    [
        (True, False, 1, 1),
        (False, True, 2, 1),
        (True, True, 1, 2),
    ],
)
def test_bottleneck_transform_forward(has_bn, use_srelu, stride, num_gs):
    torch.manual_seed(0)
    w_in, w_b, w_out = 8, 4, 16
    model = BottleneckTransform(w_in, w_out, stride, has_bn, use_srelu, w_b, num_gs)
    input_tensor = torch.randn(2, w_in, 32, 32)
    output = model(input_tensor)
    expected_shape = (2, w_out, 32 // stride, 32 // stride)
    assert output.shape == expected_shape
