"""
Define the ISONet model for the CIFAR datasets.
"""

import kale.predict.isonet as isonet


def get_config(cfg):
    """
    Sets the hypermeters (architecture) for ISONet using the config file

    Args:
        cfg: A YACS config object.
    """
    config_params = {
        "net_params": {
            "use_dirac": cfg.ISON.DIRAC_INIT,
            "use_dropout": cfg.ISON.DROPOUT,
            "dropout_rate": cfg.ISON.DROPOUT_RATE,
            "nc": cfg.DATASET.NUM_CLASSES,
            "depths": cfg.ISON.DEPTH,
            "has_bn": cfg.ISON.HAS_BN,
            "use_srelu": cfg.ISON.SReLU,
            "transfun": cfg.ISON.TRANS_FUN,
            "has_st": cfg.ISON.HAS_ST,
        }
    }
    return config_params


# Inherite and override
class CifarIsoNet(isonet.ISONet):
    """Constructs the ISONet for CIFAR datasets

    Args:
        cfg: A YACS config object.
    """

    def __init__(self, net_params):
        super(CifarIsoNet, self).__init__(net_params)
        # define network structures (override)
        self._construct(net_params)
        # initialization
        self._network_init(net_params["use_dirac"])

    def _construct(self, net_params):
        assert (
            net_params["depths"] - 2
        ) % 6 == 0, "Model depth should be of the format 6n + 2 for cifar"  # Seems because this is a ResNet
        # Each stage has the same number of blocks for cifar

        d = int((net_params["depths"] - 2) / 6)
        # Stem: (N, 3, 32, 32) -> (N, 16, 32, 32)
        self.stem = isonet.ResStem(w_in=3, w_out=16, net_params=net_params, kernelsize=3, stride=1, padding=1)
        # Stage 1: (N, 16, 32, 32) -> (N, 16, 32, 32)
        self.s1 = isonet.ResStage(w_in=16, w_out=16, stride=1, net_params=net_params, d=d)
        # Stage 2: (N, 16, 32, 32) -> (N, 32, 16, 16)
        self.s2 = isonet.ResStage(w_in=16, w_out=32, stride=2, net_params=net_params, d=d)
        # Stage 3: (N, 32, 16, 16) -> (N, 64, 8, 8)
        self.s3 = isonet.ResStage(w_in=32, w_out=64, stride=2, net_params=net_params, d=d)
        # Head: (N, 64, 8, 8) -> (N, num_classes)
        self.head = isonet.ResHead(w_in=64, net_params=net_params)


def get_model(cfg):
    """
    Builds and returns an ISONet model for CIFAR datasets according to the config
    object passed.

    Args:
        cfg: A YACS config object.
    """

    config_params = get_config(cfg)
    net_params = config_params["net_params"]
    net = CifarIsoNet(net_params)
    return net
