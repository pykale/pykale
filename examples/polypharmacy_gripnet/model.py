from yacs.config import CfgNode

from kale.predict.decode import GripNetLinkPrediction
from kale.prepdata.supergraph_construct import SuperGraph, SuperVertexParaSetting


def get_supervertex(sv_configs: CfgNode) -> SuperVertexParaSetting:
    """Get supervertex parameter setting from configurations."""

    exter_list = sv_configs.EXTER_AGG_CHANNELS_LIST

    if len(exter_list):
        exter_dict = {k: v for k, v in exter_list}

        return SuperVertexParaSetting(
            sv_configs.NAME,
            sv_configs.INTER_FEAT_CHANNELS,
            sv_configs.INTER_AGG_CHANNELS_LIST,
            exter_agg_channels_dict=exter_dict,
            mode=sv_configs.MODE,
        )

    return SuperVertexParaSetting(sv_configs.NAME, sv_configs.INTER_FEAT_CHANNELS, sv_configs.INTER_AGG_CHANNELS_LIST,)


def get_model(supergraph: SuperGraph, cfg: CfgNode) -> GripNetLinkPrediction:
    """Get model from the supergraph and configurations."""

    learning_rate = cfg.SOLVER.BASE_LR
    epsilon = cfg.SOLVER.EPSILON

    return GripNetLinkPrediction(supergraph, learning_rate, epsilon)
