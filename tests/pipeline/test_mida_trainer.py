import numpy as np
import pytest
from numpy import testing
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

from kale.pipeline.mida_trainer import MIDATrainer

N_SAMPLES = 200


@pytest.fixture(scope="module")
def toy_data():
    np.random.seed(29118)
    # Generate toy data
    n_samples = N_SAMPLES

    xs, ys = make_blobs(n_samples, n_features=3, centers=[[0, 0, 0], [0, 2, 1]], cluster_std=[0.3, 0.35])
    xt, yt = make_blobs(n_samples, n_features=3, centers=[[2, -2, 2], [2, 0.2, -1]], cluster_std=[0.35, 0.4])

    groups = np.zeros(n_samples * 2)
    groups[:n_samples] = 1

    return xs, ys, xt, yt, groups


@pytest.mark.parametrize("transformer", [None, MinMaxScaler()])
@pytest.mark.parametrize("estimator_param_grid", [None, {"C": [0.1, 1.0]}])
# @pytest.mark.parametrize("mida_param_grid", [None, {"eta": [0.1, 10.], "mu": [0.1, 10.]}])
@pytest.mark.parametrize("mida_param_grid", [None])
def test_mida_trainer_transformer(transformer, estimator_param_grid, mida_param_grid, toy_data):
    """

    Args:
        toy_data:

    Returns:

    """
    estimator = LogisticRegression()
    xs, ys, xt, yt, groups = toy_data
    x = np.concatenate([xs, xt], axis=0)
    if transformer is not None:
        transformer_param_grid = {"feature_range": [(-2, 2)]}
    else:
        transformer_param_grid = None
    mida_trainer = MIDATrainer(
        estimator=estimator,
        transformer=transformer,
        mida_param_grid=mida_param_grid,
        transformer_param_grid=transformer_param_grid,
        estimator_param_grid=estimator_param_grid,
    )
    mida_trainer.fit(x, ys, groups=groups)

    y_pred = mida_trainer.predict(xt)
    testing.assert_equal(y_pred.shape, yt.shape)
