import numpy as np
import pytest
from numpy import testing
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

from kale.pipeline.mida_trainer import MIDATrainer


@pytest.fixture(scope="module")
def toy_data():
    """Generate two toy sets, shape (200, 3), from two distributions for testing."""
    np.random.seed(29118)
    # Generate toy data
    n_samples = 200

    xs, ys = make_blobs(n_samples, n_features=3, centers=[[0, 0, 0], [0, 2, 1]], cluster_std=[0.3, 0.35])
    xt, yt = make_blobs(n_samples, n_features=3, centers=[[2, -2, 2], [2, 0.2, -1]], cluster_std=[0.35, 0.4])

    groups = np.zeros(n_samples * 2)
    groups[:n_samples] = 1

    return xs, ys, xt, yt, groups


TRANSFORMER = [None, MinMaxScaler()]
ESTIMATOR_PARAM_GRID = [None, {"C": [0.1, 1.0]}]
MIDA_PARAM_GRID = [None, {"fit_label": [True], "augmentation": [True]}]


@pytest.mark.parametrize("transformer", TRANSFORMER)
@pytest.mark.parametrize("estimator_param_grid", ESTIMATOR_PARAM_GRID)
@pytest.mark.parametrize("mida_param_grid", MIDA_PARAM_GRID)
def test_mida_trainer(transformer, estimator_param_grid, mida_param_grid, toy_data):
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
    n_samples = yt.shape[0] + ys.shape[0]
    y_pred = mida_trainer.fit_predict(x, ys, groups=groups)
    y_pred_proba = mida_trainer.predict_proba(x, groups=groups)
    y_score = mida_trainer.decision_function(x, groups=groups)
    testing.assert_equal(y_pred.shape[0], n_samples)
    testing.assert_equal(y_pred_proba.shape[0], n_samples)
    testing.assert_equal(y_score.shape[0], n_samples)
