import numpy as np
import pytest
from numpy import testing
from scipy.io import loadmat
from sklearn.metrics import accuracy_score, roc_auc_score

from kale.pipeline.mpca_trainer import MPCATrainer

gait = loadmat("tests/test_data/gait_gallery_data.mat")
x = gait["fea3D"].transpose((3, 0, 1, 2))
x = x[:20, :]
y = gait["gnd"][:20].reshape(-1)

CLASSIFIERS = ["svc", "linear_svc", "lr"]
PARAMS = [
    {"classifier_params": "auto", "mpca_params": None, "n_features": None, "search_params": None},
    {
        "classifier_params": {"C": 1},
        "mpca_params": {"var_ratio": 0.9, "return_vector": True},
        "n_features": 100,
        "search_params": {"cv": 3},
    },
]


@pytest.mark.parametrize("classifier", CLASSIFIERS)
@pytest.mark.parametrize("params", PARAMS)
def test_mpca_trainer(classifier, params):
    trainer = MPCATrainer(classifier=classifier, **params)
    trainer.fit(x, y)
    y_pred = trainer.predict(x)
    testing.assert_equal(np.unique(y), np.unique(y_pred))
    assert accuracy_score(y, y_pred) >= 0.8

    if classifier == "linear_svc":
        with pytest.raises(Exception):
            y_proba = trainer.predict_proba(x)
    else:
        y_proba = trainer.predict_proba(x)
        assert np.max(y_proba) <= 1.0
        assert np.min(y_proba) >= 0.0
        y_ = np.zeros(y.shape)
        y_[np.where(y == 1)] = 1
        assert roc_auc_score(y_, y_proba[:, 0]) >= 0.8

    y_dec_score = trainer.decision_function(x)
    assert roc_auc_score(y, y_dec_score) >= 0.8


def test_invalid_init():
    with pytest.raises(Exception):
        MPCATrainer(classifier="Ridge")
    with pytest.raises(Exception):
        MPCATrainer(classifier_params=False)
