import matplotlib.figure
import numpy as np
import pytest
from numpy import testing
from sklearn.metrics import accuracy_score, roc_auc_score

from kale.interpret import model_weights, visualize
from kale.pipeline.mpca_trainer import MPCATrainer

CLASSIFIERS = ["svc", "linear_svc", "lr"]
PARAMS = [
    {"classifier_params": "auto", "mpca_params": None, "n_features": None, "search_params": None},
    {
        "classifier_params": {"C": 1},
        "mpca_params": {"var_ratio": 0.9, "vectorize": True},
        "n_features": 100,
        "search_params": {"cv": 3},
    },
]


@pytest.mark.parametrize("classifier", CLASSIFIERS)
@pytest.mark.parametrize("params", PARAMS)
def test_mpca_trainer(classifier, params, gait):
    x = gait["fea3D"].transpose((3, 0, 1, 2))
    x = x[:20, :]
    y = gait["gnd"][:20].reshape(-1)
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

    if classifier == "svc" and trainer.clf.kernel == "rbf":
        with pytest.raises(Exception):
            trainer.mpca.inverse_transform(trainer.clf.coef_)
    else:
        weights = trainer.mpca.inverse_transform(trainer.clf.coef_) - trainer.mpca.mean_
        top_weights = model_weights.select_top_weight(weights, select_ratio=0.1)
        fig = visualize.plot_weights(top_weights[0][0], background_img=x[0][0])
        assert type(fig) == matplotlib.figure.Figure


def test_invalid_init():
    with pytest.raises(Exception):
        MPCATrainer(classifier="Ridge")
    with pytest.raises(Exception):
        MPCATrainer(classifier_params=False)
