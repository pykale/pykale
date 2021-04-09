import numpy as np
import pytest
from numpy import testing
from scipy.io import loadmat

from kale.pipeline.mpca_trainer import MPCATrainer

gait = loadmat("../test_data/gait_gallery_data.mat")
x = gait["fea3D"].transpose((3, 0, 1, 2))
x = x[:20, :]
y = gait["gnd"][:20]

CLASSIFIERS = ["svc", "lr"]
CLASSIFIER_PARAMS = ["auto", {"C": 1}]
MPCA_PARAMS = [None, {"var_ratio": 0.9, "return_vector": True}]
N_FEATURES = [None, 100]
SEARCH_PARAMS = [None, {'cv': 3}]


@pytest.mark.parametrize("classifier", CLASSIFIERS)
@pytest.mark.parametrize("classifier_params", CLASSIFIER_PARAMS)
@pytest.mark.parametrize("mpca_params", MPCA_PARAMS)
@pytest.mark.parametrize("n_features", N_FEATURES)
@pytest.mark.parametrize("search_params", SEARCH_PARAMS)
def test_mpca_trainer(classifier, classifier_params, mpca_params, n_features, search_params):
    trainer = MPCATrainer(classifier=classifier, classifier_params=classifier_params, mpca_params=mpca_params,
                          n_features=n_features, search_params=search_params)
    trainer.fit(x, y)
    y_pred = trainer.predict(x)
    testing.assert_equal(np.unique(y), np.unique(y_pred))

    y_proba = trainer.predict_proba(x)
    assert np.max(y_proba) <= 1.0
    assert np.min(y_proba) >= 0.0

    y_dec_score = trainer.decision_function(x)
    testing.assert_equal(y_dec_score.shape, (x.shape[0],))

