import numpy as np
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from kale.evaluate import cross_validation
from kale.pipeline.multi_domain_adapter import CoIRLS
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from ..helpers.toy_dataset import make_domain_shifted_dataset


@pytest.fixture(scope="module")
def sample_data():
    # Sample data for testing
    X, y, groups = make_domain_shifted_dataset(
        n_domains=2,
        n_samples_per_class=100,
        n_features=40,
        class_sep=1.0,
        centroid_shift_scale=5.0,
        random_state=0,
    )

    # Convert labels to strings for testing
    groups = np.where(groups == 0, "A", "B")
    return X, y, groups


# Checks leave-one-group-out results for above sample data
def check_leave_one_group_out_results(sample_data, estimator, domain_adaptation=False):
    x, y, groups = sample_data
    results = cross_validation.leave_one_group_out(x, y, groups, estimator, domain_adaptation)

    # Check if all keys are present in the result dictionary
    assert "Target" in results
    assert "Num_samples" in results
    assert "Accuracy" in results

    # Check if the length of 'Target', 'Num_samples', and 'Accuracy' lists is the same
    assert len(results["Target"]) == len(results["Num_samples"]) == len(results["Accuracy"])

    # Check if the last element of 'Target' is "Average"
    assert results["Target"] == ["A", "B", "Average"]

    # Check if the number of samples is correct for each target group
    assert results["Num_samples"] == [200, 200, 400]

    # Check if the accuracy scores are within the expected range
    for accuracy in results["Accuracy"]:
        assert 0 <= accuracy <= 1


def test_leave_one_group_out_without_domain_adaptation(sample_data):
    estimator = DummyClassifier()
    check_leave_one_group_out_results(sample_data, estimator, domain_adaptation=False)


def test_leave_one_group_out_with_domain_adaptation(sample_data):
    estimator = CoIRLS(kernel="linear", lambda_=1.0, alpha=1.0)
    check_leave_one_group_out_results(sample_data, estimator, domain_adaptation=True)


@pytest.mark.parametrize("cv", [5, 10, LeaveOneGroupOut()])
@pytest.mark.parametrize("estimator", [DummyClassifier(), LogisticRegression()])
@pytest.mark.parametrize("transformer", [None, StandardScaler(), PCA()])
@pytest.mark.parametrize("scoring", ["accuracy", "f1", "roc_auc", ["accuracy", "f1", "roc_auc"]])
def test_cross_validate(sample_data, cv, estimator, transformer, scoring):
    X, y, groups = sample_data

    factors = OneHotEncoder(sparse_output=False).fit_transform(groups.reshape(-1, 1))

    fit_args = None
    if isinstance(estimator, CoIRLS):
        fit_args = {"covariates": factors}

    results = cross_validation.cross_validate(
        estimator,
        X,
        y,
        groups=groups,
        cv=cv,
        transformer=transformer,
        scoring=scoring,
        fit_args=fit_args,
        error_score="raise",
    )

    print(results)
