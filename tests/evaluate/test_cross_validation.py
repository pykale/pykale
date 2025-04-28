import numpy as np
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from kale.embed.factorization import MIDA
from kale.evaluate import cross_validation
from kale.pipeline.multi_domain_adapter import CoIRLS

from ..helpers.toy_dataset import make_domain_shifted_dataset


@pytest.fixture(scope="module")
def sample_data():
    # Sample data for testing
    x, y, groups = make_domain_shifted_dataset(
        num_domains=2,
        num_samples_per_class=100,
        num_features=40,
        class_sep=1.0,
        centroid_shift_scale=5.0,
        random_state=0,
    )

    # Convert labels to strings for testing
    groups = np.where(groups == 0, "A", "B")
    factors = OneHotEncoder(sparse_output=False).fit_transform(groups.reshape(-1, 1))

    return x, y, groups, factors


# Checks leave-one-group-out results for above sample data
def check_leave_one_group_out_results(sample_data, estimator, domain_adaptation=False):
    x, y, groups, _ = sample_data
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


@pytest.mark.parametrize("cv", [3, LeaveOneGroupOut()])
@pytest.mark.parametrize("transformer", [None, StandardScaler()])
@pytest.mark.parametrize("domain_adapter", [None, MIDA()])
@pytest.mark.parametrize("scoring", ["accuracy", "f1", "roc_auc", ["accuracy", "f1", "roc_auc"]])
@pytest.mark.parametrize("return_indices", [True, False])
@pytest.mark.parametrize("return_train_score", [True, False])
@pytest.mark.parametrize("return_estimator", [True, False])
def test_cross_validate(
    sample_data,
    cv,
    transformer,
    domain_adapter,
    scoring,
    return_indices,
    return_train_score,
    return_estimator,
):
    estimator = DummyClassifier()

    x, y, groups, factors = sample_data

    fit_args = None
    if isinstance(estimator, CoIRLS):
        fit_args = {"covariates": factors}

    results = cross_validation.cross_validate(
        estimator,
        x,
        y,
        groups=groups,
        cv=cv,
        transformer=transformer,
        domain_adapter=domain_adapter,
        factors=factors,
        scoring=scoring,
        fit_args=fit_args,
        return_indices=return_indices,
        return_train_score=return_train_score,
        error_score="raise",
        return_estimator=return_estimator,
    )

    if isinstance(scoring, str):
        scoring = ["score"]

    assert "fit_time" in results, "fit_time not in results"
    assert "score_time" in results, "score_time not in results"
    assert "indices" in results if return_indices else True, "indices not in results"
    assert any(f"test_{score}" in results for score in scoring), f"Expected at least one of {scoring} in results"
    assert (
        any(f"train_{score}" in results for score in scoring) if return_train_score else True
    ), "train_score not in results"
    assert "estimator" in results if return_estimator else True, "estimator not in results"
