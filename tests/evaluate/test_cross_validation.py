import numpy as np
import pytest
from sklearn.dummy import DummyClassifier

from kale.evaluate import cross_validation
from kale.pipeline.multi_domain_adapter import CoIRLS


@pytest.fixture
def sample_data():
    # Sample data for testing
    x = np.array([np.random.rand(100)] * 8)
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    groups = np.array(["A", "A", "B", "B", "A", "A", "B", "B"])
    return x, y, groups


def test_leave_one_group_out_without_domain_adaptation(sample_data):
    x, y, groups = sample_data
    estimator = DummyClassifier()
    result = cross_validation.leave_one_group_out(x, y, groups, estimator, domain_adaptation=False)

    # Check if all keys are present in the result dictionary
    assert "Target" in result
    assert "Num_samples" in result
    assert "Accuracy" in result

    # Check if the length of 'Target', 'Num_samples', and 'Accuracy' lists is the same
    assert len(result["Target"]) == len(result["Num_samples"]) == len(result["Accuracy"])

    # Check if the 'Target' list contains correct groups
    assert result["Target"] == ["A", "B", "Average"]

    # Check if the number of samples is correct for each target group
    assert result["Num_samples"] == [4, 4, 8]

    # Check if the accuracy scores are within the expected range
    for accuracy in result["Accuracy"]:
        assert 0 <= accuracy <= 1


def test_leave_one_group_out_with_domain_adaptation(sample_data):
    x, y, groups = sample_data
    estimator = CoIRLS(kernel="linear", lambda_=1.0, alpha=1.0)
    result = cross_validation.leave_one_group_out(x, y, groups, estimator, domain_adaptation=True)

    # Check if all keys are present in the result dictionary
    assert "Target" in result
    assert "Num_samples" in result
    assert "Accuracy" in result

    # Check if the length of 'Target', 'Num_samples', and 'Accuracy' lists is the same
    assert len(result["Target"]) == len(result["Num_samples"]) == len(result["Accuracy"])

    # Check if the 'Target' list contains correct groups
    assert result["Target"] == ["A", "B", "Average"]

    # Check if the number of samples is correct for each target group
    assert result["Num_samples"] == [4, 4, 8]

    # Check if the accuracy scores are within the expected range
    for accuracy in result["Accuracy"]:
        assert 0 <= accuracy <= 1
