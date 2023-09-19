"""
Functions implementing cross-validation methods for assessing model fit.
"""

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import OneHotEncoder


def leave_one_group_out(x, y, groups, estimator, domain_adaptation=False) -> dict:
    """
    Perform leave one group out cross validation for a given estimator.

    Args:
        x: Input data [n_samples, n_features].
        y: Target labels [n_samples].
        groups: Group labels to be left out [n_samples].
        estimator: Machine learning estimator to be evaluated from kale or scikit-learn.
        domain_adaptation: Whether to use domain adaptation during training.

    Returns:
        dict: A dictionary containing results for each target group with 3 keys.

            - 'Target': List of unique target groups or classes.
            - 'Num_samples': List containing number of samples in each target group.
            - 'Accuracy': List of accuracy scores for each target group.
    """
    enc = OneHotEncoder(handle_unknown="ignore")
    group_mat = enc.fit_transform(groups.reshape(-1, 1)).toarray()
    target, num_samples, accuracy = np.unique(groups).tolist(), [], []

    for train, test in LeaveOneGroupOut().split(x, y, groups=groups):
        x_src, x_tgt = x[train], x[test]
        y_src, y_tgt = y[train], y[test]

        if domain_adaptation:
            estimator.fit(np.concatenate((x_src, x_tgt)), y_src, np.concatenate((group_mat[train], group_mat[test])))
        else:
            estimator.fit(x_src, y_src)

        y_pred = estimator.predict(x_tgt)
        num_samples.append(x_tgt.shape[0])
        accuracy.append(accuracy_score(y_tgt, y_pred))

    mean_acc = sum([num_samples[i] * accuracy[i] for i in range(len(target))]) / x.shape[0]
    target.append("Average")
    num_samples.append(x.shape[0])
    accuracy.append(mean_acc)

    return {
        "Target": target,
        "Num_samples": num_samples,
        "Accuracy": accuracy,
    }
