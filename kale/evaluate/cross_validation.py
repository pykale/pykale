import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder


def leave_one_out_cross_validate(x, y, covariates, estimator, domain_adaptation=False) -> dict:
    """
    Perform leave-one-out cross validation (LOOCV) for a given estimator.

    Args:
        x: Input data [n_samples, n_features].
        y: Target labels [n_samples].
        covariates: Covariates indicating classes for LOOCV [n_samples].
        estimator: Machine learning estimator to be evaluated from kale or scikit-learn.
        domain_adaptation: Whether to use domain adaptation during training.

    Returns:
        dict: A dictionary containing LOOCV results for each target group with 3 keys.

            - 'Target': List of unique target groups or classes.
            - 'Num_samples': List containing number of samples in each target group.
            - 'Accuracy': List of accuracy scores for each target group.
    """
    target, num_samples, accuracy = [], [], []
    enc = OneHotEncoder(handle_unknown="ignore")
    covariate_mat = enc.fit_transform(covariates.reshape(-1, 1)).toarray()
    unique_covariates = np.unique(covariates)

    for tgt in unique_covariates:
        idx_tgt = np.where(covariates == tgt)
        idx_src = np.where(covariates != tgt)
        x_tgt = x[idx_tgt]
        x_src = x[idx_src]
        y_tgt = y[idx_tgt]
        y_src = y[idx_src]

        if domain_adaptation:
            estimator.fit(
                np.concatenate((x_src, x_tgt)), y_src, np.concatenate((covariate_mat[idx_src], covariate_mat[idx_tgt]))
            )
        else:
            estimator.fit(x_src, y_src)
        y_pred = estimator.predict(x_tgt)
        target.append(tgt)
        num_samples.append(x_tgt.shape[0])
        accuracy.append(accuracy_score(y_tgt, y_pred))

    mean_acc = sum([num_samples[i] * accuracy[i] for i in range(len(unique_covariates))]) / x.shape[0]
    target.append("Average")
    num_samples.append(x.shape[0])
    accuracy.append(mean_acc)

    return {
        "Target": target,
        "Num_samples": num_samples,
        "Accuracy": accuracy,
    }
