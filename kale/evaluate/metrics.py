import torch
from sklearn import metrics


def concord_index(y, y_pred):
    """
    Calculate the Concordance Index (CI), which is a metric to measure the proportion of `concordant pairs
    <https://en.wikipedia.org/wiki/Concordant_pair>`_ between real and
    predict values.

    Args:
        y (array): real values.
        y_pred (array): predicted values.
    """
    total_loss = 0
    pair = 0
    for i in range(1, len(y)):
        for j in range(0, i):
            if i is not j:
                if y[i] > y[j]:
                    pair += 1
                    total_loss += 1 * (y_pred[i] > y_pred[j]) + 0.5 * (y_pred[i] == y_pred[j])

    if pair:
        return total_loss / pair
    else:
        return 0


def auprc_auroc_ap(target: torch.Tensor, score: torch.Tensor):
    """
    auprc: area under the precision-recall curve
    auroc: area under the receiver operating characteristic curve
    ap: average precision

    Copy-paste from https://github.com/NYXFLOWER/GripNet
    """
    y = target.detach().cpu().numpy()
    pred = score.detach().cpu().numpy()
    auroc, ave_precision = metrics.roc_auc_score(y, pred), metrics.average_precision_score(y, pred)
    precision, recall, _ = metrics.precision_recall_curve(y, pred)
    auprc = metrics.auc(recall, precision)

    return auprc, auroc, ave_precision
