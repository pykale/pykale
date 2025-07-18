# =============================================================================
# Author: Shuo Zhou, sz144@outlook.com
#         Sina Tabakhi, sina.tabakhi@gmail.com
#         Peizhen Bai, https://github.com/peizhenbai
#         Mohammod Suvon, m.suvon@sheffield.ac.uk
# =============================================================================

"""Commonly used metrics, losses, and distances, part from domain adaptation package
https://github.com/criteo-research/pytorch-ada/blob/master/adalib/ada/models/losses.py
"""
from enum import Enum

import torch
import torch.nn as nn
from sklearn import metrics
from torch.autograd import grad
from torch.nn import functional as F


def cross_entropy_logits(output, target, weights=None):
    """Computes cross entropy with logits

    Args:
        output (Tensor): The output of the last layer of the network, before softmax.
        target (Tensor): The ground truth label.
        weights (Tensor, optional): The weight of each sample. Defaults to None.

    Returns:
        loss (Tensor): The computed loss value.
        is_correct (Tensor): A tensor of Boolean values. Each True in is_correct indicates a correctly predicted
            label, while each False indicates an incorrect prediction.

    Examples:
        See DANN, WDGRL, and MMD trainers in kale.pipeline.domain_adapter
    """

    class_output = F.log_softmax(output, dim=1)
    max_class = class_output.max(1)
    y_hat = max_class[1]  # get the index of the max log-probability
    is_correct = y_hat.eq(target.view(target.size(0)).type_as(y_hat))
    if weights is None:
        loss = nn.NLLLoss()(class_output, target.type_as(y_hat).view(target.size(0)))
    else:
        losses = nn.NLLLoss(reduction="none")(class_output, target.type_as(y_hat).view(target.size(0)))
        loss = torch.sum(weights * losses) / torch.sum(weights)
    return loss, is_correct


def binary_cross_entropy(output, target):
    """
    Compute binary cross-entropy loss between predicted output and true labels.

    Args:
        output (Tensor): The output of the last layer of the network, before softmax.
        target (Tensor): The ground truth label.
    """
    loss_fct = torch.nn.BCELoss()
    m = nn.Sigmoid()
    n = torch.squeeze(m(output), 1)
    loss = loss_fct(n, target)
    return n, loss


def topk_accuracy(output, target, topk=(1,)):
    """Computes the top-num_att_maps accuracy for the specified values of num_att_maps.

    Args:
        output (Tensor): The output of the last layer of the network, before softmax. Shape: (batch_size, class_count).
        target (Tensor): The ground truth label. Shape: (batch_size)
        topk (tuple(int)): Compute accuracy at top-num_att_maps for the values of num_att_maps specified in this parameter.
    Returns:
        list(Tensor): A list of tensors of the same length as topk.
        Each tensor consists of boolean variables to show if this prediction ranks top num_att_maps with each value of num_att_maps.
        True means the prediction ranks top num_att_maps and False means not.
        The shape of tensor is batch_size, i.e. the number of predictions.

    Examples:
        >>> output = torch.tensor(([0.3, 0.2, 0.1], [0.3, 0.2, 0.1]))
        >>> target = torch.tensor((0, 1))
        >>> top1, top2 = topk_accuracy(output, target, topk=(1, 2)) # get the boolean value
        >>> top1_value = top1.double().mean() # get the top 1 accuracy score
        >>> top2_value = top2.double().mean() # get the top 2 accuracy score
    """

    maxk = max(topk)

    # returns the num_att_maps largest elements and their indexes of inputs along a given dimension.
    _, pred = output.topk(maxk, 1, True, True)

    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    result = []
    for k in topk:
        correct_k = torch.ge(correct[:k].float().sum(0), 1)
        result.append(correct_k)
    return result


def multitask_topk_accuracy(output, target, topk=(1,)):
    """Computes the top-num_att_maps accuracy for the specified values of num_att_maps for multitask input.

    Args:
        output (tuple(Tensor)): A tuple of generated predictions. Each tensor is of shape [batch_size, class_count],
            class_count can vary per task basis, i.e. outputs[i].shape[1] can differ from outputs[j].shape[1].
        target (tuple(Tensor)): A tuple of ground truth. Each tensor is of shape [batch_size]
        topk (tuple(int)): Compute accuracy at top-num_att_maps for the values of num_att_maps specified in this parameter.
    Returns:
        list(Tensor): A list of tensors of the same length as topk.
        Each tensor consists of boolean variables to show if predictions of multitask ranks top num_att_maps with each value of num_att_maps.
        True means predictions of this output for all tasks ranks top num_att_maps and False means not.
        The shape of tensor is batch_size, i.e. the number of predictions.

        Examples:
            >>> first_output = torch.tensor(([0.3, 0.2, 0.1], [0.3, 0.2, 0.1]))
            >>> first_target = torch.tensor((0, 2))
            >>> second_output = torch.tensor(([0.2, 0.1], [0.2, 0.1]))
            >>> second_target = torch.tensor((0, 1))
            >>> output = (first_output, second_output)
            >>> target = (first_target, second_target)
            >>> top1, top2 = multitask_topk_accuracy(output, target, topk=(1, 2)) # get the boolean value
            >>> top1_value = top1.double().mean() # get the top 1 accuracy score
            >>> top2_value = top2.double().mean() # get the top 2 accuracy score
    """

    maxk = max(topk)
    batch_size = target[0].size(0)
    task_count = len(output)
    all_correct = torch.zeros(maxk, batch_size).type(torch.ByteTensor).to(output[0].device)

    for output, target in zip(output, target):
        # returns the num_att_maps largest elements and their indexes of inputs along a given dimension.
        _, pred = output.topk(maxk, 1, True, True)

        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        all_correct.add_(correct)

    result = []
    for k in topk:
        all_correct_k = torch.ge(all_correct[:k].float().sum(0), task_count)
        result.append(all_correct_k)
    return result


def entropy_logits(linear_output):
    """Computes entropy logits in CDAN with entropy conditioning (CDAN+E)

    Examples:
        See CDANTrainer in kale.pipeline.domain_adapter
    """
    p = F.softmax(linear_output, dim=1)
    loss_ent = -torch.sum(p * (torch.log(p + 1e-5)), dim=1)
    return loss_ent


def entropy_logits_loss(linear_output):
    """Computes entropy logits loss in semi-supervised or few-shot domain adaptation

    Examples:
        See FewShotDANNTrainer in kale.pipeline.domain_adapter
    """
    return torch.mean(entropy_logits(linear_output))


def gradient_penalty(critic, h_s, h_t):
    """Computes gradient penalty in Wasserstein distance guided representation learning

    Examples:
        See WDGRLTrainer and WDGRLTrainerMod in kale.pipeline.domain_adapter
    """

    alpha = torch.rand(h_s.size(0), 1)
    alpha = alpha.expand(h_s.size()).type_as(h_s)
    # try:
    differences = h_t - h_s

    interpolates = h_s + (alpha * differences)
    interpolates = torch.cat((interpolates, h_s, h_t), dim=0).requires_grad_()

    preds = critic(interpolates)
    gradients = grad(
        preds,
        interpolates,
        grad_outputs=torch.ones_like(preds),
        retain_graph=True,
        create_graph=True,
    )[0]
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = ((gradient_norm - 1) ** 2).mean()
    # except:
    #     gradient_penalty = 0

    return gradient_penalty


def gaussian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """
    Code from XLearn: computes the full kernel matrix, which is less than optimal since we don't use all of it
    with the linear MMD estimate.

    Examples:
        See DANTrainer and JANTrainer in kale.pipeline.domain_adapter
    """
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    l2_distance = ((total0 - total1) ** 2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(l2_distance.data) / (n_samples**2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-l2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)  # /len(kernel_val)


def compute_mmd_loss(kernel_values, batch_size):
    """Computes the Maximum Mean Discrepancy (MMD) between domains.

    Examples:
        See DANTrainer and JANTrainer in kale.pipeline.domain_adapter
    """
    loss = 0
    for i in range(batch_size):
        s1, s2 = i, (i + 1) % batch_size
        t1, t2 = s1 + batch_size, s2 + batch_size
        loss += kernel_values[s1, s2] + kernel_values[t1, t2]
        loss -= kernel_values[s1, t2] + kernel_values[s2, t1]
    return loss / float(batch_size)


def hsic(kx, ky, device):
    """
    Perform independent test with Hilbert-Schmidt Independence Criterion (HSIC) between two sets of variables x and y.

    Args:
        kx (2-D tensor): kernel matrix of x, shape (n_samples, n_samples)
        ky (2-D tensor): kernel matrix of y, shape (n_samples, n_samples)
        device (torch.device): the desired device of returned tensor

    Returns:
        [tensor]: Independent test score >= 0

    Reference:
        [1] Gretton, Arthur, Bousquet, Olivier, Smola, Alex, and Schölkopf, Bernhard. Measuring Statistical Dependence
            with Hilbert-Schmidt Norms. In Algorithmic Learning Theory (ALT), pp. 63–77. 2005.
        [2] Gretton, Arthur, Fukumizu, Kenji, Teo, Choon H., Song, Le, Schölkopf, Bernhard, and Smola, Alex J. A Kernel
            Statistical Test of Independence. In Advances in Neural Information Processing Systems, pp. 585–592. 2008.
    """

    n = kx.shape[0]
    if ky.shape[0] != n:
        raise ValueError("kx and ky are expected to have the same sample sizes.")
    ctr_mat = torch.eye(n, device=device) - torch.ones((n, n), device=device) / n
    return torch.trace(torch.mm(torch.mm(torch.mm(kx, ctr_mat), ky), ctr_mat)) / (n**2)


def euclidean(x1, x2):
    """
    Compute the Euclidean distance between two sets of variables.

    Args:
        x1 (torch.Tensor): variables set 1
        x2 (torch.Tensor): variables set 2

    Returns:
        torch.Tensor: Euclidean distance
    """
    return ((x1 - x2) ** 2).sum().sqrt()


def _moment_k(x: torch.Tensor, domain_labels: torch.Tensor, k_order=2):
    """Compute the num_att_maps-th moment distance

    Args:
        x (torch.Tensor): input data, shape (n_samples, n_features)
        domain_labels (torch.Tensor): labels indicating which domain the instance is from, shape (n_samples,)
        k_order (int, optional): moment order. Defaults to 2.

    Returns:
        torch.Tensor: the num_att_maps-th moment distance

    The code is based on:
        https://github.com/KaiyangZhou/Dassl.pytorch/blob/master/dassl/engine/da/m3sda.py#L153
        https://github.com/VisionLearningGroup/VisionLearningGroup.github.io/blob/master/M3SDA/code_MSDA_digit/metric/msda.py#L6
    """
    unique_domain_ = torch.unique(domain_labels)
    n_unique_domain_ = len(unique_domain_)
    x_k_order = []
    for domain_label_ in unique_domain_:
        domain_idx = torch.where(domain_labels == domain_label_)[0]
        x_mean = x[domain_idx].mean(0)
        if k_order == 1:
            x_k_order.append(x_mean)
        else:
            x_k_order.append(((x[domain_idx] - x_mean) ** k_order).mean(0))
    moment_sum = 0
    n_pair = 0
    for i in range(n_unique_domain_):
        for j in range(i + 1, n_unique_domain_):
            moment_sum += euclidean(x_k_order[i], x_k_order[j])
            n_pair += 1
    return moment_sum / n_pair


class protonet_loss:
    """ProtoNet loss function.

    This is a loss function for prototypical networks. It computes the loss and accuracy of the model by measuring the Euclidean distance between features of samples in support and query sets.
    Because this loss requests some constant hyperparameters, it is not a general function but defined as a class.

    - :math:`N`-way: The number of classes under a particular setting. The model is presented with samples from these :math:`N` classes and needs to classify them. For example, 3-way means the model has to classify 3 different classes.

    - :math:`K`-shot: The number of samples for each class in the support set. For example, in a 2-shot setting, two support samples are provided per class.

    - Support set: It is a small, labeled dataset used to train the model with a few samples of each class. The support set consists of :math:`N` classes (:math:`N`-way), with :math:`K` samples (:math:`K`-shot) for each class. For example, under a 3-way-2-shot setting, the support set has 3 classes with 2 samples per class, totaling 6 samples.

    - Query set: It evaluates the model's ability to generalize what it has learned from the support set. It contains samples from the same :math:`N` classes but not included in the support set. Continuing with the 3-way-2-shot example, the query set would include additional samples from the 3 classes, which the model must classify after learning from the support set.

    Args:
        num_classes (int): Number of classes in a task. Default: 5
        num_query_samples (int): Number of samples per class in the query set. Default: 15
        device (torch.device): The device in computation. Default: torch.device("cuda")

    Examples:
        >>> loss_fn = protonet_loss(num_classes=5, num_query_samples=15, device=torch.device("cuda"))
        >>> loss, acc = loss_fn(feature_support, feature_query)

    Reference:
        Snell, J., Swersky, K. and Zemel, R., 2017. Prototypical networks for few-shot learning. Advances in Neural Information Processing Systems, 30.
    """

    def __init__(
        self, num_classes: int = 5, num_query_samples: int = 15, device: torch.device = torch.device("cuda")
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_query_samples = num_query_samples
        self.device = device

    def __call__(self, feature_support: torch.Tensor, feature_query: torch.Tensor) -> tuple:
        """
        Args:
            feature_support (torch.Tensor): Feature vectors of support samples: (num_classes, k_shot, feature_dim)
            feature_query (torch.Tensor): Feature vectors of query samples: (num_classes * num_query_samples, feature_dim)

        Returns:
            tuple: (loss, acc)
            loss (torch.Tensor): Loss value
            acc (torch.Tensor): Accuracy value
        """
        feature_support = feature_support.to(self.device)
        feature_query = feature_query.to(self.device)
        prototypes = feature_support.mean(dim=1)
        dists = self.euclidean_dist_for_tensor_group(feature_query, prototypes)
        log_p_y = F.log_softmax(-dists, dim=1)
        log_p_y = log_p_y.view(self.num_classes, self.num_query_samples, -1)
        labels = torch.arange(self.num_classes).to(self.device)
        labels = labels.view(self.num_classes, 1, 1)
        labels = labels.expand(self.num_classes, self.num_query_samples, 1).long()
        loss = -log_p_y.gather(2, labels).squeeze().view(-1).mean()
        _, y_hat = log_p_y.max(2)
        acc = torch.eq(y_hat, labels.squeeze()).float().mean()
        return loss, acc

    def euclidean_dist_for_tensor_group(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the Euclidean distance between two batches.

        Args:
            x (torch.Tensor): Variables set 1: (N, D)
            y (torch.Tensor): Variables set 2: (M, D)

        Returns:
            torch.Tensor: Euclidean distance: (N, M)
        """
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        if d != y.size(1):
            raise Exception
        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)
        return torch.pow(x - y, 2).sum(2)


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


class DistanceMetric(Enum):
    COSINE = "COSINE"


def calculate_distance(
    x1: torch.Tensor, x2: torch.Tensor = None, eps: float = 1e-8, metric: DistanceMetric = DistanceMetric.COSINE
) -> torch.Tensor:
    r"""Returns similarity between :math:`x_1` and :math:`x_2`, computed along `dim`=1. This method calculates the
    similarity between each pair of data points in two input matrices.

    Note that this implementation differs from the existing implementations in
    `PyTorch <https://pytorch.org/docs/stable/generated/torch.nn.CosineSimilarity.html>`__, as they calculate the
    similarity between each row of one matrix with its corresponding row in the other matrix (i.e., pairwise distance
    between columns of input matrices).

    Args:
        x1 (torch.Tensor): The tensor input data.
        x2 (torch.Tensor, optional): The tensor input data. (default ``None``)
        eps (float, optional): Small value to avoid division by zero. (default: 1e-8)
        metric (DistanceMetric, optional): The metric to compute distance between input matrices. (default: ``DistanceMetric.COSINE``)

        Returns:
        torch.Tensor: The computed similarity tensor between :math:`x_1` and :math:`x_2`.
    """
    if metric == DistanceMetric.COSINE:
        x2 = x1 if x2 is None else x2
        w1 = torch.norm(x1, p=2, dim=1, keepdim=True)
        w2 = w1 if x2 is None else torch.norm(x2, p=2, dim=1, keepdim=True)
        return torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)

    raise Exception("This metric is not still implemented")


def signal_image_elbo_loss(
    recon_image,
    target_image,
    recon_signal,
    target_signal,
    mean,
    log_var,
    lambda_image=1.0,
    lambda_signal=1.0,
    annealing_factor=1.0,
    scale_factor=1e-4,
):
    """
    Computes a multimodal ELBO loss for VAE with image and signal modalities.
    """
    eps = 1e-8
    image_mse = 0.0
    signal_mse = 0.0

    if recon_image is not None and target_image is not None:
        image_mse = F.mse_loss(recon_image, target_image, reduction="sum") * lambda_image
    if recon_signal is not None and target_signal is not None:
        signal_mse = F.mse_loss(recon_signal, target_signal, reduction="sum") * lambda_signal

    recon_loss = (image_mse + signal_mse) * scale_factor
    log_var = torch.clamp(log_var, min=-10, max=10)
    kl_div = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp() + eps, dim=1)
    kl_div = kl_div.sum()
    loss = recon_loss + annealing_factor * kl_div
    return loss
