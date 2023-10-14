"""Domain adaptation systems (pipelines) with three types of architectures

This module takes individual modules as input and organises them into an architecture. This is taken directly from
https://github.com/criteo-research/pytorch-ada/blob/master/adalib/ada/models/architectures.py with minor changes.

This module uses `PyTorch Lightning <https://github.com/Lightning-AI/lightning>`_ to standardize the flow.
"""

from enum import Enum

import numpy as np
import pytorch_lightning as pl
import torch
from torch.autograd import Function

import kale.predict.losses as losses


class GradReverse(Function):
    """The gradient reversal layer (GRL)

    This is defined in the DANN paper http://jmlr.org/papers/volume17/15-239/15-239.pdf

    Forward pass: identity transformation.
    Backward propagation: flip the sign of the gradient.

    From https://github.com/criteo-research/pytorch-ada/blob/master/adalib/ada/models/layers.py
    """

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


def set_requires_grad(model, requires_grad=True):
    """
    Configure whether gradients are required for a model
    """
    for param in model.parameters():
        param.requires_grad = requires_grad


class Method(Enum):
    """
    Lists the available methods.
    Provides a few methods that group the methods by type.
    """

    Source = "Source"
    DANN = "DANN"
    CDAN = "CDAN"
    CDAN_E = "CDAN-E"
    FSDANN = "FSDANN"
    MME = "MME"
    WDGRL = "WDGRL"  # Wasserstein Distance Guided Representation Learning
    WDGRLMod = "WDGRLMod"
    DAN = "DAN"  # Deep Adaptation Networks
    JAN = "JAN"  # Joint Adaptation Networks

    def is_mmd_method(self):
        return self in (Method.DAN, Method.JAN)

    def is_dann_method(self):
        return self in (Method.DANN, Method.Source)

    def is_cdan_method(self):
        return self in (Method.CDAN, Method.CDAN_E)

    def is_fewshot_method(self):
        return self in (Method.FSDANN, Method.MME, Method.Source)

    def allow_supervised(self):
        return self.is_fewshot_method()


def create_mmd_based(method: Method, dataset, feature_extractor, task_classifier, **train_params):
    """MMD-based deep learning methods for domain adaptation: DAN and JAN"""
    if not method.is_mmd_method():
        raise ValueError(f"Unsupported MMD method: {method}")
    if method is Method.DAN:
        return DANTrainer(dataset, feature_extractor, task_classifier, method=method, **train_params)
    if method is Method.JAN:
        return JANTrainer(
            dataset,
            feature_extractor,
            task_classifier,
            method=method,
            kernel_mul=[2.0, 2.0],
            kernel_num=[5, 1],
            **train_params,
        )


def create_dann_like(method: Method, dataset, feature_extractor, task_classifier, critic, **train_params):
    """DANN-based deep learning methods for domain adaptation: DANN, CDAN, CDAN+E"""
    if dataset.is_semi_supervised():
        return create_fewshot_trainer(method, dataset, feature_extractor, task_classifier, critic, **train_params)

    if method.is_dann_method():
        alpha = 0.0 if method is Method.Source else 1.0
        return DANNTrainer(
            alpha=alpha,
            dataset=dataset,
            feature_extractor=feature_extractor,
            task_classifier=task_classifier,
            critic=critic,
            method=method,
            **train_params,
        )
    elif method.is_cdan_method():
        return CDANTrainer(
            dataset=dataset,
            feature_extractor=feature_extractor,
            task_classifier=task_classifier,
            critic=critic,
            method=method,
            use_entropy=method is Method.CDAN_E,
            **train_params,
        )
    elif method is Method.WDGRL:
        return WDGRLTrainer(
            dataset=dataset,
            feature_extractor=feature_extractor,
            task_classifier=task_classifier,
            critic=critic,
            method=method,
            **train_params,
        )
    elif method is Method.WDGRLMod:
        return WDGRLTrainerMod(
            dataset=dataset,
            feature_extractor=feature_extractor,
            task_classifier=task_classifier,
            critic=critic,
            method=method,
            **train_params,
        )
    else:
        raise ValueError(f"Unsupported method: {method}")


def create_fewshot_trainer(method: Method, dataset, feature_extractor, task_classifier, critic, **train_params):
    """DANN-based few-shot deep learning methods for domain adaptation: FSDANN, MME"""
    if not dataset.is_semi_supervised():
        raise ValueError("Dataset must be semi-supervised for few-shot methods.")

    if method.is_fewshot_method():
        alpha = 0 if method is Method.Source else 1
        return FewShotDANNTrainer(
            alpha=alpha,
            dataset=dataset,
            feature_extractor=feature_extractor,
            task_classifier=task_classifier,
            critic=critic,
            method=method,
            **train_params,
        )
    else:
        raise ValueError(f"Unsupported semi-supervised method: {method}")


class BaseAdaptTrainer(pl.LightningModule):
    r"""Base class for all domain adaptation architectures.

    This class implements the classic building blocks used in all the derived architectures
    for domain adaptation.
    If you inherit from this class, you will have to implement only:
        - a forward pass
        - a `compute_loss` function that returns the task loss :math:`\mathcal{L}_c` and adaptation loss
        :math:`\mathcal{L}_a`, as well as a dictionary for summary statistics and other metrics you may want to have
        access to.

    The default training step uses only the task loss :math:`\mathcal{L}_c` during warmup,
    then uses the loss defined as:

    :math:`\mathcal{L} = \mathcal{L}_c + \lambda \mathcal{L}_a`,

    where :math:`\lambda` will follow the schedule defined by the DANN paper:

    :math:`\lambda_p = \frac{2}{1 + \exp{(-\gamma \cdot p)}} - 1` where :math:`p` the learning progress
    changes linearly from 0 to 1.

    Args:
        dataset (kale.loaddata.multi_domain): the multi-domain datasets to be used for train, validation, and tests.
        feature_extractor (torch.nn.Module): the feature extractor network (mapping inputs :math:`x\in\mathcal{X}`
            to a latent space :math:`\mathcal{Z}`,).
        task_classifier (torch.nn.Module): the task classifier network that learns to predict labels
            :math:`y \in \mathcal{Y}` from latent vectors.
        method (Method, optional): the method implemented by the class. Defaults to None.
            Mostly useful when several methods may be implemented using the same class.
        lambda_init (float, optional): weight attributed to the adaptation part of the loss. Defaults to 1.0.
        adapt_lambda (bool, optional): whether to make lambda grow from 0 to 1 following the schedule from
            the DANN paper. Defaults to True.
        adapt_lr (bool, optional): whether to use the schedule for the learning rate as defined
            in the DANN paper. Defaults to True.
        nb_init_epochs (int, optional): number of warmup epochs (during which lambda=0, training only on the source).
            Defaults to 10.
        nb_adapt_epochs (int, optional): number of training epochs. Defaults to 50.
        batch_size (int, optional): defaults to 32.
        init_lr (float, optional): initial learning rate. Defaults to 1e-3.
        optimizer (dict, optional): optimizer parameters, a dictionary with 2 keys:
            "type": a string in ("SGD", "Adam", "AdamW")
            "optim_params": kwargs for the above PyTorch optimizer.
            Defaults to None.
    """

    def __init__(
        self,
        dataset,
        feature_extractor,
        task_classifier,
        method: str = None,
        lambda_init: float = 1.0,
        adapt_lambda: bool = True,
        adapt_lr: bool = True,
        nb_init_epochs: int = 10,
        nb_adapt_epochs: int = 50,
        batch_size: int = 32,
        init_lr: float = 1e-3,
        optimizer: dict = None,
    ):
        super().__init__()
        self._method = method

        self._init_lambda = lambda_init
        self.lamb_da = lambda_init
        self._adapt_lambda = adapt_lambda
        self._adapt_lr = adapt_lr

        self._init_epochs = nb_init_epochs
        self._non_init_epochs = nb_adapt_epochs - self._init_epochs
        assert self._non_init_epochs > 0
        self._batch_size = batch_size
        self._init_lr = init_lr
        self._lr_fact = 1.0
        self._grow_fact = 0.0
        self._dataset = dataset
        self.feat = feature_extractor
        self.classifier = task_classifier
        self._dataset.prepare_data_loaders()
        self._nb_training_batches = None  # to be set by method train_dataloader
        self._optimizer_params = optimizer

    @property
    def method(self):
        return self._method

    def _update_batch_epoch_factors(self, batch_id):
        if self.current_epoch >= self._init_epochs:
            delta_epoch = self.current_epoch - self._init_epochs
            p = (batch_id + delta_epoch * self._nb_training_batches) / (
                self._non_init_epochs * self._nb_training_batches
            )
            self._grow_fact = 2.0 / (1.0 + np.exp(-10 * p)) - 1

            if self._adapt_lr:
                self._lr_fact = 1.0 / ((1.0 + 10 * p) ** 0.75)

        if self._adapt_lambda:
            self.lamb_da = self._init_lambda * self._grow_fact

    def forward(self, x):
        raise NotImplementedError("Forward pass needs to be defined.")

    def compute_loss(self, batch, split_name="valid"):
        """Define the loss of the model

        Args:
            batch (tuple): batches returned by the MultiDomainLoader.
            split_name (str, optional): learning stage (one of ["train", "valid", "test"]).
                Defaults to "valid" for validation. "train" is for training and "test" for testing.
                This is currently used only for naming the metrics used for logging.

        Returns:
            a 3-element tuple with task_loss, adv_loss, log_metrics.
            log_metrics should be a dictionary.

        Raises:
            NotImplementedError: children of this classes should implement this method.
        """
        raise NotImplementedError("Loss needs to be defined.")

    #########################################
    # @profile  # For getting active GPU peak memory. Ignore this when training.
    #########################################
    def training_step(self, batch, batch_nb):
        """The most generic of training steps

        Args:
            batch (tuple): the batch as returned by the MultiDomainLoader dataloader iterator:
                2 tuples: (x_source, y_source), (x_target, y_target) in the unsupervised setting
                3 tuples: (x_source, y_source), (x_target_labeled, y_target_labeled), (x_target_unlabeled, y_target_unlabeled) in the semi-supervised setting
            batch_nb (int): id of the current batch.

        Returns:
            dict: must contain a "loss" key with the loss to be used for back-propagation.
                see pytorch-lightning for more details.
        """
        self._update_batch_epoch_factors(batch_nb)

        task_loss, adv_loss, log_metrics = self.compute_loss(batch, split_name="train")
        if self.current_epoch < self._init_epochs:
            # init phase doesn't use few-shot learning
            # ad-hoc decision but makes models more comparable between each other
            loss = task_loss
        else:
            loss = task_loss + self.lamb_da * adv_loss

        log_metrics["train_total_loss"] = loss
        log_metrics["train_adv_loss"] = adv_loss
        log_metrics["train_task_loss"] = task_loss

        self.log_dict(log_metrics, on_step=True, on_epoch=False)

        # logging alpha and lambda when they exist (they exist for DANN and CDAN but not for DAN and JAN)
        self.log("alpha", self.alpha, on_step=False, on_epoch=True) if hasattr(self, "alpha") else None
        self.log("lambda", self.lamb_da, on_step=False, on_epoch=True) if hasattr(self, "lamb_da") else None

        return loss  # required, for backward pass

    def validation_step(self, batch, batch_nb):
        task_loss, adv_loss, log_metrics = self.compute_loss(batch, split_name="valid")
        loss = task_loss + self.lamb_da * adv_loss
        log_metrics["valid_loss"] = loss
        log_metrics["valid_task_loss"] = task_loss
        log_metrics["valid_adv_loss"] = adv_loss
        self.log_dict(log_metrics, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_nb):
        task_loss, adv_loss, log_metrics = self.compute_loss(batch, split_name="test")
        loss = task_loss + self.lamb_da * adv_loss
        log_metrics["test_loss"] = loss
        log_metrics["test_task_loss"] = task_loss
        log_metrics["test_adv_loss"] = adv_loss
        self.log_dict(log_metrics, on_step=False, on_epoch=True)

    def _configure_optimizer(self, parameters):
        if self._optimizer_params is None:
            optimizer = torch.optim.Adam(parameters, lr=self._init_lr, betas=(0.8, 0.999), weight_decay=1e-5,)
            return [optimizer]
        if self._optimizer_params["type"] == "Adam":
            optimizer = torch.optim.Adam(parameters, lr=self._init_lr, **self._optimizer_params["optim_params"],)
            return [optimizer]
        if self._optimizer_params["type"] == "SGD":
            optimizer = torch.optim.SGD(parameters, lr=self._init_lr, **self._optimizer_params["optim_params"],)

            if self._adapt_lr:
                feature_sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: self._lr_fact)
                return [optimizer], [feature_sched]
            return [optimizer]
        raise NotImplementedError(f"Unknown optimizer type {self._optimizer_params['type']}")

    def configure_optimizers(self):
        return self._configure_optimizer(self.parameters())

    def train_dataloader(self):
        dataloader = self._dataset.get_domain_loaders(split="train", batch_size=self._batch_size)
        self._nb_training_batches = len(dataloader)
        return dataloader

    def val_dataloader(self):
        return self._dataset.get_domain_loaders(split="valid", batch_size=self._batch_size)

    def test_dataloader(self):
        return self._dataset.get_domain_loaders(split="test", batch_size=self._batch_size)


class BaseDANNLike(BaseAdaptTrainer):
    """Common API for DANN-based methods: DANN, CDAN, CDAN+E, WDGRL, MME, FSDANN"""

    def __init__(
        self,
        dataset,
        feature_extractor,
        task_classifier,
        critic,
        alpha=1.0,
        entropy_reg=0.0,  # not used
        adapt_reg=True,  # not used
        batch_reweighting=False,  # not used
        **base_params,
    ):
        super().__init__(dataset, feature_extractor, task_classifier, **base_params)

        self.alpha = alpha

        self._entropy_reg_init = entropy_reg  # not used
        self._entropy_reg = entropy_reg  # not used
        self._adapt_reg = adapt_reg  # not used

        self._reweight_beta = 4  # not used
        self._do_dynamic_batch_weight = batch_reweighting  # not used

        self.domain_classifier = critic

    def _update_batch_epoch_factors(self, batch_id):
        super()._update_batch_epoch_factors(batch_id)
        if self._adapt_reg:
            self._entropy_reg = self._entropy_reg_init * self._grow_fact

    def compute_loss(self, batch, split_name="valid"):
        if len(batch) == 3:
            raise NotImplementedError("DANN does not support semi-supervised setting.")
        (x_s, y_s), (x_tu, y_tu) = batch
        batch_size = len(y_s)

        _, y_hat, d_hat = self.forward(x_s)
        _, y_t_hat, d_t_hat = self.forward(x_tu)

        loss_cls, ok_src = losses.cross_entropy_logits(y_hat, y_s)
        _, ok_tgt = losses.cross_entropy_logits(y_t_hat, y_tu)
        ok_src = ok_src.double().mean()
        ok_tgt = ok_tgt.double().mean()

        loss_dmn_src, dok_src = losses.cross_entropy_logits(d_hat, torch.zeros(batch_size))
        loss_dmn_tgt, dok_tgt = losses.cross_entropy_logits(d_t_hat, torch.ones(batch_size))

        dok = torch.cat((dok_src, dok_tgt)).double().mean()
        dok_src = dok_src.double().mean()
        dok_tgt = dok_tgt.double().mean()

        adv_loss = loss_dmn_src + loss_dmn_tgt
        task_loss = loss_cls

        log_metrics = {
            f"{split_name}_source_acc": ok_src,
            f"{split_name}_target_acc": ok_tgt,
            f"{split_name}_domain_acc": dok,
            f"{split_name}_source_domain_acc": dok_src,
            f"{split_name}_target_domain_acc": dok_tgt,
        }
        return task_loss, adv_loss, log_metrics


class DANNTrainer(BaseDANNLike):
    """
    This class implements the DANN architecture from
    Ganin, Yaroslav, et al.
    "Domain-adversarial training of neural networks."
    The Journal of Machine Learning Research (2016)
    https://arxiv.org/abs/1505.07818

    """

    def __init__(
        self, dataset, feature_extractor, task_classifier, critic, method=None, **base_params,
    ):
        super().__init__(dataset, feature_extractor, task_classifier, critic, **base_params)

        if method is None:
            self._method = Method.DANN
        else:
            self._method = Method(method)
            assert self._method.is_dann_method()

    def forward(self, x):
        if self.feat is not None:
            x = self.feat(x)
        feature = x.view(x.size(0), -1)

        reverse_feature = GradReverse.apply(feature, self.alpha)
        class_output = self.classifier(feature)
        adversarial_output = self.domain_classifier(reverse_feature)
        return feature, class_output, adversarial_output


class CDANTrainer(BaseDANNLike):
    """
    Implements CDAN: Long, Mingsheng, et al. "Conditional adversarial domain adaptation."
    Advances in Neural Information Processing Systems. 2018.
    https://papers.nips.cc/paper/7436-conditional-adversarial-domain-adaptation.pdf
    """

    def __init__(
        self,
        dataset,
        feature_extractor,
        task_classifier,
        critic,
        use_entropy=False,
        use_random=False,
        random_dim=1024,
        **base_params,
    ):
        super().__init__(dataset, feature_extractor, task_classifier, critic, **base_params)
        self.random_layer = None
        self.random_dim = random_dim
        self.entropy = use_entropy
        if use_random:
            nb_inputs = self.feat.output_size() * self.classifier.n_classes()
            self.random_layer = torch.nn.Linear(in_features=nb_inputs, out_features=self.random_dim, bias=False)
            torch.nn.init.normal_(self.random_layer.weight, mean=0, std=1)
            for param in self.random_layer.parameters():
                param.requires_grad = False

    def forward(self, x):
        if self.feat is not None:
            x = self.feat(x)
        x = x.view(x.size(0), -1)

        class_output = self.classifier(x)

        # The GRL hook is applied to all inputs to the adversary
        reverse_feature = GradReverse.apply(x, self.alpha)

        softmax_output = torch.nn.Softmax(dim=1)(class_output)
        reverse_out = GradReverse.apply(softmax_output, self.alpha)

        feature = torch.bmm(reverse_out.unsqueeze(2), reverse_feature.unsqueeze(1))
        feature = feature.view(-1, reverse_out.size(1) * reverse_feature.size(1))
        if self.random_layer:
            random_out = self.random_layer.forward(feature)
            adversarial_output = self.domain_classifier(random_out.view(-1, random_out.size(1)))
        else:
            adversarial_output = self.domain_classifier(feature)

        return x, class_output, adversarial_output

    def _compute_entropy_weights(self, logits):
        entropy = losses.entropy_logits(logits)
        entropy = GradReverse.apply(entropy, self.alpha)
        entropy_w = 1.0 + torch.exp(-entropy)
        return entropy_w

    def compute_loss(self, batch, split_name="valid"):
        if len(batch) == 3:
            raise NotImplementedError("CDAN does not support semi-supervised setting.")
        (x_s, y_s), (x_tu, y_tu) = batch
        batch_size = len(y_s)

        _, y_hat, d_hat = self.forward(x_s)
        _, y_t_hat, d_t_hat = self.forward(x_tu)

        loss_cls, ok_src = losses.cross_entropy_logits(y_hat, y_s)
        _, ok_tgt = losses.cross_entropy_logits(y_t_hat, y_tu)
        ok_src = ok_src.double().mean()
        ok_tgt = ok_tgt.double().mean()

        if self.entropy:
            e_s = self._compute_entropy_weights(y_hat)
            e_t = self._compute_entropy_weights(y_t_hat)
            source_weight = e_s / torch.sum(e_s)
            target_weight = e_t / torch.sum(e_t)
        else:
            source_weight = None
            target_weight = None

        loss_dmn_src, dok_src = losses.cross_entropy_logits(d_hat, torch.zeros(batch_size), source_weight)
        loss_dmn_tgt, dok_tgt = losses.cross_entropy_logits(d_t_hat, torch.ones(len(d_t_hat)), target_weight)

        dok = torch.cat((dok_src, dok_tgt)).double().mean()
        dok_src = dok_src.double().mean()
        dok_tgt = dok_tgt.double().mean()

        adv_loss = loss_dmn_src + loss_dmn_tgt
        task_loss = loss_cls

        log_metrics = {
            f"{split_name}_source_acc": ok_src,
            f"{split_name}_target_acc": ok_tgt,
            f"{split_name}_domain_acc": dok,
            f"{split_name}_source_domain_acc": dok_src,
            f"{split_name}_target_domain_acc": dok_tgt,
        }
        return task_loss, adv_loss, log_metrics


class WDGRLTrainer(BaseDANNLike):
    """
    Implements WDGRL as described in
    Shen, Jian, et al.
    "Wasserstein distance guided representation learning for domain adaptation."
    Thirty-Second AAAI Conference on Artificial Intelligence. 2018.
    https://arxiv.org/pdf/1707.01217.pdf

    This class also implements the asymmetric ($\beta$) variant described in:
    Wu, Yifan, et al.
    "Domain adaptation with asymmetrically-relaxed distribution alignment."
    ICML (2019)
    https://arxiv.org/pdf/1903.01689.pdf
    """

    def __init__(
        self, dataset, feature_extractor, task_classifier, critic, k_critic=5, gamma=10, beta_ratio=0, **base_params,
    ):
        """
        parameters:

            k_critic: number of steps to train critic (called n in Algorithm 1 of the paper)
        """
        super().__init__(dataset, feature_extractor, task_classifier, critic, **base_params)
        self._k_critic = k_critic
        self._beta_ratio = beta_ratio
        self._gamma = gamma

    def forward(self, x):
        if self.feat is not None:
            x = self.feat(x)
        x = x.view(x.size(0), -1)

        class_output = self.classifier(x)
        adversarial_output = self.domain_classifier(x)
        return x, class_output, adversarial_output

    def compute_loss(self, batch, split_name="valid"):
        if len(batch) == 3:
            raise NotImplementedError("WDGRL does not support semi-supervised setting.")
        (x_s, y_s), (x_tu, y_tu) = batch
        batch_size = len(y_s)

        _, y_hat, d_hat = self.forward(x_s)
        _, y_t_hat, d_t_hat = self.forward(x_tu)

        loss_cls, ok_src = losses.cross_entropy_logits(y_hat, y_s)
        _, ok_tgt = losses.cross_entropy_logits(y_t_hat, y_tu)
        ok_src = ok_src.double().mean()
        ok_tgt = ok_tgt.double().mean()

        _, dok_src = losses.cross_entropy_logits(d_hat, torch.zeros(batch_size))
        _, dok_tgt = losses.cross_entropy_logits(d_t_hat, torch.ones(len(d_t_hat)))

        dok = torch.cat((dok_src, dok_tgt)).double().mean()
        dok_src = dok_src.double().mean()
        dok_tgt = dok_tgt.double().mean()

        wasserstein_distance = d_hat.mean() - (1 + self._beta_ratio) * d_t_hat.mean()
        adv_loss = wasserstein_distance
        task_loss = loss_cls

        log_metrics = {
            f"{split_name}_source_acc": ok_src,
            f"{split_name}_target_acc": ok_tgt,
            f"{split_name}_domain_acc": dok,
            f"{split_name}_source_domain_acc": dok_src,
            f"{split_name}_target_domain_acc": dok_tgt,
            f"{split_name}_wasserstein_dist": wasserstein_distance,
        }
        return task_loss, adv_loss, log_metrics

    def critic_update_steps(self, batch):
        if self.current_epoch < self._init_epochs:
            return

        set_requires_grad(self.feat, requires_grad=False)
        set_requires_grad(self.domain_classifier, requires_grad=True)

        (x_s, y_s), (x_tu, _) = batch
        with torch.no_grad():
            h_s = self.feat(x_s).data.view(x_s.shape[0], -1)
            h_t = self.feat(x_tu).data.view(x_tu.shape[0], -1)
        for _ in range(self._k_critic):
            gp = losses.gradient_penalty(self.domain_classifier, h_s, h_t)

            critic_s = self.domain_classifier(h_s)
            critic_t = self.domain_classifier(h_t)
            wasserstein_distance = critic_s.mean() - (1 + self._beta_ratio) * critic_t.mean()

            critic_cost = -wasserstein_distance + self._gamma * gp

            self.critic_opt.zero_grad()
            critic_cost.backward()
            self.critic_opt.step()
            if self.critic_sched:
                self.critic_sched.step()

        set_requires_grad(self.feat, requires_grad=True)
        set_requires_grad(self.domain_classifier, requires_grad=False)

    def training_step(self, batch, batch_id):
        self._update_batch_epoch_factors(batch_id)
        self.critic_update_steps(batch)

        task_loss, adv_loss, log_metrics = self.compute_loss(batch, split_name="train")
        if self.current_epoch < self._init_epochs:
            # init phase doesn't use few-shot learning
            # ad-hoc decision but makes models more comparable between each other
            loss = task_loss
        else:
            loss = task_loss + self.lamb_da * adv_loss

        log_metrics["train_total_loss"] = loss
        log_metrics["train_adv_loss"] = adv_loss
        log_metrics["train_task_loss"] = task_loss

        self.log_dict(log_metrics, on_step=True, on_epoch=False)

        # logging alpha and lambda when they exist (they exist for DANN and CDAN but not for DAN and JAN)
        self.log("alpha", self.alpha, on_step=False, on_epoch=True) if hasattr(self, "alpha") else None
        self.log("lambda", self.lamb_da, on_step=False, on_epoch=True) if hasattr(self, "lamb_da") else None

        return loss  # required, for backward pass

    def configure_optimizers(self):
        nets = [self.feat, self.classifier]
        parameters = set()
        for net in nets:
            parameters |= set(net.parameters())

        if self._adapt_lr:
            task_feat_optimizer, task_feat_sched = self._configure_optimizer(parameters)
            self.critic_opt, self.critic_sched = self._configure_optimizer(self.domain_classifier.parameters())
            self.critic_opt = self.critic_opt[0]
            self.critic_sched = self.critic_sched[0]
            return task_feat_optimizer, task_feat_sched
        else:
            task_feat_optimizer = self._configure_optimizer(parameters)
            self.critic_opt = self._configure_optimizer(self.domain_classifier.parameters())
            self.critic_sched = None
            self.critic_opt = self.critic_opt[0]
        return task_feat_optimizer


class WDGRLTrainerMod(WDGRLTrainer):
    """
    Implements a modified version WDGRL as described in
    Shen, Jian, et al.
    "Wasserstein distance guided representation learning for domain adaptation."
    Thirty-Second AAAI Conference on Artificial Intelligence. 2018.
    https://arxiv.org/pdf/1707.01217.pdf

    This class also implements the asymmetric ($\beta$) variant described in:
    Wu, Yifan, et al.
    "Domain adaptation with asymmetrically-relaxed distribution alignment."
    ICML (2019)
    https://arxiv.org/pdf/1903.01689.pdf
    """

    def __init__(
        self, dataset, feature_extractor, task_classifier, critic, k_critic=5, gamma=10, beta_ratio=0, **base_params,
    ):
        """
        parameters:

            k_critic: number of steps to train critic (called n in Algorithm 1 of the paper)
        """
        super().__init__(dataset, feature_extractor, task_classifier, critic, **base_params)
        self._k_critic = k_critic
        self._beta_ratio = beta_ratio
        self._gamma = gamma
        self.automatic_optimization = False

    def critic_update_steps(self, batch):
        (x_s, y_s), (x_tu, _) = batch
        with torch.no_grad():
            h_s = self.feat(x_s).data.view(x_s.shape[0], -1)
            h_t = self.feat(x_tu).data.view(x_tu.shape[0], -1)

        gp = losses.gradient_penalty(self.domain_classifier, h_s, h_t)

        critic_s = self.domain_classifier(h_s)
        critic_t = self.domain_classifier(h_t)
        wasserstein_distance = critic_s.mean() - (1 + self._beta_ratio) * critic_t.mean()

        critic_cost = -wasserstein_distance + self._gamma * gp

        log_metrics = {"train_critic_loss": critic_cost}

        return {
            "loss": critic_cost,  # required, for backward pass
            "progress_bar": {"critic loss": critic_cost},
            "log": log_metrics,
        }

    # def training_step(self, batch, batch_id, optimizer_idx):
    def training_step(self, batch, batch_id):
        # optimizer_step is not used in the new version of PyTorch Lightning,
        # so we need to implement staged optimizer in training_step.
        # This may casue the implementation to be a little different from the old version.

        self._update_batch_epoch_factors(batch_id)

        # Retrieve the critic and task optimizers
        critic_opt, task_opt = self.optimizers()

        task_loss, adv_loss, log_metrics = self.compute_loss(batch, split_name="train")
        if self.current_epoch < self._init_epochs:
            # init phase doesn't use few-shot learning
            # ad-hoc decision but makes models more comparable between each other
            loss = task_loss

            # do not update critic
            task_opt.step()
            task_opt.zero_grad()
        else:
            loss = task_loss + self.lamb_da * adv_loss

            critic_opt.step()
            critic_opt.zero_grad()

            # update discriminator opt every k_critic steps
            if (batch_id + 1) % self._k_critic == 0:
                task_opt.step()
                task_opt.zero_grad()

        log_metrics["train_total_loss"] = loss
        log_metrics["train_task_loss"] = task_loss

        self.log_dict(log_metrics, on_step=True, on_epoch=False)

        return loss  # required, for backward pass

    # PyTorch Lightning 2.0+ has a different optimizer_step, so we implement this staged optimization in training_step
    # above. We will retain optimizer_step until we have assessed WDGRLTrainerMod, after which it will be removed.
    # Add on_tpu=False etc following https://github.com/PyTorchLightning/pytorch-lightning/issues/2934
    # to fix error for WDGRLMod: TypeError: optimizer_step() got an unexpected keyword argument 'on_tpu'
    # def optimizer_step(
    #     self,
    #     current_epoch,
    #     batch_nb,
    #     optimizer,
    #     # optimizer_i,
    #     second_order_closure=None,
    #     # on_tpu=False,
    #     # using_native_amp=False,
    #     # using_lbfgs=False,
    # ):
    #     if current_epoch < self._init_epochs:
    #         # do not update critic
    #         if optimizer_i == 0:
    #             pass
    #         if optimizer_i == 1:
    #             optimizer.step()
    #             optimizer.zero_grad()
    #     else:
    #         if optimizer_i == 0:
    #             optimizer.step()
    #             optimizer.zero_grad()
    #
    #         # update discriminator opt every k_critic steps
    #         if optimizer_i == 1:
    #             if (batch_nb + 1) % self._k_critic == 0:
    #                 optimizer.step()
    #             optimizer.zero_grad()
    #
    #     optimizer.step(closure=second_order_closure)

    def configure_optimizers(self):
        nets = [self.feat, self.classifier]
        parameters = set()
        for net in nets:
            parameters |= set(net.parameters())

        optimizer = torch.optim.Adam(parameters, lr=self._init_lr, betas=(0.5, 0.999))

        critic_opt = torch.optim.Adam(self.domain_classifier.parameters(), lr=self._init_lr, betas=(0.5, 0.999))
        return [critic_opt, optimizer], []


class FewShotDANNTrainer(BaseDANNLike):
    """Implements adaptations of DANN to the semi-supervised setting

    naive: task classifier is trained on labeled target data, in addition to source
    data.
    MME: immplements Saito, Kuniaki, et al.
    "Semi-supervised domain adaptation via minimax entropy."
    Proceedings of the IEEE International Conference on Computer Vision. 2019
    https://arxiv.org/pdf/1904.06487.pdf

    """

    def __init__(self, dataset, feature_extractor, task_classifier, critic, method, **base_params):
        super().__init__(dataset, feature_extractor, task_classifier, critic, **base_params)
        self._method = Method(method)

    def forward(self, x):
        if self.feat is not None:
            x = self.feat(x)
        x = x.view(x.size(0), -1)

        reverse_feature = GradReverse.apply(x, self.alpha)
        class_output = self.classifier(x)
        adversarial_output = self.domain_classifier(reverse_feature)
        return x, class_output, adversarial_output

    def compute_loss(self, batch, split_name="valid"):
        assert len(batch) == 3
        (x_s, y_s), (x_tl, y_tl), (x_tu, y_tu) = batch
        batch_size = len(y_s)

        _, y_hat, d_hat = self.forward(x_s)
        _, y_tl_hat, d_tl_hat = self.forward(x_tl)
        _, y_tu_hat, d_tu_hat = self.forward(x_tu)
        d_target_pred = torch.cat((d_tl_hat, d_tu_hat))

        loss_cls_s, ok_src = losses.cross_entropy_logits(y_hat, y_s)
        loss_cls_tl, ok_tl = losses.cross_entropy_logits(y_tl_hat, y_tl)
        _, ok_tu = losses.cross_entropy_logits(y_tu_hat, y_tu)
        ok_tgt = torch.cat((ok_tl, ok_tu)).double().mean()
        ok_src = ok_src.double().mean()

        if self.current_epoch < self._init_epochs:
            # init phase doesn't use few-shot learning
            # ad-hoc decision but makes models more comparable between each other
            task_loss = loss_cls_s
        else:
            task_loss = (batch_size * loss_cls_s + len(y_tl) * loss_cls_tl) / (batch_size + len(y_tl))

        loss_dmn_src, dok_src = losses.cross_entropy_logits(d_hat, torch.zeros(batch_size))
        loss_dmn_tgt, dok_tgt = losses.cross_entropy_logits(d_target_pred, torch.ones(len(d_target_pred)))

        if self._method is Method.MME:
            # only keep accuracy, overwrite "domain" loss
            loss_dmn_src = 0
            loss_dmn_tgt = losses.entropy_logits_loss(y_tu_hat)

        adv_loss = loss_dmn_src + loss_dmn_tgt

        dok = torch.cat((dok_src, dok_tgt)).double().mean()
        dok_src = dok_src.double().mean()
        dok_tgt = dok_tgt.double().mean()

        log_metrics = {
            f"{split_name}_source_acc": ok_src,
            f"{split_name}_target_acc": ok_tgt,
            f"{split_name}_domain_acc": dok,
            f"{split_name}_source_domain_acc": dok_src,
            f"{split_name}_target_domain_acc": dok_tgt,
        }
        return task_loss, adv_loss, log_metrics


class BaseMMDLike(BaseAdaptTrainer):
    """Common API for MME-based deep learning DA methods: DAN, JAN"""

    def __init__(
        self, dataset, feature_extractor, task_classifier, kernel_mul=2.0, kernel_num=5, **base_params,
    ):
        super().__init__(dataset, feature_extractor, task_classifier, **base_params)

        self._kernel_mul = kernel_mul
        self._kernel_num = kernel_num

    def forward(self, x):
        if self.feat is not None:
            x = self.feat(x)
        x = x.view(x.size(0), -1)
        class_output = self.classifier(x)
        return x, class_output

    def _compute_mmd(self, phi_s, phi_t, y_hat, y_t_hat):
        raise NotImplementedError("You need to implement a MMD-loss")

    def compute_loss(self, batch, split_name="valid"):
        if len(batch) == 3:
            raise NotImplementedError("MMD does not support semi-supervised setting.")
        (x_s, y_s), (x_tu, y_tu) = batch

        phi_s, y_hat = self.forward(x_s)
        phi_t, y_t_hat = self.forward(x_tu)

        loss_cls, ok_src = losses.cross_entropy_logits(y_hat, y_s)
        _, ok_tgt = losses.cross_entropy_logits(y_t_hat, y_tu)
        ok_src = ok_src.double().mean()
        ok_tgt = ok_tgt.double().mean()

        mmd = self._compute_mmd(phi_s, phi_t, y_hat, y_t_hat)
        task_loss = loss_cls

        log_metrics = {
            f"{split_name}_source_acc": ok_src,
            f"{split_name}_target_acc": ok_tgt,
            f"{split_name}_domain_acc": mmd,
        }
        return task_loss, mmd, log_metrics


class DANTrainer(BaseMMDLike):
    """
    This is an implementation of DAN
    Long, Mingsheng, et al.
    "Learning Transferable Features with Deep Adaptation Networks."
    International Conference on Machine Learning. 2015.
    http://proceedings.mlr.press/v37/long15.pdf
    code based on https://github.com/thuml/Xlearn.
    """

    def __init__(self, dataset, feature_extractor, task_classifier, **base_params):
        super().__init__(dataset, feature_extractor, task_classifier, **base_params)

    def _compute_mmd(self, phi_s, phi_t, y_hat, y_t_hat):
        batch_size = int(phi_s.size()[0])
        kernels = losses.gaussian_kernel(phi_s, phi_t, kernel_mul=self._kernel_mul, kernel_num=self._kernel_num,)
        return losses.compute_mmd_loss(kernels, batch_size)


class JANTrainer(BaseMMDLike):
    """
    This is an implementation of JAN
    Long, Mingsheng, et al.
    "Deep transfer learning with joint adaptation networks."
    International Conference on Machine Learning, 2017.
    https://arxiv.org/pdf/1605.06636.pdf
    code based on https://github.com/thuml/Xlearn.
    """

    def __init__(
        self, dataset, feature_extractor, task_classifier, kernel_mul=(2.0, 2.0), kernel_num=(5, 1), **base_params,
    ):
        super().__init__(
            dataset, feature_extractor, task_classifier, kernel_mul=kernel_mul, kernel_num=kernel_num, **base_params,
        )

    def _compute_mmd(self, phi_s, phi_t, y_hat, y_t_hat):
        softmax_layer = torch.nn.Softmax(dim=-1)
        source_list = [phi_s, softmax_layer(y_hat)]
        target_list = [phi_t, softmax_layer(y_t_hat)]
        batch_size = int(phi_s.size()[0])

        joint_kernels = None
        for source, target, k_mul, k_num, sigma in zip(
            source_list, target_list, self._kernel_mul, self._kernel_num, [None, 1.68]
        ):
            kernels = losses.gaussian_kernel(source, target, kernel_mul=k_mul, kernel_num=k_num, fix_sigma=sigma)
            if joint_kernels is not None:
                joint_kernels = joint_kernels * kernels
            else:
                joint_kernels = kernels

        return losses.compute_mmd_loss(joint_kernels, batch_size)
