# =============================================================================
# Author: Jiayang Zhang, jiayang.zhang@sheffield.ac.uk
# =============================================================================

"""
This module contains the DrugBAN trainer class and related functions. It trains a Interpretable bilinear attention
network with or without domain adaptation model for drug-target interaction prediction.

This is refactored from: https://github.com/peizhenbai/DrugBAN/blob/main/trainer.py
"""


import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchmetrics import Accuracy, AUROC, F1Score, MetricCollection, Recall, Specificity

from kale.embed.ban import RandomLayer
from kale.evaluate.metrics import binary_cross_entropy, cross_entropy_logits, entropy_logits
from kale.pipeline.domain_adapter import GradReverse as ReverseLayerF
from kale.predict.class_domain_nets import Discriminator


class DrugbanTrainer(pl.LightningModule):
    """
    Pytorch Lightning DrugBAN Trainer class

    The DrugBAN Trainer class encapsulates the logic for training a DrugBAN model, optionally with domain adaptation (DA),
    and manages the optimization, logging, and saving of the best-performing model.

    Args:
        model (torch.nn.Module): The model to be trained.
        config (dict): Configuration dictionary containing model, DA, and training settings.
    """

    def __init__(
        self,
        model,
        solver_lr,
        num_classes,
        batch_size,
        is_da,
        solver_da_lr,
        da_init_epoch,
        da_method,
        original_random,
        use_da_entropy,
        da_random_layer,
        da_random_dim,
        decoder_in_dim,
        **kwargs,
    ):
        super(DrugbanTrainer, self).__init__()

        self.model = model
        self.solver_lr = solver_lr
        self.num_classes = num_classes
        self.batch_size = batch_size

        # --- domain adaptation parameters ---
        self.is_da = is_da
        self.solver_da_lr = solver_da_lr
        self.da_init_epoch = da_init_epoch
        self.epoch_lamb_da = 0
        self.da_method = da_method
        self.original_random = original_random
        self.alpha = 1
        self.use_da_entropy = use_da_entropy

        # ---- setup domain adaption model and optimizer ----
        if self.is_da:  # If domain adaptation is used
            # set up discriminator
            if da_random_layer:
                # Initialize the Discriminator with an input size from the random dimension specified in the config
                self.domain_discriminator = Discriminator(input_size=da_random_dim, n_class=self.num_classes)
            else:
                # Initialize the Discriminator with an input size derived from the decoder's input dimension
                self.domain_discriminator = Discriminator(
                    input_size=decoder_in_dim * self.num_classes, n_class=self.num_classes
                )

            # setup random layer
            if da_random_layer and not self.original_random:
                self.random_layer = nn.Linear(
                    in_features=decoder_in_dim * self.num_classes,
                    out_features=da_random_dim,
                    bias=False,
                )
                torch.nn.init.normal_(self.random_layer.weight, mean=0, std=1)
                for param in self.random_layer.parameters():
                    param.requires_grad = False

            elif da_random_layer and self.original_random:
                self.random_layer = RandomLayer([decoder_in_dim, self.num_classes], da_random_dim)
            else:
                self.random_layer = False

        # --- Activate manual optimization ---
        self.automatic_optimization = False

        # --- Metrics ---
        if self.num_classes <= 2:
            metrics = MetricCollection(
                [
                    AUROC("binary", average="none", num_classes=self.num_classes),
                    F1Score("binary", average="none", num_classes=self.num_classes),
                    Recall("binary", average="none", num_classes=self.num_classes),
                    Specificity("binary", average="none", num_classes=self.num_classes),
                    Accuracy("binary", average="none", num_classes=self.num_classes),
                    # AveragePrecision("binary", average="none", num_classes=self.num_classes),
                ]
            )
        else:
            metrics = MetricCollection(
                [
                    AUROC("multiclass", average="none", num_classes=self.num_classes),
                    F1Score("multiclass", average="none", num_classes=self.num_classes),
                    Recall("multiclass", average="none", num_classes=self.num_classes),
                    Specificity("multiclass", average="none", num_classes=self.num_classes),
                    Accuracy("multiclass", average="none", num_classes=self.num_classes),
                    # AveragePrecision("multiclass", average="none", num_classes=self.num_classes),
                ]
            )
        self.valid_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

    def configure_optimizers(self):
        """
        Configure the optimizers for the DrugBAN model and the Domain Discriminator model if DA is used.

        """
        if not self.is_da:  # If domain adaptation is not used
            # Initialize the optimizer for the DrugBAN model
            opt = torch.optim.Adam(self.model.parameters(), lr=self.solver_lr)
            return opt

        else:  # if domain adaptation is used
            # Initialize the optimizer for the DrugBAN model
            opt = torch.optim.Adam(self.model.parameters(), lr=self.solver_lr)
            # Initialize the optimizer for the Domain Discriminator model
            opt_da = torch.optim.Adam(self.domain_discriminator.parameters(), lr=self.solver_da_lr)
            return opt, opt_da

    def on_train_epoch_start(self):
        """
        Update the epoch_lamb_da if in the DA phase.
        """
        # ----- Update epoch_lamb_da if in the DA phase -----
        if self.current_epoch >= self.da_init_epoch:
            self.epoch_lamb_da = 1
        self.log("DA loss lambda", self.epoch_lamb_da, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def _compute_entropy_weights(self, logits):
        entropy = entropy_logits(logits)
        entropy = ReverseLayerF.apply(entropy, self.alpha)
        entropy_w = 1.0 + torch.exp(-entropy)
        return entropy_w

    def training_step(self, train_batch, batch_idx):
        """
        Training step for the DrugBAN model with or without domain adaptation.
        """
        if not self.is_da:
            # ---- optimisers ----
            opt = self.optimizers()
            opt.zero_grad()

            # ---- data ----
            vec_drug, vec_protein, labels = train_batch
            labels = labels.float()

            # forward pass
            vec_drug, vec_protein, f, score_src = self.model(vec_drug, vec_protein)

            # loss calculation
            if self.num_classes == 1:
                n, loss = binary_cross_entropy(score_src, labels)
            else:
                loss, _ = cross_entropy_logits(score_src, labels)

            # ---- log the loss ----
            self.manual_backward(loss)
            opt.step()

            # log the loss - lightning
            self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

            return loss

        else:
            # ---- optimisers ----
            opt, opt_da = self.optimizers()
            opt.zero_grad()
            opt_da.zero_grad()

            # ---- data ----
            batch_source, batch_target = train_batch
            vec_drug, vec_protein, labels = batch_source[0], batch_source[1], batch_source[2].float()
            vec_drug_tgt, vec_protein_tgt = batch_target[0], batch_target[1]

            # ---- model: forward pass ----
            vec_drug, vec_protein, f, score_src = self.model(vec_drug, vec_protein)
            if self.num_classes == 1:
                n, model_loss = binary_cross_entropy(score_src, labels)
            else:
                # n = F.softmax(score, dim=1)[:, 1]
                model_loss, _ = cross_entropy_logits(score_src, labels)

            # ---- domain discriminator: forward pass ----
            if self.current_epoch >= self.da_init_epoch:
                vec_drug_tgt, vec_protein_tgt, f_t, score_tgt = self.model(vec_drug_tgt, vec_protein_tgt)

                if self.da_method == "CDAN":
                    # ---- source----
                    reverse_f = ReverseLayerF.apply(f, self.alpha)
                    softmax_output = torch.nn.Softmax(dim=1)(score_src)
                    softmax_output = softmax_output.detach()

                    if self.original_random:
                        random_out = self.random_layer.forward([reverse_f, softmax_output])
                        adv_output_score_src = self.domain_discriminator(random_out.view(-1, random_out.size(1)))
                    else:
                        feature = torch.bmm(softmax_output.unsqueeze(2), reverse_f.unsqueeze(1))
                        feature = feature.view(-1, softmax_output.size(1) * reverse_f.size(1))
                        if self.random_layer:
                            random_out = self.random_layer.forward(feature)
                            adv_output_score_src = self.domain_discriminator(random_out)
                        else:
                            adv_output_score_src = self.domain_discriminator(feature)

                    # ---- target ----
                    reverse_f_t = ReverseLayerF.apply(f_t, self.alpha)
                    softmax_output_t = torch.nn.Softmax(dim=1)(score_tgt)
                    softmax_output_t = softmax_output_t.detach()

                    if self.original_random:
                        random_out_tgt = self.random_layer.forward([reverse_f_t, softmax_output_t])
                        adv_output_score_tgt = self.domain_discriminator(
                            random_out_tgt.view(-1, random_out_tgt.size(1))
                        )
                    else:
                        feature_tgt = torch.bmm(softmax_output_t.unsqueeze(2), reverse_f_t.unsqueeze(1))
                        feature_tgt = feature_tgt.view(-1, softmax_output_t.size(1) * reverse_f_t.size(1))
                        if self.random_layer:
                            random_out_tgt = self.random_layer.forward(feature_tgt)
                            adv_output_score_tgt = self.domain_discriminator(random_out_tgt)
                        else:
                            adv_output_score_tgt = self.domain_discriminator(feature_tgt)

                    if self.use_da_entropy:
                        entropy_src = self._compute_entropy_weights(score_src)
                        entropy_tgt = self._compute_entropy_weights(score_tgt)
                        weight_src = entropy_src / torch.sum(entropy_src)
                        weight_tgt = entropy_tgt / torch.sum(entropy_tgt)
                    else:
                        weight_src = None
                        weight_tgt = None

                    loss_cdan_src, _ = cross_entropy_logits(
                        adv_output_score_src, torch.zeros(self.batch_size), weights=weight_src
                    )

                    loss_cdan_tgt, _ = cross_entropy_logits(
                        adv_output_score_tgt, torch.ones(self.batch_size), weights=weight_tgt
                    )

                    da_loss = loss_cdan_src + loss_cdan_tgt

                else:
                    raise ValueError(f"The da method {self.da_method} is not supported")
                loss = model_loss + da_loss
            else:
                loss = model_loss

            # ---- log the loss ----
            self.manual_backward(loss)
            opt.step()
            opt_da.step()

            self.log(
                "train_step model loss", model_loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True
            )
            self.log("train_step total loss", loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
            if self.current_epoch >= self.da_init_epoch:
                self.log("train_step da loss", da_loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True)

            return loss

    def validation_step(self, val_batch, batch_idx):
        v_d, v_p, labels = val_batch
        labels = labels.float()
        v_d, v_p, f, score = self.model(v_d, v_p)
        if self.num_classes == 1:
            n, loss = binary_cross_entropy(score, labels)
        else:
            n = F.softmax(score, dim=1)[:, 1]
            loss, _ = cross_entropy_logits(score, labels)

        # ---- metrics update ----
        self.valid_metrics.update(n, labels.long())

        # ---- log the loss ----
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return

    def on_validation_epoch_end(self):
        """
        At the end of the validation epoch, compute and log the validation metrics, and reset the validation metrics.
        """
        output = self.valid_metrics.compute()
        self.log_dict(output)
        self.valid_metrics.reset()

    def test_step(self, test_batch, batch_idx):
        v_d, v_p, labels = test_batch
        labels = labels.float()
        v_d, v_p, f, score = self.model(v_d, v_p)
        if self.num_classes == 1:
            n, loss = binary_cross_entropy(score, labels)
        else:
            n = F.softmax(score, dim=1)[:, 1]
            loss, _ = cross_entropy_logits(score, labels)

        # ---- metrics update ----
        self.test_metrics.update(n, labels.long())

        # ---- log the loss ----
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def on_test_epoch_end(self):
        """
        At the end of the test epoch, compute and log the test metrics, and reset the test metrics.
        """
        output = self.test_metrics.compute()
        self.log_dict(output)
        self.test_metrics.reset()
