# =============================================================================
# Author: Jiayang Zhang, jiayang.zhang@sheffield.ac.uk
# =============================================================================

"""
This module contains the DrugBAN trainer class and related functions. It trains a Interpretable bilinear attention
network with or without domain adaptation model for drug-target interaction prediction.

This is refactored from: https://github.com/peizhenbai/DrugBAN/blob/main/trainer.py
"""

import copy
import logging
import os

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from prettytable import PrettyTable
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    roc_auc_score,
    roc_curve,
)
from torch.nn import functional as F
from torchmetrics import AUROC, AveragePrecision, MetricCollection
from tqdm import tqdm

from kale.embed.ban import RandomLayer
from kale.pipeline.domain_adapter import GradReverse as ReverseLayerF
from kale.predict.losses import binary_cross_entropy, cross_entropy_logits, entropy_logits


class Trainer(object):
    """
    DrugBAN Trainer class

    The Trainer class encapsulates the logic for training a DrugBAN model, optionally with domain adaptation (DA),
    and manages the optimization, logging, and saving of the best-performing model.

    Args:
        model (torch.nn.Module): The model to be trained.
        optim (torch.optim.Optimizer): The optimizer for the model.
        device (torch.device): The device to run the training on (e.g., GPU or CPU).
        train_dataloader (DataLoader): DataLoader for training data.
        val_dataloader (DataLoader): DataLoader for validation data.
        test_dataloader (DataLoader): DataLoader for test data.
        opt_da (torch.optim.Optimizer, optional): The optimizer for domain adaptation (DA) if used.
        discriminator (nn.Module, optional): The domain discriminator used for DA if used.
        experiment (CometExperiment, optional): An experiment logger for Comet ML to log metrics.
        alpha (float): A coefficient for controlling the influence of the DA component.
        config (dict): Configuration dictionary containing model, DA, and training settings.


    """

    def __init__(
        self,
        model,
        optim,
        device,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        opt_da=None,
        discriminator=None,
        experiment=None,
        alpha=1,
        **config,
    ):
        self.model = model
        self.optim = optim
        self.device = device
        self.epochs = config["SOLVER"]["MAX_EPOCH"]
        self.current_epoch = 0
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.is_da = config["DA"]["USE"]
        self.alpha = alpha
        self.n_class = config["DECODER"]["BINARY"]

        if opt_da:
            self.optim_da = opt_da
        if self.is_da:
            """If domain adaptation is used"""
            self.da_method = config["DA"]["METHOD"]
            self.domain_dmm = discriminator

            if config["DA"]["RANDOM_LAYER"] and not config["DA"]["ORIGINAL_RANDOM"]:
                self.random_layer = nn.Linear(
                    in_features=config["DECODER"]["IN_DIM"] * self.n_class,
                    out_features=config["DA"]["RANDOM_DIM"],
                    bias=False,
                ).to(self.device)

                torch.nn.init.normal_(self.random_layer.weight, mean=0, std=1)

                for param in self.random_layer.parameters():
                    param.requires_grad = False

            elif config["DA"]["RANDOM_LAYER"] and config["DA"]["ORIGINAL_RANDOM"]:
                self.random_layer = RandomLayer([config["DECODER"]["IN_DIM"], self.n_class], config["DA"]["RANDOM_DIM"])
                if torch.cuda.is_available():
                    self.random_layer.cuda()
            else:
                self.random_layer = False

        self.da_init_epoch = config["DA"]["INIT_EPOCH"]
        self.init_lamb_da = config["DA"]["LAMB_DA"]
        self.batch_size = config["SOLVER"]["BATCH_SIZE"]
        self.use_da_entropy = config["DA"]["USE_ENTROPY"]
        self.nb_training = len(self.train_dataloader)
        self.step = 0
        self.experiment = experiment

        self.best_model = None
        self.best_epoch = None
        self.best_auroc = 0

        self.train_loss_epoch = []
        self.train_model_loss_epoch = []
        self.train_da_loss_epoch = []
        self.val_loss_epoch, self.val_auroc_epoch = [], []
        self.test_metrics = {}
        self.config = config
        self.output_dir = config["RESULT"]["OUTPUT_DIR"]

        valid_metric_header = ["# Epoch", "AUROC", "AUPRC", "Val_loss"]
        test_metric_header = [
            "# Best Epoch",
            "AUROC",
            "AUPRC",
            "F1",
            "Sensitivity",
            "Specificity",
            "Accuracy",
            "Threshold",
            "Test_loss",
        ]

        if not self.is_da:
            train_metric_header = ["# Epoch", "Train_loss"]
        else:
            train_metric_header = ["# Epoch", "Train_loss", "Model_loss", "epoch_lamb_da", "da_loss"]
        self.val_table = PrettyTable(valid_metric_header)
        self.test_table = PrettyTable(test_metric_header)
        self.train_table = PrettyTable(train_metric_header)

        self.original_random = config["DA"]["ORIGINAL_RANDOM"]

    def train(self):
        """
        Train the DrugBAN model across multiple epochs, optionally including domain adaptation (DA).

        This method manages the training loop, handling both standard and domain adaptation (DA) training. It logs
        metrics after each epoch, including training loss, validation metrics (AUROC, AUPRC, loss), and updates the
        best-performing model based on validation AUROC. The method also saves training progress and the best model
        to the specified output directory.

        During each epoch:
        - If DA is not used, the method calls `train_epoch` to train the model normally.
        - If DA is used, the method calls `train_da_epoch` to include domain adaptation components in training.
        - After each epoch, validation metrics are computed and logged.
        - At the end of training, the method evaluates the model on the test dataset and logs the results.
        """
        # float2str = lambda x: "%0.4f" % x
        for i in range(self.epochs):
            self.current_epoch += 1
            if not self.is_da:
                train_loss = self.train_epoch()
                train_lst = ["epoch " + str(self.current_epoch)] + [f"{train_loss:.4f}"]
                if self.experiment:
                    self.experiment.log_metric("train_epoch model loss", train_loss, epoch=self.current_epoch)
            else:
                train_loss, model_loss, da_loss, epoch_lamb = self.train_da_epoch()
                train_lst = ["epoch " + str(self.current_epoch)] + [
                    f"{x:.4f}" for x in [train_loss, model_loss, epoch_lamb, da_loss]
                ]
                self.train_model_loss_epoch.append(model_loss)
                self.train_da_loss_epoch.append(da_loss)
                if self.experiment:
                    self.experiment.log_metric("train_epoch total loss", train_loss, epoch=self.current_epoch)
                    self.experiment.log_metric("train_epoch model loss", model_loss, epoch=self.current_epoch)
                    if self.current_epoch >= self.da_init_epoch:
                        self.experiment.log_metric("train_epoch da loss", da_loss, epoch=self.current_epoch)
            self.train_table.add_row(train_lst)
            self.train_loss_epoch.append(train_loss)
            auroc, auprc, val_loss = self.test(dataloader="val")
            if self.experiment:
                self.experiment.log_metric("valid_epoch model loss", val_loss, epoch=self.current_epoch)
                self.experiment.log_metric("valid_epoch auroc", auroc, epoch=self.current_epoch)
                self.experiment.log_metric("valid_epoch auprc", auprc, epoch=self.current_epoch)
            val_lst = ["epoch " + str(self.current_epoch)] + [f"{x:.4f}" for x in [auroc, auprc, val_loss]]
            self.val_table.add_row(val_lst)
            self.val_loss_epoch.append(val_loss)
            self.val_auroc_epoch.append(auroc)
            if auroc >= self.best_auroc:
                self.best_model = copy.deepcopy(self.model)
                self.best_auroc = auroc
                self.best_epoch = self.current_epoch
            logging.info(
                "Validation at Epoch " + str(self.current_epoch) + " with validation loss " + str(val_loss),
                " AUROC " + str(auroc) + " AUPRC " + str(auprc),
            )
        auroc, auprc, f1, sensitivity, specificity, accuracy, test_loss, thred_optim, precision = self.test(
            dataloader="test"
        )

        test_lst = [f"epoch {self.best_epoch}"] + [
            f"{x:.4f}" for x in [auroc, auprc, f1, sensitivity, specificity, accuracy, thred_optim, test_loss]
        ]
        self.test_table.add_row(test_lst)
        logging.info(
            "Test at Best Model of Epoch " + str(self.best_epoch) + " with test loss " + str(test_loss),
            " AUROC "
            + str(auroc)
            + " AUPRC "
            + str(auprc)
            + " Sensitivity "
            + str(sensitivity)
            + " Specificity "
            + str(specificity)
            + " Accuracy "
            + str(accuracy)
            + " Thred_optim "
            + str(thred_optim),
        )
        self.test_metrics["auroc"] = auroc
        self.test_metrics["auprc"] = auprc
        self.test_metrics["test_loss"] = test_loss
        self.test_metrics["sensitivity"] = sensitivity
        self.test_metrics["specificity"] = specificity
        self.test_metrics["accuracy"] = accuracy
        self.test_metrics["thred_optim"] = thred_optim
        self.test_metrics["best_epoch"] = self.best_epoch
        self.test_metrics["F1"] = f1
        self.test_metrics["Precision"] = precision
        self.save_result()
        if self.experiment:
            self.experiment.log_metric("valid_best_auroc", self.best_auroc)
            self.experiment.log_metric("valid_best_epoch", self.best_epoch)
            self.experiment.log_metric("test_auroc", self.test_metrics["auroc"])
            self.experiment.log_metric("test_auprc", self.test_metrics["auprc"])
            self.experiment.log_metric("test_sensitivity", self.test_metrics["sensitivity"])
            self.experiment.log_metric("test_specificity", self.test_metrics["specificity"])
            self.experiment.log_metric("test_accuracy", self.test_metrics["accuracy"])
            self.experiment.log_metric("test_threshold", self.test_metrics["thred_optim"])
            self.experiment.log_metric("test_f1", self.test_metrics["F1"])
            self.experiment.log_metric("test_precision", self.test_metrics["Precision"])
        return self.test_metrics

    def save_result(self):
        """
        Save the model's state_dict and training, validation and testing results.

        This method saves the best model's state dictionary, the current model's state dictionary,
        and various training metrics to files in the specified output directory. Additionally,
        it writes the training, validation, and test results to markdown table files for easy
        viewing.

        Files saved:
        - `best_model_epoch_{best_epoch}.pth`: Best model's state dictionary.
        - `model_epoch_{current_epoch}.pth`: Current model's state dictionary.
        - `result_metrics.pt`: A dictionary containing all relevant metrics and configurations.
        - `valid_markdowntable.txt`: Validation metrics in markdown table format.
        - `test_markdowntable.txt`: Test metrics in markdown table format.
        - `train_markdowntable.txt`: Training metrics in markdown table format.
        """
        # ---- save models ---
        if self.config["RESULT"]["SAVE_MODEL"]:
            torch.save(
                self.best_model.state_dict(), os.path.join(self.output_dir, f"best_model_epoch_{self.best_epoch}.pth")
            )
            torch.save(self.model.state_dict(), os.path.join(self.output_dir, f"model_epoch_{self.current_epoch}.pth"))
        state = {
            "train_epoch_loss": self.train_loss_epoch,
            "val_epoch_loss": self.val_loss_epoch,
            "test_metrics": self.test_metrics,
            "config": self.config,
        }
        if self.is_da:
            # If domain adaptation (DA) is used
            state["train_model_loss"] = self.train_model_loss_epoch
            state["train_da_loss"] = self.train_da_loss_epoch
            state["da_init_epoch"] = self.da_init_epoch
        torch.save(state, os.path.join(self.output_dir, "result_metrics.pt"))

        # ---- save markdown tables ----
        val_prettytable_file = os.path.join(self.output_dir, "valid_markdowntable.txt")
        test_prettytable_file = os.path.join(self.output_dir, "test_markdowntable.txt")
        train_prettytable_file = os.path.join(self.output_dir, "train_markdowntable.txt")
        with open(val_prettytable_file, "w") as fp:
            fp.write(self.val_table.get_string())
        with open(test_prettytable_file, "w") as fp:
            fp.write(self.test_table.get_string())
        with open(train_prettytable_file, "w") as fp:
            fp.write(self.train_table.get_string())

    def _compute_entropy_weights(self, logits):
        """
        Compute entropy weights for domain adaptation.

        This method computes weights based on the entropy of the logits. The entropy is first
        calculated, then passed through a gradient reversal layer, and finally transformed
        into weights.

        Args:
        logits (torch.Tensor): The logits output from the model.

        Returns:
        torch.Tensor: The computed entropy-based weights for each sample in the batch.
        """
        entropy = entropy_logits(logits)
        entropy = ReverseLayerF.apply(entropy, self.alpha)
        entropy_w = 1.0 + torch.exp(-entropy)
        return entropy_w

    def train_epoch(self):
        """
        Perform a single epoch of training.

        This method trains the model for one full pass (epoch) over the training dataset.
        """
        self.model.train()
        loss_epoch = 0
        num_batches = len(self.train_dataloader)
        for i, (v_d, v_p, labels) in enumerate(tqdm(self.train_dataloader)):
            self.step += 1
            v_d, v_p, labels = v_d.to(self.device), v_p.to(self.device), labels.float().to(self.device)
            self.optim.zero_grad()
            v_d, v_p, f, score = self.model(v_d, v_p)
            if self.n_class == 1:
                n, loss = binary_cross_entropy(score, labels)
            else:
                # n = F.softmax(score, dim=1)[:, 1]
                loss, _ = cross_entropy_logits(score, labels)
            loss.backward()
            self.optim.step()
            loss_epoch += loss.item()
            if self.experiment:
                self.experiment.log_metric("train_step model loss", loss.item(), step=self.step)
        loss_epoch = loss_epoch / num_batches
        logging.info("Training at Epoch " + str(self.current_epoch) + " with training loss " + str(loss_epoch))
        return loss_epoch

    def train_da_epoch(self):
        """
        Perform a single epoch of training with Domain Adaptation (DA).

        This method trains the model for one full pass (epoch) over the training dataset, with an emphasis on CDAN.
        """
        self.model.train()
        total_loss_epoch = 0
        model_loss_epoch = 0
        da_loss_epoch = 0
        epoch_lamb_da = 0

        # ----- Update epoch_lamb_da if in the DA phase -----
        if self.current_epoch >= self.da_init_epoch:
            # epoch_lamb_da = self.da_lambda_decay()
            epoch_lamb_da = 1
            if self.experiment:
                self.experiment.log_metric("DA loss lambda", epoch_lamb_da, epoch=self.current_epoch)

        num_batches = len(self.train_dataloader)
        for i, (batch_s, batch_t) in enumerate(tqdm(self.train_dataloader)):
            self.step += 1
            v_d, v_p, labels = (
                batch_s[0].to(self.device),
                batch_s[1].to(self.device),
                batch_s[2].float().to(self.device),
            )

            v_d_t, v_p_t = batch_t[0].to(self.device), batch_t[1].to(self.device)
            self.optim.zero_grad()
            self.optim_da.zero_grad()
            v_d, v_p, f, score = self.model(v_d, v_p)

            if self.n_class == 1:
                n, model_loss = binary_cross_entropy(score, labels)

            else:
                # n = F.softmax(score, dim=1)[:, 1]
                model_loss, _ = cross_entropy_logits(score, labels)

            if self.current_epoch >= self.da_init_epoch:
                v_d_t, v_p_t, f_t, t_score = self.model(v_d_t, v_p_t)
                if self.da_method == "CDAN":
                    reverse_f = ReverseLayerF.apply(f, self.alpha)
                    softmax_output = torch.nn.Softmax(dim=1)(score)
                    softmax_output = softmax_output.detach()
                    # reverse_output = ReverseLayerF.apply(softmax_output, self.alpha)
                    if self.original_random:
                        random_out = self.random_layer.forward([reverse_f, softmax_output])
                        adv_output_src_score = self.domain_dmm(random_out.view(-1, random_out.size(1)))
                    else:
                        feature = torch.bmm(softmax_output.unsqueeze(2), reverse_f.unsqueeze(1))
                        feature = feature.view(-1, softmax_output.size(1) * reverse_f.size(1))
                        if self.random_layer:
                            random_out = self.random_layer.forward(feature)
                            adv_output_src_score = self.domain_dmm(random_out)
                        else:
                            adv_output_src_score = self.domain_dmm(feature)

                    reverse_f_t = ReverseLayerF.apply(f_t, self.alpha)
                    softmax_output_t = torch.nn.Softmax(dim=1)(t_score)
                    softmax_output_t = softmax_output_t.detach()
                    # reverse_output_t = ReverseLayerF.apply(softmax_output_t, self.alpha)
                    if self.original_random:
                        random_out_t = self.random_layer.forward([reverse_f_t, softmax_output_t])
                        adv_output_tgt_score = self.domain_dmm(random_out_t.view(-1, random_out_t.size(1)))
                    else:
                        feature_t = torch.bmm(softmax_output_t.unsqueeze(2), reverse_f_t.unsqueeze(1))
                        feature_t = feature_t.view(-1, softmax_output_t.size(1) * reverse_f_t.size(1))
                        if self.random_layer:
                            random_out_t = self.random_layer.forward(feature_t)
                            adv_output_tgt_score = self.domain_dmm(random_out_t)
                        else:
                            adv_output_tgt_score = self.domain_dmm(feature_t)

                    if self.use_da_entropy:
                        entropy_src = self._compute_entropy_weights(score)
                        entropy_tgt = self._compute_entropy_weights(t_score)
                        src_weight = entropy_src / torch.sum(entropy_src)
                        tgt_weight = entropy_tgt / torch.sum(entropy_tgt)
                    else:
                        src_weight = None
                        tgt_weight = None

                    # n_src = F.softmax(adv_output_src_score, dim=1)[:, 1]
                    loss_cdan_src, _ = cross_entropy_logits(
                        adv_output_src_score, torch.zeros(self.batch_size).to(self.device), weights=src_weight
                    )

                    # n_tgt = F.softmax(adv_output_tgt_score, dim=1)[:, 1]
                    loss_cdan_tgt, _ = cross_entropy_logits(
                        adv_output_tgt_score, torch.ones(self.batch_size).to(self.device), weights=tgt_weight
                    )

                    da_loss = loss_cdan_src + loss_cdan_tgt
                else:
                    raise ValueError(f"The da method {self.da_method} is not supported")
                loss = model_loss + da_loss
            else:
                loss = model_loss
            loss.backward()
            self.optim.step()
            self.optim_da.step()
            total_loss_epoch += loss.item()
            model_loss_epoch += model_loss.item()
            if self.experiment:
                self.experiment.log_metric("train_step model loss", model_loss.item(), step=self.step)
                self.experiment.log_metric("train_step total loss", loss.item(), step=self.step)
            if self.current_epoch >= self.da_init_epoch:
                da_loss_epoch += da_loss.item()
                if self.experiment:
                    self.experiment.log_metric("train_step da loss", da_loss.item(), step=self.step)
        total_loss_epoch = total_loss_epoch / num_batches
        model_loss_epoch = model_loss_epoch / num_batches
        da_loss_epoch = da_loss_epoch / num_batches
        if self.current_epoch < self.da_init_epoch:
            logging.info(
                "Training at Epoch " + str(self.current_epoch) + " with model training loss " + str(total_loss_epoch)
            )
        else:
            logging.info(
                "Training at Epoch "
                + str(self.current_epoch)
                + " model training loss "
                + str(model_loss_epoch)
                + ", da loss "
                + str(da_loss_epoch)
                + ", total training loss "
                + str(total_loss_epoch)
                + ", DA lambda "
                + str(epoch_lamb_da)
            )
        return total_loss_epoch, model_loss_epoch, da_loss_epoch, epoch_lamb_da

    def test(self, dataloader="test"):
        """
        Evaluate the model on the validation or test dataset.

        It handles both binary and multi-class classification scenarios and supports logging of ROC and precision-recall curves if a Comet experiment is provided.

        Args:
            dataloader (str, optional): Specifies which dataset to use for evaluation.
                                        - "test": Uses the test dataset.
                                        - "val": Uses the validation dataset.
                                        Default is "test".
        """
        test_loss = 0
        y_label, y_pred = [], []
        if dataloader == "test":
            data_loader = self.test_dataloader
        elif dataloader == "val":
            data_loader = self.val_dataloader
        else:
            raise ValueError(f"Error key value {dataloader}")
        num_batches = len(data_loader)
        with torch.no_grad():
            self.model.eval()
            for i, (v_d, v_p, labels) in enumerate(data_loader):
                v_d, v_p, labels = v_d.to(self.device), v_p.to(self.device), labels.float().to(self.device)
                if dataloader == "val":
                    v_d, v_p, f, score = self.model(v_d, v_p)
                elif dataloader == "test":
                    v_d, v_p, f, score = self.best_model(v_d, v_p)
                if self.n_class == 1:
                    n, loss = binary_cross_entropy(score, labels)
                else:
                    n = F.softmax(score, dim=1)[:, 1]
                    loss, _ = cross_entropy_logits(score, labels)
                test_loss += loss.item()
                y_label = y_label + labels.to("cpu").tolist()
                y_pred = y_pred + n.to("cpu").tolist()
        auroc = roc_auc_score(y_label, y_pred)
        auprc = average_precision_score(y_label, y_pred)
        test_loss = test_loss / num_batches

        if dataloader == "test":
            fpr, tpr, thresholds = roc_curve(y_label, y_pred)
            prec, recall, _ = precision_recall_curve(y_label, y_pred)
            precision = tpr / (tpr + fpr)
            f1 = 2 * precision * tpr / (tpr + precision + 0.00001)
            thred_optim = thresholds[5:][np.argmax(f1[5:])]
            y_pred_s = [1 if i else 0 for i in (y_pred >= thred_optim)]
            cm1 = confusion_matrix(y_label, y_pred_s)
            accuracy = (cm1[0, 0] + cm1[1, 1]) / sum(sum(cm1))
            sensitivity = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
            specificity = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])
            if self.experiment:
                self.experiment.log_curve("test_roc curve", fpr, tpr)
                self.experiment.log_curve("test_pr curve", recall, prec)
            precision1 = precision_score(y_label, y_pred_s)
            return auroc, auprc, np.max(f1[5:]), sensitivity, specificity, accuracy, test_loss, thred_optim, precision1
        else:
            return auroc, auprc, test_loss


class DrugbanTrainer(pl.LightningModule):
    def __init__(self, model, **config):
        super(DrugbanTrainer, self).__init__()
        # model
        self.model = model

        self.solver_lr = config["SOLVER"]["LR"]
        self.n_class = config["DECODER"]["BINARY"]

        # Metrics
        if self.n_class <= 2:
            metrics = MetricCollection(
                [
                    AUROC("binary", average="none", num_classes=self.n_class),
                    AveragePrecision("binary", average="none", num_classes=self.n_class),
                ]
            )
        else:
            metrics = MetricCollection(
                [
                    AUROC("multiclass", average="none", num_classes=self.n_class),
                    AveragePrecision("multiclass", average="none", num_classes=self.n_class),
                ]
            )

        self.valid_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

    def forward(self, v_d, v_p):
        v_d, v_p, f, score = self.model(v_d, v_p)
        return f, score

    def configure_optimizers(self):
        """If domain adaptation is not used, only initialize the optimizer for the DrugBAN model"""
        opt = torch.optim.Adam(self.model.parameters(), lr=self.solver_lr)
        return opt

    def training_step(self, train_batch, batch_idx):
        v_d, v_p, labels = train_batch
        labels = labels.float()

        # forward pass
        f, score = self(v_d, v_p)

        # loss calculation
        if self.n_class == 1:
            n, loss = binary_cross_entropy(score, labels)
        else:
            loss, _ = cross_entropy_logits(score, labels)

        # log the loss - lightning
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, val_batch, batch_idx):
        v_d, v_p, labels = val_batch
        labels = labels.float()
        f, score = self(v_d, v_p)
        if self.n_class == 1:
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
        output = self.valid_metrics.compute()
        self.log_dict(output)
        self.valid_metrics.reset()

    def test_step(self, test_batch, batch_idx):
        v_d, v_p, labels = test_batch
        labels = labels.float()
        f, score = self(v_d, v_p)
        if self.n_class == 1:
            n, loss = binary_cross_entropy(score, labels)
        else:
            n = F.softmax(score, dim=1)[:, 1]
            loss, _ = cross_entropy_logits(score, labels)

        # ---- metrics update ----
        self.test_metrics.update(n, labels.long())

        # ---- log the loss ----
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def on_test_epoch_end(self):
        output = self.test_metrics.compute()
        self.log_dict(output)
        self.test_metrics.reset()


class DrugbanDATrainer(pl.LightningModule):
    def __init__(self, model, discriminator, **config):
        super(DrugbanDATrainer, self).__init__()

        # model
        self.model = model
        self.solver_lr = config["SOLVER"]["LR"]

        # general
        self.n_class = config["DECODER"]["BINARY"]

        # domain adaptation -- parameters
        self.solver_da_lr = config["SOLVER"]["DA_LR"]
        self.da_init_epoch = config["DA"]["INIT_EPOCH"]
        self.epoch_lamb_da = 0
        self.da_method = config["DA"]["METHOD"]
        self.original_random = config["DA"]["ORIGINAL_RANDOM"]

        # domain adaptation -- discriminator
        if config["DA"]["RANDOM_LAYER"]:
            # Initialize the Discriminator with an input size from the random dimension specified in the config
            self.domain_dmm = discriminator(input_size=config["DA"]["RANDOM_DIM"], n_class=self.n_class)
        else:
            # Initialize the Discriminator with an input size derived from the decoder's input dimension
            self.domain_dmm = discriminator(input_size=config["DECODER"]["IN_DIM"] * self.n_class, n_class=self.n_class)

        # domain adaption -- random layer
        if config["DA"]["RANDOM_LAYER"] and not config["DA"]["ORIGINAL_RANDOM"]:
            self.random_layer = nn.Linear(
                in_features=config["DECODER"]["IN_DIM"] * self.n_class,
                out_features=config["DA"]["RANDOM_DIM"],
                bias=False,
            )
            torch.nn.init.normal_(self.random_layer.weight, mean=0, std=1)
            for param in self.random_layer.parameters():
                param.requires_grad = False

        elif config["DA"]["RANDOM_LAYER"] and config["DA"]["ORIGINAL_RANDOM"]:
            self.random_layer = RandomLayer([config["DECODER"]["IN_DIM"], self.n_class], config["DA"]["RANDOM_DIM"])
        else:
            self.random_layer = False

        # Activate manual optimization
        self.automatic_optimization = False

        # Metrics
        if self.n_class <= 2:
            metrics = MetricCollection(
                [
                    AUROC("binary", average="none", num_classes=self.n_class),
                    AveragePrecision("binary", average="none", num_classes=self.n_class),
                ]
            )
        else:
            metrics = MetricCollection(
                [
                    AUROC("multiclass", average="none", num_classes=self.n_class),
                    AveragePrecision("multiclass", average="none", num_classes=self.n_class),
                ]
            )

        self.valid_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

    def configure_optimizers(self):
        """If domain adaptation is used"""
        # Initialize the optimizer for the DrugBAN model
        opt = torch.optim.Adam(self.model.parameters(), lr=self.solver_lr)
        # Initialize the optimizer for the Domain Discriminator model
        opt_da = torch.optim.Adam(self.domain_dmm.parameters(), lr=self.solver_da_lr)

        return opt, opt_da

    def on_train_epoch_start(self):
        # ----- Update epoch_lamb_da if in the DA phase -----
        if self.current_epoch >= self.da_init_epoch:
            self.epoch_lamb_da = 1
        self.log("DA loss lambda", self.epoch_lamb_da, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def training_step(self, train_batch, batch_idx):
        # ---- optimisers ----
        opt, opt_da = self.optimizers()
        opt.zero_grad()
        opt_da.zero_grad()

        # ---- data ----
        batch_source, batch_target = train_batch
        v_d, v_p, labels = batch_source[0], batch_source[1], batch_source[2].float()
        v_d_t, v_p_t = batch_target[0], batch_target[1]

        # ---- model: forward pass ----
        v_d, v_p, f, score = self.model(v_d, v_p)
        if self.n_class == 1:
            n, model_loss = binary_cross_entropy(score, labels)
        else:
            # n = F.softmax(score, dim=1)[:, 1]
            model_loss, _ = cross_entropy_logits(score, labels)

        # ---- domain discriminator: forward pass ----
        if self.current_epoch >= self.da_init_epoch:
            v_d_t, v_p_t, f_t, t_score = self.model(v_d_t, v_p_t)

            if self.da_method == "CDAN":
                # ---- source----
                reverse_f = ReverseLayerF.apply(f, self.alpha)
                softmax_output = torch.nn.Softmax(dim=1)(score)
                softmax_output = softmax_output.detach()

                if self.original_random:
                    random_out = self.random_layer.forward([reverse_f, softmax_output])
                    adv_output_src_score = self.domain_dmm(random_out.view(-1, random_out.size(1)))
                else:
                    feature = torch.bmm(softmax_output.unsqueeze(2), reverse_f.unsqueeze(1))
                    feature = feature.view(-1, softmax_output.size(1) * reverse_f.size(1))
                    if self.random_layer:
                        random_out = self.random_layer.forward(feature)
                        adv_output_src_score = self.domain_dmm(random_out)
                    else:
                        adv_output_src_score = self.domain_dmm(feature)

                # ---- target ----
                reverse_f_t = ReverseLayerF.apply(f_t, self.alpha)
                softmax_output_t = torch.nn.Softmax(dim=1)(t_score)
                softmax_output_t = softmax_output_t.detach()

                if self.original_random:
                    random_out_t = self.random_layer.forward([reverse_f_t, softmax_output_t])
                    adv_output_tgt_score = self.domain_dmm(random_out_t.view(-1, random_out_t.size(1)))
                else:
                    feature_t = torch.bmm(softmax_output_t.unsqueeze(2), reverse_f_t.unsqueeze(1))
                    feature_t = feature_t.view(-1, softmax_output_t.size(1) * reverse_f_t.size(1))
                    if self.random_layer:
                        random_out_t = self.random_layer.forward(feature_t)
                        adv_output_tgt_score = self.domain_dmm(random_out_t)
                    else:
                        adv_output_tgt_score = self.domain_dmm(feature_t)

                if self.use_da_entropy:
                    entropy_src = self._compute_entropy_weights(score)
                    entropy_tgt = self._compute_entropy_weights(t_score)
                    src_weight = entropy_src / torch.sum(entropy_src)
                    tgt_weight = entropy_tgt / torch.sum(entropy_tgt)
                else:
                    src_weight = None
                    tgt_weight = None

                loss_cdan_src, _ = cross_entropy_logits(
                    adv_output_src_score, torch.zeros(self.batch_size).to(self.device), weights=src_weight
                )

                loss_cdan_tgt, _ = cross_entropy_logits(
                    adv_output_tgt_score, torch.ones(self.batch_size).to(self.device), weights=tgt_weight
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

        self.log("train_step model loss", model_loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_step total loss", loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        if self.current_epoch >= self.da_init_epoch:
            self.log("train_step da loss", da_loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, val_batch, batch_idx):
        v_d, v_p, labels = val_batch
        labels = labels.float()
        v_d, v_p, f, score = self.model(v_d, v_p)
        if self.n_class == 1:
            n, loss = binary_cross_entropy(score, labels)
        else:
            n = F.softmax(score, dim=1)[:, 1]
            loss, _ = cross_entropy_logits(score, labels)

        # ---- metrics update ----
        self.valid_metrics.update(n, labels)

        # ---- log the loss ----
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return

    def on_validation_epoch_end(self):
        output = self.valid_metrics.compute()
        self.log_dict(output)
        self.valid_metrics.reset()

    def test_step(self, test_batch, batch_idx):
        v_d, v_p, labels = test_batch
        labels = labels.float()
        v_d, v_p, f, score = self.model(v_d, v_p)
        if self.n_class == 1:
            n, loss = binary_cross_entropy(score, labels)
        else:
            n = F.softmax(score, dim=1)[:, 1]
            loss, _ = cross_entropy_logits(score, labels)

        # ---- metrics update ----
        self.test_metrics.update(n, labels)

        # ---- log the loss ----
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def on_test_epoch_end(self):
        output = self.test_metrics.compute()
        self.log_dict(output)
        self.test_metrics.reset()
