import pytorch_lightning as pl
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.nn import functional as F

from kale.embed.ban import RandomLayer
from kale.predict.losses import binary_cross_entropy, cross_entropy_logits

from torchmetrics import MetricCollection, AUROC, Precision

class DrugbanTrainer(pl.LightningModule):
    def __init__(self, model, **config):
        super(DrugbanTrainer, self).__init__()
        # model
        self.model = model

        self.solver_lr = config["SOLVER"]["LR"]
        self.n_class = config["DECODER"]["BINARY"]

        self.val_preds, self.val_labels = [], []
        self.test_preds, self.test_labels = [], []

        # Metrics
        if self.n_class <= 2:
            metrics = MetricCollection([
                AUROC("binary", average='none', num_classes=self.n_class),
                Precision("binary", average='none', num_classes=self.n_class),
            ])
        else:
            metrics = MetricCollection([
                AUROC("multiclass", average='none', num_classes=self.n_class),
                Precision("multiclass", average='none', num_classes=self.n_class)
            ])

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
        self.valid_metrics.update(n, labels)

        # ---- store the predictions and labels for the validation set ----
        self.val_preds.append(n.detach().cpu())
        self.val_labels.append(labels.detach().cpu())

        # ---- log the loss ----
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return

    def on_validation_epoch_end(self):
        # print("Validation finished")
        val_preds = torch.cat(self.val_preds).numpy()
        val_labels = torch.cat(self.val_labels).numpy()

        # ---- calculate the metrics ----
        auroc = roc_auc_score(val_labels, val_preds)
        auprc = average_precision_score(val_labels, val_preds)

        # ---- log the metrics ----
        self.log("sklearn_val_auroc", auroc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("sklearn_val_auprc", auprc, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # ---- clear the lists for the next epoch ----
        self.val_preds.clear()
        self.val_labels.clear()

        # ---- torch metrics: log the metrics ----
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
        self.test_metrics.update(n, labels)

        # ---- store the predictions and labels for the test set ----
        self.test_preds.append(n.detach().cpu())
        self.test_labels.append(labels.detach().cpu())

        # ---- log the loss ----
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def on_test_epoch_end(self) -> None:
        # print("Testing finished")
        test_preds = torch.cat(self.test_preds).numpy()
        test_labels = torch.cat(self.test_labels).numpy()

        # ---- calculate the metrics ----
        auroc = roc_auc_score(test_labels, test_preds)
        auprc = average_precision_score(test_labels, test_preds)

        # ---- log the metrics ----
        self.log("sklearn_test_auroc", auroc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("sklearn_test_auprc", auprc, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # --- clear the lists for the next epoch ---
        self.test_preds.clear()
        self.test_labels.clear()

        # ---- torch metrics: log the metrics ----
        output = self.test_metrics.compute()
        self.log_dict(output)
        self.test_metrics.reset()
