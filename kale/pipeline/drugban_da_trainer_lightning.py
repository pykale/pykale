import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn import functional as F

from kale.embed.ban import RandomLayer
from kale.predict.losses import binary_cross_entropy, cross_entropy_logits
from kale.pipeline.domain_adapter import GradReverse as ReverseLayerF

from torchmetrics import MetricCollection, AUROC, Precision

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

    def configure_optimizers(self):
        """If domain adaptation is used"""
        # Initialize the optimizer for the DrugBAN model
        # optimizer0 = torch.optim.Adam(self.model.parameters(), lr=self.solver_lr)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.solver_lr)
        # Initialize the optimizer for the Domain Discriminator model
        # optimizer_da1 = torch.optim.Adam(self.domain_dmm.parameters(), lr=self.solver_da_lr)
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
            n = F.softmax(score, dim=1)[:, 1]
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