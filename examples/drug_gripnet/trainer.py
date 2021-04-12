import time

import numpy as np
import torch
from utils import auprc_auroc_ap, EPS


class Trainer(object):
    def __init__(self, cfg, device, data, model, optim, logger, output_dir):
        # misc
        self.cfg = cfg
        self.device = device
        self.output_dir = output_dir
        self.logger = logger
        # data loader
        self.data = data
        # model setting
        self.model = model
        self.optim = optim
        # ---- loss settings ----
        self.loss = 0
        self.train_auprc, self.train_auroc, self.train_ap = [], [], []
        self.val_auprc, self.val_auroc, self.val_ap = [], [], []
        self.best_ap = 0
        # ---- others ----
        self.epochs = 1
        self.EPS = 1e-13

    def train(self):
        while self.epochs <= self.cfg.SOLVER.MAX_EPOCHS:
            self.adjust_learning_rate()
            self.train_epoch()
            self.val()
            self.epochs += 1

    def train_epoch(self):
        self.model.train()
        self.optim.zero_grad()
        self.loss = 0
        epoch_t = time.time()
        pos_score, neg_score = self.model(
            self.data.g_feat,
            self.data.gg_edge_index,
            self.data.edge_weight,
            self.data.gd_edge_index,
            self.data.train_idx,
            self.data.train_et,
            self.data.train_range,
            self.device,
        )
        pos_loss = -torch.log(pos_score + self.EPS).mean()
        neg_loss = -torch.log(1 - neg_score + self.EPS).mean()
        loss = pos_loss + neg_loss

        loss.backward()
        self.optim.step()

        self.loss = loss.item()
        record = np.zeros((3, self.data.n_dd_edge_type))  # auprc, auroc, ap
        for i in range(self.data.train_range.shape[0]):
            [start, end] = self.data.train_range[i]
            p_s = pos_score[start:end]
            n_s = neg_score[start:end]

            pos_target = torch.ones(p_s.shape[0])
            neg_target = torch.zeros(n_s.shape[0])

            score = torch.cat([p_s, n_s])
            target = torch.cat([pos_target, neg_target])
            # TODO
            if score.tolist() and target.tolist():
                record[0, i], record[1, i], record[2, i] = auprc_auroc_ap(target, score)

        [auprc, auroc, ap] = record.mean(axis=1)
        info_str = "train Epoch: {:3d}\tloss: {:0.4f}\tauprc: {:0.4f}\t auroc: {:0.4f}\tap@50: {:0.4f}s\ttime: {:0.2f}".format(
            self.epochs, self.loss, auprc, auroc, ap, time.time() - epoch_t
        )
        self.logger.info(info_str)
        print(info_str)
        self.train_auprc.append(auprc)
        self.train_auroc.append(auroc)
        self.train_ap.append(ap)

    def val(self):
        self.model.eval()
        self.loss = 0
        epoch_t = time.time()
        pos_score, neg_score = self.model(
            self.data.g_feat,
            self.data.gg_edge_index,
            self.data.edge_weight,
            self.data.gd_edge_index,
            self.data.test_idx,
            self.data.test_et,
            self.data.test_range,
            self.device,
        )
        pos_loss = -torch.log(pos_score + EPS).mean()
        neg_loss = -torch.log(1 - neg_score + EPS).mean()
        loss = pos_loss + neg_loss
        self.loss = loss

        record = np.zeros((3, self.data.n_dd_edge_type))
        for i in range(self.data.test_range.shape[0]):
            [start, end] = self.data.test_range[i]
            p_s = pos_score[start:end]
            n_s = neg_score[start:end]

            pos_target = torch.ones(p_s.shape[0])
            neg_target = torch.zeros(n_s.shape[0])

            score = torch.cat([p_s, n_s])
            target = torch.cat([pos_target, neg_target])
            # TODO
            if score.tolist() and target.tolist():
                record[0, i], record[1, i], record[2, i] = auprc_auroc_ap(target, score)
        [auprc, auroc, ap] = record.mean(axis=1)
        if ap > self.best_ap:
            self.snapshot("best")
        self.snapshot("latest")

        self.best_ap = max(self.best_ap, ap)
        info_str = "valid Epoch: {:3d}\tloss: {:0.4f}\tauprc: {:0.4f}\t auroc: {:0.4f}\tap@50: {:0.4f}\ttime: {:0.2f}s\tbest_ap: {:0.4f}".format(
            self.epochs, self.loss, auprc, auroc, ap, time.time() - epoch_t, self.best_ap
        )
        self.logger.info(info_str)
        print(info_str)
        self.train_auprc.append(auprc)
        self.train_auroc.append(auroc)
        self.train_ap.append(ap)

    def adjust_learning_rate(self):
        c = self.cfg

        # Linear warmup
        if c.SOLVER.WARMUP and self.epochs < c.SOLVER.WARMUP_EPOCHS:
            lr = c.SOLVER.BASE_LR * self.epochs / c.SOLVER.WARMUP_EPOCHS
        else:
            # normal (step) scheduling
            lr = c.SOLVER.BASE_LR
            for m_epoch in c.SOLVER.LR_MILESTONES:
                if self.epochs > m_epoch:
                    lr *= c.SOLVER.LR_GAMMA

        for param_group in self.optim.param_groups:
            param_group["lr"] = lr
            if "scaling" in param_group:
                param_group["lr"] *= param_group["scaling"]

    def snapshot(self, name=None):
        state = {
            "net": self.model.state_dict(),
            "optim": self.optim.state_dict(),
            "epoch": self.epochs,
            "train_auprc": self.train_auprc,
            "val_auprc": self.val_auprc,
            "train_auroc": self.train_auroc,
            "val_auroc": self.val_auroc,
            "train_ap@50": self.train_ap,
            "val_ap@50": self.val_ap,
        }

        if name is None:
            torch.save(state, f"{self.cfg.OUTPUT_DIR}/epoch-{self.epochs}.pt")
        else:
            torch.save(state, f"{self.cfg.OUTPUT_DIR}/{name}.pt")
