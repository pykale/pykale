# Created by Raivo Koot from modifying https://github.com/HaozhiQi/ISONet/blob/master/isonet/trainer.py
# Under the MIT License
import time

import torch
import torch.nn as nn

from kale.utils.print import pprint_without_newline, tprint


class Trainer(object):
    def __init__(self, device, train_loader, val_loader, model, optim, logger, cfg):
        self.device = device

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.optim = optim
        self.logger = logger
        self.cfg = cfg

        # ---- loss settings ----
        self.criterion = nn.NLLLoss()
        self.loss = 0
        self.train_acc, self.val_acc = [], []
        self.best_val_acc = 0

        # ---- others ----
        self.ave_time = 0
        self.epochs = 1

    def train(self):
        while self.epochs <= self.cfg.SOLVER.MAX_EPOCHS:
            self.adjust_learning_rate()
            self.train_epoch()
            self.val()
            self.epochs += 1

    def train_epoch(self):
        self.model.train()
        self.loss = 0
        self.ave_time = 0
        correct = 0
        total = 0
        epoch_t = time.time()

        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            iter_t = time.time()
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optim.zero_grad()
            batch_size = inputs.shape[0]

            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optim.step()

            _, predicted = outputs.max(1)
            self.loss += loss.item()
            total += batch_size

            correct += predicted.eq(targets).sum().item()

            self.ave_time += time.time() - iter_t
            tprint(
                f"train Epoch: {self.epochs} | {batch_idx + 1} / {len(self.train_loader)} | "
                f"Acc: {100. * correct / total:.3f} | Loss: {self.loss / (batch_idx + 1):.3f} | "
                f"time: {self.ave_time / (batch_idx + 1):.3f}s"
            )

            info_str = (
                f"train Epoch: {self.epochs} | Acc: {100. * correct / total:.3f} | "
                f"Loss: {self.loss / (batch_idx + 1):.3f} | "
                f"time: {time.time() - epoch_t:.2f}s |"
            )

        self.logger.info(info_str)
        pprint_without_newline(info_str)
        self.train_acc.append(100.0 * correct / total)

    def val(self):
        self.model.eval()
        self.loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.val_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)

                loss = self.criterion(outputs, targets)
                self.loss = loss

                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        if 100.0 * correct / total > self.best_val_acc:
            self.snapshot("best")
        self.snapshot("latest")

        self.best_val_acc = max(self.best_val_acc, 100.0 * correct / total)
        info_str = (
            f"valid | Acc: {100. * correct / total:.3f} | "
            f"Loss: {self.loss / len(self.val_loader):.3f} | "
            f"best: {self.best_val_acc:.3f} | "
        )

        print(info_str)
        self.logger.info(info_str)
        self.val_acc.append(100.0 * correct / total)

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
            "train_accuracy": self.train_acc,
            "test_accuracy": self.val_acc,
        }

        if name is None:
            torch.save(state, f"{self.cfg.OUTPUT_DIR}/epoch-{self.epochs}.pt")
        else:
            torch.save(state, f"{self.cfg.OUTPUT_DIR}/{name}.pt")
