# Created by Haiping Lu from modifying https://github.com/HaozhiQi/ISONet/blob/master/isonet/trainer.py
# Under the MIT License
import time
import torch
import torch.nn as nn
from kale.utils.print import tprint, pprint_without_newline
from config import C


class Trainer(object):
    def __init__(self, device, train_loader, val_loader, model, optim, logger, output_dir):
        # misc
        self.device = device
        self.output_dir = output_dir
        # data loader
        self.train_loader = train_loader
        self.val_loader = val_loader
        # nn setting
        self.model = model
        self.optim = optim
        # lr setting
        self.criterion = nn.CrossEntropyLoss()
        # training loop settings
        self.epochs = 1
        # loss settings
        self.train_acc, self.val_acc = [], []
        self.best_valid_acc = 0
        self.ce_loss, self.ortho_loss = 0, 0
        # others
        self.ave_time = 0
        self.logger = logger

    def train(self):
        while self.epochs <= C.SOLVER.MAX_EPOCHS:
            self.adjust_learning_rate()
            self.train_epoch()
            self.val()
            self.epochs += 1

    def train_epoch(self):
        self.model.train()
        self.ce_loss = 0
        self.ortho_loss = 0
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
            loss = self.loss(outputs, targets)
            loss.backward()
            self.optim.step()

            _, predicted = outputs.max(1)
            total += batch_size

            correct += predicted.eq(targets).sum().item()

            self.ave_time += time.time() - iter_t
            tprint(f'train Epoch: {self.epochs} | {batch_idx + 1} / {len(self.train_loader)} | '
                   f'Acc: {100. * correct / total:.3f} | CE: {self.ce_loss / (batch_idx + 1):.3f} | '
                   f'O: {self.ortho_loss / (batch_idx + 1):.3f} | time: {self.ave_time / (batch_idx + 1):.3f}s')

        info_str = f'train Epoch: {self.epochs} | Acc: {100. * correct / total:.3f} | ' \
                   f'CE: {self.ce_loss / (batch_idx + 1):.3f} | ' \
                   f'time: {time.time() - epoch_t:.2f}s |'
        self.logger.info(info_str)
        pprint_without_newline(info_str)
        self.train_acc.append(100. * correct / total)

    def val(self):
        self.model.eval()
        self.ce_loss = 0
        self.ortho_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.val_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.loss(outputs, targets)

                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        if 100. * correct / total > self.best_valid_acc:
            self.snapshot('best')
        self.snapshot('latest')
        self.best_valid_acc = max(self.best_valid_acc, 100. * correct / total)
        info_str = f'valid | Acc: {100. * correct / total:.3f} | ' \
                   f'CE: {self.ce_loss / len(self.val_loader):.3f} | ' \
                   f'O: {self.ortho_loss / len(self.val_loader):.3f} | ' \
                   f'best: {self.best_valid_acc:.3f} | '
        print(info_str)
        self.logger.info(info_str)
        self.val_acc.append(100. * correct / total)

    def loss(self, outputs, targets):
        loss = self.criterion(outputs, targets)
        self.ce_loss += loss.item()

        if C.ISON.ORTHO_COEFF > 0:
            o_loss = self.model.module.ortho(self.device)
            self.ortho_loss += o_loss.item()
            loss += o_loss * C.ISON.ORTHO_COEFF
        return loss

    def adjust_learning_rate(self):
        # if do linear warmup
        if C.SOLVER.WARMUP and self.epochs < C.SOLVER.WARMUP_EPOCH:
            lr = C.SOLVER.BASE_LR * self.epochs / C.SOLVER.WARMUP_EPOCH
        else:
            # normal (step) scheduling
            lr = C.SOLVER.BASE_LR
            for m_epoch in C.SOLVER.LR_MILESTONES:
                if self.epochs > m_epoch:
                    lr *= C.SOLVER.LR_GAMMA

        for param_group in self.optim.param_groups:
            param_group['lr'] = lr
            if 'scaling' in param_group:
                param_group['lr'] *= param_group['scaling']

    def snapshot(self, name=None):
        state = {
            'net': self.model.state_dict(),
            'optim': self.optim.state_dict(),
            'epoch': self.epochs,
            'train_accuracy': self.train_acc,
            'test_accuracy': self.val_acc
        }
        if name is None:
            torch.save(state, f'{self.output_dir}/{self.epochs}.pt')
        else:
            torch.save(state, f'{self.output_dir}/{name}.pt')
