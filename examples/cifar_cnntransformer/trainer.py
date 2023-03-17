import time
from copy import deepcopy

import torch
import torch.nn as nn
import pytorch_lightning as pl

#from kale.utils.print import pprint_without_newline, tprint

class CNNTransformerTrainer(pl.LightningModule):
  
    """Pytorch Lightning trainer for cifar-cnntransformer
    Args:
        model (torch.nn.Sequential): model according to the config
        optim (dict): parameters of the model
        cfg: A YACS config object
    """
    def __init__(self, model, optim, cfg):
        super().__init__()
        #self.train_loader = train_loader
        #self.valid_loader = valid_loader
        self.model = model
        self.optim = optim
        self.cfg = cfg
        
        self.loss_fn = nn.NLLLoss()
        self.train_acc, self.valid_acc = [], []
        self.best_valid_acc = 0
        
        self.ave_time = 0
        self.epochs = 1

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
 
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        
        outputs = self.model(x)

        loss = self.loss_fn(outputs, y)
        _, predicted = outputs.max(1)
        acc = (predicted == y).sum().item() / y.size(0)
        
        self.log('train_loss', loss)
        self.log('train_acc', acc * 100, prog_bar=True)


        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        
        outputs = self.model(x)
        loss = self.loss_fn(outputs, y)
        _, predicted = outputs.max(1)
        acc = (predicted == y).sum().item() / y.size(0)
        
        self.log('val_loss', loss)
        self.log('val_acc', acc * 100, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        
        outputs = self.model(x)
        loss = self.loss_fn(outputs, y)
        _, predicted = outputs.max(1)
        acc = (predicted == y).sum().item() / y.size(0)
        
        self.log('val_loss', loss)
        self.log('val_acc', acc * 100, prog_bar=True)
        return loss

    def configure_optimizers(self):
        c = self.cfg
        if c.SOLVER.WARMUP and self.epochs < c.SOLVER.WARMUP_EPOCHS:
            lr = c.SOLVER.BASE_LR * self.epochs / c.SOLVER.WARMUP_EPOCHS
        else:
            # normal (step) scheduling
            lr = c.SOLVER.BASE_LR
            for m_epoch in c.SOLVER.LR_MILESTONES:
                if self.epochs > m_epoch:
                    lr *= c.SOLVER.LR_GAMMA

        for param_group in self.optim["param_groups"]:
            
            param_group['lr'] = lr
            if "scaling" in param_group:
                param_group["lr"] *= param_group["scaling"]
        # set the learning rate scheduler
        #lr_scheduler = None
        #if c.SOLVER.WARMUP:
           # linear warmup
        #    warmup_epochs = c.SOLVER.WARMUP_EPOCHS
        #    def warmup_lr_lambda(current_step):
        #        if current_step < warmup_epochs * len(self.train_loader):
        #            return current_step / (warmup_epochs * len(self.train_loader))
        #        return 1
        #    lr_scheduler = {
        #        'scheduler': torch.optim.lr_scheduler.LambdaLR(self.optim, lr_lambda=warmup_lr_lambda),
        #        'interval': 'step'
        #    }
        
        # set the optimizer
        train_params = self.optim["param_groups"][0]
        train_params_local = deepcopy(train_params)
        try:
            del train_params_local['lr']
        except KeyError:
            pass
        
        optimizer = torch.optim.SGD(self.parameters(), lr=self.optim["param_groups"][0]["lr"],)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.epochs, last_epoch=-1)
        return [optimizer], [scheduler]
