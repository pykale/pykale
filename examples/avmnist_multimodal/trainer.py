import torch
import torch.nn as nn
import sklearn.metrics
from kale.evaluate.robustness import relative_robustness, effective_robustness, single_plot

class Trainer:
    def __init__(self, device, model, train_dataloader, valid_dataloader, test_dataloader, total_epochs,
                 is_packed=False, early_stop=True, optimtype=torch.optim.RMSprop, lr=0.001,
                 weight_decay=0.0, objective=nn.CrossEntropyLoss(), save='best.pt',clip_val=8):
        """
    Handle running a simple supervised training loop.

    :param total_epochs: maximum number of epochs to train
    :param is_packed: whether the input modalities are packed in one list or not (default is False, which means we expect input of [tensor(20xmodal1_size),(20xmodal2_size),(20xlabel_size)] for batch size 20 and 2 input modalities)
    :param early_stop: whether to stop early if valid performance does not improve over 7 epochs
    :param optimtype: type of optimizer to use
    :param lr: learning rate
    :param weight_decay: weight decay of optimizer
    :param objective: objective function, which is either one of CrossEntropyLoss, MSELoss or BCEWithLogitsLoss or a custom objective function that takes in three arguments: prediction, ground truth, and an argument dictionary.
    :param save: the name of the saved file for the model with current best validation performance
    :param validtime: whether to show valid time in seconds or not
    :param clip_val: grad clipping limit

    """
        self.device = device
        self.model = model
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader
        self.total_epochs = total_epochs
        self.is_packed = is_packed
        self.early_stop = early_stop
        self.optimtype = optimtype
        self.lr = lr
        self.weight_decay = weight_decay
        self.objective = objective
        self.save = save
        self.clip_val = clip_val

    def train(self):
        op = self.optimtype([p for p in self.model.parameters() if p.requires_grad], lr=self.lr, weight_decay=self.weight_decay)
        bestacc = 0
        patience = 0

        for epoch in range(self.total_epochs):
            totalloss = 0.0
            totals = 0
            self.model.train()
            for j in self.train_dataloader:
                op.zero_grad()
                if self.is_packed:
                    with torch.backends.cudnn.flags(enabled=False):
                        self.model.train()
                        out = self.model([[i.float().to(self.device) for i in j[0]], j[1]])
                else:
                    self.model.train()
                    out = self.model([i.float().to(self.device) for i in j[:-1]])

                loss = self.deal_with_objective(out, j[-1])

                totalloss += loss * len(j[-1])
                totals += len(j[-1])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_val)
                op.step()
            print("Epoch "+str(epoch)+" train loss: "+str(totalloss/totals))

            self.model.eval()
            with torch.no_grad():
                totalloss = 0.0
                pred = []
                true = []
                for j in self.valid_dataloader:
                    if self.is_packed:
                        out = self.model([[i.float().to(self.device) for i in j[0]], j[1]])
                    else:
                        out = self.model([i.float().to(self.device) for i in j[:-1]])
                    loss = self.deal_with_objective(out, j[-1])
                    totalloss += loss*len(j[-1])

                    pred.append(torch.argmax(out, 1))

                    true.append(j[-1])

            if pred:
                pred = torch.cat(pred, 0)

            true = torch.cat(true, 0)
            totals = true.shape[0]
            valloss = totalloss/totals

            acc = sklearn.metrics.accuracy_score(true.cpu().numpy(), pred.cpu().numpy())
            print("Epoch "+str(epoch)+" valid loss: "+str(valloss) + " acc: "+str(acc))
            if acc > bestacc:
                patience = 0
                bestacc = acc
            else:
                patience += 1

            if self.early_stop and patience > 7:
                break

    def deal_with_objective(self, pred, truth):
        """Alter inputs depending on objective function, to deal with different objective arguments."""
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if type(self.objective) == nn.CrossEntropyLoss:
            if len(truth.size()) == len(pred.size()):
                truth1 = truth.squeeze(len(pred.size())-1)
            else:
                truth1 = truth
            return self.objective(pred, truth1.long().to(self.device))
        elif type(self.objective) == nn.MSELoss or type(self.objective) == nn.modules.loss.BCEWithLogitsLoss or type(self.objective) == nn.L1Loss:
            return self.objective(pred, truth.float().to(self.device))
        else:
            return self.objective(pred, truth)

    def single_test(self):
        """Run single test for model.
        Args:
            model (nn.Module): Model to test
            test_dataloader (torch.utils.data.Dataloader): Test dataloader
            is_packed (bool, optional): Whether the input data is packed or not. Defaults to False.
            criterion (_type_, optional): Loss function. Defaults to nn.CrossEntropyLoss().
        """

        with torch.no_grad():
            totalloss = 0.0
            pred = []
            true = []
            for j in self.test_dataloader:
                self.model.eval()
                if self.is_packed:
                    out = self.model([[i.float().to(self.device)
                                for i in j[0]], j[1]])
                else:
                    out = self.model([i.float().float().to(self.device)
                                for i in j[:-1]])
                if type(self.objective) == torch.nn.modules.loss.BCEWithLogitsLoss or type(self.objective) == torch.nn.MSELoss:
                    loss = self.objective(out, j[-1].float().to(self.device))

                elif type(self.objective) == nn.CrossEntropyLoss:
                    if len(j[-1].size()) == len(out.size()):
                        truth1 = j[-1].squeeze(len(out.size())-1)
                    else:
                        truth1 = j[-1]
                    loss = self.objective(out, truth1.long().to(self.device))
                else:
                    loss = self.objective(out, j[-1].to(self.device))
                totalloss += loss*len(j[-1])

                pred.append(torch.argmax(out, 1))

                true.append(j[-1])

            if pred:
                pred = torch.cat(pred, 0)
            true = torch.cat(true, 0)

            accuracy = sklearn.metrics.accuracy_score(true.cpu().numpy(), pred.cpu().numpy())
            print("Test acc: "+str(accuracy))
            return {'Accuracy': accuracy}

