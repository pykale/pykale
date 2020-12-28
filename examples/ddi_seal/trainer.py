from tqdm import tqdm
import torch
from torch.nn import BCEWithLogitsLoss


class Trainer(object):
    def __init__(self, cfg, model, emb, train_loader, val_loader, test_loader, device, optim, evaluator):
        self.cfg = cfg
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.optim = optim
        self.emb = emb
        self.evaluator = evaluator
        self.epochs = 1

    def train(self):
        while self.epochs <= self.cfg.SOLVER.MAX_EPOCHS:
            self.train_epoch()
            if self.epochs % 5 == 0:
                self.val_epoch()
            self.epochs += 1

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        pbar = tqdm(self.train_loader)
        for batch in pbar:
            batch = batch.to(self.device)
            self.optim.zero_grad()
            node_id = batch.node_id if self.emb else None
            logits = self.model(batch.z, batch.edge_index, batch.batch, node_id)
            loss = BCEWithLogitsLoss()(logits.view(-1), batch.y.to(torch.float))
            loss.backward()
            self.optim.step()
            total_loss += loss.item() * batch.num_graphs
        total_loss /= len(self.train_loader.dataset)
        info_str = 'train Epoch: {:3d}\tloss: {:0.4f}'.format(self.epochs, total_loss)
        print(info_str)

    @torch.no_grad()
    def val_epoch(self):
        self.model.eval()
        y_pred, y_true = [], []
        val_loss, test_loss = 0, 0
        for batch in tqdm(self.val_loader):
            batch = batch.to(self.device)
            node_id = batch.node_id if self.emb else None
            logits = self.model(batch.z, batch.edge_index, batch.batch, node_id)
            loss = BCEWithLogitsLoss()(logits.view(-1), batch.y.to(torch.float))
            val_loss += loss.item() * batch.num_graphs
            y_pred.append(logits.view(-1).cpu())
            y_true.append(batch.y.view(-1).cpu().to(torch.float))
        val_loss /= len(self.val_loader.dataset)
        val_pred, val_true = torch.cat(y_pred), torch.cat(y_true)
        pos_val_pred = val_pred[val_true == 1]
        neg_val_pred = val_pred[val_true == 0]

        y_pred, y_true = [], []
        for batch in tqdm(self.test_loader):
            batch = batch.to(self.device)
            node_id = batch.node_id if self.emb else None
            logits = self.model(batch.z, batch.edge_index, batch.batch, node_id)
            loss = BCEWithLogitsLoss()(logits.view(-1), batch.y.to(torch.float))
            test_loss += loss.item() * batch.num_graphs
            y_pred.append(logits.view(-1).cpu())
            y_true.append(batch.y.view(-1).cpu().to(torch.float))
        test_loss /= len(self.test_loader.dataset)
        test_pred, test_true = torch.cat(y_pred), torch.cat(y_true)
        pos_test_pred = test_pred[test_true == 1]
        neg_test_pred = test_pred[test_true == 0]
        results = self.evaluate_hits(pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred)
        info_str = 'val Epoch: {:3d}\tval loss: {:0.4f}\ttest loss: {:0.4f}'.format(self.epochs, val_loss, test_loss)
        print(info_str)

    def evaluate_hits(self, pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred):
        results = {}
        for K in [20, 50, 100]:
            self.evaluator.K = K
            valid_hits = self.evaluator.eval({
                'y_pred_pos': pos_val_pred,
                'y_pred_neg': neg_val_pred,
            })[f'hits@{K}']
            test_hits = self.evaluator.eval({
                'y_pred_pos': pos_test_pred,
                'y_pred_neg': neg_test_pred,
            })[f'hits@{K}']

            results[f'Hits@{K}'] = (valid_hits, test_hits)

        return results
