import torch
from create_dataloader import DTIDataset
from tdc.multi_pred import DTI
from torch_geometric.data import DataLoader
from model import DrugGCNEncoder, TargetConvEncoder, MLPDecoder
from torch.nn import functional as F
import pytorch_lightning as pl

TRAIN_BATCH_SIZE, TEST_BATCH_SIZE = 512, 512
NUM_EPOCH = 100
LR = 0.005


class LitGraphDTA(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.drug_encoder = DrugGCNEncoder()
        self.target_encoder = TargetConvEncoder()
        self.mlp_decoder = MLPDecoder()

    def forward(self, x_d, x_t, edge_index, batch):
        drug_emb = self.drug_encoder(x_d, edge_index, batch)
        target_emb = self.target_encoder(x_t)
        com_emb = torch.cat((drug_emb, target_emb), dim=1)
        output = self.mlp_decoder(com_emb)
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=LR)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x_d, x_t, edge_index, batch = train_batch.x, train_batch.target, train_batch.edge_index, train_batch.batch
        y_hat = self(x_d, x_t, edge_index, batch)
        loss = F.mse_loss(y_hat, train_batch.y.view(-1, 1))
        self.log("train_loss", loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x_d, x_t, edge_index, batch = val_batch.x, val_batch.target, val_batch.edge_index, val_batch.batch
        y_hat = self(x_d, x_t, edge_index, batch)
        loss = F.mse_loss(y_hat, val_batch.y.view(-1, 1))
        self.log("val_loss", loss)
        return loss

    def test_step(self, test_batch, batch_idx):
        x_d, x_t, edge_index, batch = test_batch.x, test_batch.target, test_batch.edge_index, test_batch.batch
        y_hat = self(x_d, x_t, edge_index, batch)
        loss = F.mse_loss(y_hat, test_batch.y.view(-1, 1))
        self.log("test_loss", loss)
        return loss


if __name__ == "__main__":
    dataset = "DAVIS"
    data = DTI(name=dataset)
    split = data.get_split()
    train_data = DTIDataset(dataset=dataset + f"_train", root="data")
    valid_data = DTIDataset(dataset=dataset + f"_valid", root="data")
    test_data = DTIDataset(dataset=dataset + f"_test", root="data")

    train_loader = DataLoader(dataset=train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(dataset=valid_data, batch_size=TEST_BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(dataset=test_data, batch_size=TEST_BATCH_SIZE, shuffle=False)

    model = LitGraphDTA()
    trainer = pl.Trainer(max_epochs=2)
    trainer.fit(model, train_dataloader=train_loader, val_dataloaders=valid_loader)
    trainer.test(model, test_dataloaders=test_loader)
