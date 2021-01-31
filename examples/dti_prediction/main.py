import torch
from graph_dataloader import DTIDataset
from tdc.multi_pred import DTI
from torch_geometric.data import DataLoader
from cnn_dataloader import DTIDeepDataset
from kale.embed.deepdta import MLPDecoder, DrugGCNEncoder, DeepDTAEncoder
from kale.utils.chemchar_label import CHARISOSMILEN, CHARPROTLEN
from config import get_cfg_defaults
from torch.nn import functional as F
import pytorch_lightning as pl

LR = 0.005


class LitGraphDTA(pl.LightningModule):
    def __init__(self, drug_encoding, target_encoding, decoding, **config):
        super().__init__()
        if drug_encoding == "CNN":
            self.drug_encoder = DeepDTAEncoder(num_embeddings=CHARISOSMILEN, embedding_dim=config["DRUG_DIM"],
                                               sequence_length=config["DRUG_LENGTH"], num_kernels=config["NUM_FILTERS"],
                                               kernel_length=config["DRUG_FILTER_LENGTH"])
        elif drug_encoding == "GCN":
            self.drug_encoder = DrugGCNEncoder()

        if target_encoding == "CNN":
            self.target_encoder = DeepDTAEncoder(num_embeddings=CHARPROTLEN, embedding_dim=config["TARGET_DIM"],
                                                 sequence_length=config["TARGET_LENGTH"], num_kernels=config["NUM_FILTERS"],
                                                 kernel_length=config["TARGET_FILTER_LENGTH"])

        if decoding == "MLP":
            self.decoder = MLPDecoder(in_dim=config["DECODER_IN_DIM"], hidden_dim=config["MLP_HIDDEN_DIM"],
                                      out_dim=config["MLP_OUT_DIM"], dropout=config["MLP_DROPOUT"])

        self.drug_encoding = drug_encoding
        self.target_encoding = target_encoding
        self.decoding = decoding

    def forward(self, x_d, x_t, **kwargs):
        if self.drug_encoding == "CNN":
            drug_emb = self.drug_encoder(x_d)
        elif self.drug_encoding == "GCN":
            drug_emb = self.drug_encoder(x_d, kwargs["edge_index"], kwargs["batch"])

        if self.target_encoding == "CNN":
            target_emb = self.target_encoder(x_t)
        com_emb = torch.cat((drug_emb, target_emb), dim=1)
        output = self.decoder(com_emb)
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=LR)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        if self.drug_encoding == "CNN":
            x_d, x_t, y = train_batch
            y_hat = self(x_d, x_t)
        elif self.drug_encoding == "GCN":
            x_d, x_t, edge_index, batch = train_batch.x, train_batch.target, train_batch.edge_index, train_batch.batch
            y = train_batch.y
            y_hat = self(x_d, x_t, edge_index, batch)
        loss = F.mse_loss(y_hat, y.view(-1, 1))
        self.log("train_loss", loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        if self.drug_encoding == "CNN":
            x_d, x_t, y = val_batch
            y_hat = self(x_d, x_t)
        elif self.drug_encoding == "GCN":
            x_d, x_t, edge_index, batch = val_batch.x, val_batch.target, val_batch.edge_index, val_batch.batch
            y = val_batch.y
            y_hat = self(x_d, x_t, edge_index, batch)
        loss = F.mse_loss(y_hat, y.view(-1, 1))
        self.log("val_loss", loss)
        return loss

    def test_step(self, test_batch, batch_idx):
        if self.drug_encoding == "CNN":
            x_d, x_t, y = test_batch
            y_hat = self(x_d, x_t)
        elif self.drug_encoding == "GCN":
            x_d, x_t, edge_index, batch = test_batch.x, test_batch.target, test_batch.edge_index, test_batch.batch
            y = test_batch.y
            y_hat = self(x_d, x_t, edge_index, batch)
        loss = F.mse_loss(y_hat, y.view(-1, 1))
        self.log("test_loss", loss)
        return loss


if __name__ == "__main__":
    cfg = get_cfg_defaults()
    train_dataset = DTIDeepDataset(dataset="DAVIS", split="train")
    val_dataset = DTIDeepDataset(dataset="DAVIS", split="valid")
    test_dataset = DTIDeepDataset(dataset="DAVIS", split="test")

    train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=cfg.SOLVER.TRAIN_BATCH_SIZE)
    val_loader = DataLoader(dataset=val_dataset, shuffle=True, batch_size=cfg.SOLVER.TEST_BATCH_SIZE)
    test_loader = DataLoader(dataset=test_dataset, shuffle=True, batch_size=cfg.SOLVER.TEST_BATCH_SIZE)

    model = LitGraphDTA(drug_encoding=cfg.MODEL.DRUG_ENCODER, target_encoding=cfg.MODEL.TARGET_ENCODER,
                        decoding=cfg.MODEL.DECODER, **cfg.MODEL)
    trainer = pl.Trainer(max_epochs=2)
    trainer.fit(model, train_dataloader=train_loader, val_dataloaders=val_loader)
    trainer.test(model, test_dataloaders=test_loader)



