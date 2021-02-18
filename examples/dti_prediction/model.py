import pytorch_lightning as pl
import torch
from torch.nn import functional as F

from kale.embed.deepdta import DeepDTAEncoder, MLPDecoder


class LitDeepDTA(pl.LightningModule):
    def __init__(self, num_drug_embeddings, drug_dim, drug_length, num_filters, drug_filter_length,
                 num_target_embeddings, target_dim, target_length, target_filter_length, decoder_in_dim,
                 decoder_hidden_dim, decoder_out_dim, dropout):
        super().__init__()
        self.drug_encoder = DeepDTAEncoder(num_embeddings=num_drug_embeddings, embedding_dim=drug_dim,
                                           sequence_length=drug_length, num_kernels=num_filters,
                                           kernel_length=drug_filter_length)

        self.target_encoder = DeepDTAEncoder(num_embeddings=num_target_embeddings, embedding_dim=target_dim,
                                             sequence_length=target_length, num_kernels=num_filters,
                                             kernel_length=target_filter_length)
        self.decoder = MLPDecoder(in_dim=decoder_in_dim, hidden_dim=decoder_hidden_dim,
                                  out_dim=decoder_out_dim, dropout=dropout)

    def forward(self, x_d, x_t):
        drug_emb = self.drug_encoder(x_d)
        target_emb = self.target_encoder(x_t)
        com_emb = torch.cat((drug_emb, target_emb), dim=1)
        output = self.decoder(com_emb)
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x_d, x_t, y = train_batch
        y_hat = self(x_d, x_t)
        loss = F.mse_loss(y_hat, y.view(-1, 1))
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x_d, x_t, y = val_batch
        y_hat = self(x_d, x_t)
        loss = F.mse_loss(y_hat, y.view(-1, 1))
        self.log("val_loss", loss, on_step=False, on_epoch=True)

    def test_step(self, test_batch, batch_idx):
        x_d, x_t, y = test_batch
        y_hat = self(x_d, x_t)
        loss = F.mse_loss(y_hat, y.view(-1, 1))
        self.log("test_loss", loss, on_step=False, on_epoch=True)

def get_model(cfg):
    # ---- encoder hyper-parameter ----
    num_drug_embeddings = cfg.MODEL.NUM_SMILE_CHAR
    num_target_embeddings = cfg.MODEL.NUM_ATOM_CHAR
    drug_dim = cfg.MODEL.DRUG_DIM
    target_dim = cfg.MODEL.TARGET_DIM
    drug_length = cfg.MODEL.DRUG_LENGTH
    target_length = cfg.MODEL.TARGET_LENGTH
    num_filters = cfg.MODEL.NUM_FILTERS
    drug_filter_length = cfg.MODEL.DRUG_FILTER_LENGTH
    target_filter_length = cfg.MODEL.TARGET_FILTER_LENGTH

    # ---- decoder hyper-parameter ----
    decoder_in_dim = cfg.MODEL.MLP_IN_DIM
    decoder_hidden_dim = cfg.MODEL.MLP_HIDDEN_DIM
    decoder_out_dim = cfg.MODEL.MLP_OUT_DIM
    dropout = cfg.MODEL.MLP_DROPOUT

    model = LitDeepDTA(num_drug_embeddings, drug_dim, drug_length, num_filters, drug_filter_length,
                       num_target_embeddings, target_dim, target_length, target_filter_length, decoder_in_dim,
                       decoder_hidden_dim, decoder_out_dim, dropout)

    return model
