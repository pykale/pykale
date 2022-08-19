import pytest
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from kale.embed.seq_nn import CNNEncoder
from kale.loaddata.tdc_datasets import BindingDBDataset
from kale.pipeline.deepdta import BaseDTATrainer, DeepDTATrainer
from kale.predict.decode import MLPDecoder

DATASET = "BindingDB_Kd"


def test_deep_data(download_path):
    test_dataset = BindingDBDataset(name=DATASET, split="test", path=download_path)
    subset_indices = list(range(0, 32, 2))
    test_subset = torch.utils.data.Subset(test_dataset, subset_indices)
    test_dataloader = DataLoader(dataset=test_subset, shuffle=False, batch_size=8)
    valid_dataloader = DataLoader(dataset=test_subset, shuffle=True, batch_size=8)
    train_dataloader = DataLoader(dataset=test_subset, shuffle=True, batch_size=4)
    test_batch = next(iter(test_dataloader))

    drug_encoder = CNNEncoder(num_embeddings=64, embedding_dim=128, sequence_length=85, num_kernels=32, kernel_length=8)
    target_encoder = CNNEncoder(
        num_embeddings=25, embedding_dim=128, sequence_length=1200, num_kernels=32, kernel_length=8
    )
    decoder = MLPDecoder(in_dim=192, hidden_dim=16, out_dim=16)
    # test deep_dta trainer
    save_parameters = {"seed": 2020, "batch_size": 256}
    model = DeepDTATrainer(drug_encoder, target_encoder, decoder, lr=0.001, ci_metric=True, **save_parameters).eval()
    trainer = pl.Trainer(max_epochs=1, gpus=0)
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)
    trainer.test(dataloaders=test_dataloader)
    assert isinstance(model.drug_encoder, CNNEncoder)
    assert isinstance(model.target_encoder, CNNEncoder)
    assert isinstance(model.decoder, MLPDecoder)
    # model.configure_optimizers()

    # test base_dta trainer
    model = BaseDTATrainer(drug_encoder, target_encoder, decoder, lr=0.001, ci_metric=True, **save_parameters).eval()
    with pytest.raises(NotImplementedError) as excinfo:
        model.training_step(test_batch, 0)
        assert "Forward pass needs to be defined" in str(excinfo.value)
