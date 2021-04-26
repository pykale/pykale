import pytest
import torch
from torch.utils.data import DataLoader

from kale.embed.seq_nn import CNNEncoder
from kale.loaddata.tdc_datasets import BindingDBDataset
from kale.pipeline.deep_dti import BaseDTATrainer, DeepDTATrainer
from kale.predict.decode import MLPDecoder

DATASET = "BindingDB_Kd"


def test_deep_data(download_path):
    test_dataset = BindingDBDataset(name=DATASET, split="test", path=download_path)
    test_batch = next(iter(DataLoader(dataset=test_dataset, shuffle=True, batch_size=32)))

    drug_encoder = CNNEncoder(num_embeddings=64, embedding_dim=128, sequence_length=85, num_kernels=32, kernel_length=8)
    target_encoder = CNNEncoder(
        num_embeddings=25, embedding_dim=128, sequence_length=1200, num_kernels=32, kernel_length=8
    )
    decoder = MLPDecoder(in_dim=192, hidden_dim=16, out_dim=16)
    # test deep_dta trainer
    save_parameters = {"seed": 2020, "batch_size": 256}
    model = DeepDTATrainer(drug_encoder, target_encoder, decoder, lr=0.001, **save_parameters).eval()
    assert isinstance(model.drug_encoder, CNNEncoder)
    assert isinstance(model.target_encoder, CNNEncoder)
    assert isinstance(model.decoder, MLPDecoder)
    model.configure_optimizers()
    assert torch.is_tensor(model.validation_step(test_batch, 0))
    assert torch.is_tensor(model.test_step(test_batch, 0))
    with pytest.raises(AttributeError) as excinfo:
        model.training_step(test_batch, 0)
        assert "log_metrics" in str(excinfo.value)

    # test base_dta trainer
    model = BaseDTATrainer(drug_encoder, target_encoder, decoder, lr=0.001, **save_parameters).eval()
    with pytest.raises(NotImplementedError) as excinfo:
        model.training_step(test_batch, 0)
        assert "Forward pass needs to be defined" in str(excinfo.value)
