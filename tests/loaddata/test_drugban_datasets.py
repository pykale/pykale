import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Batch, Data

from kale.loaddata.drugban_datasets import DTIDataset, graph_collate_func, MultiDataLoader


@pytest.fixture(scope="module")
def sample_data():
    # Create a sample DataFrame for testing
    data = {
        "SMILES": ["CCO", "CCN", "CCC", "CCCl"],
        "Protein": ["MTEYK", "GAGDE", "VKHG", "KRTG"],
        "Y": [1.0, 0.0, 1.0, 0.0],
    }
    return pd.DataFrame(data)


@pytest.fixture(scope="module")
def dataset(sample_data):
    list_IDs = list(range(len(sample_data)))
    return DTIDataset(list_IDs, sample_data, max_drug_nodes=50)


@pytest.fixture
def dataloader(dataset):
    return DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=graph_collate_func)


@pytest.fixture
def multidataloader(dataloader):
    return MultiDataLoader(dataloaders=[dataloader, dataloader], n_batches=2)


def test_dataset_length(dataset):
    assert len(dataset) == 4


def test_dataset_item(dataset):
    graph, protein, label = dataset[0]

    assert isinstance(graph, Data)
    assert isinstance(protein, np.ndarray)
    assert isinstance(label, float)

    assert graph.x.shape[0] == dataset.max_drug_nodes  # Includes virtual node padding
    assert graph.edge_attr.ndim == 2
    assert graph.edge_index.shape[0] == 2
    assert protein.shape[0] > 0


def test_graph_collate_func(dataloader):
    for batch in dataloader:
        graphs, proteins, labels = batch

        assert isinstance(graphs, Batch)
        assert isinstance(proteins, torch.Tensor)
        assert isinstance(labels, torch.Tensor)

        assert proteins.shape[0] == 2
        assert labels.shape[0] == 2
        break


def test_multidataloader_length(multidataloader):
    assert len(multidataloader) == 2


def test_multidataloader_iteration(multidataloader):
    for batch_group in multidataloader:
        assert len(batch_group) == 2

        for graph, protein, label in batch_group:
            assert isinstance(graph, Batch)
            assert isinstance(protein, torch.Tensor)
            assert isinstance(label, torch.Tensor)
        break


def test_multidataloader_with_zero_batches():
    with pytest.raises(ValueError, match="n_batches should be > 0"):
        MultiDataLoader(dataloaders=[], n_batches=0)
