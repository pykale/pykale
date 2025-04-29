import dgl
import numpy as np
import pandas as pd
import pytest
import torch
from dgl import DGLGraph
from torch.utils.data import DataLoader

from kale.loaddata.drugban_datasets import DTIDataset, graph_collate_func, MultiDataLoader


@pytest.fixture(scope="module")
def sample_data():
    # Create a sample DataFrame for testing
    data = {
        "SMILES": ["CCO", "CCN", "CCC", "CCCl"],  # Example SMILES strings
        "Protein": ["MTEYK", "GAGDE", "VKHG", "KRTG"],  # Example protein sequences
        "Y": [1.0, 0.0, 1.0, 0.0],  # Example labels
    }
    df = pd.DataFrame(data)
    return df


@pytest.fixture(scope="module")
def dataset(sample_data):
    # Create a DTIDataset instance for testing
    list_IDs = list(range(len(sample_data)))
    return DTIDataset(list_IDs, sample_data)


@pytest.fixture
def dataloader(dataset):
    # Create a DataLoader for the DTIDataset
    return DataLoader(dataset, batch_size=2, collate_fn=graph_collate_func, shuffle=True)


@pytest.fixture
def multidataloader(dataloader):
    # Create a MultiDataLoader for testing with two instances of the same DataLoader
    return MultiDataLoader(dataloaders=[dataloader, dataloader], n_batches=2)


def test_dataset_length(dataset):
    # Test the length of the dataset
    assert len(dataset) == 4  # We have 4 entries in sample_data


def test_dataset_item(dataset):
    # Test retrieval of a single item from the dataset
    v_d, v_p, y = dataset[0]

    assert isinstance(v_d, DGLGraph)  # The drug representation should be a DGLGraph
    assert isinstance(v_p, np.ndarray)  # The protein sequence should be a numpy array
    assert isinstance(y, float)  # The label should be a float

    # Check if the DGLGraph has the expected number of nodes
    assert v_d.number_of_nodes() == dataset.max_drug_nodes  # Should have max_drug_nodes nodes due to virtual nodes

    # Check if the protein sequence tensor has the correct shape
    assert v_p.shape[0] > 0  # Ensure the protein sequence was correctly converted to a tensor of integers


def test_graph_collate_func(dataloader):
    # Test the graph_collate_func within the DataLoader
    for batch in dataloader:
        dgl_graphs, protein_tensors, labels = batch

        assert isinstance(dgl_graphs, dgl.DGLGraph)  # The batched drug graphs should be a single DGLGraph
        assert isinstance(protein_tensors, torch.Tensor)  # The batched protein sequences should be a tensor
        assert isinstance(labels, torch.Tensor)  # The batched labels should be a tensor

        assert dgl_graphs.batch_size == 2  # Ensure the batch size is correct
        assert protein_tensors.shape[0] == 2  # Ensure the protein tensor batch size matches
        assert labels.shape[0] == 2  # Ensure the labels tensor batch size matches

        break  # Only check the first batch to keep the test efficient


def test_multidataloader_length(multidataloader):
    # Test the length of the MultiDataLoader
    assert len(multidataloader) == 2  # We specified n_batches=2


def test_multidataloader_iteration(multidataloader):
    # Test iterating over MultiDataLoader
    for batch in multidataloader:
        assert len(batch) == 2  # Two batches (from two DataLoaders)
        assert isinstance(batch[0], tuple)  # Each batch should be a tuple from the DataLoader
        assert isinstance(batch[0][0], dgl.DGLGraph)  # First element in tuple should be a DGLGraph
        assert isinstance(batch[0][1], torch.Tensor)  # Second element should be a Tensor (protein sequences)
        assert isinstance(batch[0][2], torch.Tensor)  # Third element should be a Tensor (labels)

        assert isinstance(batch[1], tuple)  # Repeat for the second DataLoader
        assert isinstance(batch[1][0], dgl.DGLGraph)
        assert isinstance(batch[1][1], torch.Tensor)
        assert isinstance(batch[1][2], torch.Tensor)

        break  # Only check the first batch to keep the test efficient


def test_multidataloader_with_zero_batches():
    # Test MultiDataLoader initialization with zero batches, expecting an exception
    with pytest.raises(ValueError, match="n_batches should be > 0"):
        MultiDataLoader(dataloaders=[], n_batches=0)
