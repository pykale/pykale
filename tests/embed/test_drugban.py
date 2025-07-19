import torch

from kale.embed.model_lib.drugban import DrugBAN
from tests.helpers.mock_graph import create_mock_batch_graph

BATCH_SIZE = 64

CONFIG = {
    "DRUG": {
        "NODE_IN_FEATS": 100,
        "NODE_IN_EMBEDDING": 128,
        "HIDDEN_LAYERS": [128, 64],
        "PADDING": True,
    },
    "PROTEIN": {
        "EMBEDDING_DIM": 128,
        "NUM_FILTERS": [32, 64, 128],
        "KERNEL_SIZE": [3, 3, 3],
        "PADDING": True,
    },
    "DECODER": {
        "IN_DIM": 64,
        "HIDDEN_DIM": 32,
        "OUT_DIM": 16,
        "BINARY": 1,
    },
    "BCN": {
        "HEADS": 8,
    },
}


# Mock Protein Input (protein sequence data)
def create_mock_protein_input(batch_size, sequence_length):
    return torch.randint(0, 25, (batch_size, sequence_length))  # Simulating protein sequences as random integers


# Pytest for DrugBAN initialization
def test_drugban_initialization():
    model = DrugBAN(**CONFIG)
    assert isinstance(model, DrugBAN), "Model should be an instance of DrugBAN"


# Pytest for forward pass in training mode
def test_drugban_forward_train():
    model = DrugBAN(**CONFIG)
    model.eval()

    batch_size = 64
    sequence_length = 200  # Protein sequence length

    # Create mock inputs
    bg_d = create_mock_batch_graph(batch_size, in_feats=CONFIG["DRUG"]["NODE_IN_FEATS"])
    v_p = create_mock_protein_input(batch_size, sequence_length)

    # Forward pass in training mode
    v_d, v_p_out, fused_output, score = model(bg_d, v_p, mode="train")

    assert isinstance(v_d, torch.Tensor), "Drug output should be a tensor"
    assert isinstance(v_p_out, torch.Tensor), "Protein output should be a tensor"
    assert isinstance(fused_output, torch.Tensor), "Fused output should be a tensor"
    assert isinstance(score, torch.Tensor), "Score output should be a tensor"
    assert score.shape[0] == batch_size


def test_drugban_forward_eval_and_invalid():
    model = DrugBAN(**CONFIG)
    model.eval()
    batch_size = 8
    sequence_length = 50
    bg_d = create_mock_batch_graph(batch_size, in_feats=CONFIG["DRUG"]["NODE_IN_FEATS"])
    v_p = create_mock_protein_input(batch_size, sequence_length)
    # Eval mode
    v_d, v_p_out, f, score, att = model(bg_d, v_p, mode="eval")
    assert isinstance(score, torch.Tensor)
    assert isinstance(att, torch.Tensor)
    # Invalid mode
    try:
        model(bg_d, v_p, mode="invalid")
        assert False, "Should raise an error for invalid mode"
    except Exception:
        pass
