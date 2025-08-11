from typing import Any, Dict

import pytest
import torch

from kale.embed.attention import BANLayer
from kale.embed.model_lib.drugban import BCNConfig, DecoderConfig, DrugBAN, DrugConfig, FullConfig, ProteinConfig
from tests.helpers.mock_graph import create_mock_batch_graph

BATCH_SIZE = 4

CONFIG: Dict[str, Dict[str, Any]] = {
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


@pytest.fixture(scope="module")
def model_config() -> FullConfig:
    return FullConfig(
        DRUG=DrugConfig(**CONFIG["DRUG"]),
        PROTEIN=ProteinConfig(**CONFIG["PROTEIN"]),
        DECODER=DecoderConfig(**CONFIG["DECODER"]),
        BCN=BCNConfig(**CONFIG["BCN"]),
    )


# Mock Protein Input (protein sequence data)
def create_mock_protein_input(batch_size, sequence_length):
    return torch.randint(0, 25, (batch_size, sequence_length))  # Simulating protein sequences as random integers


# Pytest for DrugBAN initialization
def test_drugban_initialization(model_config):
    model = DrugBAN(model_config)
    assert isinstance(model, DrugBAN), "Model should be an instance of DrugBAN"


# Pytest for forward pass in training mode
def test_drugban_forward_train(model_config):
    model = DrugBAN(model_config)
    model.eval()

    batch_size = BATCH_SIZE
    sequence_length = 200  # Protein sequence length

    # Create mock inputs
    bg_d = create_mock_batch_graph(batch_size, in_feats=model_config.DRUG.NODE_IN_FEATS)
    v_p = create_mock_protein_input(batch_size, sequence_length)

    # Forward pass in training mode
    v_d, v_p_out, fused_output, score = model(bg_d, v_p, mode="train")

    assert isinstance(v_d, torch.Tensor), "Drug output should be a tensor"
    assert isinstance(v_p_out, torch.Tensor), "Protein output should be a tensor"
    assert isinstance(fused_output, torch.Tensor), "Fused output should be a tensor"
    assert isinstance(score, torch.Tensor), "Score output should be a tensor"
    assert score.shape[0] == batch_size


def test_drugban_forward_eval_and_invalid(model_config):
    model = DrugBAN(model_config)
    model.eval()
    batch_size = BATCH_SIZE
    sequence_length = 50
    bg_d = create_mock_batch_graph(batch_size, in_feats=model_config.DRUG.NODE_IN_FEATS)
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


def test_ban_layer_forward():
    v_dim = 64
    q_dim = 64
    h_dim = 128
    h_out = 8
    batch_size = BATCH_SIZE
    v_seq_len = 10
    q_seq_len = 12

    # Initialize BANLayer model
    model = BANLayer(v_dim, q_dim, h_dim, h_out)

    # Create mock input tensors for v and q
    v = torch.randn(batch_size, v_seq_len, v_dim)
    q = torch.randn(batch_size, q_seq_len, q_dim)

    # Forward pass through the model
    logits, att_maps = model(v, q)

    # Check output types and shape
    assert isinstance(logits, torch.Tensor), "Logits should be a tensor"
    assert isinstance(att_maps, torch.Tensor), "Attention maps should be a tensor"
    assert logits.shape == torch.Size([batch_size, h_dim]), "Logits shape should match batch size and hidden_dim"
    assert att_maps.shape == torch.Size([batch_size, h_out, v_seq_len, q_seq_len]), "Attention maps shape should match"


def test_ban_layer_attention_pooling():
    v_dim = 16
    q_dim = 16
    h_dim = 32
    h_out = 2
    batch_size = BATCH_SIZE
    seq_len = 3
    model = BANLayer(v_dim, q_dim, h_dim, h_out)
    v = torch.randn(batch_size, seq_len, h_dim * model.num_att_maps)
    q = torch.randn(batch_size, seq_len, h_dim * model.num_att_maps)
    att_map = torch.randn(batch_size, seq_len, seq_len)
    pooled = model.attention_pooling(v, q, att_map)
    assert pooled.shape[0] == batch_size


@pytest.mark.parametrize(
    "v_dim, q_dim, hidden_dim, num_out_heads, num_att_maps, batch_size, seq_len_v, seq_len_q",
    [
        # This set triggers the `else` branch (uses `h_net`)
        (32, 32, 16, 64, 3, 2, 10, 12),
        # This set triggers the `if num_out_heads <= self.c` branch (uses h_mat + einsum)
        (16, 16, 32, 8, 3, 4, 5, 6),
    ],
)
def test_ban_layer_forward_hout_leq_c(
    v_dim, q_dim, hidden_dim, num_out_heads, num_att_maps, batch_size, seq_len_v, seq_len_q
):
    model = BANLayer(v_dim, q_dim, hidden_dim, num_out_heads=num_out_heads, num_att_maps=num_att_maps)
    v = torch.randn(batch_size, seq_len_v, v_dim)
    q = torch.randn(batch_size, seq_len_q, q_dim)
    logits, att_maps = model(v, q)
    assert logits.shape[0] == batch_size
    assert att_maps.shape[0] == batch_size
    # Also test with softmax=True
    logits_sm, att_maps_sm = model(v, q, softmax=True)
    assert logits_sm.shape[0] == batch_size
    assert att_maps_sm.shape[0] == batch_size
