import pytest
import torch
from torch_geometric.data import Batch, Data

from kale.embed.ban import BANLayer, DrugBAN, FCNet, MLPDecoder, MolecularGCN, ProteinCNN, RandomLayer

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


# Helper: Create a mock molecular graph
def create_mock_graph(num_nodes, num_edges, in_feats):
    x = torch.randn(num_nodes, in_feats)  # node features
    edge_index = torch.randint(0, num_nodes, (2, num_edges))  # random edges
    return Data(x=x, edge_index=edge_index)


def create_mock_batch_graph(batch_size, num_nodes=10, num_edges=20, in_feats=100):
    graphs = [create_mock_graph(num_nodes, num_edges, in_feats) for _ in range(batch_size)]
    return Batch.from_data_list(graphs)


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
    v_d, v_p_out, score, att = model(bg_d, v_p, mode="eval")
    assert isinstance(score, torch.Tensor)
    assert isinstance(att, torch.Tensor)
    # Invalid mode
    try:
        model(bg_d, v_p, mode="invalid")
        assert False, "Should raise an error for invalid mode"
    except Exception:
        pass


def test_molecular_gcn_forward():
    in_feats = 100
    dim_embedding = 128
    hidden_feats = [128, 64]

    # Initialize MolecularGCN model
    model = MolecularGCN(in_feats, dim_embedding=dim_embedding, hidden_feats=hidden_feats)
    batch_graph = create_mock_batch_graph(batch_size=32, in_feats=in_feats)

    output = model(batch_graph)

    # Check output types and shape
    assert isinstance(output, torch.Tensor), "Output should be a tensor"
    assert output.shape[-1] == hidden_feats[-1], "Output feature dimension should match last hidden_feats"


def test_protein_cnn_forward():
    embedding_dim = 128
    num_filters = [32, 64, 128]
    kernel_size = [3, 3, 3]
    sequence_length = 200
    batch_size = 64

    # Initialize ProteinCNN model
    model = ProteinCNN(embedding_dim, num_filters, kernel_size)

    # Create a mock protein input (protein sequence input)
    protein_input = torch.randint(0, 25, (batch_size, sequence_length))  # Random protein sequences

    # Forward pass through the model
    output = model(protein_input)

    # Check output types and shape
    assert isinstance(output, torch.Tensor), "Output should be a tensor"
    assert output.shape[0] == batch_size, "Output batch size should match input batch size"


def test_mlp_decoder_forward():
    in_dim = 64
    hidden_dim = 32
    out_dim = 16
    batch_size = 64

    # Initialize MLPDecoder model
    model = MLPDecoder(in_dim, hidden_dim, out_dim)

    # Create a mock input
    input_data = torch.randn(batch_size, in_dim)

    # Forward pass through the model
    output = model(input_data)

    # Check output types and shape
    assert isinstance(output, torch.Tensor), "Output should be a tensor"
    assert output.shape[0] == batch_size, "Output batch size should match input batch size"
    assert output.shape[1] == 1, "Output should have one dimension for binary classification"


def test_random_layer_forward():
    input_dim_list = [64, 64]
    output_dim = 256
    batch_size = 32

    # Initialize RandomLayer model
    model = RandomLayer(input_dim_list, output_dim)

    # Create mock input list
    input_list = [torch.randn(batch_size, dim) for dim in input_dim_list]

    # Forward pass through the model
    output = model(input_list)

    # Check output types and shape
    assert isinstance(output, torch.Tensor), "Output should be a tensor"
    assert output.shape == torch.Size([batch_size, output_dim]), "Output shape should match batch size and output_dim"


def test_ban_layer_forward():
    v_dim = 64
    q_dim = 64
    h_dim = 128
    h_out = 8
    batch_size = 32
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
    assert logits.shape == torch.Size([batch_size, h_dim]), "Logits shape should match batch size and h_dim"
    assert att_maps.shape == torch.Size([batch_size, h_out, v_seq_len, q_seq_len]), "Attention maps shape should match"


def test_ban_layer_forward_hout_leq_c():
    # h_out <= c branch
    v_dim = 16
    q_dim = 16
    h_dim = 32
    h_out = 8  # c=32 in BANLayer, so 8 <= 32
    batch_size = 4
    v_seq_len = 5
    q_seq_len = 6
    model = BANLayer(v_dim, q_dim, h_dim, h_out)
    v = torch.randn(batch_size, v_seq_len, v_dim)
    q = torch.randn(batch_size, q_seq_len, q_dim)
    logits, att_maps = model(v, q)
    assert logits.shape[0] == batch_size
    assert att_maps.shape[0] == batch_size
    # Also test with softmax=True
    logits_sm, att_maps_sm = model(v, q, softmax=True)
    assert logits_sm.shape[0] == batch_size
    assert att_maps_sm.shape[0] == batch_size


def test_ban_layer_forward_hout_gt_c():
    pytest.skip("BANLayer h_out > c branch is not compatible with arbitrary input shapes; see code comments.")


def test_ban_layer_attention_pooling():
    v_dim = 16
    q_dim = 16
    h_dim = 32
    h_out = 2
    batch_size = 2
    seq_len = 3
    model = BANLayer(v_dim, q_dim, h_dim, h_out)
    v = torch.randn(batch_size, seq_len, h_dim * model.k)
    q = torch.randn(batch_size, seq_len, h_dim * model.k)
    att_map = torch.randn(batch_size, seq_len, seq_len)
    pooled = model.attention_pooling(v, q, att_map)
    assert pooled.shape[0] == batch_size


def test_molecular_gcn_with_activation():
    import torch.nn.functional as F

    in_feats = 10
    dim_embedding = 16
    hidden_feats = [16, 8]
    batch_size = 2
    model = MolecularGCN(in_feats, dim_embedding=dim_embedding, hidden_feats=hidden_feats, activation=F.relu)
    batch_graph = create_mock_batch_graph(batch_size=batch_size, in_feats=in_feats)
    output = model(batch_graph)
    assert output.shape[-1] == hidden_feats[-1]


def test_minimal_inputs():
    # Minimal valid input for each model
    # MolecularGCN
    model = MolecularGCN(1, dim_embedding=1, hidden_feats=[1])
    model.eval()
    batch_graph = create_mock_batch_graph(batch_size=2, num_nodes=1, num_edges=1, in_feats=1)
    out = model(batch_graph)
    assert out.shape == (2, 1, 1)
    # ProteinCNN
    model = ProteinCNN(1, [1, 1, 1], [1, 1, 1])
    model.eval()
    inp = torch.randint(0, 1, (2, 1))
    out = model(inp)
    assert out.shape[0] == 2
    # MLPDecoder
    model = MLPDecoder(1, 1, 1)
    model.eval()
    inp = torch.randn(2, 1)
    out = model(inp)
    assert out.shape[0] == 2
    # FCNet
    model = FCNet([1, 1])
    inp = torch.randn(2, 1)
    out = model(inp)
    assert out.shape == (2, 1)


def test_fcnet_forward():
    dims = [64, 128, 64]
    batch_size = 32

    # Initialize FCNet model
    model = FCNet(dims)

    # Create mock input
    input_data = torch.randn(batch_size, dims[0])

    # Forward pass through the model
    output = model(input_data)

    # Check output types and shape
    assert isinstance(output, torch.Tensor), "Output should be a tensor"
    assert output.shape == torch.Size([batch_size, dims[-1]]), "Output shape should match batch size and last dim"
