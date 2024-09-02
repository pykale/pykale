import torch

from kale.embed.drugban import BANLayer, DrugBAN, FCNet, MLPDecoder, MolecularGCN, ProteinCNN, RandomLayer

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

# Test data
GRAPH_INPUT = torch.randn(BATCH_SIZE, 100)  # Example graph input for MolecularGCN
PROTEIN_INPUT = torch.randint(0, 25, (BATCH_SIZE, 200))  # Example protein sequence input for ProteinCNN


def test_drugban_shapes():
    model = DrugBAN(**CONFIG)
    model.eval()
    drug_output, protein_output, fused_output, score = model(GRAPH_INPUT, PROTEIN_INPUT)
    assert drug_output.shape == (BATCH_SIZE, CONFIG["DRUG"]["HIDDEN_LAYERS"][-1]), "Unexpected drug output shape"
    assert protein_output.shape == (
        BATCH_SIZE,
        PROTEIN_INPUT.size(1),
        CONFIG["PROTEIN"]["NUM_FILTERS"][-1],
    ), "Unexpected protein output shape"
    assert fused_output.shape == (BATCH_SIZE, CONFIG["DECODER"]["IN_DIM"]), "Unexpected fused output shape"
    assert score.shape == (BATCH_SIZE, CONFIG["DECODER"]["BINARY"]), "Unexpected score output shape"


def test_moleculargcn_shapes():
    model = MolecularGCN(
        in_feats=CONFIG["DRUG"]["NODE_IN_FEATS"],
        dim_embedding=CONFIG["DRUG"]["NODE_IN_EMBEDDING"],
        padding=CONFIG["DRUG"]["PADDING"],
        hidden_feats=CONFIG["DRUG"]["HIDDEN_LAYERS"],
    )
    model.eval()
    output = model(GRAPH_INPUT)
    assert output.shape == (
        BATCH_SIZE,
        GRAPH_INPUT.size(1),
        CONFIG["DRUG"]["HIDDEN_LAYERS"][-1],
    ), "Unexpected MolecularGCN output shape"


def test_proteincnn_shapes():
    model = ProteinCNN(
        embedding_dim=CONFIG["PROTEIN"]["EMBEDDING_DIM"],
        num_filters=CONFIG["PROTEIN"]["NUM_FILTERS"],
        kernel_size=CONFIG["PROTEIN"]["KERNEL_SIZE"],
        padding=CONFIG["PROTEIN"]["PADDING"],
    )
    model.eval()
    output = model(PROTEIN_INPUT)
    assert output.shape == (
        BATCH_SIZE,
        PROTEIN_INPUT.size(1),
        CONFIG["PROTEIN"]["NUM_FILTERS"][-1],
    ), "Unexpected ProteinCNN output shape"


def test_mlpdecoder_shapes():
    mlp_input = torch.randn(BATCH_SIZE, CONFIG["DECODER"]["IN_DIM"])
    model = MLPDecoder(
        in_dim=CONFIG["DECODER"]["IN_DIM"],
        hidden_dim=CONFIG["DECODER"]["HIDDEN_DIM"],
        out_dim=CONFIG["DECODER"]["OUT_DIM"],
        binary=CONFIG["DECODER"]["BINARY"],
    )
    model.eval()
    output = model(mlp_input)
    assert output.shape == (BATCH_SIZE, CONFIG["DECODER"]["BINARY"]), "Unexpected MLPDecoder output shape"


def test_randomlayer_shapes():
    input_dims = [CONFIG["DECODER"]["IN_DIM"], CONFIG["DECODER"]["IN_DIM"]]
    inputs = [torch.randn(BATCH_SIZE, dim) for dim in input_dims]
    model = RandomLayer(input_dim_list=input_dims, output_dim=256)
    model.eval()
    output = model(inputs)
    assert output.shape == (BATCH_SIZE, 256), "Unexpected RandomLayer output shape"


def test_banlayer_shapes():
    v_dim = CONFIG["DRUG"]["HIDDEN_LAYERS"][-1]
    q_dim = CONFIG["PROTEIN"]["NUM_FILTERS"][-1]
    h_dim = CONFIG["DECODER"]["IN_DIM"]
    h_out = CONFIG["BCN"]["HEADS"]
    v = torch.randn(BATCH_SIZE, 10, v_dim)
    q = torch.randn(BATCH_SIZE, 15, q_dim)
    model = BANLayer(v_dim=v_dim, q_dim=q_dim, h_dim=h_dim, h_out=h_out)
    model.eval()
    logits, att_maps = model(v, q)
    assert logits.shape == (BATCH_SIZE, h_dim), "Unexpected BANLayer logits shape"
    assert att_maps.shape == (BATCH_SIZE, h_out, 10, 15), "Unexpected BANLayer attention maps shape"


def test_fcnet_shapes():
    dims = [128, 64, 32, 16]
    model = FCNet(dims=dims, act="ReLU", dropout=0.2)
    x = torch.randn(BATCH_SIZE, dims[0])
    model.eval()
    output = model(x)
    assert output.shape == (BATCH_SIZE, dims[-1]), "Unexpected FCNet output shape"
