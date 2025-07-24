import torch

from kale.embed.cnn import CNNEncoder, ProteinCNN


def test_cnn_encoder():
    num_embeddings, embedding_dim = 64, 128
    sequence_length = 85
    num_filters = 32
    filter_length = 8
    cnn_encoder = CNNEncoder(
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        sequence_length=sequence_length,
        num_kernels=num_filters,
        kernel_length=filter_length,
    ).eval()
    # assert cnn encoder shape
    assert cnn_encoder.embedding.weight.size() == (num_embeddings + 1, embedding_dim)
    assert cnn_encoder.conv1.__repr__() == "Conv1d(85, 32, kernel_size=(8,), stride=(1,))"
    assert cnn_encoder.conv2.__repr__() == "Conv1d(32, 64, kernel_size=(8,), stride=(1,))"
    assert cnn_encoder.conv3.__repr__() == "Conv1d(64, 96, kernel_size=(8,), stride=(1,))"
    assert cnn_encoder.global_max_pool.__repr__() == "AdaptiveMaxPool1d(output_size=1)"

    input_batch = torch.randint(high=num_embeddings, size=(8, sequence_length))
    output_encoding = cnn_encoder(input_batch)
    # assert output shape
    assert output_encoding.size() == (8, 96)


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


def test_protein_cnn_minimal_inputs():
    # ProteinCNN
    model = ProteinCNN(1, [1, 1, 1], [1, 1, 1])
    model.eval()
    inp = torch.randint(0, 1, (2, 1))
    out = model(inp)
    assert out.shape[0] == 2
