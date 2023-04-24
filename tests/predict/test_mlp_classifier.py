import torch

from kale.predict.mlp_classifier import MLPClassifier


def test_mlp_classifier():
    # Test 1: Check if the MLPClassifier instance can be created
    in_dim = 10
    hidden_dim = 5
    out_dim = 3
    dropout_rate = 0.1
    model = MLPClassifier(in_dim, hidden_dim, out_dim, dropout_rate)
    assert isinstance(model, MLPClassifier), "Failed to create MLPClassifier instance."

    # Test 2: Check if the MLPClassifier can process input and return output of correct shape
    batch_size = 32
    input_tensor = torch.randn(batch_size, in_dim)
    output = model(input_tensor)
    assert output.shape == (
        batch_size,
        out_dim,
    ), f"Output shape mismatch. Expected: ({batch_size}, {out_dim}), Got: {output.shape}"

    # Test 3: Check if the MLPClassifier returns output without dropout during evaluation mode
    model.eval()
    input_tensor = torch.ones(batch_size, in_dim)
    output = model(input_tensor)
    assert not torch.isnan(output).any(), "Output contains NaN values in evaluation mode."
