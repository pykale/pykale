import torch
from torch import nn

from kale.pipeline.multimodal_deep_learning import MultiModalDeepLearning


class _TestEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(_TestEncoder, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)


class _TestFusion(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(_TestFusion, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x_concat = torch.cat(x, dim=1)
        return self.fc(x_concat)


class _TestClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(_TestClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


def test_multimodal_module():
    encoders = [_TestEncoder(10, 32), _TestEncoder(15, 32)]
    fusion = _TestFusion(64, 64)
    head = _TestClassifier(64, 3)
    model = MultiModalDeepLearning(encoders, fusion, head)

    input1 = torch.randn(32, 10)
    input2 = torch.randn(32, 15)
    inputs = [input1, input2]
    output = model(inputs)
    assert output.shape == (32, 3)
    assert model.modalities_reps[0].shape == (32, 32)
    assert model.modalities_reps[1].shape == (32, 32)
    assert model.fusion_output.shape == (32, 64)
