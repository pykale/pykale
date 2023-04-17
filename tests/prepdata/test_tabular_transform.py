import pytest
import torch
import numpy as np

from kale.prepdata.tabular_transform import ToTensor, ToOneHotEncoding


class TestToTensor:
    def test_to_tensor_output(self):
        data = np.array([[.1, .2, .3], [.4, .5, .6], [.7, .8, .9]])
        to_tensor = ToTensor()
        output = to_tensor(data)
        assert isinstance(output, torch.Tensor)

    def test_to_tensor_dtype(self):
        data = np.array([[.1, .2, .3], [.4, .5, .6], [.7, .8, .9]])
        to_tensor = ToTensor(dtype=torch.float32)
        output = to_tensor(data)
        assert output.dtype == torch.float32

    def test_to_tensor_device(self):
        data = np.array([[.1, .2, .3], [.4, .5, .6], [.7, .8, .9]])
        to_tensor = ToTensor(device=torch.device('cpu'))
        output = to_tensor(data)
        assert output.device == torch.device('cpu')


class TestToOneHotEncoding:
    @pytest.mark.parametrize("num_classes", [3, -1])
    def test_onehot_encoding_output(self, num_classes: int):
        labels = [1, 0, 2]
        to_onehot = ToOneHotEncoding(num_classes=num_classes)
        output = to_onehot(labels)
        assert output.tolist() == [[0, 1, 0], [1, 0, 0], [0, 0, 1]]

    @pytest.mark.parametrize("num_classes, shape", [(3, (3, 3)), (4, (3, 4)), (5, (3, 5))])
    def test_onehot_encoding_shape(self, num_classes: int, shape: tuple):
        labels = [1, 0, 2]
        to_onehot = ToOneHotEncoding(num_classes=num_classes)
        output = to_onehot(labels)
        assert output.shape == shape

    def test_onehot_encoding_dtype(self):
        data = [1, 0, 2]
        to_onehot = ToOneHotEncoding(dtype=torch.float32)
        output = to_onehot(data)
        assert output.dtype == torch.float32

    def test_onehot_encoding_device(self):
        data = [1, 0, 2]
        to_onehot = ToOneHotEncoding(device=torch.device('cpu'))
        output = to_onehot(data)
        assert output.device == torch.device('cpu')
