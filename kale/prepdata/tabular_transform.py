from typing import Optional, Any

import torch
import torch.nn.functional as F


Tensor = torch.Tensor


class ToTensor(object):
    r"""Convert an array_like data to a tensor of the same shape

    Args:
        dtype (torch.dtype, optional): The desired data type of returned tensor. Default: if :obj:`None`, infers data
            type from data.
        device (torch.device, optional): The device of the constructed tensor. If :obj:`None` and data is a tensor then
            the device of data is used. If None and data is not a tensor then the result tensor is constructed on the
            CPU.
    """

    def __init__(self,
                 dtype: Optional[torch.dtype] = None,
                 device: Optional[torch.device] = None
                 ) -> None:
        self.dtype = dtype
        self.device = device

    def __call__(self, data: Any) -> Tensor:
        """
        Args:
            data (array_like): Initial data for the tensor. Can be a list, tuple, NumPy ndarray, scalar, and other
                types.

        Returns:
            Data converted to tensor.
        """
        return torch.tensor(data, dtype=self.dtype, device=self.device)


class ToOneHotEncoding(object):
    r"""Convert an array_like of class values of shape ``(*,)`` to a tensor of shape ``(*, num_classes)`` that have
    zeros everywhere except where the index of last dimension matches the corresponding value of the input tensor, in
    which case it will be 1.

    Args:
        num_classes (int, optional): Total number of classes. If set to -1, the number of classes will be inferred as
            one greater than the largest class value in the input data.
        dtype (torch.dtype, optional): The desired data type of returned tensor. Default: if :obj:`None`, infers data
            type from data.
        device (torch.device, optional): The device of the constructed tensor. If :obj:`None` and data is a tensor then
            the device of data is used. If None and data is not a tensor then the result tensor is constructed on the
            CPU.
    """

    def __init__(self,
                 num_classes: Optional[int] = -1,
                 dtype: Optional[torch.dtype] = None,
                 device: Optional[torch.device] = None
                 ) -> None:
        self.num_classes = num_classes
        self.dtype = dtype
        self.device = device

    def __call__(self, data: Any) -> Tensor:
        """
        Args:
            data (list): Class values of any shape.

        Returns:
            torch.Tensor: The constructed tensor that has one more dimension with 1 values at the index of last
                dimension indicated by the input, and 0 everywhere else.
        """
        data_tensor = torch.tensor(data, dtype=torch.long)
        data_tensor = F.one_hot(data_tensor, num_classes=self.num_classes)
        return torch.tensor(data_tensor, dtype=self.dtype, device=self.device)
