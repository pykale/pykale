"""
Functions for manipulating/transforming tabular data
"""
# =============================================================================
# Author: Sina Tabakhi, sina.tabakhi@gmail.com
#         Lawrance Schobs, lawrenceschobs@gmail.com
# =============================================================================

import os
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor


class ToTensor(object):
    r"""Convert an array_like data to a tensor of the same shape. This class provides a callable object that allows
    instances of the class to be called as a function. In other words, this class wraps the functionality of
    `torch.tensor <https://pytorch.org/docs/stable/generated/torch.tensor.html>`__ and allows users to use it as a
    callable instance.

    Args:
        dtype (torch.dtype, optional): The desired data type of returned tensor. Default: if :obj:`None`, infers data
            type from data.
        device (torch.device, optional): The device of the constructed tensor. If :obj:`None` and data is a tensor then
            the device of data is used. If None and data is not a tensor then the result tensor is constructed on the
            CPU.
    """

    def __init__(self, dtype: Optional[torch.dtype] = None, device: Optional[torch.device] = None) -> None:
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

    Note that this class provides a callable object that allows instances of the class to be called as a function. In
    other words, this class wraps the functionality of the one_hot method in the
    `PyTorch <https://pytorch.org/docs/stable/generated/torch.nn.functional.one_hot.html>`__ and allows users to use it
     as a callable instance.

    Args:
        num_classes (int, optional): Total number of classes. If set to -1, the number of classes will be inferred as
            one greater than the largest class value in the input data.
        dtype (torch.dtype, optional): The desired data type of returned tensor. Default: if :obj:`None`, infers data
            type from data.
        device (torch.device, optional): The device of the constructed tensor. If :obj:`None` and data is a tensor then
            the device of data is used. If None and data is not a tensor then the result tensor is constructed on the
            CPU.
    """

    def __init__(
        self,
        num_classes: Optional[int] = -1,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
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


def apply_confidence_inversion(data: pd.DataFrame, uncertainty_measure: str) -> Tuple[Any, Any]:
    """Invert a list of numbers, add a small number to avoid division by zero.

    Args:
        data (Dict): Dictionary of data to invert.
        uncertainty_measure (str): Key of dict to invert.

    Returns:
        Dict: Dictionary with inverted data.
    """

    if uncertainty_measure not in data:
        raise KeyError("The key %s not in the dictionary provided" % uncertainty_measure)

    # Make sure no value is less than zero.
    min_not_zero = min(i for i in data[uncertainty_measure] if i > 0)
    data.loc[data[uncertainty_measure] < 0, uncertainty_measure] = min_not_zero
    data[uncertainty_measure] = 1 / data[uncertainty_measure] + 0.0000000000001
    return data


def generate_struct_for_qbin(
    models_to_compare: List[str], targets: List[int], saved_bins_path_pre: str, dataset: str
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """
    Returns dictionaries of pandas dataframes for:
        a) all error and prediction info (all prediction data across targets for each model),
        b) target indices for separated error and prediction info (prediction data for each model and each target),
        c) all estimated error bounds (estimated error bounds across targets for each model),
        d) target separated estimated error bounds (estimated error bounds for each model and each target).

    Args:
        models_to_compare: List of set models to add to data struct.
        targets: List of targets to add to data struct.
        saved_bins_path_pre: Preamble to path of where the predicted quantile bins are saved.
        dataset: String of what dataset you're measuring.

    Returns:
        data_structs: Dictionary where keys are model names and values are pandas dataframes containing
                      all prediction data across targets for that model.

        data_struct_sep: Dictionary where keys are a combination of model names and target indices (e.g., "model1 T1"),
                         and values are pandas dataframes containing prediction data for the corresponding model and target.

        data_struct_bounds: Dictionary where keys are a combination of model names and the string " Error Bounds"
                            (e.g., "model1 Error Bounds"), and values are pandas dataframes containing all estimated
                            error bounds across targets for that model.

        data_struct_bounds_sep: Dictionary where keys are a combination of model names, target indices and the string
                                "Error Bounds" (e.g., "model1 Error Bounds L1"), and values are pandas dataframes containing
                                estimated error bounds for the corresponding model and target.
    """
    data_structs = {}
    data_struct_sep = {}  #

    data_struct_bounds = {}
    data_struct_bounds_sep = {}

    for model in models_to_compare:
        all_targets = []
        all_err_bounds = []

        for target_idx in targets:
            bin_pred_path = os.path.join(saved_bins_path_pre, model, dataset, "res_predicted_bins_t" + str(target_idx))
            bin_preds = pd.read_csv(bin_pred_path + ".csv", header=0)
            bin_preds["target_idx"] = target_idx

            error_bounds_path = os.path.join(
                saved_bins_path_pre, model, dataset, "estimated_error_bounds_t" + str(target_idx)
            )
            error_bounds_pred = pd.read_csv(error_bounds_path + ".csv", header=0)
            error_bounds_pred["target"] = target_idx

            all_targets.append(bin_preds)
            all_err_bounds.append(error_bounds_pred)
            data_struct_sep[model + " L" + str(target_idx)] = bin_preds
            data_struct_bounds_sep[model + "Error Bounds L" + str(target_idx)] = error_bounds_pred

        data_structs[model] = pd.concat(all_targets, axis=0, ignore_index=True)
        data_struct_bounds[model + " Error Bounds"] = pd.concat(all_err_bounds, axis=0, ignore_index=True)

    return data_structs, data_struct_sep, data_struct_bounds, data_struct_bounds_sep
