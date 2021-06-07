import torch


def get_class_subset(dataset, class_ids):
    """
    Args:
        dataset: a torch.utils.data.Dataset
        class_ids (list, optional): List of chosen subset of class ids.
    Returns: a torch.utils.data.Dataset
        Dataset: a torch.utils.data.Dataset with only classes in class_ids
    """
    sub_indices = [i for i in range(0, len(dataset)) if dataset[i][1] in class_ids]
    return torch.utils.data.Subset(dataset, sub_indices)
