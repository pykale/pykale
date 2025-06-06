import torch


def setup_device():
    """
    Sets up the computing device for PyTorch operations.

    Returns
    -------
    torch.device
        A CUDA device if available, otherwise CPU.

    Notes
    -----
    Also calls `torch.cuda.empty_cache()` to clear any cached GPU memory.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    return device
