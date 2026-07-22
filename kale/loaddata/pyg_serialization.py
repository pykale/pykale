"""Helpers for deserializing datasets saved with ``torch.save``.

``torch.load(..., weights_only=False)`` unpickles arbitrary Python objects and can therefore execute
arbitrary code from a malicious or substituted file. Loading with ``weights_only=True`` removes that
risk, but it also refuses any class that is not a tensor by default, so objects such as PyTorch
Geometric ``Data`` cannot be rebuilt without explicitly trusting the classes they are made of.

This module keeps that trusted set in one place, so loaders can stay on the safe ``weights_only=True``
path while still reading the graph datasets shipped with PyKale.
"""

from typing import Any, List

import torch


def get_pyg_safe_globals() -> List[Any]:
    """Classes required to rebuild PyTorch Geometric ``Data`` objects under ``weights_only=True``.

    Only container and array types are included. None of them execute code on unpickling, so
    allow-listing them keeps the guarantees of ``weights_only=True``: any other class in the file is
    still rejected.

    Returns:
        list: Classes to pass to :func:`torch.serialization.safe_globals`.
    """
    import numpy as np
    import torch_geometric.data.data as pyg_data
    from torch_geometric.data.storage import GlobalStorage

    allowed: List[Any] = [pyg_data.Data, GlobalStorage, np.ndarray, np.dtype]

    # numpy's array-reconstruction helper and its concrete dtype classes live in internal modules
    # whose import paths differ across numpy versions. Resolve them defensively: if one is absent, that
    # payload simply cannot be rebuilt (a clear load-time error), which is far better than an
    # ImportError that would break every dataset loader that imports this module.
    try:
        from numpy._core.multiarray import _reconstruct

        allowed.append(_reconstruct)
    except ImportError:  # pragma: no cover - depends on the installed numpy version
        pass

    # The concrete dtype class depends on the stored array, so allow the whole family rather than
    # chase them one at a time. They describe layout only and cannot execute code.
    try:
        import numpy.dtypes as np_dtypes

        allowed.extend(
            getattr(np_dtypes, name) for name in dir(np_dtypes) if isinstance(getattr(np_dtypes, name, None), type)
        )
    except ImportError:  # pragma: no cover - depends on the installed numpy version
        pass

    # Present only in newer PyTorch Geometric releases, so resolve them defensively.
    for name in ("DataEdgeAttr", "DataTensorAttr"):
        attribute = getattr(pyg_data, name, None)
        if attribute is not None:
            allowed.append(attribute)

    return allowed


def load_pyg_data(path: str, **kwargs) -> Any:
    """Load a PyTorch Geometric object saved with ``torch.save``, without allowing code execution.

    Args:
        path (str): Path of the file to load.
        **kwargs: Additional keyword arguments forwarded to :func:`torch.load`. Any ``weights_only``
            value is ignored; this loader always uses ``weights_only=True``.

    Returns:
        The deserialized object, typically a ``Data`` instance or a list of them.

    Example:
        >>> data = load_pyg_data("data.pt")
    """
    # weights_only is fixed to True so the safe path cannot be overridden by a caller; dropping any
    # supplied value also avoids "got multiple values for keyword argument" from torch.load.
    kwargs.pop("weights_only", None)
    with torch.serialization.safe_globals(get_pyg_safe_globals()):
        return torch.load(path, weights_only=True, **kwargs)
