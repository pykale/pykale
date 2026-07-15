import pickle

import pytest
import torch
from torch_geometric.data import Data

from kale.utils.serialization import get_pyg_safe_globals, load_pyg_data


class Exploit:
    """Stands in for a malicious payload; ``os.system`` would run on an unrestricted unpickle."""

    def __reduce__(self):
        return (eval, ("__import__('pathlib').Path(MARKER).write_text('executed')",))


def test_load_pyg_data_reads_data_object(tmp_path):
    """A saved Data object round-trips through the safe loader."""
    path = tmp_path / "data.pt"
    torch.save(Data(x=torch.rand(3, 2), edge_index=torch.tensor([[0, 1], [1, 2]])), path)

    loaded = load_pyg_data(str(path))

    assert isinstance(loaded, Data)
    assert loaded.x.shape == (3, 2)


def test_load_pyg_data_reads_list_of_data_objects(tmp_path):
    """Multiomics stores a list of Data objects, which must load as a list."""
    path = tmp_path / "data.pt"
    torch.save([Data(x=torch.rand(2, 2)), Data(x=torch.rand(2, 2))], path)

    loaded = load_pyg_data(str(path))

    assert isinstance(loaded, list)
    assert all(isinstance(item, Data) for item in loaded)


def test_load_pyg_data_rejects_arbitrary_objects(tmp_path):
    """A class outside the allow-list is refused rather than reconstructed.

    This is the guarantee that separates the safe loader from weights_only=False: allow-listing the
    container types needed by Data must not reopen the door to arbitrary code execution.
    """
    path = tmp_path / "payload.pt"
    with open(path, "wb") as handle:
        pickle.dump(Exploit(), handle)

    with pytest.raises(Exception) as excinfo:
        load_pyg_data(str(path))

    assert "eval" in str(excinfo.value) or "Unsupported" in str(excinfo.value) or "Weights only" in str(excinfo.value)


def test_pyg_safe_globals_contain_no_callables_that_execute():
    """The allow-list holds only container/array types, never arbitrary builtins."""
    forbidden = {"eval", "exec", "system", "Popen", "apply"}

    for allowed in get_pyg_safe_globals():
        assert getattr(allowed, "__name__", "") not in forbidden
