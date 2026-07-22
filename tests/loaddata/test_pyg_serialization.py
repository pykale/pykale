import pytest
import torch
from torch_geometric.data import Data

from kale.loaddata.pyg_serialization import get_pyg_safe_globals, load_pyg_data


class Exploit:
    """Stands in for a malicious payload.

    On an unrestricted unpickle its ``__reduce__`` runs ``eval`` and writes ``marker_path``, giving the
    test an observable side effect to assert against. Under ``weights_only=True`` ``eval`` is not in the
    allow-list, so the load is rejected and the marker is never written.
    """

    def __init__(self, marker_path):
        self.marker_path = str(marker_path)

    def __reduce__(self):
        return (eval, (f"__import__('pathlib').Path(r'{self.marker_path}').write_text('executed')",))


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
    container types needed by Data must not reopen the door to arbitrary code execution. The payload is
    written with ``torch.save`` so it travels the same ``torch.load`` path the loader uses, and the
    marker assertion proves the payload was rejected rather than executed.
    """
    marker = tmp_path / "marker.txt"
    path = tmp_path / "payload.pt"
    torch.save(Exploit(marker), path)

    with pytest.raises(Exception):
        load_pyg_data(str(path))

    assert not marker.exists(), "the payload executed - the safe loader failed to reject it"


def test_pyg_safe_globals_contain_no_callables_that_execute():
    """The allow-list holds only container/array types, never arbitrary builtins."""
    forbidden = {"eval", "exec", "system", "Popen", "apply"}

    for allowed in get_pyg_safe_globals():
        assert getattr(allowed, "__name__", "") not in forbidden
