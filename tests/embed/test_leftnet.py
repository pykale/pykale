# test_leftnet.py
from types import SimpleNamespace

import pytest
import torch

# Adjust import path to where you placed the modules
from kale.embed.model_lib.LEFTNet import LEFTNetProp, LEFTNetZ, NeighborEmb


def _make_dummy_leftnet_batch(num_graphs=2, nodes_per_graph=(3, 2), with_atom_fea=False, atom_fea_dim=10):
    """Minimal batch namespace expected by LEFTNet models."""
    torch.manual_seed(0)
    total_nodes = sum(nodes_per_graph)

    # Place each graph's atoms in a small cluster so radius_graph will find neighbors
    positions = []
    batch_idx = []
    base = 0.0
    for g, n in enumerate(nodes_per_graph):
        # offset each graph in space so they're distinct (even though batch_idx also isolates them)
        center = torch.tensor([10.0 * g, 0.0, 0.0])
        local = torch.randn(n, 3) * 0.5 + center
        positions.append(local)
        batch_idx.append(torch.full((n,), g, dtype=torch.long))
        base += n
    positions = torch.cat(positions, dim=0)  # [N, 3]
    batch_idx = torch.cat(batch_idx, dim=0)  # [N]

    atom_num = torch.randint(low=1, high=30, size=(total_nodes,), dtype=torch.long)

    ns = SimpleNamespace(
        positions=positions,
        batch_idx=batch_idx,
        atom_num=atom_num,
    )

    if with_atom_fea:
        ns.atom_fea = torch.randn(total_nodes, atom_fea_dim)

    return ns


@pytest.mark.parametrize("readout", ["mean", "sum"])
def test_leftnetz_forward_backward(readout):
    # LEFTNetZ uses one-hot of size 95 internally → set atom_fea_dim accordingly
    model = LEFTNetZ(
        atom_fea_dim=95,  # must match the one-hot length in prop_setup
        num_targets=1,
        cutoff=3.0,
        num_layers=1,
        hidden_channels=16,
        num_radial=8,
        readout=readout,
        y_mean=0.0,
        y_std=1.0,
    )

    batch = _make_dummy_leftnet_batch(num_graphs=2, nodes_per_graph=(3, 2))
    out = model(batch)

    assert model.radial_lin[0].__repr__() == "Linear(in_features=8, out_features=16, bias=True)"
    assert model.radial_lin[1].__repr__() == "SiLU(inplace=True)"
    assert model.radial_lin[2].__repr__() == "Linear(in_features=16, out_features=16, bias=True)"
    assert isinstance(model.neighbor_emb, NeighborEmb)
    assert hasattr(model, "S_vector") and isinstance(model.S_vector, torch.nn.Module)
    assert model.lin[0].__repr__() == "Linear(in_features=3, out_features=4, bias=True)"
    assert model.lin[1].__repr__() == "SiLU(inplace=True)"
    assert model.lin[2].__repr__() == "Linear(in_features=4, out_features=1, bias=True)"
    assert model.last_layer.__repr__() == "Linear(in_features=16, out_features=1, bias=True)"

    # shape: [num_graphs, num_targets]
    assert isinstance(out, torch.Tensor)
    assert out.shape == (2, 1)

    # Backward pass works
    loss = out.pow(2).mean()
    loss.backward()
    assert any(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_leftnetprop_forward_backward():
    atom_fea_dim = 10
    model = LEFTNetProp(
        atom_fea_dim=atom_fea_dim,
        num_targets=1,
        cutoff=3.0,
        num_layers=1,
        hidden_channels=16,
        num_radial=8,
        readout="mean",
        y_mean=0.0,
        y_std=1.0,
    )

    batch = _make_dummy_leftnet_batch(
        num_graphs=3, nodes_per_graph=(2, 3, 2), with_atom_fea=True, atom_fea_dim=atom_fea_dim
    )
    out = model(batch)

    assert isinstance(out, torch.Tensor)
    assert out.shape == (3, 1)

    loss = out.sum()
    loss.backward()
    assert any(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_leftnetz_prop_setup_shape():
    model = LEFTNetZ(atom_fea_dim=95, num_targets=2, num_layers=1, hidden_channels=8, num_radial=4)
    batch = _make_dummy_leftnet_batch(num_graphs=1, nodes_per_graph=(5,))
    one_hot = model.prop_setup(batch, device=next(model.parameters()).device)
    assert one_hot.shape == (5, 95)  # 95-class one-hot encoding


def test_leftnetprop_prop_setup_passthrough():
    atom_fea_dim = 12
    model = LEFTNetProp(atom_fea_dim=atom_fea_dim, num_targets=2, num_layers=1, hidden_channels=8, num_radial=4)
    batch = _make_dummy_leftnet_batch(num_graphs=1, nodes_per_graph=(5,), with_atom_fea=True, atom_fea_dim=atom_fea_dim)
    fea = model.prop_setup(batch, device=next(model.parameters()).device)
    assert fea.shape == (5, atom_fea_dim)
