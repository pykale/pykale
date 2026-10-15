from types import SimpleNamespace

import pytest
import torch

# Adjust the import to match your actual module path
from kale.embed.model_lib.cartnet import CartNet, CartNetLayer, GeometricGraphEncoder


def _make_cartnet_batch(num_nodes=10, num_graphs=3, atom_max=30):
    """Minimal batch matching CartNet/GeometricGraphEncoder expectations."""
    torch.manual_seed(0)
    pos = torch.randn(num_nodes, 3)  # positions
    batch_idx = torch.arange(num_nodes) % num_graphs
    atom_num = torch.randint(1, atom_max, (num_nodes,))  # categorical atom types
    target = torch.randn(num_graphs, 1)

    return SimpleNamespace(positions=pos, batch_idx=batch_idx, atom_num=atom_num, target=target)


def test_geometric_graph_encoder_shapes():
    dim_in, dim_rbf, radius = 32, 16, 4.0
    enc = GeometricGraphEncoder(
        dim_in=dim_in, dim_rbf=dim_rbf, radius=radius, invariant=False, temperature=False, atom_types=True
    )

    batch = _make_cartnet_batch(num_nodes=12, num_graphs=4)
    out = enc(batch)

    # basic attributes present
    assert hasattr(out, "x")
    assert hasattr(out, "edge_index")
    assert hasattr(out, "edge_attr")
    assert hasattr(out, "cart_dist")
    assert hasattr(out, "cart_dir")

    # shapes consistent
    atom = out.x.shape[0]
    edge = out.edge_index.shape[1]
    assert out.x.shape == (atom, dim_in)
    assert out.edge_index.shape == (2, edge)
    assert out.edge_attr.shape[0] == edge and out.edge_attr.shape[1] == dim_in
    assert out.cart_dist.shape == (edge,)
    assert out.cart_dir.shape == (edge, 3)

    # finite values
    assert torch.isfinite(out.x).all()
    assert torch.isfinite(out.edge_attr).all()
    assert torch.isfinite(out.cart_dist).all()


@pytest.mark.parametrize("use_envelope", [True, False])
def test_cartnet_layer_step_shapes(use_envelope):
    dim_in, radius = 32, 4.0
    # Build encoded batch first
    enc = GeometricGraphEncoder(
        dim_in=dim_in, dim_rbf=16, radius=radius, invariant=False, temperature=False, atom_types=True
    )
    batch = _make_cartnet_batch(num_nodes=9, num_graphs=3)
    batch = enc(batch)

    layer = CartNetLayer(dim_in=dim_in, radius=radius, use_envelope=use_envelope)

    x_prev = batch.x.clone()
    e_prev = batch.edge_attr.clone()
    out = layer(batch)

    # shapes preserved
    atom = out.x.shape[0]
    edge = out.edge_index.shape[1]
    assert out.x.shape == (atom, dim_in)
    assert out.edge_attr.shape == (edge, dim_in)

    # values finite
    assert torch.isfinite(out.x).all()
    assert torch.isfinite(out.edge_attr).all()

    # layer has an effect (allow tiny numerical differences)
    assert torch.sum(torch.abs(out.x - x_prev)) > 0
    assert torch.sum(torch.abs(out.edge_attr - e_prev)) > 0


def test_cartnet():
    dim_in, dim_rbf, num_layers, radius = 32, 12, 3, 4.0
    model = CartNet(
        dim_in=dim_in,
        dim_rbf=dim_rbf,
        num_layers=num_layers,
        radius=radius,
        invariant=False,
        temperature=False,
        use_envelope=True,
        atom_types=True,
    )

    batch = _make_cartnet_batch(num_nodes=15, num_graphs=5)
    batch = model.encoder(batch)

    # step through each layer and check shapes after each
    for layer in model.layers:
        prev_x = batch.x.clone()
        prev_e = batch.edge_attr.clone()

        batch = layer(batch)

        atom = batch.x.shape[0]
        edge = batch.edge_index.shape[1]
        assert batch.x.shape == (atom, dim_in)
        assert batch.edge_attr.shape == (edge, dim_in)
        assert torch.isfinite(batch.x).all()
        assert torch.isfinite(batch.edge_attr).all()
        assert torch.sum(torch.abs(batch.x - prev_x)) > 0
        assert torch.sum(torch.abs(batch.edge_attr - prev_e)) > 0

    pred, true = model(batch)

    assert pred.shape == true.shape == (5, 1)
    assert torch.isfinite(pred).all()
