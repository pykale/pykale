from types import SimpleNamespace

import torch

from kale.embed.model_lib.CGCNN import CGCNNEncoderLayer, CrystalGCN


def _make_dummy_cgcnn_batch(num_graphs=2, nodes_per_graph=(3, 2), atom_fea_len=8, nbr_fea_len=6, max_nbr=3):
    """Build a minimal batch object mimicking CIFData.collate_fn output."""
    total_nodes = sum(nodes_per_graph)

    atom_fea = torch.randn(total_nodes, atom_fea_len)
    nbr_fea = torch.randn(total_nodes, max_nbr, nbr_fea_len)
    nbr_fea_idx = torch.empty(total_nodes, max_nbr, dtype=torch.long)

    crystal_atom_idx = []
    base = 0
    for n in nodes_per_graph:
        idx = torch.arange(base, base + n, dtype=torch.long)
        crystal_atom_idx.append(idx)
        for k, node in enumerate(idx):
            local = idx[(k + torch.arange(max_nbr)) % n]  # cyclic neighbors
            nbr_fea_idx[node] = local
        base += n

    target = torch.randn(num_graphs, 1)

    return SimpleNamespace(
        atom_fea=atom_fea,
        nbr_fea=nbr_fea,
        nbr_fea_idx=nbr_fea_idx,
        crystal_atom_idx=crystal_atom_idx,
        target=target,
        batch_size=num_graphs,
    )


def test_cgcnn_encoder_layer():
    atom_fea_len, nbr_fea_len, max_nbr = 8, 6, 3
    batch = _make_dummy_cgcnn_batch(
        num_graphs=2, nodes_per_graph=(3, 2), atom_fea_len=atom_fea_len, nbr_fea_len=nbr_fea_len, max_nbr=max_nbr
    )

    model = CGCNNEncoderLayer(atom_fea_len=atom_fea_len, nbr_fea_len=nbr_fea_len)
    assert (
        model.fc_full.__repr__()
        == f"Linear(in_features={2 * atom_fea_len + nbr_fea_len}, out_features={2 * atom_fea_len}, bias=True)"
    )
    assert (
        model.bn1.__repr__()
        == f"BatchNorm1d({2 * atom_fea_len}, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)"
    )
    assert (
        model.bn2.__repr__()
        == f"BatchNorm1d({atom_fea_len}, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)"
    )

    model.eval()  # stabilize BatchNorm for comparison

    out = model(batch.atom_fea, batch.nbr_fea, batch.nbr_fea_idx)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (batch.atom_fea.shape[0], atom_fea_len)

    # Permute neighbors consistently in features and indices
    perm = torch.randperm(max_nbr)
    nbr_fea_perm = batch.nbr_fea[:, perm, :]
    nbr_fea_idx_perm = batch.nbr_fea_idx[:, perm]

    out_perm = model(batch.atom_fea, nbr_fea_perm, nbr_fea_idx_perm)

    # Summation over neighbors should make the output invariant to neighbor ordering
    assert torch.allclose(out, out_perm, atol=1e-6)


def test_crystalgcn_forward_and_backward():
    orig_atom_fea_len, nbr_fea_len = 8, 6
    batch = _make_dummy_cgcnn_batch(
        num_graphs=3, nodes_per_graph=(2, 3, 2), atom_fea_len=orig_atom_fea_len, nbr_fea_len=nbr_fea_len, max_nbr=3
    )

    model = CrystalGCN(
        orig_atom_fea_len=orig_atom_fea_len,
        nbr_fea_len=nbr_fea_len,
        atom_fea_len=16,
        n_conv=2,
        h_fea_len=32,
        n_h=2,
        classification=False,
    )

    assert model.embedding.__repr__() == f"Linear(in_features={orig_atom_fea_len}, out_features=16, bias=True)"
    assert model.conv_to_fc.__repr__() == "Linear(in_features=16, out_features=32, bias=True)"
    assert model.fcs[0].__repr__() == "Linear(in_features=32, out_features=32, bias=True)"

    pred = model(batch)
    assert isinstance(pred, torch.Tensor)
    assert pred.shape == (len(batch.crystal_atom_idx), 1)

    loss = (pred - batch.target).pow(2).mean()
    loss.backward()  # should run without error
