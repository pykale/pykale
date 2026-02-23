import pytest
import torch

from kale.embed.materials_equivariant import (
    CosineCutoff,
    EquiMessagePassing,
    EquiOutput,
    ExpNormalSmearing,
    FTE,
    GatedEquivariantBlock,
    NeighborEmb,
    rbf_emb,
    S_vector,
)


def test_rbf_emb():
    # (num_rbf, rbound_upper)
    model = rbf_emb(6, 6.0)
    assert model.reset_parameters() is None
    # (num_edges, 1)
    dist = torch.randn(4).abs()
    out = model(dist)
    assert out.shape == (4, 6)
    assert model.__repr__() is not None


def test_neighbor_emb_basic():
    """Checks shape, baseline (no-neighbor) behavior, effect of neighbors, gradients, and error on bad shapes."""
    hid_dim, in_dim = 8, 5
    emb = NeighborEmb(hid_dim=hid_dim, input_dim=in_dim)

    # Tiny directed graph: 0<->1, 1<->2, 2<->3  (6 edges total)
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long)
    num_nodes = int(edge_index.max()) + 1

    # Node features
    z = torch.randn(num_nodes, in_dim, requires_grad=True)

    # ---- Case 1: zero edge embeddings -> output should equal LN(FC(z)) (no neighbor contribution) ----
    embs_zero = torch.zeros(edge_index.size(1), hid_dim)
    out_zero = emb(z, edge_index, embs_zero)
    baseline = emb.ln_emb(emb.fc(z))
    assert out_zero.shape == (num_nodes, hid_dim)
    assert torch.allclose(out_zero, baseline, atol=1e-6)

    # ---- Case 2: ones edge embeddings -> should differ from baseline and be finite ----
    embs_one = torch.ones_like(embs_zero)
    out_one = emb(z, edge_index, embs_one)
    assert out_one.shape == (num_nodes, hid_dim)
    assert not torch.allclose(out_one, baseline)
    assert torch.isfinite(out_one).all()

    # ---- Gradients flow (w.r.t. inputs and parameters) ----
    loss = out_one.sum()
    loss.backward()
    assert z.grad is not None and z.grad.shape == z.shape
    assert emb.fc.weight.grad is not None

    # ---- Case 3: empty edge set -> should reduce to baseline ----
    empty_edges = torch.empty(2, 0, dtype=torch.long)
    embs_empty = torch.empty(0, hid_dim)
    out_empty = emb(z.detach().requires_grad_(True), empty_edges, embs_empty)
    baseline2 = emb.ln_emb(emb.fc(z.detach()))
    assert torch.allclose(out_empty, baseline2, atol=1e-6)

    # ---- Case 4: bad embs shape triggers a runtime error ----
    bad_embs = torch.randn(edge_index.size(1), hid_dim + 1)
    with pytest.raises(RuntimeError):
        emb(z, edge_index, bad_embs)


def test_s_vector():
    torch.manual_seed(0)
    hid_dim = 8
    num_nodes = 4

    # small symmetric graph: 0<->1, 1<->2, 2<->3  (6 directed edges)
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long)
    num_edges = edge_index.size(1)

    s = torch.randn(num_nodes, hid_dim, requires_grad=True)  # node invariants
    v = torch.randn(num_edges, 3, 1)  # edge directions (E,3,1)
    emb = torch.randn(num_edges, hid_dim, requires_grad=True)  # edge RBF (E,H)

    layer = S_vector(hid_dim)

    # ----- basic forward / shape -----
    out = layer(s, v, edge_index, emb)
    assert out.shape == (num_nodes, 3, hid_dim)
    assert torch.isfinite(out).all()

    # ----- zeros in emb -> zero output -----
    out_zero_emb = layer(s.detach().clone(), v, edge_index, torch.zeros_like(emb))
    assert torch.allclose(out_zero_emb, torch.zeros_like(out_zero_emb), atol=1e-6)

    # ----- zeros in v -> zero output (because emb.unsqueeze(1) * v == 0) -----
    out_zero_v = layer(s.detach().clone(), torch.zeros_like(v), edge_index, emb.detach().clone())
    assert torch.allclose(out_zero_v, torch.zeros_like(out_zero_v), atol=1e-6)

    # ----- gradients flow -----
    out.sum().backward()
    assert s.grad is not None and s.grad.shape == s.shape
    assert emb.grad is not None and emb.grad.shape == emb.shape

    # ----- bad shapes raise -----
    with pytest.raises(RuntimeError):
        bad_emb = torch.randn(num_edges, hid_dim + 1)
        layer(s, v, edge_index, bad_emb)

    with pytest.raises(RuntimeError):
        bad_v = torch.randn(num_edges, 3)  # missing last dim
        layer(s, bad_v, edge_index, emb)


def test_equi_message_passing():
    torch.manual_seed(0)

    # tiny synthetic graph
    num_nodes = 5
    num_edges = 10
    hidden = 8
    num_radial = 6

    # random directed edges (allows self-edges too)
    src = torch.randint(0, num_nodes, (num_edges,))  # source nodes
    dst = torch.randint(0, num_nodes, (num_edges,))  # destination nodes
    edge_index = torch.stack([src, dst], dim=0)

    # node invariants and equivariants
    x = torch.randn(num_nodes, hidden, requires_grad=True)  # [num_nodes, hidden]
    vec = torch.randn(num_nodes, 3, hidden, requires_grad=True)  # [num_nodes, 3, hidden]

    # edge features
    edge_rbf = torch.randn(num_edges, num_radial, requires_grad=True)  # [num_edges, num_radial]
    weight = torch.randn(num_edges, 3 * hidden + num_radial, requires_grad=True)  # [num_edges, 3 * hidden + num_radial]
    edge_vec = torch.randn(num_edges, 3)  # [num_edges, 3]

    layer = EquiMessagePassing(hidden_channels=hidden, num_radial=num_radial)

    # ---- forward: shapes ----
    dx, dvec = layer(x, vec, edge_index, edge_rbf, weight, edge_vec)
    assert isinstance(dx, torch.Tensor) and isinstance(dvec, torch.Tensor)
    assert dx.shape == (num_nodes, hidden)
    assert dvec.shape == (num_nodes, 3, hidden)
    assert torch.isfinite(dx).all() and torch.isfinite(dvec).all()

    # ---- gradients flow ----
    (dx.sum() + dvec.sum()).backward()
    assert x.grad is not None and x.grad.shape == x.shape
    assert vec.grad is not None and vec.grad.shape == vec.shape
    assert edge_rbf.grad is not None and edge_rbf.grad.shape == edge_rbf.shape
    assert weight.grad is not None and weight.grad.shape == weight.shape

    # ---- zero-RBF -> zero output (because rbf_proj(edge_rbf)=0 => messages=0) ----
    layer_zero = EquiMessagePassing(hidden_channels=hidden, num_radial=num_radial)
    dx0, dvec0 = layer_zero(x.detach(), vec.detach(), edge_index, torch.zeros_like(edge_rbf), weight.detach(), edge_vec)
    assert torch.allclose(dx0, torch.zeros_like(dx0), atol=1e-6)
    assert torch.allclose(dvec0, torch.zeros_like(dvec0), atol=1e-6)

    # ---- bad shapes should error ----
    with pytest.raises(RuntimeError):
        wrong_vec = torch.randn(num_nodes, hidden)  # missing the 3-axis
        layer(x.detach(), wrong_vec, edge_index, edge_rbf.detach(), weight.detach(), edge_vec)


def test_fte_forward_shapes_and_determinism():
    hidden_channels, num_nodes = 16, 4

    torch.manual_seed(0)

    x = torch.randn(num_nodes, hidden_channels)
    vec = torch.randn(num_nodes, 3, hidden_channels)
    # LEFTNet passes a placeholder "node_frame"; FTE ignores it internally
    node_frame = torch.zeros(num_nodes, 3, 3)

    fte = FTE(hidden_channels)

    # check output shapes
    dx, dvec = fte(x, vec, node_frame)
    assert isinstance(dx, torch.Tensor) and isinstance(dvec, torch.Tensor)
    assert dx.shape == (num_nodes, hidden_channels)
    assert dvec.shape == (num_nodes, 3, hidden_channels)

    # determinism in eval mode
    fte.eval()
    with torch.no_grad():
        dx1, dvec1 = fte(x, vec, node_frame)
        dx2, dvec2 = fte(x, vec, node_frame)
    assert torch.allclose(dx1, dx2)
    assert torch.allclose(dvec1, dvec2)


def test_fte_backprop_and_reset():
    torch.manual_seed(0)
    hidden_channels, num_nodes = 16, 4

    x = torch.randn(num_nodes, hidden_channels, requires_grad=True)
    vec = torch.randn(num_nodes, 3, hidden_channels, requires_grad=True)
    node_frame = torch.zeros(num_nodes, 3, 3)

    fte = FTE(hidden_channels)
    fte.train()

    dx, dvec = fte(x, vec, node_frame)
    loss = dx.mean() + dvec.mean()
    loss.backward()

    # gradients should flow to inputs and parameters
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert vec.grad is not None and torch.isfinite(vec.grad).all()

    has_param_grads = any(
        p.grad is not None and torch.isfinite(p.grad).all() for p in fte.parameters() if p.requires_grad
    )
    assert has_param_grads, "No parameter received gradients."

    # reset parameters should run without error
    fte.reset_parameters()


def test_equioutput_forward_and_determinism():
    hidden_channels, num_nodes = 16, 5
    torch.manual_seed(0)
    x = torch.randn(num_nodes, hidden_channels)
    vec = torch.randn(num_nodes, 3, hidden_channels)

    layer = EquiOutput(hidden_channels)

    # shape check
    out = layer(x, vec)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (num_nodes, 3)  # expects final vec with 1 channel, squeezed

    # determinism in eval mode
    layer.eval()
    with torch.no_grad():
        out1 = layer(x, vec)
        out2 = layer(x, vec)
    assert torch.allclose(out1, out2)


def test_equioutput_backprop_and_reset():
    torch.manual_seed(0)
    hidden_channels, num_nodes = 16, 5
    x = torch.randn(num_nodes, hidden_channels, requires_grad=True)
    vec = torch.randn(num_nodes, 3, hidden_channels, requires_grad=True)

    layer = EquiOutput(hidden_channels)
    layer.train()

    out = layer(x, vec)  # (num_nodes, 3)
    loss = out.pow(2).mean()
    loss.backward()

    # grads on inputs
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert vec.grad is not None and torch.isfinite(vec.grad).all()

    # some parameter received gradients
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in layer.parameters() if p.requires_grad)

    # reset should run without error
    layer.reset_parameters()


def test_gated_equivariant():
    torch.manual_seed(0)

    # --- 1) Shape & type checks across a couple configs ---
    hidden_channels, out_channels, num_nodes = (16, 8, 4)
    block = GatedEquivariantBlock(hidden_channels, out_channels)
    x = torch.randn(num_nodes, hidden_channels)
    v = torch.randn(num_nodes, 3, hidden_channels)

    x_out, v_out = block(x, v)

    assert isinstance(x_out, torch.Tensor)
    assert isinstance(v_out, torch.Tensor)
    assert x_out.shape == (num_nodes, out_channels)
    assert v_out.shape == (num_nodes, 3, out_channels)

    # --- 2) Determinism in eval mode ---
    block = GatedEquivariantBlock(16, 8).eval()
    x = torch.randn(5, 16)
    v = torch.randn(5, 3, 16)
    with torch.no_grad():
        x1, v1 = block(x, v)
        x2, v2 = block(x, v)
    assert torch.allclose(x1, x2)
    assert torch.allclose(v1, v2)

    # --- 3) Backprop works & reset_parameters runs ---
    block = GatedEquivariantBlock(16, 8).train()
    x = torch.randn(6, 16, requires_grad=True)
    v = torch.randn(6, 3, 16, requires_grad=True)

    x_out, v_out = block(x, v)
    loss = x_out.pow(2).mean() + v_out.pow(2).mean()
    loss.backward()

    # grads on inputs
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert v.grad is not None and torch.isfinite(v.grad).all()

    # some parameters received gradients
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in block.parameters() if p.requires_grad)

    # reset should execute without error
    block.reset_parameters()


def test_exp_normal_smearing_all():
    # shape & cutoff behavior
    cutoff = 5.0
    num_rbf = 16
    layer = ExpNormalSmearing(cutoff_lower=0.0, cutoff_upper=cutoff, num_rbf=num_rbf, trainable=False)

    dist = torch.tensor([0.0, cutoff / 2.0, cutoff])
    out = layer(dist)

    # shape
    assert out.shape == (3, num_rbf)
    # non-negative
    assert torch.all(out >= 0)
    # at the upper cutoff, output ~ 0 due to CosineCutoff
    assert torch.allclose(out[-1], torch.zeros(num_rbf), atol=1e-6)
    # at distance 0, output should be > 0
    assert torch.all(out[0] > 0)

    # trainable flags, gradient flow, reset_parameters
    cutoff2 = 6.0
    num_rbf2 = 8
    trainable_layer = ExpNormalSmearing(cutoff_lower=0.0, cutoff_upper=cutoff2, num_rbf=num_rbf2, trainable=True)
    assert trainable_layer.means.requires_grad
    assert trainable_layer.betas.requires_grad

    frozen_layer = ExpNormalSmearing(cutoff_lower=0.0, cutoff_upper=cutoff2, num_rbf=num_rbf2, trainable=False)
    assert not frozen_layer.means.requires_grad
    assert not frozen_layer.betas.requires_grad

    # gradients w.r.t. distances should flow: keep a leaf tensor and pass an expression
    d = torch.rand(10, requires_grad=True)  # leaf
    out = trainable_layer(d * cutoff2).sum()  # expression uses the leaf
    out.backward()
    assert d.grad is not None and torch.isfinite(d.grad).all()


def test_cosine_cutoff():
    # -------- Case A: cutoff_lower = 0.0 (common path) --------
    upper = 5.0
    layer_a = CosineCutoff(cutoff_lower=0.0, cutoff_upper=upper)

    dists_a = torch.tensor([0.0, upper / 2.0, upper, upper + 1e-6])
    out_a = layer_a(dists_a)

    # shape & dtype
    assert out_a.shape == dists_a.shape
    assert out_a.dtype == dists_a.dtype

    # in-range values are within [0, 1]
    assert torch.all(out_a[:-1].ge(0) & out_a[:-1].le(1))

    # at d=0 => 0.5*(cos(0)+1) = 1
    assert torch.allclose(out_a[0], torch.tensor(1.0), atol=1e-6)

    # at d=upper => masked to 0 (because distances < upper)
    assert torch.allclose(out_a[2], torch.tensor(0.0), atol=1e-6)

    # beyond upper => 0
    assert torch.allclose(out_a[3], torch.tensor(0.0), atol=1e-6)

    # -------- Case B: cutoff_lower > 0.0 (window (lower, upper)) --------
    lower = 1.0
    upper = 6.0
    layer_b = CosineCutoff(cutoff_lower=lower, cutoff_upper=upper)

    # Include values below lower, exactly at lower, inside window, at upper, and above upper
    dists_b = torch.tensor([0.5 * lower, lower, (lower + upper) / 2.0, upper, upper + 1e-6])
    out_b = layer_b(dists_b)

    # shape & dtype
    assert out_b.shape == dists_b.shape
    assert out_b.dtype == dists_b.dtype

    # below lower => 0
    assert torch.allclose(out_b[0], torch.tensor(0.0), atol=1e-6)

    # exactly at lower => strictly greater condition makes it 0
    assert torch.allclose(out_b[1], torch.tensor(0.0), atol=1e-6)

    # inside (lower, upper): allow up to 1 (midpoint can be exactly 1)
    assert (out_b[2] > 0) and (out_b[2] <= 1)

    # at/above upper => 0
    assert torch.allclose(out_b[3], torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(out_b[4], torch.tensor(0.0), atol=1e-6)
