"""Routing math on synthetic tensors (CPU): Definition 1, Lemma 1, losses, gap metrics."""

import math

import pytest
import torch

from routing import (
    DISTANCE_BINS,
    attention_probs,
    bernoulli_kl,
    causal_loss,
    causal_target,
    far_target_mask,
    head_query_routing,
    head_row_routing_blockwise,
    js_divergence,
    kl_rows,
    mass_loss,
    node_mass,
    per_query_route_loss,
    query_routing,
    route_loss,
    routing_gap_by_bin,
    row_average,
    valid_rows,
)


def _layout(step_lengths, prompt=3, question=4):
    """key_nodes for [template x prompt][q x question][steps...][answer x 3][stop]."""
    nodes = [-1] * prompt + [0] * question
    for index, length in enumerate(step_lengths, start=1):
        nodes += [index] * length
    nodes += [len(step_lengths) + 1] * 3 + [-1]
    return torch.tensor(nodes), len(step_lengths) + 2


def test_far_mask_matches_definition():
    far = far_target_mask(8, d_min=4)
    assert not far[0].any()
    # F(i) = {0} U {1..i-4}
    assert far[4].nonzero().squeeze(-1).tolist() == [0]
    assert far[6].nonzero().squeeze(-1).tolist() == [0, 1, 2]
    assert valid_rows(far).nonzero().squeeze(-1).tolist() == [5, 6, 7]


def test_node_mass_and_query_routing_normalize():
    torch.manual_seed(0)
    key_nodes, N = _layout([3, 2, 4, 3, 2, 5])
    attn = torch.softmax(torch.randn(5, key_nodes.numel()), -1)
    mass = node_mass(attn, key_nodes, N)
    assert torch.allclose(mass.sum(-1), attn[:, key_nodes >= 0].sum(-1))
    far = far_target_mask(N, 2)
    rows = torch.tensor([3, 4, 5, 6, 7])
    R, Z = query_routing(mass, rows, far)
    assert torch.allclose(R.sum(-1), torch.ones(5))
    assert torch.all(R[~far[rows]] == 0)
    assert torch.allclose(Z, (mass * far[rows]).sum(-1))


def test_lemma1_gradient_sum_equals_routing_deviation():
    """Lemma 1 (ii): sum_{u in I(v_j)} dl/de_tu = R_j - P_j, (i): zero on keys outside F(i)."""
    torch.manual_seed(1)
    key_nodes, N = _layout([2, 3, 2, 4, 3, 2, 3])
    i, d_min = 7, 2
    far = far_target_mask(N, d_min)
    logits = torch.randn(key_nodes.numel(), requires_grad=True)
    attn = torch.softmax(logits, -1).unsqueeze(0)
    R, _ = query_routing(node_mass(attn, key_nodes, N), torch.tensor([i]), far)
    P = torch.rand(N) * far[i]
    P = P / P.sum()
    loss = kl_rows(P.unsqueeze(0), R)[0]
    loss.backward()
    grad = logits.grad
    for j in range(N):
        total = grad[key_nodes == j].sum()
        if far[i, j]:
            assert torch.isclose(total, R[0, j] - P[j], atol=1e-6)
        else:
            assert torch.allclose(grad[key_nodes == j], torch.zeros_like(grad[key_nodes == j]), atol=1e-7)
    assert torch.allclose(grad[key_nodes < 0], torch.zeros_like(grad[key_nodes < 0]), atol=1e-7)


def test_lemma1_signal_does_not_vanish_for_tiny_mass():
    """The per-target gradient is R_j - P_j even when the student gives node j almost no attention."""
    key_nodes, N = _layout([2] * 8)
    far = far_target_mask(N, 2)
    logits = torch.zeros(key_nodes.numel())
    logits[key_nodes == 1] = -12.0  # ~e-12 attention on node 1
    logits.requires_grad_(True)
    R, _ = query_routing(node_mass(torch.softmax(logits, -1).unsqueeze(0), key_nodes, N), torch.tensor([9]), far)
    P = far[9].float() / far[9].sum()
    kl_rows(P.unsqueeze(0), R)[0].backward()
    assert torch.isclose(logits.grad[key_nodes == 1].sum(), R[0, 1] - P[1], atol=1e-5)
    assert abs(float(logits.grad[key_nodes == 1].sum())) > 0.05


def test_route_loss_finite_differences():
    torch.manual_seed(2)
    key_nodes, N = _layout([3, 3, 2, 4, 3, 3])
    T, d = key_nodes.numel(), 8
    q = torch.randn(4, d, dtype=torch.float64, requires_grad=True)
    k = torch.randn(T, d, dtype=torch.float64)
    positions = torch.tensor([13, 16, 19, 22])
    owners = key_nodes[positions]
    far = far_target_mask(N, 2)
    P = torch.rand(N, N, dtype=torch.float64) * far
    P = P / P.sum(-1, keepdim=True).clamp_min(1e-12)
    rows = valid_rows(far)

    def f(q_):
        R, _ = head_query_routing(q_, k, positions, key_nodes, owners, far, 0.35)
        return route_loss(P, row_average(R, owners, N)[0], rows)

    assert torch.autograd.gradcheck(lambda q_: f(q_).double(), (q,), eps=1e-6, atol=1e-5)


def test_checkpointed_head_routing_matches_plain():
    torch.manual_seed(3)
    key_nodes, N = _layout([3, 4, 2, 3, 3, 2])
    T = key_nodes.numel()
    q = torch.randn(5, 8, requires_grad=True)
    k = torch.randn(T, 8, requires_grad=True)
    positions = torch.tensor([10, 12, 15, 20, 24])
    owners = key_nodes[positions]
    far = far_target_mask(N, 2)
    outs = []
    for flag in (False, True):
        q.grad = k.grad = None
        R, Z = head_query_routing(q, k, positions, key_nodes, owners, far, 0.3, use_checkpoint=flag)
        (R.square().sum() + Z.sum()).backward()
        outs.append((R.detach(), q.grad.clone(), k.grad.clone()))
    for a, b in zip(*outs):
        assert torch.allclose(a, b, atol=1e-6)


def test_blockwise_teacher_matches_full_attention():
    """Appendix D: block-wise node masses == full attention (tolerance 1e-3; here exact in fp32)."""
    torch.manual_seed(4)
    key_nodes, N = _layout([5, 4, 6, 3, 5, 4, 6])
    T, d = key_nodes.numel(), 16
    q, k = torch.randn(2, T, d), torch.randn(2, T, d)
    far = far_target_mask(N, 2)
    out = head_row_routing_blockwise(q, k, key_nodes, far, d**-0.5, block=7)
    positions = torch.nonzero((key_nodes >= 1) & valid_rows(far)[key_nodes.clamp_min(0)]).squeeze(-1)
    owners = key_nodes[positions]
    for h in range(2):
        full = torch.softmax((q[h] @ k[h].T) * d**-0.5 + torch.triu(torch.full((T, T), -math.inf), 1), -1)[positions]
        R, Z = query_routing(node_mass(full, key_nodes, N), owners, far)
        R_row, _ = row_average(R, owners, N)
        Z_row, _ = row_average(Z, owners, N)
        assert torch.allclose(out["R"][h], R_row, atol=1e-5)
        assert torch.allclose(out["Z"][h], Z_row, atol=1e-5)
        expected = (positions.float() - full @ torch.arange(T).float()).mean()
        assert torch.isclose(out["mean_distance"][h], expected, atol=1e-4)
    single = head_row_routing_blockwise(q[1], k[1], key_nodes, far, d**-0.5, block=5)
    assert torch.allclose(single["R"], out["R"][1], atol=1e-6)


def test_attention_probs_is_causal():
    probs = attention_probs(torch.randn(3, 4), torch.randn(6, 4), torch.tensor([0, 2, 5]), 0.5)
    assert torch.allclose(probs.sum(-1), torch.ones(3))
    assert probs[0, 1:].abs().sum() == 0 and probs[1, 3:].abs().sum() == 0


def test_losses_zero_at_match_and_positive_otherwise():
    far = far_target_mask(8, 2)
    rows = valid_rows(far)
    P = far.float() / far.float().sum(-1, keepdim=True).clamp_min(1)
    assert float(route_loss(P, P, rows)) == pytest.approx(0.0, abs=1e-7)
    Q = torch.rand(8, 8) * far
    Q = Q / Q.sum(-1, keepdim=True).clamp_min(1e-12)
    assert float(route_loss(P, Q, rows)) > 0
    z = torch.rand(8)
    assert float(mass_loss(z, z, rows)) == pytest.approx(0.0, abs=1e-6)
    assert float(bernoulli_kl(torch.tensor(0.7), torch.tensor(0.2))) > 0
    # anchor weighting reweights rows only
    w = torch.ones(8)
    assert float(route_loss(P, Q, rows, w)) == pytest.approx(float(route_loss(P, Q, rows)), rel=1e-6)


def test_per_query_loss_upper_bounds_pooled():
    torch.manual_seed(5)
    far = far_target_mask(9, 2)
    rows = valid_rows(far)
    owners = torch.tensor([4, 4, 4, 6, 6, 8, 8, 8])
    R = torch.rand(owners.numel(), 9) * far[owners]
    R = R / R.sum(-1, keepdim=True)
    P = torch.rand(9, 9) * far
    P = P / P.sum(-1, keepdim=True).clamp_min(1e-12)
    pooled = row_average(R, owners, 9)[0]
    rows_q = torch.zeros(9, dtype=torch.bool)
    rows_q[owners] = True
    # per-row Jensen: mean_t KL(P||R_t) >= KL(P||mean_t R_t)
    for i in owners.unique():
        per_q = kl_rows(P[i].expand(int((owners == i).sum()), -1), R[owners == i]).mean()
        assert per_q >= kl_rows(P[i : i + 1], pooled[i : i + 1])[0] - 1e-6
    assert float(per_query_route_loss(P, R, owners, rows & rows_q)) >= 0


def test_causal_target_normalizes_over_far_and_J():
    N = 9
    far = far_target_mask(N, 2)
    C = torch.full((N, N), float("nan"))
    J = torch.tensor([0, 2, 3])
    C[:, J] = torch.rand(N, 3)
    c_tilde, support = causal_target(C, J, far)
    for i in range(N):
        if support[i].any():
            assert torch.isclose(c_tilde[i].sum(), torch.tensor(1.0))
            assert set(support[i].nonzero().squeeze(-1).tolist()) <= set(J.tolist()) & set(far[i].nonzero().squeeze(-1).tolist())
    Q = far.float() / far.float().sum(-1, keepdim=True).clamp_min(1)
    assert float(causal_loss(c_tilde, Q, support)) >= 0


def test_routing_gap_bins():
    N = 80
    far = far_target_mask(N, 4)
    rows = valid_rows(far)
    P = far.float() / far.float().sum(-1, keepdim=True).clamp_min(1)
    terms = routing_gap_by_bin(P, P, rows)
    assert [t["bin"][0] for t in terms] == [low for low, _ in DISTANCE_BINS]
    assert all(t["rg_sum"] == pytest.approx(0.0, abs=1e-6) for t in terms)
    assert terms[-1]["rg_count"] > 0  # rows >= 66 have >= 2 targets at distance >= 64
    Q = P.clone()
    Q[:, 1:] *= 0.5
    Q = Q / Q.sum(-1, keepdim=True).clamp_min(1e-12)
    shifted = routing_gap_by_bin(P, Q, rows)
    assert all(t["dmu_sum"] < 0 for t in shifted if t["dmu_count"])  # mass moved to node 0 (no bin)
    assert torch.allclose(js_divergence(P[rows], Q[rows]), js_divergence(Q[rows], P[rows]), atol=1e-6)
