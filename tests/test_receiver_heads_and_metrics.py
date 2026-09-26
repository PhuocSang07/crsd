"""Receiver-head scoring, pass@k, chunked CE, anchors and diagnostics helpers."""

import itertools
import math

import numpy as np
import pytest
import torch

from anchor_labels import ANCHOR_LABELS, heuristic_label
from diagnostics import ndcg_at_k
from masked_loss import IGNORE_INDEX, chunked_cross_entropy, masked_cross_entropy
from pass_at_k import holm_bonferroni, mean_pass_at_k, pass_at_k
from receiver_heads import band_layers, receiver_scores, select_heads, vertical_scores


def test_band_layers_qwen3_depths():
    teacher, student = band_layers(36), band_layers(28)
    assert teacher == [list(range(14, 25)), list(range(25, 36))]
    assert student == [list(range(11, 19)), list(range(19, 28))]


def test_vertical_scores_use_only_far_rows():
    R = torch.zeros(1, 6, 6)
    R[0, 5, 1] = 1.0
    R[0, 3, 2] = 1.0  # distance 1: below d_min, ignored
    rows = torch.tensor([False, False, False, True, True, True])
    nu = vertical_scores(R, rows, d_min=2)[0]
    assert nu[1] == pytest.approx(1 / 3)  # rows 3, 4, 5 are >= 2 after node 1
    assert nu[2] == 0.0  # rows 4, 5 read node 2 with 0 mass
    assert math.isnan(nu[5].item())


def test_receiver_scores_prefer_peaked_heads_and_remove_background():
    torch.manual_seed(0)
    n = 40
    sink = torch.zeros(n)
    sink[0] = 5.0  # every head attends to node 0 (sink-like shared pattern)
    # realistic scale: hundreds of heads, so one head's peak barely moves the background mean
    background = [sink + 0.2 * torch.rand(n) for _ in range(200)]
    flat = sink + 0.2 * torch.rand(n)
    peaked = sink + 0.2 * torch.rand(n)
    peaked[17] += 3.0
    nu = torch.stack([flat, peaked] + background)
    raw = receiver_scores(nu, "kurtosis")
    bg = receiver_scores(nu, "excess_bg")
    assert bg[1] > bg[0] + 10  # after background removal only the specific peak counts
    assert raw[0] > 20  # raw kurtosis is fooled by the shared sink
    heads = select_heads(np.arange(12, dtype=float).reshape(4, 3), k_per_band=2)
    assert heads == [[(2, 2), (2, 1)], [(3, 2), (3, 1)]]


def test_pass_at_k_matches_enumeration():
    for n, c, k in [(16, 3, 1), (16, 3, 3), (4, 0, 3), (4, 4, 1), (6, 2, 3)]:
        samples = [1] * c + [0] * (n - c)
        combos = list(itertools.combinations(range(n), k))
        exact = np.mean([max(samples[i] for i in combo) for combo in combos])
        assert pass_at_k(n, c, k) == pytest.approx(exact)
    assert mean_pass_at_k([[1, 0, 0, 1], [0, 0, 0, 0]], 1) == pytest.approx(0.25)
    adjusted = holm_bonferroni({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adjusted["a"] == pytest.approx(0.03) and adjusted["b"] >= adjusted["c"]


def test_chunked_cross_entropy_matches_full():
    torch.manual_seed(0)
    head = torch.nn.Linear(8, 50, bias=False)
    hidden = torch.randn(1, 30, 8, requires_grad=True)
    labels = torch.randint(0, 50, (1, 30))
    labels[0, :10] = IGNORE_INDEX
    full = masked_cross_entropy(head(hidden), labels)
    g_full = torch.autograd.grad(full, (hidden, head.weight))
    chunked = chunked_cross_entropy(hidden, head, labels, chunk=7)
    g_chunk = torch.autograd.grad(chunked, (hidden, head.weight))
    assert torch.isclose(full, chunked, atol=1e-6)
    for a, b in zip(g_full, g_chunk):
        assert torch.allclose(a, b, atol=1e-6)
    assert torch.isclose(chunked_cross_entropy(hidden, head, labels, denominator=40, chunk=4), full * 20 / 40, atol=1e-6)


def test_heuristic_anchor_labels():
    assert heuristic_label("Wait, that can't be right.") == "uncertainty_management"
    assert heuristic_label("Let me check this by plugging in x = 2.") == "self_checking"
    assert heuristic_label("First, I need to find the radius.") == "planning"
    assert heuristic_label("12 * 7 = 84, and 84 + 6 = 90.") == "active_computation"
    assert heuristic_label("So the answer is \\boxed{90}.") == "final_answer"
    assert ANCHOR_LABELS == {"planning", "uncertainty_management", "self_checking"}


def test_ndcg():
    relevance = np.array([3.0, 2.0, 1.0, 0.0])
    assert ndcg_at_k(relevance, relevance, 3) == pytest.approx(1.0)
    assert ndcg_at_k(relevance, -relevance, 3) < 0.5
