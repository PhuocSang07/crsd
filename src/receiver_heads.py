"""Receiver-head scoring and selection (proposal Sec. 4.3).

Vertical score of node j at head (l, h): nu_j = E_{i : i - j >= d_min} R[i, j] -- how much the steps
far after j read it back. A receiver head concentrates nu on a few nodes, measured by kurtosis over
j. Two scores:
    "kurtosis"     raw Pearson kurtosis of nu_j (Bogdan et al., 2025)
    "excess_bg"    (default) subtract the mean vertical profile over *all* heads of the model first,
                   then Fisher (excess) kurtosis -- removes patterns every head shares, like the
                   attention sink (Hiding in Plain Sight, 2026)
Scores are averaged over a calibration set; the top K heads of each relative-depth band are kept.
"""

import numpy as np
import torch

DEPTH_BANDS = ((0.4, 0.7), (0.7, 1.0))
SCORE_MODES = ("excess_bg", "kurtosis")


def band_layers(num_layers: int, bands=DEPTH_BANDS) -> list[list[int]]:
    """Layer indices per band by relative depth l / (L - 1); the last band is closed at 1.0."""
    depth = [layer / max(1, num_layers - 1) for layer in range(num_layers)]
    out = []
    for index, (low, high) in enumerate(bands):
        closed = index == len(bands) - 1
        out.append([l for l, d in enumerate(depth) if low <= d and (d <= high if closed else d < high)])
    return out


def vertical_scores(R: torch.Tensor, rows: torch.Tensor, d_min: int) -> torch.Tensor:
    """R: [H, N, N] row routing of H heads -> nu [H, N], NaN for nodes no row is far enough from."""
    num_nodes = R.size(-1)
    i = torch.arange(num_nodes, device=R.device).unsqueeze(1)
    j = torch.arange(num_nodes, device=R.device).unsqueeze(0)
    eligible = ((i - j) >= d_min) & rows.to(torch.bool).unsqueeze(1)  # [N_i, N_j]
    counts = eligible.sum(0).float()
    sums = (R.float() * eligible.float()).sum(-2)
    nu = sums / counts.clamp_min(1)
    return torch.where(counts > 0, nu, torch.full_like(nu, float("nan")))


def _kurtosis(values: torch.Tensor, excess: bool) -> torch.Tensor:
    """Kurtosis along the last dim ignoring NaN; 0 where the spread is degenerate."""
    finite = torch.isfinite(values)
    n = finite.sum(-1).float().clamp_min(1)
    x = torch.where(finite, values, torch.zeros_like(values))
    mean = x.sum(-1, keepdim=True) / n.unsqueeze(-1)
    centered = torch.where(finite, values - mean, torch.zeros_like(values))
    m2 = (centered**2).sum(-1) / n
    m4 = (centered**4).sum(-1) / n
    kurt = m4 / m2.clamp_min(1e-20) ** 2
    kurt = torch.where(m2 > 1e-20, kurt, torch.zeros_like(kurt))
    return kurt - 3.0 if excess else kurt


def receiver_scores(nu: torch.Tensor, mode: str = "excess_bg") -> torch.Tensor:
    """nu [H, N] over all heads of one trace -> one score per head (see module doc)."""
    if mode == "kurtosis":
        return _kurtosis(nu, excess=False)
    if mode == "excess_bg":
        background = torch.nanmean(nu, dim=0, keepdim=True)
        return _kurtosis(nu - background, excess=True)
    raise ValueError(f"unknown receiver score {mode!r}; expected one of {SCORE_MODES}")


def select_heads(mean_scores: np.ndarray, k_per_band: int, bands=DEPTH_BANDS) -> list[list[tuple[int, int]]]:
    """mean_scores [L, H] -> per band the top-k (layer, head) pairs by score, best first."""
    num_layers, num_heads = mean_scores.shape
    selection = []
    for layers in band_layers(num_layers, bands):
        candidates = [(float(mean_scores[l, h]), l, h) for l in layers for h in range(num_heads)]
        candidates.sort(key=lambda item: -item[0])
        selection.append([(l, h) for _, l, h in candidates[:k_per_band]])
    return selection


def split_half_stability(per_trace: np.ndarray, k_per_band: int, bands=DEPTH_BANDS, seed: int = 0) -> dict:
    """Reliability of the selection: random half/half split of the calibration traces.

    Reports the Spearman correlation of the two halves' head scores (Bogdan et al. report r=.67)
    and, per band, the overlap |top-k(A) & top-k(B)| / k.
    """
    from scipy.stats import spearmanr

    if per_trace.shape[0] < 4:
        return {"spearman": float("nan"), "topk_overlap": [float("nan")] * len(bands)}
    order = np.random.default_rng(seed).permutation(per_trace.shape[0])
    half = len(order) // 2
    a = np.nanmean(per_trace[order[:half]], axis=0)
    b = np.nanmean(per_trace[order[half:]], axis=0)
    rho = spearmanr(a.ravel(), b.ravel(), nan_policy="omit").statistic
    sel_a, sel_b = select_heads(a, k_per_band, bands), select_heads(b, k_per_band, bands)
    overlap = [len(set(x) & set(y)) / max(1, k_per_band) for x, y in zip(sel_a, sel_b)]
    return {"spearman": float(rho), "topk_overlap": overlap}
