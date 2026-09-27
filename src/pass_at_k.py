"""Unbiased pass@k (Chen et al., 2021; proposal Eq. 8) and the two-level bootstrap CI.

    pass@k = E_problems [ 1 - C(n - c, k) / C(n, k) ]

Computed in the numerically stable product form; with k = 1 it is the mean accuracy.
"""

import numpy as np


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k of one problem with c correct out of n samples (requires k <= n)."""
    if k > n:
        raise ValueError(f"pass@{k} needs at least {k} samples, got {n}")
    if n - c < k:
        return 1.0
    return 1.0 - float(np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


def mean_pass_at_k(labels: list[list[int]], k: int) -> float:
    """Benchmark-level pass@k from per-problem 0/1 labels (each problem may have its own n)."""
    return float(np.mean([pass_at_k(len(row), int(sum(row)), k) for row in labels])) if labels else 0.0


def bootstrap_ci(labels: list[list[int]], k: int, resamples: int = 2000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    """Two-level bootstrap (problems, then samples within each problem), percentile CI of pass@k."""
    rng = np.random.default_rng(seed)
    rows = [np.asarray(row) for row in labels]
    stats = []
    for _ in range(resamples):
        problems = rng.integers(0, len(rows), len(rows))
        values = []
        for p in problems:
            row = rows[p]
            sample = row[rng.integers(0, len(row), len(row))]
            values.append(pass_at_k(len(sample), int(sample.sum()), k))
        stats.append(np.mean(values))
    return float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2))


def paired_permutation_test(a: list[float], b: list[float], resamples: int = 10000, seed: int = 0) -> float:
    """Two-sided sign-flip test on per-problem differences a - b (method vs SFT)."""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    observed = abs(diff.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(resamples, diff.size))
    return float((np.abs((signs * diff).mean(axis=1)) >= observed - 1e-12).mean())


def holm_bonferroni(p_values: dict[str, float]) -> dict[str, float]:
    """Holm-adjusted p-values for several configurations compared against SFT."""
    order = sorted(p_values, key=p_values.get)
    adjusted, running = {}, 0.0
    for rank, name in enumerate(order):
        running = max(running, min(1.0, (len(order) - rank) * p_values[name]))
        adjusted[name] = running
    return adjusted
