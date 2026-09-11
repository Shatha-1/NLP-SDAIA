"""Lab 6 starter: bootstrap confidence intervals."""
import numpy as np


def bootstrap_ci(values, *, n_boot=2000, seed=42, alpha=0.05):
    """Return (point_estimate, ci_lo, ci_hi) via the percentile bootstrap."""
    values = np.asarray(values, dtype=float)
    point = float(values.mean())

    rng = np.random.default_rng(seed)
    n = len(values)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        sample_idx = rng.integers(0, n, size=n)
        boot_means[i] = values[sample_idx].mean()

    lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


def paired_bootstrap_diff(a, b, *, n_boot=2000, seed=42, alpha=0.05):
    """Return (mean_delta, ci_lo, ci_hi) for a - b via paired resampling."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) != len(b):
        raise ValueError("a and b must be the same length (paired observations)")

    diffs = a - b
    point = float(diffs.mean())

    rng = np.random.default_rng(seed)
    n = len(diffs)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        sample_idx = rng.integers(0, n, size=n)
        boot_means[i] = diffs[sample_idx].mean()

    lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)
