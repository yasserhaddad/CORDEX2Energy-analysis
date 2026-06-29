"""Shared GPD / right-tail Anderson-Darling engine.

Used by both threshold-selection methods (POT and TAILS).  All functions here
are pure-NumPy with no domain knowledge; they operate on generic float arrays.
"""
from __future__ import annotations

import numpy as np
import scipy.stats as stats


def _gpd_right_tail_ad_statistic(z_sorted: np.ndarray) -> float:
    """Right-tail Anderson-Darling statistic for n sorted PIT values in (0, 1).

    Formula (Choulakian & Stephens 2001; Bader, Yan & Zhang 2018 ``eva`` package):

        A²_right = -n/2 - (1/n) Σᵢ₌₁ⁿ (2n − 2i + 1) ln(1 − z_i)

    where z_i are ordered probability-integral-transform values of the excesses
    under the fitted GPD.  Larger values indicate worse fit in the upper tail.
    """
    n = len(z_sorted)
    z = np.clip(z_sorted, 1e-10, 1.0 - 1e-10)
    i = np.arange(1, n + 1)
    return float(-n / 2.0 - np.sum((2.0 * n - 2.0 * i + 1.0) * np.log(1.0 - z)) / n)


def _gpd_right_tail_ad_batch(z_sorted_2d: np.ndarray) -> np.ndarray:
    """Vectorised right-tail AD statistic for a 2-D array of PIT rows.

    Parameters
    ----------
    z_sorted_2d : (m, n) array
        Each row contains n sorted PIT values in (0, 1).

    Returns
    -------
    (m,) array of A²_right statistics.
    """
    m, n = z_sorted_2d.shape
    z = np.clip(z_sorted_2d, 1e-10, 1.0 - 1e-10)
    weights = 2.0 * n - 2.0 * np.arange(1, n + 1) + 1.0  # (n,)
    return -n / 2.0 - (weights * np.log(1.0 - z)).sum(axis=1) / n  # (m,)


def _lmoment_gpd_bootstrap_refit(
    c: float,
    scale: float,
    n_exc: int,
    n_bootstrap: int,
    rng: np.random.Generator,
):
    """Closed-form L-moment GPD refit for ``n_bootstrap`` parametric resamples.

    Simulate from the fitted GPD, refit each bootstrap replicate via
    probability-weighted-moment / L-moment estimators (no iterative optimiser).
    L-moments: λ1 = σ/(1−ξ), λ2 = σ/((1−ξ)(2−ξ)) → ξ = 2 − λ1/λ2

    Returns
    -------
    sim_sorted : (n_bootstrap, n_exc) — sorted bootstrap draws
    xi_b : (n_bootstrap,) — L-moment shape estimates
    sigma_b : (n_bootstrap,) — L-moment scale estimates
    valid : (n_bootstrap,) bool — rows with physically plausible parameters
    """
    sim_sorted = np.sort(
        stats.genpareto.rvs(c, loc=0, scale=scale, size=(n_bootstrap, n_exc), random_state=rng),
        axis=1,
    )

    w_b1 = np.arange(n_exc) / max(n_exc - 1, 1)  # (n_exc,) weights for β1
    l1_b = sim_sorted.mean(axis=1)                # (n_bootstrap,)
    b1_b = (sim_sorted * w_b1).mean(axis=1)
    l2_b = 2.0 * b1_b - l1_b
    xi_b = 2.0 - l1_b / l2_b
    sigma_b = l1_b * (1.0 - xi_b)

    valid = (sigma_b > 0) & np.isfinite(xi_b) & np.isfinite(sigma_b) & (xi_b < 1.0)
    return sim_sorted, xi_b, sigma_b, valid


def _gpd_cdf_vectorised(
    x_v: np.ndarray,
    xi_v: np.ndarray,
    sig_v: np.ndarray,
) -> np.ndarray:
    """Vectorised GPD CDF, safe for xi → 0 (exponential limit).

    Parameters
    ----------
    x_v : (n_valid, n_exc) — exceedances per bootstrap replicate
    xi_v : (n_valid, 1) — shape parameters
    sig_v : (n_valid, 1) — scale parameters

    Returns
    -------
    (n_valid, n_exc) array of CDF values in (0, 1).
    """
    inner = np.maximum(1.0 + xi_v * x_v / sig_v, 1e-10)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        z_pow = 1.0 - inner ** (-1.0 / xi_v)
        z_exp = 1.0 - np.exp(-x_v / sig_v)  # limit as ξ → 0
        z = np.where(np.abs(xi_v) > 1e-6, z_pow, z_exp)
    return z
