"""Generalised Pareto (GPD) goodness-of-fit statistics for the tail thresholds.

Shared by the peaks-over-threshold (EVT) and TAILS threshold-selection methods in
:mod:`cordex2energy_analysis.extremes_thresholds`. The functions are plain NumPy and
operate on generic float arrays.

Notation: the GPD with shape ``xi`` and scale ``sigma`` (location 0) has the CDF
``F(x) = 1 - (1 + xi x / sigma) ** (-1 / xi)``, with the exponential limit
``F(x) = 1 - exp(-x / sigma)`` as ``xi -> 0``. A probability-integral-transform (PIT)
value is ``F(x)`` evaluated at an observed or simulated excess ``x``.
"""
from __future__ import annotations

import numpy as np
import scipy.stats as stats

# PIT values are clipped to [_PIT_CLIP, 1 - _PIT_CLIP] so that log(1 - PIT) stays finite.
_PIT_CLIP = 1e-10


def _gpd_right_tail_ad_statistic(pit_sorted: np.ndarray) -> float:
    """Right-tail Anderson-Darling statistic of one sample.

    Parameters
    ----------
    pit_sorted : numpy.ndarray
        1-D array of ``n`` PIT values in (0, 1), sorted ascending.

    Returns
    -------
    float
        The statistic ``A2_right``; larger values mean a worse fit in the upper tail.

    Notes
    -----
    Choulakian & Stephens (2001); Bader, Yan & Zhang (2018), R package ``eva``::

        A2_right = -n / 2 - (1 / n) * sum_{i=1..n} (2n - 2i + 1) * ln(1 - z_i)

    where ``z_i`` is the i-th smallest PIT value.
    """
    n = len(pit_sorted)
    pit = np.clip(pit_sorted, _PIT_CLIP, 1.0 - _PIT_CLIP)
    i = np.arange(1, n + 1)
    return float(-n / 2.0 - np.sum((2.0 * n - 2.0 * i + 1.0) * np.log(1.0 - pit)) / n)


def _gpd_right_tail_ad_batch(pit_sorted_rows: np.ndarray) -> np.ndarray:
    """Right-tail Anderson-Darling statistic for each row of a 2-D array.

    Parameters
    ----------
    pit_sorted_rows : numpy.ndarray
        Array of shape ``(m, n)``; each row holds ``n`` PIT values in (0, 1), sorted
        ascending.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(m,)`` with the ``A2_right`` statistic of each row (same
        formula as :func:`_gpd_right_tail_ad_statistic`).
    """
    _, n = pit_sorted_rows.shape
    pit = np.clip(pit_sorted_rows, _PIT_CLIP, 1.0 - _PIT_CLIP)
    weights = 2.0 * n - 2.0 * np.arange(1, n + 1) + 1.0
    return -n / 2.0 - (weights * np.log(1.0 - pit)).sum(axis=1) / n


def _lmoment_gpd_bootstrap_refit(
    shape: float,
    scale: float,
    n_excesses: int,
    n_bootstrap: int,
    random_generator: np.random.Generator,
):
    """Simulate parametric GPD bootstrap samples and refit each by L-moments.

    Parameters
    ----------
    shape : float
        GPD shape ``xi`` of the fitted model (scipy's ``c``).
    scale : float
        GPD scale ``sigma`` of the fitted model.
    n_excesses : int
        Sample size of each bootstrap replicate (the number of observed excesses).
    n_bootstrap : int
        Number of bootstrap replicates.
    random_generator : numpy.random.Generator
        Source of the random draws.

    Returns
    -------
    simulated_sorted : numpy.ndarray
        ``(n_bootstrap, n_excesses)`` simulated excesses, each row sorted ascending.
    shape_refit : numpy.ndarray
        ``(n_bootstrap,)`` L-moment estimates of the shape.
    scale_refit : numpy.ndarray
        ``(n_bootstrap,)`` L-moment estimates of the scale.
    valid : numpy.ndarray
        ``(n_bootstrap,)`` boolean mask of replicates with finite estimates,
        positive scale and shape below 1.

    Notes
    -----
    The refit is closed-form (no optimiser). With the first two L-moments
    ``l1 = sigma / (1 - xi)`` and ``l2 = sigma / ((1 - xi)(2 - xi))``, the estimators
    are ``xi = 2 - l1 / l2`` and ``sigma = l1 (1 - xi)``. ``l2`` is obtained from the
    probability-weighted moment ``b1`` as ``l2 = 2 b1 - l1``.
    """
    simulated_sorted = np.sort(
        stats.genpareto.rvs(shape, loc=0, scale=scale, size=(n_bootstrap, n_excesses),
                            random_state=random_generator),
        axis=1,
    )

    rank_weights = np.arange(n_excesses) / max(n_excesses - 1, 1)  # (j - 1) / (n - 1) for b1
    l1 = simulated_sorted.mean(axis=1)
    b1 = (simulated_sorted * rank_weights).mean(axis=1)
    l2 = 2.0 * b1 - l1
    shape_refit = 2.0 - l1 / l2
    scale_refit = l1 * (1.0 - shape_refit)

    valid = (scale_refit > 0) & np.isfinite(shape_refit) & np.isfinite(scale_refit) & (shape_refit < 1.0)
    return simulated_sorted, shape_refit, scale_refit, valid


def _gpd_cdf_vectorised(
    excesses: np.ndarray,
    shape: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    """GPD CDF evaluated row-wise with one parameter pair per row.

    Parameters
    ----------
    excesses : numpy.ndarray
        ``(m, n)`` excesses, one bootstrap replicate per row.
    shape : numpy.ndarray
        ``(m, 1)`` shape parameters ``xi``.
    scale : numpy.ndarray
        ``(m, 1)`` scale parameters ``sigma``.

    Returns
    -------
    numpy.ndarray
        ``(m, n)`` CDF values. The exponential limit is used where ``|xi| <= 1e-6``.
    """
    inner = np.maximum(1.0 + shape * excesses / scale, 1e-10)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        cdf_power = 1.0 - inner ** (-1.0 / shape)
        cdf_exponential = 1.0 - np.exp(-excesses / scale)
        cdf = np.where(np.abs(shape) > 1e-6, cdf_power, cdf_exponential)
    return cdf
