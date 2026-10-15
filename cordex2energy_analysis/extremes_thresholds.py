"""Deficit thresholds for extreme residual-load events.

Three methods are available through :func:`compute_active_threshold` (annual) and
:func:`calibrate_seasonal_active_thresholds` (per season):

``"PERCENTILE"`` (:func:`find_fixed_percentile_threshold`)
    A fixed percentile of the strictly positive residual load. Used in the paper.
``"EVT"`` (:func:`find_evt_pot_threshold`)
    Peaks over threshold: the lowest candidate threshold at which a generalised
    Pareto distribution (GPD) fitted to the excesses is not rejected by a
    parametric-bootstrap right-tail Anderson-Darling test.
``"TAILS"`` (:func:`find_tails_threshold`)
    The threshold-selection steps of the TAILS framework (Supplementary Method 1,
    eqs. 1-9): knee-point lower bound, fine quantile grid, filter on goodness of fit
    and event count, minimum quantile-distance optimum, and the range of thresholds
    indistinguishable from it. Bootstrap refits use L-moments and the filter uses the
    right-tail Anderson-Darling statistic.

Every threshold is finally floored at zero (a genuine supply deficit) and zones with
too few deficit steps are flagged as degenerate (:func:`_apply_deficit_floor`).
The GPD statistics are in :mod:`cordex2energy_analysis.extremes_statistics`.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.stats as stats
import xarray as xr

from .extremes_statistics import (
    _gpd_right_tail_ad_statistic,
    _gpd_right_tail_ad_batch,
    _lmoment_gpd_bootstrap_refit,
    _gpd_cdf_vectorised,
)

# Fewest excesses for which a GPD is fitted at a candidate threshold.
_MIN_EXCESSES = 30


# ---------------------------------------------------------------------------
# Peaks over threshold (EVT)
# ---------------------------------------------------------------------------


def find_evt_pot_threshold(
    residual_load,
    min_quantile: float = 0.9,
    max_quantile: float = 0.995,
    num_candidates: int = 50,
    n_bootstrap: int = 199,
    p_value_threshold: float = 0.05,
) -> float:
    """Peaks-over-threshold threshold from a GPD goodness-of-fit scan.

    Parameters
    ----------
    residual_load : numpy.ndarray
        1-D reference-period residual-load values. Only strictly positive values are
        used.
    min_quantile : float, default 0.9
        Lowest candidate, as a quantile level of the positive values.
    max_quantile : float, default 0.995
        Highest candidate, as a quantile level of the positive values.
    num_candidates : int, default 50
        Number of candidate quantile levels, evenly spaced between the two bounds.
    n_bootstrap : int, default 199
        Parametric-bootstrap replicates for each p-value.
    p_value_threshold : float, default 0.05
        Significance level: a candidate is accepted when its p-value exceeds it.

    Returns
    -------
    float
        The lowest accepted candidate, in the units of ``residual_load``; the 99th
        percentile of the positive values if no candidate is accepted.

    Notes
    -----
    Candidates are scanned from low to high. At each candidate ``u`` a GPD (location
    0) is fitted to the excesses ``x - u`` by maximum likelihood and the right-tail
    Anderson-Darling statistic ``A2_right`` of the fit is computed. Its p-value is
    estimated by a parametric bootstrap: samples are drawn from the fitted GPD,
    refitted by L-moments (closed form, valid for shape < 1), and
    ``p = (#{A2_boot >= A2_obs} + 1) / (n_valid + 1)``. The scan stops at the first
    accepted candidate, or when fewer than 30 excesses remain. The random seed is
    fixed (0), so results are reproducible.
    """
    positive_values = residual_load[residual_load > 0]

    candidates = np.percentile(
        positive_values, np.linspace(min_quantile * 100, max_quantile * 100, num_candidates)
    )

    best_threshold = None
    random_generator = np.random.default_rng(0)

    for candidate in candidates:
        excesses = positive_values[positive_values > candidate] - candidate

        if len(excesses) < _MIN_EXCESSES:
            break

        try:
            shape, location, scale = stats.genpareto.fit(excesses, floc=0)
        except Exception:
            continue

        pit_observed = np.sort(stats.genpareto.cdf(excesses, shape, loc=location, scale=scale))
        statistic_observed = _gpd_right_tail_ad_statistic(pit_observed)

        simulated_sorted, shape_refit, scale_refit, valid = _lmoment_gpd_bootstrap_refit(
            shape, scale, len(excesses), n_bootstrap, random_generator
        )
        n_valid = int(valid.sum())
        if n_valid == 0:
            continue

        # Rows are sorted and the GPD CDF is monotone, so the PIT rows are sorted too.
        pit_bootstrap = _gpd_cdf_vectorised(
            simulated_sorted[valid], shape_refit[valid, np.newaxis], scale_refit[valid, np.newaxis]
        )
        statistic_bootstrap = _gpd_right_tail_ad_batch(pit_bootstrap)

        p_value = (float(np.sum(statistic_bootstrap >= statistic_observed)) + 1.0) / (n_valid + 1.0)

        if p_value > p_value_threshold:
            best_threshold = candidate
            break

    if best_threshold is None:
        best_threshold = np.percentile(positive_values, 99)

    return float(best_threshold)


def compute_evt_thresholds_for_all_zones(
    reference_residual_load, min_quantile=0.9, p_value_threshold=0.05, zone_dim="zone"
) -> "tuple[xr.DataArray, xr.DataArray]":
    """Run :func:`find_evt_pot_threshold` for every zone.

    Parameters
    ----------
    reference_residual_load : xarray.DataArray
        Reference-period residual load with a ``zone_dim`` dimension. Non-finite
        values are dropped per zone.
    min_quantile : float, default 0.9
        Passed to :func:`find_evt_pot_threshold`.
    p_value_threshold : float, default 0.05
        Passed to :func:`find_evt_pot_threshold`.
    zone_dim : str, default "zone"
        Name of the zone dimension.

    Returns
    -------
    thresholds : xarray.DataArray
        Threshold per zone (name ``evt_threshold``), in the input units.
    percentiles : xarray.DataArray
        Empirical percentile (0-100) of each threshold among the zone's finite values
        (name ``evt_threshold_pct``).
    """
    threshold_by_zone: dict = {}
    percentile_by_zone: dict = {}

    for zone in reference_residual_load[zone_dim].values:
        zone_values = reference_residual_load.sel({zone_dim: zone}).values.flatten()
        zone_values = zone_values[np.isfinite(zone_values)]

        threshold = find_evt_pot_threshold(
            zone_values, min_quantile=min_quantile, p_value_threshold=p_value_threshold
        )
        threshold_by_zone[zone] = threshold
        percentile_by_zone[zone] = float(np.mean(zone_values <= threshold) * 100)

    zones = list(threshold_by_zone.keys())
    thresholds = xr.DataArray(
        list(threshold_by_zone.values()),
        coords={zone_dim: zones},
        dims=[zone_dim],
        name="evt_threshold",
    )
    percentiles = xr.DataArray(
        list(percentile_by_zone.values()),
        coords={zone_dim: zones},
        dims=[zone_dim],
        name="evt_threshold_pct",
    )
    return thresholds, percentiles


def calibrate_seasonal_evt_thresholds(
    reference_residual_load: xr.Dataset,
    zones,
    season_order: list[str],
    season_months: dict[str, list[int]],
    zone_dim: str = "zone",
) -> "tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]":
    """Run :func:`find_evt_pot_threshold` for every (zone, season).

    Parameters
    ----------
    reference_residual_load : xarray.Dataset
        Reference-period dataset with a ``residual_load`` variable and a ``time``
        coordinate.
    zones : sequence
        Zone coordinate values.
    season_order : list of str
        Season names, e.g. ``["DJF", "MAM", "JJA", "SON"]``.
    season_months : dict
        Mapping ``season -> list of calendar months``.
    zone_dim : str, default "zone"
        Name of the zone dimension.

    Returns
    -------
    thresholds : dict
        ``{zone: {season: threshold}}`` in the input units; zone keys are strings.
    percentiles : dict
        ``{zone: {season: percentile}}``: empirical percentile (0-100) of each
        threshold among all values of that season.
    """
    times = pd.DatetimeIndex(reference_residual_load["time"].values)
    thresholds: dict[str, dict[str, float]] = {}
    percentiles: dict[str, dict[str, float]] = {}
    for zone in zones:
        zone_key = str(zone)
        thresholds[zone_key] = {}
        percentiles[zone_key] = {}
        zone_values = reference_residual_load["residual_load"].sel({zone_dim: zone}).values
        for season in season_order:
            season_values = zone_values[np.isin(times.month, season_months[season])]
            threshold = find_evt_pot_threshold(season_values)
            thresholds[zone_key][season] = threshold
            percentiles[zone_key][season] = float(np.mean(season_values <= threshold) * 100)
    return thresholds, percentiles


# ---------------------------------------------------------------------------
# TAILS
# ---------------------------------------------------------------------------


def _knee_point_quantile(values: np.ndarray, quantile_levels: np.ndarray) -> float:
    """Knee of the empirical quantile curve, the TAILS lower bound (eq. 2).

    Parameters
    ----------
    values : numpy.ndarray
        1-D sample.
    quantile_levels : numpy.ndarray
        Increasing quantile levels in [0, 1] at which the curve is evaluated.

    Returns
    -------
    float
        The quantile level at which the min-max normalised quantile curve lies
        furthest above the straight line joining its end points.
    """
    quantiles = np.quantile(values, quantile_levels)

    levels_normalised = (quantile_levels - quantile_levels.min()) / (quantile_levels.max() - quantile_levels.min())
    quantiles_normalised = (quantiles - quantiles.min()) / (quantiles.max() - quantiles.min() + 1e-12)

    chord = (quantiles_normalised[0]
             + (quantiles_normalised[-1] - quantiles_normalised[0]) * (levels_normalised - levels_normalised[0])
             / (levels_normalised[-1] - levels_normalised[0]))
    knee_index = int(np.argmax(quantiles_normalised - chord))
    return float(quantile_levels[knee_index])


def _z_normalize(series: np.ndarray, years: np.ndarray, threshold_type: str) -> np.ndarray:
    """Standardise a series to zero mean and unit standard deviation (TAILS eq. 4).

    Parameters
    ----------
    series : numpy.ndarray
        1-D values.
    years : numpy.ndarray
        Calendar year of each value.
    threshold_type : {"yearly", "global"}
        ``"yearly"`` standardises each year with its own mean and standard deviation;
        any other value uses a single mean and standard deviation for the whole series.

    Returns
    -------
    numpy.ndarray
        The standardised series. A zero standard deviation is replaced by 1.
    """
    if threshold_type == "yearly":
        standardised = np.empty_like(series, dtype=float)
        for year in np.unique(years):
            in_year = years == year
            mean, std = series[in_year].mean(), series[in_year].std(ddof=0)
            std = std if std > 0 else 1.0
            standardised[in_year] = (series[in_year] - mean) / std
        return standardised
    mean, std = series.mean(), series.std(ddof=0)
    std = std if std > 0 else 1.0
    return (series - mean) / std


def _count_events(exceed_mask: np.ndarray) -> int:
    """Count contiguous runs of ``True`` in a boolean array.

    Parameters
    ----------
    exceed_mask : numpy.ndarray
        1-D boolean exceedance mask.

    Returns
    -------
    int
        Number of runs (run-length declustered events). Used only for the TAILS
        minimum-event-count filter (eq. 7).
    """
    if exceed_mask.sum() == 0:
        return 0
    steps = np.diff(exceed_mask.astype(int))
    return int(np.sum(steps == 1) + (1 if exceed_mask[0] else 0))


def _bootstrap_quantile_distances(
    simulated: np.ndarray,
    shape: np.ndarray,
    scale: np.ndarray,
    probabilities: np.ndarray,
) -> np.ndarray:
    """Mean absolute distance between sample quantiles and GPD quantiles (TAILS eq. 6).

    Parameters
    ----------
    simulated : numpy.ndarray
        ``(m, n)`` bootstrap samples of excesses.
    shape : numpy.ndarray
        ``(m, 1)`` GPD shapes refitted to each sample.
    scale : numpy.ndarray
        ``(m, 1)`` GPD scales refitted to each sample.
    probabilities : numpy.ndarray
        ``(k,)`` probability levels at which quantiles are compared.

    Returns
    -------
    numpy.ndarray
        ``(m,)`` mean over the ``k`` levels of ``|sample quantile - GPD quantile|``.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gpd_quantiles = stats.genpareto.ppf(probabilities[np.newaxis, :], shape, loc=0, scale=scale)
    sample_quantiles = np.quantile(simulated, probabilities, axis=1).T
    return np.mean(np.abs(sample_quantiles - gpd_quantiles), axis=1)


@dataclass
class TailsResult:
    """Result of :func:`find_tails_threshold`.

    Thresholds are in the units of the standardised (Z-normalised) series.

    Attributes
    ----------
    threshold : float or None
        Selected threshold (the 0.99 quantile when ``fallback_used``).
    threshold_quantile : float or None
        Quantile level of ``threshold``.
    threshold_range : tuple of float or None
        Lower and upper edge of the indistinguishable threshold range (eq. 9); None on
        fallback.
    threshold_range_quantile : tuple of float or None
        Quantile levels of ``threshold_range``; None on fallback.
    knee_quantile : float
        Quantile level of the knee point (eq. 2).
    q_lower : float
        Lowest candidate quantile level: the knee rounded up to 0.01, at least the
        minimum quantile.
    candidates_quantile : numpy.ndarray
        Candidate quantile levels (eq. 3).
    distances : numpy.ndarray
        Quantile distance per candidate (eq. 6); NaN where no fit was made.
    ad_pvalues : numpy.ndarray
        Bootstrap right-tail Anderson-Darling p-value per candidate; NaN where no fit
        was made.
    n_events : numpy.ndarray
        Run-length event count per candidate.
    passed_filter : numpy.ndarray
        Boolean mask of candidates passing the eq. 7 filter.
    fallback_used : bool
        True when no candidate passed the filter.
    diagnostics : dict
        Filter counts and settings; on success also ``ci_band`` and ``best_distance``,
        on fallback a ``reason``.
    """

    threshold: float | None
    threshold_quantile: float | None
    threshold_range: tuple[float, float] | None
    threshold_range_quantile: tuple[float, float] | None
    knee_quantile: float
    q_lower: float
    candidates_quantile: np.ndarray
    distances: np.ndarray
    ad_pvalues: np.ndarray
    n_events: np.ndarray
    passed_filter: np.ndarray
    fallback_used: bool = False
    diagnostics: dict = field(default_factory=dict)


def find_tails_threshold(
    series: np.ndarray,
    years: np.ndarray,
    invert: bool = False,
    threshold_type: str = "global",
    min_quantile: float = 0.95,
    knee_grid_size: int = 200,
    quantile_step: float = 0.001,
    n_bootstrap: int = 199,
    p_value_threshold: float = 0.05,
    min_events_per_year: float = 1.0,
    n_quantile_levels: int = 50,
    random_seed: int = 0,
) -> TailsResult:
    """TAILS automated threshold selection (Supplementary Method 1, eqs. 1-9).

    Parameters
    ----------
    series : numpy.ndarray
        1-D reference-period values of the indicator.
    years : numpy.ndarray
        Calendar year of each value.
    invert : bool, default False
        Negate the series first, for indicators whose stress is low values (eq. 1).
    threshold_type : {"global", "yearly"}, default "global"
        Standardisation convention (eq. 4); see :func:`_z_normalize`.
    min_quantile : float, default 0.95
        Lowest admissible candidate quantile level (``q_min`` in eq. 2).
    knee_grid_size : int, default 200
        Number of quantile levels in [0.5, 1] used to locate the knee.
    quantile_step : float, default 0.001
        Spacing of the candidate quantile grid (eq. 3).
    n_bootstrap : int, default 199
        Bootstrap replicates per candidate.
    p_value_threshold : float, default 0.05
        A candidate passes the goodness-of-fit filter when its p-value exceeds this.
    min_events_per_year : float, default 1.0
        Minimum number of run-length events per year of data (eq. 7).
    n_quantile_levels : int, default 50
        Number of probability levels ``i / (n + 1)`` in the distance metric (eq. 6).
    random_seed : int, default 0
        Seed of the bootstrap random generator.

    Returns
    -------
    TailsResult
        Selected threshold, its indistinguishable range and per-candidate diagnostics,
        all on the standardised series.

    Notes
    -----
    1. Negate the series if ``invert`` and standardise it (eqs. 1, 4).
    2. Lower bound: the knee quantile level on [0.5, 1] rounded up to 0.01, at least
       ``min_quantile`` (eq. 2).
    3. Candidates: quantile levels from the lower bound to ``1 - quantile_step`` in
       steps of ``quantile_step`` (eq. 3).
    4. At each candidate with at least 30 excesses, fit a GPD by maximum likelihood,
       draw ``n_bootstrap`` samples from it and refit each by L-moments (eq. 5).
       Record the bootstrap p-value of the right-tail Anderson-Darling statistic and
       the distance: the mean over replicates of the mean absolute difference between
       the replicate's sample quantiles and its refitted GPD quantiles (eq. 6).
    5. Filter: finite distance, p-value above ``p_value_threshold`` and at least
       ``round(min_events_per_year * n_years)`` events (eq. 7). If nothing passes, the
       0.99 quantile is returned with ``fallback_used=True``.
    6. The optimum is the passing candidate with the smallest distance (eq. 8).
    7. The bootstrap is repeated at the optimum; the contiguous run of passing
       candidates around it whose distance lies within the 16th-84th percentile band
       of the optimum's bootstrap distances is the indistinguishable range (eq. 9).
    """
    random_generator = np.random.default_rng(random_seed)
    n_years = len(np.unique(years))
    min_events = max(1, round(min_events_per_year * n_years))

    transformed = -series if invert else series.astype(float)
    standardised = _z_normalize(transformed, years, threshold_type)

    knee_quantile = _knee_point_quantile(standardised, np.linspace(0.50, 1.0, knee_grid_size))
    q_lower = max(np.ceil(knee_quantile * 100) / 100.0, min_quantile)

    candidate_quantiles = np.arange(q_lower, 1.0 - quantile_step, quantile_step)
    if candidate_quantiles.size == 0:
        candidate_quantiles = np.array([min_quantile])

    probabilities = (np.arange(1, n_quantile_levels + 1)) / (n_quantile_levels + 1.0)

    distances = np.full(candidate_quantiles.shape, np.nan)
    ad_pvalues = np.full(candidate_quantiles.shape, np.nan)
    n_events = np.zeros(candidate_quantiles.shape, dtype=int)

    for k, quantile_level in enumerate(candidate_quantiles):
        candidate = np.quantile(standardised, quantile_level)
        exceed_mask = standardised > candidate
        excesses = standardised[exceed_mask] - candidate

        n_events[k] = _count_events(exceed_mask)

        if len(excesses) < _MIN_EXCESSES:
            continue

        try:
            shape, _, scale = stats.genpareto.fit(excesses, floc=0)
        except Exception:
            continue
        if scale <= 0:
            continue

        simulated_sorted, shape_refit, scale_refit, valid = _lmoment_gpd_bootstrap_refit(
            shape, scale, len(excesses), n_bootstrap, random_generator
        )
        n_valid = int(valid.sum())
        if n_valid == 0:
            continue

        shape_valid = shape_refit[valid, np.newaxis]
        scale_valid = scale_refit[valid, np.newaxis]
        simulated_valid = simulated_sorted[valid]

        pit_bootstrap = _gpd_cdf_vectorised(simulated_valid, shape_valid, scale_valid)
        pit_observed = np.sort(stats.genpareto.cdf(excesses, shape, loc=0, scale=scale))
        statistic_observed = _gpd_right_tail_ad_statistic(pit_observed)
        statistic_bootstrap = _gpd_right_tail_ad_batch(np.sort(pit_bootstrap, axis=1))
        ad_pvalues[k] = (float(np.sum(statistic_bootstrap >= statistic_observed)) + 1.0) / (n_valid + 1.0)

        distances[k] = float(np.mean(
            _bootstrap_quantile_distances(simulated_valid, shape_valid, scale_valid, probabilities)
        ))

    passed = (
        np.isfinite(distances)
        & (ad_pvalues > p_value_threshold)
        & (n_events >= min_events)
    )

    filter_diagnostics = {
        "n_candidates":    len(candidate_quantiles),
        "n_fit_ok":        int(np.sum(np.isfinite(distances))),
        "n_pass_ad":       int(np.sum(np.isfinite(distances) & (ad_pvalues > p_value_threshold))),
        "n_pass_events":   int(np.sum(n_events >= min_events)),
        "n_pass_all":      int(passed.sum()),
        "min_events_applied": min_events,
        "ad_p_threshold":  p_value_threshold,
    }

    if not passed.any():
        fallback_quantile = 0.99
        return TailsResult(
            threshold=float(np.quantile(standardised, fallback_quantile)),
            threshold_quantile=fallback_quantile,
            threshold_range=None,
            threshold_range_quantile=None,
            knee_quantile=knee_quantile,
            q_lower=q_lower,
            candidates_quantile=candidate_quantiles,
            distances=distances,
            ad_pvalues=ad_pvalues,
            n_events=n_events,
            passed_filter=passed,
            fallback_used=True,
            diagnostics={
                "reason": "no candidate passed eq. 7 filter",
                "n_years": n_years,
                **filter_diagnostics,
            },
        )

    passed_indices = np.where(passed)[0]
    best_index = passed_indices[np.argmin(distances[passed_indices])]
    best_quantile = candidate_quantiles[best_index]
    best_threshold = np.quantile(standardised, best_quantile)

    excesses_best = standardised[standardised > best_threshold] - best_threshold
    shape_best, _, scale_best = stats.genpareto.fit(excesses_best, floc=0)
    simulated_best, shape_refit_best, scale_refit_best, valid_best = _lmoment_gpd_bootstrap_refit(
        shape_best, scale_best, len(excesses_best), n_bootstrap, random_generator
    )
    distances_at_optimum = _bootstrap_quantile_distances(
        simulated_best[valid_best],
        shape_refit_best[valid_best, np.newaxis],
        scale_refit_best[valid_best, np.newaxis],
        probabilities,
    )
    band_low, band_high = np.percentile(distances_at_optimum, [16, 84])

    in_band = (distances >= band_low) & (distances <= band_high) & passed
    range_start = best_index
    while range_start - 1 >= 0 and in_band[range_start - 1]:
        range_start -= 1
    range_end = best_index
    while range_end + 1 < len(candidate_quantiles) and in_band[range_end + 1]:
        range_end += 1

    range_quantiles = (float(candidate_quantiles[range_start]), float(candidate_quantiles[range_end]))
    range_values = (
        float(np.quantile(standardised, range_quantiles[0])),
        float(np.quantile(standardised, range_quantiles[1])),
    )

    return TailsResult(
        threshold=float(best_threshold),
        threshold_quantile=float(best_quantile),
        threshold_range=range_values,
        threshold_range_quantile=range_quantiles,
        knee_quantile=knee_quantile,
        q_lower=q_lower,
        candidates_quantile=candidate_quantiles,
        distances=distances,
        ad_pvalues=ad_pvalues,
        n_events=n_events,
        passed_filter=passed,
        fallback_used=False,
        diagnostics={
            "ci_band": (float(band_low), float(band_high)),
            "best_distance": float(distances[best_index]),
            "n_years": n_years,
            **filter_diagnostics,
        },
    )


def compute_tails_thresholds_for_all_zones(
    reference_data: xr.DataArray,
    time_dim: str = "time",
    zone_dim: str = "zone",
    invert: bool = False,
    threshold_type: str = "yearly",
    min_quantile: float = 0.95,
    quantile_step: float = 0.001,
    n_bootstrap: int = 199,
    p_value_threshold: float = 0.05,
    min_events_per_year: float = 1.0,
    random_seed: int = 0,
) -> "tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]":
    """Run :func:`find_tails_threshold` for every zone.

    Parameters
    ----------
    reference_data : xarray.DataArray
        Reference-period indicator with ``time_dim`` and ``zone_dim`` dimensions.
        Non-finite values are dropped per zone; years are taken from ``time_dim``.
    time_dim : str, default "time"
        Name of the time dimension.
    zone_dim : str, default "zone"
        Name of the zone dimension.
    invert, threshold_type, min_quantile, quantile_step, n_bootstrap, p_value_threshold, min_events_per_year, random_seed
        Passed to :func:`find_tails_threshold`.

    Returns
    -------
    thresholds : xarray.DataArray
        Threshold per zone in standardised units (name ``tails_threshold``).
    threshold_percentiles : xarray.DataArray
        Empirical percentile (0-100) of each threshold within the zone's standardised
        series (name ``tails_threshold_pct``).
    threshold_range : xarray.DataArray
        ``(zone, bound)`` lower and upper edge of the indistinguishable range, in
        standardised units; NaN on fallback (name ``tails_threshold_range``).
    threshold_range_percentiles : xarray.DataArray
        ``(zone, bound)`` quantile levels of that range times 100; NaN on fallback
        (name ``tails_threshold_range_pct``).
    """
    threshold_by_zone: dict = {}
    percentile_by_zone: dict = {}
    range_lower: dict = {}
    range_upper: dict = {}
    range_lower_percentile: dict = {}
    range_upper_percentile: dict = {}

    for zone in reference_data[zone_dim].values:
        da_zone = reference_data.sel({zone_dim: zone})

        zone_values = da_zone.values.flatten()
        zone_years = da_zone[time_dim].dt.year.values
        if zone_years.shape != zone_values.shape:
            zone_years = np.broadcast_to(
                zone_years.reshape(da_zone[time_dim].size, *([1] * (da_zone.ndim - 1))),
                da_zone.shape,
            ).flatten()

        finite = np.isfinite(zone_values)
        zone_values = zone_values[finite]
        zone_years = zone_years[finite]

        result = find_tails_threshold(
            zone_values,
            years=zone_years,
            invert=invert,
            threshold_type=threshold_type,
            min_quantile=min_quantile,
            quantile_step=quantile_step,
            n_bootstrap=n_bootstrap,
            p_value_threshold=p_value_threshold,
            min_events_per_year=min_events_per_year,
            random_seed=random_seed,
        )

        threshold_by_zone[zone] = result.threshold
        if result.threshold is not None:
            transformed = -zone_values if invert else zone_values.astype(float)
            standardised = _z_normalize(transformed, zone_years, threshold_type)
            percentile_by_zone[zone] = float(np.nanmean(standardised <= result.threshold) * 100)
        else:
            percentile_by_zone[zone] = float("nan")
        if result.threshold_range is not None:
            range_lower[zone], range_upper[zone] = result.threshold_range
        else:
            range_lower[zone], range_upper[zone] = float("nan"), float("nan")
        if result.threshold_range_quantile is not None:
            range_lower_percentile[zone] = result.threshold_range_quantile[0] * 100.0
            range_upper_percentile[zone] = result.threshold_range_quantile[1] * 100.0
        else:
            range_lower_percentile[zone] = float("nan")
            range_upper_percentile[zone] = float("nan")

    zones = list(threshold_by_zone.keys())
    thresholds = xr.DataArray(
        list(threshold_by_zone.values()),
        coords={zone_dim: zones},
        dims=[zone_dim],
        name="tails_threshold",
    )
    threshold_percentiles = xr.DataArray(
        list(percentile_by_zone.values()),
        coords={zone_dim: zones},
        dims=[zone_dim],
        name="tails_threshold_pct",
    )
    threshold_range = xr.DataArray(
        np.array([[range_lower[zone], range_upper[zone]] for zone in zones]),
        coords={zone_dim: zones, "bound": ["lower", "upper"]},
        dims=[zone_dim, "bound"],
        name="tails_threshold_range",
    )
    threshold_range_percentiles = xr.DataArray(
        np.array([[range_lower_percentile[zone], range_upper_percentile[zone]] for zone in zones]),
        coords={zone_dim: zones, "bound": ["lower", "upper"]},
        dims=[zone_dim, "bound"],
        name="tails_threshold_range_pct",
    )
    return thresholds, threshold_percentiles, threshold_range, threshold_range_percentiles


# ---------------------------------------------------------------------------
# Deficit floor and coverage
# ---------------------------------------------------------------------------

#: Minimum fraction of reference-period steps with residual load > 0 for a zone (or
#: zone-season) to have a genuine deficit regime. In the CERRA reference,
#: surplus-dominated zones (e.g. Nordic, about 2.5 % positive steps) fall far below it
#: and every deficit-prone zone is at 31 % or more, even per season.
DEFAULT_MIN_POSITIVE_FRACTION = 0.05


def _apply_deficit_floor(
    threshold_gwh: float,
    residual_load: np.ndarray,
    min_positive_fraction: float = DEFAULT_MIN_POSITIVE_FRACTION,
) -> "tuple[float, bool, float]":
    """Floor a threshold at zero and flag zones without a deficit regime.

    An extreme event must at least be a supply deficit (residual load > 0). For
    deficit-prone zones the floor has no effect; for surplus-dominated zones it
    replaces a threshold at or below zero.

    Parameters
    ----------
    threshold_gwh : float
        Candidate threshold (GWh per step).
    residual_load : numpy.ndarray
        1-D reference-period residual load of the zone (or zone-season), same units.
    min_positive_fraction : float, default DEFAULT_MIN_POSITIVE_FRACTION
        Minimum fraction of finite values above zero for a non-degenerate regime.

    Returns
    -------
    floored_threshold : float
        ``max(threshold_gwh, 0)``.
    is_degenerate : bool
        True when the positive fraction is below ``min_positive_fraction`` or the
        threshold is at or below zero.
    positive_fraction : float
        Fraction of finite values above zero (0 if there are none).
    """
    finite = residual_load[np.isfinite(residual_load)]
    positive_fraction = float(np.mean(finite > 0)) if finite.size else 0.0
    floored_threshold = max(float(threshold_gwh), 0.0)
    is_degenerate = positive_fraction < min_positive_fraction or float(threshold_gwh) <= 0.0
    return floored_threshold, is_degenerate, positive_fraction


def _coverage_row(zone_key, season, threshold, values, min_positive_fraction) -> dict:
    """One row of :func:`diagnose_threshold_coverage`.

    Parameters
    ----------
    zone_key : str
        Zone label.
    season : str
        Season label, or ``"ALL"``.
    threshold : float
        Threshold to annotate; NaN if none was given.
    values : numpy.ndarray
        Residual-load values of the zone (or zone-season).
    min_positive_fraction : float
        Passed to :func:`_apply_deficit_floor`.

    Returns
    -------
    dict
        The row; a NaN threshold is evaluated as 0 for the degeneracy flag.
    """
    floored_threshold, degenerate, positive_fraction = _apply_deficit_floor(
        threshold if np.isfinite(threshold) else 0.0, values, min_positive_fraction
    )
    return {
        "zone": zone_key, "season": season,
        "positive_fraction": positive_fraction,
        "threshold_gwh": threshold,
        "threshold_floored_gwh": floored_threshold if np.isfinite(threshold) else float("nan"),
        "degenerate": degenerate,
    }


def diagnose_threshold_coverage(
    residual_load: xr.DataArray,
    thresholds=None,
    zone_dim: str = "zone",
    season_map: "dict[int, str] | None" = None,
    season_order: "list[str] | None" = None,
    min_positive_fraction: float = DEFAULT_MIN_POSITIVE_FRACTION,
) -> pd.DataFrame:
    """Tabulate the deficit fraction and threshold degeneracy per zone (and season).

    A zone whose positive fraction is below ``min_positive_fraction`` has no deficit
    tail; its threshold is degenerate and it should be left out of cross-zone tail
    comparisons.

    Parameters
    ----------
    residual_load : xarray.DataArray
        Reference-period residual load with a ``zone_dim`` dimension (and ``time``
        when ``season_map`` is given).
    thresholds : xarray.DataArray or dict, optional
        Zone-indexed DataArray (used when ``season_map`` is None) or a
        ``{zone: {season: threshold}}`` mapping (used when ``season_map`` is given).
    zone_dim : str, default "zone"
        Name of the zone dimension.
    season_map : dict, optional
        Mapping ``calendar month -> season label``. When given, one row per
        (zone, season) is reported.
    season_order : list of str, optional
        Seasons to report, in order; defaults to the sorted labels of ``season_map``.
    min_positive_fraction : float, default DEFAULT_MIN_POSITIVE_FRACTION
        Passed to :func:`_apply_deficit_floor`.

    Returns
    -------
    pandas.DataFrame
        Columns ``zone``, ``season`` (``"ALL"`` without ``season_map``),
        ``positive_fraction``, ``threshold_gwh``, ``threshold_floored_gwh`` and
        ``degenerate``.
    """
    rows = []
    times = pd.DatetimeIndex(residual_load["time"].values) if "time" in residual_load.dims else None

    for zone in list(residual_load[zone_dim].values):
        zone_key = str(zone)
        zone_values = residual_load.sel({zone_dim: zone}).values.flatten()
        if season_map is None:
            threshold = float("nan")
            if isinstance(thresholds, xr.DataArray):
                threshold = float(thresholds.sel({zone_dim: zone}).values)
            rows.append(_coverage_row(zone_key, "ALL", threshold, zone_values, min_positive_fraction))
        else:
            seasons = season_order or sorted(set(season_map.values()))
            season_of_step = np.array([season_map[month] for month in times.month])
            for season in seasons:
                threshold = float("nan")
                if isinstance(thresholds, dict):
                    threshold = float(thresholds.get(zone_key, {}).get(season, float("nan")))
                rows.append(_coverage_row(zone_key, season, threshold,
                                          zone_values[season_of_step == season], min_positive_fraction))

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Fixed percentile
# ---------------------------------------------------------------------------

#: Default percentile (0-100) of the positive residual load for the "PERCENTILE" method.
DEFAULT_FIXED_PERCENTILE = 95.0


def find_fixed_percentile_threshold(
    residual_load,
    percentile: float = DEFAULT_FIXED_PERCENTILE,
) -> float:
    """Fixed percentile of the strictly positive residual load.

    Parameters
    ----------
    residual_load : array_like
        Reference-period residual load; flattened, and non-finite and non-positive
        values are dropped.
    percentile : float, default DEFAULT_FIXED_PERCENTILE
        Percentile in [0, 100].

    Returns
    -------
    float
        The threshold in the input units, or NaN if there are no positive values.
    """
    values = np.asarray(residual_load, dtype=float).ravel()
    positive_values = values[np.isfinite(values) & (values > 0)]
    if positive_values.size == 0:
        return float("nan")
    return float(np.percentile(positive_values, percentile))


# ---------------------------------------------------------------------------
# Method-agnostic calibration
# ---------------------------------------------------------------------------


def compute_active_threshold(
    residual_load: xr.DataArray,
    method: str,
    zone_dim: str = "zone",
    p_value_threshold: float = 0.1,
    percentile: float = DEFAULT_FIXED_PERCENTILE,
    min_positive_fraction: float = DEFAULT_MIN_POSITIVE_FRACTION,
) -> xr.DataArray:
    """Annual extreme-event threshold per zone with the chosen method.

    Each threshold is floored at zero and zones without a deficit regime are flagged
    (see :func:`_apply_deficit_floor`); every flagged zone also emits a warning.

    Parameters
    ----------
    residual_load : xarray.DataArray
        Reference-period residual load with a ``zone_dim`` dimension.
    method : {"PERCENTILE", "EVT", "TAILS"}
        ``"PERCENTILE"``: :func:`find_fixed_percentile_threshold`. ``"EVT"``:
        :func:`find_evt_pot_threshold`. ``"TAILS"``: :func:`find_tails_threshold` with
        yearly standardisation; the selected percentile of the standardised series is
        mapped back to the same percentile of the raw values.
    zone_dim : str, default "zone"
        Name of the zone dimension.
    p_value_threshold : float, default 0.1
        Significance level for ``"EVT"``.
    percentile : float, default DEFAULT_FIXED_PERCENTILE
        Percentile (0-100) of the positive residual load for ``"PERCENTILE"``.
    min_positive_fraction : float, default DEFAULT_MIN_POSITIVE_FRACTION
        Minimum deficit fraction for a non-degenerate zone.

    Returns
    -------
    xarray.DataArray
        Threshold per zone (name ``threshold_gwh``) in the input units. ``attrs``
        holds ``degenerate_zones`` (list of zone labels), ``positive_fraction``
        (zone -> fraction) and ``min_positive_fraction``.

    Raises
    ------
    ValueError
        If ``method`` is unknown.
    """
    zones = residual_load[zone_dim].values

    def zone_values(zone):
        """All residual-load values of one zone, flattened."""
        return residual_load.sel({zone_dim: zone}).values.flatten()

    if method == "EVT":
        evt_thresholds, _ = compute_evt_thresholds_for_all_zones(
            residual_load, p_value_threshold=p_value_threshold, zone_dim=zone_dim
        )
        raw = {str(zone): float(evt_thresholds.sel({zone_dim: zone}).values) for zone in zones}
    elif method == "TAILS":
        _, tails_percentiles, _, _ = compute_tails_thresholds_for_all_zones(
            residual_load, zone_dim=zone_dim, threshold_type="yearly"
        )
        raw = {
            str(zone): float(np.nanpercentile(
                zone_values(zone), float(tails_percentiles.sel({zone_dim: zone}).values)
            ))
            for zone in zones
        }
    elif method == "PERCENTILE":
        raw = {str(zone): find_fixed_percentile_threshold(zone_values(zone), percentile) for zone in zones}
    else:
        raise ValueError(
            f"Unknown method {method!r}. Use 'EVT', 'TAILS' or 'PERCENTILE'."
        )

    floored, degenerate_zones, positive_fraction = {}, [], {}
    for zone in zones:
        zone_key = str(zone)
        floored_threshold, is_degenerate, zone_positive_fraction = _apply_deficit_floor(
            raw[zone_key], zone_values(zone), min_positive_fraction
        )
        floored[zone_key] = floored_threshold
        positive_fraction[zone_key] = zone_positive_fraction
        if is_degenerate:
            degenerate_zones.append(zone_key)
            warnings.warn(
                f"[{method}] Zone {zone_key!r} has a degenerate extreme threshold "
                f"(raw={raw[zone_key]:.1f} GWh, deficit fraction={zone_positive_fraction:.1%} "
                f"< {min_positive_fraction:.0%}); floored to {floored_threshold:.1f} GWh. "
                f"Exclude from cross-zone tail comparisons.",
                stacklevel=2,
            )

    thresholds = xr.DataArray(
        [floored[str(zone)] for zone in zones],
        coords={zone_dim: zones},
        dims=[zone_dim],
        name="threshold_gwh",
    )
    thresholds.attrs["degenerate_zones"] = degenerate_zones
    thresholds.attrs["positive_fraction"] = positive_fraction
    thresholds.attrs["min_positive_fraction"] = min_positive_fraction
    return thresholds


def calibrate_seasonal_active_thresholds(
    reference_residual_load: xr.DataArray,
    zones,
    season_order: "list[str]",
    season_months: "dict[str, list[int]]",
    method: str,
    zone_dim: str = "zone",
    percentile: float = DEFAULT_FIXED_PERCENTILE,
    min_positive_fraction: float = DEFAULT_MIN_POSITIVE_FRACTION,
) -> "tuple[dict, dict]":
    """Per-season extreme-event threshold for every zone with the chosen method.

    Thresholds are computed on the finite values of each (zone, season), floored at
    zero, and degenerate zone-seasons emit a warning (see :func:`_apply_deficit_floor`;
    the deficit fraction is measured on all values of the season).

    Parameters
    ----------
    reference_residual_load : xarray.DataArray
        Reference-period residual load with ``zone_dim`` and ``time`` dimensions.
    zones : sequence
        Zone coordinate values.
    season_order : list of str
        Season names, e.g. ``["DJF", "MAM", "JJA", "SON"]``.
    season_months : dict
        Mapping ``season -> list of calendar months``.
    method : {"PERCENTILE", "EVT", "TAILS"}
        As in :func:`compute_active_threshold`; any value other than ``"EVT"`` and
        ``"PERCENTILE"`` runs TAILS. ``"EVT"`` uses the default significance level of
        :func:`find_evt_pot_threshold`.
    zone_dim : str, default "zone"
        Name of the zone dimension.
    percentile : float, default DEFAULT_FIXED_PERCENTILE
        Percentile (0-100) of the season's positive residual load for ``"PERCENTILE"``.
    min_positive_fraction : float, default DEFAULT_MIN_POSITIVE_FRACTION
        Minimum deficit fraction for a non-degenerate zone-season.

    Returns
    -------
    thresholds : dict
        ``{zone: {season: threshold}}`` in the input units, floored at zero; zone keys
        are strings.
    percentiles : dict
        ``{zone: {season: percentile}}``: empirical percentile (0-100) of the unfloored
        threshold among the season's finite values.
    """
    times = pd.DatetimeIndex(reference_residual_load["time"].values)
    years = times.year.values

    thresholds: dict = {}
    percentiles: dict = {}

    for zone in zones:
        zone_key = str(zone)
        thresholds[zone_key] = {}
        percentiles[zone_key] = {}
        zone_values = reference_residual_load.sel({zone_dim: zone}).values.flatten()

        for season in season_order:
            in_season = np.isin(times.month, season_months[season])
            season_values = zone_values[in_season]
            season_years = years[in_season]

            finite = np.isfinite(season_values)
            season_values_finite = season_values[finite]
            season_years_finite = season_years[finite]

            if method == "EVT":
                threshold = find_evt_pot_threshold(season_values_finite)
                threshold_percentile = float(np.mean(season_values_finite <= threshold) * 100)
            elif method == "PERCENTILE":
                threshold = find_fixed_percentile_threshold(season_values_finite, percentile)
                threshold_percentile = (
                    float(np.mean(season_values_finite <= threshold) * 100)
                    if np.isfinite(threshold) else float("nan")
                )
            else:
                result = find_tails_threshold(
                    season_values_finite,
                    years=season_years_finite,
                    threshold_type="yearly",
                    random_seed=0,
                )
                standardised = _z_normalize(season_values_finite.astype(float), season_years_finite, "yearly")
                threshold_percentile = float(np.nanmean(standardised <= result.threshold) * 100)
                threshold = float(np.nanpercentile(season_values_finite, threshold_percentile))

            floored_threshold, is_degenerate, positive_fraction = _apply_deficit_floor(
                threshold, season_values, min_positive_fraction
            )
            if is_degenerate:
                warnings.warn(
                    f"[{method}] Zone {zone_key!r} season {season!r} has a degenerate "
                    f"extreme threshold (raw={threshold:.1f} GWh, deficit fraction="
                    f"{positive_fraction:.1%} < {min_positive_fraction:.0%}); floored to "
                    f"{floored_threshold:.1f} GWh. Exclude from cross-zone tail comparisons.",
                    stacklevel=2,
                )

            thresholds[zone_key][season] = floored_threshold
            percentiles[zone_key][season] = threshold_percentile

    return thresholds, percentiles
