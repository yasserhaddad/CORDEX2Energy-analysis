"""POT and TAILS automated threshold selection, plus xarray calibration wrappers.

Both methods share the GPD / A²_right engine from ``extremes_statistics``.

POT (``find_evt_pot_threshold``)
    Scans candidate percentiles low→high and stops at the first threshold where
    a right-tail-weighted Anderson-Darling bootstrap test does not reject the
    GPD fit at the given significance level.

TAILS (``find_tails_threshold``)
    Implements the threshold-selection architecture of the TAILS framework
    (knee-point lower bound → fine quantile grid → filter → minimum-distance
    optimum → indistinguishable threshold range), with two substitutions for
    speed and tail sensitivity: L-moment bootstrap refits and A²_right filter
    statistic instead of the paper's standard AD.
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


# ---------------------------------------------------------------------------
# POT threshold selection
# ---------------------------------------------------------------------------


def find_evt_pot_threshold(
    zone_data,
    min_p: float = 0.9,
    max_p: float = 0.995,
    num_candidates: int = 50,
    n_bootstrap: int = 199,
    p_value_threshold: float = 0.05,
) -> float:
    """Find a statistically robust POT threshold using the right-tail Anderson-Darling test.

    Scans candidate thresholds from low to high percentile (90th → 99.5th), fitting a
    GPD to the excesses and testing the fit with the right-tail weighted Anderson-Darling
    (A²_right) test.  The first (lowest) threshold where the test does not reject at
    α = p_value_threshold is returned, the default being 0.05 to maximise the number of
    exceedances and minimise estimation variance.

    The p-value is estimated via a parametric bootstrap.  To keep the bootstrap fast,
    the bootstrap refits use L-moment estimators (closed-form, no optimiser) rather than
    MLE.  L-moments are consistent for GPD when ξ < 1 (all finite-mean cases encountered
    in practice) and produce a valid null distribution for the purpose of calibrating the
    test.  MLE is still used for the single fit to the observed excesses.

    Parameters
    ----------
    zone_data :
        1-D array of 3-hourly residual loads for the reference period.
    min_p :
        Lowest percentile (of positive RL values) to consider as a candidate.
    max_p :
        Highest percentile to consider.
    num_candidates :
        Number of evenly spaced candidate thresholds.
    n_bootstrap :
        Bootstrap replicates for the p-value estimate.

    Returns
    -------
    float
        Robust POT threshold in GW.  Falls back to the 99th percentile of
        positive RL values if no candidate passes the test.
    """
    positive_rl = zone_data[zone_data > 0]

    candidates = np.percentile(
        positive_rl, np.linspace(min_p * 100, max_p * 100, num_candidates)
    )

    best_threshold = None
    rng = np.random.default_rng(0)  # fixed seed → deterministic across models

    # Scan LOW→HIGH: stop at the first (lowest) threshold where GPD holds.
    # This maximises exceedances and minimises estimation variance.
    for u in candidates:
        excesses = positive_rl[positive_rl > u] - u

        if len(excesses) < 30:
            break  # going higher will only reduce exceedances further

        # MLE for the observed excesses (accurate single fit)
        try:
            c, loc, scale = stats.genpareto.fit(excesses, floc=0)
        except Exception:
            continue

        z_obs = np.sort(stats.genpareto.cdf(excesses, c, loc=loc, scale=scale))
        t_obs = _gpd_right_tail_ad_statistic(z_obs)

        n_exc = len(excesses)
        sim_sorted, xi_b, sigma_b, valid = _lmoment_gpd_bootstrap_refit(
            c, scale, n_exc, n_bootstrap, rng
        )
        n_valid = int(valid.sum())
        if n_valid == 0:
            continue

        xi_v = xi_b[valid, np.newaxis]
        sig_v = sigma_b[valid, np.newaxis]
        x_v = sim_sorted[valid]

        # sim_sorted rows are already sorted, and GPD CDF is monotone → z_b rows are sorted
        z_b = _gpd_cdf_vectorised(x_v, xi_v, sig_v)
        t_b = _gpd_right_tail_ad_batch(z_b)

        p_value = (float(np.sum(t_b >= t_obs)) + 1.0) / (n_valid + 1.0)

        if p_value > p_value_threshold:
            best_threshold = u
            break  # lowest threshold where GPD holds — stop here

    if best_threshold is None:
        best_threshold = np.percentile(positive_rl, 99)

    return float(best_threshold)


def compute_evt_thresholds_for_all_zones(
    ds_reference, min_p=0.9, p_value_threshold=0.05, country_dim="zone"
) -> "tuple[xr.DataArray, xr.DataArray]":
    """Vectorized wrapper to run the EVT threshold identification across all zones.

    Returns
    -------
    da_thresholds :
        EVT threshold in GW for each zone.
    min_p :
        Minimum percentile of positive RL values to consider as a candidate.
    da_percentiles :
        Empirical percentile (0–100) of the threshold within the finite
        (non-NaN) values of the reference period.  NaN entries (e.g. from a
        prior ``.where()`` filter) are stripped before both threshold finding
        and percentile computation, so the percentile is always relative to
        the domain on which the threshold was actually calibrated.
    """
    threshold_dict: dict = {}
    percentile_dict: dict = {}

    for zone in ds_reference[country_dim].values:
        zone_series = ds_reference.sel({country_dim: zone}).values.flatten()

        valid_mask = np.isfinite(zone_series)
        zone_series = zone_series[valid_mask]

        u_robust = find_evt_pot_threshold(zone_series, min_p=min_p, p_value_threshold=p_value_threshold)
        threshold_dict[zone] = u_robust
        percentile_dict[zone] = float(np.mean(zone_series <= u_robust) * 100)

    _zones = list(threshold_dict.keys())
    da_thresholds = xr.DataArray(
        list(threshold_dict.values()),
        coords={country_dim: _zones},
        dims=[country_dim],
        name="evt_threshold",
    )
    da_percentiles = xr.DataArray(
        list(percentile_dict.values()),
        coords={country_dim: _zones},
        dims=[country_dim],
        name="evt_threshold_pct",
    )

    return da_thresholds, da_percentiles


def calibrate_seasonal_evt_thresholds(
    ds_rl_ref: xr.Dataset,
    zones,
    season_order: list[str],
    season_months: dict[str, list[int]],
    zone_dim: str = "zone",
) -> "tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]":
    """Calibrate per-season EVT thresholds for all zones from the reference period.

    Parameters
    ----------
    ds_rl_ref :
        Reference-period residual load dataset (must contain ``residual_load``).
    zones :
        Zone coordinate values.
    season_order :
        Season names in display order (e.g. ``["DJF", "MAM", "JJA", "SON"]``).
    season_months :
        Mapping ``{season: [month_ints]}``.
    zone_dim :
        Name of the zone dimension (default ``"zone"``).

    Returns
    -------
    thresholds :
        ``{zone: {season: float}}`` — EVT threshold in GW for each (zone, season).
    percentiles :
        ``{zone: {season: float}}`` — empirical percentile (0–100) of each
        threshold within all 3-hourly RL values of that season's reference slice.
    """
    times_ref = pd.DatetimeIndex(ds_rl_ref["time"].values)
    thresholds: dict[str, dict[str, float]] = {}
    percentiles: dict[str, dict[str, float]] = {}
    for zone in zones:
        zk = str(zone)
        thresholds[zk] = {}
        percentiles[zk] = {}
        rl_ref_z = ds_rl_ref["residual_load"].sel({zone_dim: zone}).values
        for season in season_order:
            mask = np.isin(times_ref.month, season_months[season])
            season_rl = rl_ref_z[mask]
            thr = find_evt_pot_threshold(season_rl)
            thresholds[zk][season] = thr
            percentiles[zk][season] = float(np.mean(season_rl <= thr) * 100)
    return thresholds, percentiles


# ---------------------------------------------------------------------------
# TAILS threshold selection
# ---------------------------------------------------------------------------
# Implements the threshold-selection architecture of the TAILS framework
# (Supp. Method 1, eqs. 1-9 only — no SPA event identification), with the
# L-moment bootstrap engine and A²_right filter statistic substituted in for
# speed and tail sensitivity.
#
# Context this version is built for
# -----------------------------------
# - A FIXED reference period (1991-2020, 30 years).
# - Multiple climate data SOURCES over the same reference period.  Call
#   `find_tails_threshold` independently per (data source x region x indicator);
#   there is no ensemble-pooling logic baked in here on purpose.
# - "Event" here means a single run-length-declustered exceedance, NOT the
#   sequent-peak-algorithm (SPA) event count from the paper.
# ---------------------------------------------------------------------------


def _knee_point_quantile(values: np.ndarray, q_grid: np.ndarray) -> float:
    """Locate the quantile of maximum curvature ("knee") of the tail-transformed
    empirical distribution, used as the candidate lower bound (Supp. Method 1,
    eq. 2).

    Returns
    -------
    float
        Quantile level (in [0, 1]) of the detected knee.
    """
    x = q_grid
    y = np.quantile(values, q_grid)

    x_n = (x - x.min()) / (x.max() - x.min())
    y_n = (y - y.min()) / (y.max() - y.min() + 1e-12)

    line = y_n[0] + (y_n[-1] - y_n[0]) * (x_n - x_n[0]) / (x_n[-1] - x_n[0])
    diff = y_n - line
    knee_idx = int(np.argmax(diff))
    return float(x[knee_idx])


def _z_normalize(series: np.ndarray, years: np.ndarray, threshold_type: str) -> np.ndarray:
    """Z-normalisation per Supp. Method 1, eq. 4.

    threshold_type='yearly' -> per-year mean/std
    threshold_type='global' -> single mean/std across the whole reference period.
    """
    if threshold_type == "yearly":
        out = np.empty_like(series, dtype=float)
        for yr in np.unique(years):
            mask = years == yr
            mu, sigma = series[mask].mean(), series[mask].std(ddof=0)
            sigma = sigma if sigma > 0 else 1.0
            out[mask] = (series[mask] - mu) / sigma
        return out
    else:
        mu, sigma = series.mean(), series.std(ddof=0)
        sigma = sigma if sigma > 0 else 1.0
        return (series - mu) / sigma


def _count_events(exceed_mask: np.ndarray) -> int:
    """Decluster consecutive exceedances into events by run-length.

    One contiguous run of timesteps above threshold = one event.  Used only
    for the TAILS eq. 7 minimum-event-count floor; not the energy-system event
    detection (see ``detect_events`` in ``extremes_events``).
    """
    if exceed_mask.sum() == 0:
        return 0
    d = np.diff(exceed_mask.astype(int))
    starts = np.sum(d == 1) + (1 if exceed_mask[0] else 0)
    return int(starts)


@dataclass
class TailsResult:
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
    q_min_baseline: float = 0.95,
    knee_grid_size: int = 200,
    quantile_step: float = 0.001,
    n_bootstrap: int = 199,
    ad_p_threshold: float = 0.05,
    min_events_per_year: float = 1.0,
    n_quantile_levels: int = 50,
    random_seed: int = 0,
) -> TailsResult:
    """TAILS automated threshold selection (Supp. Method 1, eqs. 1-9 only).

    Designed for a FIXED reference period (e.g. 1991-2020) and run
    independently per climate data source x region x indicator.

    Parameters
    ----------
    series :
        Raw 1-D indicator time series for the reference period, before tail
        transformation.
    years :
        Integer calendar-year label per timestep.
    invert :
        Set True for indicators where stress = low values (e.g. renewable
        supply), so the tail transform (eq. 1) flips the series.
    threshold_type :
        'global' or 'yearly' — controls the Z-normalisation convention (eq. 4).
    q_min_baseline :
        Minimum admissible percentile (q_min in eq. 2), default 0.95.
    knee_grid_size :
        Number of quantile points used to locate the knee.
    quantile_step :
        Step size for the fine candidate grid (eq. 3), default 0.001.
    n_bootstrap :
        Bootstrap replicates per candidate threshold.
    ad_p_threshold :
        Filter cutoff on the mean A²_right-calibrated bootstrap p-value (eq. 7).
    min_events_per_year :
        Minimum identified events per year of reference period.
    n_quantile_levels :
        Z, number of evenly spaced probability levels used in the
        empirical-vs-GPD distance metric (eq. 6).
    random_seed :
        Fixed seed for reproducible bootstrap draws.

    Returns
    -------
    TailsResult
    """
    rng = np.random.default_rng(random_seed)
    n_years = len(np.unique(years))
    min_events = max(1, round(min_events_per_year * n_years))

    # Step 1: Tail transformation (eq. 1)
    transformed = -series if invert else series.astype(float)

    # Step 4: Z-normalisation (eq. 4)
    z_series = _z_normalize(transformed, years, threshold_type)

    # Step 2: Knee-point analysis (eq. 2)
    knee_grid = np.linspace(0.50, 1.0, knee_grid_size)
    knee_q = _knee_point_quantile(z_series, knee_grid)
    q_lower = max(np.ceil(knee_q * 100) / 100.0, q_min_baseline)

    # Step 3: Candidate threshold grid (eq. 3)
    candidates_q = np.arange(q_lower, 1.0 - quantile_step, quantile_step)
    if candidates_q.size == 0:
        candidates_q = np.array([q_min_baseline])

    probs = (np.arange(1, n_quantile_levels + 1)) / (n_quantile_levels + 1.0)

    distances = np.full(candidates_q.shape, np.nan)
    ad_pvalues = np.full(candidates_q.shape, np.nan)
    n_events_arr = np.zeros(candidates_q.shape, dtype=int)

    for idx, q in enumerate(candidates_q):
        u = np.quantile(z_series, q)
        exceed_mask = z_series > u
        excesses = z_series[exceed_mask] - u
        n_exc = len(excesses)

        n_events_arr[idx] = _count_events(exceed_mask)

        if n_exc < 30:
            continue

        # Step 5: GPD fit + bootstrap (eq. 5)
        try:
            c, _, scale = stats.genpareto.fit(excesses, floc=0)
        except Exception:
            continue
        if scale <= 0:
            continue

        sim_sorted, xi_b, sigma_b, valid = _lmoment_gpd_bootstrap_refit(
            c, scale, n_exc, n_bootstrap, rng
        )
        n_valid = int(valid.sum())
        if n_valid == 0:
            continue

        xi_v = xi_b[valid, np.newaxis]
        sig_v = sigma_b[valid, np.newaxis]
        x_v = sim_sorted[valid]

        z_b = _gpd_cdf_vectorised(x_v, xi_v, sig_v)

        # Filter statistic: A²_right instead of standard AD
        z_obs = np.sort(stats.genpareto.cdf(excesses, c, loc=0, scale=scale))
        t_obs = _gpd_right_tail_ad_statistic(z_obs)
        t_b = _gpd_right_tail_ad_batch(np.sort(z_b, axis=1))
        p_value = (float(np.sum(t_b >= t_obs)) + 1.0) / (n_valid + 1.0)
        ad_pvalues[idx] = p_value

        # Step 6: Quantile-quantile distance metric (eq. 6)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            theo_q = stats.genpareto.ppf(probs[np.newaxis, :], xi_v, loc=0, scale=sig_v)
        emp_q_boot = np.quantile(x_v, probs, axis=1).T  # (n_valid, Z)
        mean_abs_dist = np.mean(np.abs(emp_q_boot - theo_q), axis=1)  # (n_valid,)
        distances[idx] = float(np.mean(mean_abs_dist))

    # Step 7: Threshold filtering (eq. 7)
    passed = (
        np.isfinite(distances)
        & (ad_pvalues > ad_p_threshold)
        & (n_events_arr >= min_events)
    )

    _filter_diag = {
        "n_candidates":    len(candidates_q),
        "n_fit_ok":        int(np.sum(np.isfinite(distances))),
        "n_pass_ad":       int(np.sum(np.isfinite(distances) & (ad_pvalues > ad_p_threshold))),
        "n_pass_events":   int(np.sum(n_events_arr >= min_events)),
        "n_pass_all":      int(passed.sum()),
        "min_events_applied": min_events,
        "ad_p_threshold":  ad_p_threshold,
    }

    if not passed.any():
        fallback_q = 0.99
        fallback_val = float(np.quantile(z_series, fallback_q))
        return TailsResult(
            threshold=fallback_val,
            threshold_quantile=fallback_q,
            threshold_range=None,
            threshold_range_quantile=None,
            knee_quantile=knee_q,
            q_lower=q_lower,
            candidates_quantile=candidates_q,
            distances=distances,
            ad_pvalues=ad_pvalues,
            n_events=n_events_arr,
            passed_filter=passed,
            fallback_used=True,
            diagnostics={
                "reason": "no candidate passed eq. 7 filter",
                "n_years": n_years,
                **_filter_diag,
            },
        )

    # Step 8: Optimal threshold = argmin distance among filtered
    filtered_idx = np.where(passed)[0]
    best_local = filtered_idx[np.argmin(distances[filtered_idx])]
    best_q = candidates_q[best_local]
    best_val = float(np.quantile(z_series, best_q))
    best_dist = distances[best_local]

    # Step 9: Indistinguishable threshold range T* (eq. 9)
    u_best = np.quantile(z_series, best_q)
    exceed_mask_best = z_series > u_best
    excesses_best = z_series[exceed_mask_best] - u_best
    c_b, _, scale_b = stats.genpareto.fit(excesses_best, floc=0)
    sim_sorted_b, xi_bb, sigma_bb, valid_b = _lmoment_gpd_bootstrap_refit(
        c_b, scale_b, len(excesses_best), n_bootstrap, rng
    )
    xi_vb = xi_bb[valid_b, np.newaxis]
    sig_vb = sigma_bb[valid_b, np.newaxis]
    x_vb = sim_sorted_b[valid_b]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        theo_qb = stats.genpareto.ppf(probs[np.newaxis, :], xi_vb, loc=0, scale=sig_vb)
    emp_qb = np.quantile(x_vb, probs, axis=1).T
    boot_dists_at_opt = np.mean(np.abs(emp_qb - theo_qb), axis=1)
    ci_low, ci_high = np.percentile(boot_dists_at_opt, [16, 84])

    in_band = (distances >= ci_low) & (distances <= ci_high) & passed
    lo = best_local
    while lo - 1 >= 0 and in_band[lo - 1]:
        lo -= 1
    hi = best_local
    while hi + 1 < len(candidates_q) and in_band[hi + 1]:
        hi += 1

    q_range = (float(candidates_q[lo]), float(candidates_q[hi]))
    val_range = (
        float(np.quantile(z_series, q_range[0])),
        float(np.quantile(z_series, q_range[1])),
    )

    return TailsResult(
        threshold=best_val,
        threshold_quantile=float(best_q),
        threshold_range=val_range,
        threshold_range_quantile=q_range,
        knee_quantile=knee_q,
        q_lower=q_lower,
        candidates_quantile=candidates_q,
        distances=distances,
        ad_pvalues=ad_pvalues,
        n_events=n_events_arr,
        passed_filter=passed,
        fallback_used=False,
        diagnostics={
            "ci_band": (float(ci_low), float(ci_high)),
            "best_distance": float(best_dist),
            "n_years": n_years,
            **_filter_diag,
        },
    )


def compute_tails_thresholds_for_all_zones(
    ds_reference: xr.DataArray,
    time_dim: str = "time",
    zone_dim: str = "zone",
    invert: bool = False,
    threshold_type: str = "yearly",
    q_min_baseline: float = 0.95,
    quantile_step: float = 0.001,
    n_bootstrap: int = 199,
    ad_p_threshold: float = 0.05,
    min_events_per_year: float = 1.0,
    random_seed: int = 0,
) -> "tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]":
    """Vectorised wrapper to run TAILS threshold identification across all zones.

    Mirrors the calling convention of ``compute_evt_thresholds_for_all_zones``.
    ``years`` is derived directly from ``ds_reference[time_dim]`` so the
    underlying calendar is respected automatically.

    Parameters
    ----------
    ds_reference :
        DataArray with at least a ``time_dim`` and a ``zone_dim`` dimension.
    time_dim, zone_dim :
        Dimension names in ``ds_reference``.
    invert, threshold_type, q_min_baseline, quantile_step, n_bootstrap,
    ad_p_threshold, min_events_per_year, random_seed :
        Passed through to ``find_tails_threshold``.

    Returns
    -------
    da_thresholds :
        TAILS threshold value (in Z-normalised units) for each zone.
    da_threshold_pct :
        Empirical percentile (0–100) of the threshold within the finite
        (non-NaN) values of the zone's Z-normalised series.  NaN entries
        are stripped before both threshold finding and percentile computation,
        matching the convention of ``compute_evt_thresholds_for_all_zones``.
    da_threshold_range :
        (zone, bound) DataArray with the lower/upper edge of the
        indistinguishable threshold range T* (eq. 9).  NaN on fallback.
    da_threshold_range_pct :
        (zone, bound) DataArray with T* expressed as empirical percentiles.
        NaN on fallback.
    """
    threshold_dict: dict = {}
    pct_dict: dict = {}
    range_lo_dict: dict = {}
    range_hi_dict: dict = {}
    range_lo_pct_dict: dict = {}
    range_hi_pct_dict: dict = {}

    for zone in ds_reference[zone_dim].values:
        da_zone = ds_reference.sel({zone_dim: zone})

        zone_series = da_zone.values.flatten()
        zone_years = da_zone[time_dim].dt.year.values
        if zone_years.shape != zone_series.shape:
            zone_years = np.broadcast_to(
                zone_years.reshape(da_zone[time_dim].size, *([1] * (da_zone.ndim - 1))),
                da_zone.shape,
            ).flatten()

        valid_mask = np.isfinite(zone_series)
        zone_series = zone_series[valid_mask]
        zone_years = zone_years[valid_mask]

        result = find_tails_threshold(
            zone_series,
            years=zone_years,
            invert=invert,
            threshold_type=threshold_type,
            q_min_baseline=q_min_baseline,
            quantile_step=quantile_step,
            n_bootstrap=n_bootstrap,
            ad_p_threshold=ad_p_threshold,
            min_events_per_year=min_events_per_year,
            random_seed=random_seed,
        )

        threshold_dict[zone] = result.threshold
        if result.threshold is not None:
            transformed = -zone_series if invert else zone_series.astype(float)
            z_series_for_pct = _z_normalize(transformed, zone_years, threshold_type)
            pct_dict[zone] = float(np.nanmean(z_series_for_pct <= result.threshold) * 100)
        else:
            pct_dict[zone] = float("nan")
        if result.threshold_range is not None:
            range_lo_dict[zone], range_hi_dict[zone] = result.threshold_range
        else:
            range_lo_dict[zone], range_hi_dict[zone] = float("nan"), float("nan")
        if result.threshold_range_quantile is not None:
            range_lo_pct_dict[zone] = result.threshold_range_quantile[0] * 100.0
            range_hi_pct_dict[zone] = result.threshold_range_quantile[1] * 100.0
        else:
            range_lo_pct_dict[zone] = float("nan")
            range_hi_pct_dict[zone] = float("nan")

    _zones = list(threshold_dict.keys())
    da_thresholds = xr.DataArray(
        list(threshold_dict.values()),
        coords={zone_dim: _zones},
        dims=[zone_dim],
        name="tails_threshold",
    )
    da_threshold_pct = xr.DataArray(
        list(pct_dict.values()),
        coords={zone_dim: _zones},
        dims=[zone_dim],
        name="tails_threshold_pct",
    )
    da_threshold_range = xr.DataArray(
        np.array([[range_lo_dict[z], range_hi_dict[z]] for z in _zones]),
        coords={zone_dim: _zones, "bound": ["lower", "upper"]},
        dims=[zone_dim, "bound"],
        name="tails_threshold_range",
    )
    da_threshold_range_pct = xr.DataArray(
        np.array([[range_lo_pct_dict[z], range_hi_pct_dict[z]] for z in _zones]),
        coords={zone_dim: _zones, "bound": ["lower", "upper"]},
        dims=[zone_dim, "bound"],
        name="tails_threshold_range_pct",
    )

    return da_thresholds, da_threshold_pct, da_threshold_range, da_threshold_range_pct


# ---------------------------------------------------------------------------
# Deficit floor + coverage gate (degenerate-threshold handling)
# ---------------------------------------------------------------------------

# Minimum fraction of reference-period hours that must be in genuine deficit
# (residual load > 0) for an "extreme deficit" regime to be considered to exist.
# Sized from the CERRA reference distribution: surplus-dominated zones (e.g.
# Nordic, ~2.5 % positive hours) fall far below this, while every deficit-prone
# zone sits at >=31 % even seasonally — so 5 % cleanly separates the two.
DEFAULT_MIN_POSITIVE_FRACTION = 0.05


def _apply_deficit_floor(
    threshold_gwh: float,
    rl_values: np.ndarray,
    min_positive_fraction: float = DEFAULT_MIN_POSITIVE_FRACTION,
) -> "tuple[float, bool, float]":
    """Floor an extreme-event threshold at genuine deficit (RL > 0) and flag degeneracy.

    An "extreme energy event" must at minimum be a real supply deficit (demand
    exceeds all in-zone generation + storage). For deficit-prone zones the tail
    threshold is already well above 0, so the floor is a no-op. For
    surplus-dominated zones (e.g. hydro-rich Nordic) the unconstrained tail can
    land at or below 0 GWh — a physically empty "extreme" — which this floor
    removes.

    Parameters
    ----------
    threshold_gwh :
        Candidate threshold in GWh (original residual-load units).
    rl_values :
        1-D reference-period residual-load values for the zone (or zone-season),
        in the same units, used to measure the positive (deficit) fraction.
    min_positive_fraction :
        Minimum fraction of hours in deficit for a non-degenerate regime.

    Returns
    -------
    floored_threshold :
        ``max(threshold_gwh, 0.0)``.
    is_degenerate :
        True when the zone has too few deficit hours (positive fraction below
        ``min_positive_fraction``) or the raw threshold was already <= 0.
    positive_fraction :
        Fraction of finite values with RL > 0.
    """
    finite = rl_values[np.isfinite(rl_values)]
    positive_fraction = float(np.mean(finite > 0)) if finite.size else 0.0
    floored = max(float(threshold_gwh), 0.0)
    is_degenerate = (
        positive_fraction < min_positive_fraction or float(threshold_gwh) <= 0.0
    )
    return floored, is_degenerate, positive_fraction


def diagnose_threshold_coverage(
    da_rl: xr.DataArray,
    thresholds=None,
    zone_dim: str = "zone",
    season_map: "dict[int, str] | None" = None,
    season_order: "list[str] | None" = None,
    min_positive_fraction: float = DEFAULT_MIN_POSITIVE_FRACTION,
) -> pd.DataFrame:
    """Report deficit coverage and threshold degeneracy per zone (and season).

    Use this as a notebook diagnostic to see which zones have a genuine extreme
    deficit regime. A zone whose ``positive_fraction`` falls below
    ``min_positive_fraction`` has essentially no deficit tail; its TAILS/EVT
    threshold is degenerate (near or below 0 GWh) and it should be excluded from
    cross-zone tail comparisons.

    Parameters
    ----------
    da_rl :
        Reference-period residual load with a ``zone_dim`` (and ``time``) dim.
    thresholds :
        Optional thresholds to annotate. Either a zone-indexed ``xr.DataArray``
        (annual) or a ``{zone: {season: float}}`` mapping (seasonal). When given
        with ``season_map``, the seasonal mapping is reported per season.
    zone_dim, season_map, season_order, min_positive_fraction :
        See module/usage notes.

    Returns
    -------
    pandas.DataFrame
        One row per zone (or zone-season) with positive_fraction, threshold_gwh,
        floored threshold, and the ``degenerate`` flag.
    """
    rows = []
    zones = list(da_rl[zone_dim].values)
    times = pd.DatetimeIndex(da_rl["time"].values) if "time" in da_rl.dims else None

    for z in zones:
        zk = str(z)
        x_all = da_rl.sel({zone_dim: z}).values.flatten()
        if season_map is None:
            thr = float("nan")
            if isinstance(thresholds, xr.DataArray):
                thr = float(thresholds.sel({zone_dim: z}).values)
            floored, degenerate, pos = _apply_deficit_floor(
                thr if np.isfinite(thr) else 0.0, x_all, min_positive_fraction
            )
            rows.append({
                "zone": zk, "season": "ALL",
                "positive_fraction": pos,
                "threshold_gwh": thr,
                "threshold_floored_gwh": floored if np.isfinite(thr) else float("nan"),
                "degenerate": degenerate,
            })
        else:
            _seasons = season_order or sorted(set(season_map.values()))
            seas = np.array([season_map[m] for m in times.month])
            for s in _seasons:
                m = seas == s
                x_s = x_all[m]
                thr = float("nan")
                if isinstance(thresholds, dict):
                    thr = float(thresholds.get(zk, {}).get(s, float("nan")))
                floored, degenerate, pos = _apply_deficit_floor(
                    thr if np.isfinite(thr) else 0.0, x_s, min_positive_fraction
                )
                rows.append({
                    "zone": zk, "season": s,
                    "positive_fraction": pos,
                    "threshold_gwh": thr,
                    "threshold_floored_gwh": floored if np.isfinite(thr) else float("nan"),
                    "degenerate": degenerate,
                })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Fixed-percentile threshold selection
# ---------------------------------------------------------------------------

# Default percentile (0-100) of the positive residual-load distribution used by
# the ``"PERCENTILE"`` method. The 95th percentile of positive RL is a simple,
# transparent tail cut-off with no distributional-fit assumptions.
DEFAULT_FIXED_PERCENTILE = 95.0


def find_fixed_percentile_threshold(
    zone_data,
    percentile: float = DEFAULT_FIXED_PERCENTILE,
) -> float:
    """Fixed-percentile threshold of the positive residual-load distribution.

    A model-free alternative to the EVT/TAILS tail-selection engines: the
    threshold is simply the ``percentile``-th percentile of the strictly
    positive (deficit) residual-load values. Non-positive and non-finite values
    are dropped first so the percentile is always taken over the genuine
    deficit tail, matching the deficit-floor convention used elsewhere.

    Parameters
    ----------
    zone_data :
        1-D (or flattenable) array of residual loads for the reference period.
    percentile :
        Percentile in [0, 100] of the positive-RL distribution (default 95).

    Returns
    -------
    float
        Threshold in the original (GWh) units, or NaN if there are no positive
        values to compute a percentile from.
    """
    x = np.asarray(zone_data, dtype=float).ravel()
    positive_rl = x[np.isfinite(x) & (x > 0)]
    if positive_rl.size == 0:
        return float("nan")
    return float(np.percentile(positive_rl, percentile))


# ---------------------------------------------------------------------------
# Unified threshold calibration (method-agnostic wrappers)
# ---------------------------------------------------------------------------


def compute_active_threshold(
    da_rl: xr.DataArray,
    method: str,
    zone_dim: str = "zone",
    p_value_threshold: float = 0.1,
    percentile: float = DEFAULT_FIXED_PERCENTILE,
    min_positive_fraction: float = DEFAULT_MIN_POSITIVE_FRACTION,
) -> xr.DataArray:
    """Compute the extreme-event threshold for all zones using the chosen method.

    Thresholds are floored at genuine deficit (RL > 0): an extreme energy event
    must at minimum be a real supply deficit. This is a no-op for deficit-prone
    zones (tail already well above 0) and removes the degenerate near-/sub-zero
    "extreme" of surplus-dominated zones (e.g. hydro-rich Nordic). Zones with a
    deficit fraction below ``min_positive_fraction`` are flagged as degenerate
    via ``result.attrs`` (and a warning) so callers can exclude them from
    cross-zone tail comparisons. See :func:`diagnose_threshold_coverage`.

    Parameters
    ----------
    da_rl :
        Reference-period residual load DataArray with a ``zone_dim`` dimension.
    method :
        ``"EVT"`` for Peaks-Over-Threshold / GPD stability test,
        ``"TAILS"`` for TAILS automated distance minimisation, or
        ``"PERCENTILE"`` for a fixed percentile of the positive-RL distribution.
    zone_dim :
        Name of the zone dimension (default ``"zone"``).
    p_value_threshold :
        Passed to :func:`find_evt_pot_threshold` when ``method="EVT"``.
    percentile :
        Percentile (0-100) of the positive-RL distribution used when
        ``method="PERCENTILE"`` (default 95).
    min_positive_fraction :
        Minimum fraction of reference-period hours in deficit (RL > 0) for a
        non-degenerate extreme regime (default 5 %).

    Returns
    -------
    xr.DataArray
        Zone-indexed DataArray of thresholds in the original (GWh) units of
        ``da_rl``. ``.attrs['degenerate_zones']`` lists flagged zones and
        ``.attrs['positive_fraction']`` maps zone -> deficit fraction.
    """
    if method == "EVT":
        da_thr, _ = compute_evt_thresholds_for_all_zones(
            da_rl, p_value_threshold=p_value_threshold, country_dim=zone_dim
        )
        raw = {str(z): float(da_thr.sel({zone_dim: z}).values) for z in da_rl[zone_dim].values}
    elif method == "TAILS":
        _, da_pct, _, _ = compute_tails_thresholds_for_all_zones(
            da_rl, zone_dim=zone_dim, threshold_type="yearly"
        )
        raw = {
            str(z): float(np.nanpercentile(
                da_rl.sel({zone_dim: z}).values.flatten(),
                float(da_pct.sel({zone_dim: z}).values),
            ))
            for z in da_rl[zone_dim].values
        }
    elif method == "PERCENTILE":
        raw = {
            str(z): find_fixed_percentile_threshold(
                da_rl.sel({zone_dim: z}).values.flatten(), percentile
            )
            for z in da_rl[zone_dim].values
        }
    else:
        raise ValueError(
            f"Unknown method {method!r}. Use 'EVT', 'TAILS' or 'PERCENTILE'."
        )

    # Deficit floor + coverage gate (applies to both methods).
    floored, degenerate_zones, positive_fraction = {}, [], {}
    for z in da_rl[zone_dim].values:
        zk = str(z)
        thr_f, is_deg, pos = _apply_deficit_floor(
            raw[zk], da_rl.sel({zone_dim: z}).values.flatten(), min_positive_fraction
        )
        floored[zk] = thr_f
        positive_fraction[zk] = pos
        if is_deg:
            degenerate_zones.append(zk)
            warnings.warn(
                f"[{method}] Zone {zk!r} has a degenerate extreme threshold "
                f"(raw={raw[zk]:.1f} GWh, deficit fraction={pos:.1%} "
                f"< {min_positive_fraction:.0%}); floored to {thr_f:.1f} GWh. "
                f"Exclude from cross-zone tail comparisons.",
                stacklevel=2,
            )

    da_out = xr.DataArray(
        [floored[str(z)] for z in da_rl[zone_dim].values],
        coords={zone_dim: da_rl[zone_dim].values},
        dims=[zone_dim],
        name="threshold_gwh",
    )
    da_out.attrs["degenerate_zones"] = degenerate_zones
    da_out.attrs["positive_fraction"] = positive_fraction
    da_out.attrs["min_positive_fraction"] = min_positive_fraction
    return da_out


def calibrate_seasonal_active_thresholds(
    da_rl_ref: xr.DataArray,
    zones,
    season_order: "list[str]",
    season_months: "dict[str, list[int]]",
    method: str,
    zone_dim: str = "zone",
    percentile: float = DEFAULT_FIXED_PERCENTILE,
    min_positive_fraction: float = DEFAULT_MIN_POSITIVE_FRACTION,
) -> "tuple[dict, dict]":
    """Calibrate per-season thresholds for all zones using the chosen method.

    As in :func:`compute_active_threshold`, each (zone, season) threshold is
    floored at genuine deficit (RL > 0) — a no-op for deficit-prone
    zone-seasons and a correction for surplus-dominated ones (e.g. Nordic JJA,
    whose unconstrained TAILS threshold is negative). Degenerate zone-seasons
    (deficit fraction < ``min_positive_fraction``) emit a warning.

    Parameters
    ----------
    da_rl_ref :
        Reference-period residual load DataArray with ``zone_dim`` and ``time``.
    zones :
        Zone coordinate values.
    season_order :
        Season names in display order (e.g. ``["DJF", "MAM", "JJA", "SON"]``).
    season_months :
        Mapping ``{season: [month_ints]}``.
    method :
        ``"EVT"``, ``"TAILS"`` or ``"PERCENTILE"``.
    zone_dim :
        Name of the zone dimension (default ``"zone"``).
    percentile :
        Percentile (0-100) of the season's positive-RL distribution used when
        ``method="PERCENTILE"`` (default 95).
    min_positive_fraction :
        Minimum deficit fraction for a non-degenerate seasonal regime (default 5 %).

    Returns
    -------
    thresholds :
        ``{zone: {season: float}}`` — threshold in GWh for each (zone, season),
        floored at 0.
    percentiles :
        ``{zone: {season: float}}`` — empirical percentile (0–100) of each threshold
        within the season's own reference-period RL distribution.
    """
    times_ref = pd.DatetimeIndex(da_rl_ref["time"].values)
    years_ref  = times_ref.year.values

    thresholds: dict = {}
    percentiles: dict = {}

    for zone in zones:
        zk = str(zone)
        thresholds[zk] = {}
        percentiles[zk] = {}
        rl_zone = da_rl_ref.sel({zone_dim: zone}).values.flatten()

        for season in season_order:
            mask         = np.isin(times_ref.month, season_months[season])
            season_rl    = rl_zone[mask]
            season_years = years_ref[mask]

            # Strip NaN (e.g. from a prior .where() filter applied by the caller)
            # for threshold finding and percentile computation.  The full season_rl
            # is kept for _apply_deficit_floor so positive_fraction is correct.
            valid        = np.isfinite(season_rl)
            season_rl_v  = season_rl[valid]
            season_yrs_v = season_years[valid]

            if method == "EVT":
                thr = find_evt_pot_threshold(season_rl_v)
                pct = float(np.mean(season_rl_v <= thr) * 100)
            elif method == "PERCENTILE":
                thr = find_fixed_percentile_threshold(season_rl_v, percentile)
                pct = (
                    float(np.mean(season_rl_v <= thr) * 100)
                    if np.isfinite(thr) else float("nan")
                )
            else:  # TAILS
                result = find_tails_threshold(
                    season_rl_v,
                    years=season_yrs_v,
                    threshold_type="yearly",
                    random_seed=0,
                )
                z_season = _z_normalize(season_rl_v.astype(float), season_yrs_v, "yearly")
                pct = float(np.nanmean(z_season <= result.threshold) * 100)
                thr = float(np.nanpercentile(season_rl_v, pct))

            # Deficit floor + coverage gate (uses full season_rl for positive_fraction).
            thr_f, is_deg, pos = _apply_deficit_floor(thr, season_rl, min_positive_fraction)
            if is_deg:
                warnings.warn(
                    f"[{method}] Zone {zk!r} season {season!r} has a degenerate "
                    f"extreme threshold (raw={thr:.1f} GWh, deficit fraction="
                    f"{pos:.1%} < {min_positive_fraction:.0%}); floored to "
                    f"{thr_f:.1f} GWh. Exclude from cross-zone tail comparisons.",
                    stacklevel=2,
                )

            thresholds[zk][season] = thr_f
            percentiles[zk][season] = pct

    return thresholds, percentiles
