"""Event detection primitives and per-event metric computation.

Public entry points
-------------------
detect_events(rl, threshold, time_step_h, method)
    Identify exceedance events in a 1-D residual-load array.
    method="run_length" (default) — contiguous blocks above threshold.
    method="spa" — Sequent Peak Algorithm; brief dips below threshold are
    absorbed into an ongoing event as long as the cumulative surplus has not
    been fully discharged.

detect_seasonal_events(rl, times, threshold, season_months, time_step_h, method)
    Same as above, but restricts events to a specific meteorological season
    so they cannot span season boundaries.

calculate_climatological_baseline(ds_reference_period, time_dim)
    Compute 3-hourly seasonal mean baselines for demand and generation assets.

compute_continuous_system_metrics(ds_residual_load, ds_components, ds_baseline, ...)
    Augment a dataset with ramping rates, PCIT, and climate anomalies.

compute_comprehensive_event_metrics(ds_with_continuous_metrics, da_thresholds, ...)
    Annual event metrics (frequency, duration, energy severity, attribution).

compute_seasonal_evt_metrics(ds_rl_p, ds_met_p, ..., zones, zone_dim)
    Seasonal-conditional event metrics for one model-period slice.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import label as _label


# ---------------------------------------------------------------------------
# Standalone event-detection primitives
# ---------------------------------------------------------------------------


def _detect_events_run_length(
    rl: np.ndarray,
    threshold: float,
    time_step_h: float,
) -> list[dict]:
    """Run-length event detection: each contiguous block of rl > threshold is one event."""
    labeled, n_events = _label(rl > threshold)
    events: list[dict] = []
    for i in range(1, n_events + 1):
        idx = np.where(labeled == i)[0]
        events.append({
            "indices": idx,
            "duration_h": float(len(idx) * time_step_h),
            "peak_gw": float(np.max(rl[idx])),
            "severity_gwh": float(np.sum(np.maximum(rl[idx] - threshold, 0.0)) * time_step_h),
        })
    return events


def _detect_events_spa(
    rl: np.ndarray,
    threshold: float,
    time_step_h: float,
) -> list[dict]:
    """Sequent Peak Algorithm event detection.

    Computes the storage state K[t] = max(0, K[t-1] + (rl[t] - threshold)).
    An event is a maximal connected interval where K[t] > 0, allowing brief
    dips below the threshold to be absorbed into an ongoing event as long as
    the cumulative surplus has not been fully discharged.

    K is computed without a Python loop via:

        S[t]               = cumsum(rl - threshold)[t]
        running_min[t]     = min(0, S[0], …, S[t-1])
        K[t]               = max(0, S[t] − running_min[t])

    This is O(n) and fully vectorised — processing 175 000-step arrays in
    roughly the same wall-clock time as scipy.ndimage.label.
    """
    x = rl - threshold
    S = np.cumsum(x)
    # running_min[t] = min(0, S[0], ..., S[t-1])  (includes the "S[-1] = 0" sentinel)
    running_min = np.minimum.accumulate(np.concatenate([[0.0], S[:-1]]))
    K = np.maximum(0.0, S - running_min)

    labeled, n_events = _label(K > 0)
    events: list[dict] = []
    for i in range(1, n_events + 1):
        idx = np.where(labeled == i)[0]
        events.append({
            "indices": idx,
            "duration_h": float(len(idx) * time_step_h),
            "peak_gw": float(np.max(rl[idx])),
            # Gross excess only (non-negative even when event spans dips below threshold)
            "severity_gwh": float(np.sum(np.maximum(rl[idx] - threshold, 0.0)) * time_step_h),
        })
    return events


_DETECTION_METHODS = {"run_length": _detect_events_run_length, "spa": _detect_events_spa}


def detect_events(
    rl: np.ndarray,
    threshold: float,
    time_step_h: float = 3.0,
    method: str = "run_length",
) -> list[dict]:
    """Identify exceedance events in a 1-D residual-load array.

    Parameters
    ----------
    rl :
        1-D array of residual-load values (GW).
    threshold :
        Exceedance threshold (GW).
    time_step_h :
        Length of one time step in hours (default 3 for 3-hourly data).
    method :
        Detection algorithm:

        ``"run_length"`` *(default)*
            Each contiguous block of timesteps where ``rl > threshold`` is
            one event.  Identical to the previous behaviour.
        ``"spa"``
            Sequent Peak Algorithm.  Brief dips below *threshold* are
            absorbed into an ongoing event as long as the cumulative energy
            surplus above the threshold has not yet been fully discharged.
            Implemented via ``cumsum`` + ``np.minimum.accumulate`` — O(n),
            no Python loop, scales to 175 k-step arrays without slowdown.

    Returns
    -------
    list[dict]
        One dict per detected event.  Keys are identical for both methods:

        ``indices``
            Integer index array into *rl*.
        ``duration_h``
            Duration in hours (``len(indices) * time_step_h``).
        ``peak_gw``
            Maximum residual load during the event (GW).
        ``severity_gwh``
            Total gross energy above *threshold* during the event (GWh).
            Uses ``max(rl - threshold, 0)`` so the value is non-negative
            even when the SPA event spans dips below the threshold.
    """
    if method not in _DETECTION_METHODS:
        raise ValueError(f"method must be one of {list(_DETECTION_METHODS)}; got {method!r}")
    return _DETECTION_METHODS[method](rl, threshold, time_step_h)


DEFAULT_DURATION_BINS: dict[str, tuple[float, float]] = {
    "flash":     (0,    3),
    "short":     (3,    24),
    "multi_day": (24,   168),
    "prolonged": (168,  float("inf")),
}


def classify_event_duration(
    duration_h: float,
    bins: dict[str, tuple[float, float]] | None = None,
) -> str:
    """Return the duration-class label for a single event.

    Parameters
    ----------
    duration_h :
        Event duration in hours.
    bins :
        Ordered dict mapping label → (lo_exclusive, hi_inclusive) hour bounds.
        Defaults to ``DEFAULT_DURATION_BINS``.

    Returns
    -------
    str
        The matching label, or the last label if none match (catches fp edge cases).
    """
    bins = bins or DEFAULT_DURATION_BINS
    for label, (lo, hi) in bins.items():
        if lo == 0 and duration_h <= hi:
            return label
        if lo < duration_h <= hi:
            return label
    return list(bins)[-1]


def detect_seasonal_events(
    rl: np.ndarray,
    times: pd.DatetimeIndex,
    threshold: float,
    season_months: list[int],
    time_step_h: float = 3.0,
    method: str = "run_length",
) -> list[dict]:
    """Detect events within a specific meteorological season.

    Non-season timesteps are masked before event detection so that events
    cannot span season boundaries.  The returned event dicts are identical
    in structure to those from :func:`detect_events`.

    Parameters
    ----------
    rl :
        1-D array of residual-load values (GW).
    times :
        DatetimeIndex aligned with *rl*.
    threshold :
        Exceedance threshold (GW).
    season_months :
        List of integer month numbers that belong to this season
        (e.g. ``[12, 1, 2]`` for DJF).
    time_step_h :
        Length of one time step in hours (default 3).
    method :
        Detection algorithm — ``"run_length"`` or ``"spa"``.
        Forwarded to :func:`detect_events`.  For ``"spa"``, non-season
        timesteps are set to ``-inf`` so the storage state K is guaranteed
        to reset to zero at every season boundary (rather than draining
        gradually as with a small negative value).

    Returns
    -------
    list[dict]
        Same structure as :func:`detect_events`.
    """
    s_mask = np.isin(times.month, season_months)
    if method == "spa":
        # The -inf masking trick breaks SPA: once cumsum absorbs -inf the running
        # minimum becomes -inf, and S - (-inf) = nan for all subsequent in-season
        # steps, yielding zero events.  Instead, run SPA independently on each
        # contiguous in-season block (~60 blocks for 60 years — loop is negligible)
        # and remap the local indices back to positions in the original array.
        labeled_seasons, n_seasons = _label(s_mask)
        events: list[dict] = []
        for i in range(1, n_seasons + 1):
            seg_idx = np.where(labeled_seasons == i)[0]
            for ev in _detect_events_spa(rl[seg_idx], threshold, time_step_h):
                ev["indices"] = seg_idx[ev["indices"]]
                events.append(ev)
        return events
    # run_length: masking with a sub-threshold value is sufficient — no cumsum.
    rl_masked = rl.copy()
    rl_masked[~s_mask] = threshold - 1.0
    return _detect_events_run_length(rl_masked, threshold, time_step_h)


# ---------------------------------------------------------------------------
# Baseline and continuous metrics
# ---------------------------------------------------------------------------


def calculate_climatological_baseline(ds_reference_period, time_dim="time"):
    """Computes expected 3-hourly seasonal baselines for demand and generation assets.

    Returns a dataset grouped by (month, hour) MultiIndex representing the
    long-run mean for each variable across the reference period.
    """
    months = ds_reference_period[time_dim].dt.month
    hours = ds_reference_period[time_dim].dt.hour

    multi_idx = pd.MultiIndex.from_arrays(
        [months.values, hours.values], names=["month", "hour"]
    )

    ds_stacked = ds_reference_period.assign_coords(
        month_hour=(time_dim, multi_idx)
    )

    target_vars = ["total_demand", "gen_pv", "gen_wind_on", "gen_wind_off", "gen_hydro_ror"]
    if "reservoir_dispatch_gwh" in ds_reference_period.data_vars:
        target_vars.append("reservoir_dispatch_gwh")
    for _bat_var in ["battery_dispatch_gwh", "prosumer_dispatch_gwh", "utility_dispatch_gwh"]:
        if _bat_var in ds_reference_period.data_vars:
            target_vars.append(_bat_var)

    baseline = (
        ds_stacked[target_vars]
        .groupby("month_hour")
        .mean(dim=time_dim)
        .unstack("month_hour")
    )

    return baseline


def compute_continuous_system_metrics(
    ds_residual_load, ds_components, ds_baseline, zone_dim="zone", time_dim="time"
):
    """Compute continuous time-series metrics (Ramping and PCIT) before slicing
    into isolated extreme event blocks.

    Augments *ds_residual_load* with:

    * ``ramping_rate`` — 3-hourly ramping rate (GW/hour).
    * ``pcit`` — Potential Cross-border Interconnection Transport (GW).
    * ``net_continental_imbalance`` / ``net_system_residual_load``
    * ``anomaly_*`` — demand and generation anomalies relative to the baseline.
    """
    ds_metrics = ds_residual_load.copy()

    # 1. Ramping rates (GW/hour)
    rl_prev = ds_metrics["residual_load"].shift({time_dim: 1})
    ds_metrics["ramping_rate"] = (ds_metrics["residual_load"] - rl_prev) / 3.0
    ds_metrics["ramping_rate"] = ds_metrics["ramping_rate"].fillna(0.0)

    # 2. PCIT
    sum_absolute_imbalances = np.abs(ds_metrics["residual_load"]).sum(dim=zone_dim)
    net_continental_imbalance = np.abs(ds_metrics["residual_load"].sum(dim=zone_dim))
    ds_metrics["net_continental_imbalance"] = net_continental_imbalance
    ds_metrics["net_system_residual_load"] = ds_metrics["residual_load"].sum(dim=zone_dim)
    ds_metrics["pcit"] = 0.5 * (sum_absolute_imbalances - net_continental_imbalance)

    # 3. Climate anomalies
    current_months = ds_metrics[time_dim].dt.month
    current_hours = ds_metrics[time_dim].dt.hour

    expected_d = ds_baseline["total_demand"].sel(month=current_months, hour=current_hours)
    ds_metrics["anomaly_demand"] = (ds_components["total_demand"] - expected_d).clip(min=0.0)

    for tech in ["gen_pv", "gen_wind_on", "gen_wind_off", "gen_hydro_ror"]:
        expected_tech = ds_baseline[tech].sel(month=current_months, hour=current_hours)
        ds_metrics[f"anomaly_{tech}"] = (expected_tech - ds_components[tech]).clip(min=0.0)

    if (
        "reservoir_dispatch_gwh" in ds_components.data_vars
        and "reservoir_dispatch_gwh" in ds_baseline.data_vars
    ):
        expected_res = ds_baseline["reservoir_dispatch_gwh"].sel(
            month=current_months, hour=current_hours
        )
        ds_metrics["anomaly_reservoir_dispatch_gwh"] = (
            expected_res - ds_components["reservoir_dispatch_gwh"]
        ).clip(min=0.0)

    if (
        "battery_dispatch_gwh" in ds_components.data_vars
        and "battery_dispatch_gwh" in ds_baseline.data_vars
    ):
        expected_bat = ds_baseline["battery_dispatch_gwh"].sel(
            month=current_months, hour=current_hours
        )
        ds_metrics["anomaly_battery_dispatch_gwh"] = (
            expected_bat - ds_components["battery_dispatch_gwh"]
        ).clip(min=0.0)

    return ds_metrics


# ---------------------------------------------------------------------------
# Per-zone event metric aggregation
# ---------------------------------------------------------------------------


def compute_comprehensive_event_metrics(
    ds_with_continuous_metrics,
    da_thresholds,
    zone_dim: str = "zone",
    method: str = "run_length",
):
    """Compute Frequency, Duration, Peak Capacity, and Energy Integral metrics,
    contextually extracting Ramping and PCIT signatures during threshold violations.
    """
    metrics_summary = {}

    year_series = ds_with_continuous_metrics.time.dt.year.values
    all_years = np.unique(year_series)

    for zone in ds_with_continuous_metrics[zone_dim].values:
        H = float(da_thresholds.sel({zone_dim: zone}).values)

        rl_series = ds_with_continuous_metrics["residual_load"].sel({zone_dim: zone}).values
        ramp_series = ds_with_continuous_metrics["ramping_rate"].sel({zone_dim: zone}).values

        d_anom_series = ds_with_continuous_metrics["anomaly_demand"].sel({zone_dim: zone}).values
        pv_anom_series = ds_with_continuous_metrics["anomaly_gen_pv"].sel({zone_dim: zone}).values
        w_anom_series = ds_with_continuous_metrics["anomaly_gen_wind_on"].sel({zone_dim: zone}).values
        woff_anom_series = ds_with_continuous_metrics["anomaly_gen_wind_off"].sel({zone_dim: zone}).values
        hydro_anom_series = ds_with_continuous_metrics["anomaly_gen_hydro_ror"].sel({zone_dim: zone}).values
        reservoir_anom_series = (
            ds_with_continuous_metrics["anomaly_reservoir_dispatch_gwh"].sel({zone_dim: zone}).values
            if "anomaly_reservoir_dispatch_gwh" in ds_with_continuous_metrics
            else np.zeros(len(rl_series))
        )
        battery_anom_series = (
            ds_with_continuous_metrics["anomaly_battery_dispatch_gwh"].sel({zone_dim: zone}).values
            if "anomaly_battery_dispatch_gwh" in ds_with_continuous_metrics
            else np.zeros(len(rl_series))
        )

        pcit_series = ds_with_continuous_metrics["pcit"].values
        net_system_rl_series = ds_with_continuous_metrics["net_system_residual_load"].values

        n_years = len(all_years)

        events = detect_events(rl_series, H, method=method)
        num_features = len(events)

        if num_features == 0:
            metrics_summary[zone] = {
                "Frequency": 0,
                "Avg_Duration_Hours": 0.0,
                "Max_Duration_Hours": 0.0,
                "Max_Peak_Intensity_GW": 0.0,
                "Max_Upward_Ramp_GW_per_hr": 0.0,
                "Total_Annual_Energy_Severity_GWh": 0.0,
                "Avg_Energy_Severity_GWh": 0.0,
                "Max_Single_Event_Severity_GWh": 0.0,
                "Avg_PCIT_During_Events_GW": 0.0,
                "Max_PCIT_During_Events_GW": 0.0,
                "Avg_Net_System_Residual_Load_During_Events_GW": 0.0,
                "Max_Net_System_Residual_Load_During_Events_GW": 0.0,
                "Std_Annual_Energy_Severity_GWh": 0.0,
                "P90_Annual_Energy_Severity_GWh": 0.0,
                "CV_Annual_Energy_Severity": 0.0,
                "Demand_Driven_Share_%": 0.0,
                "Generation_Driven_Share_%": 0.0,
                "PV_Driven_Share_%": 0.0,
                "Wind_Onshore_Driven_Share_%": 0.0,
                "Wind_Offshore_Driven_Share_%": 0.0,
                "Hydro_RoR_Driven_Share_%": 0.0,
                "Reservoir_Hydro_Driven_Share_%": 0.0,
                "Battery_Driven_Share_%": 0.0,
                "Avg_Demand_Anomaly_GWh": 0.0,
                "Avg_PV_Anomaly_GWh": 0.0,
                "Avg_Wind_Onshore_Anomaly_GWh": 0.0,
                "Avg_Wind_Offshore_Anomaly_GWh": 0.0,
                "Avg_Hydro_RoR_Anomaly_GWh": 0.0,
                "Avg_Reservoir_Anomaly_GWh": 0.0,
                "Avg_Battery_Anomaly_GWh": 0.0,
            }
            continue

        (
            event_durations,
            event_peaks,
            event_integrals,
            event_max_ramps,
            event_mean_pcits,
            event_mean_net_system_rls,
        ) = [], [], [], [], [], []
        per_year_severity: dict = {}
        tot_d_anom = 0.0
        tot_pv_anom = 0.0
        tot_w_anom = 0.0
        tot_woff_anom = 0.0
        tot_hydro_anom = 0.0
        tot_reservoir_anom = 0.0
        tot_battery_anom = 0.0

        for ev in events:
            idx = ev["indices"]
            event_durations.append(ev["duration_h"])
            event_peaks.append(ev["peak_gw"])
            event_integrals.append(ev["severity_gwh"])

            event_year = year_series[idx[0]]
            per_year_severity[event_year] = per_year_severity.get(event_year, 0.0) + ev["severity_gwh"]

            event_max_ramps.append(float(np.max(ramp_series[idx])))
            event_mean_pcits.append(float(np.mean(pcit_series[idx])))
            event_mean_net_system_rls.append(float(np.mean(net_system_rl_series[idx])))

            tot_d_anom += float(np.sum(d_anom_series[idx]))
            tot_pv_anom += float(np.sum(pv_anom_series[idx]))
            tot_w_anom += float(np.sum(w_anom_series[idx]))
            tot_woff_anom += float(np.sum(woff_anom_series[idx]))
            tot_hydro_anom += float(np.sum(hydro_anom_series[idx]))
            tot_reservoir_anom += float(np.sum(reservoir_anom_series[idx]))
            tot_battery_anom += float(np.sum(battery_anom_series[idx]))

        annual_vals = np.array([per_year_severity.get(yr, 0.0) for yr in all_years])
        std_annual = float(np.std(annual_vals))
        p90_annual = float(np.percentile(annual_vals, 90))
        mean_annual = float(np.sum(event_integrals)) / n_years
        cv_annual = std_annual / mean_annual if mean_annual > 0 else 0.0

        denominator = (
            tot_d_anom + tot_pv_anom + tot_w_anom + tot_woff_anom
            + tot_hydro_anom + tot_reservoir_anom + tot_battery_anom
        )

        if denominator > 0:
            d_share = (tot_d_anom / denominator) * 100
            pv_share = (tot_pv_anom / denominator) * 100
            w_share = (tot_w_anom / denominator) * 100
            woff_share = (tot_woff_anom / denominator) * 100
            hydro_share = (tot_hydro_anom / denominator) * 100
            reservoir_share = (tot_reservoir_anom / denominator) * 100
            battery_share = (tot_battery_anom / denominator) * 100
        else:
            d_share = pv_share = w_share = woff_share = hydro_share = reservoir_share = battery_share = 100.0 / 7

        metrics_summary[zone] = {
            "Frequency": int(num_features),
            "Avg_Duration_Hours": float(np.mean(event_durations)),
            "Max_Duration_Hours": float(np.max(event_durations)),
            "Max_Peak_Intensity_GW": float(np.max(event_peaks)),
            "Max_Upward_Ramp_GW_per_hr": float(np.max(event_max_ramps)),
            "Total_Annual_Energy_Severity_GWh": float(np.sum(event_integrals)) / n_years,
            "Avg_Energy_Severity_GWh": float(np.mean(event_integrals)),
            "Max_Single_Event_Severity_GWh": float(np.max(event_integrals)),
            "Avg_PCIT_During_Events_GW": float(np.mean(event_mean_pcits)),
            "Max_PCIT_During_Events_GW": float(np.max(event_mean_pcits)),
            "Avg_Net_System_Residual_Load_During_Events_GW": float(np.mean(event_mean_net_system_rls)),
            "Max_Net_System_Residual_Load_During_Events_GW": float(np.max(event_mean_net_system_rls)),
            "Std_Annual_Energy_Severity_GWh": std_annual,
            "P90_Annual_Energy_Severity_GWh": p90_annual,
            "CV_Annual_Energy_Severity": cv_annual,
            "Demand_Driven_Share_%": round(float(d_share), 1),
            "PV_Driven_Share_%": round(float(pv_share), 1),
            "Wind_Onshore_Driven_Share_%": round(float(w_share), 1),
            "Wind_Offshore_Driven_Share_%": round(float(woff_share), 1),
            "Hydro_RoR_Driven_Share_%": round(float(hydro_share), 1),
            "Reservoir_Hydro_Driven_Share_%": round(float(reservoir_share), 1),
            "Battery_Driven_Share_%": round(float(battery_share), 1),
            "Avg_Demand_Anomaly_GWh": round(float(tot_d_anom * 3 / num_features), 2),
            "Avg_PV_Anomaly_GWh": round(-float(tot_pv_anom * 3 / num_features), 2),
            "Avg_Wind_Onshore_Anomaly_GWh": round(-float(tot_w_anom * 3 / num_features), 2),
            "Avg_Wind_Offshore_Anomaly_GWh": round(-float(tot_woff_anom * 3 / num_features), 2),
            "Avg_Hydro_RoR_Anomaly_GWh": round(-float(tot_hydro_anom * 3 / num_features), 2),
            "Avg_Reservoir_Anomaly_GWh": round(-float(tot_reservoir_anom * 3 / num_features), 2),
            "Avg_Battery_Anomaly_GWh": round(-float(tot_battery_anom * 3 / num_features), 2),
        }

    return metrics_summary


def compute_seasonal_evt_metrics(
    ds_rl_p: xr.Dataset,
    ds_met_p: xr.Dataset,
    season_order: list[str],
    season_months: dict[str, list[int]],
    season_thresholds: dict[str, dict[str, float]],
    model: str,
    period_name: str,
    zones,
    zone_dim: str = "zone",
    method: str = "run_length",
) -> list[dict]:
    """Compute seasonal-conditional EVT event metrics for one model-period slice.

    For each (zone, season) the function:
    - Zeroes out non-season timesteps so events cannot span season boundaries.
    - Detects connected exceedance blocks above the season-specific threshold.
    - Accumulates durations, energy severities, PCIT, and climate driver shares.

    Parameters
    ----------
    ds_rl_p :
        Post-battery residual load for the target period (``residual_load`` variable).
    ds_met_p :
        Output of ``compute_continuous_system_metrics`` for the same period.
    season_order :
        Season names in display order.
    season_months :
        ``{season: [month_ints]}``.
    season_thresholds :
        ``{zone: {season: float}}`` from ``calibrate_seasonal_evt_thresholds``.
    model :
        Model chain ID string (carried into output rows).
    period_name :
        Period label string (carried into output rows).
    zones :
        Zone coordinate values.
    zone_dim :
        Name of the zone dimension (default ``"zone"``).

    Returns
    -------
    list[dict]
        One row-dict per (zone, season) ready for ``pd.DataFrame``.
    """
    times_p = pd.DatetimeIndex(ds_rl_p["time"].values)
    pcit_series = ds_met_p["pcit"].values
    has_res = "anomaly_reservoir_dispatch_gwh" in ds_met_p
    has_bat = "anomaly_battery_dispatch_gwh" in ds_met_p
    rows: list[dict] = []

    for season in season_order:
        s_mask = np.isin(times_p.month, season_months[season])
        s_years = np.unique(ds_rl_p["time"].dt.year.values[s_mask])

        for zone in zones:
            zone_key = str(zone)
            rl = ds_rl_p["residual_load"].sel({zone_dim: zone}).values
            thr = season_thresholds[zone_key][season]

            events = detect_seasonal_events(rl, times_p, thr, season_months[season], method=method)
            n_ev = len(events)

            if n_ev == 0:
                rows.append({
                    "model": model, "zone": zone_key, "season": season, "Period": period_name,
                    "Frequency": 0,
                    "Avg_Duration_Hours": np.nan, "Max_Duration_Hours": np.nan,
                    "Avg_Energy_Severity_GWh": np.nan, "Max_Energy_Severity_GWh": np.nan,
                    "Total_Annual_Energy_Severity_GWh": 0.0,
                    "P90_Annual_Energy_Severity_GWh": 0.0,
                    "Std_Annual_Energy_Severity_GWh": 0.0,
                    "CV_Annual_Energy_Severity": np.nan,
                    "Avg_PCIT_During_Events_GW": np.nan,
                    "Max_PCIT_During_Events_GW": np.nan,
                    "Demand_Driven_Share_%": np.nan, "PV_Driven_Share_%": np.nan,
                    "Wind_Onshore_Driven_Share_%": np.nan, "Wind_Offshore_Driven_Share_%": np.nan,
                    "Hydro_RoR_Driven_Share_%": np.nan, "Reservoir_Hydro_Driven_Share_%": np.nan,
                    "Battery_Driven_Share_%": np.nan,
                    "Avg_Demand_Anomaly_GWh": np.nan, "Avg_PV_Anomaly_GWh": np.nan,
                    "Avg_Wind_Onshore_Anomaly_GWh": np.nan, "Avg_Wind_Offshore_Anomaly_GWh": np.nan,
                    "Avg_Hydro_RoR_Anomaly_GWh": np.nan, "Avg_Reservoir_Anomaly_GWh": np.nan,
                    "Avg_Battery_Anomaly_GWh": np.nan,
                })
                continue

            d_anom    = ds_met_p["anomaly_demand"].sel({zone_dim: zone}).values
            pv_anom   = ds_met_p["anomaly_gen_pv"].sel({zone_dim: zone}).values
            won_anom  = ds_met_p["anomaly_gen_wind_on"].sel({zone_dim: zone}).values
            woff_anom = ds_met_p["anomaly_gen_wind_off"].sel({zone_dim: zone}).values
            hror_anom = ds_met_p["anomaly_gen_hydro_ror"].sel({zone_dim: zone}).values
            res_anom  = (
                ds_met_p["anomaly_reservoir_dispatch_gwh"].sel({zone_dim: zone}).values
                if has_res else np.zeros(len(rl))
            )
            bat_anom  = (
                ds_met_p["anomaly_battery_dispatch_gwh"].sel({zone_dim: zone}).values
                if has_bat else np.zeros(len(rl))
            )

            durations: list[float] = []
            severities: list[float] = []
            event_pcit: list[float] = []
            annual_sev: dict[int, float] = {int(yr): 0.0 for yr in s_years}
            tot_d = tot_pv = tot_won = tot_woff = tot_hror = tot_res = tot_bat = 0.0

            for ev in events:
                ev_idx = ev["indices"]
                sev = ev["severity_gwh"]
                ev_year = int(times_p[ev_idx[0]].year)
                durations.append(ev["duration_h"])
                severities.append(sev)
                annual_sev[ev_year] = annual_sev.get(ev_year, 0.0) + sev
                event_pcit.extend(pcit_series[ev_idx].tolist())
                tot_d    += float(d_anom[ev_idx].sum())
                tot_pv   += float(pv_anom[ev_idx].sum())
                tot_won  += float(won_anom[ev_idx].sum())
                tot_woff += float(woff_anom[ev_idx].sum())
                tot_hror += float(hror_anom[ev_idx].sum())
                tot_res  += float(res_anom[ev_idx].sum())
                tot_bat  += float(bat_anom[ev_idx].sum())

            ann_arr = np.array(list(annual_sev.values()))
            denom = tot_d + tot_pv + tot_won + tot_woff + tot_hror + tot_res + tot_bat
            if denom > 0:
                d_sh    = tot_d    / denom * 100
                pv_sh   = tot_pv   / denom * 100
                won_sh  = tot_won  / denom * 100
                woff_sh = tot_woff / denom * 100
                hror_sh = tot_hror / denom * 100
                res_sh  = tot_res  / denom * 100
                bat_sh  = tot_bat  / denom * 100
            else:
                d_sh = pv_sh = won_sh = woff_sh = hror_sh = res_sh = bat_sh = 100.0 / 7

            rows.append({
                "model": model, "zone": zone_key, "season": season, "Period": period_name,
                "Frequency": n_ev,
                "Avg_Duration_Hours": float(np.mean(durations)),
                "Max_Duration_Hours": float(np.max(durations)),
                "Avg_Energy_Severity_GWh": float(np.mean(severities)),
                "Max_Energy_Severity_GWh": float(np.max(severities)),
                "Total_Annual_Energy_Severity_GWh": float(ann_arr.mean()),
                "P90_Annual_Energy_Severity_GWh": float(np.percentile(ann_arr, 90)),
                "Std_Annual_Energy_Severity_GWh": float(ann_arr.std()),
                "CV_Annual_Energy_Severity": (
                    float(ann_arr.std() / ann_arr.mean()) if ann_arr.mean() > 0 else np.nan
                ),
                "Avg_PCIT_During_Events_GW": float(np.mean(event_pcit)),
                "Max_PCIT_During_Events_GW": float(np.max(event_pcit)),
                "Demand_Driven_Share_%": d_sh, "PV_Driven_Share_%": pv_sh,
                "Wind_Onshore_Driven_Share_%": won_sh, "Wind_Offshore_Driven_Share_%": woff_sh,
                "Hydro_RoR_Driven_Share_%": hror_sh, "Reservoir_Hydro_Driven_Share_%": res_sh,
                "Battery_Driven_Share_%": bat_sh,
                "Avg_Demand_Anomaly_GWh": round( tot_d   * 3 / n_ev, 2),
                "Avg_PV_Anomaly_GWh":     round(-tot_pv  * 3 / n_ev, 2),
                "Avg_Wind_Onshore_Anomaly_GWh":  round(-tot_won  * 3 / n_ev, 2),
                "Avg_Wind_Offshore_Anomaly_GWh": round(-tot_woff * 3 / n_ev, 2),
                "Avg_Hydro_RoR_Anomaly_GWh":     round(-tot_hror * 3 / n_ev, 2),
                "Avg_Reservoir_Anomaly_GWh":     round(-tot_res  * 3 / n_ev, 2),
                "Avg_Battery_Anomaly_GWh":       round(-tot_bat  * 3 / n_ev, 2),
            })

    return rows
