"""Event detection on residual-load series, and the continuous system metrics.

Public functions
----------------
detect_events
    Exceedance events in a 1-D residual-load array, by run length or by the
    sequent peak algorithm (SPA).
detect_seasonal_events
    The same, restricted to one meteorological season so events cannot cross a
    season boundary.
classify_event_duration
    Duration-class label (flash, short, multi-day, prolonged) of one event.
calculate_climatological_baseline
    Mean demand and generation per (month, hour of day) over a reference period.
compute_continuous_system_metrics
    Ramping rate, potential cross-border interconnection transport (PCIT) and
    demand/generation anomalies, computed on the full time series.

Units
-----
Every residual-load and component series is energy per time step (GWh per 3-hourly
step), not power. An event's energy is therefore the plain sum of its per-step excess;
the step length is used only to convert a step count into hours.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import label as _label

#: Length of one time step of the residual-load series, in hours.
STEP_HOURS = 3.0


def _event_record(residual_load: np.ndarray, indices: np.ndarray, threshold: float,
                  time_step_h: float) -> dict:
    """Summarise one event given the time-step indices it covers.

    Parameters
    ----------
    residual_load : numpy.ndarray
        Full 1-D residual-load array (energy per step).
    indices : numpy.ndarray
        Integer indices of the event's time steps.
    threshold : float
        Exceedance threshold, same units as ``residual_load``.
    time_step_h : float
        Length of one time step in hours.

    Returns
    -------
    dict
        Keys ``indices``, ``duration_h``, ``peak_gw`` and ``severity_gwh`` as described
        in :func:`detect_events`.
    """
    return {
        "indices": indices,
        "duration_h": float(len(indices) * time_step_h),
        "peak_gw": float(np.max(residual_load[indices])),
        "severity_gwh": float(np.sum(np.maximum(residual_load[indices] - threshold, 0.0))),
    }


def _detect_events_run_length(
    residual_load: np.ndarray,
    threshold: float,
    time_step_h: float,
) -> list[dict]:
    """Detect events as contiguous runs of ``residual_load > threshold``.

    Parameters
    ----------
    residual_load : numpy.ndarray
        1-D residual-load array (energy per step).
    threshold : float
        Exceedance threshold, same units as ``residual_load``.
    time_step_h : float
        Length of one time step in hours.

    Returns
    -------
    list of dict
        One record per event; see :func:`detect_events`.
    """
    labeled, n_events = _label(residual_load > threshold)
    return [
        _event_record(residual_load, np.where(labeled == event_number)[0], threshold, time_step_h)
        for event_number in range(1, n_events + 1)
    ]


def _detect_events_spa(
    residual_load: np.ndarray,
    threshold: float,
    time_step_h: float,
) -> list[dict]:
    """Detect events with the sequent peak algorithm (SPA).

    Parameters
    ----------
    residual_load : numpy.ndarray
        1-D residual-load array (energy per step).
    threshold : float
        Exceedance threshold, same units as ``residual_load``.
    time_step_h : float
        Length of one time step in hours.

    Returns
    -------
    list of dict
        One record per event; see :func:`detect_events`.

    Notes
    -----
    The SPA tracks a cumulative deficit ("storage") state

        K[t] = max(0, K[t-1] + (r[t] - threshold)),   K[-1] = 0,

    where ``r`` is the residual load. An event is a maximal interval with ``K > 0``,
    so short dips below the threshold stay inside an event until the accumulated
    excess has been fully discharged. The recursion is evaluated without a loop as

        S[t] = sum_{s<=t} (r[s] - threshold)
        K[t] = max(0, S[t] - min(0, S[0], ..., S[t-1])).
    """
    cumulative_excess = np.cumsum(residual_load - threshold)
    previous_minimum = np.minimum.accumulate(np.concatenate([[0.0], cumulative_excess[:-1]]))
    storage_state = np.maximum(0.0, cumulative_excess - previous_minimum)

    labeled, n_events = _label(storage_state > 0)
    return [
        _event_record(residual_load, np.where(labeled == event_number)[0], threshold, time_step_h)
        for event_number in range(1, n_events + 1)
    ]


_DETECTION_METHODS = {"run_length": _detect_events_run_length, "spa": _detect_events_spa}


def detect_events(
    residual_load: np.ndarray,
    threshold: float,
    time_step_h: float = 3.0,
    method: str = "run_length",
) -> list[dict]:
    """Detect exceedance events in a 1-D residual-load array.

    Parameters
    ----------
    residual_load : numpy.ndarray
        1-D residual-load values, energy per step (GWh per 3-hourly step).
    threshold : float
        Exceedance threshold, same units as ``residual_load``.
    time_step_h : float, default 3.0
        Length of one time step in hours. Used only to convert a step count into
        ``duration_h``.
    method : {"run_length", "spa"}, default "run_length"
        ``"run_length"``: each contiguous block of steps above ``threshold`` is one
        event. ``"spa"``: sequent peak algorithm; dips below ``threshold`` are kept
        inside an event while the accumulated excess is not yet discharged (see
        :func:`_detect_events_spa`).

    Returns
    -------
    list of dict
        One dict per event, in time order, with keys

        ``indices``
            Integer indices of the event's steps into ``residual_load``.
        ``duration_h``
            ``len(indices) * time_step_h``.
        ``peak_gw``
            Maximum residual load during the event (same units as the input).
        ``severity_gwh``
            Sum over the event of ``max(residual_load - threshold, 0)``; non-negative
            even when an SPA event spans steps below the threshold.

    Raises
    ------
    ValueError
        If ``method`` is not one of the supported names.
    """
    if method not in _DETECTION_METHODS:
        raise ValueError(f"method must be one of {list(_DETECTION_METHODS)}; got {method!r}")
    return _DETECTION_METHODS[method](residual_load, threshold, time_step_h)


#: Duration classes: label -> (lower bound exclusive, upper bound inclusive) in hours.
#: The first class also includes its lower bound of 0.
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
    """Return the duration-class label of one event.

    Parameters
    ----------
    duration_h : float
        Event duration in hours.
    bins : dict, optional
        Ordered mapping ``label -> (lower_exclusive, upper_inclusive)`` in hours; a bin
        with lower bound 0 also includes 0. Defaults to :data:`DEFAULT_DURATION_BINS`.

    Returns
    -------
    str
        The first matching label, or the last label if no bin matches.
    """
    bins = bins or DEFAULT_DURATION_BINS
    for label, (lower_hours, upper_hours) in bins.items():
        if lower_hours == 0 and duration_h <= upper_hours:
            return label
        if lower_hours < duration_h <= upper_hours:
            return label
    return list(bins)[-1]


def detect_seasonal_events(
    residual_load: np.ndarray,
    times: pd.DatetimeIndex,
    threshold: float,
    season_months: list[int],
    time_step_h: float = 3.0,
    method: str = "run_length",
) -> list[dict]:
    """Detect events within one meteorological season.

    Events are confined to the season: they never span a season boundary.

    Parameters
    ----------
    residual_load : numpy.ndarray
        1-D residual-load values, energy per step.
    times : pandas.DatetimeIndex
        Time stamps aligned with ``residual_load``.
    threshold : float
        Exceedance threshold, same units as ``residual_load``.
    season_months : list of int
        Calendar months of the season, e.g. ``[12, 1, 2]`` for DJF.
    time_step_h : float, default 3.0
        Length of one time step in hours.
    method : {"run_length", "spa"}, default "run_length"
        Detection method, as in :func:`detect_events`.

    Returns
    -------
    list of dict
        Same records as :func:`detect_events`; ``indices`` refer to positions in the
        full ``residual_load`` array.

    Notes
    -----
    For ``"run_length"`` the out-of-season steps are set to ``threshold - 1`` before
    detection. For ``"spa"`` each contiguous in-season block is processed separately,
    so the storage state starts from zero in every season.
    """
    in_season = np.isin(times.month, season_months)
    if method == "spa":
        labeled_blocks, n_blocks = _label(in_season)
        events: list[dict] = []
        for block_number in range(1, n_blocks + 1):
            block_indices = np.where(labeled_blocks == block_number)[0]
            for event in _detect_events_spa(residual_load[block_indices], threshold, time_step_h):
                event["indices"] = block_indices[event["indices"]]
                events.append(event)
        return events
    residual_load_masked = residual_load.copy()
    residual_load_masked[~in_season] = threshold - 1.0
    return _detect_events_run_length(residual_load_masked, threshold, time_step_h)


def calculate_climatological_baseline(ds_reference_period, time_dim="time"):
    """Mean demand and generation per (month, hour of day) over a reference period.

    Parameters
    ----------
    ds_reference_period : xarray.Dataset
        Component series for the reference period. Must contain ``total_demand``,
        ``gen_pv``, ``gen_wind_on``, ``gen_wind_off`` and ``gen_hydro_ror``; the
        optional ``reservoir_dispatch_gwh``, ``battery_dispatch_gwh``,
        ``prosumer_dispatch_gwh`` and ``utility_dispatch_gwh`` are included when present.
    time_dim : str, default "time"
        Name of the time dimension.

    Returns
    -------
    xarray.Dataset
        The selected variables averaged over ``time_dim`` for each (``month``,
        ``hour``) pair, with ``month`` and ``hour`` as dimensions.
    """
    months = ds_reference_period[time_dim].dt.month
    hours = ds_reference_period[time_dim].dt.hour
    month_hour_index = pd.MultiIndex.from_arrays(
        [months.values, hours.values], names=["month", "hour"]
    )
    ds_stacked = ds_reference_period.assign_coords(month_hour=(time_dim, month_hour_index))

    target_vars = ["total_demand", "gen_pv", "gen_wind_on", "gen_wind_off", "gen_hydro_ror"]
    optional_vars = ["reservoir_dispatch_gwh", "battery_dispatch_gwh",
                     "prosumer_dispatch_gwh", "utility_dispatch_gwh"]
    target_vars += [name for name in optional_vars if name in ds_reference_period.data_vars]

    return (
        ds_stacked[target_vars]
        .groupby("month_hour")
        .mean(dim=time_dim)
        .unstack("month_hour")
    )


def compute_continuous_system_metrics(
    ds_residual_load, ds_components, ds_baseline, zone_dim="zone", time_dim="time"
):
    """Add ramping, PCIT and climate-anomaly series to a residual-load dataset.

    Parameters
    ----------
    ds_residual_load : xarray.Dataset
        Must contain ``residual_load`` (energy per 3-hourly step) with dimensions
        ``zone_dim`` and ``time_dim``.
    ds_components : xarray.Dataset
        Component series on the same grid: ``total_demand``, ``gen_pv``,
        ``gen_wind_on``, ``gen_wind_off``, ``gen_hydro_ror`` and optionally
        ``reservoir_dispatch_gwh`` and ``battery_dispatch_gwh``.
    ds_baseline : xarray.Dataset
        Output of :func:`calculate_climatological_baseline`.
    zone_dim : str, default "zone"
        Name of the zone dimension.
    time_dim : str, default "time"
        Name of the time dimension.

    Returns
    -------
    xarray.Dataset
        A copy of ``ds_residual_load`` with the added variables

        ``ramping_rate``
            Change of power per hour (GW/h): the series is energy per 3 h step, so
            power is ``residual_load / 3`` and its hourly change is
            ``diff(residual_load) / 9``. The first step is 0.
        ``net_system_residual_load``
            Sum of ``residual_load`` over zones.
        ``net_continental_imbalance``
            Absolute value of ``net_system_residual_load``.
        ``pcit``
            Potential cross-border interconnection transport,
            ``0.5 * (sum_zones |residual_load| - |sum_zones residual_load|)``.
        ``anomaly_demand``
            ``max(total_demand - baseline, 0)``.
        ``anomaly_<tech>``
            ``max(baseline - generation, 0)`` for each generation technology, and for
            ``reservoir_dispatch_gwh`` / ``battery_dispatch_gwh`` when present in both
            ``ds_components`` and ``ds_baseline``.
    """
    ds_metrics = ds_residual_load.copy()
    residual_load = ds_metrics["residual_load"]

    previous_step = residual_load.shift({time_dim: 1})
    ds_metrics["ramping_rate"] = ((residual_load - previous_step) / (STEP_HOURS * STEP_HOURS)).fillna(0.0)

    sum_absolute_imbalances = np.abs(residual_load).sum(dim=zone_dim)
    net_continental_imbalance = np.abs(residual_load.sum(dim=zone_dim))
    ds_metrics["net_continental_imbalance"] = net_continental_imbalance
    ds_metrics["net_system_residual_load"] = residual_load.sum(dim=zone_dim)
    ds_metrics["pcit"] = 0.5 * (sum_absolute_imbalances - net_continental_imbalance)

    current_months = ds_metrics[time_dim].dt.month
    current_hours = ds_metrics[time_dim].dt.hour

    def expected(variable):
        """Climatological value of a variable at the event's months and hours."""
        return ds_baseline[variable].sel(month=current_months, hour=current_hours)

    ds_metrics["anomaly_demand"] = (ds_components["total_demand"] - expected("total_demand")).clip(min=0.0)
    for tech in ["gen_pv", "gen_wind_on", "gen_wind_off", "gen_hydro_ror"]:
        ds_metrics[f"anomaly_{tech}"] = (expected(tech) - ds_components[tech]).clip(min=0.0)

    for dispatch in ["reservoir_dispatch_gwh", "battery_dispatch_gwh"]:
        if dispatch in ds_components.data_vars and dispatch in ds_baseline.data_vars:
            ds_metrics[f"anomaly_{dispatch}"] = (expected(dispatch) - ds_components[dispatch]).clip(min=0.0)

    return ds_metrics
