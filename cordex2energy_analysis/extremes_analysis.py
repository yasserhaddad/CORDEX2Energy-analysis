"""Event, attribution and storage-attenuation tables per climate chain and warming level.

Every function takes per-chain datasets (``{model: xarray.Dataset}``) and per-chain period
windows, and returns tidy tables or arrays for the figures.

Units: residual load and its components are energy per 3-hourly step (GWh), so an event's
severity is the plain sum of its positive exceedance, with no step-length factor.

Sign convention of the attribution helpers: residual load decomposes as

    RL = demand - PV - wind_on - wind_off - hydro_ror - reservoir - battery

so a rise in demand and a fall in generation or storage both raise the deficit.  Contributions are
reported with demand entering with ``+`` and every generation and storage term negated, so a
positive contribution always deepens the deficit.

Functions
---------
resolve_gwl_period_definitions
    Per-chain warming-level windows from a warming-levels YAML.
battery_split_mode
    How battery dispatch is disaggregated in a components dataset.
compute_flexibility_attenuation_metrics
    Battery and reservoir energy deployed during events, and the severity they remove.
compute_duration_stratified_metrics
    Event frequency, duration, severity and attribution per duration class.
compute_event_season_metrics
    Event counts per season and duration class from one annual detection pass.
compute_evening_peak_attribution
    Mean of every residual-load component over a window of hours of the day.
compute_multi_day_table
    Multi-day event frequency and the within-season component means that attribute them.
attribution_change_by_model
    Per-chain driver contributions to the change in residual load between two levels.
multi_day_event_anomaly_by_model
    Per-chain composition of multi-day events relative to their season's mean.
event_anomaly_vs_hourly_climatology
    Event anomaly per driver against the hour-of-day-matched seasonal climatology.
compute_deficit_budget
    Sub-daily against multi-day share of the annual deficit, and stressed-day diurnal profiles.
compute_survival_x_grid, compute_exceedance_survival, compute_exceedance_probability
    Residual-load survival curves and exceedance probabilities.
compute_country_gwl_field, compute_c2e_gwl_field, mask_structural_zeros
    Per-chain, per-country mean annual or seasonal totals over a warming-level window.
compute_annual_operation_curves
    Day-of-year storage operation curves, one per chain-year.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .analysis import load_warming_level_periods, model_spread_bounds, resolve_model_periods
from .extremes_events import (
    DEFAULT_DURATION_BINS,
    classify_event_duration,
    detect_events,
    detect_seasonal_events,
)


def _available_windows(period_windows: dict, model: str) -> dict:
    """Return ``{period: (start, end)}`` for the periods ``model`` has a window in."""
    return {period: window for period, window in resolve_model_periods(period_windows, model).items()
            if window is not None}


# ---------------------------------------------------------------------------
# Warming-level windows
# ---------------------------------------------------------------------------

#: CORDEX driving-GCM identifiers mapped to the CMIP5 names used in the warming-levels YAML.
_CORDEX_GCM_TO_CMIP5: dict[str, str] = {
    "CNRM-CERFACS-CNRM-CM5": "CNRM-CM5",
    "MOHC-HadGEM2-ES":        "HadGEM2-ES",
    "NCC-NorESM1-M":          "NorESM1-M",
    "MPI-M-MPI-ESM-LR":       "MPI-ESM-LR",
    "ICHEC-EC-EARTH":         "EC-EARTH",
}

_GWL_COLORS = ["#4A90E2", "#F5A623", "#E24A4A", "#7B2D8B"]


def resolve_gwl_period_definitions(
    warming_levels: list[float],
    models: list[str],
    chain_meta_path,
    gwl_yaml_path,
    climate_scenario: str = "rcp85",
) -> tuple[dict, dict, dict]:
    """Build per-chain warming-level windows from a warming-levels YAML.

    Parameters
    ----------
    warming_levels : list of float
        Warming levels to include, e.g. ``[1.5, 2.0, 3.0]``.
    models : list of str
        Model chain identifiers.
    chain_meta_path : str or pathlib.Path
        ``models.json`` listing the driving GCM and ensemble member of each chain.
    gwl_yaml_path : str or pathlib.Path
        Warming-levels YAML (Mathias Hauser format).
    climate_scenario : str, default "rcp85"
        Scenario whose crossing years are used.

    Returns
    -------
    period_windows : dict
        ``{"+X.X°C": {chain_id: (start_year, end_year)}}``, years as strings.  A chain whose GCM
        is not mapped, or which never reaches a level, is absent from that level's dict.
    period_label : dict
        ``{"+X.X°C": "N_+X.X°C"}``, with ``N`` the 1-based position in ``warming_levels``.
    period_colors : dict
        ``{"+X.X°C": hex_color}``.
    """
    with open(Path(chain_meta_path)) as handle:
        chain_meta = {chain["id"]: chain for chain in json.load(handle)["model_chains"]}

    ensemble_by_gcm: dict[str, str] = {}
    for chain_id in models:
        meta = chain_meta.get(chain_id, {})
        cmip5_gcm = _CORDEX_GCM_TO_CMIP5.get(meta.get("gcm", ""))
        if cmip5_gcm:
            ensemble_by_gcm[cmip5_gcm] = meta.get("ensemble", "r1i1p1")

    crossing_windows = load_warming_level_periods(
        yaml_path=Path(gwl_yaml_path),
        warming_levels=warming_levels,
        gcms=ensemble_by_gcm,
    )

    period_windows: dict = {}
    period_label: dict = {}
    period_colors: dict = {}
    for i, warming_level in enumerate(warming_levels):
        label = f"+{warming_level:.1f}°C"
        windows: dict[str, tuple[str, str]] = {}
        for chain_id in models:
            meta = chain_meta.get(chain_id, {})
            cmip5_gcm = _CORDEX_GCM_TO_CMIP5.get(meta.get("gcm", ""))
            if cmip5_gcm is None:
                continue
            window = (
                crossing_windows
                .get((cmip5_gcm, meta.get("ensemble", "r1i1p1")), {})
                .get(climate_scenario, {})
                .get(warming_level)
            )
            if window:
                windows[chain_id] = (str(window["start_year"]), str(window["end_year"]))
        period_windows[label] = windows
        period_label[label] = f"{i + 1}_{label}"
        period_colors[label] = _GWL_COLORS[i % len(_GWL_COLORS)]

    return period_windows, period_label, period_colors


# ---------------------------------------------------------------------------
# Storage attenuation
# ---------------------------------------------------------------------------


def battery_split_mode(components) -> str:
    """Return how battery dispatch is disaggregated in a components dataset.

    Parameters
    ----------
    components : xarray.Dataset
        Components dataset of one chain.

    Returns
    -------
    str
        ``"split"`` when ``battery_``, ``prosumer_`` and ``utility_dispatch_gwh`` are all present
        (``market`` flexibility scheme); ``"partial"`` when battery and exactly one of prosumer and
        utility are present (``market`` with zonal utility dispatch, read at country level);
        ``"lumped"`` when only ``battery_dispatch_gwh`` is present (``system`` scheme);
        ``"none"`` when there is no battery dispatch.

    Notes
    -----
    In every mode ``battery_dispatch_gwh`` is the total storage dispatch, so the combined battery
    metrics and the battery/reservoir Shapley shares are comparable across schemes; only the
    intra-battery breakdown depends on the mode.
    """
    variables = components.data_vars
    if "battery_dispatch_gwh" not in variables:
        return "none"
    n_split = ("prosumer_dispatch_gwh" in variables) + ("utility_dispatch_gwh" in variables)
    return {2: "split", 1: "partial", 0: "lumped"}[n_split]


def _onset_fill_pct(state, first_index: int) -> float:
    """State of charge at ``first_index`` as a per cent of the series maximum (NaN if unknown)."""
    if state is None:
        return np.nan
    top = float(np.nanmax(state))
    return float(state[first_index]) / top * 100 if top > 0 else np.nan


def _nan_mean(values: list) -> float:
    """Mean ignoring NaN; NaN for an empty list."""
    array = np.array(values, dtype=float)
    return float(np.nanmean(array)) if array.size else np.nan


def _flex_event_metrics(
    events, residual_load, threshold,
    battery_dispatch, prosumer_dispatch, utility_dispatch, reservoir_dispatch,
    has_prosumer, has_utility, has_battery, has_reservoir,
    reservoir_storage=None, battery_charge=None, per_event=None,
) -> dict:
    """Average storage deployment and attenuation over a list of events.

    Parameters
    ----------
    events : list of dict
        Events from :func:`~cordex2energy_analysis.extremes_events.detect_events`.
    residual_load : numpy.ndarray
        Post-dispatch residual load, GWh per step.
    threshold : float
        Deficit threshold, GWh per step.
    battery_dispatch, prosumer_dispatch, utility_dispatch, reservoir_dispatch : numpy.ndarray
        Dispatch series aligned with ``residual_load`` (positive = discharge), GWh per step.
    has_prosumer, has_utility, has_battery, has_reservoir : bool
        Whether the asset exists; per-asset attenuation is NaN when it does not.
    reservoir_storage, battery_charge : numpy.ndarray, optional
        Reservoir storage and battery state of charge, used only for the per-event onset columns.
    per_event : list, optional
        If given, one dict per event is appended to it (see Notes).

    Returns
    -------
    dict
        ``n_events``, ``*_deployed_gwh_per_event``, ``*_attenuation_pct``,
        ``combined_attenuation_pct``, ``battery_attenuation_share_pct`` and
        ``reservoir_attenuation_share_pct``, each the mean over events (NaN when there are none).

    Notes
    -----
    Severity is ``sum(max(RL - threshold, 0))`` over the event's steps.  The counterfactual
    severity without an asset adds that asset's signed dispatch back to the residual load
    (discharge raises it, charging inside the event lowers it), and the
    attenuation is ``(counterfactual - actual) / counterfactual * 100``.  Per-asset attenuations
    are leave-one-out and can overlap; the combined attenuation is split additively between battery
    and reservoir with the two-player Shapley value

        phi_battery   = ((S_none - S_reservoir) + (S_battery - S_actual)) / 2
        phi_reservoir = ((S_none - S_battery) + (S_reservoir - S_actual)) / 2

    where ``S_x`` is the severity with only asset ``x`` removed and ``S_none`` with both removed,
    so ``phi_battery + phi_reservoir = S_none - S_actual`` exactly.

    Each per-event dict holds ``duration_h``, ``sev_actual``, ``sev_no_storage``, the Shapley
    split in GWh (``absorbed_reservoir``, ``absorbed_battery``), the combined attenuation
    (``atten``), the energy deployed (``res_dep``, ``bat_dep``), and ``res_onset_pct`` /
    ``bat_onset_pct``: the storage level at the event's first step as a per cent of the series
    maximum.
    """
    battery_deployed, prosumer_deployed, utility_deployed, reservoir_deployed = [], [], [], []
    battery_attenuation, prosumer_attenuation, utility_attenuation = [], [], []
    reservoir_attenuation, combined_attenuation = [], []
    battery_share, reservoir_share = [], []

    for event in events:
        indices = event["indices"]
        severity_actual = float(np.maximum(residual_load[indices] - threshold, 0.0).sum())

        def severity_without(dispatch):
            """Event severity with the given signed dispatch added back to the residual load."""
            return float(np.maximum((residual_load[indices] + dispatch) - threshold, 0.0).sum())

        def attenuation(severity_counterfactual, present):
            """Per-cent reduction of a counterfactual severity; NaN if the asset is absent."""
            if not present:
                return np.nan
            if severity_counterfactual > 0:
                return (severity_counterfactual - severity_actual) / severity_counterfactual * 100
            return 0.0

        battery_discharge = np.maximum(battery_dispatch[indices], 0.0)
        prosumer_discharge = np.maximum(prosumer_dispatch[indices], 0.0)
        utility_discharge = np.maximum(utility_dispatch[indices], 0.0)
        reservoir_discharge = np.maximum(reservoir_dispatch[indices], 0.0)

        # The counterfactual removes the signed dispatch: discharge is added back, and charging
        # inside the event (which raised the residual load) is taken out.
        battery_signed = battery_dispatch[indices]
        reservoir_signed = reservoir_dispatch[indices]
        severity_no_battery = severity_without(battery_signed)
        severity_no_prosumer = (severity_without(prosumer_dispatch[indices]) if has_prosumer
                                else np.nan)
        severity_no_utility = (severity_without(utility_dispatch[indices]) if has_utility
                               else np.nan)
        severity_no_reservoir = severity_without(reservoir_signed)
        severity_no_storage = severity_without(battery_signed + reservoir_signed)

        battery_deployed.append(float(battery_discharge.sum()))
        prosumer_deployed.append(float(prosumer_discharge.sum()) if has_prosumer else np.nan)
        utility_deployed.append(float(utility_discharge.sum()) if has_utility else np.nan)
        reservoir_deployed.append(float(reservoir_discharge.sum()))

        battery_attenuation.append(attenuation(severity_no_battery, has_battery))
        prosumer_attenuation.append(attenuation(severity_no_prosumer, has_prosumer))
        utility_attenuation.append(attenuation(severity_no_utility, has_utility))
        reservoir_attenuation.append(attenuation(severity_no_reservoir, has_reservoir))
        combined_attenuation.append(attenuation(severity_no_storage, True))

        phi_battery = (0.5 * (severity_no_storage - severity_no_reservoir)
                       + 0.5 * (severity_no_battery - severity_actual))
        phi_reservoir = (0.5 * (severity_no_storage - severity_no_battery)
                         + 0.5 * (severity_no_reservoir - severity_actual))
        if severity_no_storage > 0:
            battery_share.append(phi_battery / severity_no_storage * 100)
            reservoir_share.append(phi_reservoir / severity_no_storage * 100)
        else:
            battery_share.append(0.0)
            reservoir_share.append(0.0)

        if per_event is not None:
            per_event.append({
                "duration_h": float(event["duration_h"]),
                "sev_actual": severity_actual,
                "sev_no_storage": severity_no_storage,
                "absorbed_reservoir": phi_reservoir,
                "absorbed_battery": phi_battery,
                "atten": combined_attenuation[-1],
                "res_dep": reservoir_deployed[-1],
                "bat_dep": battery_deployed[-1],
                "res_onset_pct": _onset_fill_pct(reservoir_storage, indices[0]),
                "bat_onset_pct": _onset_fill_pct(battery_charge, indices[0]),
            })

    return {
        "n_events": len(events),
        "battery_deployed_gwh_per_event": _nan_mean(battery_deployed),
        "prosumer_deployed_gwh_per_event": _nan_mean(prosumer_deployed),
        "utility_deployed_gwh_per_event": _nan_mean(utility_deployed),
        "reservoir_deployed_gwh_per_event": _nan_mean(reservoir_deployed),
        "battery_attenuation_pct": _nan_mean(battery_attenuation),
        "prosumer_attenuation_pct": _nan_mean(prosumer_attenuation),
        "utility_attenuation_pct": _nan_mean(utility_attenuation),
        "reservoir_attenuation_pct": _nan_mean(reservoir_attenuation),
        "combined_attenuation_pct": _nan_mean(combined_attenuation),
        "battery_attenuation_share_pct": _nan_mean(battery_share),
        "reservoir_attenuation_share_pct": _nan_mean(reservoir_share),
    }


def _zero_outside(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Copy of ``values`` with every step outside ``mask`` set to zero."""
    masked = values.copy()
    masked[~mask] = 0.0
    return masked


def compute_flexibility_attenuation_metrics(
    residual_load_by_model: dict,
    components_by_model: dict,
    threshold_by_model: dict,
    period_windows: dict,
    models,
    zones,
    season_map: "dict[int, str] | None" = None,
    zone_dim: str = "zone",
    method: str = "run_length",
    duration_bins: "dict | None" = None,
    return_events: bool = False,
) -> pd.DataFrame:
    """Battery and reservoir energy deployed during events, and the severity it removes.

    Parameters
    ----------
    residual_load_by_model : dict
        ``{model: xarray.Dataset}`` with the post-dispatch ``residual_load`` (GWh per step).
    components_by_model : dict
        ``{model: xarray.Dataset}`` with ``battery_dispatch_gwh``, ``prosumer_dispatch_gwh``,
        ``utility_dispatch_gwh`` and ``reservoir_dispatch_gwh``; a missing variable is read as
        zero dispatch and its per-asset metrics are NaN (see :func:`battery_split_mode`).
        ``reservoir_storage_gwh`` and ``battery_soc_gwh`` are read only when ``return_events``.
    threshold_by_model : dict
        Annual ``{model: xarray.DataArray}`` thresholds over ``zone_dim``, or seasonal
        ``{model: {zone: {season: float}}}`` thresholds (detected by type).
    period_windows : dict
        Fixed ``{period: (start, end)}`` or per-chain ``{period: {model: (start, end)}}`` windows.
    models : iterable of str
        Model chain identifiers.
    zones : iterable
        Zone coordinate values.
    season_map : dict, optional
        ``{month: season}``.  When given, metrics are computed per season, with events restricted
        to the season's months.
    zone_dim : str, default "zone"
        Name of the zone dimension.
    method : {"run_length", "spa"}, default "run_length"
        Event-detection algorithm.
    duration_bins : dict, optional
        ``{label: (low_exclusive, high_inclusive)}`` in hours.  When given, events are partitioned
        into duration classes and one row is emitted per class.
    return_events : bool, default False
        Also return one row per event (see Returns).

    Returns
    -------
    pandas.DataFrame
        One row per (model, period, zone[, season][, duration_class]) with the string column
        ``battery_mode`` and the metrics of :func:`_flex_event_metrics`.  The column schema is the
        same in every battery mode, so ``market`` and ``system`` runs can be concatenated.
    pandas.DataFrame
        Only when ``return_events``: one row per event with the same keys and the per-event
        columns of :func:`_flex_event_metrics`.

    Notes
    -----
    The per-asset ``*_attenuation_pct`` are leave-one-out quantities with separate counterfactual
    denominators, so they can sum to more than 100 %.  ``battery_attenuation_share_pct`` and
    ``reservoir_attenuation_share_pct`` are the additive Shapley split of
    ``combined_attenuation_pct``.

    With ``season_map``, dispatch is zeroed outside the season.  For ``"spa"`` events are detected
    on each contiguous in-season block; for ``"run_length"`` out-of-season steps are set just below
    the threshold before detection.
    """
    if season_map is not None:
        season_names: list = list(dict.fromkeys(season_map.values()))
    else:
        season_names = [None]

    records: list[dict] = []
    event_records: list[dict] = []

    for model in models:
        residual_load_dataset = residual_load_by_model[model]
        components_dataset = components_by_model[model]
        model_threshold = threshold_by_model[model]
        seasonal_threshold = isinstance(model_threshold, dict)

        has_battery = "battery_dispatch_gwh" in components_dataset.data_vars
        has_prosumer = "prosumer_dispatch_gwh" in components_dataset.data_vars
        has_utility = "utility_dispatch_gwh" in components_dataset.data_vars
        has_reservoir = "reservoir_dispatch_gwh" in components_dataset.data_vars
        # On every row, so a NaN per-asset metric is distinguishable from a genuine zero.
        battery_mode = battery_split_mode(components_dataset)

        for period_name, (start, end) in _available_windows(period_windows, model).items():
            residual_load_window = residual_load_dataset.sel(time=slice(start, end))
            components_window = components_dataset.sel(time=slice(start, end))
            time_months = residual_load_window["time"].dt.month.values
            times = pd.DatetimeIndex(residual_load_window["time"].values)

            for zone in zones:
                zone_residual_load = residual_load_window["residual_load"].sel({zone_dim: zone}).values

                def zone_series(variable, present):
                    """The zone's series of a component, or zeros if the asset is absent."""
                    if present:
                        return components_window[variable].sel({zone_dim: zone}).values
                    return np.zeros_like(zone_residual_load)

                def optional_zone_series(variable):
                    """The zone's series of a component, or None if the run does not store it."""
                    if variable in components_window.data_vars:
                        return components_window[variable].sel({zone_dim: zone}).values
                    return None

                battery_full = zone_series("battery_dispatch_gwh", has_battery)
                prosumer_full = zone_series("prosumer_dispatch_gwh", has_prosumer)
                utility_full = zone_series("utility_dispatch_gwh", has_utility)
                reservoir_full = zone_series("reservoir_dispatch_gwh", has_reservoir)
                reservoir_storage = battery_charge = None
                if return_events:
                    reservoir_storage = optional_zone_series("reservoir_storage_gwh")
                    battery_charge = optional_zone_series("battery_soc_gwh")

                for season in season_names:
                    if seasonal_threshold and season is not None:
                        threshold = float(model_threshold[str(zone)][season])
                    else:
                        threshold = float(model_threshold.sel({zone_dim: zone}).values)

                    if season is None:
                        residual_load = zone_residual_load
                        battery, prosumer = battery_full, prosumer_full
                        utility, reservoir = utility_full, reservoir_full
                        events = detect_events(residual_load, threshold, method=method)
                    else:
                        season_months = sorted({month for month, name in season_map.items()
                                                if name == season})
                        in_season = np.isin(time_months, season_months)
                        battery = _zero_outside(battery_full, in_season)
                        prosumer = _zero_outside(prosumer_full, in_season)
                        utility = _zero_outside(utility_full, in_season)
                        reservoir = _zero_outside(reservoir_full, in_season)
                        if method == "spa":
                            residual_load = zone_residual_load
                            events = detect_seasonal_events(
                                zone_residual_load, times, threshold, season_months, method="spa")
                        else:
                            residual_load = zone_residual_load.copy()
                            residual_load[~in_season] = threshold - 1.0
                            events = detect_events(residual_load, threshold, method=method)

                    base_row = {"model": model, "period": period_name, "zone": str(zone),
                                "battery_mode": battery_mode}
                    if season is not None:
                        base_row["season"] = season

                    def summarise(cell_events, row):
                        """Append the storage metrics of a set of events, and optionally its per-event rows."""
                        collected = [] if return_events else None
                        summary = _flex_event_metrics(
                            cell_events, residual_load, threshold,
                            battery, prosumer, utility, reservoir,
                            has_prosumer, has_utility, has_battery, has_reservoir,
                            reservoir_storage=reservoir_storage, battery_charge=battery_charge,
                            per_event=collected)
                        for entry in collected or ():
                            event_records.append({**row, **entry})
                        records.append({**row, **summary})

                    if duration_bins is None:
                        summarise(events, base_row)
                    else:
                        by_class = {duration_class: [] for duration_class in duration_bins}
                        for event in events:
                            by_class[classify_event_duration(event["duration_h"], duration_bins)].append(event)
                        for duration_class, class_events in by_class.items():
                            summarise(class_events, {**base_row, "duration_class": duration_class})

    if return_events:
        return pd.DataFrame(records), pd.DataFrame(event_records)
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Duration-stratified and seasonal event metrics
# ---------------------------------------------------------------------------

#: (anomaly variable in the metrics dataset, share-column stem, anomaly-column stem, sign, optional).
#: The sign turns a generation or storage anomaly into its contribution to the deficit.
_ATTRIBUTION_DRIVERS = (
    ("anomaly_demand", "Demand", "Demand", 1.0, False),
    ("anomaly_gen_pv", "PV", "PV", -1.0, False),
    ("anomaly_gen_wind_on", "Wind_Onshore", "Wind_Onshore", -1.0, False),
    ("anomaly_gen_wind_off", "Wind_Offshore", "Wind_Offshore", -1.0, False),
    ("anomaly_gen_hydro_ror", "Hydro_RoR", "Hydro_RoR", -1.0, False),
    ("anomaly_reservoir_dispatch_gwh", "Reservoir_Hydro", "Reservoir", -1.0, True),
    ("anomaly_battery_dispatch_gwh", "Battery", "Battery", -1.0, True),
)
_SHARE_COLUMNS = [f"{stem}_Driven_Share_%" for _, stem, _, _, _ in _ATTRIBUTION_DRIVERS]
_ANOMALY_COLUMNS = [f"Avg_{stem}_Anomaly_GWh" for _, _, stem, _, _ in _ATTRIBUTION_DRIVERS]


def _attribution_shares(anomaly_totals: list[float]) -> dict:
    """Return each driver's share (%) of the summed anomaly totals.

    Parameters
    ----------
    anomaly_totals : list of float
        Summed event anomaly of each driver, in the order of ``_ATTRIBUTION_DRIVERS``.

    Returns
    -------
    dict
        ``{share column: per cent}``; equal shares when the total is not positive.
    """
    denominator = sum(anomaly_totals)
    if denominator > 0:
        return {column: total / denominator * 100
                for column, total in zip(_SHARE_COLUMNS, anomaly_totals)}
    return {column: 100.0 / 7 for column in _SHARE_COLUMNS}


def _empty_duration_row(model: str, period: str, zone: str, duration_class: str) -> dict:
    """Return the row of a (model, period, zone, duration_class) cell without events."""
    row = {
        "model": model, "period": period, "zone": zone,
        "duration_class": duration_class,
        "n_events": 0, "freq_per_year": 0.0,
        "avg_duration_h": np.nan, "max_duration_h": np.nan,
        "avg_severity_gwh": np.nan, "max_severity_gwh": np.nan,
        "total_annual_severity_gwh": 0.0, "p90_annual_severity_gwh": 0.0,
    }
    row.update({column: np.nan for column in _SHARE_COLUMNS})
    row.update({column: np.nan for column in _ANOMALY_COLUMNS})
    return row


def compute_duration_stratified_metrics(
    residual_load_by_model: dict,
    metrics_by_model: dict,
    threshold_by_model: dict,
    period_windows: dict,
    models,
    zones,
    zone_dim: str = "zone",
    method: str = "spa",
    duration_bins: dict | None = None,
) -> pd.DataFrame:
    """Event frequency, duration, severity and attribution per duration class.

    Parameters
    ----------
    residual_load_by_model : dict
        ``{model: xarray.Dataset}`` with the post-dispatch ``residual_load`` (GWh per step).
    metrics_by_model : dict
        ``{model: xarray.Dataset}`` from
        :func:`~cordex2energy_analysis.extremes_events.compute_continuous_system_metrics`, with the
        component anomalies ``anomaly_demand``, ``anomaly_gen_pv``, ``anomaly_gen_wind_on``,
        ``anomaly_gen_wind_off``, ``anomaly_gen_hydro_ror`` and, optionally,
        ``anomaly_reservoir_dispatch_gwh`` and ``anomaly_battery_dispatch_gwh``.
    threshold_by_model : dict
        ``{model: xarray.DataArray}`` annual thresholds over ``zone_dim``.
    period_windows : dict
        Fixed or per-chain period windows (as returned by :func:`resolve_gwl_period_definitions`).
    models : iterable of str
        Model chain identifiers.
    zones : iterable
        Zone coordinate values.
    zone_dim : str, default "zone"
        Name of the zone dimension.
    method : {"spa", "run_length"}, default "spa"
        Event-detection algorithm.
    duration_bins : dict, optional
        ``{label: (low_exclusive, high_inclusive)}`` in hours; defaults to
        ``DEFAULT_DURATION_BINS``.

    Returns
    -------
    pandas.DataFrame
        One row per (model, period, zone, duration_class) with ``n_events``, ``freq_per_year``,
        average and maximum duration (h) and severity (GWh), the mean annual severity, the 90th
        percentile of annual severity, ``Avg_*_Anomaly_GWh`` (mean anomaly per event, as a
        contribution to the deficit) and ``*_Driven_Share_%``.

    Notes
    -----
    Events are detected once over the window and partitioned by duration.  An event's year is that
    of its first step.  Shares are each driver's summed contribution divided by the sum over all
    drivers.
    """
    bins = duration_bins or DEFAULT_DURATION_BINS
    records: list[dict] = []

    for model in models:
        residual_load_dataset = residual_load_by_model[model]
        metrics_dataset = metrics_by_model[model]
        model_threshold = threshold_by_model[model]

        for period_name, (start, end) in _available_windows(period_windows, model).items():
            residual_load_window = residual_load_dataset.sel(time=slice(start, end))
            metrics_window = metrics_dataset.sel(time=slice(start, end))
            step_years = residual_load_window["time"].dt.year.values
            all_years = np.unique(step_years)
            n_years = len(all_years)

            for zone in zones:
                zone_name = str(zone)
                threshold = float(model_threshold.sel({zone_dim: zone}).values)
                residual_load = residual_load_window["residual_load"].sel({zone_dim: zone}).values
                anomalies = [
                    metrics_window[variable].sel({zone_dim: zone}).values
                    if not optional or variable in metrics_window else np.zeros(len(residual_load))
                    for variable, _, _, _, optional in _ATTRIBUTION_DRIVERS
                ]

                by_class: dict[str, list[dict]] = {duration_class: [] for duration_class in bins}
                for event in detect_events(residual_load, threshold, method=method):
                    by_class[classify_event_duration(event["duration_h"], bins)].append(event)

                for duration_class, class_events in by_class.items():
                    if not class_events:
                        records.append(_empty_duration_row(
                            model, period_name, zone_name, duration_class))
                        continue

                    n_events = len(class_events)
                    durations: list[float] = []
                    severities: list[float] = []
                    severity_by_year: dict = {int(year): 0.0 for year in all_years}
                    anomaly_totals = [0.0] * len(_ATTRIBUTION_DRIVERS)

                    for event in class_events:
                        indices = event["indices"]
                        severity = event["severity_gwh"]
                        event_year = int(step_years[indices[0]])
                        durations.append(event["duration_h"])
                        severities.append(severity)
                        severity_by_year[event_year] = severity_by_year.get(event_year, 0.0) + severity
                        for k, anomaly in enumerate(anomalies):
                            anomaly_totals[k] += float(anomaly[indices].sum())

                    annual_severity = np.array(list(severity_by_year.values()))
                    record = {
                        "model": model, "period": period_name, "zone": zone_name,
                        "duration_class": duration_class,
                        "n_events": n_events,
                        "freq_per_year": n_events / n_years,
                        "avg_duration_h": float(np.mean(durations)),
                        "max_duration_h": float(np.max(durations)),
                        "avg_severity_gwh": float(np.mean(severities)),
                        "max_severity_gwh": float(np.max(severities)),
                        "total_annual_severity_gwh": float(np.sum(severities)) / n_years,
                        "p90_annual_severity_gwh": float(np.percentile(annual_severity, 90)),
                    }
                    for column, total, (_, _, _, sign, _) in zip(
                            _ANOMALY_COLUMNS, anomaly_totals, _ATTRIBUTION_DRIVERS):
                        record[column] = round((total if sign > 0 else -total) / n_events, 2)
                    record.update(_attribution_shares(anomaly_totals))
                    records.append(record)

    return pd.DataFrame(records)


def compute_event_season_metrics(
    residual_load_by_model: dict,
    threshold_by_model: dict,
    period_windows: dict,
    models,
    zones,
    season_map: dict,
    season_order,
    *,
    zone_dim: str = "zone",
    method: str = "spa",
    duration_bins: dict | None = None,
) -> pd.DataFrame:
    """Event counts per season and duration class from one annual detection pass.

    Parameters
    ----------
    residual_load_by_model : dict
        ``{model: xarray.Dataset}`` with ``residual_load`` (GWh per step).
    threshold_by_model : dict
        ``{model: xarray.DataArray}`` annual thresholds over ``zone_dim``.
    period_windows : dict
        Fixed or per-chain period windows.
    models : iterable of str
        Model chain identifiers.
    zones : iterable
        Zone coordinate values.
    season_map : dict
        ``{month: season}``.
    season_order : sequence of str
        Seasons to report; must cover every value of ``season_map``.
    zone_dim : str, default "zone"
        Name of the zone dimension.
    method : {"spa", "run_length"}, default "spa"
        Event-detection algorithm.
    duration_bins : dict, optional
        ``{label: (low_exclusive, high_inclusive)}`` in hours; defaults to
        ``DEFAULT_DURATION_BINS``.

    Returns
    -------
    pandas.DataFrame
        One row per (model, period, zone, season, duration_class) with ``n_events``,
        ``freq_per_year`` and ``avg_duration_h`` (mean full duration of the season's events).

    Notes
    -----
    Events are detected once over the whole window, as in
    :func:`compute_duration_stratified_metrics`, and each event is assigned whole to the season
    holding most of its steps (a tie goes to the season of its middle step).  The seasonal
    frequencies therefore sum exactly to the annual frequency, and an event's duration class is
    decided on its full duration rather than on the part inside one season.
    """
    bins = duration_bins or DEFAULT_DURATION_BINS
    records: list[dict] = []

    for model in models:
        for period_name, (start, end) in _available_windows(period_windows, model).items():
            residual_load_window = residual_load_by_model[model].sel(time=slice(start, end))
            times = pd.DatetimeIndex(residual_load_window["time"].values)
            step_seasons = np.array([season_map[int(month)] for month in times.month])
            n_years = len(np.unique(times.year))

            for zone in zones:
                series = residual_load_window["residual_load"].sel({zone_dim: zone}).values
                threshold = float(threshold_by_model[model].sel({zone_dim: zone}).values)
                durations_by_cell: dict[tuple, list] = {
                    (season, duration_class): [] for season in season_order for duration_class in bins}

                for event in detect_events(series, threshold, method=method):
                    indices = np.asarray(event["indices"])
                    labels, counts = np.unique(step_seasons[indices], return_counts=True)
                    winners = labels[counts == counts.max()]
                    season = (str(step_seasons[indices[len(indices) // 2]]) if len(winners) > 1
                              else str(winners[0]))
                    duration_class = classify_event_duration(event["duration_h"], bins)
                    durations_by_cell[(season, duration_class)].append(float(event["duration_h"]))

                for (season, duration_class), durations in durations_by_cell.items():
                    records.append(dict(
                        model=model, period=period_name, zone=str(zone), season=season,
                        duration_class=duration_class, n_events=len(durations),
                        freq_per_year=len(durations) / n_years,
                        avg_duration_h=float(np.mean(durations)) if durations else np.nan))
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Driver attribution
# ---------------------------------------------------------------------------

#: Residual-load components mapped to their display names, in the stacking order of the
#: attribution bars.
PAPER_COMPONENT_LABELS = {
    "total_demand": "Demand",
    "gen_pv": "PV",
    "gen_wind_on": "Wind Onshore",
    "gen_wind_off": "Wind Offshore",
    "gen_hydro_ror": "Hydro RoR",
    "reservoir_dispatch_gwh": "Reservoir Hydro",
    "battery_dispatch_gwh": "Battery",
}

#: Hours (UTC, step start) of the evening demand peak, after PV has faded.
EVENING_PEAK_HOURS = (15, 18, 21)

#: Hours (UTC, step start) occupied by the deficit: the evening peak and the night, i.e. every
#: step except the midday PV hours.  Sub-daily events run on from the evening into the night.
DEFICIT_WINDOW_HOURS = (0, 3, 6, 15, 18, 21)


def _zone_components(components_window, zone, labels: dict) -> dict:
    """Return ``{variable: values}`` for the labelled components present in the dataset."""
    return {variable: components_window[variable].sel(zone=zone).values
            for variable in labels if variable in components_window}


def compute_evening_peak_attribution(
    residual_load_by_model: dict,
    components_by_model: dict,
    period_windows: dict,
    zones,
    season_map: dict,
    *,
    hours=EVENING_PEAK_HOURS,
    component_labels: dict | None = None,
) -> pd.DataFrame:
    """Mean of every residual-load component over a window of hours of the day.

    Parameters
    ----------
    residual_load_by_model : dict
        ``{model: xarray.Dataset}`` with ``residual_load`` (GWh per step) over ``zone``.
    components_by_model : dict
        ``{model: xarray.Dataset}`` with the component variables of ``component_labels``.
    period_windows : dict
        Fixed or per-chain period windows.
    zones : iterable
        Zone coordinate values.
    season_map : dict
        ``{month: season}``; must contain ``"DJF"`` and ``"JJA"`` to produce seasonal rows.
    hours : sequence of int, default EVENING_PEAK_HOURS
        Step-start hours (UTC) included in the mean; the figures pass
        :data:`DEFICIT_WINDOW_HOURS`.
    component_labels : dict, optional
        ``{variable: display name}``; defaults to :data:`PAPER_COMPONENT_LABELS`.

    Returns
    -------
    pandas.DataFrame
        One row per (model, period, zone, season), ``season`` in ``"ALL"``, ``"DJF"``,
        ``"JJA"``, with ``RL`` and one column per component display name (GWh per step, unsigned).
        A season with no step in the window is omitted.
    """
    labels = component_labels or PAPER_COMPONENT_LABELS
    records = []
    for model in residual_load_by_model:
        residual_load_dataset = residual_load_by_model[model]
        components_dataset = components_by_model[model]
        for period, (start, end) in _available_windows(period_windows, model).items():
            residual_load_window = residual_load_dataset.sel(time=slice(start, end))
            components_window = components_dataset.sel(time=slice(start, end))
            hour = residual_load_window["time"].dt.hour.values
            step_seasons = np.array([season_map[int(month)]
                                     for month in residual_load_window["time"].dt.month.values])
            in_hours = np.isin(hour, list(hours))
            for zone in zones:
                zone_residual_load = residual_load_window["residual_load"].sel(zone=zone).values
                zone_components = _zone_components(components_window, zone, labels)
                for season_mask, season_label in [(np.ones_like(in_hours, bool), "ALL"),
                                                  (step_seasons == "DJF", "DJF"),
                                                  (step_seasons == "JJA", "JJA")]:
                    mask = in_hours & season_mask
                    if mask.sum() == 0:
                        continue
                    row = dict(model=model, period=period, zone=str(zone), season=season_label,
                               RL=float(zone_residual_load[mask].mean()))
                    for variable, values in zone_components.items():
                        row[labels[variable]] = float(values[mask].mean())
                    records.append(row)
    return pd.DataFrame(records)


def compute_multi_day_table(
    residual_load_by_model: dict,
    components_by_model: dict,
    threshold_by_model: dict,
    period_windows: dict,
    zones,
    season_map: dict,
    season_order,
    *,
    method: str = "spa",
    min_duration_h: float = 24.0,
    component_labels: dict | None = None,
    attribution_seasons=("DJF", "JJA"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Multi-day event frequency and the within-season component means that attribute them.

    Parameters
    ----------
    residual_load_by_model : dict
        ``{model: xarray.Dataset}`` with ``residual_load`` (GWh per step) over ``zone``.
    components_by_model : dict
        ``{model: xarray.Dataset}`` with the component variables of ``component_labels``.
    threshold_by_model : dict
        ``{model: xarray.DataArray}`` annual thresholds over ``zone``.
    period_windows : dict
        Fixed or per-chain period windows.
    zones : iterable
        Zone coordinate values.
    season_map : dict
        ``{month: season}``.
    season_order : sequence of str
        Seasons of the per-season table.
    method : {"spa", "run_length"}, default "spa"
        Event-detection algorithm.
    min_duration_h : float, default 24.0
        Events strictly longer than this are multi-day.
    component_labels : dict, optional
        ``{variable: display name}``; defaults to :data:`PAPER_COMPONENT_LABELS`.
    attribution_seasons : sequence of str, default ("DJF", "JJA")
        Seasons for which the component means are reported.

    Returns
    -------
    multi_day_events : pandas.DataFrame
        One row per (model, period, zone) with ``md_events_yr``, ``md_meandur`` (h) and, per
        attribution season, ``{season}_RL`` and ``{season}_{driver}`` (season means over all
        steps), ``{season}_EV_steps``, ``{season}_EV_RL`` and ``{season}_EV_{driver}`` (means over
        the steps inside a multi-day event; NaN when the season has none).
    multi_day_by_season : pandas.DataFrame
        One row per (model, period, zone, season) with ``md_events_yr`` and ``md_hours_yr``,
        each event counted in the season of its first step.

    Notes
    -----
    The attribution is within season because warming shifts multi-day events between seasons; a
    season-pooled decomposition would mix that shift with the physical change.
    """
    labels = component_labels or PAPER_COMPONENT_LABELS
    event_rows, season_rows = [], []
    for model in residual_load_by_model:
        residual_load_dataset = residual_load_by_model[model]
        components_dataset = components_by_model[model]
        for period, (start, end) in _available_windows(period_windows, model).items():
            residual_load_window = residual_load_dataset.sel(time=slice(start, end))
            components_window = components_dataset.sel(time=slice(start, end))
            n_years = len(np.unique(residual_load_window["time"].dt.year.values))
            months = residual_load_window["time"].dt.month.values
            step_seasons = np.array([season_map[int(month)] for month in months])

            for zone in zones:
                threshold = float(threshold_by_model[model].sel(zone=zone).values)
                zone_residual_load = residual_load_window["residual_load"].sel(zone=zone).values
                events = [event for event in detect_events(zone_residual_load, threshold, method=method)
                          if event["duration_h"] > min_duration_h]
                durations = [event["duration_h"] for event in events]
                event_seasons = [season_map[int(months[event["indices"][0]])] for event in events]
                zone_components = _zone_components(components_window, zone, labels)

                in_event = np.zeros(len(zone_residual_load), dtype=bool)
                for event in events:
                    in_event[event["indices"]] = True

                row = dict(model=model, period=period, zone=str(zone),
                           md_events_yr=len(durations) / n_years,
                           md_meandur=float(np.mean(durations)) if durations else np.nan)
                for season in attribution_seasons:
                    mask = step_seasons == season
                    row[f"{season}_RL"] = float(zone_residual_load[mask].mean())
                    for variable, values in zone_components.items():
                        row[f"{season}_{labels[variable]}"] = float(values[mask].mean())

                    # NaN, not zero, where the season holds no event step.
                    event_mask = mask & in_event
                    present = bool(event_mask.any())
                    row[f"{season}_EV_steps"] = int(event_mask.sum())
                    row[f"{season}_EV_RL"] = (float(zone_residual_load[event_mask].mean())
                                              if present else np.nan)
                    for variable, values in zone_components.items():
                        row[f"{season}_EV_{labels[variable]}"] = (
                            float(values[event_mask].mean()) if present else np.nan)
                event_rows.append(row)

                for season in season_order:
                    season_rows.append(dict(
                        model=model, period=period, zone=str(zone), season=season,
                        md_events_yr=float(sum(1 for event_season in event_seasons
                                               if event_season == season)) / n_years,
                        md_hours_yr=float(np.sum([duration for duration, event_season
                                                  in zip(durations, event_seasons)
                                                  if event_season == season])) / n_years))
    return pd.DataFrame(event_rows), pd.DataFrame(season_rows)


def attribution_change_by_model(
    components: pd.DataFrame,
    zone: str,
    season: str,
    reference_period: str,
    warmest_period: str,
    drivers,
    *,
    column_template: str = "{driver}",
    net_column: str = "RL",
    demand_driver: str = "Demand",
) -> pd.DataFrame:
    """Per-chain driver contributions to the change in residual load between two periods.

    Parameters
    ----------
    components : pandas.DataFrame
        Output of :func:`compute_evening_peak_attribution` (has a ``season`` column, driver
        columns named directly) or the first table of :func:`compute_multi_day_table` (no
        ``season`` column; pass ``column_template="{season}_{driver}"`` and
        ``net_column="{season}_RL"``).
    zone, season : str
        Cell to select.
    reference_period, warmest_period : str
        Periods differenced (warmest minus reference).
    drivers : sequence of str
        Driver display names.
    column_template : str, default "{driver}"
        Column name of a driver, formatted with ``driver`` and ``season``.
    net_column : str, default "RL"
        Column name of the residual load, formatted with ``season``.
    demand_driver : str, default "Demand"
        The driver entering with ``+``; every other driver is negated.

    Returns
    -------
    pandas.DataFrame
        Indexed by the models present in both periods, one column per driver plus ``net``
        (the change in residual load).
    """
    rows = components[components.zone == zone]
    if "season" in rows.columns:
        rows = rows[rows.season == season]

    columns = {driver: column_template.format(driver=driver, season=season) for driver in drivers}
    net = net_column.format(season=season)

    reference = rows[rows.period == reference_period].set_index("model")
    warmest = rows[rows.period == warmest_period].set_index("model")
    common = reference.index.intersection(warmest.index)

    wanted = [net] + list(columns.values())
    delta = warmest.loc[common, wanted] - reference.loc[common, wanted]

    out = pd.DataFrame(index=common)
    for driver in drivers:
        sign = 1.0 if driver == demand_driver else -1.0
        out[driver] = sign * delta[columns[driver]]
    out["net"] = delta[net]
    return out


def multi_day_event_anomaly_by_model(
    multi_day_events: pd.DataFrame,
    zone: str,
    season: str,
    period: str,
    drivers,
    *,
    scale: float = 1.0,
    demand_driver: str = "Demand",
) -> pd.DataFrame:
    """Per-chain composition of multi-day events relative to their season's mean.

    Parameters
    ----------
    multi_day_events : pandas.DataFrame
        First table of :func:`compute_multi_day_table`.
    zone, season, period : str
        Cell to select.
    drivers : sequence of str
        Driver display names.
    scale : float, default 1.0
        Divisor applied to every term, e.g. the zone threshold; additivity is preserved.
    demand_driver : str, default "Demand"
        The driver entering with ``+``; every other driver is negated.

    Returns
    -------
    pandas.DataFrame
        Indexed by model, one column per driver plus ``net``: the mean over multi-day event steps
        minus the season mean.  NaN where the season holds no multi-day event.
    """
    rows = multi_day_events[(multi_day_events.zone == zone)
                            & (multi_day_events.period == period)].set_index("model")

    out = pd.DataFrame(index=rows.index)
    for driver in drivers:
        sign = 1.0 if driver == demand_driver else -1.0
        out[driver] = sign * (rows[f"{season}_EV_{driver}"] - rows[f"{season}_{driver}"]) / scale
    out["net"] = (rows[f"{season}_EV_RL"] - rows[f"{season}_RL"]) / scale
    return out


def event_anomaly_vs_hourly_climatology(
    residual_load_by_model: dict,
    components_by_model: dict,
    threshold_by_model: dict,
    period_windows: dict,
    zones,
    season_map: dict,
    *,
    method: str = "spa",
    max_duration_h: float | None = None,
    component_labels: dict | None = None,
) -> pd.DataFrame:
    """Event anomaly per driver against the hour-of-day-matched seasonal climatology.

    Parameters
    ----------
    residual_load_by_model : dict
        ``{model: xarray.Dataset}`` with ``residual_load`` (GWh per step) over ``zone``.
    components_by_model : dict
        ``{model: xarray.Dataset}`` with the component variables of ``component_labels``.
    threshold_by_model : dict
        ``{model: xarray.DataArray}`` annual thresholds over ``zone``.
    period_windows : dict
        Fixed or per-chain period windows.
    zones : iterable
        Zone coordinate values.
    season_map : dict
        ``{month: season}``.
    method : {"spa", "run_length"}, default "spa"
        Event-detection algorithm.
    max_duration_h : float, optional
        Keep only events no longer than this (e.g. 24 for sub-daily events); ``None`` keeps all.
    component_labels : dict, optional
        ``{variable: display name}``; defaults to :data:`PAPER_COMPONENT_LABELS`.

    Returns
    -------
    pandas.DataFrame
        One row per (model, period, zone, season) with ``n_steps`` (event steps in the season),
        ``evening_share`` (fraction of those steps in :data:`EVENING_PEAK_HOURS`) and, for the
        residual load (``net``) and each driver:

        ``{driver}``
            weather anomaly: event mean minus the hour-matched climatology;
        ``TOT_{driver}``
            total anomaly: event mean minus the season mean;
        ``DIU_{driver}``
            diurnal term: hour-matched climatology, averaged over the event steps, minus the
            season mean.

        All NaN when the season holds no event step.

    Notes
    -----
    For every event step the reference is the season mean at that step's hour of day, so the
    diurnal cycle is removed however the event's steps are spread over the day, and
    ``total = diurnal + weather`` exactly.  Demand and ``net`` enter with ``+``; generation and
    storage are negated, so a positive value deepens the deficit.
    """
    labels = component_labels or PAPER_COMPONENT_LABELS
    rows = []
    for model in residual_load_by_model:
        residual_load_dataset = residual_load_by_model[model]
        components_dataset = components_by_model[model]
        for period, (start, end) in _available_windows(period_windows, model).items():
            residual_load_window = residual_load_dataset.sel(time=slice(start, end))
            components_window = components_dataset.sel(time=slice(start, end))
            months = residual_load_window["time"].dt.month.values
            hours = residual_load_window["time"].dt.hour.values
            step_seasons = np.array([season_map[int(month)] for month in months])

            for zone in zones:
                threshold = float(threshold_by_model[model].sel(zone=zone).values)
                zone_residual_load = residual_load_window["residual_load"].sel(zone=zone).values
                events = [event for event in detect_events(zone_residual_load, threshold, method=method)
                          if max_duration_h is None or event["duration_h"] <= max_duration_h]
                in_event = np.zeros(len(zone_residual_load), dtype=bool)
                for event in events:
                    in_event[event["indices"]] = True
                series = {"net": zone_residual_load}
                series.update({labels[variable]: components_window[variable].sel(zone=zone).values
                               for variable in labels if variable in components_window})

                for season in sorted(set(step_seasons)):
                    in_season = step_seasons == season
                    mask = in_event & in_season
                    row = dict(model=model, period=period, zone=str(zone), season=str(season),
                               n_steps=int(mask.sum()),
                               evening_share=(float(np.isin(hours[mask], EVENING_PEAK_HOURS).mean())
                                              if mask.any() else np.nan))
                    for name, values in series.items():
                        sign = 1.0 if name in (labels.get("total_demand"), "net") else -1.0
                        if not mask.any():
                            row[name] = row[f"TOT_{name}"] = row[f"DIU_{name}"] = np.nan
                            continue
                        season_mean = float(values[in_season].mean())
                        # One climatology value per hour, then indexed per event step.
                        climatology = {hour: float(values[in_season & (hours == hour)].mean())
                                       for hour in np.unique(hours)}
                        matched = np.array([climatology[hour] for hour in hours[mask]])
                        row[name] = sign * float((values[mask] - matched).mean())
                        row[f"TOT_{name}"] = sign * float(values[mask].mean() - season_mean)
                        row[f"DIU_{name}"] = sign * float(matched.mean() - season_mean)
                    rows.append(row)
    return pd.DataFrame(rows)


def compute_deficit_budget(
    residual_load_by_model: dict,
    threshold_by_model: dict,
    period_windows: dict,
    zones,
    *,
    method: str = "spa",
    sub_daily_max_h: float = 24.0,
) -> tuple[pd.DataFrame, dict]:
    """Sub-daily against multi-day share of the annual deficit, and stressed-day profiles.

    Parameters
    ----------
    residual_load_by_model : dict
        ``{model: xarray.Dataset}`` with ``residual_load`` (GWh per step) over ``zone``.
    threshold_by_model : dict
        ``{model: xarray.DataArray}`` annual thresholds over ``zone``.
    period_windows : dict
        Fixed or per-chain period windows.
    zones : iterable
        Zone coordinate values.
    method : {"spa", "run_length"}, default "spa"
        Event-detection algorithm.
    sub_daily_max_h : float, default 24.0
        Longest duration (h) counted as sub-daily.

    Returns
    -------
    deficit_budget : pandas.DataFrame
        One row per (model, period, zone) with ``sub_events_yr``, ``multi_events_yr``,
        ``sub_hours_yr`` and ``multi_hours_yr``.
    diurnal_profiles : dict
        ``{period: [pandas.Series]}``, each Series indexed by hour and named ``"model|zone"``:
        mean residual load divided by the threshold over stressed days (calendar days with at
        least one step above the threshold).
    """
    rows = []
    diurnal: dict[str, list] = {period: [] for period in period_windows}
    for model in residual_load_by_model:
        residual_load_dataset = residual_load_by_model[model]
        for period, (start, end) in _available_windows(period_windows, model).items():
            residual_load_window = residual_load_dataset.sel(time=slice(start, end))
            n_years = len(np.unique(residual_load_window["time"].dt.year.values))
            hours = residual_load_window["time"].dt.hour.values
            days = residual_load_window["time"].dt.floor("D").values

            for zone in zones:
                threshold = float(threshold_by_model[model].sel(zone=zone).values)
                values = residual_load_window["residual_load"].sel(zone=zone).values
                exceeds = values > threshold
                events = detect_events(values, threshold, method=method)
                durations = (np.array([event["duration_h"] for event in events]) if events
                             else np.array([]))
                is_sub_daily = (durations > 0) & (durations <= sub_daily_max_h)
                is_multi_day = durations > sub_daily_max_h

                frame = pd.DataFrame({"day": days, "hour": hours, "residual_load": values,
                                      "exceeds": exceeds})
                stressed = frame[frame.groupby("day")["exceeds"].transform("any")]
                if len(stressed):
                    diurnal[period].append(
                        (stressed.groupby("hour")["residual_load"].mean() / threshold)
                        .rename(f"{model}|{zone}"))

                rows.append(dict(
                    model=model, period=period, zone=str(zone),
                    sub_events_yr=int(is_sub_daily.sum()) / n_years,
                    multi_events_yr=int(is_multi_day.sum()) / n_years,
                    sub_hours_yr=float(durations[is_sub_daily].sum()) / n_years,
                    multi_hours_yr=float(durations[is_multi_day].sum()) / n_years))
    return pd.DataFrame(rows), diurnal


# ---------------------------------------------------------------------------
# Residual-load distribution tails
# ---------------------------------------------------------------------------


def _season_values_by_period(residual_load_by_model, period_windows, zone, months, period_names,
                             *, sort: bool = True) -> dict:
    """Return ``{period: [values per chain]}`` of finite residual load in ``months`` for one zone."""
    out = {}
    for period in period_names:
        per_model = []
        for model, dataset in residual_load_by_model.items():
            window = period_windows[period].get(model)
            if window is None:
                continue
            data = dataset["residual_load"].sel(
                zone=zone, time=slice(str(window[0]), str(window[1])))
            in_season = np.isin(data["time"].dt.month.values, list(months))
            values = data.values[in_season]
            values = values[np.isfinite(values)]
            per_model.append(np.sort(values) if sort else values)
        out[period] = per_model
    return out


def compute_survival_x_grid(
    residual_load_by_model: dict,
    period_windows: dict,
    zone: str,
    month_lists,
    period_names,
    *,
    threshold: float | None = None,
    n_points: int = 220,
) -> np.ndarray:
    """Residual-load grid shared by several seasons of one zone.

    Parameters
    ----------
    residual_load_by_model : dict
        ``{model: xarray.Dataset}`` with ``residual_load`` over ``zone``.
    period_windows : dict
        Per-chain period windows ``{period: {model: (start, end)}}``.
    zone : str
        Zone to evaluate.
    month_lists : sequence of sequence of int
        Months of each season pooled into the grid.
    period_names : sequence of str
        Periods pooled into the grid.
    threshold : float, optional
        Extend the grid to at least ``1.02 * threshold`` so the threshold is always in range.
    n_points : int, default 220
        Number of grid points.

    Returns
    -------
    numpy.ndarray
        Evenly spaced grid from the 40th to the 99.95th percentile of the pooled values.
    """
    pooled = []
    for months in month_lists:
        by_period = _season_values_by_period(residual_load_by_model, period_windows, zone, months,
                                             period_names, sort=False)
        pooled.extend(values for per_model in by_period.values() for values in per_model)
    pooled = np.concatenate(pooled)

    low = float(np.percentile(pooled, 40))
    high = float(np.percentile(pooled, 99.95))
    if threshold is not None:
        high = max(high, threshold * 1.02)
    return np.linspace(low, high, n_points)


def compute_exceedance_survival(
    residual_load_by_model: dict,
    period_windows: dict,
    zone: str,
    months,
    period_names,
    *,
    spread: str = "iqr",
    n_points: int = 220,
    grid: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """Survival curve ``P(RL > x)`` per chain, reduced to ensemble median and spread.

    Parameters
    ----------
    residual_load_by_model : dict
        ``{model: xarray.Dataset}`` with ``residual_load`` over ``zone``.
    period_windows : dict
        Per-chain period windows ``{period: {model: (start, end)}}``.
    zone : str
        Zone to evaluate.
    months : sequence of int
        Months of the season.
    period_names : sequence of str
        Periods to evaluate.
    spread : str, default "iqr"
        Inter-model spread convention of
        :func:`~cordex2energy_analysis.analysis.model_spread_bounds`.
    n_points : int, default 220
        Grid size when ``grid`` is not given.
    grid : numpy.ndarray, optional
        Evaluation grid, e.g. from :func:`compute_survival_x_grid`; by default the 40th to
        99.95th percentile of the pooled values.

    Returns
    -------
    grid : numpy.ndarray
        The evaluation grid.
    curves : dict
        ``{period: (median, low, high)}``, each an array over ``grid``.
    """
    sorted_values = _season_values_by_period(residual_load_by_model, period_windows, zone, months,
                                             period_names)

    if grid is None:
        pooled = np.concatenate([np.concatenate(per_model) for per_model in sorted_values.values()])
        grid = np.linspace(np.percentile(pooled, 40), np.percentile(pooled, 99.95), n_points)

    curves = {}
    for period in period_names:
        survival = np.vstack([1.0 - np.searchsorted(values, grid, side="right") / len(values)
                              for values in sorted_values[period]])
        curves[period] = (np.median(survival, axis=0),
                          *model_spread_bounds(survival, spread, axis=0))
    return grid, curves


def compute_exceedance_probability(
    residual_load_by_model: dict,
    period_windows: dict,
    zone: str,
    months,
    period: str,
    threshold: float,
) -> float:
    """Ensemble-median probability that the residual load exceeds a threshold.

    Parameters
    ----------
    residual_load_by_model : dict
        ``{model: xarray.Dataset}`` with ``residual_load`` over ``zone``.
    period_windows : dict
        Per-chain period windows ``{period: {model: (start, end)}}``.
    zone : str
        Zone to evaluate.
    months : sequence of int
        Months of the season.
    period : str
        Period to evaluate.
    threshold : float
        Threshold, GWh per step.

    Returns
    -------
    float
        Median over chains of the fraction of finite in-season steps above ``threshold``.
    """
    probabilities = []
    for model, dataset in residual_load_by_model.items():
        window = period_windows[period].get(model)
        if window is None:
            continue
        data = dataset["residual_load"].sel(zone=zone, time=slice(str(window[0]), str(window[1])))
        in_season = np.isin(data["time"].dt.month.values, list(months))
        values = data.values[in_season]
        values = values[np.isfinite(values)]
        probabilities.append(float(np.mean(values > threshold)))
    return float(np.median(probabilities))


# ---------------------------------------------------------------------------
# Country fields and operation curves
# ---------------------------------------------------------------------------


def compute_country_gwl_field(
    country_output_by_model: dict,
    period_windows: dict,
    period: str,
    expression,
    *,
    scale: float = 1e-3,
    months: "list[int] | None" = None,
) -> "xr.DataArray":
    """Per-chain mean annual (or seasonal) total of a country quantity over a window.

    Parameters
    ----------
    country_output_by_model : dict
        ``{model: xarray.Dataset}`` of country-level output.
    period_windows : dict
        Per-chain period windows ``{period: {model: (start, end)}}``.
    period : str
        Period to evaluate.
    expression : callable
        ``expression(dataset) -> DataArray(country, time)`` in GWh per time step.
    scale : float, default 1e-3
        Factor applied to the total (GWh to TWh).
    months : list of int, optional
        Restrict to these months; the result is then the mean seasonal total.

    Returns
    -------
    xarray.DataArray
        ``(model, country)``: mean over the window's years of the calendar-year sum.
    """
    per_model = []
    for model, dataset in country_output_by_model.items():
        window = period_windows[period].get(model)
        if window is None:
            continue
        data = expression(dataset).sel(time=slice(str(window[0]), str(window[1])))
        if months is not None:
            data = data.sel(time=data["time"].dt.month.isin(months))
        total = data.resample(time="YE").sum("time")
        per_model.append((total.mean("time") * scale).expand_dims(model=[model]))
    return xr.concat(per_model, dim="model")


def compute_c2e_gwl_field(
    c2e_by_variable: dict,
    period_windows: dict,
    period: str,
    variable: str,
    *,
    scale: float = 1e-3,
    months: "list[int] | None" = None,
) -> "xr.DataArray":
    """Per-chain mean annual (or seasonal) total of a c2e variable over a window.

    Parameters
    ----------
    c2e_by_variable : dict
        ``{variable: DataArray(model, country, time)}``.
    period_windows : dict
        Per-chain period windows ``{period: {model: (start, end)}}``.
    period : str
        Period to evaluate.
    variable : str
        Key of ``c2e_by_variable``.
    scale : float, default 1e-3
        Factor applied to the total.
    months : list of int, optional
        Restrict to these months; the result is then the mean seasonal total.

    Returns
    -------
    xarray.DataArray
        ``(model, country)`` for the chains with a window in ``period``.
    """
    per_model = []
    for model, window in period_windows[period].items():
        if window is None:
            continue
        data = c2e_by_variable[variable].sel(
            model=model, time=slice(str(window[0]), str(window[1])))
        if months is not None:
            data = data.sel(time=data["time"].dt.month.isin(months))
        per_model.append(
            (data.resample(time="YE").sum("time").mean("time") * scale).expand_dims(model=[model]))
    return xr.concat(per_model, dim="model")


def mask_structural_zeros(field: "xr.DataArray") -> "xr.DataArray":
    """Replace zeros with NaN so countries without a technology plot as missing.

    Parameters
    ----------
    field : xarray.DataArray
        Field to mask.

    Returns
    -------
    xarray.DataArray
        ``field`` with every exact zero set to NaN.
    """
    return field.where(field != 0)


def compute_annual_operation_curves(
    components_by_model: dict,
    period_windows: dict,
    period: str,
    variable: str,
    selection: dict,
    *,
    mode: str = "sum",
    smooth_days: int = 7,
) -> "np.ndarray":
    """Day-of-year operation curves over a window, one per (chain, year).

    Parameters
    ----------
    components_by_model : dict
        ``{model: xarray.Dataset}`` holding ``variable``.
    period_windows : dict
        Per-chain period windows ``{period: {model: (start, end)}}``.
    period : str
        Period to evaluate.
    variable : str
        Variable to aggregate.
    selection : dict
        Slice of the panel, e.g. ``{"country": "NO"}`` or ``{"zone": "Nordic"}``.
    mode : {"sum", "mean", "throughput"}, default "sum"
        Daily aggregation: ``"sum"`` daily total (e.g. reservoir dispatch, GWh/day); ``"mean"``
        daily mean (e.g. reservoir storage, GWh); ``"throughput"`` energy cycled per day,
        ``sum(|x|) / 2`` (for batteries, whose net daily dispatch is only the round-trip loss).
    smooth_days : int, default 7
        Centred rolling-mean window in days; no smoothing when 1 or less.

    Returns
    -------
    numpy.ndarray
        ``(n_curves, 365)``.  Years with fewer than 365 days in the window are skipped, and
        leap years are truncated to 365 days.
    """
    curves = []
    for model, dataset in components_by_model.items():
        window = period_windows[period].get(model)
        if window is None:
            continue
        series = dataset[variable].sel(**selection).sel(
            time=slice(str(window[0]), str(window[1])))
        if mode == "throughput":
            daily = np.abs(series).resample(time="1D").sum() / 2
        elif mode == "mean":
            daily = series.resample(time="1D").mean()
        else:
            daily = series.resample(time="1D").sum()

        for _, year_block in daily.groupby("time.year"):
            values = year_block.values
            if len(values) < 365:
                continue
            values = values[:365]
            if smooth_days > 1:
                values = (pd.Series(values)
                          .rolling(smooth_days, center=True, min_periods=1).mean().values)
            curves.append(values)
    return np.array(curves)
