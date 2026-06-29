"""Derived multi-model analyses and period/GWL orchestration.

Functions
---------
compile_and_compare_periods
    Assemble per-zone metrics from multiple climate periods into a DataFrame
    and compute percentage changes relative to the reference period.

resolve_gwl_period_definitions
    Build per-model GWL period windows from a warming-levels YAML.

compute_seasonal_exceedance_fractions
    Seasonal distribution of annual EVT exceedances per model/period/zone.

compute_zone_pair_cooccurrence
    Zone-pair co-occurrence of EVT exceedances with a dilation window.

extract_evt_duration_severity_records
    Individual event records (duration, severity) for power-law analysis.

compute_unserved_energy_fraction
    Annual share of unserved energy per model, period, zone, and year.

compute_flexibility_attenuation_metrics
    Battery and reservoir flexibility deployed during EVT events.

extract_event_diurnal_profiles
    Distribution of event timesteps across hours of the day, with PV context.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import binary_dilation

from .extremes_events import (
    DEFAULT_DURATION_BINS,
    classify_event_duration,
    detect_events,
    detect_seasonal_events,
)


# ---------------------------------------------------------------------------
# Period comparison
# ---------------------------------------------------------------------------


def compile_and_compare_periods(*period_dicts, period_names=None):
    """Compile metrics from multiple eras into a structured DataFrame.

    Parameters
    ----------
    *period_dicts :
        One dict per period, each keyed by zone and containing metric dicts.
    period_names : list[str], optional
        Internal labels assigned to each period.  When omitted and exactly
        three dicts are supplied the function uses
        ``["1_Reference", "2_Mid-Century", "3_End-Century"]`` for backward
        compatibility.  Required for any other number of periods.
    """
    _legacy_names = ["1_Reference", "2_Mid-Century", "3_End-Century"]
    if period_names is None:
        if len(period_dicts) != 3:
            raise ValueError(
                "period_names is required when passing a number of periods other than 3"
            )
        period_names = _legacy_names

    records = []
    for period_name, period_data in zip(period_names, period_dicts):
        for zone, metrics in period_data.items():
            row = {"Period": period_name, "Zone": zone}
            row.update(metrics)
            records.append(row)

    df = pd.DataFrame(records)
    df_pivot = df.set_index(["Zone", "Period"]).sort_index()

    summary_metrics = [
        "Frequency",
        "Avg_Duration_Hours",
        "Max_Duration_Hours",
        "Total_Annual_Energy_Severity_GWh",
        "Avg_Energy_Severity_GWh",
        "Max_Single_Event_Severity_GWh",
        "Avg_PCIT_During_Events_GW",
        "Max_PCIT_During_Events_GW",
        "Std_Annual_Energy_Severity_GWh",
        "P90_Annual_Energy_Severity_GWh",
        "CV_Annual_Energy_Severity",
        "Demand_Driven_Share_%",
        "PV_Driven_Share_%",
        "Wind_Onshore_Driven_Share_%",
        "Wind_Offshore_Driven_Share_%",
        "Hydro_RoR_Driven_Share_%",
        "Reservoir_Hydro_Driven_Share_%",
        "Battery_Driven_Share_%",
        "Avg_Demand_Anomaly_GWh",
        "Avg_PV_Anomaly_GWh",
        "Avg_Wind_Onshore_Anomaly_GWh",
        "Avg_Wind_Offshore_Anomaly_GWh",
        "Avg_Hydro_RoR_Anomaly_GWh",
        "Avg_Reservoir_Anomaly_GWh",
        "Avg_Battery_Anomaly_GWh",
    ]

    ref_label = period_names[0]
    _use_legacy_suffixes = list(period_names) == _legacy_names

    deltas = {}
    for zone in df["Zone"].unique():
        deltas[zone] = {}
        for metric in summary_metrics:
            if metric not in df_pivot.columns:
                continue
            ref_val = df_pivot.loc[(zone, ref_label), metric]
            for j, other_label in enumerate(period_names[1:], start=1):
                other_val = df_pivot.loc[(zone, other_label), metric]
                pct = ((other_val - ref_val) / ref_val * 100) if ref_val > 0 else 0
                if _use_legacy_suffixes:
                    suffix = "Mid" if j == 1 else "End"
                else:
                    suffix = other_label.split("_", 1)[-1] if "_" in other_label else other_label
                deltas[zone][f"{metric}_{suffix}_Change_%"] = round(pct, 1)

    df_deltas = pd.DataFrame.from_dict(deltas, orient="index")
    df_deltas.index.name = "Zone"

    return df_pivot, df_deltas


# ---------------------------------------------------------------------------
# GWL period resolution
# ---------------------------------------------------------------------------

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
    """Build per-model GWL period definitions from a warming-levels YAML.

    Parameters
    ----------
    warming_levels :
        GWL values to include (e.g. ``[1.5, 2.0, 3.0]``).
    models :
        Model chain IDs.
    chain_meta_path :
        Path to ``models.json`` listing GCM/ensemble info per chain.
    gwl_yaml_path :
        Path to the warming-levels YAML (Mathias Hauser format).
    climate_scenario :
        RCP scenario string (default ``"rcp85"``).

    Returns
    -------
    period_defs : dict
        ``{"+X.X°C": {chain_id: (start_str, end_str)}}``
    period_label : dict
        ``{"+X.X°C": "N_+X.X°C"}``
    period_colors : dict
        ``{"+X.X°C": hex_color}``
    """
    import json as _json
    from pathlib import Path as _Path
    from cordex2energy_analysis.analysis import load_warming_level_periods

    chain_meta_path = _Path(chain_meta_path)
    gwl_yaml_path = _Path(gwl_yaml_path)

    _models_config = _json.load(open(chain_meta_path))
    _chain_meta = {m["id"]: m for m in _models_config["model_chains"]}

    _gcm_ens_map: dict[str, str] = {}
    for chain_id in models:
        meta = _chain_meta.get(chain_id, {})
        cmip5 = _CORDEX_GCM_TO_CMIP5.get(meta.get("gcm", ""))
        if cmip5:
            _gcm_ens_map[cmip5] = meta.get("ensemble", "r1i1p1")

    gwl_data = load_warming_level_periods(
        yml_path=gwl_yaml_path,
        warming_levels=warming_levels,
        gcms=_gcm_ens_map,
    )

    period_defs: dict = {}
    period_label: dict = {}
    period_colors: dict = {}

    for i, wl in enumerate(warming_levels):
        label = f"+{wl:.1f}°C"
        windows: dict[str, tuple[str, str]] = {}
        for chain_id in models:
            meta = _chain_meta.get(chain_id, {})
            cmip5 = _CORDEX_GCM_TO_CMIP5.get(meta.get("gcm", ""))
            if cmip5 is None:
                continue
            r = (
                gwl_data
                .get((cmip5, meta.get("ensemble", "r1i1p1")), {})
                .get(climate_scenario, {})
                .get(wl)
            )
            if r:
                windows[chain_id] = (str(r["start_year"]), str(r["end_year"]))
        period_defs[label] = windows
        period_label[label] = f"{i + 1}_{label}"
        period_colors[label] = _GWL_COLORS[i % len(_GWL_COLORS)]

    return period_defs, period_label, period_colors


# ---------------------------------------------------------------------------
# Seasonal exceedance fractions
# ---------------------------------------------------------------------------


def compute_seasonal_exceedance_fractions(
    cache_rl: dict,
    evt_thresholds: dict,
    period_defs: dict,
    models,
    season_map: dict[int, str],
    season_order: list[str],
    zone_dim: str = "zone",
) -> pd.DataFrame:
    """Compute the seasonal distribution of annual EVT exceedances.

    For each (model, period, zone), counts what fraction of timesteps that
    exceed the annual common threshold fall in each meteorological season.

    Parameters
    ----------
    cache_rl :
        ``{model: ds_rl}`` — post-battery residual load datasets.
    evt_thresholds :
        ``{model: da_thresholds}`` — annual EVT threshold DataArrays.
    period_defs :
        Fixed or per-model period definitions.
    models :
        Iterable of model chain IDs.
    season_map :
        ``{month_int: season_str}`` mapping.
    season_order :
        Season names in display order.
    zone_dim :
        Name of the zone dimension (default ``"zone"``).

    Returns
    -------
    pd.DataFrame
        Columns: model, zone, period, DJF, MAM, JJA, SON (fractions in %).
    """
    from cordex2energy_analysis.plotting_extremes import resolve_model_periods

    records: list[dict] = []
    for model in models:
        ds_rl_m = cache_rl[model]
        da_evt_m = evt_thresholds[model]
        resolved = {
            k: v for k, v in resolve_model_periods(period_defs, model).items()
            if v is not None
        }
        for period_name, (t0, t1) in resolved.items():
            ds_rl_p = ds_rl_m.sel(time=slice(t0, t1))
            times_p = pd.DatetimeIndex(ds_rl_p["time"].values)
            zones = ds_rl_p[zone_dim].values
            for zone in zones:
                rl = ds_rl_p["residual_load"].sel({zone_dim: zone}).values
                thr = float(da_evt_m.sel({zone_dim: zone}).values)
                exc_months = times_p[rl > thr].month
                seasons = np.array([season_map[int(m)] for m in exc_months])
                total = len(seasons)
                row: dict = {"model": model, "zone": str(zone), "period": period_name}
                for s in season_order:
                    row[s] = float((seasons == s).sum()) / total * 100 if total > 0 else 0.0
                records.append(row)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Zone-pair co-occurrence
# ---------------------------------------------------------------------------


def _cooccurrence_on_array(
    exc_masks: dict,
    zones,
    dilation_struct,
) -> dict:
    """Compute dilated masks and return {zone: dilated_bool_array}."""
    return {
        str(z): binary_dilation(exc_masks[str(z)], structure=dilation_struct)
        for z in zones
    }


def compute_zone_pair_cooccurrence(
    cache_rl: dict,
    evt_thresholds: dict,
    period_defs: dict,
    models,
    zones,
    window_steps: int = 24,
    season_map: "dict[int, str] | None" = None,
    zone_dim: str = "zone",
) -> pd.DataFrame:
    """Compute zone-pair co-occurrence of EVT exceedances with a dilation window.

    For each (model, period, zone_a, zone_b): the fraction of Zone A exceedance
    timesteps that co-occur with a Zone B exceedance within ±``window_steps``
    time steps (default ±72 h at 3 h resolution).

    Parameters
    ----------
    cache_rl :
        ``{model: ds_rl}`` — post-battery residual load datasets.
    evt_thresholds :
        Annual ``{model: xr.DataArray}`` or seasonal
        ``{model: {zone: {season: float}}}`` thresholds (auto-detected when
        ``season_map`` is provided).
    period_defs :
        Fixed or per-model period definitions.
    models :
        Iterable of model chain IDs.
    zones :
        Zone coordinate values.
    window_steps :
        Half-width of the dilation kernel in time steps (default 24 = ±72 h).
    season_map :
        Optional ``{month_int: season_name}`` mapping.  When provided the
        function computes co-occurrence separately for each season.
    zone_dim :
        Name of the zone dimension (default ``"zone"``).

    Returns
    -------
    pd.DataFrame
        Columns: model, period, zone_a, zone_b, co_occurrence_pct[, season].
    """
    from cordex2energy_analysis.plotting_extremes import resolve_model_periods

    dilation_struct = np.ones(2 * window_steps + 1, dtype=bool)
    records: list[dict] = []

    if season_map is not None:
        seen: dict = {}
        for s in season_map.values():
            seen[s] = None
        season_names = list(seen.keys())
    else:
        season_names = [None]  # sentinel: single "full period" pass

    for model in models:
        ds_rl_m = cache_rl[model]
        thr_m = evt_thresholds[model]
        seasonal_thr = isinstance(thr_m, dict)
        resolved = {
            k: v for k, v in resolve_model_periods(period_defs, model).items()
            if v is not None
        }
        for period_name, (t0, t1) in resolved.items():
            ds_rl_p = ds_rl_m.sel(time=slice(t0, t1))
            time_months = ds_rl_p["time"].dt.month.values

            for season in season_names:
                if season is None:
                    s_mask = np.ones(len(time_months), dtype=bool)
                else:
                    target_months = {m for m, s in season_map.items() if s == season}
                    s_mask = np.isin(time_months, list(target_months))

                exc_masks: dict = {}
                for zone in zones:
                    rl_full = ds_rl_p["residual_load"].sel({zone_dim: zone}).values
                    if seasonal_thr and season is not None:
                        thr = float(thr_m[str(zone)][season])
                    else:
                        thr = float(thr_m.sel({zone_dim: zone}).values)
                    exc = (rl_full > thr) & s_mask
                    exc_masks[str(zone)] = exc

                dilated_masks = _cooccurrence_on_array(exc_masks, zones, dilation_struct)

                for za in zones:
                    za_key = str(za)
                    n_a = int((exc_masks[za_key] & s_mask).sum())
                    for zb in zones:
                        zb_key = str(zb)
                        frac = (
                            float((exc_masks[za_key] & dilated_masks[zb_key]).sum()) / n_a * 100
                            if n_a > 0 else np.nan
                        )
                        row = {
                            "model": model, "period": period_name,
                            "zone_a": za_key, "zone_b": zb_key,
                            "co_occurrence_pct": frac,
                        }
                        if season is not None:
                            row["season"] = season
                        records.append(row)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Duration-severity event extraction
# ---------------------------------------------------------------------------


def extract_evt_duration_severity_records(
    cache_rl: dict,
    evt_thresholds: dict,
    period_defs: dict,
    models,
    zones,
    zone_dim: str = "zone",
    method: str = "run_length",
) -> pd.DataFrame:
    """Extract individual EVT event records (duration, severity) for power-law analysis.

    Parameters
    ----------
    cache_rl :
        ``{model: ds_rl}`` — post-battery residual load datasets.
    evt_thresholds :
        ``{model: da_thresholds}`` — annual EVT threshold DataArrays.
    period_defs :
        Fixed or per-model period definitions.
    models :
        Iterable of model chain IDs.
    zones :
        Zone coordinate values.
    zone_dim :
        Name of the zone dimension (default ``"zone"``).

    Returns
    -------
    pd.DataFrame
        Columns: model, period, zone, duration_h, severity_gwh.
    """
    from cordex2energy_analysis.plotting_extremes import resolve_model_periods

    records: list[dict] = []
    for model in models:
        ds_rl_m = cache_rl[model]
        da_evt_m = evt_thresholds[model]
        resolved = {
            k: v for k, v in resolve_model_periods(period_defs, model).items()
            if v is not None
        }
        for period_name, (t0, t1) in resolved.items():
            ds_rl_p = ds_rl_m.sel(time=slice(t0, t1))
            for zone in zones:
                rl = ds_rl_p["residual_load"].sel({zone_dim: zone}).values
                thr = float(da_evt_m.sel({zone_dim: zone}).values)
                for ev in detect_events(rl, thr, method=method):
                    records.append({
                        "model": model, "period": period_name, "zone": str(zone),
                        "duration_h": int(ev["duration_h"]),
                        "severity_gwh": ev["severity_gwh"],
                    })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Unserved energy fraction
# ---------------------------------------------------------------------------


def compute_unserved_energy_fraction(
    cache_rl: dict,
    cache_comp: dict,
    period_defs: dict,
    models,
    zones,
    season_map: "dict[int, str] | None" = None,
    zone_dim: str = "zone",
    demand_var: str = "total_demand",
) -> pd.DataFrame:
    """Annual share of unserved energy (USE) per model, period, zone, and year.

    USE is defined as the fraction of total electricity demand not covered by
    generation in the post-battery residual-load series::

        USE_year[, season] = Σ_t max(RL_t, 0) / Σ_t demand_t

    Parameters
    ----------
    cache_rl :
        ``{model: ds_rl}`` — post-battery residual load datasets.
    cache_comp :
        ``{model: ds_comp}`` — generation/demand component datasets.
    period_defs :
        Fixed or per-model GWL/calendar period definitions.
    models :
        Iterable of model chain IDs.
    zones :
        Zone coordinate values.
    season_map :
        Optional ``{month_int: season_name}`` mapping.  When provided USE is
        computed per season; the returned DataFrame gains a ``season`` column.
    zone_dim :
        Name of the zone dimension (default ``"zone"``).
    demand_var :
        Name of the demand variable in *cache_comp* (default ``"total_demand"``).

    Returns
    -------
    pd.DataFrame
        Columns: ``model``, ``period``, ``zone``, ``year``[, ``season``], ``use_fraction``.
    """
    from cordex2energy_analysis.plotting_extremes import resolve_model_periods

    if season_map is not None:
        seen: dict = {}
        for s in season_map.values():
            seen[s] = None
        season_names: list = list(seen.keys())
    else:
        season_names = [None]

    records: list[dict] = []

    for model in models:
        ds_rl_m   = cache_rl[model]
        ds_comp_m = cache_comp[model]
        resolved  = {
            k: v for k, v in resolve_model_periods(period_defs, model).items()
            if v is not None
        }
        for period_name, (t0, t1) in resolved.items():
            ds_rl_p    = ds_rl_m.sel(time=slice(t0, t1))
            ds_comp_p  = ds_comp_m.sel(time=slice(t0, t1))
            time_months = ds_rl_p["time"].dt.month.values
            time_years  = ds_rl_p["time"].dt.year.values
            years       = np.unique(time_years)

            for zone in zones:
                rl     = ds_rl_p["residual_load"].sel({zone_dim: zone}).values
                demand = ds_comp_p[demand_var].sel({zone_dim: zone}).values

                for season in season_names:
                    if season is None:
                        s_mask = np.ones(len(time_months), dtype=bool)
                    else:
                        target_months = {m for m, s in season_map.items() if s == season}
                        s_mask = np.isin(time_months, list(target_months))

                    for year in years:
                        y_s_mask = s_mask & (time_years == year)
                        if not y_s_mask.any():
                            continue
                        pos_rl_sum = float(np.maximum(rl[y_s_mask], 0.0).sum())
                        demand_sum = float(demand[y_s_mask].sum())
                        use = pos_rl_sum / demand_sum if demand_sum > 0 else np.nan
                        row = {
                            "model":        model,
                            "period":       period_name,
                            "zone":         str(zone),
                            "year":         int(year),
                            "use_fraction": use,
                        }
                        if season is not None:
                            row["season"] = season
                        records.append(row)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Flexibility attenuation metrics
# ---------------------------------------------------------------------------


def compute_flexibility_attenuation_metrics(
    cache_rl: dict,
    cache_comp: dict,
    evt_thresholds: dict,
    period_defs: dict,
    models,
    zones,
    season_map: "dict[int, str] | None" = None,
    zone_dim: str = "zone",
    method: str = "run_length",
) -> pd.DataFrame:
    """Quantify battery and reservoir flexibility deployed during EVT events.

    For each detected extreme event (connected block above the EVT threshold in
    the post-flexibility residual load) the function computes:

    * **deployed GWh** — total flexibility energy dispatched during the event.
    * **counterfactual severity** — severity without that flexibility source.
    * **attenuation %** — ``(counterfactual − actual) / counterfactual × 100``.

    Parameters
    ----------
    cache_rl :
        ``{model: ds_rl}`` — post-dispatch residual load datasets.
    cache_comp :
        ``{model: ds_comp}`` — component datasets.  Battery variables
        (``battery_dispatch_gwh``, ``prosumer_dispatch_gwh``,
        ``utility_dispatch_gwh``) and ``reservoir_dispatch_gwh`` are read
        from here.  Missing variables produce NaN metrics.
    evt_thresholds :
        Annual ``{model: xr.DataArray}`` or seasonal
        ``{model: {zone: {season: float}}}`` thresholds (auto-detected).
    period_defs :
        Period definitions.
    models :
        Iterable of model chain IDs.
    zones :
        Zone coordinate values.
    season_map :
        Optional ``{month_int: season_name}`` mapping.  When provided metrics
        are computed per season; non-season timesteps are masked out.
    zone_dim :
        Name of the zone dimension (default ``"zone"``).

    Returns
    -------
    pd.DataFrame
        One row per (model, period, zone[, season]).  Columns include
        ``n_events``, ``*_deployed_gwh_per_event``, ``*_attenuation_pct``, and
        ``combined_attenuation_pct``.
    """
    from cordex2energy_analysis.plotting_extremes import resolve_model_periods

    if season_map is not None:
        seen: dict = {}
        for s in season_map.values():
            seen[s] = None
        season_names: list = list(seen.keys())
    else:
        season_names = [None]

    records: list[dict] = []

    for model in models:
        ds_rl_m   = cache_rl[model]
        ds_comp_m = cache_comp[model]
        thr_m     = evt_thresholds[model]
        seasonal_thr = isinstance(thr_m, dict)

        has_battery   = "battery_dispatch_gwh"   in ds_comp_m.data_vars
        has_prosumer  = "prosumer_dispatch_gwh"  in ds_comp_m.data_vars
        has_utility   = "utility_dispatch_gwh"   in ds_comp_m.data_vars
        has_reservoir = "reservoir_dispatch_gwh" in ds_comp_m.data_vars

        resolved = {
            k: v for k, v in resolve_model_periods(period_defs, model).items()
            if v is not None
        }
        for period_name, (t0, t1) in resolved.items():
            ds_rl_p     = ds_rl_m.sel(time=slice(t0, t1))
            ds_comp_p   = ds_comp_m.sel(time=slice(t0, t1))
            time_months = ds_rl_p["time"].dt.month.values
            times_p     = pd.DatetimeIndex(ds_rl_p["time"].values)

            for zone in zones:
                rl_full = ds_rl_p["residual_load"].sel({zone_dim: zone}).values

                def _get(var, has_flag):
                    return (
                        ds_comp_p[var].sel({zone_dim: zone}).values
                        if has_flag else np.zeros_like(rl_full)
                    )

                bat_disp_full  = _get("battery_dispatch_gwh",  has_battery)
                pro_disp_full  = _get("prosumer_dispatch_gwh", has_prosumer)
                util_disp_full = _get("utility_dispatch_gwh",  has_utility)
                res_disp_full  = _get("reservoir_dispatch_gwh", has_reservoir)

                for season in season_names:
                    if seasonal_thr and season is not None:
                        thr = float(thr_m[str(zone)][season])
                    else:
                        thr = float(thr_m.sel({zone_dim: zone}).values)

                    if season is None:
                        rl        = rl_full
                        bat_disp  = bat_disp_full
                        pro_disp  = pro_disp_full
                        util_disp = util_disp_full
                        res_disp  = res_disp_full
                        events = detect_events(rl, thr, method=method)
                    else:
                        target_months = sorted({m for m, s in season_map.items() if s == season})
                        s_mask = np.isin(time_months, target_months)
                        # Component series are always zeroed outside the season regardless of method.
                        bat_disp  = bat_disp_full.copy();  bat_disp[~s_mask]  = 0.0
                        pro_disp  = pro_disp_full.copy();  pro_disp[~s_mask]  = 0.0
                        util_disp = util_disp_full.copy(); util_disp[~s_mask] = 0.0
                        res_disp  = res_disp_full.copy();  res_disp[~s_mask]  = 0.0
                        if method == "spa":
                            rl = rl_full
                            events = detect_seasonal_events(
                                rl_full, times_p, thr, target_months, method="spa"
                            )
                        else:
                            rl = rl_full.copy()
                            rl[~s_mask] = thr - 1.0
                            events = detect_events(rl, thr, method=method)
                    n_ev = len(events)

                    base_row = {"model": model, "period": period_name, "zone": str(zone)}
                    if season is not None:
                        base_row["season"] = season

                    if n_ev == 0:
                        records.append({
                            **base_row,
                            "n_events": 0,
                            "battery_deployed_gwh_per_event":  np.nan,
                            "prosumer_deployed_gwh_per_event": np.nan,
                            "utility_deployed_gwh_per_event":  np.nan,
                            "reservoir_deployed_gwh_per_event": np.nan,
                            "battery_attenuation_pct":  np.nan,
                            "prosumer_attenuation_pct": np.nan,
                            "utility_attenuation_pct":  np.nan,
                            "reservoir_attenuation_pct": np.nan,
                            "combined_attenuation_pct": np.nan,
                        })
                        continue

                    bat_dep, pro_dep, util_dep, res_dep = [], [], [], []
                    bat_att, pro_att, util_att, res_att, comb_att = [], [], [], [], []

                    for ev in events:
                        ev_idx = ev["indices"]
                        sev_actual = float(np.maximum(rl[ev_idx] - thr, 0.0).sum()) * 3

                        def _sev(discharge):
                            # discharge must be the non-negative (discharge-only) portion.
                            # Using raw dispatch (which includes negative/charging values)
                            # would give a counterfactual severity below actual.
                            return float(np.maximum((rl[ev_idx] + discharge) - thr, 0.0).sum()) * 3

                        bat_dis  = np.maximum(bat_disp[ev_idx],  0.0)
                        pro_dis  = np.maximum(pro_disp[ev_idx],  0.0)
                        util_dis = np.maximum(util_disp[ev_idx], 0.0)
                        res_dis  = np.maximum(res_disp[ev_idx],  0.0)

                        sev_no_bat  = _sev(bat_dis)
                        sev_no_pro  = _sev(pro_dis)  if has_prosumer else np.nan
                        sev_no_util = _sev(util_dis) if has_utility  else np.nan
                        sev_no_res  = _sev(res_dis)
                        sev_no_both = _sev(bat_dis + res_dis)

                        def _atten(sev_no, flag):
                            if not flag:
                                return np.nan
                            return (sev_no - sev_actual) / sev_no * 100 if sev_no > 0 else 0.0

                        bat_dep.append(float(bat_dis.sum())  * 3)
                        pro_dep.append(float(pro_dis.sum())  * 3 if has_prosumer else np.nan)
                        util_dep.append(float(util_dis.sum()) * 3 if has_utility  else np.nan)
                        res_dep.append(float(res_dis.sum())  * 3)

                        bat_att.append(_atten(sev_no_bat,  has_battery))
                        pro_att.append(_atten(sev_no_pro,  has_prosumer))
                        util_att.append(_atten(sev_no_util, has_utility))
                        res_att.append(_atten(sev_no_res,  has_reservoir))
                        comb_att.append(
                            (sev_no_both - sev_actual) / sev_no_both * 100
                            if sev_no_both > 0 else 0.0
                        )

                    def _mean(lst):
                        arr = np.array(lst, dtype=float)
                        return float(np.nanmean(arr)) if arr.size else np.nan

                    records.append({
                        **base_row,
                        "n_events": n_ev,
                        "battery_deployed_gwh_per_event":  _mean(bat_dep),
                        "prosumer_deployed_gwh_per_event": _mean(pro_dep),
                        "utility_deployed_gwh_per_event":  _mean(util_dep),
                        "reservoir_deployed_gwh_per_event": _mean(res_dep),
                        "battery_attenuation_pct":  _mean(bat_att),
                        "prosumer_attenuation_pct": _mean(pro_att),
                        "utility_attenuation_pct":  _mean(util_att),
                        "reservoir_attenuation_pct": _mean(res_att),
                        "combined_attenuation_pct": _mean(comb_att),
                    })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Diurnal profile of extreme events
# ---------------------------------------------------------------------------


def extract_event_diurnal_profiles(
    cache_rl: dict,
    cache_comp: dict,
    season_thresholds: dict,
    period_defs: dict,
    models,
    zones,
    season_order: list[str],
    season_months: dict[str, list[int]],
    cache_clim: "dict | None" = None,
    zone_dim: str = "zone",
    method: str = "run_length",
    time_step_h: float = 3.0,
    pv_var: str = "gen_pv",
) -> pd.DataFrame:
    """Distribution of extreme-event timesteps across hours of the day.

    For each ``(model, period, zone, season, hour)`` combination, computes how
    often event timesteps occur, along with mean residual-load and PV during
    events versus the reference-period climatology.

    Parameters
    ----------
    cache_rl :
        ``{model: ds_rl}`` — post-battery residual-load datasets.
    cache_comp :
        ``{model: ds_comp}`` — component datasets (PV, wind, demand, …).
    season_thresholds :
        ``{model: {zone_str: {season: float}}}`` — seasonal EVT thresholds.
    period_defs :
        Fixed or per-model period definitions.
    models :
        Iterable of model chain IDs.
    zones :
        Zone coordinate values.
    season_order :
        Season labels, e.g. ``["DJF", "MAM", "JJA", "SON"]``.
    season_months :
        ``{season: [month_ints]}``.
    cache_clim :
        ``{model: ds_clim}`` — reference-period climatological baselines from
        ``calculate_climatological_baseline``.  When provided, ``mean_pv_clim``
        is the ``(month, hour)``-grouped reference-period mean PV, averaged
        across the season's months — directly comparable to the anomaly baseline
        used in driver attribution.  ``None`` → ``mean_pv_clim`` is always NaN.
    zone_dim :
        Name of the zone dimension (default ``"zone"``).
    method :
        Event detection method: ``"run_length"`` or ``"spa"``.
    time_step_h :
        Time resolution in hours (default ``3.0``).
    pv_var :
        Variable name for PV generation in ``cache_comp`` (default ``"gen_pv"``).

    Returns
    -------
    pd.DataFrame
        One row per ``(model, period, zone, season, hour)`` with columns:

        * ``n_event_steps``  — count of event timesteps at that hour
        * ``n_total_steps``  — count of all season timesteps at that hour
        * ``event_frac``     — fraction of season timesteps that are event steps
        * ``propensity``     — ``event_frac * n_hour_bins`` (1 = uniform)
        * ``pct_event_steps``— share of all season event timesteps at that hour (%)
        * ``mean_rl_event``  — mean RL (GW) during event timesteps at that hour
        * ``mean_pv_event``  — mean PV (GW) during event timesteps at that hour
        * ``mean_pv_clim``   — reference-period climatological mean PV (GW) at
                               that (season, hour), averaged across season months
    """
    from cordex2energy_analysis.plotting_extremes import resolve_model_periods

    records: list[dict] = []

    for model in models:
        ds_rl_m   = cache_rl[model]
        ds_comp_m = cache_comp[model]
        has_pv    = pv_var in ds_comp_m.data_vars
        ds_clim_m = cache_clim[model] if cache_clim is not None else None
        has_clim  = ds_clim_m is not None and pv_var in ds_clim_m.data_vars

        resolved = {
            k: v for k, v in resolve_model_periods(period_defs, model).items()
            if v is not None
        }

        for period_name, (t0, t1) in resolved.items():
            ds_rl_p   = ds_rl_m.sel(time=slice(t0, t1))
            ds_comp_p = ds_comp_m.sel(time=slice(t0, t1))
            times_p   = pd.DatetimeIndex(ds_rl_p["time"].values)
            hours     = times_p.hour

            for zone in zones:
                zone_str = str(zone)
                rl = ds_rl_p["residual_load"].sel({zone_dim: zone}).values
                pv = ds_comp_p[pv_var].sel({zone_dim: zone}).values if has_pv else None

                for season in season_order:
                    s_months = season_months[season]
                    s_mask   = np.isin(times_p.month, s_months)
                    thr      = float(season_thresholds[model][zone_str][season])

                    events = detect_seasonal_events(
                        rl, times_p, thr, s_months,
                        time_step_h=time_step_h, method=method,
                    )

                    event_mask = np.zeros(len(rl), dtype=bool)
                    for ev in events:
                        event_mask[ev["indices"]] = True

                    hour_bins = sorted(set(hours[s_mask]))
                    n_hour_bins = len(hour_bins)
                    n_event_total = int(event_mask[s_mask].sum())

                    # Reference-period climatological PV per hour, averaged across
                    # the season's months — same baseline as driver attribution.
                    clim_pv: "dict[int, float]" = {}
                    if has_clim:
                        da_clim_pv = ds_clim_m[pv_var].sel({zone_dim: zone})
                        for hour in hour_bins:
                            clim_pv[hour] = float(
                                da_clim_pv.sel(month=s_months, hour=hour)
                                .mean("month")
                                .values
                            )

                    for hour in hour_bins:
                        h_mask  = (hours == hour) & s_mask
                        he_mask = h_mask & event_mask
                        n_total = int(h_mask.sum())
                        n_event = int(he_mask.sum())
                        e_frac  = float(n_event / n_total) if n_total > 0 else np.nan

                        records.append({
                            "model":            model,
                            "period":           period_name,
                            "zone":             zone_str,
                            "season":           season,
                            "hour":             int(hour),
                            "n_event_steps":    n_event,
                            "n_total_steps":    n_total,
                            "event_frac":       e_frac,
                            "propensity":       e_frac * n_hour_bins if np.isfinite(e_frac) else np.nan,
                            "pct_event_steps":  float(n_event / n_event_total * 100) if n_event_total > 0 else np.nan,
                            "mean_rl_event":    float(rl[he_mask].mean()) if n_event > 0 else np.nan,
                            "mean_pv_event":    float(pv[he_mask].mean()) if (pv is not None and n_event > 0) else np.nan,
                            "mean_pv_clim":     clim_pv.get(hour, np.nan),
                        })

    df = pd.DataFrame(records)
    return df


# ---------------------------------------------------------------------------
# Duration-stratified metrics
# ---------------------------------------------------------------------------


def _attribution_shares(
    tot_d: float,
    tot_pv: float,
    tot_won: float,
    tot_woff: float,
    tot_hror: float,
    tot_res: float,
    tot_bat: float,
) -> dict:
    """Compute component attribution shares (%) from accumulated anomaly sums."""
    denom = tot_d + tot_pv + tot_won + tot_woff + tot_hror + tot_res + tot_bat
    if denom > 0:
        return {
            "Demand_Driven_Share_%":          tot_d    / denom * 100,
            "PV_Driven_Share_%":              tot_pv   / denom * 100,
            "Wind_Onshore_Driven_Share_%":    tot_won  / denom * 100,
            "Wind_Offshore_Driven_Share_%":   tot_woff / denom * 100,
            "Hydro_RoR_Driven_Share_%":       tot_hror / denom * 100,
            "Reservoir_Hydro_Driven_Share_%": tot_res  / denom * 100,
            "Battery_Driven_Share_%":         tot_bat  / denom * 100,
        }
    equal = 100.0 / 7
    return {k: equal for k in (
        "Demand_Driven_Share_%", "PV_Driven_Share_%",
        "Wind_Onshore_Driven_Share_%", "Wind_Offshore_Driven_Share_%",
        "Hydro_RoR_Driven_Share_%", "Reservoir_Hydro_Driven_Share_%",
        "Battery_Driven_Share_%",
    )}


def _empty_duration_row(
    model: str,
    period: str,
    zone: str,
    duration_class: str,
    extra: dict | None = None,
) -> dict:
    """Return a zero/NaN row for a (model, period, zone, duration_class) with no events."""
    row = {
        "model": model, "period": period, "zone": zone,
        "duration_class": duration_class,
        "n_events": 0, "freq_per_year": 0.0,
        "avg_duration_h": np.nan, "max_duration_h": np.nan,
        "avg_severity_gwh": np.nan, "max_severity_gwh": np.nan,
        "total_annual_severity_gwh": 0.0, "p90_annual_severity_gwh": 0.0,
        "Demand_Driven_Share_%": np.nan, "PV_Driven_Share_%": np.nan,
        "Wind_Onshore_Driven_Share_%": np.nan, "Wind_Offshore_Driven_Share_%": np.nan,
        "Hydro_RoR_Driven_Share_%": np.nan, "Reservoir_Hydro_Driven_Share_%": np.nan,
        "Battery_Driven_Share_%": np.nan,
        "Avg_Demand_Anomaly_GWh": np.nan, "Avg_PV_Anomaly_GWh": np.nan,
        "Avg_Wind_Onshore_Anomaly_GWh": np.nan, "Avg_Wind_Offshore_Anomaly_GWh": np.nan,
        "Avg_Hydro_RoR_Anomaly_GWh": np.nan, "Avg_Reservoir_Anomaly_GWh": np.nan,
        "Avg_Battery_Anomaly_GWh": np.nan,
    }
    if extra:
        row.update(extra)
    return row


def compute_duration_stratified_metrics(
    cache_rl: dict,
    cache_met: dict,
    evt_thresholds: dict,
    period_defs: dict,
    models,
    zones,
    zone_dim: str = "zone",
    method: str = "spa",
    duration_bins: dict | None = None,
) -> pd.DataFrame:
    """Compute EVT metrics stratified by event duration class — annual scale.

    For each (model, period, zone, duration_class) the function detects all
    exceedance events, partitions them by duration class, then aggregates
    frequency, severity, and component attribution shares for each class.

    Parameters
    ----------
    cache_rl :
        ``{model: ds_rl}`` — post-battery residual load datasets.
    cache_met :
        ``{model: ds_met}`` — output of ``compute_continuous_system_metrics``
        (provides component anomaly series and PCIT).
    evt_thresholds :
        ``{model: da_thresholds}`` — annual EVT threshold DataArrays.
    period_defs :
        Fixed or per-model period definitions (same format as
        ``resolve_gwl_period_definitions`` output).
    models :
        Iterable of model chain IDs.
    zones :
        Zone coordinate values.
    zone_dim :
        Name of the zone dimension (default ``"zone"``).
    method :
        Event-detection algorithm — ``"spa"`` (default) or ``"run_length"``.
    duration_bins :
        Ordered dict mapping label → (lo_exclusive, hi_inclusive) hour bounds.
        Defaults to ``DEFAULT_DURATION_BINS``.

    Returns
    -------
    pd.DataFrame
        One row per (model, period, zone, duration_class).  Columns mirror
        the output of ``compute_comprehensive_event_metrics`` extended with
        ``duration_class``, ``n_events``, and ``freq_per_year``.
    """
    from .plotting_extremes import resolve_model_periods

    bins = duration_bins or DEFAULT_DURATION_BINS
    records: list[dict] = []

    for model in models:
        ds_rl_m = cache_rl[model]
        ds_met_m = cache_met[model]
        da_thr_m = evt_thresholds[model]
        resolved = {
            k: v for k, v in resolve_model_periods(period_defs, model).items()
            if v is not None
        }

        for period_name, (t0, t1) in resolved.items():
            ds_rl_p  = ds_rl_m.sel(time=slice(t0, t1))
            ds_met_p = ds_met_m.sel(time=slice(t0, t1))
            year_series = ds_rl_p["time"].dt.year.values
            all_years   = np.unique(year_series)
            n_years     = len(all_years)

            has_res = "anomaly_reservoir_dispatch_gwh" in ds_met_p
            has_bat = "anomaly_battery_dispatch_gwh" in ds_met_p

            for zone in zones:
                zone_str = str(zone)
                thr = float(da_thr_m.sel({zone_dim: zone}).values)
                rl  = ds_rl_p["residual_load"].sel({zone_dim: zone}).values

                d_anom    = ds_met_p["anomaly_demand"].sel({zone_dim: zone}).values
                pv_anom   = ds_met_p["anomaly_gen_pv"].sel({zone_dim: zone}).values
                won_anom  = ds_met_p["anomaly_gen_wind_on"].sel({zone_dim: zone}).values
                woff_anom = ds_met_p["anomaly_gen_wind_off"].sel({zone_dim: zone}).values
                hror_anom = ds_met_p["anomaly_gen_hydro_ror"].sel({zone_dim: zone}).values
                res_anom  = (
                    ds_met_p["anomaly_reservoir_dispatch_gwh"].sel({zone_dim: zone}).values
                    if has_res else np.zeros(len(rl))
                )
                bat_anom = (
                    ds_met_p["anomaly_battery_dispatch_gwh"].sel({zone_dim: zone}).values
                    if has_bat else np.zeros(len(rl))
                )

                all_events = detect_events(rl, thr, method=method)

                # Partition events into duration classes
                classified: dict[str, list[dict]] = {dc: [] for dc in bins}
                for ev in all_events:
                    dc = classify_event_duration(ev["duration_h"], bins)
                    classified[dc].append(ev)

                for dc, dc_events in classified.items():
                    if not dc_events:
                        records.append(_empty_duration_row(
                            model, period_name, zone_str, dc
                        ))
                        continue

                    n_ev     = len(dc_events)
                    durations: list[float] = []
                    severities: list[float] = []
                    annual_sev: dict = {int(yr): 0.0 for yr in all_years}
                    tot_d = tot_pv = tot_won = tot_woff = tot_hror = tot_res = tot_bat = 0.0

                    for ev in dc_events:
                        idx = ev["indices"]
                        sev = ev["severity_gwh"]
                        ev_year = int(year_series[idx[0]])
                        durations.append(ev["duration_h"])
                        severities.append(sev)
                        annual_sev[ev_year] = annual_sev.get(ev_year, 0.0) + sev
                        tot_d    += float(d_anom[idx].sum())
                        tot_pv   += float(pv_anom[idx].sum())
                        tot_won  += float(won_anom[idx].sum())
                        tot_woff += float(woff_anom[idx].sum())
                        tot_hror += float(hror_anom[idx].sum())
                        tot_res  += float(res_anom[idx].sum())
                        tot_bat  += float(bat_anom[idx].sum())

                    ann_arr = np.array(list(annual_sev.values()))
                    shares  = _attribution_shares(
                        tot_d, tot_pv, tot_won, tot_woff, tot_hror, tot_res, tot_bat
                    )

                    records.append({
                        "model": model, "period": period_name, "zone": zone_str,
                        "duration_class": dc,
                        "n_events":                    n_ev,
                        "freq_per_year":               n_ev / n_years,
                        "avg_duration_h":              float(np.mean(durations)),
                        "max_duration_h":              float(np.max(durations)),
                        "avg_severity_gwh":            float(np.mean(severities)),
                        "max_severity_gwh":            float(np.max(severities)),
                        "total_annual_severity_gwh":   float(np.sum(severities)) / n_years,
                        "p90_annual_severity_gwh":     float(np.percentile(ann_arr, 90)),
                        "Avg_Demand_Anomaly_GWh":      round( tot_d    * 3 / n_ev, 2),
                        "Avg_PV_Anomaly_GWh":          round(-tot_pv   * 3 / n_ev, 2),
                        "Avg_Wind_Onshore_Anomaly_GWh":  round(-tot_won  * 3 / n_ev, 2),
                        "Avg_Wind_Offshore_Anomaly_GWh": round(-tot_woff * 3 / n_ev, 2),
                        "Avg_Hydro_RoR_Anomaly_GWh":     round(-tot_hror * 3 / n_ev, 2),
                        "Avg_Reservoir_Anomaly_GWh":     round(-tot_res  * 3 / n_ev, 2),
                        "Avg_Battery_Anomaly_GWh":       round(-tot_bat  * 3 / n_ev, 2),
                        **shares,
                    })

    return pd.DataFrame(records)


def compute_seasonal_duration_stratified_metrics(
    cache_rl: dict,
    cache_met: dict,
    seasonal_thresholds: dict,
    period_defs: dict,
    models,
    zones,
    season_order: list[str],
    season_months: dict[str, list[int]],
    zone_dim: str = "zone",
    method: str = "spa",
    duration_bins: dict | None = None,
) -> pd.DataFrame:
    """Compute EVT metrics stratified by event duration class — seasonal scale.

    Mirrors ``compute_seasonal_evt_metrics`` but adds a ``duration_class``
    column so each (zone, season) produces one row per duration class.

    Parameters
    ----------
    cache_rl :
        ``{model: ds_rl}`` — post-battery residual load datasets.
    cache_met :
        ``{model: ds_met}`` — output of ``compute_continuous_system_metrics``.
    seasonal_thresholds :
        ``{model: {zone: {season: float}}}`` season-specific EVT thresholds.
    period_defs :
        Fixed or per-model period definitions.
    models :
        Iterable of model chain IDs.
    zones :
        Zone coordinate values.
    season_order :
        Season names in display order.
    season_months :
        ``{season: [month_ints]}``.
    zone_dim :
        Name of the zone dimension (default ``"zone"``).
    method :
        Event-detection algorithm — ``"spa"`` (default) or ``"run_length"``.
    duration_bins :
        Ordered dict mapping label → (lo_exclusive, hi_inclusive) hour bounds.
        Defaults to ``DEFAULT_DURATION_BINS``.

    Returns
    -------
    pd.DataFrame
        One row per (model, period, zone, season, duration_class).
    """
    from .plotting_extremes import resolve_model_periods

    bins = duration_bins or DEFAULT_DURATION_BINS
    records: list[dict] = []

    for model in models:
        ds_rl_m  = cache_rl[model]
        ds_met_m = cache_met[model]
        thr_m    = seasonal_thresholds[model]
        resolved = {
            k: v for k, v in resolve_model_periods(period_defs, model).items()
            if v is not None
        }

        for period_name, (t0, t1) in resolved.items():
            ds_rl_p  = ds_rl_m.sel(time=slice(t0, t1))
            ds_met_p = ds_met_m.sel(time=slice(t0, t1))
            times_p  = pd.DatetimeIndex(ds_rl_p["time"].values)

            has_res = "anomaly_reservoir_dispatch_gwh" in ds_met_p
            has_bat = "anomaly_battery_dispatch_gwh" in ds_met_p

            for season in season_order:
                s_mask  = np.isin(times_p.month, season_months[season])
                s_years = np.unique(ds_rl_p["time"].dt.year.values[s_mask])

                for zone in zones:
                    zone_key = str(zone)
                    rl  = ds_rl_p["residual_load"].sel({zone_dim: zone}).values
                    thr = thr_m[zone_key][season]

                    d_anom    = ds_met_p["anomaly_demand"].sel({zone_dim: zone}).values
                    pv_anom   = ds_met_p["anomaly_gen_pv"].sel({zone_dim: zone}).values
                    won_anom  = ds_met_p["anomaly_gen_wind_on"].sel({zone_dim: zone}).values
                    woff_anom = ds_met_p["anomaly_gen_wind_off"].sel({zone_dim: zone}).values
                    hror_anom = ds_met_p["anomaly_gen_hydro_ror"].sel({zone_dim: zone}).values
                    res_anom  = (
                        ds_met_p["anomaly_reservoir_dispatch_gwh"].sel({zone_dim: zone}).values
                        if has_res else np.zeros(len(rl))
                    )
                    bat_anom = (
                        ds_met_p["anomaly_battery_dispatch_gwh"].sel({zone_dim: zone}).values
                        if has_bat else np.zeros(len(rl))
                    )

                    all_events = detect_seasonal_events(
                        rl, times_p, thr, season_months[season], method=method
                    )

                    # Partition events into duration classes
                    classified: dict[str, list[dict]] = {dc: [] for dc in bins}
                    for ev in all_events:
                        dc = classify_event_duration(ev["duration_h"], bins)
                        classified[dc].append(ev)

                    for dc, dc_events in classified.items():
                        if not dc_events:
                            records.append(_empty_duration_row(
                                model, period_name, zone_key, dc,
                                extra={"season": season},
                            ))
                            continue

                        n_ev       = len(dc_events)
                        durations: list[float]  = []
                        severities: list[float] = []
                        annual_sev: dict = {int(yr): 0.0 for yr in s_years}
                        tot_d = tot_pv = tot_won = tot_woff = tot_hror = tot_res = tot_bat = 0.0

                        for ev in dc_events:
                            idx     = ev["indices"]
                            sev     = ev["severity_gwh"]
                            ev_year = int(times_p[idx[0]].year)
                            durations.append(ev["duration_h"])
                            severities.append(sev)
                            annual_sev[ev_year] = annual_sev.get(ev_year, 0.0) + sev
                            tot_d    += float(d_anom[idx].sum())
                            tot_pv   += float(pv_anom[idx].sum())
                            tot_won  += float(won_anom[idx].sum())
                            tot_woff += float(woff_anom[idx].sum())
                            tot_hror += float(hror_anom[idx].sum())
                            tot_res  += float(res_anom[idx].sum())
                            tot_bat  += float(bat_anom[idx].sum())

                        ann_arr = np.array(list(annual_sev.values()))
                        shares  = _attribution_shares(
                            tot_d, tot_pv, tot_won, tot_woff, tot_hror, tot_res, tot_bat
                        )

                        records.append({
                            "model": model, "period": period_name, "zone": zone_key,
                            "season": season, "duration_class": dc,
                            "n_events":                    n_ev,
                            "freq_per_year":               n_ev / len(s_years),
                            "avg_duration_h":              float(np.mean(durations)),
                            "max_duration_h":              float(np.max(durations)),
                            "avg_severity_gwh":            float(np.mean(severities)),
                            "max_severity_gwh":            float(np.max(severities)),
                            "total_annual_severity_gwh":   float(np.sum(severities)) / len(s_years),
                            "p90_annual_severity_gwh":     float(np.percentile(ann_arr, 90)),
                            "Avg_Demand_Anomaly_GWh":      round( tot_d    * 3 / n_ev, 2),
                            "Avg_PV_Anomaly_GWh":          round(-tot_pv   * 3 / n_ev, 2),
                            "Avg_Wind_Onshore_Anomaly_GWh":  round(-tot_won  * 3 / n_ev, 2),
                            "Avg_Wind_Offshore_Anomaly_GWh": round(-tot_woff * 3 / n_ev, 2),
                            "Avg_Hydro_RoR_Anomaly_GWh":     round(-tot_hror * 3 / n_ev, 2),
                            "Avg_Reservoir_Anomaly_GWh":     round(-tot_res  * 3 / n_ev, 2),
                            "Avg_Battery_Anomaly_GWh":       round(-tot_bat  * 3 / n_ev, 2),
                            **shares,
                        })

    return pd.DataFrame(records)
