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


def compute_zone_pair_cooccurrence_by_duration(
    cache_rl: dict,
    evt_thresholds: dict,
    period_defs: dict,
    models,
    zones,
    window_steps: int = 24,
    method: str = "spa",
    duration_bins: dict | None = None,
    zone_dim: str = "zone",
) -> pd.DataFrame:
    """Zone-pair co-occurrence conditioned on event duration class (gap G2).

    Like :func:`compute_zone_pair_cooccurrence`, but instead of dilating *all*
    exceedance timesteps, events are first detected per zone and partitioned
    into duration classes; co-occurrence is then computed *within* each class.
    This tests whether long, synoptic ``multi_day`` events are more spatially
    coherent across zones (blocking is large-scale) than short ``sub_daily``
    events (more zone-specific) — the systemic-risk question behind the
    importance of cross-border balancing.

    For each (model, period, duration_class, zone_a, zone_b) the fraction of
    Zone A *class-member* timesteps that fall within ±``window_steps`` steps of a
    Zone B *class-member* timestep is reported, together with ``n_a`` (the count
    of Zone A class-member timesteps) so sparse cells can be masked downstream.

    Parameters
    ----------
    cache_rl :
        ``{model: ds_rl}`` — post-battery residual load datasets.
    evt_thresholds :
        Annual ``{model: xr.DataArray}`` thresholds (one per zone).
    period_defs :
        Fixed or per-model period definitions.
    models, zones :
        Model chain IDs and zone coordinate values.
    window_steps :
        Half-width of the dilation kernel in time steps (default 24 = ±72 h),
        matching :func:`compute_zone_pair_cooccurrence`. ±72 h comfortably spans
        the synoptic timescale of blocking, so a multi_day event in one zone and
        a multi_day event in another are counted as co-occurring when their
        active windows overlap within three days.
    method :
        Event-detection algorithm — ``"spa"`` (default) or ``"run_length"``.
    duration_bins :
        Ordered ``{label: (lo_exclusive, hi_inclusive)}`` hour bounds. Defaults
        to ``DEFAULT_DURATION_BINS``; pass ``{"sub_daily": (0, 24),
        "multi_day": (24, inf)}`` for the two-regime split.
    zone_dim :
        Name of the zone dimension (default ``"zone"``).

    Returns
    -------
    pd.DataFrame
        Columns: model, period, duration_class, zone_a, zone_b,
        co_occurrence_pct, n_a.
    """
    from cordex2energy_analysis.plotting_extremes import resolve_model_periods

    bins = duration_bins or DEFAULT_DURATION_BINS
    dilation_struct = np.ones(2 * window_steps + 1, dtype=bool)
    records: list[dict] = []

    for model in models:
        ds_rl_m = cache_rl[model]
        thr_m = evt_thresholds[model]
        resolved = {
            k: v for k, v in resolve_model_periods(period_defs, model).items()
            if v is not None
        }
        for period_name, (t0, t1) in resolved.items():
            ds_rl_p = ds_rl_m.sel(time=slice(t0, t1))
            n_steps = ds_rl_p.sizes["time"]

            # Per-duration-class, per-zone timestep masks (True at indices that
            # belong to an event of that class).
            dc_masks: dict = {dc: {} for dc in bins}
            for zone in zones:
                rl = ds_rl_p["residual_load"].sel({zone_dim: zone}).values
                thr = float(thr_m.sel({zone_dim: zone}).values)
                per_dc = {dc: np.zeros(n_steps, dtype=bool) for dc in bins}
                for ev in detect_events(rl, thr, method=method):
                    dc = classify_event_duration(ev["duration_h"], bins)
                    per_dc[dc][ev["indices"]] = True
                for dc in bins:
                    dc_masks[dc][str(zone)] = per_dc[dc]

            for dc in bins:
                exc_masks = dc_masks[dc]
                dilated = _cooccurrence_on_array(exc_masks, zones, dilation_struct)
                for za in zones:
                    za_key = str(za)
                    n_a = int(exc_masks[za_key].sum())
                    for zb in zones:
                        zb_key = str(zb)
                        frac = (
                            float((exc_masks[za_key] & dilated[zb_key]).sum()) / n_a * 100
                            if n_a > 0 else np.nan
                        )
                        records.append({
                            "model": model, "period": period_name,
                            "duration_class": dc,
                            "zone_a": za_key, "zone_b": zb_key,
                            "co_occurrence_pct": frac, "n_a": n_a,
                        })

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


#: Per-asset metric prefixes that only exist when the battery dispatch is split.
_SPLIT_ONLY_PREFIXES = ("prosumer_", "utility_")

#: Metric columns emitted by :func:`compute_flexibility_attenuation_metrics`, by kind.
_FLEX_METRIC_GROUPS = {
    "deployed": [
        "battery_deployed_gwh_per_event", "prosumer_deployed_gwh_per_event",
        "utility_deployed_gwh_per_event", "reservoir_deployed_gwh_per_event",
    ],
    "attenuation": [
        "battery_attenuation_pct", "prosumer_attenuation_pct",
        "utility_attenuation_pct", "reservoir_attenuation_pct",
        "combined_attenuation_pct",
    ],
    "share": ["battery_attenuation_share_pct", "reservoir_attenuation_share_pct"],
}


def battery_split_mode(ds_comp) -> str:
    """How battery dispatch is disaggregated in a components dataset.

    Returns one of:

    - ``"split"``   — ``battery_``, ``prosumer_`` and ``utility_dispatch_gwh`` all present
      (the ``market`` flexibility scheme, both assets dispatched separately).
    - ``"partial"`` — battery plus exactly one of prosumer/utility.  This is a real
      state: under ``market`` with ``--utility_dispatch_level zone`` the *country-level*
      ``components.nc`` holds ``prosumer_dispatch_gwh`` but not ``utility_dispatch_gwh``,
      because utility is dispatched after zonal aggregation.
    - ``"lumped"``  — only the combined ``battery_dispatch_gwh`` (the ``system`` scheme,
      where prosumer and utility are one storage asset).
    - ``"none"``    — no battery dispatch at all.

    In every mode ``battery_dispatch_gwh`` is the **total** storage dispatch, so the
    combined battery metrics and the battery/reservoir Shapley shares stay comparable
    across schemes; only the intra-battery breakdown depends on the mode.
    """
    dv = ds_comp.data_vars
    if "battery_dispatch_gwh" not in dv:
        return "none"
    n_split = ("prosumer_dispatch_gwh" in dv) + ("utility_dispatch_gwh" in dv)
    return {2: "split", 1: "partial", 0: "lumped"}[n_split]


def flex_metric_columns(df, kind: str = "attenuation", drop_all_nan: bool = True) -> list:
    """Flexibility metric columns of *df* that are meaningful for its ``battery_mode``.

    Use this instead of hardcoding metric lists, so a lumped (``system``) run plots the
    combined battery metrics without empty prosumer/utility panels.

    Parameters
    ----------
    df :
        DataFrame from :func:`compute_flexibility_attenuation_metrics`.
    kind :
        ``"deployed"``, ``"attenuation"``, ``"share"``, or ``"all"``.
    drop_all_nan :
        Also drop columns that are present but entirely NaN.  This is what removes the
        per-asset entries under ``lumped`` (and the missing one under ``partial``).

    Returns
    -------
    list of str
        Column names, in canonical order, that exist in *df* and carry data.
    """
    if kind == "all":
        candidates = [c for g in _FLEX_METRIC_GROUPS.values() for c in g]
    elif kind in _FLEX_METRIC_GROUPS:
        candidates = list(_FLEX_METRIC_GROUPS[kind])
    else:
        raise ValueError(
            f"Unknown kind {kind!r}; expected one of "
            f"{tuple(_FLEX_METRIC_GROUPS) + ('all',)}."
        )

    mode = None
    if "battery_mode" in df.columns:
        modes = df["battery_mode"].dropna().unique()
        mode = modes[0] if len(modes) == 1 else None  # mixed -> fall back to data checks

    cols = []
    for c in candidates:
        if c not in df.columns:
            continue
        if mode == "lumped" and c.startswith(_SPLIT_ONLY_PREFIXES):
            continue
        if drop_all_nan and df[c].isna().all():
            continue
        cols.append(c)
    return cols


def _flex_event_metrics(
    events, rl, thr, bat_disp, pro_disp, util_disp, res_disp,
    has_prosumer, has_utility, has_battery, has_reservoir, time_step_h=3.0,
    res_state=None, bat_state=None, per_event=None,
) -> dict:
    """Aggregate per-event flexibility deployed / attenuation metrics for a list
    of events (NaNs when empty).  Shared by the pooled and duration-stratified
    paths of :func:`compute_flexibility_attenuation_metrics`.

    When *per_event* is a list, one row per event is appended to it as a side effect: the
    quantities the ratios are built from, kept rather than averaged away, plus each asset's state
    of charge at the event's first timestep as a per cent of that asset's own maximum over the
    window.  A ratio cannot answer WHY attenuation degrades, and the two questions it has to answer
    both need the individual events — whether the fall survives conditioning on event size (it is
    otherwise indistinguishable from "the events simply got bigger"), and whether the assets were
    full when the event began.
    """
    _nan = float("nan")
    if not events:
        return {
            "n_events": 0,
            "battery_deployed_gwh_per_event": _nan, "prosumer_deployed_gwh_per_event": _nan,
            "utility_deployed_gwh_per_event": _nan, "reservoir_deployed_gwh_per_event": _nan,
            "battery_attenuation_pct": _nan, "prosumer_attenuation_pct": _nan,
            "utility_attenuation_pct": _nan, "reservoir_attenuation_pct": _nan,
            "combined_attenuation_pct": _nan,
            "battery_attenuation_share_pct": _nan, "reservoir_attenuation_share_pct": _nan,
        }
    bat_dep, pro_dep, util_dep, res_dep = [], [], [], []
    bat_att, pro_att, util_att, res_att, comb_att = [], [], [], [], []
    bat_share, res_share = [], []
    for ev in events:
        ev_idx = ev["indices"]
        # Energy per step in, so energy is a plain sum (see extremes_events.STEP_HOURS).
        sev_actual = float(np.maximum(rl[ev_idx] - thr, 0.0).sum())

        def _sev(discharge):
            return float(np.maximum((rl[ev_idx] + discharge) - thr, 0.0).sum())

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

        bat_dep.append(float(bat_dis.sum()))
        pro_dep.append(float(pro_dis.sum()) if has_prosumer else np.nan)
        util_dep.append(float(util_dis.sum()) if has_utility else np.nan)
        res_dep.append(float(res_dis.sum()))

        bat_att.append(_atten(sev_no_bat,  has_battery))
        pro_att.append(_atten(sev_no_pro,  has_prosumer))
        util_att.append(_atten(sev_no_util, has_utility))
        res_att.append(_atten(sev_no_res,  has_reservoir))
        comb_att.append(
            (sev_no_both - sev_actual) / sev_no_both * 100 if sev_no_both > 0 else 0.0
        )

        # Shapley split of combined attenuation between battery and reservoir.
        phi_bat = 0.5 * (sev_no_both - sev_no_res) + 0.5 * (sev_no_bat - sev_actual)
        phi_res = 0.5 * (sev_no_both - sev_no_bat) + 0.5 * (sev_no_res - sev_actual)
        if sev_no_both > 0:
            bat_share.append(phi_bat / sev_no_both * 100)
            res_share.append(phi_res / sev_no_both * 100)
        else:
            bat_share.append(0.0)
            res_share.append(0.0)

        if per_event is not None:
            # State of charge at the FIRST timestep of the event, as a per cent of that asset's own
            # maximum over the window. Normalising by the observed maximum rather than by a rated
            # capacity keeps this readable without a capacity table, and the question asked of it —
            # "was the asset full when the event began?" — is scale-free.
            def _onset(state):
                if state is None:
                    return np.nan
                top = float(np.nanmax(state))
                return float(state[ev_idx[0]]) / top * 100 if top > 0 else np.nan

            per_event.append({
                "duration_h": float(ev["duration_h"]),
                "sev_actual": sev_actual,
                "sev_no_storage": sev_no_both,
                # phi_res + phi_bat == sev_no_both - sev_actual identically, so the absolute split
                # closes for the same reason the percentage one does.
                "absorbed_reservoir": phi_res,
                "absorbed_battery": phi_bat,
                "atten": comb_att[-1],
                "res_dep": res_dep[-1],
                "bat_dep": bat_dep[-1],
                "res_onset_pct": _onset(res_state),
                "bat_onset_pct": _onset(bat_state),
            })

    def _mean(lst):
        arr = np.array(lst, dtype=float)
        return float(np.nanmean(arr)) if arr.size else np.nan

    return {
        "n_events": len(events),
        "battery_deployed_gwh_per_event":  _mean(bat_dep),
        "prosumer_deployed_gwh_per_event": _mean(pro_dep),
        "utility_deployed_gwh_per_event":  _mean(util_dep),
        "reservoir_deployed_gwh_per_event": _mean(res_dep),
        "battery_attenuation_pct":  _mean(bat_att),
        "prosumer_attenuation_pct": _mean(pro_att),
        "utility_attenuation_pct":  _mean(util_att),
        "reservoir_attenuation_pct": _mean(res_att),
        "combined_attenuation_pct": _mean(comb_att),
        "battery_attenuation_share_pct":   _mean(bat_share),
        "reservoir_attenuation_share_pct": _mean(res_share),
    }


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
    duration_bins: "dict | None" = None,
    return_events: bool = False,
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

        Under the lumped ``system`` flexibility scheme only ``battery_dispatch_gwh``
        exists, so the per-asset ``prosumer_*`` / ``utility_*`` metrics are NaN and the
        emitted ``battery_mode`` column is ``"lumped"`` — see :func:`battery_split_mode`.
        Use :func:`flex_metric_columns` to select the metrics that carry data instead of
        hardcoding column lists.
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
    return_events :
        Also return a second frame with ONE ROW PER EVENT, carrying ``duration_h``,
        ``sev_actual``, ``sev_no_storage``, the Shapley split in GWh, ``atten``, each asset's
        dispatched energy, and ``res_onset_pct`` / ``bat_onset_pct`` — the state of charge at the
        event's first timestep, as a per cent of that asset's own maximum over the window.  These
        are the quantities the summary averages away, and they are what a mechanism question needs:
        an aggregate cannot distinguish "storage degraded" from "the events got bigger", and only
        the individual events can be conditioned on their own counterfactual severity.  Requires
        ``reservoir_storage_gwh`` / ``battery_soc_gwh`` in *cache_comp* for the onset columns; they
        are NaN when absent.

    Returns
    -------
    pd.DataFrame
        One row per (model, period, zone[, season][, duration_class]).  When
        ``duration_bins`` is given, events are partitioned into duration classes
        and one row is emitted per class.  Columns include
        ``n_events``, ``*_deployed_gwh_per_event``, ``*_attenuation_pct``,
        ``combined_attenuation_pct``, and the string column ``battery_mode``.

        The column schema is the same in every ``battery_mode`` (per-asset entries are
        NaN when unavailable), so ``market`` and ``system`` runs can be concatenated and
        compared.  ``battery_mode`` is *not* numeric — restrict broad aggregations
        (``.mean()``, ``.median()``) to numeric columns, or select via
        :func:`flex_metric_columns`.

        Because ``battery_dispatch_gwh`` is the **total** storage dispatch in every
        scheme, ``battery_deployed_gwh_per_event``, ``battery_attenuation_pct``,
        ``combined_attenuation_pct`` and the Shapley shares are directly comparable
        across ``market`` and ``system``; only the intra-battery breakdown is not.

        Note that the per-resource ``*_attenuation_pct`` are *marginal*
        (leave-one-out) quantities computed against independent counterfactual
        denominators, so when battery and reservoir act on the same deficits
        they overlap and can sum to more than 100 %. For an additive split use
        ``battery_attenuation_share_pct`` and ``reservoir_attenuation_share_pct``
        — a two-player Shapley decomposition of ``combined_attenuation_pct`` that
        sums (at the per-event mean level) exactly to it.

    (pd.DataFrame, pd.DataFrame)
        When ``return_events`` is true: the summary above, and the per-event frame described under
        that parameter, keyed by the same (model, period, zone[, season][, duration_class]).
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
    event_records: list[dict] = []

    for model in models:
        ds_rl_m   = cache_rl[model]
        ds_comp_m = cache_comp[model]
        thr_m     = evt_thresholds[model]
        seasonal_thr = isinstance(thr_m, dict)

        has_battery   = "battery_dispatch_gwh"   in ds_comp_m.data_vars
        has_prosumer  = "prosumer_dispatch_gwh"  in ds_comp_m.data_vars
        has_utility   = "utility_dispatch_gwh"   in ds_comp_m.data_vars
        has_reservoir = "reservoir_dispatch_gwh" in ds_comp_m.data_vars
        # Reported on every row so a NaN per-asset metric is distinguishable from a
        # genuine zero, and so `market` and `system` runs can be concatenated.
        battery_mode = battery_split_mode(ds_comp_m)

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
                # STATE, not dispatch: read only for the per-event export, and only when asked, so
                # the default path opens nothing extra.
                res_state = bat_state = None
                if return_events:
                    for var, name in (("reservoir_storage_gwh", "res"), ("battery_soc_gwh", "bat")):
                        series = (ds_comp_p[var].sel({zone_dim: zone}).values
                                  if var in ds_comp_p.data_vars else None)
                        if name == "res":
                            res_state = series
                        else:
                            bat_state = series

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
                    base_row = {
                        "model": model, "period": period_name, "zone": str(zone),
                        "battery_mode": battery_mode,
                    }
                    if season is not None:
                        base_row["season"] = season

                    _flex_args = (rl, thr, bat_disp, pro_disp, util_disp, res_disp,
                                  has_prosumer, has_utility, has_battery, has_reservoir)
                    _flex_kwargs = (dict(res_state=res_state, bat_state=bat_state)
                                    if return_events else {})

                    def _run(ev_list, row):
                        """One cell of the summary, and its events tagged with that cell's keys."""
                        collected = [] if return_events else None
                        summary = _flex_event_metrics(ev_list, *_flex_args, per_event=collected,
                                                      **_flex_kwargs)
                        for entry in collected or ():
                            event_records.append({**row, **entry})
                        records.append({**row, **summary})

                    if duration_bins is None:
                        _run(events, base_row)
                    else:
                        _classified = {dc: [] for dc in duration_bins}
                        for ev in events:
                            _classified[classify_event_duration(ev["duration_h"], duration_bins)].append(ev)
                        for dc, ev_list in _classified.items():
                            _run(ev_list, {**base_row, "duration_class": dc})

    if return_events:
        return pd.DataFrame(records), pd.DataFrame(event_records)
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
                        "Avg_Demand_Anomaly_GWh":      round( tot_d    / n_ev, 2),
                        "Avg_PV_Anomaly_GWh":          round(-tot_pv   / n_ev, 2),
                        "Avg_Wind_Onshore_Anomaly_GWh":  round(-tot_won  / n_ev, 2),
                        "Avg_Wind_Offshore_Anomaly_GWh": round(-tot_woff / n_ev, 2),
                        "Avg_Hydro_RoR_Anomaly_GWh":     round(-tot_hror / n_ev, 2),
                        "Avg_Reservoir_Anomaly_GWh":     round(-tot_res  / n_ev, 2),
                        "Avg_Battery_Anomaly_GWh":       round(-tot_bat  / n_ev, 2),
                        **shares,
                    })

    return pd.DataFrame(records)


def compute_event_season_metrics(
    cache_rl: dict,
    threshold_by_model: dict,
    period_defs: dict,
    models,
    zones,
    season_map: dict,
    season_order,
    *,
    zone_dim: str = "zone",
    method: str = "spa",
    duration_bins: dict | None = None,
) -> pd.DataFrame:
    """Seasonal event metrics from ONE annual detection pass, events binned by prevailing season.

    Returns one row per (model, period, zone, season, duration_class) with ``n_events``,
    ``freq_per_year`` and ``avg_duration_h`` — the columns the paper's figures consume.

    Why this and not :func:`compute_seasonal_duration_stratified_metrics`, which detects inside each
    season's months: that pass and the annual pass of
    :func:`compute_duration_stratified_metrics` are different runs of the detector, so their counts
    do not agree.  Measured on this paper's run the four seasons sum to within ~1 % of the annual
    count, 4.5 % at worst, which makes a "share of the zone's events" impossible to define
    consistently.  Worse, splitting an event at a season boundary can leave both halves under 24 h,
    so a genuine multi-day event is demoted to two sub-daily ones — 11 events on this run.

    Here the detector runs once over the window, exactly as the annual table does, and each event is
    assigned whole to the season carrying most of its timesteps.  Consequences, all of them wanted:

    * the four seasonal frequencies sum to the annual frequency EXACTLY, because every event is
      counted once and the denominator is the window's year count for every season alike;
    * an event's duration class is decided once, on its full duration;
    * a season's ``avg_duration_h`` is the mean FULL duration of the events belonging to it, so
      ``freq_per_year * avg_duration_h`` is the hours those events carry, not the hours that happen
      to land inside the season's months.  For a 1.2 % straddle rate the two differ negligibly, but
      the figures that use it should say which they mean.

    Ties — an event split evenly between two seasons — go to the season of the event's midpoint
    timestep, which is deterministic and needs no ordering convention over the seasons.
    """
    from .plotting_extremes import resolve_model_periods

    bins = duration_bins or DEFAULT_DURATION_BINS
    records: list[dict] = []

    for model in models:
        resolved = {k: v for k, v in resolve_model_periods(period_defs, model).items()
                    if v is not None}
        for period_name, (t0, t1) in resolved.items():
            ds_rl = cache_rl[model].sel(time=slice(t0, t1))
            times = pd.DatetimeIndex(ds_rl["time"].values)
            seasons = np.array([season_map[int(m)] for m in times.month])
            n_years = len(np.unique(times.year))

            for zone in zones:
                series = ds_rl["residual_load"].sel({zone_dim: zone}).values
                threshold = float(threshold_by_model[model].sel({zone_dim: zone}).values)
                buckets: dict[tuple, list] = {(s, dc): [] for s in season_order for dc in bins}

                for event in detect_events(series, threshold, method=method):
                    indices = np.asarray(event["indices"])
                    labels, counts = np.unique(seasons[indices], return_counts=True)
                    winners = labels[counts == counts.max()]
                    season = (str(seasons[indices[len(indices) // 2]]) if len(winners) > 1
                              else str(winners[0]))
                    buckets[(season, classify_event_duration(event["duration_h"], bins))].append(
                        float(event["duration_h"]))

                for (season, duration_class), durations in buckets.items():
                    records.append(dict(
                        model=model, period=period_name, zone=str(zone), season=season,
                        duration_class=duration_class, n_events=len(durations),
                        freq_per_year=len(durations) / n_years,
                        avg_duration_h=float(np.mean(durations)) if durations else np.nan))
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

    .. note::
       The paper's figures do NOT use this function.  It detects events inside each season's months,
       which is a different run of the detector from the annual pass, so its counts do not sum to
       the annual ones and an event straddling a season boundary can be split into two shorter
       events — sometimes demoting a multi-day event to two sub-daily ones.  The figures use
       :func:`compute_event_season_metrics`, which detects once and assigns each whole event to its
       prevailing season.  This function is kept because per-season detection with per-season
       thresholds is a legitimate alternative convention, but the two must not be mixed.

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
                            "Avg_Demand_Anomaly_GWh":      round( tot_d    / n_ev, 2),
                            "Avg_PV_Anomaly_GWh":          round(-tot_pv   / n_ev, 2),
                            "Avg_Wind_Onshore_Anomaly_GWh":  round(-tot_won  / n_ev, 2),
                            "Avg_Wind_Offshore_Anomaly_GWh": round(-tot_woff / n_ev, 2),
                            "Avg_Hydro_RoR_Anomaly_GWh":     round(-tot_hror / n_ev, 2),
                            "Avg_Reservoir_Anomaly_GWh":     round(-tot_res  / n_ev, 2),
                            "Avg_Battery_Anomaly_GWh":       round(-tot_bat  / n_ev, 2),
                            **shares,
                        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Paper figures: derived tables
#
# Everything below feeds the residual-load extremes paper
# (``notebooks/paper_main_figures.ipynb``).  Each function takes its inputs
# explicitly -- no reliance on notebook globals -- so the notebook can stay a
# narrative and these can be tested and reused.
#
# Convention shared by the attribution helpers: residual load decomposes as
#     RL = demand - PV - wind_on - wind_off - hydro_ror - reservoir - battery
# so a *rise in demand* and a *fall in generation/storage* both RAISE the
# deficit.  Contributions are therefore reported with demand entering at +sign
# and every generation/storage term negated.
# ---------------------------------------------------------------------------

#: Residual-load components, mapped to their display names (the stacking order of the
#: attribution bars).
PAPER_COMPONENT_LABELS = {
    "total_demand": "Demand",
    "gen_pv": "PV",
    "gen_wind_on": "Wind Onshore",
    "gen_wind_off": "Wind Offshore",
    "gen_hydro_ror": "Hydro RoR",
    "reservoir_dispatch_gwh": "Reservoir Hydro",
    "battery_dispatch_gwh": "Battery",
}

#: Hours (UTC) of the evening demand peak, when PV has already faded.
EVENING_PEAK_HOURS = (15, 18, 21)

#: Hours (UTC) the deficit actually occupies: the evening peak plus the overnight stretch, i.e.
#: everything except the midday PV hours.  Sub-daily events run 10-14 h -- three to five 3-hourly
#: steps -- so they spill out of the evening and into the night: measured on the LOW run at +3.0 C,
#: only 44 % (Iberia DJF) to 74 % (South-East JJA) of their timesteps fall in EVENING_PEAK_HOURS,
#: against 97-100 % here.  This is the window the diurnal panel of Figure 3 shades.
DEFICIT_WINDOW_HOURS = (0, 3, 6, 15, 18, 21)


def compute_evening_peak_attribution(
    residual_load_by_model: dict,
    components_by_model: dict,
    period_defs: dict,
    zones,
    season_map: dict,
    *,
    hours=EVENING_PEAK_HOURS,
    component_labels: dict | None = None,
) -> pd.DataFrame:
    """Mean of every residual-load component over a window of hours, per chain/period/zone/season.

    Returns one row per (model, period, zone, season) with a ``RL`` column and one column per
    driver.  ``season`` is ``"ALL"``, ``"DJF"`` or ``"JJA"`` -- the seasonal split is what exposes
    the summer cooling-demand signal behind the sub-daily surge.

    The window is the ``hours`` argument; :data:`EVENING_PEAK_HOURS` is only its default, kept so
    older callers keep their meaning.  The paper's figures pass :data:`DEFICIT_WINDOW_HOURS`, which
    is where the events actually sit -- see that constant for the measurement.
    """
    from .plotting_extremes import resolve_model_periods

    labels = component_labels or PAPER_COMPONENT_LABELS
    records = []
    for model in residual_load_by_model:
        rl_all, comp_all = residual_load_by_model[model], components_by_model[model]
        windows = {k: v for k, v in resolve_model_periods(period_defs, model).items()
                   if v is not None}
        for period, (start, end) in windows.items():
            rl = rl_all.sel(time=slice(start, end))
            comp = comp_all.sel(time=slice(start, end))
            hour = rl["time"].dt.hour.values
            season = np.array([season_map[int(m)] for m in rl["time"].dt.month.values])
            evening = np.isin(hour, list(hours))
            for zone in zones:
                rl_zone = rl["residual_load"].sel(zone=zone).values
                comp_zone = {v: comp[v].sel(zone=zone).values for v in labels if v in comp}
                for season_mask, season_label in [(np.ones_like(evening, bool), "ALL"),
                                                  (season == "DJF", "DJF"),
                                                  (season == "JJA", "JJA")]:
                    mask = evening & season_mask
                    if mask.sum() == 0:
                        continue
                    row = dict(model=model, period=period, zone=str(zone), season=season_label,
                               RL=float(rl_zone[mask].mean()))
                    for var, values in comp_zone.items():
                        row[labels[var]] = float(values[mask].mean())
                    records.append(row)
    return pd.DataFrame(records)


def compute_multi_day_table(
    residual_load_by_model: dict,
    components_by_model: dict,
    threshold_by_model: dict,
    period_defs: dict,
    zones,
    season_map: dict,
    season_order,
    *,
    method: str = "spa",
    min_duration_h: float = 24.0,
    component_labels: dict | None = None,
    attribution_seasons=("DJF", "JJA"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Multi-day events plus the WITHIN-SEASON component means used to attribute them.

    Returns ``(multi_day_events, multi_day_by_season)``.

    The attribution columns are named ``"{season}_{driver}"`` (and ``"{season}_RL"``) because the
    attribution *must* be done within season: warming shifts these events from winter to summer in
    the Mediterranean, so a season-pooled decomposition confounds the recomposition of the event
    population with the physical change.
    """
    from .plotting_extremes import resolve_model_periods

    labels = component_labels or PAPER_COMPONENT_LABELS
    event_rows, season_rows = [], []
    for model in residual_load_by_model:
        rl_all, comp_all = residual_load_by_model[model], components_by_model[model]
        windows = {k: v for k, v in resolve_model_periods(period_defs, model).items()
                   if v is not None}
        for period, (start, end) in windows.items():
            rl = rl_all.sel(time=slice(start, end))
            comp = comp_all.sel(time=slice(start, end))
            n_years = len(np.unique(rl["time"].dt.year.values))
            months = rl["time"].dt.month.values
            seasons = np.array([season_map[int(m)] for m in months])

            for zone in zones:
                threshold = float(threshold_by_model[model].sel(zone=zone).values)
                rl_zone = rl["residual_load"].sel(zone=zone).values
                events = [e for e in detect_events(rl_zone, threshold, method=method)
                          if e["duration_h"] > min_duration_h]
                durations = [e["duration_h"] for e in events]
                event_seasons = [season_map[int(months[e["indices"][0]])] for e in events]
                comp_zone = {v: comp[v].sel(zone=zone).values for v in labels if v in comp}

                # Which timesteps belong to a multi-day event, for the EV_ columns below.
                in_event = np.zeros(len(rl_zone), dtype=bool)
                for event in events:
                    in_event[event["indices"]] = True

                row = dict(model=model, period=period, zone=str(zone),
                           md_events_yr=len(durations) / n_years,
                           md_meandur=float(np.mean(durations)) if durations else np.nan)
                for season in attribution_seasons:
                    mask = seasons == season
                    row[f"{season}_RL"] = float(rl_zone[mask].mean())
                    for var, values in comp_zone.items():
                        row[f"{season}_{labels[var]}"] = float(values[mask].mean())

                    # The same means over the timesteps INSIDE a multi-day event. The unprefixed
                    # columns above are the seasonal background the events form against; these are
                    # the events themselves, and the difference of the two is what a multi-day
                    # event is made of. NaN, not zero, where the season carries no event: three of
                    # the six zones have no summer multi-day events in any chain at any warming
                    # level, and a zero there would draw an event with no anomaly.
                    event_mask = mask & in_event
                    present = bool(event_mask.any())
                    row[f"{season}_EV_steps"] = int(event_mask.sum())
                    row[f"{season}_EV_RL"] = float(rl_zone[event_mask].mean()) if present else np.nan
                    for var, values in comp_zone.items():
                        row[f"{season}_EV_{labels[var]}"] = (
                            float(values[event_mask].mean()) if present else np.nan)
                event_rows.append(row)

                for season in season_order:
                    season_rows.append(dict(
                        model=model, period=period, zone=str(zone), season=season,
                        md_events_yr=float(sum(1 for s in event_seasons if s == season)) / n_years,
                        md_hours_yr=float(np.sum([d for d, s in zip(durations, event_seasons)
                                                  if s == season])) / n_years))
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
    """PER-CHAIN driver contributions to the change in residual load (keeps the inter-model spread).

    Works with both attribution tables:

    * ``compute_evening_peak_attribution`` -- a ``season`` column, driver columns named directly
      (defaults);
    * ``compute_multi_day_table`` -- no ``season`` column, driver columns named
      ``"{season}_{driver}"`` (pass ``column_template="{season}_{driver}"``,
      ``net_column="{season}_RL"``).

    Demand enters with a ``+`` sign; every generation/storage term is negated, because a *drop* in
    generation raises the residual load.
    """
    rows = components[components.zone == zone]
    if "season" in rows.columns:
        rows = rows[rows.season == season]

    columns = {d: column_template.format(driver=d, season=season) for d in drivers}
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
    """PER-CHAIN composition of a multi-day event, relative to its own season's mean.

    The companion of :func:`attribution_change_by_model`, and the same shape (one column per driver
    plus ``net``), so the same panel builder draws both.  Where that one differences two warming
    levels over a fixed sample, this one differences two samples within one level: the timesteps
    inside a multi-day event against every timestep of the season.

    It answers "what makes an event an event" and NOT "what does warming change" — measured on this
    paper's run the two are different by one to two orders of magnitude, and the event anomaly is
    almost the same at +1.5 °C as at +3.0 °C (+189 against +186 GWh/3h in the British Isles in
    winter).  That is why the change panels decompose the seasonal background instead.

    ``scale`` divides every term, so a caller can normalise by the zone's deficit threshold and put
    zones of very different size on one axis; additivity is untouched because the divisor is shared.

    Rows where the season carries no multi-day event come back as NaN, which the panel draws as a
    gap.  Signs follow the change panels: demand enters with ``+``, generation and storage with
    ``-``, so a positive contribution deepens the deficit.
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
    period_defs: dict,
    zones,
    season_map: dict,
    *,
    method: str = "spa",
    max_duration_h: float | None = None,
    component_labels: dict | None = None,
) -> pd.DataFrame:
    """What an event is, once the time of day it happens at is controlled for.

    The companion of :func:`multi_day_event_anomaly_by_model`, which differences the event mean
    against the plain season mean.  That reference carries the diurnal cycle with it, so its answer
    mixes "the weather was anomalous" with "this happened in the evening, when the residual load is
    higher anyway".  This one subtracts, for every event timestep, the season mean **for that hour
    of day**, so the diurnal term is removed exactly however the event's steps happen to be spread.

    That distinction is not academic here: measured on the HIGH/2050 run, sub-daily winter events
    average 9-12 h — three or four 3-hourly steps — and only 46-61 % of their timesteps fall in the
    15-21 h evening window, the rest running on into the night.  An evening-window reference
    over-corrects those overnight steps; an hour-matched one cannot.

    Returns one row per (model, period, zone, season) with three blocks of columns, so a caller can
    read the exact additive split ``total = diurnal + weather``:

    ``{driver}``      the weather anomaly: event mean minus the hour-matched climatology
    ``TOT_{driver}``  the total anomaly: event mean minus the plain season mean
    ``DIU_{driver}``  the diurnal term: hour-matched climatology minus the season mean, averaged
                      over the event's own hours

    plus ``net``, ``TOT_net`` and ``DIU_net`` for the residual load itself, and ``n_steps``.
    Signs follow the other attribution helpers — demand enters ``+``, generation and storage ``-``
    — so a positive contribution deepens the deficit and
    :func:`~cordex2energy_analysis.plotting_publication.panel_driver_attribution` draws any of the
    three blocks unchanged.

    ``max_duration_h`` keeps only events no longer than that (24 for the sub-daily class); ``None``
    keeps every detected event.  A (zone, season) carrying no event yields NaN, which the panel
    draws as a gap.
    """
    from .plotting_extremes import resolve_model_periods

    labels = component_labels or PAPER_COMPONENT_LABELS
    rows = []
    for model in residual_load_by_model:
        rl_all, comp_all = residual_load_by_model[model], components_by_model[model]
        windows = {k: v for k, v in resolve_model_periods(period_defs, model).items()
                   if v is not None}
        for period, (start, end) in windows.items():
            rl = rl_all.sel(time=slice(start, end))
            comp = comp_all.sel(time=slice(start, end))
            months = rl["time"].dt.month.values
            hours = rl["time"].dt.hour.values
            seasons = np.array([season_map[int(m)] for m in months])

            for zone in zones:
                threshold = float(threshold_by_model[model].sel(zone=zone).values)
                rl_zone = rl["residual_load"].sel(zone=zone).values
                events = [e for e in detect_events(rl_zone, threshold, method=method)
                          if max_duration_h is None or e["duration_h"] <= max_duration_h]
                in_event = np.zeros(len(rl_zone), dtype=bool)
                for event in events:
                    in_event[event["indices"]] = True
                series = {"net": rl_zone}
                series.update({labels[v]: comp[v].sel(zone=zone).values
                               for v in labels if v in comp})

                for season in sorted(set(seasons)):
                    in_season = seasons == season
                    mask = in_event & in_season
                    # `evening_share` is the evidence for the hour-matched reference: it is the
                    # fraction of the event's own timesteps falling in the evening window, and it
                    # is why an evening-window reference is the wrong control when it is not ~1.
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
                        # The climatology matched to each event step's own hour. Computed ONCE per
                        # hour and then indexed: done per event step it is O(steps x season length),
                        # which over nine chains, three levels and six zones is ~10^9 operations.
                        clim = {h: float(values[in_season & (hours == h)].mean())
                                for h in np.unique(hours)}
                        matched = np.array([clim[h] for h in hours[mask]])
                        row[name] = sign * float((values[mask] - matched).mean())
                        row[f"TOT_{name}"] = sign * float(values[mask].mean() - season_mean)
                        row[f"DIU_{name}"] = sign * float(matched.mean() - season_mean)
                    rows.append(row)
    return pd.DataFrame(rows)


def compute_deficit_budget(
    residual_load_by_model: dict,
    threshold_by_model: dict,
    period_defs: dict,
    zones,
    *,
    method: str = "spa",
    sub_daily_max_h: float = 24.0,
) -> tuple[pd.DataFrame, dict]:
    """How the annual deficit budget splits between the sub-daily and multi-day timescales.

    Returns ``(deficit_budget, diurnal_profiles)``.  ``diurnal_profiles[period]`` is a list of
    Series (indexed by hour, named ``"model|zone"``) giving the residual load divided by the zone
    threshold on *stressed days* -- calendar days carrying at least one exceedance.  That ratio is
    what shows the midday solar trough and the evening peak.
    """
    from .plotting_extremes import resolve_model_periods

    rows = []
    diurnal: dict[str, list] = {period: [] for period in period_defs}
    for model in residual_load_by_model:
        rl_all = residual_load_by_model[model]
        windows = {k: v for k, v in resolve_model_periods(period_defs, model).items()
                   if v is not None}
        for period, (start, end) in windows.items():
            rl = rl_all.sel(time=slice(start, end))
            n_years = len(np.unique(rl["time"].dt.year.values))
            hours = rl["time"].dt.hour.values
            days = rl["time"].dt.floor("D").values

            for zone in zones:
                threshold = float(threshold_by_model[model].sel(zone=zone).values)
                values = rl["residual_load"].sel(zone=zone).values
                exceeds = values > threshold
                events = detect_events(values, threshold, method=method)
                durations = (np.array([e["duration_h"] for e in events]) if events
                             else np.array([]))
                is_sub = (durations > 0) & (durations <= sub_daily_max_h)
                is_multi = durations > sub_daily_max_h

                frame = pd.DataFrame({"day": days, "hour": hours, "rl": values, "exc": exceeds})
                stressed = frame[frame.groupby("day")["exc"].transform("any")]
                if len(stressed):
                    diurnal[period].append(
                        (stressed.groupby("hour")["rl"].mean() / threshold)
                        .rename(f"{model}|{zone}"))

                rows.append(dict(
                    model=model, period=period, zone=str(zone),
                    sub_events_yr=int(is_sub.sum()) / n_years,
                    multi_events_yr=int(is_multi.sum()) / n_years,
                    sub_hours_yr=float(durations[is_sub].sum()) / n_years,
                    multi_hours_yr=float(durations[is_multi].sum()) / n_years))
    return pd.DataFrame(rows), diurnal


def _season_values_by_period(residual_load_by_model, period_defs, zone, months, period_names,
                             *, sort: bool = True) -> dict:
    """Per-chain residual-load values for one zone, restricted to ``months``, for each period."""
    out = {}
    for period in period_names:
        per_model = []
        for model, dataset in residual_load_by_model.items():
            window = period_defs[period].get(model)
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
    period_defs: dict,
    zone: str,
    month_lists,
    period_names,
    *,
    threshold: float | None = None,
    n_points: int = 220,
) -> np.ndarray:
    """One x grid for a zone, pooled over SEVERAL month lists (i.e. across the displayed seasons).

    Giving both seasons of a zone the same x-axis is what makes them comparable: the threshold then
    sits at the same place in each, and a season whose distribution stops short of the threshold is
    visibly a season with no exceedance -- no annotation required.

    ``threshold``, if given, is kept inside the grid so the deficit level is always in frame even
    where it lies above the whole distribution.
    """
    pooled = []
    for months in month_lists:
        by_period = _season_values_by_period(residual_load_by_model, period_defs, zone, months,
                                             period_names, sort=False)
        pooled.extend(v for per_model in by_period.values() for v in per_model)
    pooled = np.concatenate(pooled)

    low = float(np.percentile(pooled, 40))
    high = float(np.percentile(pooled, 99.95))
    if threshold is not None:
        high = max(high, threshold * 1.02)
    return np.linspace(low, high, n_points)


def compute_exceedance_survival(
    residual_load_by_model: dict,
    period_defs: dict,
    zone: str,
    months,
    period_names,
    *,
    spread: str = "iqr",
    n_points: int = 220,
    x: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """Survival curve ``P(RL > x)`` per chain -> ensemble median + inter-model band.

    A density plot cannot show a change confined to the top few per cent of the distribution; on a
    log survival axis the warming levels fan out exactly where the events live.

    Pass ``x`` (from :func:`compute_survival_x_grid`) to evaluate several seasons of a zone on one
    shared grid, so their panels can be compared directly.
    """
    from .plotting_extremes import model_spread_bounds

    sorted_values = _season_values_by_period(residual_load_by_model, period_defs, zone, months,
                                             period_names)

    if x is None:
        pooled = np.concatenate([np.concatenate(v) for v in sorted_values.values()])
        x = np.linspace(np.percentile(pooled, 40), np.percentile(pooled, 99.95), n_points)

    curves = {}
    for period in period_names:
        survival = np.vstack([1.0 - np.searchsorted(v, x, side="right") / len(v)
                              for v in sorted_values[period]])
        curves[period] = (np.median(survival, axis=0),
                          *model_spread_bounds(survival, spread, axis=0))
    return x, curves


def compute_exceedance_probability(
    residual_load_by_model: dict,
    period_defs: dict,
    zone: str,
    months,
    period: str,
    threshold: float,
) -> float:
    """Ensemble-median ``P(RL > threshold)``.

    Evaluated exactly at the threshold rather than at the nearest grid point of a survival curve, so
    a threshold lying outside the plotted range still returns the right (zero) answer.
    """
    probabilities = []
    for model, dataset in residual_load_by_model.items():
        window = period_defs[period].get(model)
        if window is None:
            continue
        data = dataset["residual_load"].sel(zone=zone, time=slice(str(window[0]), str(window[1])))
        in_season = np.isin(data["time"].dt.month.values, list(months))
        values = data.values[in_season]
        values = values[np.isfinite(values)]
        probabilities.append(float(np.mean(values > threshold)))
    return float(np.median(probabilities))


def compute_country_gwl_field(
    country_output_by_model: dict,
    period_defs: dict,
    period: str,
    expression,
    *,
    scale: float = 1e-3,
    months: "list[int] | None" = None,
) -> "xr.DataArray":
    """PER-CHAIN mean total of a country-level quantity over each chain's GWL window.

    ``expression(dataset) -> DataArray(country, time)`` in GWh per timestep; ``scale`` converts the
    total (default GWh -> TWh).  Returns ``(model, country)``.

    With ``months=None`` (default) this is the mean ANNUAL total.  Pass a list of month numbers (e.g.
    ``[12, 1, 2]`` for DJF) to restrict to a SEASON: the result is then the mean *seasonal* total
    (the per-year sum over those months, averaged over the window's years).
    """
    per_model = []
    for model, dataset in country_output_by_model.items():
        window = period_defs[period].get(model)
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
    period_defs: dict,
    period: str,
    variable: str,
    *,
    scale: float = 1e-3,
    months: "list[int] | None" = None,
) -> "xr.DataArray":
    """As :func:`compute_country_gwl_field`, for the c2e demand cache (``(model, country, time)``).

    ``months`` restricts to a season (mean seasonal total) exactly as in
    :func:`compute_country_gwl_field`; ``None`` gives the mean annual total.
    """
    per_model = []
    for model in period_defs[period]:
        window = period_defs[period].get(model)
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
    """Turn structural zeros into NaN.

    A country with no capacity for a technology stores 0, which on a sequential colour scale reads
    as "low" rather than "absent".  NaN makes it grey.
    """
    return field.where(field != 0)


def compute_annual_operation_curves(
    components_by_model: dict,
    period_defs: dict,
    period: str,
    variable: str,
    selection: dict,
    *,
    mode: str = "sum",
    smooth_days: int = 7,
) -> "np.ndarray":
    """Day-of-year operating curves over a GWL window — one curve per (chain, year).

    Returns ``(n_curves, 365)``: every chain-year in the window, on a common day-of-year axis, ready
    to draw as spaghetti or reduce to a median + percentile band.

    ``selection`` picks the panel's slice, e.g. ``{"country": "NO"}`` or ``{"zone": "Nordic"}``.

    ``mode``:
      * ``"sum"``   — daily total (reservoir dispatch, GWh/day).
      * ``"mean"``  — daily mean (reservoir storage, GWh).
      * ``"throughput"`` — daily energy cycled, ``sum(|x|)/2``. **The only meaningful daily metric for
        a battery**: its net daily dispatch is just the round-trip loss (~412 GWh/day cycled nets
        ~-69 GWh/day at 85 % efficiency), so a net-dispatch curve shows the loss term, not operation.

    Leap days are dropped (curves are truncated to 365) so every year shares one axis.
    """
    import pandas as _pd

    curves = []
    for model, dataset in components_by_model.items():
        window = period_defs[period].get(model)
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
            if len(values) < 365:          # partial year at a window edge
                continue
            values = values[:365]
            if smooth_days > 1:
                values = (_pd.Series(values)
                          .rolling(smooth_days, center=True, min_periods=1).mean().values)
            curves.append(values)
    return np.array(curves)


def summarise_event_frequency(
    events: pd.DataFrame,
    duration_class: str,
    zones,
    period_names,
    n_models: int,
) -> pd.DataFrame:
    """Headline table: events/yr per zone and period, with the % change and the chain agreement."""
    subset = events[events.duration_class == duration_class]
    table = (subset.pivot_table(index="zone", columns="period", values="freq_per_year",
                                aggfunc="median")
             .reindex(index=list(zones))[list(period_names)])
    out = table.round(1).copy()

    reference = period_names[0]
    for period in period_names[1:]:
        changes, agreements = [], []
        for zone in zones:
            per_zone = subset[subset.zone == zone]
            ref = per_zone[per_zone.period == reference].set_index("model")["freq_per_year"]
            warm = per_zone[per_zone.period == period].set_index("model")["freq_per_year"]
            base = table.loc[zone, reference]
            changes.append(100 * (table.loc[zone, period] - base) / base if base > 0 else np.nan)
            delta = (warm - ref).reindex(ref.index).dropna()
            sign = np.sign(np.median(delta))
            agreements.append(int((np.sign(delta) == sign).sum()) if sign != 0 else 0)
        out[f"%chg {period}"] = np.round(changes, 0)
        out[f"agree {period}"] = [f"{a}/{n_models}" for a in agreements]
    return out


def compute_zone_season_change(
    events_by_season: pd.DataFrame,
    zones,
    season_order,
    period_names,
    *,
    spread: str = "iqr",
) -> dict:
    """% change in seasonal event count per (zone, season), with spread and chain agreement.

    Returns ``{duration_class: {period: (median, low, high, n_agree)}}``, each a (zone, season)
    array.  A cell is NaN where there were **no events at the reference level** -- that is a result
    (the zone has no deficit problem in that season), not a gap.
    """
    from .plotting_extremes import model_spread_bounds

    reference = period_names[0]
    shape = (len(zones), len(season_order))
    out: dict = {}

    for duration_class in events_by_season["duration_class"].unique():
        subset = events_by_season[events_by_season.duration_class == duration_class]
        out[duration_class] = {}
        for period in period_names[1:]:
            median = np.full(shape, np.nan)
            low = np.full(shape, np.nan)
            high = np.full(shape, np.nan)
            n_agree = np.zeros(shape)
            for i, zone in enumerate(zones):
                for j, season in enumerate(season_order):
                    cell = subset[(subset.zone == zone) & (subset.season == season)]
                    ref = cell[cell.period == reference].set_index("model")["n_events"]
                    warm = cell[cell.period == period].set_index("model")["n_events"]
                    common = ref.index.intersection(warm.index)
                    if len(common) == 0 or ref.loc[common].sum() == 0:
                        continue
                    change = (100 * (warm.loc[common] - ref.loc[common])
                              / ref.loc[common].replace(0, np.nan))
                    change = change.replace([np.inf, -np.inf], np.nan).dropna()
                    if change.empty:
                        continue
                    median[i, j] = change.median()
                    low[i, j], high[i, j] = model_spread_bounds(change.values, spread, axis=0)
                    n_agree[i, j] = (np.sign(change) == np.sign(change.median())).sum()
            out[duration_class][period] = (median, low, high, n_agree)
    return out


def compute_event_significance(
    events_by_season: pd.DataFrame,
    zones,
    season_order,
    period_names,
) -> pd.DataFrame:
    """Paired Wilcoxon signed-rank test on the per-chain seasonal event counts, plus agreement."""
    from scipy.stats import wilcoxon

    reference = period_names[0]
    rows = []
    for duration_class in events_by_season["duration_class"].unique():
        subset = events_by_season[events_by_season.duration_class == duration_class]
        for period in period_names[1:]:
            for zone in zones:
                for season in season_order:
                    cell = subset[(subset.zone == zone) & (subset.season == season)]
                    ref = cell[cell.period == reference].set_index("model")["n_events"]
                    warm = cell[cell.period == period].set_index("model")["n_events"]
                    common = ref.index.intersection(warm.index)
                    x, y = ref.loc[common].values, warm.loc[common].values
                    delta = y - x
                    if len(delta) < 5 or np.all(delta == 0):
                        pvalue = np.nan
                    else:
                        try:
                            pvalue = wilcoxon(x, y, zero_method="zsplit").pvalue
                        except Exception:
                            pvalue = np.nan
                    sign = np.sign(np.median(delta))
                    rows.append(dict(
                        duration_class=duration_class,
                        comparison=f"{period} vs {reference}",
                        zone=zone, season=season, p=pvalue,
                        agree=(np.sign(delta) == sign).sum() / max(len(delta), 1)))
    return pd.DataFrame(rows)


def compute_threshold_sensitivity(
    residual_load_by_model: dict,
    period_defs: dict,
    zones,
    season_order,
    season_months: dict,
    zone_threshold,
    seasonal_thresholds: dict,
    *,
    method: str = "spa",
    sub_daily_max_h: float = 24.0,
) -> pd.DataFrame:
    """Event rates under the single annual threshold vs a season-specific one.

    The seasonal masking is held fixed and only the threshold varies, so the comparison isolates the
    threshold choice.  A season-specific p95 forces ~5 % exceedance in *every* season of the
    reference climate by construction: it re-baselines each season to its own tail (erasing the
    winter->summer migration) and manufactures events in seasons that have no deficit problem.
    """
    from .plotting_extremes import resolve_model_periods

    rows = []
    for model, dataset in residual_load_by_model.items():
        windows = {k: v for k, v in resolve_model_periods(period_defs, model).items()
                   if v is not None}
        for period, (start, end) in windows.items():
            data = dataset.sel(time=slice(start, end))
            times = pd.DatetimeIndex(data["time"].values)
            n_years = len(np.unique(times.year))
            for zone in zones:
                values = data["residual_load"].sel(zone=zone).values
                for season in season_order:
                    schemes = {
                        "annual": float(zone_threshold.sel(zone=zone)),
                        "seasonal": float(seasonal_thresholds[str(zone)][season]),
                    }
                    for scheme, threshold in schemes.items():
                        events = detect_seasonal_events(
                            values, times, threshold, season_months[season],
                            time_step_h=3.0, method=method)
                        durations = (np.array([e["duration_h"] for e in events]) if events
                                     else np.array([]))
                        rows.append(dict(
                            model=model, period=period, zone=str(zone), season=season,
                            scheme=scheme,
                            sub_daily=int((durations <= sub_daily_max_h).sum()) / n_years,
                            multi_day=int((durations > sub_daily_max_h).sum()) / n_years))
    return pd.DataFrame(rows)
