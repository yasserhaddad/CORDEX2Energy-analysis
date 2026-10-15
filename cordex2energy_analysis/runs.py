"""Loading the residual-load runs and the tables built from them.

A *run* is one residual-load simulation (one TYNDP build-out and flexibility setup, identified by
its run tag) for every climate-model chain. Each chain's output lives in
``<residual_load>/<chain>_<scenario>/<run tag>/``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from cordex2energy.utils.constants import REFERENCE_PERIOD

from .analysis import resolve_model_periods
from .config import (CLIMATE_SCENARIO, DURATION_BINS, EVENT_DETECTION, RUN_TAG, SEASON_MAP,
                     SEASON_MONTHS, SEASON_ORDER, THRESHOLD_PERCENTILE, ZONE_ORDER)
from .extremes_analysis import (DEFICIT_WINDOW_HOURS, compute_annual_operation_curves,
                                compute_duration_stratified_metrics,
                                compute_evening_peak_attribution, compute_event_season_metrics,
                                compute_flexibility_attenuation_metrics, compute_multi_day_table)
from .extremes_events import calculate_climatological_baseline, compute_continuous_system_metrics
from .extremes_thresholds import compute_active_threshold
from .figure_cache import cached


def run_files(context, filename: str, run_tag: str = RUN_TAG) -> dict[str, Path]:
    """Path of one output file of a run, for every chain.

    Parameters
    ----------
    context : PaperContext
        Supplies the data paths and the chains.
    filename : str
        File inside the run folder, e.g. ``"components.nc"``.
    run_tag : str
        The run.

    Returns
    -------
    dict
        ``{chain: path}``.
    """
    return {model: context.paths.residual_load / f"{model}_{CLIMATE_SCENARIO}" / run_tag / filename
            for model in context.models}


def operation_curves(components_by_model: dict, period_windows: dict, variable: str,
                     dimension: str, keys, mode: str, smooth_days: int = 7) -> dict:
    """Annual operation cycles of one variable, for several countries or zones.

    Parameters
    ----------
    components_by_model : dict
        ``{chain: xarray.Dataset}`` of run components.
    period_windows : dict
        ``{warming level: {chain: (first year, last year)}}``.
    variable : str
        Component to draw, e.g. ``"reservoir_storage_gwh"``.
    dimension : str
        ``"country"`` or ``"zone"``.
    keys : sequence of str
        Countries or zones, one panel each.
    mode : str
        Passed to :func:`~cordex2energy_analysis.extremes_analysis.compute_annual_operation_curves`.
    smooth_days : int
        Width of the centred running mean, in days.

    Returns
    -------
    dict
        ``{key: {warming level: array (chain-years, 365)}}``.
    """
    return {key: {period: compute_annual_operation_curves(components_by_model, period_windows,
                                                          period, variable, {dimension: key},
                                                          mode=mode, smooth_days=smooth_days)
                  for period in period_windows}
            for key in keys}


def reservoir_operation_curves(context, run_tag: str = RUN_TAG):
    """Daily reservoir storage and dispatch over the year, per country with a reservoir fleet.

    Countries, not zones: a zonal sum would hide Norway inside the Nordic zone. Cached on disk.

    Parameters
    ----------
    context : PaperContext
        Data paths, chains and warming-level windows.
    run_tag : str
        The run.

    Returns
    -------
    countries : list of str
        Countries (ISO2) whose reservoirs dispatch at all.
    storage : dict
        Storage curves (GWh), see :func:`operation_curves`; ``mode="mean"``, no smoothing.
    dispatch : dict
        Dispatch curves (GWh per day), ``mode="sum"``, 7-day smoothing.
    """
    files = run_files(context, "components.nc", run_tag)

    def compute():
        """Reservoir storage and dispatch curves of the countries with a reservoir."""
        components = {model: xr.open_dataset(path) for model, path in files.items()}
        probe = components[context.models[0]]
        countries = [str(country) for country in probe["country"].values
                     if float(probe["reservoir_dispatch_gwh"].sel(country=country).sum()) > 0]
        storage = operation_curves(components, context.period_windows, "reservoir_storage_gwh",
                                   "country", countries, mode="mean", smooth_days=1)
        dispatch = operation_curves(components, context.period_windows, "reservoir_dispatch_gwh",
                                    "country", countries, mode="sum")
        return countries, storage, dispatch

    return cached("note2_reservoir_curves", compute,
                  key=dict(run_tag=run_tag, models=context.models, periods=context.period_windows),
                  files=files.values())


def battery_throughput_curves(context, run_tag: str = RUN_TAG) -> dict:
    """Energy cycled through the batteries each day of the year, per balancing zone.

    Parameters
    ----------
    context : PaperContext
        Data paths, chains and warming-level windows.
    run_tag : str
        The run.

    Returns
    -------
    dict
        ``{zone: {warming level: array (chain-years, 365)}}`` in GWh per day.
    """
    zonal_components = {model: xr.open_dataset(path)
                        for model, path in run_files(context, "components_zonal.nc", run_tag).items()}
    return operation_curves(zonal_components, context.period_windows, "battery_dispatch_gwh",
                            "zone", ZONE_ORDER, mode="throughput")


# ---------------------------------------------------------------------------
# The residual-load run and its event tables
# ---------------------------------------------------------------------------

@dataclass
class ResidualLoadRun:
    """One residual-load run for every chain, with the threshold its events are scored against.

    Attributes
    ----------
    run_tag : str
        The run.
    models : list of str
        Chains with output for this run.
    zones : list of str
        Balancing zones, in the paper's order.
    residual_load : dict
        ``{chain: Dataset}`` of the zonal residual load (GWh per 3h step), on the time steps it
        shares with the components.
    components : dict
        ``{chain: Dataset}`` of the zonal supply, demand and storage components.
    metrics : dict
        ``{chain: Dataset}`` of the continuous system metrics (anomalies against the 1991-2020
        climatology of each chain).
    threshold : xarray.DataArray
        Deficit threshold per zone (GWh per 3h step).
    cerra_deficit : xarray.DataArray or None
        The CERRA positive residual load the threshold was calibrated on, if it was.
    files : list of pathlib.Path
        Every input file read, for cache keys.
    """

    run_tag: str
    models: list
    zones: list
    residual_load: dict = field(repr=False)
    components: dict = field(repr=False)
    metrics: dict = field(repr=False)
    threshold: xr.DataArray = field(repr=False)
    cerra_deficit: xr.DataArray | None = field(default=None, repr=False)
    files: list = field(default_factory=list, repr=False)

    @property
    def threshold_by_model(self) -> dict:
        """The same threshold for every chain, the form the event functions take."""
        return {model: self.threshold for model in self.models}


def cerra_threshold(context, run_tag: str = RUN_TAG,
                    percentile: float = THRESHOLD_PERCENTILE) -> tuple[xr.DataArray, xr.DataArray]:
    """Deficit threshold: a percentile of the positive CERRA residual load, per zone.

    Parameters
    ----------
    context : PaperContext
        Supplies the data paths.
    run_tag : str
        The run whose CERRA pass calibrates the threshold.
    percentile : float
        Percentile of the positive residual load.

    Returns
    -------
    threshold : xarray.DataArray
        Threshold per zone (GWh per 3h step).
    cerra_deficit : xarray.DataArray
        The positive CERRA residual load (other time steps NaN).
    """
    cerra_dir = context.paths.residual_load / "cerra" / run_tag
    residual_load = xr.open_dataset(cerra_dir / "residual_load_zonal.nc")
    components = xr.open_dataset(cerra_dir / "components_zonal.nc")
    shared_time = np.intersect1d(residual_load.time.values, components.time.values)
    residual_load, components = residual_load.sel(time=shared_time), components.sel(time=shared_time)
    metrics = compute_continuous_system_metrics(
        residual_load, components, calculate_climatological_baseline(components))
    cerra_deficit = metrics["residual_load"].where(metrics["residual_load"] > 0)
    threshold = compute_active_threshold(cerra_deficit, "PERCENTILE", percentile=percentile)
    return threshold, cerra_deficit


def load_run(context, run_tag: str = RUN_TAG, threshold: xr.DataArray | None = None
             ) -> ResidualLoadRun:
    """Load a residual-load run for every chain and prepare what the event statistics need.

    Parameters
    ----------
    context : PaperContext
        Data paths and chains.
    run_tag : str
        The run.
    threshold : xarray.DataArray, optional
        Threshold per zone to score the run against. By default the run's own CERRA p95
        (:func:`cerra_threshold`): right when the system itself differs. Pass another run's
        threshold to judge two systems by one yardstick.

    Returns
    -------
    ResidualLoadRun
        The chains with output for the run, their zonal series and system metrics, and the
        threshold.
    """
    residual_load_files = run_files(context, "residual_load_zonal.nc", run_tag)
    component_files = run_files(context, "components_zonal.nc", run_tag)
    models = [model for model in context.models if residual_load_files[model].exists()]
    residual_load_all = xr.concat(
        [xr.open_dataset(residual_load_files[model]).expand_dims(model=[model]) for model in models],
        dim="model", join="outer")
    components_all = xr.concat(
        [xr.open_dataset(component_files[model]).expand_dims(model=[model]) for model in models],
        dim="model", join="outer")
    zones = [zone for zone in ZONE_ORDER
             if zone in [str(name) for name in residual_load_all["zone"].values]]

    cerra_deficit = None
    if threshold is None:
        threshold, cerra_deficit = cerra_threshold(context, run_tag)

    residual_load, components, metrics = {}, {}, {}
    for model in models:
        chain_residual_load = (residual_load_all.sel(model=model).drop_vars("model", errors="ignore")
                               .dropna("time", how="all"))
        chain_components = (components_all.sel(model=model).drop_vars("model", errors="ignore")
                            .dropna("time", how="all"))
        shared_time = np.intersect1d(chain_residual_load.time.values, chain_components.time.values)
        residual_load[model] = chain_residual_load.sel(time=shared_time)
        components[model] = chain_components.sel(time=shared_time)
        metrics[model] = compute_continuous_system_metrics(
            residual_load[model], components[model],
            calculate_climatological_baseline(components[model].sel(time=slice(*REFERENCE_PERIOD))))

    cerra_dir = context.paths.residual_load / "cerra" / run_tag
    files = ([residual_load_files[model] for model in models]
             + [component_files[model] for model in models]
             + [cerra_dir / "residual_load_zonal.nc", cerra_dir / "components_zonal.nc"])
    return ResidualLoadRun(run_tag=run_tag, models=models, zones=zones,
                           residual_load=residual_load, components=components, metrics=metrics,
                           threshold=threshold, cerra_deficit=cerra_deficit, files=files)


def event_tables(run: ResidualLoadRun, context) -> dict:
    """Every event, attribution and storage table the figures are drawn from. Cached on disk.

    One detection pass per table; each event is assigned whole to the season holding most of its
    time steps, so the four seasonal counts sum to the annual count exactly.

    Parameters
    ----------
    run : ResidualLoadRun
        The run, with its threshold.
    context : PaperContext
        Warming-level windows.

    Returns
    -------
    dict
        ``events_annual`` (events per chain, level, zone and duration class),
        ``events_by_season`` (the same split by season, with ``hours_yr`` = event hours per year),
        ``storage_attenuation`` and ``storage_events`` (severity balanced by storage, per cell and
        per event), ``evening_peak_components`` (driver anomalies over the deficit window),
        ``multi_day_events`` and ``multi_day_by_season`` (multi-day event composition).
    """
    periods = context.period_windows

    def compute():
        """Every event table of the run, one detection pass each."""
        events_annual = compute_duration_stratified_metrics(
            run.residual_load, run.metrics, run.threshold_by_model, periods, run.models, run.zones,
            method=EVENT_DETECTION, duration_bins=DURATION_BINS)
        events_by_season = compute_event_season_metrics(
            run.residual_load, run.threshold_by_model, periods, run.models, run.zones,
            SEASON_MAP, SEASON_ORDER, method=EVENT_DETECTION, duration_bins=DURATION_BINS)
        storage_attenuation, storage_events = compute_flexibility_attenuation_metrics(
            run.residual_load, run.components, run.threshold_by_model, periods, run.models,
            run.zones, duration_bins=DURATION_BINS, method=EVENT_DETECTION, return_events=True)
        evening_peak_components = compute_evening_peak_attribution(
            run.residual_load, run.components, periods, run.zones, SEASON_MAP,
            hours=DEFICIT_WINDOW_HOURS)
        multi_day_events, multi_day_by_season = compute_multi_day_table(
            run.residual_load, run.components, run.threshold_by_model, periods, run.zones,
            SEASON_MAP, SEASON_ORDER, method=EVENT_DETECTION, attribution_seasons=SEASON_ORDER)
        return dict(events_annual=events_annual, events_by_season=events_by_season,
                    storage_attenuation=storage_attenuation, storage_events=storage_events,
                    evening_peak_components=evening_peak_components,
                    multi_day_events=multi_day_events, multi_day_by_season=multi_day_by_season)

    tables = cached(
        "event_tables", compute,
        key=dict(run_tag=run.run_tag, models=run.models, zones=run.zones, threshold=run.threshold,
                 periods=periods, reference=REFERENCE_PERIOD, method=EVENT_DETECTION,
                 durations=DURATION_BINS, seasons=SEASON_MAP, season_order=SEASON_ORDER,
                 window=DEFICIT_WINDOW_HOURS),
        files=run.files)
    events_by_season = tables["events_by_season"]
    tables["events_by_season"] = events_by_season.assign(
        hours_yr=events_by_season["freq_per_year"] * events_by_season["avg_duration_h"])
    return tables


def annual_event_counts(run: ResidualLoadRun, context) -> pd.DataFrame:
    """Events per chain, warming level, zone and duration class, scored on the run's threshold.

    Parameters
    ----------
    run : ResidualLoadRun
        The run, with its threshold.
    context : PaperContext
        Warming-level windows.

    Returns
    -------
    pandas.DataFrame
        As ``events_annual`` of :func:`event_tables`.
    """
    return compute_duration_stratified_metrics(
        run.residual_load, run.metrics, run.threshold_by_model, context.period_windows,
        run.models, run.zones, method=EVENT_DETECTION, duration_bins=DURATION_BINS)


def exceedance_by_hour(run: ResidualLoadRun, context) -> pd.DataFrame:
    """Hours per year above the threshold, by chain, level, zone, season and hour of day.

    Detector-free: every time step above the zone's threshold counts, whether or not the event
    detector would call it part of an event.

    Parameters
    ----------
    run : ResidualLoadRun
        The run, with its threshold.
    context : PaperContext
        Warming-level windows.

    Returns
    -------
    pandas.DataFrame
        Columns ``model, period, zone, season, hour, hours_yr``.
    """
    rows = []
    for model in run.models:
        series = run.residual_load[model]["residual_load"]
        for period in context.period_names:
            window = resolve_model_periods(context.period_windows, model).get(period)
            if window is None:
                continue
            chunk = series.sel(time=slice(*window))
            times = pd.DatetimeIndex(chunk["time"].values)
            years = len(np.unique(times.year))
            for zone in run.zones:
                above = chunk.sel(zone=zone).values > float(run.threshold.sel(zone=zone))
                for season in SEASON_ORDER:
                    in_season = np.isin(times.month, SEASON_MONTHS[season])
                    for hour in np.unique(times.hour):
                        at_hour = times.hour == hour
                        rows.append(dict(model=model, period=period, zone=zone, season=season,
                                         hour=int(hour),
                                         hours_yr=3.0 * int((above & in_season & at_hour).sum())
                                         / years))
    return pd.DataFrame(rows)


def pooled_exceedance_by_hour(hour_exceedance: pd.DataFrame, zones) -> pd.DataFrame:
    """Exceedance hours of several zones summed per chain (before any median is taken).

    Parameters
    ----------
    hour_exceedance : pandas.DataFrame
        From :func:`exceedance_by_hour`.
    zones : sequence of str
        Zones to pool, e.g. the two southern ones.

    Returns
    -------
    pandas.DataFrame
        Columns ``model, period, season, hour, hours_yr``.
    """
    return (hour_exceedance[hour_exceedance.zone.isin(zones)]
            .groupby(["model", "period", "season", "hour"], as_index=False)["hours_yr"].sum())


def positive_residual_load_share(run: ResidualLoadRun, context, label: str) -> pd.DataFrame:
    """Annual positive residual load, as energy and as a share of annual demand.

    Only positive (deficit) time steps count: surplus hours would otherwise net the deficit away.

    Parameters
    ----------
    run : ResidualLoadRun
        The run.
    context : PaperContext
        Warming-level windows.
    label : str
        Scenario label written into the ``scenario`` column.

    Returns
    -------
    pandas.DataFrame
        Columns ``scenario, model, period, zone, twh, demand_twh, share_of_demand_pct`` for the
        reference and the warmest level.
    """
    rows = []
    for model, dataset in run.residual_load.items():
        windows = resolve_model_periods(context.period_windows, model)
        for period in (context.reference, context.warmest):
            window = windows.get(period)
            if window is None:
                continue
            for zone in run.zones:
                block = dataset["residual_load"].sel(zone=zone, time=slice(*window))
                demand = run.components[model]["total_demand"].sel(zone=zone, time=slice(*window))
                years = pd.DatetimeIndex(block["time"].values).year.nunique()
                values = block.values
                positive = values[np.isfinite(values) & (values > 0)]
                energy = positive.sum() / years / 1e3          # energy per 3h step: plain sum
                total = float(np.nansum(demand.values)) / years / 1e3
                rows.append(dict(scenario=label, model=model, period=period, zone=zone,
                                 twh=energy, demand_twh=total,
                                 share_of_demand_pct=energy / total * 100))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Threshold diagnostics and event anomalies (Supplementary Note 3)
# ---------------------------------------------------------------------------

def reference_period_thresholds(run: ResidualLoadRun, context,
                                percentile: float = THRESHOLD_PERCENTILE) -> tuple[dict, dict]:
    """Each chain's system metrics in the reference window, and the threshold it would imply.

    Parameters
    ----------
    run : ResidualLoadRun
        The run.
    context : PaperContext
        Warming-level windows.
    percentile : float
        Percentile of the positive residual load.

    Returns
    -------
    reference_metrics : dict
        ``{chain: Dataset}`` restricted to the chain's +1.5 °C window.
    chain_thresholds : dict
        ``{chain: DataArray}`` of the per-zone threshold calibrated on that window.
    """
    reference_metrics, chain_thresholds = {}, {}
    for model in run.models:
        window = context.period_windows[context.reference][model]
        metrics = run.metrics[model].sel(time=slice(str(window[0]), str(window[1])))
        reference_metrics[model] = metrics
        deficit = metrics["residual_load"].where(metrics["residual_load"] > 0)
        chain_thresholds[model] = compute_active_threshold(deficit, "PERCENTILE",
                                                           percentile=percentile)
    return reference_metrics, chain_thresholds


def cerra_system_metrics(context, run_tag: str = RUN_TAG) -> xr.Dataset:
    """System metrics of the CERRA pass, over its full record.

    Parameters
    ----------
    context : PaperContext
        Data paths.
    run_tag : str
        The run.

    Returns
    -------
    xarray.Dataset
        Continuous system metrics, anomalies against the full-record climatology.
    """
    cerra_dir = context.paths.residual_load / "cerra" / run_tag
    residual_load = xr.open_dataset(cerra_dir / "residual_load_zonal.nc")
    components = xr.open_dataset(cerra_dir / "components_zonal.nc")
    return compute_continuous_system_metrics(residual_load, components,
                                             calculate_climatological_baseline(components))


def reservoir_hydro_by_season(run: ResidualLoadRun, context, zone: str) -> pd.DataFrame:
    """Reservoir inflow and dispatch of one zone per chain, level and season, per year.

    Parameters
    ----------
    run : ResidualLoadRun
        The run (supplies the dispatch).
    context : PaperContext
        Data paths and warming-level windows (the inflow comes from the hydro-inflow cache).
    zone : str
        Balancing zone.

    Returns
    -------
    pandas.DataFrame
        Columns ``model, period, season, quantity ("inflow" | "dispatch"), per_year`` (GWh).
    """
    from .power_system import REALTIME_BALANCING_ZONES

    inflow = xr.open_dataarray(context.paths.gwl_grids / "c2e_cache"
                               / f"c2e_hydro_inflow_{CLIMATE_SCENARIO}.nc")
    countries = [country for country in REALTIME_BALANCING_ZONES[zone]
                 if country in {str(name) for name in inflow["country"].values}]
    rows = []
    for model in run.models:
        windows = resolve_model_periods(context.period_windows, model)
        dispatch_all = run.components[model]["reservoir_dispatch_gwh"].sel(zone=zone)
        inflow_all = inflow.sel(model=model, country=countries).sum("country")
        for period, window in windows.items():
            if window is None:
                continue
            for quantity, series in (("inflow", inflow_all), ("dispatch", dispatch_all)):
                block = series.sel(time=slice(*window))
                times = pd.DatetimeIndex(block["time"].values)
                years = len(set(times.year))
                for season, months in SEASON_MONTHS.items():
                    total = float(block.values[np.isin(times.month, months)].sum())
                    rows.append(dict(model=model, period=period, season=season,
                                     quantity=quantity, per_year=total / years))
    return pd.DataFrame(rows)


def seasonal_event_counts(run: ResidualLoadRun, context) -> pd.DataFrame:
    """Events per chain, warming level, zone, season and duration class, on the run's threshold.

    Each event is assigned whole to the season holding most of its time steps.

    Parameters
    ----------
    run : ResidualLoadRun
        The run, with its threshold.
    context : PaperContext
        Warming-level windows.

    Returns
    -------
    pandas.DataFrame
        As ``events_by_season`` of :func:`event_tables`, without ``hours_yr``.
    """
    return compute_event_season_metrics(
        run.residual_load, run.threshold_by_model, context.period_windows, run.models, run.zones,
        SEASON_MAP, SEASON_ORDER, method=EVENT_DETECTION, duration_bins=DURATION_BINS)
