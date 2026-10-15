"""Data behind the Supplementary Note 1 figures (S1-S16): the conversion pipeline.

Loaders for the gridded warming-level means, the per-country generation and demand fields, the
CMIP5 global-mean temperature, the chain table and the European seasonal area means, and the
:class:`~cordex2energy_analysis.plotting_publication.MapRow` builders the map figures are drawn
from.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from .analysis import load_warming_level_periods, resolve_model_periods
from .config import CLIMATE_SCENARIO, SEASON_MONTHS, SEASON_ORDER, TURBINE, WARMING_LEVELS
from .extremes_analysis import compute_c2e_gwl_field, mask_structural_zeros
from .fields import country_field, demand_field, load_country_output, load_demand_by_month
from .figure_cache import cached, monthly_totals
from .plotting_publication import QUANTITY_CMAP, MapRow, season_label

#: Season names in the gridded warming-level files, and the meteorological season keys.
GRID_SEASON_RENAME = {"Winter": "DJF", "Spring": "MAM", "Summer": "JJA", "Fall": "SON"}
#: The bias-corrected climate variables of S3-S7.
CLIMATE_VARIABLES = ["tas", "rsds", "sfcWind", "pr", "mrro"]
#: Runoff before and after river routing (S8).
ROUTING_VARIABLES = ["mrro", "Qrouted"]
CLIMATE_CMAPS = {"tas": "inferno", "rsds": "plasma", "sfcWind": "GnBu", "pr": "YlGnBu",
                 "mrro": "PuBu", "Qrouted": "GnBu",
                 "pv": "plasma", "wind": "Greens"}
#: Runoff is stored as a daily accumulation in "kg m**-2" (the CERRA convention), i.e. mm per day.
CLIMATE_UNITS = {"tas": "K", "rsds": "W m$^{-2}$", "sfcWind": "m s$^{-1}$", "pr": "mm d$^{-1}$",
                 "mrro": "mm d$^{-1}$", "Qrouted": "m$^3$ s$^{-1}$",
                 "pv": "capacity factor", "wind": "capacity factor"}
#: Channel-concentrated variables, whose absolute scale is clipped at a percentile.
ROBUST_ABS_VMAX = {"Qrouted"}

#: Display names of the regional and the driving global models.
RCM_DISPLAY = {"CNRM-ALADIN63": "ALADIN63", "MOHC-HadREM3-GA7-05": "HadREM3-GA7-05"}
GCM_DISPLAY = {"CNRM-CERFACS-CNRM-CM5_r1i1p1": "CNRM-CM5",
               "ICHEC-EC-EARTH_r12i1p1": "EC-EARTH",
               "MOHC-HadGEM2-ES_r1i1p1": "HadGEM2-ES",
               "MPI-M-MPI-ESM-LR_r1i1p1": "MPI-ESM-LR",
               "NCC-NorESM1-M_r1i1p1": "NorESM1-M"}
#: The CMIP5 ensemble member each driving GCM ran with.
DRIVING_MEMBERS = {"CNRM-CM5": "r1i1p1", "EC-EARTH": "r12i1p1", "HadGEM2-ES": "r1i1p1",
                   "MPI-ESM-LR": "r1i1p1", "NorESM1-M": "r1i1p1"}
#: Keys of the driving GCMs in the CMIP5 GMST table.
DRIVING_KEYS = {gcm: f"{gcm}_{member}" for gcm, member in DRIVING_MEMBERS.items()}

#: The analysis domain of the European area means.
EUROPE_EXTENT = {"lat": slice(30, 72.6), "lon": slice(-12, 39)}

#: Change-class edges (%) of the hydro rows, shared with Figure 2.
HYDRO_BINS = [-20, -10, -5, -2, -1, 1, 2, 5, 10, 20]
#: Change-class edges (%) of the demand rows, shared with Figure 3.
DEMAND_BINS = {
    "Heating demand": [-50, -30, -20, -15, -10, -1, 1, 10, 15, 20, 30, 50],
    "Cooling demand": [-200, -150, -100, -50, -25, -1, 1, 25, 50, 100, 150, 200],
}
#: Conversion steps drawn on the gridded capacity factors, and their variable names.
GENERATION_GRID_VARIABLE = {"Solar power": "pv", "Wind power": "wind"}


def load_gwl_grids(gwl_grid_dir, kind: str, period_names, seasonal: bool = False,
                   turbine: str = TURBINE) -> dict:
    """Gridded ensemble means at every warming level.

    Parameters
    ----------
    gwl_grid_dir : pathlib.Path
        Root of the warming-level products (``<processed>/energy/analysis``).
    kind : str
        ``"climate"`` for the bias-corrected climate variables, ``""`` for the energy variables.
    period_names : sequence of str
        Warming-level labels, in the order of :data:`~cordex2energy_analysis.config.WARMING_LEVELS`.
    seasonal : bool
        Read the meteorological-season files instead of the annual ones.
    turbine : str, optional
        Turbine of the wind capacity factors in the energy files; default the paper's.

    Returns
    -------
    dict
        ``{warming level: xarray.Dataset}`` with a ``model`` dimension and, if ``seasonal``, a
        ``season`` dimension in :data:`~cordex2energy_analysis.config.SEASON_ORDER`.
    """
    folder = gwl_grid_dir / ("seasonal" if seasonal else "annual") / "gwl"
    turbine_tag = "_" + turbine.replace("/", "_")
    suffix = {("climate", False): "_climate", ("climate", True): "_climate_meteorological",
              ("", False): turbine_tag,
              ("", True): f"{turbine_tag}_meteorological"}[(kind, seasonal)]
    grids = {}
    for level, period in zip(WARMING_LEVELS, period_names):
        dataset = xr.open_dataset(folder / f"mean_gwl_{int(round(level * 10))}{suffix}.nc")
        if seasonal:
            dataset = dataset.assign_coords(
                season=[GRID_SEASON_RENAME[str(season)] for season in dataset["season"].values]
            ).sel(season=SEASON_ORDER)
        grids[period] = dataset
    shapes = {period: (dataset.sizes.get("model"), dataset.sizes["lat"], dataset.sizes["lon"])
              for period, dataset in grids.items()}
    if len(set(shapes.values())) > 1:
        raise RuntimeError(f"GWL {kind or 'energy'} files disagree on shape: {shapes}")
    return grids


def climate_rows(grids: dict, season=None, variables=None) -> list:
    """One map row per climate variable, on a continuous ramp.

    Parameters
    ----------
    grids : dict
        ``{warming level: xarray.Dataset}`` from :func:`load_gwl_grids`.
    season : str, optional
        Season to select from seasonal grids.
    variables : sequence of str, optional
        Variables to draw; :data:`CLIMATE_VARIABLES` by default.

    Returns
    -------
    list of MapRow
        One row per variable. Temperature changes are absolute, the others relative; the change
        ramps keep red for warming and blue for wetter.
    """
    reference = next(iter(grids))
    rows = []
    for variable in variables or CLIMATE_VARIABLES:
        attributes = grids[reference][variable].attrs

        def field(period, variable=variable):
            """The variable's grid at one level, for the row's season if it has one."""
            values = grids[period][variable]
            return (values.sel(season=season)
                    if season is not None and "season" in values.dims else values)

        rows.append(MapRow(
            label=variable, field=field, cmap=CLIMATE_CMAPS[variable], kind="grid",
            unit=CLIMATE_UNITS[variable],
            change="absolute" if attributes["change_type"] == "absolute" else "percent",
            change_cmap="RdBu_r" if attributes["change_type"] == "absolute" else "RdBu",
            change_label=(f"Δ {CLIMATE_UNITS[variable]}"
                    if attributes["change_type"] == "absolute" else "% change"),
            robust_abs_vmax=variable in ROBUST_ABS_VMAX))
    return rows


def annual_country_cooling(context, country: str = "Spain") -> pd.DataFrame:
    """Annual cooling demand of one country, per chain and year of each warming-level window.

    Parameters
    ----------
    context : PaperContext
        Supplies the demand directory, the chains and their warming-level windows.
    country : str
        Country name as indexed in the demand product.

    Returns
    -------
    pandas.DataFrame
        Columns ``model``, ``period``, ``year_in_window`` (1-based) and ``twh`` (TWh per year).

    Notes
    -----
    The product is hourly power in GW, so the sum over a year is in GWh.
    """
    rows = []
    for model in context.models:
        with xr.open_dataset(context.paths.demand / f"{model}_{CLIMATE_SCENARIO}.nc") as dataset:
            series = dataset["cooling_demand"].sel(country=country)
            annual = series.resample(time="1YE").sum().load() / 1e3            # GW-hours -> TWh
        windows = resolve_model_periods(context.period_windows, model)
        for period in context.period_names:
            window = windows.get(period)
            if window is None:
                continue
            block = annual.sel(time=slice(str(window[0]), str(window[1])))
            for index, value in enumerate(block.values, start=1):
                rows.append(dict(model=model, period=period, year_in_window=index,
                                 twh=float(value)))
    return pd.DataFrame(rows)


def split_chain(model: str) -> tuple[str, str]:
    """Regional and driving global model of a chain.

    Parameters
    ----------
    model : str
        Chain identifier ``<RCM>_<GCM>_<member>``.

    Returns
    -------
    tuple of str
        ``(RCM, GCM_member)``.
    """
    return model.split("_")[0], "_".join(model.split("_")[1:])


def chain_table(models, period_windows: dict) -> pd.DataFrame:
    """The chains, their models and their warming-level windows.

    Parameters
    ----------
    models : sequence of str
        Chain identifiers.
    period_windows : dict
        ``{warming level: {chain: (first year, last year)}}``.

    Returns
    -------
    pandas.DataFrame
        One row per chain: RCM, driving GCM and the window at each level, sorted by GCM and RCM.
    """
    rows = []
    for model in models:
        rcm, gcm = split_chain(model)
        row = {"RCM": RCM_DISPLAY[rcm], "driving GCM": GCM_DISPLAY[gcm]}
        for period in period_windows:
            window = period_windows[period].get(model)
            row[period] = f"{int(window[0])}-{int(window[1])}" if window else "n/a"
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["driving GCM", "RCM"]).reset_index(drop=True)


def driving_gcm_rows(models) -> list[tuple[str, str]]:
    """One representative chain per driving GCM.

    Parameters
    ----------
    models : sequence of str
        Chain identifiers.

    Returns
    -------
    list of tuple
        ``(GCM_member, chain)``, ordered by the GCM's display name.
    """
    gcm_rows = []
    for model in models:
        _, gcm = split_chain(model)
        if gcm not in [listed_gcm for listed_gcm, _ in gcm_rows]:
            gcm_rows.append((gcm, model))
    gcm_rows.sort(key=lambda pair: GCM_DISPLAY[pair[0]])
    return gcm_rows


def load_cmip5_gmst(processed_dir) -> xr.DataArray:
    """Annual global-mean surface temperature anomaly of every CMIP5 run (1861-1900 baseline).

    Parameters
    ----------
    processed_dir : pathlib.Path
        Root of the processed data.

    Returns
    -------
    xarray.DataArray
        ``gmst_anomaly`` (K) on ``(model, year)``.
    """
    return xr.open_dataset(processed_dir / f"cmip5_gmst_annual_{CLIMATE_SCENARIO}.nc")["gmst_anomaly"]


def cmip5_crossing_years(warming_level_years_file) -> pd.DataFrame:
    """First year of every CMIP5 run's window at each warming level.

    Parameters
    ----------
    warming_level_years_file : pathlib.Path
        The warming-level YAML that defines the paper's windows.

    Returns
    -------
    pandas.DataFrame
        Columns ``model``, ``ensemble``, ``warming_level`` and ``start_year``.
    """
    all_windows = load_warming_level_periods(warming_level_years_file, WARMING_LEVELS, None)
    return pd.DataFrame(
        [dict(model=model, ensemble=ensemble, warming_level=level, start_year=window["start_year"])
         for (model, ensemble), scenarios in all_windows.items()
         for level, window in scenarios.get(CLIMATE_SCENARIO, {}).items()])


def area_mean(field: xr.DataArray, mask: xr.DataArray) -> float:
    """Cosine-latitude weighted spatial mean over a mask.

    Parameters
    ----------
    field : xarray.DataArray
        Field on ``(lat, lon)``.
    mask : xarray.DataArray
        Boolean mask on the same grid.

    Returns
    -------
    float
        The weighted mean over the masked cells.
    """
    weights = np.cos(np.deg2rad(field["lat"]))
    field = field.where(mask)
    return float(field.weighted(weights.broadcast_like(field).where(mask).fillna(0.0))
                 .mean(dim=["lat", "lon"]))


def european_seasonal_change(context) -> pd.DataFrame:
    """Each chain's European seasonal change in temperature, precipitation and runoff.

    Parameters
    ----------
    context : PaperContext
        Supplies the data paths.

    Returns
    -------
    pandas.DataFrame
        Columns ``model``, ``season``, ``dT`` (K), ``dP`` and ``dR`` (mm/d): the change from the
        lowest to the highest warming level, area-averaged over :data:`EUROPE_EXTENT`.

    Notes
    -----
    Temperature and precipitation are averaged over the CERRA footprint, runoff over land only,
    because the runoff product is zero over the sea.
    """
    processed = context.paths.processed
    cerra_mask = xr.open_dataarray(processed / "cerra_valid_mask.nc")
    land_mask = xr.open_dataarray(processed / "european_land_mask.nc")
    low, high = min(WARMING_LEVELS), max(WARMING_LEVELS)
    seasonal = {level: xr.open_dataset(
        context.paths.gwl_grids / "seasonal" / "gwl"
        / f"mean_gwl_{int(round(level * 10))}_climate_meteorological.nc"
    ).sel(**EUROPE_EXTENT) for level in (low, high)}
    chains = [str(chain) for chain in seasonal[low]["model"].values]

    def change(variable, chain, grid_season, mask):
        """A chain's change of a variable's area mean between the two levels."""
        return (area_mean(seasonal[high][variable].sel(model=chain, season=grid_season), mask)
                - area_mean(seasonal[low][variable].sel(model=chain, season=grid_season), mask))

    return pd.DataFrame([
        dict(model=chain.replace(f"_{CLIMATE_SCENARIO}", ""),
             season=GRID_SEASON_RENAME[grid_season],
             dT=change("tas", chain, grid_season, cerra_mask),
             dP=change("pr", chain, grid_season, cerra_mask),
             dR=change("mrro", chain, grid_season, land_mask))
        for chain in chains for grid_season in GRID_SEASON_RENAME])


def _annual_and_seasonal(field_functions: dict, period_names) -> tuple[dict, dict]:
    """Evaluate field functions at every warming level, annually and per season.

    Parameters
    ----------
    field_functions : dict
        ``{label: function(period, months) -> DataArray}``; ``months=None`` is the annual total.
    period_names : sequence of str
        Warming levels.

    Returns
    -------
    annual : dict
        ``{label: {warming level: DataArray}}``.
    seasonal : dict
        ``{label: {season: {warming level: DataArray}}}``.
    """
    annual = {label: {period: function(period, None) for period in period_names}
              for label, function in field_functions.items()}
    seasonal = {label: {season: {period: function(period, SEASON_MONTHS[season])
                                 for period in period_names}
                        for season in SEASON_ORDER}
                for label, function in field_functions.items()}
    return annual, seasonal


def generation_fields(context) -> tuple[dict, dict]:
    """Per-country run-of-river generation and reservoir inflow at every warming level.

    Parameters
    ----------
    context : PaperContext
        Supplies the data paths, the chains and their warming-level windows.

    Returns
    -------
    annual : dict
        ``{label: {warming level: DataArray(model, country)}}``, TWh per year.
    seasonal : dict
        ``{label: {season: {warming level: DataArray(model, country)}}}``, TWh per season.
    """
    components_by_model = load_country_output(context, files=("components.nc",))
    inflow_file = context.paths.gwl_grids / "c2e_cache" / f"c2e_hydro_inflow_{CLIMATE_SCENARIO}.nc"
    hydro_inflow = cached("c2e_hydro_inflow_monthly",
                          lambda: {"hydro_inflow": monthly_totals(xr.open_dataarray(inflow_file))},
                          key={"version": 1}, files=[inflow_file], code=False)
    period_windows = context.period_windows

    def run_of_river(period, months):
        """Run-of-river generation per country, structural zeros masked."""
        return mask_structural_zeros(country_field(
            components_by_model, period_windows, lambda data: data["gen_hydro_ror"], period,
            months=months))

    def reservoir_inflow(period, months):
        """Reservoir inflow per country, structural zeros masked."""
        return mask_structural_zeros(compute_c2e_gwl_field(
            hydro_inflow, period_windows, period, "hydro_inflow", months=months))

    return _annual_and_seasonal({"Run-of-river generation": run_of_river,
                                 "Reservoir inflow": reservoir_inflow}, context.period_names)


def demand_fields(context) -> tuple[dict, dict]:
    """Per-country heating and cooling demand at every warming level.

    Parameters
    ----------
    context : PaperContext
        Supplies the data paths, the chains and their warming-level windows.

    Returns
    -------
    annual : dict
        ``{label: {warming level: DataArray(model, country)}}``, TWh per year, countries as ISO2.
    seasonal : dict
        ``{label: {season: {warming level: DataArray(model, country)}}}``, TWh per season.
    """
    demand_by_month = load_demand_by_month(context)
    period_windows = context.period_windows

    def demand(variable):
        """Field builder for a demand variable over chosen months."""
        return lambda period, months: demand_field(demand_by_month, period_windows, variable,
                                                   period, months=months)

    return _annual_and_seasonal({"Heating demand": demand("heating"),
                                 "Cooling demand": demand("cooling")}, context.period_names)


def conversion_row(label: str, country_annual: dict, country_seasonal: dict, grids_annual: dict,
                   grids_seasonal: dict, *, season=None, unit_suffix: str = "yr",
                   row_label=None) -> MapRow:
    """One map row of a conversion step, annual or for one season.

    Parameters
    ----------
    label : str
        Conversion step: a key of :data:`GENERATION_GRID_VARIABLE` (gridded), of
        :data:`DEMAND_BINS`, or a hydro quantity of ``country_annual``.
    country_annual, country_seasonal : dict
        Per-country fields from :func:`generation_fields` and :func:`demand_fields`, merged.
    grids_annual, grids_seasonal : dict
        Gridded energy variables from :func:`load_gwl_grids`.
    season : str, optional
        Season of the row; annual by default.
    unit_suffix : str
        Time unit of a per-country total (``"yr"`` or ``"season"``).
    row_label : str, optional
        Row title; ``label`` by default.

    Returns
    -------
    MapRow
        A gridded row for the capacity factors, a choropleth row with fixed change classes for
        the per-country totals.
    """
    if label in GENERATION_GRID_VARIABLE:
        grids = grids_seasonal if season is not None else grids_annual
        variable = GENERATION_GRID_VARIABLE[label]

        def field(period, variable=variable, grids=grids, season=season):
            """The variable's grid at one level, for the row's season if it has one."""
            values = grids[period][variable]
            return (values.sel(season=season)
                    if season is not None and "season" in values.dims else values)

        return MapRow(label=row_label or label, field=field, cmap=CLIMATE_CMAPS[variable],
                      kind="grid", unit=CLIMATE_UNITS[variable], change="percent",
                      change_label="% change")
    fields = country_annual[label] if season is None else country_seasonal[label][season]

    def country_values(period, row_fields=fields):
        """The precomputed country values at one level."""
        return row_fields[period]

    if label in DEMAND_BINS:
        return MapRow(label=row_label or label, field=country_values, kind="country",
                      cmap=QUANTITY_CMAP[label], unit=f"TWh {unit_suffix}$^{{-1}}$",
                      change="percent", bins=DEMAND_BINS[label], reference_center=None, change_label=None)
    return MapRow(label=row_label or label, field=country_values, kind="country",
                  cmap=QUANTITY_CMAP["Hydro"], unit=f"TWh {unit_suffix}$^{{-1}}$", change="percent",
                  bins=HYDRO_BINS)


def conversion_rows(label: str, country_annual: dict, country_seasonal: dict, grids_annual: dict,
                    grids_seasonal: dict) -> list:
    """The annual row above the four seasonal rows of one conversion step.

    Parameters
    ----------
    label : str
        Conversion step, as in :func:`conversion_row`.
    country_annual, country_seasonal, grids_annual, grids_seasonal : dict
        As in :func:`conversion_row`.

    Returns
    -------
    list of MapRow
        Five rows: annual, then :data:`~cordex2energy_analysis.config.SEASON_ORDER`.
    """
    sources = (country_annual, country_seasonal, grids_annual, grids_seasonal)
    return [conversion_row(label, *sources, row_label="Annual")] + [
        conversion_row(label, *sources, season=season, unit_suffix="season",
                       row_label=season_label(season))
        for season in SEASON_ORDER]
