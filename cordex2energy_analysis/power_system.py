"""TYNDP 2024 power-system inputs and the residual-load / storage-dispatch model.

Main functions
--------------
load_tyndp_demand_profiles       Load a TYNDP demand-profile Excel file into an xarray Dataset.
compute_estimated_generation     Multiply capacity factors by TYNDP installed-capacity projections.
preprocess_thermal_demand        Upsample, smooth, correct and rescale thermal demand.
build_thermal_targets            Build per-country thermal TWh targets from TYNDP scenario data.
compute_tyndp_baseload           Subtract thermal demand from TYNDP market profiles, with a floor.
compute_or_load_baseload         Typical-year TYNDP baseload, cached on disk.
expand_typical_year              Tile an 8760-step typical-year profile across calendar years.
compute_residual_load            Residual load = total demand - (PV + wind + run-of-river hydro).
preprocess_hydro_ror             Map country names to ISO2 and spread daily run-of-river to sub-daily.
aggregate_by_balancing_zones     Sum country series into the six real-time balancing zones.
load_tyndp_supply_capacities     Solar and wind installed capacities per bidding zone.
load_tyndp_battery_capacities    Prosumer and utility battery capacities per country.
apply_battery_dispatch           Utility battery dispatch on the residual load.
apply_prosumer_battery_dispatch  Prosumer battery dispatch.
apply_reservoir_dispatch         Reservoir hydro dispatch.
installed_capacities_by_country  Installed capacity of every modelled country under one build-out.
"""

from __future__ import annotations

import calendar
import warnings
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import xarray as xr

from cordex2energy.utils.constants import REFERENCE_PERIOD

#: Real-time balancing zone -> member countries (ISO2; the UK is ``"UK"``).
REALTIME_BALANCING_ZONES = {
    "Nordic":         ['NO', 'SE', 'FI', 'DK'],
    "Central_Western":   ['FR', 'DE', 'LU', 'BE', 'NL', 'CH', 'AT'],
    "Iberian_Peninsula":    ['ES', 'PT'],
    "Central_Eastern":   ['PL', 'CZ', 'SK', 'HU', 'RO', 'BG', 'EE', 'LV', 'LT'],
    "South_East":  ['IT', 'GR', 'SI', 'HR', 'BA', 'RS', 'ME', 'MK', 'AL', 'MT', 'CY'],
    "British_Isles":        ['UK', 'IE']
}

#: Country (ISO2) -> balancing zone.
COUNTRY_ZONE = {country: zone for zone, members in REALTIME_BALANCING_ZONES.items()
                for country in members}


# ---------------------------------------------------------------------------
# Demand profile loading
# ---------------------------------------------------------------------------


def load_tyndp_demand_profiles(path, min_hours=8000):
    """Load a TYNDP demand-profile Excel file into a country-level dataset.

    Parameters
    ----------
    path : str or Path
        Demand-profile workbook with one sheet per bidding zone, year headers on
        Excel row 12 and one row per hour.
    min_hours : int, optional
        Minimum number of non-NaN hours for a column to count as a climate year.
        The default (8000, below the 8760 hours of a year) rejects columns that
        hold only stray values under a year-like header.

    Returns
    -------
    xr.Dataset
        Variable ``"demand_profile"`` with dims ``(country, time)``. Bidding zones
        are summed by the first two characters of their code over the union of
        all zones' time stamps.

    Raises
    ------
    ValueError
        If no sheet holds a valid climate year.

    Notes
    -----
    A column is a climate year when its header parses as an integer in
    [1900, 2100] and it holds at least ``min_hours`` values. Sheet ``UK00``
    labels its climate years one year early (1994/2007/2008 instead of
    1995/2008/2009); these labels are shifted by one. A warning is printed for
    any sheet whose climate years differ from {1995, 2008, 2009}.
    """
    expected_years = {1995, 2008, 2009}

    workbook = pd.ExcelFile(path)
    series_by_zone: dict[str, pd.Series] = {}

    for sheet in workbook.sheet_names:
        table = workbook.parse(sheet, header=11)
        table = table.dropna(axis=1, how="all").dropna(how="all")

        year_of_column: dict = {}
        for column in table.columns:
            try:
                year = int(float(str(column).strip()))
                if 1900 <= year <= 2100 and table[column].notna().sum() >= min_hours:
                    year_of_column[column] = year
            except (ValueError, TypeError):
                pass

        if sheet == "UK00" and set(year_of_column.values()) == {1994, 2007, 2008}:
            year_of_column = {column: year + 1 for column, year in year_of_column.items()}

        if set(year_of_column.values()) != expected_years:
            print(
                f"Warning: sheet '{sheet}' has unexpected climate years "
                f"{set(year_of_column.values())} (expected {expected_years})."
            )

        segments = []
        for column, year in sorted(year_of_column.items(), key=lambda item: item[1]):
            values = table[column].dropna().values.astype(float)
            # date_range yields 8784 steps for leap years.
            timestamps = pd.date_range(f"{year}-01-01", periods=len(values), freq="h")
            segments.append(pd.Series(values, index=timestamps))

        if segments:
            series_by_zone[sheet] = pd.concat(segments)

    if not series_by_zone:
        raise ValueError(f"No demand-profile data found in {path}")

    all_times = pd.DatetimeIndex(
        sorted(set().union(*[series.index.tolist() for series in series_by_zone.values()]))
    )
    zones = list(series_by_zone.keys())
    values = np.stack(
        [series_by_zone[zone].reindex(all_times).values for zone in zones], axis=0
    )

    profiles = xr.DataArray(
        values,
        dims=["tyndp_zone", "time"],
        coords={"tyndp_zone": zones, "time": all_times},
        name="demand_profile",
    ).to_dataset()

    return (
        profiles.assign_coords(country=("tyndp_zone", [zone[:2] for zone in zones]))
        .groupby("country")
        .sum()
    )


# ---------------------------------------------------------------------------
# Generation estimation
# ---------------------------------------------------------------------------


def _capacity_column(table, column):
    """Extract one capacity column of a TYNDP supply table, indexed by node code.

    Parameters
    ----------
    table : pd.DataFrame
        Supply table with a ``"Country"`` column of TYNDP node codes.
    column : str
        Capacity column, ``{year}_{scenario}``.

    Returns
    -------
    pd.Series
        Capacities (float, NaN replaced by 0) indexed by node code.
    """
    return (
        table[["Country", column]]
        .dropna(subset=["Country"])
        .set_index("Country")[column]
        .fillna(0.0)
        .astype(float)
    )


def _reintegrate_dropped_zones(capacity, available_zones):
    """Move the capacity of zones missing from the capacity factors onto a peer zone.

    A dropped zone (e.g. ``DK00``, superseded by ``DKW1`` + ``DKE1``) shares the
    capacity factors of its country peers, so adding its capacity to the first
    available zone of the same country preserves the country total.

    Parameters
    ----------
    capacity : pd.Series
        Capacity indexed by TYNDP node code.
    available_zones : list of str
        Zones present in the capacity-factor dataset.

    Returns
    -------
    pd.Series
        Capacity indexed by ``available_zones`` (0 where absent).
    """
    merged = capacity.reindex(
        capacity.index.union(available_zones), fill_value=0.0
    ).copy()
    for zone in capacity.index:
        if zone not in available_zones and capacity.get(zone, 0.0) > 0:
            peers = [peer for peer in available_zones if peer.startswith(zone[:2])]
            if peers:
                merged[peers[0]] += capacity[zone]
    return merged.reindex(available_zones, fill_value=0.0)


def _as_data_array(series, dim):
    """Wrap a pandas Series as a 1-D DataArray along ``dim``.

    Parameters
    ----------
    series : pd.Series
        Values indexed by the coordinate labels.
    dim : str
        Dimension name.

    Returns
    -------
    xr.DataArray
    """
    return xr.DataArray(series.values, dims=[dim], coords={dim: series.index.values})


def _sum_zones_to_countries(data, zone_to_country, zone_dim):
    """Sum bidding-zone data into countries.

    Parameters
    ----------
    data : xr.Dataset
        Data with a ``zone_dim`` dimension.
    zone_to_country : pd.Series
        Country code indexed by zone code.
    zone_dim : str
        Bidding-zone dimension of ``data``.

    Returns
    -------
    xr.Dataset
        Data with ``zone_dim`` replaced by a sorted ``country`` dimension.
    """
    slices = []
    for country, zones in sorted(zone_to_country.groupby(zone_to_country).groups.items()):
        country_data = data.sel({zone_dim: list(zones)}).sum(zone_dim)
        slices.append(country_data.expand_dims({"country": [country]}))
    return xr.concat(slices, dim="country")


def compute_estimated_generation(
    pv_capacity_factors,
    wind_onshore_capacity_factors,
    wind_offshore_capacity_factors,
    solar_capacities,
    wind_onshore_capacities,
    wind_offshore_capacities,
    scenario,
    year,
    zone_dim="tyndp_zone",
    country_dim="country",
):
    """Multiply capacity factors by TYNDP installed capacities and sum to countries.

    Parameters
    ----------
    pv_capacity_factors, wind_onshore_capacity_factors : xr.Dataset
        Capacity factors with a ``zone_dim`` dimension of TYNDP bidding-zone codes
        (e.g. ``"FR00"``, ``"DKW1"``).
    wind_offshore_capacity_factors : xr.Dataset
        Capacity factors with a ``country_dim`` dimension.
    solar_capacities, wind_onshore_capacities, wind_offshore_capacities : pd.DataFrame
        TYNDP supply tables (see :func:`load_tyndp_supply_capacities`) with a
        ``"Country"`` column of bidding-zone codes and capacity columns
        ``{year}_{scenario}``.
    scenario : str
        Scenario suffix of the capacity columns (e.g. ``"GA"``, ``"HIGH"``).
    year : int
        Target year (2030, 2040 or 2050).
    zone_dim : str, optional
        Bidding-zone dimension of the PV and onshore-wind capacity factors.
    country_dim : str, optional
        Country dimension of the offshore-wind capacity factors.

    Returns
    -------
    pv_generation, wind_onshore_generation, wind_offshore_generation : xr.Dataset
        Capacity × capacity factor per country, in the capacity unit of the
        table (MW). Every variable carries ``attrs["unit"] = "MWh"`` and every
        dataset ``attrs["source"] = "TYNDP {year} {scenario}"``.

    Notes
    -----
    PV and onshore wind are multiplied at bidding-zone level, then summed to
    countries (first two characters of the zone code). Capacity of zones absent
    from the capacity factors is moved onto a peer zone of the same country
    (see :func:`_reintegrate_dropped_zones`); it is lost if the country has no
    zone in the capacity factors. Offshore capacities are summed to countries
    before multiplication because the offshore capacity factors are already per
    country.
    """
    column = f"{year}_{scenario}"

    solar = _capacity_column(solar_capacities, column)
    wind_onshore = _capacity_column(wind_onshore_capacities, column)
    wind_offshore = _capacity_column(wind_offshore_capacities, column)

    available_zones = list(pv_capacity_factors[zone_dim].values)
    solar_by_zone = _reintegrate_dropped_zones(solar, available_zones)
    wind_onshore_by_zone = _reintegrate_dropped_zones(wind_onshore, available_zones)

    pv_by_zone = pv_capacity_factors * _as_data_array(solar_by_zone, zone_dim)
    wind_by_zone = wind_onshore_capacity_factors * _as_data_array(wind_onshore_by_zone, zone_dim)

    zone_to_country = pd.Series({zone: zone[:2] for zone in available_zones})
    pv_generation = _sum_zones_to_countries(pv_by_zone, zone_to_country, zone_dim)
    wind_onshore_generation = _sum_zones_to_countries(wind_by_zone, zone_to_country, zone_dim)

    wind_offshore_by_country = (
        wind_offshore
        .rename(index=lambda zone: zone[:2])
        .groupby(level=0)
        .sum()
    )
    offshore_countries = list(wind_offshore_capacity_factors[country_dim].values)
    wind_offshore_aligned = wind_offshore_by_country.reindex(offshore_countries, fill_value=0.0)
    wind_offshore_generation = (
        wind_offshore_capacity_factors * _as_data_array(wind_offshore_aligned, country_dim)
    )

    for generation in [pv_generation, wind_onshore_generation, wind_offshore_generation]:
        generation.attrs["source"] = f"TYNDP {year} {scenario}"
        for variable in generation.data_vars:
            generation[variable].attrs["unit"] = "MWh"

    return pv_generation, wind_onshore_generation, wind_offshore_generation


# ---------------------------------------------------------------------------
# Thermal demand pre-processing
# ---------------------------------------------------------------------------


def _is_nested_target(targets):
    """Return whether a thermal target mapping is per variable (``{variable: {country: TWh}}``).

    Parameters
    ----------
    targets : dict or pd.Series
        Flat ``{country: TWh}`` or nested ``{variable: {country: TWh}}`` targets.

    Returns
    -------
    bool
        True when the first value is itself a dict.
    """
    return isinstance(next(iter(targets.values()), None), dict)


def preprocess_thermal_demand(
    thermal_demand,
    target_twh_per_country=None,
    loss_factor=1.176,
    rolling_window=4,
    upsample_to_1h=True,
    reference_period=REFERENCE_PERIOD,
):
    """Upsample, smooth, loss-correct and rescale thermal electricity demand.

    Each data variable is processed independently, so heating and cooling can
    be carried as two variables and each rescaled to its own target.

    1. Optionally upsample 3-hourly data to 1 h by cubic interpolation, then
       apply a centred ``rolling_window``-hour rolling mean (thermal inertia).
    2. Multiply by ``loss_factor``.
    3. If targets are given, scale each variable of each country by
       ``k = E_target / E_simulated``, where ``E_simulated`` is the mean annual
       sum over ``reference_period``. Countries without a target, or with a
       non-positive simulated total, keep ``k = 1``.

    Parameters
    ----------
    thermal_demand : xr.Dataset
        Thermal electricity demand in GW, dims ``(country, time)``; one lumped
        variable (``demand_profile``) or ``heating_demand`` + ``cooling_demand``.
    target_twh_per_country : dict or pd.Series, optional
        Annual electricity target in TWh, either nested
        ``{variable: {country: TWh}}`` (each variable to its own target) or flat
        ``{country: TWh}`` (applied to every variable). ``None`` skips step 3.
    loss_factor : float, optional
        COP/loss multiplier; default 1.176 (= 1/0.85, the ENTSO-E correction).
        Use 1.0 for an already-electric product.
    rolling_window : int, optional
        Width of the centred rolling mean in hours.
    upsample_to_1h : bool, optional
        Whether to upsample 3 h -> 1 h before smoothing; False for hourly input.
    reference_period : sequence of two str or None, optional
        ``(start, end)`` years over which ``E_simulated`` is computed; ``None``
        uses the whole series.

    Returns
    -------
    xr.Dataset
        Hourly thermal demand in GW with the input's variable names (not summed).

    Notes
    -----
    Whenever a country is rescaled, ``loss_factor`` cancels in ``k``; it only
    changes countries without a target. Anchoring ``E_simulated`` to the
    reference period keeps the future trend of the series intact.
    """
    hourly = (
        thermal_demand.resample(time="1h").interpolate("cubic")
        if upsample_to_1h else thermal_demand
    )
    smoothed = hourly.rolling(time=rolling_window, center=True, min_periods=1).mean()
    scaled = smoothed * loss_factor

    if target_twh_per_country is not None:
        nested = _is_nested_target(target_twh_per_country)

        for variable in scaled.data_vars:
            variable_targets = (
                target_twh_per_country.get(variable) if nested else target_twh_per_country
            )
            if variable_targets is None:
                continue
            scale_by_country = []
            for country in scaled[variable].country.values:
                code = str(country)
                if code in variable_targets:
                    simulated = scaled[variable].sel(country=country)
                    if reference_period is not None:
                        simulated = simulated.sel(time=slice(*reference_period))
                    # Hourly GW summed over a year is GWh; 1e-3 converts to TWh.
                    simulated_twh = (
                        float(simulated.resample(time="YS").sum().mean(dim="time").values) * 1e-3
                    )
                    scale = float(variable_targets[code]) / simulated_twh if simulated_twh > 0 else 1.0
                else:
                    scale = 1.0
                scale_by_country.append(scale)
            scale_array = xr.DataArray(
                scale_by_country, dims=["country"],
                coords={"country": scaled[variable].country.values},
            )
            scaled[variable] = scaled[variable] * scale_array

    return scaled


# ---------------------------------------------------------------------------
# Thermal targets
# ---------------------------------------------------------------------------

#: EU proxy for non-EU countries absent from the EU-27 TYNDP demand spreadsheet, chosen for
#: similar climate and building stock. The Western Balkans (AL, BA, ME, MK, RS) have no
#: comparable EU-27 country and are left out (k = 1); the floor in compute_tyndp_baseload
#: protects them.
_DEFAULT_NON_EU_PROXIES: dict[str, str] = {
    "UK": "FR",
    "CH": "AT",
    "NO": "SE",
}


def build_thermal_targets(
    demand_table,
    scenario,
    year=2050,
    non_eu_proxy_map=None,
    heating_subsectors=("Space heating", "Space heating & hot water"),
    cooling_subsectors=("Cooling",),
    market_demand=None,
):
    """Build per-country heating, cooling and total electricity targets in TWh.

    EU countries are read from the TYNDP demand spreadsheet. A non-EU country in
    the proxy map gets ``share × E_total``, where the heating (cooling) share is
    the proxy's heating (cooling) electricity over its total electricity, and
    ``E_total`` is the country's spreadsheet total or, failing that, its mean
    annual sum in ``market_demand``.

    Parameters
    ----------
    demand_table : pd.DataFrame
        The ``3_DEMAND_OUTPUT`` sheet with columns ``ENERGY_CARRIER``,
        ``SUBSECTOR``, ``COUNTRY`` and value columns flattened to
        ``{year}_{scenario}`` (e.g. ``"2050_GA"``), in TWh.
    scenario : str
        Scenario identifier (``"GA"``, ``"DE"``, ``"HIGH"``, ...).
    year : int, optional
        Target year.
    non_eu_proxy_map : dict, optional
        ``{country: proxy}`` pairs that override or extend
        :data:`_DEFAULT_NON_EU_PROXIES`.
    heating_subsectors : tuple of str, optional
        ``SUBSECTOR`` values counted as electric heating.
    cooling_subsectors : tuple of str, optional
        ``SUBSECTOR`` values counted as electric cooling.
    market_demand : xr.Dataset, optional
        TYNDP market demand profiles in MWh, dims ``(country, time)``, variable
        ``"demand_profile"``; the ``E_total`` fallback for non-EU countries.

    Returns
    -------
    dict
        ``{"heating": {country: TWh}, "cooling": {country: TWh},
        "total": {country: TWh}}``. Only positive values are kept. ``"total"``
        is the spreadsheet ``SUBSECTOR == "Total"`` electricity demand and is
        never filled from ``market_demand``.
    """
    column = f"{year}_{scenario}"
    proxy_map = {**_DEFAULT_NON_EU_PROXIES, **(non_eu_proxy_map or {})}

    def electricity_by_country(subsectors):
        """Sum the electricity rows of ``subsectors`` per country."""
        is_selected = (demand_table["ENERGY_CARRIER"].eq("Electricity")
                       & demand_table["SUBSECTOR"].isin(subsectors))
        return (
            demand_table.loc[is_selected, ["COUNTRY", column]]
            .dropna(subset=[column])
            .groupby("COUNTRY")[column]
            .sum()
        )

    heating_twh = electricity_by_country(heating_subsectors)
    cooling_twh = electricity_by_country(cooling_subsectors)
    total_twh = electricity_by_country(["Total"])

    market_twh: dict[str, float] = {}
    if market_demand is not None:
        year_count = max(1, len(np.unique(market_demand.time.dt.year.values)))
        for country in market_demand.country.values:
            # MWh -> TWh, mean per year.
            annual_twh = (
                float(market_demand["demand_profile"].sel(country=country).values.sum()) * 1e-6 / year_count
            )
            if annual_twh > 0:
                market_twh[str(country)] = annual_twh

    heating_targets: dict[str, float] = {
        country: float(value) for country, value in heating_twh.items() if float(value) > 0}
    cooling_targets: dict[str, float] = {
        country: float(value) for country, value in cooling_twh.items() if float(value) > 0}
    total_targets: dict[str, float] = {
        country: float(value) for country, value in total_twh.items() if float(value) > 0}

    for country, proxy in proxy_map.items():
        if country in heating_targets or country in cooling_targets:
            continue
        if proxy not in total_twh or float(total_twh[proxy]) <= 0:
            print(f"Warning: proxy '{proxy}' for '{country}' not found in "
                  f"TYNDP demand data - skipping.")
            continue

        heating_share = float(heating_twh.get(proxy, 0.0)) / float(total_twh[proxy])
        cooling_share = float(cooling_twh.get(proxy, 0.0)) / float(total_twh[proxy])

        if country in total_twh and float(total_twh[country]) > 0:
            country_total_twh = float(total_twh[country])
            source = "spreadsheet"
        elif country in market_twh:
            country_total_twh = market_twh[country]
            source = "market_demand"
        else:
            print(f"Warning: no total electricity demand found for '{country}' "
                  f"in TYNDP demand data or market_demand - cannot synthesise "
                  f"thermal target (falling back to k=1).")
            continue

        heating_targets[country] = heating_share * country_total_twh
        cooling_targets[country] = cooling_share * country_total_twh
        print(f"  {country}: proxy='{proxy}', a_heat={heating_share:.3f}, "
              f"a_cool={cooling_share:.3f}, E_total={country_total_twh:.1f} TWh [{source}] -> "
              f"E_heat={heating_targets[country]:.1f}, E_cool={cooling_targets[country]:.1f} TWh")

    return {"heating": heating_targets, "cooling": cooling_targets, "total": total_targets}


# ---------------------------------------------------------------------------
# Baseload extraction
# ---------------------------------------------------------------------------


def compute_tyndp_baseload(market_demand, demand_to_subtract, floor_fraction=0.15):
    """Subtract a demand component from TYNDP market demand, with a per-country floor.

    Parameters
    ----------
    market_demand : xr.Dataset
        Market demand, variable ``"demand_profile"``, dims ``(country, time)``.
    demand_to_subtract : xr.Dataset
        Demand to remove (thermal demand in the pipeline), variable
        ``"demand_profile"``, dims ``(country, time)``. Countries missing here
        are treated as zero.
    floor_fraction : float, optional
        Lower bound on the result as a fraction of each country's mean market
        demand.

    Returns
    -------
    xr.Dataset
        Variable ``"baseload"``, dims ``(country, time)``, on the intersection
        of the two time axes.
    """
    common_time = np.intersect1d(market_demand.time.values, demand_to_subtract.time.values)
    market_demand = market_demand.sel(time=common_time)
    demand_to_subtract = demand_to_subtract.sel(time=common_time)
    print(f"Aligned time axis: {len(common_time)} hours from "
          f"{pd.to_datetime(common_time[0])} to {pd.to_datetime(common_time[-1])}.")

    print(f"Filling missing countries in ds_prosumer with 0: "
          f"{set(market_demand.country.values) - set(demand_to_subtract.country.values)}")
    demand_to_subtract = demand_to_subtract.reindex(country=market_demand.country, fill_value=0.0)

    baseload = market_demand - demand_to_subtract

    # The floor prevents negative (physically impossible) baseload.
    floor = floor_fraction * market_demand.mean(dim="time")
    baseload = baseload.where(baseload >= floor, other=floor)

    return baseload.rename({"demand_profile": "baseload"})


# ---------------------------------------------------------------------------
# Typical-year expansion
# ---------------------------------------------------------------------------


def expand_typical_year(typical_year, years, typical_dim="hour_of_year"):
    """Tile an 8760-step typical-year profile across calendar years.

    Parameters
    ----------
    typical_year : xr.Dataset
        Dataset with dimension ``typical_dim`` of length 8760 and coordinate
        values 0-8759 (0 = 00:00 Jan 1 of a non-leap year).
    years : array-like of int
        Calendar years to cover; sorted before use.
    typical_dim : str, optional
        Name of the hour-of-year dimension.

    Returns
    -------
    xr.Dataset
        Hourly dataset of ``len(years) × 8760`` steps along ``time``; other
        dimensions are unchanged.

    Notes
    -----
    Feb 29 is skipped, and in leap years later dates are shifted back one day,
    so Mar 1 always maps onto the same typical-year slot.
    """
    times_per_year = []
    hours_per_year = []

    for year in sorted(years):
        year_times = pd.date_range(
            start=f"{year}-01-01", end=f"{year}-12-31 23:00", freq="1h"
        )
        if calendar.isleap(year):
            year_times = year_times[~((year_times.month == 2) & (year_times.day == 29))]

        day_of_year = year_times.day_of_year.values.copy()
        if calendar.isleap(year):
            day_of_year[year_times.month.values > 2] -= 1

        times_per_year.append(year_times)
        hours_per_year.append((day_of_year - 1) * 24 + year_times.hour.values)

    time_index = pd.DatetimeIndex(
        np.concatenate([times.values for times in times_per_year])
    )
    hour_of_year = xr.DataArray(
        np.concatenate(hours_per_year), dims=["time"], coords={"time": time_index}
    )
    return typical_year.sel({typical_dim: hour_of_year}).drop_vars(typical_dim)


# ---------------------------------------------------------------------------
# Residual load
# ---------------------------------------------------------------------------


def _sum_data_variables(dataset):
    """Sum all data variables of a dataset into one DataArray.

    Parameters
    ----------
    dataset : xr.Dataset

    Returns
    -------
    xr.DataArray
        The single variable itself if there is only one.
    """
    arrays = [dataset[variable] for variable in dataset.data_vars]
    return arrays[0] if len(arrays) == 1 else sum(arrays[1:], start=arrays[0])


def compute_residual_load(
    pv_generation,
    wind_onshore_generation,
    wind_offshore_generation,
    hydro_ror_generation,
    baseload,
    heating_demand,
    cooling_demand,
    target_twh_per_country=None,
    loss_factor=1.176,
    rolling_window_thermal_demand=4,
    rolling_window_residual_load=8,
    upsample_to_1h=True,
    reference_period=REFERENCE_PERIOD,
    country_dim="country",
    return_generation_demand_components=False,
):
    """Compute residual load = total demand − (PV + onshore + offshore wind + run-of-river hydro).

    Total demand is the hourly baseload plus the thermal demand processed by
    :func:`preprocess_thermal_demand`, summed to 3-hourly totals to match the
    generation. All quantities are in GWh per step.

    Parameters
    ----------
    pv_generation, wind_onshore_generation, wind_offshore_generation : xr.Dataset
        3-hourly generation in GWh per step, dims ``(country, time)``; all data
        variables are summed.
    hydro_ror_generation : xr.Dataset
        3-hourly run-of-river generation in GWh per step, dims ``(country, time)``.
    baseload : xr.Dataset
        Hourly non-thermal demand, variable ``"baseload"``. Either time-indexed,
        or a typical year with an ``hour_of_year`` dimension, which is expanded
        with :func:`expand_typical_year` over the years of ``pv_generation``.
    heating_demand : xr.Dataset
        Heating electricity demand, variable ``"heating_demand"``: hourly for
        ``cordex2energy_demand`` (the paper's runs), 3-hourly for demand.ninja
        (with ``upsample_to_1h``).
    cooling_demand : xr.Dataset
        Cooling electricity demand, variable ``"cooling_demand"``, at the same
        resolution as ``heating_demand``.
    target_twh_per_country : dict, optional
        Thermal targets for :func:`preprocess_thermal_demand`. A nested
        ``{variable: {country: TWh}}`` target keeps heating and cooling as
        separate variables; a flat ``{country: TWh}`` target or ``None``
        processes their sum as one ``demand_profile`` variable.
    loss_factor : float, optional
        Passed to :func:`preprocess_thermal_demand`.
    rolling_window_thermal_demand : int, optional
        Thermal-inertia smoothing window in hours.
    rolling_window_residual_load : int, optional
        Trailing rolling-mean window applied to the residual load, in 3-hourly
        steps; values ≤ 1 disable it.
    upsample_to_1h : bool, optional
        Whether the thermal demand is 3-hourly and needs upsampling.
    reference_period : sequence of two str or None, optional
        Rescaling anchor passed to :func:`preprocess_thermal_demand`.
    country_dim : str, optional
        Country dimension of all inputs.
    return_generation_demand_components : bool, optional
        Also return the aligned generation and demand components.

    Returns
    -------
    residual : xr.Dataset
        Variable ``"residual_load"`` (GWh per 3 h), dims ``(country, time)``;
        positive is unmet demand, negative is surplus generation.
    components : xr.Dataset
        Only if ``return_generation_demand_components``: ``gen_pv``,
        ``gen_wind_on``, ``gen_wind_off``, ``gen_hydro_ror`` and
        ``total_demand`` on the residual-load grid.

    Notes
    -----
    Countries are those present in both baseload and thermal demand; a country
    missing from a generation input gets zero for that technology. Time is the
    intersection of all inputs (no extrapolation).
    """
    if "hour_of_year" in baseload.dims:
        baseload = expand_typical_year(baseload, years=np.unique(pv_generation.time.dt.year.values))

    thermal_time = np.intersect1d(heating_demand.time.values, cooling_demand.time.values)
    nested_targets = (
        isinstance(target_twh_per_country, dict)
        and _is_nested_target(target_twh_per_country)
    )
    if nested_targets:
        thermal = xr.Dataset({
            "heating_demand": heating_demand["heating_demand"].sel(time=thermal_time),
            "cooling_demand": cooling_demand["cooling_demand"].sel(time=thermal_time),
        })
    else:
        thermal = xr.Dataset({
            "demand_profile": (
                heating_demand["heating_demand"].sel(time=thermal_time)
                + cooling_demand["cooling_demand"].sel(time=thermal_time)
            )
        })

    thermal_hourly = preprocess_thermal_demand(
        thermal,
        target_twh_per_country=target_twh_per_country,
        loss_factor=loss_factor,
        rolling_window=rolling_window_thermal_demand,
        upsample_to_1h=upsample_to_1h,
        reference_period=reference_period,
    )

    thermal_total_hourly = _sum_data_variables(thermal_hourly)
    countries = np.intersect1d(
        baseload[country_dim].values,
        thermal_total_hourly[country_dim].values,
    )
    demand_time = np.intersect1d(
        baseload.time.values,
        thermal_total_hourly.time.values,
    )
    total_demand_hourly = (
        baseload["baseload"].sel({country_dim: countries, "time": demand_time})
        + thermal_total_hourly.sel({country_dim: countries, "time": demand_time})
    )
    # Generation carries 3-hourly totals; summing three hourly demand steps matches it.
    total_demand_3hourly = total_demand_hourly.resample(time="3h").sum()

    common_time = total_demand_3hourly.time.values
    for generation in [pv_generation, wind_onshore_generation, wind_offshore_generation,
                       hydro_ror_generation]:
        common_time = np.intersect1d(common_time, generation.time.values)

    def aligned(generation):
        """Sum the variables of ``generation`` and align it on the residual-load grid."""
        return (
            _sum_data_variables(generation)
            .reindex({country_dim: countries}, fill_value=0.0)
            .sel(time=common_time)
        )

    components = xr.Dataset(
        {
            "gen_pv": aligned(pv_generation),
            "gen_wind_on": aligned(wind_onshore_generation),
            "gen_wind_off": aligned(wind_offshore_generation),
            "gen_hydro_ror": aligned(hydro_ror_generation),
        }
    )
    for component in components.data_vars:
        components[component].attrs["units"] = "GWh"
    total_generation = (components["gen_pv"] + components["gen_wind_on"]
                        + components["gen_wind_off"] + components["gen_hydro_ror"])

    total_demand = total_demand_3hourly.sel({country_dim: countries, "time": common_time})
    residual = total_demand - total_generation

    if rolling_window_residual_load > 1:
        residual = residual.rolling(time=rolling_window_residual_load, center=False, min_periods=1).mean()

    residual = residual.to_dataset(name="residual_load")
    residual.attrs["description"] = "Residual load = total demand − (PV + onshore wind + offshore wind + ROR hydro)"
    residual["residual_load"].attrs["units"] = "GWh"
    residual["residual_load"].attrs["long_name"] = "Residual Load"

    if return_generation_demand_components:
        components["total_demand"] = total_demand
        components["total_demand"].attrs["units"] = "GWh"
        return residual, components
    return residual


# ---------------------------------------------------------------------------
# Hydro run-of-river pre-processing
# ---------------------------------------------------------------------------


def preprocess_hydro_ror(
    daily_ror,
    name_to_iso2,
    freq="3h",
    var_name="ror",
):
    """Map run-of-river country labels to ISO2 and spread daily values evenly over sub-daily steps.

    Parameters
    ----------
    daily_ror : pd.DataFrame
        Daily run-of-river generation; rows are country labels, columns are
        date strings.
    name_to_iso2 : dict
        Full country name -> ISO2 code; labels not in it are kept (stripped).
    freq : str, optional
        Target sub-daily frequency (e.g. ``"3h"``, ``"1h"``).
    var_name : str, optional
        Name of the output variable.

    Returns
    -------
    xr.Dataset
        Dims ``(country, time)`` at ``freq``. Each day's value is divided by the
        number of steps per day (e.g. GWh/day -> GWh/3 h), so a day's steps sum
        to the daily total. The series ends at the first step of the last day.
    """
    iso2_codes = [
        name_to_iso2.get(str(label).strip(), str(label).strip())
        for label in daily_ror.index
    ]

    daily = xr.DataArray(
        daily_ror.values,
        dims=["country", "time"],
        coords={
            "country": iso2_codes,
            "time": pd.to_datetime(daily_ror.columns),
        },
        name=var_name,
    ).to_dataset()

    steps_per_day = int(pd.Timedelta("1D") / pd.Timedelta(freq))
    return daily.resample(time=freq).ffill() / steps_per_day


def aggregate_by_balancing_zones(data, country_dim="country"):
    """Sum country data into the six real-time balancing zones.

    Parameters
    ----------
    data : xr.Dataset or xr.DataArray
        Data with a country dimension of ISO2 codes.
    country_dim : str, optional
        Name of the country dimension.

    Returns
    -------
    xr.Dataset or xr.DataArray
        ``country_dim`` replaced by a ``zone`` dimension. Countries not in
        :data:`REALTIME_BALANCING_ZONES` are dropped.
    """
    available_countries = [country for country in data[country_dim].values if country in COUNTRY_ZONE]
    zone_labels = [COUNTRY_ZONE[country] for country in available_countries]
    return (
        data.sel({country_dim: available_countries})
        .assign_coords(zone=(country_dim, zone_labels))
        .groupby("zone")
        .sum(dim=country_dim)
    )


# ---------------------------------------------------------------------------
# Supply capacity loading
# ---------------------------------------------------------------------------

def _parse_supply_sheet(path_excel: Path, sheet_name: str) -> pd.DataFrame:
    """Load one sheet of the TYNDP supply-input workbook.

    Parameters
    ----------
    path_excel : Path
        Supply-input workbook.
    sheet_name : str
        Sheet to read, e.g. ``"1.1."``.

    Returns
    -------
    pd.DataFrame
        A ``"Country"`` column followed by float capacity columns
        ``{year}_{scenario}`` up to ``"2050_HIGH"``; thousands separators and
        non-breaking spaces are removed and unparsable cells become NaN.
    """
    table = pd.read_excel(path_excel, sheet_name=sheet_name, header=[2, 3])
    table.columns = [
        year_label if str(year_label) == "Country" else f"{year_label}_{scenario_label}"
        for scenario_label, year_label in table.columns
    ]
    table = table.loc[:, "Country":"2050_HIGH"]
    numeric_columns = [column for column in table.columns if column != "Country"]
    table[numeric_columns] = table[numeric_columns].apply(
        lambda values: pd.to_numeric(
            values.astype(str)
            .str.replace("\xa0", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip(),
            errors="coerce",
        )
    )
    return table


def load_tyndp_supply_capacities(
    path_excel: Union[str, Path],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load TYNDP 2024 solar, onshore-wind and offshore-wind capacity projections.

    Parameters
    ----------
    path_excel : str or Path
        ``20231103 - Final Supply Inputs for TYNDP 2024 Scenarios.xlsx`` or an
        equivalent workbook (sheets ``1.1.``, ``1.2.``, ``1.3.``).

    Returns
    -------
    solar, wind_onshore, wind_offshore : pd.DataFrame
        A ``"Country"`` column of TYNDP bidding-zone codes and float capacity
        columns (MW) named ``{year}_{scenario}``.
    """
    path_excel = Path(path_excel)
    solar = _parse_supply_sheet(path_excel, "1.1.")
    wind_onshore = _parse_supply_sheet(path_excel, "1.2.")
    wind_offshore = _parse_supply_sheet(path_excel, "1.3.")
    return solar, wind_onshore, wind_offshore


def load_tyndp_battery_capacities(
    path_excel: Union[str, Path],
    scenario: str = "HIGH",
    year: int = 2050,
    return_separate: bool = False,
) -> "pd.Series | tuple[pd.Series, pd.Series]":
    """Load TYNDP 2024 battery energy capacities per ISO2 country, in GWh.

    Parameters
    ----------
    path_excel : str or Path
        TYNDP supply-input workbook (sheets ``1.5.`` prosumer, ``1.6.`` utility;
        values in MWh).
    scenario : str, optional
        Scenario of the column header (e.g. ``"HIGH"``).
    year : int, optional
        Projection year.
    return_separate : bool, optional
        Return the prosumer and utility capacities separately instead of their sum.

    Returns
    -------
    pd.Series or tuple of pd.Series
        Total capacity, or ``(prosumer, utility)``, in GWh indexed by the first
        two characters of the node code. ``TOTAL``/``EU-TOTAL`` rows are excluded.
    """
    path_excel = Path(path_excel)
    column = f"{year}_{scenario}"

    aggregate_rows = {"TOTAL", "EU-TOTAL"}

    def sum_by_iso2(table: pd.DataFrame) -> pd.Series:
        """Sum node rows of ``table`` into ISO2 countries."""
        is_node = table["Country"].apply(
            lambda label: isinstance(label, str) and label not in aggregate_rows
        )
        rows = table.loc[is_node, ["Country", column]].copy()
        rows["iso2"] = rows["Country"].str[:2]
        return rows.groupby("iso2")[column].sum().fillna(0.0)

    prosumer_gwh = sum_by_iso2(_parse_supply_sheet(path_excel, "1.5.")) / 1_000.0
    utility_gwh = sum_by_iso2(_parse_supply_sheet(path_excel, "1.6.")) / 1_000.0
    prosumer_gwh.name = f"prosumer_battery_capacity_gwh_{year}_{scenario}"
    utility_gwh.name = f"utility_battery_capacity_gwh_{year}_{scenario}"

    if return_separate:
        return prosumer_gwh, utility_gwh

    total_gwh = prosumer_gwh.add(utility_gwh, fill_value=0.0)
    total_gwh.name = f"battery_capacity_gwh_{year}_{scenario}"
    return total_gwh


def aggregate_battery_capacities_by_zone(
    capacity_by_country: pd.Series,
) -> pd.Series:
    """Sum country battery capacities into the real-time balancing zones.

    Parameters
    ----------
    capacity_by_country : pd.Series
        Battery energy capacity in GWh indexed by ISO2 code; missing countries
        count as zero.

    Returns
    -------
    pd.Series
        Capacity in GWh per zone of :data:`REALTIME_BALANCING_ZONES`, with the
        input's name.
    """
    capacity_by_zone = {}
    for zone, countries in REALTIME_BALANCING_ZONES.items():
        capacity_by_zone[zone] = sum(float(capacity_by_country.get(country, 0.0)) for country in countries)
    return pd.Series(capacity_by_zone, name=capacity_by_country.name)


# ---------------------------------------------------------------------------
# Country code mapping
# ---------------------------------------------------------------------------

#: Countries in TYNDP bidding zones that are absent from the EU-27 demand spreadsheet.
_NON_EU_COUNTRY_CODES: dict[str, str] = {
    "AL": "Albania",
    "BA": "Bosnia and Herzegovina",
    "CH": "Switzerland",
    "HR": "Croatia",
    "ME": "Montenegro",
    "MK": "North Macedonia",
    "NO": "Norway",
    "RS": "Serbia",
    "SI": "Slovenia",
}

#: Country labels of the C2E/CERRA/CORDEX files that differ from the TYNDP names.
_DEFAULT_C2E_EXTRA_MAPPINGS: dict[str, str] = {
    "Macedonia": "MK",   # TYNDP: 'North Macedonia'
    "Poland": "PL",      # TYNDP: 'Poland (PL)'
    "Luxembourg": "LU",
    "Kosovo": "XK",      # not in the TYNDP demand model
    "Ukraine": "UA",     # not in the TYNDP demand model
}


def load_tyndp_country_codes(path_demand_excel: Union[str, Path]) -> dict[str, str]:
    """Load the ISO2 -> country-name mapping of the TYNDP demand workbook.

    Parameters
    ----------
    path_demand_excel : str or Path
        ``Demand_Scenarios_TYNDP_2024_After_Public_Consultation.xlsb`` or an
        equivalent workbook with a ``1_ETM_SCENARIO_LINKS`` sheet.

    Returns
    -------
    dict
        ``{ISO2: country_name}`` (whitespace-stripped), extended with
        :data:`_NON_EU_COUNTRY_CODES`.
    """
    links = pd.read_excel(
        path_demand_excel, sheet_name="1_ETM_SCENARIO_LINKS", header=0
    )
    codes = (
        links[["COUNTRY_ID", "COUNTRY_NAME"]]
        .drop_duplicates()
        .set_index("COUNTRY_ID")["COUNTRY_NAME"]
        .to_dict()
    )
    codes.update(_NON_EU_COUNTRY_CODES)
    return {code.strip(): name.strip() for code, name in codes.items()}


def build_c2e_name_to_iso2(
    tyndp_country_codes: dict[str, str],
    extra_mappings: dict[str, str] | None = None,
) -> dict[str, str]:
    """Build a country-label -> ISO2 lookup for the C2E/CERRA/CORDEX files.

    Parameters
    ----------
    tyndp_country_codes : dict
        ``{ISO2: full_name}``, e.g. from :func:`load_tyndp_country_codes`.
    extra_mappings : dict, optional
        Further ``{label: ISO2}`` pairs, applied last.

    Returns
    -------
    dict
        ``{label: ISO2}``: the inverted TYNDP mapping updated with
        :data:`_DEFAULT_C2E_EXTRA_MAPPINGS` and ``extra_mappings``.
    """
    name_to_iso2 = {name: code for code, name in tyndp_country_codes.items()}
    name_to_iso2.update(_DEFAULT_C2E_EXTRA_MAPPINGS)
    if extra_mappings:
        name_to_iso2.update(extra_mappings)
    return name_to_iso2


def load_c2e_demand_nc(
    path_demand_nc: Union[str, Path],
    name_to_iso2: dict,
    time_slice=None,
) -> tuple[xr.Dataset, xr.Dataset]:
    """Load an hourly ``cordex2energy_demand`` NetCDF as heating and cooling datasets.

    Parameters
    ----------
    path_demand_nc : str or Path
        ``energy/europe/demand/{model_scenario}.nc`` (or ``cerra.nc``) with
        variables ``heating_demand`` / ``cooling_demand`` in GW and dims
        ``time``/``country`` (full country names).
    name_to_iso2 : dict
        Full name -> ISO2 mapping (from :func:`build_c2e_name_to_iso2`).
    time_slice : tuple of str, optional
        ``(start, end)`` passed to ``.sel(time=slice(...))``.

    Returns
    -------
    heating, cooling : xr.Dataset
        Single-variable datasets (``heating_demand``, ``cooling_demand``) in GW
        with ISO2 ``country`` coordinates, matching the 3-hourly CSV loaders
        except for the hourly resolution (use ``upsample_to_1h=False``).

    Notes
    -----
    Names that do not map to a two-letter code are reported, since they drop
    out of later ISO2 intersections silently.
    """
    demand = xr.open_dataset(path_demand_nc)
    iso2_codes = [name_to_iso2.get(str(name).strip(), str(name).strip())
                  for name in demand["country"].values]

    unmapped = sorted({str(name) for name, code in zip(demand["country"].values, iso2_codes)
                       if not (isinstance(code, str) and len(code) == 2)})
    if unmapped:
        print(f"  [WARN] load_c2e_demand_nc: {len(unmapped)} country name(s) did not "
              f"map to ISO2 (will not intersect downstream): {unmapped}")

    demand = demand.assign_coords(country=iso2_codes)
    if time_slice is not None:
        demand = demand.sel(time=slice(*time_slice))

    return demand[["heating_demand"]], demand[["cooling_demand"]]


# ---------------------------------------------------------------------------
# Demand profile loading helpers
# ---------------------------------------------------------------------------

def load_tyndp_market_profiles(
    path_demand_profiles_dir: Union[str, Path],
    demand_scenario: str,
    demand_year: int,
) -> tuple[xr.Dataset, xr.Dataset]:
    """Load the TYNDP market and prosumer hourly demand profiles of one scenario and year.

    Parameters
    ----------
    path_demand_profiles_dir : str or Path
        Root of the demand profiles, laid out as
        ``{demand_scenario}/{demand_year}/ELECTRICITY_MARKET {demand_scenario} {demand_year}.xlsx``
        (and ``ELECTRICITY_PROSUMER ...``).
    demand_scenario : str
        Scenario identifier (e.g. ``"DE"``, ``"GA"``).
    demand_year : int
        Target year (e.g. 2040, 2050).

    Returns
    -------
    market, prosumer : xr.Dataset
        Variable ``"demand_profile"`` in MWh, dims ``(country, time)``.
    """
    scenario_dir = Path(path_demand_profiles_dir) / demand_scenario / str(demand_year)
    path_market = scenario_dir / f"ELECTRICITY_MARKET {demand_scenario} {demand_year}.xlsx"
    path_prosumer = scenario_dir / f"ELECTRICITY_PROSUMER {demand_scenario} {demand_year}.xlsx"
    market = load_tyndp_demand_profiles(path_market)
    prosumer = load_tyndp_demand_profiles(path_prosumer)
    return market, prosumer


# ---------------------------------------------------------------------------
# Typical-year profile
# ---------------------------------------------------------------------------

def compute_typical_year_profile(hourly: xr.Dataset) -> xr.Dataset:
    """Average a time-indexed dataset into an 8760-step typical year.

    Parameters
    ----------
    hourly : xr.Dataset
        Dataset with a ``time`` dimension.

    Returns
    -------
    xr.Dataset
        Mean per ``hour_of_year = (day_of_year − 1) × 24 + hour`` (0-8759), Feb 29
        excluded. Can be passed as ``baseload`` to :func:`compute_residual_load`.
    """
    is_leap_day = (hourly.time.dt.month == 2) & (hourly.time.dt.day == 29)
    without_leap_day = hourly.sel(time=~is_leap_day)
    hour_of_year = (without_leap_day.time.dt.dayofyear - 1) * 24 + without_leap_day.time.dt.hour
    return (
        without_leap_day
        .assign_coords(hour_of_year=("time", hour_of_year.values))
        .groupby("hour_of_year")
        .mean("time")
    )


def _load_cerra_demand_csv(path_csv, variable, name_to_iso2, reference_period):
    """Load a C2E/CERRA demand CSV (rows = country names, columns = times) over the reference period.

    Parameters
    ----------
    path_csv : Path
        CSV file.
    variable : str
        Name of the output variable.
    name_to_iso2 : dict
        Full name -> ISO2 mapping; unknown names are kept.
    reference_period : sequence of two str
        ``(start, end)`` time slice.

    Returns
    -------
    xr.Dataset
        One variable, dims ``(country, time)``.
    """
    table = pd.read_csv(path_csv, index_col=0)
    return (
        xr.DataArray(
            table.values, dims=["country", "time"],
            coords={"country": [name_to_iso2.get(name, name) for name in table.index],
                    "time": pd.to_datetime(table.columns)},
            name=variable,
        ).to_dataset().sel(time=slice(*reference_period))
    )


def compute_or_load_baseload(
    path_c2e_cerra: Path,
    path_baseload_cache: Path,
    path_tyndp: Path,
    demand_scenario: str,
    demand_year: int,
    demand_table: pd.DataFrame,
    name_to_iso2: dict,
    floor_fraction: float = 0.15,
    demand_model: str = "demand.ninja",
    path_energy_demand: Path = None,
    modulate_baseload: bool = True,
    reference_period=REFERENCE_PERIOD,
    legacy_thermal: bool = False,
) -> xr.Dataset:
    """Return the typical-year TYNDP baseload, computing and caching it if needed.

    The baseload is TYNDP (market + prosumer) demand minus the reference-period
    CERRA thermal demand of ``demand_model``, so that the thermal demand added
    back in :func:`compute_residual_load` comes from the same model.

    Parameters
    ----------
    path_c2e_cerra : Path
        C2E/CERRA output root (holds ``output_variables/heating-demand_CERRA.csv``
        and ``cooling-demand_CERRA.csv``); used for ``"demand.ninja"``.
    path_baseload_cache : Path
        Cache directory.
    path_tyndp : Path
        TYNDP data root (holds ``Demand Profiles/``).
    demand_scenario : str
        TYNDP demand scenario (e.g. ``"DE"``).
    demand_year : int
        Target year (e.g. 2050).
    demand_table : pd.DataFrame
        Flattened TYNDP ``3_DEMAND_OUTPUT`` table (see :func:`build_thermal_targets`).
    name_to_iso2 : dict
        Full country name -> ISO2 mapping.
    floor_fraction : float, optional
        Baseload floor as a fraction of mean total market demand.
    demand_model : str, optional
        ``"demand.ninja"`` (3-hourly CSVs) or ``"cordex2energy_demand"``
        (hourly ``cerra.nc`` under ``path_energy_demand``).
    path_energy_demand : Path, optional
        ``energy/europe/demand`` directory; required for ``"cordex2energy_demand"``.
    modulate_baseload : bool, optional
        Scale the baseload of each country in the spreadsheet ``"total"`` so its
        annual sum equals total − heating − cooling, then re-apply the floor.
        Ignored when ``legacy_thermal``.
    reference_period : sequence of two str, optional
        Years of the CERRA slice and of the rescaling anchor.
    legacy_thermal : bool, optional
        Use one lumped heating + cooling target scaled over the whole series,
        without modulation.

    Returns
    -------
    xr.Dataset
        Variable ``"baseload"`` in GWh per hour with an ``hour_of_year``
        dimension, read from or written to
        ``baseload_{demand_scenario}_{demand_year}_{model_tag}[_legacy].nc``,
        where ``model_tag`` is ``dninja`` or ``c2edem``.

    Raises
    ------
    ValueError
        If ``demand_model="cordex2energy_demand"`` and ``path_energy_demand``
        is None, or if the cached file was built with other settings.

    Notes
    -----
    Heating and cooling are rescaled to separate targets only for
    ``"cordex2energy_demand"``. demand.ninja uses one lumped target because its
    reference-period cooling is close to zero in cool countries, where a
    separate cooling scale factor would diverge.

    The file name records the scenario, year, demand model and ``legacy_thermal``.
    ``floor_fraction``, ``modulate_baseload`` and ``reference_period`` are stored as
    attributes and checked when the cache is read; a file without them (written
    before they were recorded) is used with a warning.
    """
    model_tag = {"demand.ninja": "dninja", "cordex2energy_demand": "c2edem"}.get(
        demand_model, demand_model
    )
    cache_path = path_baseload_cache / (
        f"baseload_{demand_scenario}_{demand_year}_{model_tag}"
        f"{'_legacy' if legacy_thermal else ''}.nc"
    )

    settings = {
        "floor_fraction": str(float(floor_fraction)),
        "modulate_baseload": str(bool(modulate_baseload) and not legacy_thermal),
        "reference_period": f"{reference_period[0]}-{reference_period[1]}",
    }

    if cache_path.exists():
        print(f"  Found cached baseload -> {cache_path}")
        cached_baseload = xr.open_dataset(cache_path)
        recorded = {key: str(cached_baseload.attrs[key])
                    for key in settings if key in cached_baseload.attrs}
        if not recorded:
            warnings.warn(f"{cache_path.name} does not record the settings it was built with; "
                          f"assuming {settings}.", stacklevel=2)
        elif recorded != settings:
            raise ValueError(f"{cache_path} was built with {recorded}, not {settings}; "
                             "delete it or pass the matching settings.")
        return cached_baseload

    print(f"  Baseload cache not found - computing for {demand_scenario} {demand_year} "
          f"[{demand_model}{', legacy' if legacy_thermal else ''}] ...")

    market, prosumer = load_tyndp_market_profiles(
        path_tyndp / "Demand Profiles", demand_scenario, demand_year
    )

    targets = build_thermal_targets(
        demand_table,
        scenario=demand_scenario,
        year=demand_year,
        market_demand=market,
    )

    if demand_model == "cordex2energy_demand":
        if path_energy_demand is None:
            raise ValueError(
                "path_energy_demand is required for demand_model='cordex2energy_demand'"
            )
        cerra_heating, cerra_cooling = load_c2e_demand_nc(
            Path(path_energy_demand) / "cerra.nc",
            name_to_iso2,
            time_slice=reference_period,
        )
        upsample_to_1h, loss_factor = False, 1.0          # hourly, already electric
    else:
        csv_dir = path_c2e_cerra / "output_variables"
        cerra_heating = _load_cerra_demand_csv(
            csv_dir / "heating-demand_CERRA.csv", "heating_demand", name_to_iso2, reference_period
        )
        cerra_cooling = _load_cerra_demand_csv(
            csv_dir / "cooling-demand_CERRA.csv", "cooling_demand", name_to_iso2, reference_period
        )
        upsample_to_1h, loss_factor = True, 1 / 0.85      # 3-hourly; ENTSO-E COP correction

    thermal_time = np.intersect1d(cerra_heating.time.values, cerra_cooling.time.values)
    cerra_thermal = xr.Dataset({
        "heating_demand": cerra_heating["heating_demand"].sel(time=thermal_time),
        "cooling_demand": cerra_cooling["cooling_demand"].sel(time=thermal_time),
    })

    split_targets = (demand_model == "cordex2energy_demand") and not legacy_thermal
    if split_targets:
        preprocessed = preprocess_thermal_demand(
            cerra_thermal,
            target_twh_per_country={"heating_demand": targets["heating"],
                                    "cooling_demand": targets["cooling"]},
            loss_factor=loss_factor, rolling_window=4, upsample_to_1h=upsample_to_1h,
            reference_period=reference_period,
        )
        thermal_total = preprocessed["heating_demand"] + preprocessed["cooling_demand"]
    else:
        lumped_thermal = xr.Dataset({
            "demand_profile": cerra_thermal["heating_demand"] + cerra_thermal["cooling_demand"]
        })
        lumped_targets = {country: targets["heating"].get(country, 0.0) + targets["cooling"].get(country, 0.0)
                          for country in set(targets["heating"]) | set(targets["cooling"])}
        preprocessed = preprocess_thermal_demand(
            lumped_thermal, target_twh_per_country=lumped_targets, loss_factor=loss_factor,
            rolling_window=4, upsample_to_1h=upsample_to_1h,
            reference_period=None if legacy_thermal else reference_period,
        )
        thermal_total = preprocessed["demand_profile"]
    thermal_preprocessed = xr.Dataset({"demand_profile": thermal_total})

    # Market and prosumer profiles are in MWh; 1e-3 converts to GWh.
    total_market = (market + prosumer.reindex(
        country=market.country, fill_value=0.0
    )) * 1e-3
    baseload = compute_tyndp_baseload(
        total_market,
        thermal_preprocessed,
        floor_fraction=floor_fraction,
    )

    if modulate_baseload and not legacy_thermal:
        total = targets["total"]
        year_count = max(1, len(np.unique(baseload.time.dt.year.values)))
        modulation_by_country = []
        for country in baseload.country.values:
            code = str(country)
            baseload_twh = float(baseload["baseload"].sel(country=country).sum().values) * 1e-3 / year_count
            target_twh = (total.get(code, 0.0)
                          - targets["heating"].get(code, 0.0)
                          - targets["cooling"].get(code, 0.0))
            modulation_by_country.append(
                target_twh / baseload_twh if (code in total and baseload_twh > 0 and target_twh > 0) else 1.0
            )
        modulation = xr.DataArray(modulation_by_country, dims=["country"],
                                  coords={"country": baseload.country.values})
        baseload = baseload * modulation
        # Modulation can push a country below the floor; re-apply it.
        floor = (floor_fraction * total_market["demand_profile"].mean(dim="time")).to_dataset(name="baseload")
        baseload = baseload.where(baseload >= floor, other=floor)
        modulated_count = int(sum(1 for country in baseload.country.values if str(country) in total))
        print(f"  Modulated baseload for {modulated_count} TYNDP-scenario countries "
              f"(total - heating - cooling).")

    baseload_typical = compute_typical_year_profile(baseload)
    baseload_typical.attrs.update(settings)

    baseload_typical.to_netcdf(cache_path)
    print(f"  Saved baseload cache -> {cache_path}")

    return baseload_typical


# ---------------------------------------------------------------------------
# Battery dispatch
# ---------------------------------------------------------------------------


def _start_quiet_gurobi_env():
    """Start a Gurobi environment with console and file logging disabled.

    Returns
    -------
    gurobipy.Env
        The started environment; the caller disposes of it.
    """
    import gurobipy as gp

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.setParam("LogFile", "")
    env.start()
    return env


def _optimize_battery_gurobi(
    residual_load: np.ndarray,
    energy_capacity: float,
    power_capacity: float,
    one_way_efficiency: float,
    min_soc_fraction: float | None = None,
    initial_soc: float | None = None,
    env=None,
    max_signfix_passes: int = 4,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Solve a battery dispatch QP over one horizon with Gurobi.

    Minimises ``Σ net_load[t]²`` with ``net_load = residual_load + charge − discharge``,
    subject to ``soc[t+1] = soc[t] + η·charge[t] − discharge[t]/η``, power bounds
    ``0 ≤ charge, discharge ≤ power_capacity``, storage bounds and a boundary
    condition on the storage (see ``initial_soc``).

    Parameters
    ----------
    residual_load : np.ndarray
        Residual load over the horizon (GWh per 3 h step); must be finite.
    energy_capacity : float
        Battery energy capacity (GWh).
    power_capacity : float
        Maximum charge or discharge per 3 h step (GWh = installed GW × 3).
    one_way_efficiency : float
        ``η = sqrt(roundtrip_efficiency)``, applied on charge and on discharge.
    min_soc_fraction : float, optional
        Lower bound on the state of charge as a fraction of ``energy_capacity``.
    initial_soc : float, optional
        If None, the cyclicity constraint ``soc[0] == soc[T]`` is imposed. If
        set, ``soc[0]`` is fixed to it and the terminal state is free.
    env : gurobipy.Env, optional
        Started environment to reuse across solves; if None, one is created and
        disposed of within the call.
    max_signfix_passes : int, optional
        Maximum number of sign-fixing re-solves (see Notes).

    Returns
    -------
    dispatch, net_load, soc : np.ndarray or None
        ``dispatch`` and ``net_load`` have length T, ``soc`` has length T + 1;
        ``(None, None, None)`` if a solve returns no solution.
        ``dispatch = discharge − charge`` (positive reduces the residual load);
        ``soc[k]`` is the storage entering step ``k``, so ``soc[T]`` is the
        storage at the end of the horizon.

    Notes
    -----
    The objective depends only on ``charge − discharge``, so charging and
    discharging in the same step drains energy without changing the net load.
    Because ``Σ net_load²`` penalises surplus, the relaxation exploits this to
    dispose of surplus energy and could flatten a surplus year with an
    arbitrarily small battery; round-trip losses do not prevent it.
    ``charge[t] · discharge[t] = 0`` is therefore enforced by sign-fixing:
    after each solve, each step's upper bounds allow only the direction the
    previous solution chose, and the QP is re-solved until the pattern is
    stable. Each pass is a convex QP; the result is feasible (doing nothing
    satisfies every clamp) and never overstates the battery's benefit.
    Feasible solutions that Gurobi cannot certify optimal (SUBOPTIMAL/NUMERIC)
    are accepted.
    """
    import gurobipy as gp
    from gurobipy import GRB

    horizon = len(residual_load)
    energy_floor = energy_capacity * min_soc_fraction if min_soc_fraction is not None else 0.0

    owns_env = env is None
    if owns_env:
        env = _start_quiet_gurobi_env()

    try:
        model = gp.Model("Battery_Dispatch", env=env)

        # storage has T + 1 entries: storage[k] is the level entering step k.
        storage = model.addMVar(shape=horizon + 1, lb=energy_floor, ub=energy_capacity, name="E")
        charge = model.addMVar(shape=horizon, lb=0.0, ub=power_capacity, name="charge")
        discharge = model.addMVar(shape=horizon, lb=0.0, ub=power_capacity, name="discharge")
        net_load = model.addMVar(shape=horizon, lb=-GRB.INFINITY, name="net_load")

        model.addConstr(
            storage[1:] == storage[:-1] + one_way_efficiency * charge - discharge / one_way_efficiency,
            name="SOC_Balance",
        )
        model.addConstr(net_load == residual_load + charge - discharge, name="Net_Load_Def")

        if initial_soc is not None:
            model.addConstr(storage[0] == initial_soc, name="Initial_SOC")
        else:
            model.addConstr(storage[0] == storage[horizon], name="Cyclicity")

        model.setObjective(net_load @ net_load, GRB.MINIMIZE)
        model.optimize()
        if model.SolCount == 0:
            return None, None, None

        tolerance = 1e-9
        full_bound = np.full(horizon, power_capacity, dtype=float)
        charge_upper, discharge_upper = full_bound, full_bound
        for _ in range(max_signfix_passes):
            net_charge = charge.X - discharge.X
            new_charge_upper = np.where(net_charge > tolerance, power_capacity, 0.0)
            new_discharge_upper = np.where(net_charge < -tolerance, power_capacity, 0.0)
            if (np.array_equal(new_charge_upper, charge_upper)
                    and np.array_equal(new_discharge_upper, discharge_upper)):
                break
            charge_upper, discharge_upper = new_charge_upper, new_discharge_upper
            charge.UB = charge_upper
            discharge.UB = discharge_upper
            model.optimize()
            if model.SolCount == 0:
                return None, None, None

        return discharge.X - charge.X, net_load.X, storage.X
    finally:
        if owns_env:
            env.dispose()


def _capacity_per_zone(capacity_gwh, zones, zone_dim):
    """Look up the capacity of each zone, 0 where absent.

    Parameters
    ----------
    capacity_gwh : pd.Series or xr.DataArray
        Capacity indexed by zone (Series index or ``zone_dim`` coordinate).
    zones : list
        Zones in output order.
    zone_dim : str
        Zone dimension of a DataArray ``capacity_gwh``.

    Returns
    -------
    np.ndarray
        Float capacities aligned with ``zones``.
    """
    if isinstance(capacity_gwh, pd.Series):
        return np.array([capacity_gwh.get(zone, 0.0) for zone in zones], dtype=float)
    capacity_zones = list(capacity_gwh[zone_dim].values)
    return np.array(
        [float(capacity_gwh.sel({zone_dim: zone}).values) if zone in capacity_zones else 0.0
         for zone in zones],
        dtype=float,
    )


def _battery_dispatch_result(residual_dataset, residual_load_variable, residual_by_zone,
                             original, adjusted, stored_energy, label):
    """Attach the battery dispatch outputs to a copy of the residual-load dataset.

    Parameters
    ----------
    residual_dataset : xr.Dataset
        Input dataset.
    residual_load_variable : str
        Residual-load variable; its dimension order is used for the outputs.
    residual_by_zone : xr.DataArray
        The residual load transposed to ``(zone, time)``; supplies coordinates.
    original, adjusted, stored_energy : np.ndarray
        ``(zone, time)`` residual load before and after dispatch, and state of charge.
    label : str
        ``"battery"`` or ``"prosumer battery"``, used in the ``long_name`` attributes.

    Returns
    -------
    xr.Dataset
        Copy with ``residual_load_with_batteries``, ``soc_gwh`` and
        ``battery_dispatch_gwh`` (= original − adjusted, positive = discharging).
    """
    original_dims = residual_dataset[residual_load_variable].dims

    def as_output(values, attrs):
        """Wrap a ``(zone, time)`` array in the original dimension order."""
        return xr.DataArray(
            values, coords=residual_by_zone.coords, dims=residual_by_zone.dims, attrs=attrs,
        ).transpose(*original_dims)

    result = residual_dataset.copy()
    result["residual_load_with_batteries"] = as_output(
        adjusted,
        {"long_name": f"Residual load after {label} dispatch", "units": "GWh per 3h step"},
    )
    result["soc_gwh"] = as_output(
        stored_energy, {"long_name": f"{label.capitalize()} state-of-charge", "units": "GWh"},
    )
    result["battery_dispatch_gwh"] = as_output(
        original - adjusted,
        {
            "long_name": f"{label.capitalize()} net dispatch per 3h step",
            "units": "GWh per 3h step",
            "sign_convention": "positive = discharging (reduces residual load)",
        },
    )
    return result


def _dispatch_annual_qp(residual, adjusted, stored_energy_trace, capacity_by_zone, zones, time_index,
                        battery_duration_h, one_way_efficiency, min_soc_fraction, verbose):
    """Perfect-foresight, annually cyclic QP dispatch, filling the output arrays in place.

    Parameters
    ----------
    residual : np.ndarray
        ``(zone, time)`` residual load in GWh per 3 h step.
    adjusted, stored_energy_trace : np.ndarray
        ``(zone, time)`` output arrays, overwritten for solved zone-years.
    capacity_by_zone : np.ndarray
        Energy capacity per zone (GWh).
    zones : list
        Zone labels, for messages.
    time_index : pd.DatetimeIndex
        Time stamps of ``residual``.
    battery_duration_h, one_way_efficiency, min_soc_fraction, verbose
        As in :func:`apply_battery_dispatch` (efficiency is one-way).
    """
    env = _start_quiet_gurobi_env()
    try:
        for zone_index in range(len(zones)):
            capacity = capacity_by_zone[zone_index]
            if capacity <= 0.0:
                continue
            power_per_step = capacity / battery_duration_h * 3.0  # GWh per 3h step
            for year in np.unique(time_index.year):
                in_year = time_index.year == year
                residual_year = residual[zone_index, in_year]
                if np.all(np.isnan(residual_year)):
                    continue
                # Gurobi needs finite inputs: solve with NaN as 0, restore NaN afterwards.
                is_missing = np.isnan(residual_year)
                residual_finite = np.where(is_missing, 0.0, residual_year)
                _, net_load, stored_energy = _optimize_battery_gurobi(
                    residual_finite, capacity, power_per_step, one_way_efficiency,
                    min_soc_fraction=min_soc_fraction,
                    env=env,
                )
                if net_load is None:
                    if verbose:
                        print(f"    {zones[zone_index]} {year}: Gurobi infeasible, skipping")
                    continue
                positions = np.where(in_year)[0]
                adjusted[zone_index, positions] = np.where(is_missing, np.nan, net_load)
                stored_energy_trace[zone_index, positions] = stored_energy[:-1]
            if verbose:
                print(f"  Zone {zones[zone_index]}: dispatched {capacity:.2f} GWh battery (annual QP)")
    finally:
        env.dispose()


def _dispatch_rolling_qp(residual, adjusted, stored_energy_trace, capacity_by_zone, zones,
                         battery_duration_h, one_way_efficiency, initial_soc_fraction,
                         min_soc_fraction, commit_h, dispatch_lookahead_h, verbose):
    """Rolling day-ahead QP dispatch, filling the output arrays in place.

    Each solve covers ``commit_h + dispatch_lookahead_h`` hours with the state of
    charge carried in from the previous window; only the first ``commit_h`` is
    kept, then the window slides by ``commit_h``. Keeping only the head pushes
    the end-of-window emptying of the free terminal state into the discarded
    tail. The state of charge runs continuously across years.

    Parameters
    ----------
    residual : np.ndarray
        ``(zone, time)`` residual load in GWh per 3 h step.
    adjusted, stored_energy_trace : np.ndarray
        ``(zone, time)`` output arrays, overwritten for committed steps.
    capacity_by_zone : np.ndarray
        Energy capacity per zone (GWh).
    zones : list
        Zone labels, for messages.
    battery_duration_h, one_way_efficiency, initial_soc_fraction, min_soc_fraction
        As in :func:`apply_battery_dispatch` (efficiency is one-way).
    commit_h, dispatch_lookahead_h, verbose
        As in :func:`apply_battery_dispatch` (efficiency is one-way).
    """
    step_count = residual.shape[1]
    commit_steps = max(1, round(commit_h / 3.0))
    lookahead_steps = max(0, round(dispatch_lookahead_h / 3.0))
    window_steps = commit_steps + lookahead_steps

    env = _start_quiet_gurobi_env()
    try:
        for zone_index in range(len(zones)):
            capacity = capacity_by_zone[zone_index]
            if capacity <= 0.0:
                continue
            power_per_step = capacity / battery_duration_h * 3.0  # GWh per 3h step
            start_stored_energy = capacity * initial_soc_fraction
            for start in range(0, step_count, commit_steps):
                residual_window = residual[zone_index, start : start + window_steps]
                if np.all(np.isnan(residual_window)):
                    continue  # state of charge carried over unchanged
                # Gurobi needs finite inputs: solve with NaN as 0, restore NaN afterwards.
                is_missing = np.isnan(residual_window)
                residual_finite = np.where(is_missing, 0.0, residual_window)
                _, net_load, stored_energy = _optimize_battery_gurobi(
                    residual_finite, capacity, power_per_step, one_way_efficiency,
                    min_soc_fraction=min_soc_fraction,
                    initial_soc=start_stored_energy,
                    env=env,
                )
                if net_load is None:
                    continue
                commit = min(commit_steps, len(residual_window))
                committed = slice(start, start + commit)
                adjusted[zone_index, committed] = np.where(is_missing[:commit], np.nan, net_load[:commit])
                stored_energy_trace[zone_index, committed] = stored_energy[:commit]
                # stored_energy[commit] is the storage at the end of the committed part.
                start_stored_energy = stored_energy[commit]
            if verbose:
                print(f"  Zone {zones[zone_index]}: dispatched {capacity:.2f} GWh battery")
    finally:
        env.dispose()


def _dispatch_greedy(residual, adjusted, stored_energy_trace, capacity_by_zone, zones,
                     battery_duration_h, one_way_efficiency, initial_soc_fraction,
                     lookahead_steps, verbose):
    """Greedy charge / rationed discharge dispatch, filling the output arrays in place.

    Parameters
    ----------
    residual : np.ndarray
        ``(zone, time)`` residual load in GWh per 3 h step.
    adjusted, stored_energy_trace : np.ndarray
        ``(zone, time)`` output arrays.
    capacity_by_zone : np.ndarray
        Energy capacity per zone (GWh).
    zones : list
        Zone labels, for messages.
    lookahead_steps : int
        Rationing window in steps; 0 discharges greedily. Otherwise the
        deliverable energy is divided by the number of positive steps in
        ``residual[t : t + lookahead_steps]``, which includes step ``t``.
    battery_duration_h, one_way_efficiency, initial_soc_fraction, verbose
        As in :func:`apply_battery_dispatch` (efficiency is one-way).
    """
    for zone_index in range(len(zones)):
        capacity = capacity_by_zone[zone_index]
        if capacity <= 0.0:
            continue

        power_per_step = capacity / battery_duration_h * 3.0  # GWh per 3h step
        stored_energy = capacity * initial_soc_fraction

        for t in range(residual.shape[1]):
            residual_t = residual[zone_index, t]
            if np.isnan(residual_t):
                stored_energy_trace[zone_index, t] = stored_energy
                continue
            if residual_t < 0.0:
                charge = min(-residual_t, power_per_step, (capacity - stored_energy) / one_way_efficiency)
                stored_energy += charge * one_way_efficiency
                adjusted[zone_index, t] = residual_t + charge
            elif residual_t > 0.0:
                if lookahead_steps > 0:
                    upcoming = residual[zone_index, t : t + lookahead_steps]
                    remaining = max(1, int(np.sum(upcoming > 0.0)))
                    target = stored_energy * one_way_efficiency / remaining
                else:
                    target = stored_energy * one_way_efficiency
                discharge = min(residual_t, power_per_step, target)
                stored_energy -= discharge / one_way_efficiency
                adjusted[zone_index, t] = residual_t - discharge
            stored_energy_trace[zone_index, t] = stored_energy

        if verbose:
            print(f"  Zone {zones[zone_index]}: dispatched {capacity:.2f} GWh battery")


def apply_battery_dispatch(
    residual_dataset: xr.Dataset,
    capacity_gwh: xr.DataArray | pd.Series,
    battery_duration_h: float = 4.0,
    roundtrip_efficiency: float = 0.90,
    initial_soc_fraction: float = 0.50,
    dispatch_lookahead_h: float = 0.0,
    zone_dim: str = "zone",
    residual_load_variable: str = "residual_load",
    method: str = "heuristic",
    min_soc_fraction: float | None = None,
    commit_h: float = 24.0,
    verbose: bool = True,
) -> xr.Dataset:
    """Dispatch a battery in each zone against the residual load.

    Parameters
    ----------
    residual_dataset : xr.Dataset
        Holds ``residual_load_variable`` with dims (``zone_dim``, ``"time"``) in
        GWh per 3 h step.
    capacity_gwh : xr.DataArray or pd.Series
        Battery energy capacity in GWh per zone; absent zones get no battery.
    battery_duration_h : float, optional
        Energy-to-power ratio in hours; the power limit per 3 h step is
        ``capacity / battery_duration_h × 3`` GWh.
    roundtrip_efficiency : float, optional
        Round-trip efficiency (0-1); ``sqrt`` of it is applied on charge and on
        discharge.
    initial_soc_fraction : float, optional
        Initial state of charge as a fraction of capacity (``"heuristic"`` and
        ``"optimal"``).
    dispatch_lookahead_h : float, optional
        ``"heuristic"``: discharge-rationing window in hours (0 = greedy).
        ``"optimal"``: foresight beyond ``commit_h`` that is solved but discarded.
    zone_dim : str, optional
        Zone dimension of ``residual_dataset`` (and of a DataArray ``capacity_gwh``).
    residual_load_variable : str, optional
        Residual-load variable name.
    method : {"heuristic", "optimal", "annual"}, optional
        ``"heuristic"``: charge greedily on surplus, discharge on deficit,
        optionally rationed over the lookahead window. ``"optimal"``: rolling
        day-ahead QP with limited foresight (Gurobi). ``"annual"``:
        perfect-foresight QP per zone and calendar year with cyclic storage,
        the same objective and horizon as :func:`apply_reservoir_dispatch`.
        Both QPs minimise the sum of squared net residual load
        (see :func:`_optimize_battery_gurobi`).
    min_soc_fraction : float, optional
        QP methods only: lower bound on the state of charge as a fraction of
        capacity.
    commit_h : float, optional
        ``"optimal"`` only: hours kept from each solve, and the window step.
    verbose : bool, optional
        Print per-zone progress.

    Returns
    -------
    xr.Dataset
        Copy of the input with ``residual_load_with_batteries`` (GWh per 3 h),
        ``soc_gwh`` (GWh) and ``battery_dispatch_gwh`` (GWh per 3 h; positive =
        discharging). Zones without capacity, and steps left unsolved by a QP,
        keep the input residual load and a state of charge of 0.

    Raises
    ------
    ValueError
        If ``method`` is not one of the three above.
    """
    one_way_efficiency = np.sqrt(roundtrip_efficiency)
    lookahead_steps = max(1, round(dispatch_lookahead_h / 3.0)) if dispatch_lookahead_h > 0.0 else 0

    zones = list(residual_dataset[zone_dim].values)
    capacity_by_zone = _capacity_per_zone(capacity_gwh, zones, zone_dim)

    valid_methods = ("heuristic", "optimal", "annual")
    if method not in valid_methods:
        raise ValueError(
            f"Unknown battery dispatch method {method!r}; expected one of {valid_methods}."
        )

    residual_by_zone = residual_dataset[residual_load_variable].transpose(zone_dim, "time")
    residual = residual_by_zone.values.copy()
    adjusted = residual.copy()
    stored_energy_trace = np.zeros_like(residual)

    if method == "annual":
        _dispatch_annual_qp(
            residual, adjusted, stored_energy_trace, capacity_by_zone, zones,
            pd.DatetimeIndex(residual_by_zone["time"].values),
            battery_duration_h, one_way_efficiency, min_soc_fraction, verbose,
        )
    elif method == "optimal":
        _dispatch_rolling_qp(
            residual, adjusted, stored_energy_trace, capacity_by_zone, zones,
            battery_duration_h, one_way_efficiency, initial_soc_fraction,
            min_soc_fraction, commit_h, dispatch_lookahead_h, verbose,
        )
    else:
        _dispatch_greedy(
            residual, adjusted, stored_energy_trace, capacity_by_zone, zones,
            battery_duration_h, one_way_efficiency, initial_soc_fraction,
            lookahead_steps, verbose,
        )

    return _battery_dispatch_result(residual_dataset, residual_load_variable, residual_by_zone,
                                    residual, adjusted, stored_energy_trace, "battery")


def apply_prosumer_battery_dispatch(
    residual_dataset: xr.Dataset,
    pv_generation: xr.DataArray,
    capacity_gwh: xr.DataArray | pd.Series,
    battery_duration_h: float = 4.0,
    roundtrip_efficiency: float = 0.90,
    initial_soc_fraction: float = 0.50,
    zone_dim: str = "zone",
    residual_load_variable: str = "residual_load",
) -> xr.Dataset:
    """Dispatch prosumer batteries with a PV-aware self-consumption rule.

    In each zone and step: charge only when PV > 0 and the residual load is
    negative; discharge only when PV == 0 and the residual load is positive.
    The state of charge is reset to ``initial_soc_fraction × capacity`` at every
    midnight step after the first.

    Parameters
    ----------
    residual_dataset : xr.Dataset
        Holds ``residual_load_variable`` with dims (``zone_dim``, ``"time"``) in
        GWh per 3 h step.
    pv_generation : xr.DataArray
        PV generation (any non-negative unit) with dims (``zone_dim``, ``"time"``);
        only its sign is used. Zones missing from it count as no sun. It is
        matched to the residual load by position along ``time``, not by label.
    capacity_gwh : xr.DataArray or pd.Series
        Prosumer battery energy capacity in GWh per zone.
    battery_duration_h : float, optional
        Energy-to-power ratio in hours.
    roundtrip_efficiency : float, optional
        Round-trip efficiency (0-1).
    initial_soc_fraction : float, optional
        Initial and daily-reset state of charge as a fraction of capacity.
    zone_dim : str, optional
        Zone dimension.
    residual_load_variable : str, optional
        Residual-load variable name.

    Returns
    -------
    xr.Dataset
        As :func:`apply_battery_dispatch`: ``residual_load_with_batteries``,
        ``soc_gwh``, ``battery_dispatch_gwh``.

    Notes
    -----
    Home batteries run simple self-consumption controllers, and the pipeline
    has no rooftop-PV or household-load signal that would support optimising
    them, so they are never dispatched with the QP.
    """
    one_way_efficiency = np.sqrt(roundtrip_efficiency)

    zones = list(residual_dataset[zone_dim].values)
    capacity_by_zone = _capacity_per_zone(capacity_gwh, zones, zone_dim)

    residual_by_zone = residual_dataset[residual_load_variable].transpose(zone_dim, "time")
    residual = residual_by_zone.values.copy()

    pv_by_zone = pv_generation.transpose(zone_dim, "time")
    pv_zones = list(pv_by_zone[zone_dim].values)
    if pv_zones == zones:
        pv = pv_by_zone.values
    else:
        pv = np.zeros_like(residual)
        for zone_index, zone in enumerate(zones):
            if zone in pv_zones:
                pv[zone_index] = pv_by_zone.sel({zone_dim: zone}).values

    hour_of_day = pd.DatetimeIndex(residual_by_zone["time"].values).hour

    adjusted = residual.copy()
    stored_energy_trace = np.zeros_like(residual)

    for zone_index in range(len(zones)):
        capacity = capacity_by_zone[zone_index]
        if capacity <= 0.0:
            continue

        power_per_step = capacity / battery_duration_h * 3.0
        stored_energy = capacity * initial_soc_fraction

        for t in range(residual.shape[1]):
            if t > 0 and hour_of_day[t] == 0:
                stored_energy = capacity * initial_soc_fraction

            residual_t = residual[zone_index, t]
            if np.isnan(residual_t):
                stored_energy_trace[zone_index, t] = stored_energy
                continue

            sun_up = pv[zone_index, t] > 0.0

            if sun_up and residual_t < 0.0:
                charge = min(-residual_t, power_per_step, (capacity - stored_energy) / one_way_efficiency)
                stored_energy += charge * one_way_efficiency
                adjusted[zone_index, t] = residual_t + charge
            elif (not sun_up) and residual_t > 0.0:
                discharge = min(residual_t, power_per_step, stored_energy * one_way_efficiency)
                stored_energy -= discharge / one_way_efficiency
                adjusted[zone_index, t] = residual_t - discharge

            stored_energy_trace[zone_index, t] = stored_energy

    return _battery_dispatch_result(residual_dataset, residual_load_variable, residual_by_zone,
                                    residual, adjusted, stored_energy_trace, "prosumer battery")


# ---------------------------------------------------------------------------
# Reservoir hydro dispatch
# ---------------------------------------------------------------------------


def _optimize_reservoir_gurobi(
    residual_load: np.ndarray,
    inflow: np.ndarray,
    energy_capacity: float,
    power_capacity: float,
    min_storage_level_fraction: float | None = None,
    efficiency: float = 0.9,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, int]:
    """Solve the annual reservoir dispatch QP with Gurobi.

    Minimises ``Σ (residual_load[t] − generation[t])²`` subject to
    ``storage[t+1] = storage[t] + inflow[t] − generation[t]/efficiency − spillage[t]``,
    ``0 ≤ generation ≤ power_capacity``, ``spillage ≥ 0``, storage bounds and
    ``storage[0] = storage[T]``.

    Parameters
    ----------
    residual_load : np.ndarray
        Residual load for one calendar year (GWh per 3 h step).
    inflow : np.ndarray
        Inflow over the same steps (GWh per 3 h step).
    energy_capacity : float
        Reservoir storage capacity (GWh).
    power_capacity : float
        Maximum generation per 3 h step (GWh = installed GW × 3).
    min_storage_level_fraction : float, optional
        Lower bound on storage as a fraction of ``energy_capacity``.
    efficiency : float, optional
        Turbine (water-to-wire) efficiency in (0, 1]: generating ``G`` GWh of
        electricity draws ``G / efficiency`` from storage.

    Returns
    -------
    generation, net_load, storage : np.ndarray or None
        Arrays of length T (``storage[k]`` is the level entering step ``k``), or
        None each if Gurobi found no solution.
    status : int
        Gurobi status code.

    Notes
    -----
    ``NumericFocus = 3`` is set because a year of residual load can be
    ill-conditioned. Any feasible incumbent is accepted even if not certified
    optimal (SUBOPTIMAL/NUMERIC): it satisfies the water balance exactly, which
    is preferable to leaving the country-year undispatched.
    """
    import gurobipy as gp
    from gurobipy import GRB

    horizon = len(residual_load)
    storage_floor = (energy_capacity * min_storage_level_fraction
                     if min_storage_level_fraction is not None else 0.0)

    env = _start_quiet_gurobi_env()
    model = gp.Model("Reservoir_Dispatch", env=env)

    storage = model.addMVar(shape=horizon + 1, lb=storage_floor, ub=energy_capacity, name="E")
    generation = model.addMVar(shape=horizon, lb=0.0, ub=power_capacity, name="G")
    spillage = model.addMVar(shape=horizon, lb=0.0, name="S")
    net_load = model.addMVar(shape=horizon, lb=-GRB.INFINITY, name="net_load")

    model.addConstr(
        storage[1:] == storage[:-1] + inflow - generation / efficiency - spillage,
        name="Storage_Balance",
    )
    model.addConstr(storage[0] == storage[horizon], name="Cyclicity")
    model.addConstr(net_load == residual_load - generation, name="Net_Load_Def")
    model.setObjective(net_load @ net_load, GRB.MINIMIZE)
    model.Params.NumericFocus = 3
    model.optimize()

    if model.Status == GRB.OPTIMAL or model.SolCount > 0:
        return generation.X, net_load.X, storage.X[:-1], model.Status
    return None, None, None, model.Status


def apply_reservoir_dispatch(
    residual_dataset: xr.Dataset,
    inflow_dataset: xr.Dataset,
    hydropower_capacities: pd.DataFrame,
    power_column: str = "p_nom_discharge[GW]",
    energy_column: str = "E_store[TWh]",
    min_storage_level_fraction: float | None = None,
    efficiency: float = 0.9,
    residual_load_variable: str = "residual_load",
    country_dim: str = "country",
    verbose: bool = True,
) -> xr.Dataset:
    """Dispatch reservoir hydro per country and calendar year against the residual load.

    Each country-year is solved by :func:`_optimize_reservoir_gurobi`
    (perfect foresight within the year, cyclic storage, no carry-over between
    years).

    Parameters
    ----------
    residual_dataset : xr.Dataset
        Country residual load, dims ``(country, time)``, GWh per 3 h step.
    inflow_dataset : xr.Dataset
        Weekly reservoir inflow in GWh/week, variable ``"inflow"``, ISO2
        country codes. It is linearly interpolated to 3 h and divided by 56
        (7 days × 8 steps).
    hydropower_capacities : pd.DataFrame
        Reservoir capacities indexed by ISO2 code.
    power_column : str, optional
        Column with turbine power in GW.
    energy_column : str, optional
        Column with storage capacity in TWh.
    min_storage_level_fraction : float, optional
        Lower bound on storage as a fraction of capacity.
    efficiency : float, optional
        Turbine (water-to-wire) efficiency in (0, 1].
    residual_load_variable : str, optional
        Residual-load variable name.
    country_dim : str, optional
        Country dimension.
    verbose : bool, optional
        Print per-country progress.

    Returns
    -------
    xr.Dataset
        Copy of the input with ``residual_load_variable`` replaced by the
        dispatched series and ``reservoir_dispatch_gwh`` (GWh per 3 h) and
        ``reservoir_storage_gwh`` (GWh) added. Countries without inflow, with
        non-positive capacity, or whose solve fails keep their residual load.
    """
    original_dims = residual_dataset[residual_load_variable].dims
    residual_by_country = residual_dataset[residual_load_variable].transpose(country_dim, "time")
    residual = residual_by_country.values.copy()
    time_index = pd.DatetimeIndex(residual_dataset.time.values)
    all_countries = list(residual_dataset[country_dim].values)
    inflow_countries = set(inflow_dataset[country_dim].values)

    dispatch = np.zeros_like(residual)
    storage = np.zeros_like(residual)

    for country in hydropower_capacities.index:
        if country not in inflow_countries or country not in all_countries:
            continue

        country_index = all_countries.index(country)

        energy_capacity = float(hydropower_capacities.loc[country, energy_column]) * 1000  # TWh -> GWh
        power_per_step = float(hydropower_capacities.loc[country, power_column]) * 3.0  # GW -> GWh per 3h

        if energy_capacity <= 0 or power_per_step <= 0:
            continue

        inflow_weekly = inflow_dataset["inflow"].sel({country_dim: country}).to_series()
        inflow_per_step = inflow_weekly.resample("3h").interpolate(method="linear") / (7 * 8)

        if verbose:
            print(f"  Reservoir dispatch: {country}  E_max={energy_capacity:.0f} GWh  "
                  f"P_max={power_per_step / 3:.1f} GW")

        for year in np.unique(time_index.year):
            in_year = time_index.year == year
            residual_year = residual[country_index, in_year]

            inflow_year = (
                inflow_per_step
                .reindex(time_index[in_year], method="nearest")
                .bfill()
                .ffill()
                .fillna(0.0)
                .values
            )

            horizon = min(len(residual_year), len(inflow_year))
            generation, net_load, storage_level, status = _optimize_reservoir_gurobi(
                residual_year[:horizon], inflow_year[:horizon], energy_capacity, power_per_step,
                min_storage_level_fraction,
                efficiency=efficiency,
            )

            if generation is None:
                if verbose:
                    print(f"    {year}: Gurobi infeasible (status={status}), skipping")
                continue

            positions = np.where(in_year)[0][:horizon]
            residual[country_index, positions] = net_load
            dispatch[country_index, positions] = generation
            storage[country_index, positions] = storage_level

    def as_output(values, long_name, units):
        """Wrap a ``(country, time)`` array in the original dimension order."""
        return xr.DataArray(
            values,
            dims=[country_dim, "time"],
            coords=residual_by_country.coords,
            attrs={"long_name": long_name, "units": units},
        ).transpose(*original_dims)

    result = residual_dataset.copy()
    result[residual_load_variable] = as_output(
        residual,
        "Residual load after reservoir hydro dispatch",
        residual_dataset[residual_load_variable].attrs.get("units", "GWh"),
    )
    result["reservoir_dispatch_gwh"] = as_output(
        dispatch, "Reservoir hydro dispatch", "GWh per 3h step"
    )
    result["reservoir_storage_gwh"] = as_output(
        storage, "Reservoir storage level", "GWh"
    )
    return result


# ---------------------------------------------------------------------------
# Installed capacities of the modelled countries
# ---------------------------------------------------------------------------

#: Supply technologies (GW) and storage technologies (GWh); never summed together.
GENERATION_TECHNOLOGIES = ["solar PV", "wind onshore", "wind offshore"]
BATTERY_TECHNOLOGIES = ["battery (prosumer)", "battery (utility)"]
#: The countries the residual-load model covers.
MODELLED_COUNTRIES = sorted(COUNTRY_ZONE)


def installed_capacities_by_country(supply_workbook, scenario: str, year: int) -> pd.DataFrame:
    """Installed capacity of every modelled country under one TYNDP build-out.

    Parameters
    ----------
    supply_workbook : path-like
        The TYNDP 2024 supply-input workbook.
    scenario : str
        TYNDP supply scenario, e.g. ``"LOW"`` or ``"HIGH"``.
    year : int
        Target year of the build-out.

    Returns
    -------
    pandas.DataFrame
        One row per modelled country (ISO2), one column per technology: solar and wind in GW,
        the two battery types in GWh. Countries without a technology hold 0.
    """
    solar, wind_onshore, wind_offshore = load_tyndp_supply_capacities(supply_workbook)
    column = f"{year}_{scenario}"

    def by_iso2(nodes: pd.DataFrame) -> pd.Series:
        """Sum TYNDP nodes (e.g. ``DE00``, ``DKW1``) into countries, MW -> GW."""
        nodes = nodes[nodes["Country"].astype(str).str.len() >= 2].copy()
        nodes["ISO2"] = nodes["Country"].astype(str).str[:2]
        return nodes.groupby("ISO2")[column].sum() / 1000.0

    prosumer, utility = load_tyndp_battery_capacities(
        supply_workbook, scenario=scenario, year=year, return_separate=True)
    capacities = pd.DataFrame({"solar PV": by_iso2(solar), "wind onshore": by_iso2(wind_onshore),
                               "wind offshore": by_iso2(wind_offshore),
                               "battery (prosumer)": prosumer, "battery (utility)": utility})
    return capacities.reindex(MODELLED_COUNTRIES).fillna(0.0)


def sum_by_zone(by_country: pd.DataFrame, zone_order) -> pd.DataFrame:
    """Sum country rows into balancing zones.

    Parameters
    ----------
    by_country : pandas.DataFrame
        One row per country (ISO2).
    zone_order : sequence of str
        Zones in the order the rows should come out.

    Returns
    -------
    pandas.DataFrame
        One row per zone, in ``zone_order``; zones with no country hold 0.
    """
    by_zone = by_country.groupby(by_country.index.map(COUNTRY_ZONE)).sum()
    return by_zone.reindex(zone_order).fillna(0.0)


def compare_build_outs(capacities: dict, reference: str, comparison: str, year: int,
                       zone_order) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Europe-wide and zonal difference between two build-outs.

    Parameters
    ----------
    capacities : dict
        ``{scenario: capacities by country}`` from :func:`installed_capacities_by_country`.
    reference, comparison : str
        The two scenarios, e.g. ``"LOW"`` and ``"HIGH"``.
    year : int
        Target year, used in the column labels.
    zone_order : sequence of str
        Zone order of the zonal tables.

    Returns
    -------
    totals : pandas.DataFrame
        Europe-wide capacity per technology under each scenario, and ``change %``.
    added : pandas.DataFrame
        Capacity the comparison scenario adds, per zone and technology.
    reference_by_zone : pandas.DataFrame
        The reference build-out per zone and technology.
    """
    totals = pd.DataFrame({f"{scenario} / {year}": capacities[scenario].sum()
                           for scenario in (reference, comparison)})
    totals["change %"] = (totals.iloc[:, 1] / totals.iloc[:, 0] - 1) * 100
    reference_by_zone = sum_by_zone(capacities[reference], zone_order)
    added = sum_by_zone(capacities[comparison], zone_order) - reference_by_zone
    return totals, added, reference_by_zone
