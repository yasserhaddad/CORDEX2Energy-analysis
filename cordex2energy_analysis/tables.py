"""Tables derived from the run and the energy fields, shared by the figures and the text.

Every change here is *paired*: each chain's own change between two warming levels is formed
first, and only then is the median (or any other statistic) taken across chains.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SEASON_MONTHS, ZONE_LABELS, ZONE_ORDER
from .fields import SUPPLY_VARIABLES, country_field, demand_field
from .power_system import REALTIME_BALANCING_ZONES

#: Event-duration bins (hours) of the duration spectrum, and their labels.
DURATION_EDGES = [0, 6, 12, 24, 48, 96, 192, 384, 1e4]
DURATION_BIN_LABELS = ["0-6 h", "6-12 h", "12-24 h", "1-2 d", "2-4 d", "4-8 d", "8-16 d", "> 16 d"]
#: Years in every warming-level window.
WINDOW_YEARS = 20.0
#: Columns of the Europe-wide change figure and tables: the year, winter and summer.
EUROPE_COLUMNS = {"Annual": None, "DJF": SEASON_MONTHS["DJF"], "JJA": SEASON_MONTHS["JJA"]}


# ---------------------------------------------------------------------------
# Change of a country field, by zone and Europe-wide (the numbers behind Figures 2 and 3)
# ---------------------------------------------------------------------------

def group_change(field, reference: str, warmest: str, members=None) -> tuple[float, float]:
    """Median across chains of the per-chain change of a country field summed over a group.

    Parameters
    ----------
    field : callable
        ``field(period) -> DataArray (model, country)``.
    reference, warmest : str
        The two warming levels.
    members : sequence of str, optional
        Countries (ISO2) of the group; ``None`` sums every country the field carries.

    Returns
    -------
    percent : float
        Median per-cent change. The denominator is the absolute reference total, so a shrinking
        surplus reads as a positive change.
    absolute : float
        Median absolute change, in the field's unit.
    """
    reference_field, warm_field = field(reference), field(warmest)
    if members is not None:
        available = set(map(str, reference_field["country"].values))
        present = [country for country in members if country in available]
        if not present:
            return np.nan, np.nan
        reference_field = reference_field.sel(country=present)
        warm_field = warm_field.sel(country=present)
    reference_total = reference_field.sum("country", skipna=True)
    warm_total = warm_field.sum("country", skipna=True)
    absolute = warm_total - reference_total
    percent = 100 * absolute / np.abs(reference_total)
    return float(percent.median("model")), float(absolute.median("model"))


def zone_change_table(quantities, reference: str, warmest: str, absolute_rows=()) -> pd.DataFrame:
    """Change of every mapped quantity per balancing zone and Europe-wide.

    Parameters
    ----------
    quantities : sequence of tuple
        ``(name, field, ...)`` as drawn in the maps; ``field(period)`` gives a country field.
    reference, warmest : str
        The two warming levels.
    absolute_rows : sequence of str
        Quantities reported as an absolute change (TWh per year) rather than in per cent.

    Returns
    -------
    pandas.DataFrame
        One row per zone plus ``Europe``, one column per quantity.
    """
    groups = [*((ZONE_LABELS[zone], REALTIME_BALANCING_ZONES[zone]) for zone in ZONE_ORDER),
              ("Europe", None)]
    rows = {}
    for group, members in groups:
        rows[group] = {}
        for name, field, *_ in quantities:
            percent, absolute = group_change(field, reference, warmest, members)
            rows[group][name] = absolute if name in absolute_rows else percent
    return pd.DataFrame(rows).T[[name for name, *_ in quantities]]


# ---------------------------------------------------------------------------
# The Europe-wide change, demand against supply (Figure 4, Table 1, S24)
# ---------------------------------------------------------------------------

def europe_change_rows(country_output: dict, demand_by_month: dict, period_windows: dict) -> list:
    """The quantities of the Europe-wide change figure, as ``(label, field)`` pairs.

    Heating and cooling come from the demand product, restricted to the countries of the
    residual-load run so that every row covers the same domain; the other rows come from the run.

    Parameters
    ----------
    country_output : dict
        Per-country run output (:func:`~cordex2energy_analysis.fields.load_country_output`).
    demand_by_month : dict
        Demand product (:func:`~cordex2energy_analysis.fields.load_demand_by_month`).
    period_windows : dict
        ``{warming level: {chain: (first year, last year)}}``.

    Returns
    -------
    list of tuple
        ``(label, field)`` with ``field(period, months) -> DataArray (model, country)`` in TWh.
    """
    run_countries = [str(country)
                     for country in next(iter(country_output.values()))["country"].values]

    def from_run(expression):
        """Field builder for a quantity computed from the run outputs."""
        return lambda period, months: country_field(country_output, period_windows, expression,
                                                    period, months)

    def from_demand(variable):
        """Field builder for a demand variable, restricted to the run's countries."""
        def field(period, months):
            """Demand field over chosen months; fails if a run country is missing."""
            values = demand_field(demand_by_month, period_windows, variable, period, months)
            missing = set(run_countries) - set(map(str, values["country"].values))
            if missing:
                raise ValueError(f"run countries missing from the demand product: {sorted(missing)}")
            return values.sel(country=run_countries)
        return field

    return [
        ("Heating demand", from_demand("heating")),
        ("Cooling demand", from_demand("cooling")),
        ("Total demand", from_run(lambda data: data["total_demand"])),
        ("Solar", from_run(lambda data: data["gen_pv"])),
        ("Wind", from_run(lambda data: data["gen_wind_on"] + data["gen_wind_off"])),
        ("Hydro", from_run(lambda data: data["gen_hydro_ror"] + data["reservoir_dispatch_gwh"])),
        ("Total supply", from_run(lambda data: sum(data[variable] for variable in SUPPLY_VARIABLES))),
        ("Residual load", from_run(lambda data: data["residual_load"])),
    ]


# ---------------------------------------------------------------------------
# Event duration (Figures 5 and 7, S29)
# ---------------------------------------------------------------------------

def duration_spectrum(storage_events: pd.DataFrame, value: str, by=("model", "period")) -> pd.DataFrame:
    """Events, or their severity, per year in each duration bin.

    Parameters
    ----------
    storage_events : pandas.DataFrame
        One row per event, with ``duration_h`` and ``sev_actual`` (GWh), from
        :func:`~cordex2energy_analysis.runs.event_tables`.
    value : {"counts", "severity"}
        Events per year, or severity in TWh per year.
    by : tuple of str
        Columns to keep, e.g. ``("model", "period", "zone")`` for a zonal spectrum.

    Returns
    -------
    pandas.DataFrame
        The ``by`` columns, ``bin`` and ``value``; one row per group and bin.
    """
    rows = []
    for keys, block in storage_events.groupby(list(by)):
        keys = keys if isinstance(keys, tuple) else (keys,)
        for low, high, label in zip(DURATION_EDGES[:-1], DURATION_EDGES[1:], DURATION_BIN_LABELS):
            inside = block[(block.duration_h > low) & (block.duration_h <= high)]
            amount = len(inside) if value == "counts" else inside.sev_actual.sum() / 1e3
            rows.append({**dict(zip(by, keys)), "bin": label, "value": amount / WINDOW_YEARS})
    return pd.DataFrame(rows)


def severity_by_duration(storage_events: pd.DataFrame) -> pd.DataFrame:
    """Severity storage balanced, and what it left, per year in each duration bin.

    Parameters
    ----------
    storage_events : pandas.DataFrame
        One row per event with ``duration_h``, ``sev_actual`` (with storage) and
        ``sev_no_storage`` (without), in GWh.

    Returns
    -------
    pandas.DataFrame
        Columns ``model, period, bin, sev_actual, absorbed``, in TWh per year.
    """
    events = storage_events.assign(absorbed=storage_events.sev_no_storage - storage_events.sev_actual)
    rows = []
    for (model, period), block in events.groupby(["model", "period"]):
        for low, high, label in zip(DURATION_EDGES[:-1], DURATION_EDGES[1:], DURATION_BIN_LABELS):
            inside = block[(block.duration_h > low) & (block.duration_h <= high)]
            rows.append(dict(model=model, period=period, bin=label,
                             sev_actual=inside.sev_actual.sum() / WINDOW_YEARS / 1e3,
                             absorbed=inside.absorbed.sum() / WINDOW_YEARS / 1e3))
    return pd.DataFrame(rows)


def storage_share_table(storage_attenuation: pd.DataFrame, period: str, duration_bins,
                        class_labels: dict, zones) -> pd.DataFrame:
    """Share of severity balanced by reservoirs and batteries, per zone and duration class.

    Parameters
    ----------
    storage_attenuation : pandas.DataFrame
        From :func:`~cordex2energy_analysis.runs.event_tables`.
    period : str
        Warming level.
    duration_bins : sequence of str
        Duration classes, in column order.
    class_labels : dict
        Short label per duration class.
    zones : sequence of str
        Zones, in row order.

    Returns
    -------
    pandas.DataFrame
        Median across chains, in per cent; columns ``(class, reservoir | battery | combined)``.
    """
    table = {}
    for duration_class in duration_bins:
        label = class_labels[duration_class]
        for column, part in (("reservoir_attenuation_share_pct", "reservoir"),
                             ("battery_attenuation_share_pct", "battery")):
            table[(label, part)] = {
                ZONE_LABELS[zone]: storage_attenuation[
                    (storage_attenuation.duration_class == duration_class)
                    & (storage_attenuation.period == period)
                    & (storage_attenuation.zone == zone)][column].median()
                for zone in zones}
    table = pd.DataFrame(table)
    for duration_class in duration_bins:
        label = class_labels[duration_class]
        table[(label, "combined")] = table[(label, "reservoir")] + table[(label, "battery")]
    columns = [(class_labels[duration_class], part) for duration_class in duration_bins
               for part in ("reservoir", "battery", "combined")]
    table = table[columns]
    table.columns = pd.MultiIndex.from_tuples(table.columns)
    return table


# ---------------------------------------------------------------------------
# Paired changes and the build-out comparison (Figure 8)
# ---------------------------------------------------------------------------

def paired_increment(table: pd.DataFrame, zone: str, value_column: str, reference: str,
                     warmest: str, duration_class: str | None = None):
    """Median of each chain's own change between two warming levels, and how many chains agree.

    Parameters
    ----------
    table : pandas.DataFrame
        Rows per ``model, period, zone`` (and ``duration_class``).
    zone : str
        Zone to select.
    value_column : str
        Column holding the quantity.
    reference, warmest : str
        The two warming levels.
    duration_class : str, optional
        Duration class to select.

    Returns
    -------
    median : float
        Median per-chain change.
    agree : int
        Chains whose change has the median's sign. A zero median counts the chains that are
        themselves unchanged.
    reference_median, warm_median : float
        Medians of the two levels over the same chains.
    """
    data = table if duration_class is None else table[table.duration_class == duration_class]
    levels = {period: data[(data.zone == zone) & (data.period == period)].set_index("model")[value_column]
              for period in (reference, warmest)}
    shared = levels[reference].index.intersection(levels[warmest].index)
    change = levels[warmest].loc[shared] - levels[reference].loc[shared]
    median = float(change.median())
    agree = int((np.sign(change) == np.sign(median)).sum())
    return (median, agree, float(levels[reference].loc[shared].median()),
            float(levels[warmest].loc[shared].median()))


def scenario_table(tables_by_scenario: dict, value_column: str, reference: str, warmest: str,
                   zones, n_models: int, duration_class: str | None = None,
                   digits: int = 2) -> pd.DataFrame:
    """Per zone, each build-out's two levels, their paired change and its agreement count.

    Parameters
    ----------
    tables_by_scenario : dict
        ``{scenario label: table}`` (labels starting ``LOW`` or ``HIGH``).
    value_column : str
        Column holding the quantity.
    reference, warmest : str
        The two warming levels.
    zones : sequence of str
        Zones, in row order.
    n_models : int
        Number of chains, for the ``n/N`` agreement column.
    duration_class : str, optional
        Duration class to select.
    digits : int
        Decimals of the formatted values.

    Returns
    -------
    pandas.DataFrame
        Formatted strings; for ``share_of_demand_pct`` also the damping ratio HIGH/LOW.
    """
    rows = {}
    for zone in zones:
        entry, increments = {}, {}
        for label, data in tables_by_scenario.items():
            median, agree, low, high = paired_increment(data, zone, value_column, reference,
                                                        warmest, duration_class)
            short = "LOW" if label.startswith("LOW") else "HIGH"
            entry[(short, reference)] = f"{low:.{digits}f}"
            entry[(short, warmest)] = f"{high:.{digits}f}"
            entry[(short, "Δ")] = f"{median:+.{digits}f}"
            entry[(short, "n/9")] = f"{agree}/{n_models}"
            increments[short] = median
        if value_column == "share_of_demand_pct":
            entry[("", "damping")] = (f"×{increments['HIGH'] / increments['LOW']:.2f}"
                                      if abs(increments["LOW"]) > 1e-9 else "n/a")
        rows[ZONE_LABELS[zone]] = entry
    frame = pd.DataFrame(rows).T
    frame.columns = pd.MultiIndex.from_tuples(frame.columns)
    return frame


# ---------------------------------------------------------------------------
# Supplementary Note 3
# ---------------------------------------------------------------------------

def paired_percent_change(by_season: pd.DataFrame, reference: str, warmest: str) -> pd.Series:
    """Per-chain per-cent change between two warming levels of a long table.

    Parameters
    ----------
    by_season : pandas.DataFrame
        Columns ``model, season, quantity, period, per_year``.
    reference, warmest : str
        The two warming levels.

    Returns
    -------
    pandas.Series
        Indexed by ``(model, season, quantity)``.
    """
    wide = by_season.pivot_table(index=["model", "season", "quantity"], columns="period",
                                 values="per_year")
    return ((wide[warmest] - wide[reference]) / wide[reference] * 100).rename("pct")


def zones_with_events(events_by_season: pd.DataFrame, zones, duration_bins, period_names,
                      season: str = "DJF") -> list:
    """Zones with at least one event of any duration class in a season, at any warming level.

    Parameters
    ----------
    events_by_season : pandas.DataFrame
        Events per chain, level, zone, season and duration class.
    zones : sequence of str
        Candidate zones, in output order.
    duration_bins : sequence of str
        Duration classes.
    period_names : sequence of str
        Warming levels.
    season : str
        Season to test.

    Returns
    -------
    list of str
        The zones whose median event frequency is positive in some class and level.
    """
    in_season = events_by_season[events_by_season.season == season]
    median = in_season.groupby(["zone", "duration_class", "period"])["freq_per_year"].median()
    return [zone for zone in zones
            if any(float(median.get((zone, duration_class, period), 0.0) or 0.0) > 0
                   for duration_class in duration_bins for period in period_names)]


def event_anomaly(anomaly_table: pd.DataFrame, block: str, zone: str, season: str, period: str,
                  drivers) -> pd.DataFrame:
    """Per-chain driver anomalies during events, as the driver-attribution panel draws them.

    Parameters
    ----------
    anomaly_table : pandas.DataFrame
        From :func:`~cordex2energy_analysis.extremes_analysis.event_anomaly_vs_hourly_climatology`.
    block : {"weather", "total", "diurnal"}
        ``"weather"``: against the season mean for each step's hour of day; ``"total"``: against
        the plain season mean; ``"diurnal"``: their difference.
    zone, season, period : str
        Cell to select.
    drivers : sequence of str
        Driver columns.

    Returns
    -------
    pandas.DataFrame
        One row per chain, one column per driver plus ``net`` (GWh per 3h step; positive deepens
        the deficit).
    """
    prefix = {"weather": "", "total": "TOT_", "diurnal": "DIU_"}[block]
    rows = anomaly_table[(anomaly_table.zone == zone) & (anomaly_table.season == season)
                         & (anomaly_table.period == period)].set_index("model")
    table = pd.DataFrame({driver: rows[f"{prefix}{driver}"] for driver in drivers})
    table["net"] = rows[f"{prefix}net"]
    return table
