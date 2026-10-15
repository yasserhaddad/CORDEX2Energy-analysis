"""Builders of the Supplementary Note 2 figures (S17-S22): the power system.

Each ``figure_sN`` function takes the tables it draws and returns a matplotlib figure at the
journal's artwork size; the notebooks only load the data, call the builder and save the result.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from .config import ZONE_LABELS, ZONE_ORDER
from .power_system import BATTERY_TECHNOLOGIES, GENERATION_TECHNOLOGIES, REALTIME_BALANCING_ZONES
from .plotting_publication import (
    FS,
    GWL_TITLE,
    MM,
    SCENARIO_COLORS,
    TECH_COLORS,
    WIDTH,
    ZONE_COLORS,
    fit_width,
    gwl_handles,
    legend_below,
    panel_letters,
)

#: Day of year (0-based) on which each month starts, in a 365-day year.
MONTH_STARTS = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
#: Countries with a reservoir fleet, as their panels are titled.
COUNTRY_NAMES = {"AT": "Austria", "BG": "Bulgaria", "CH": "Switzerland", "ES": "Spain",
                 "FR": "France", "IT": "Italy", "NO": "Norway", "PT": "Portugal",
                 "RO": "Romania", "SE": "Sweden"}


# ---------------------------------------------------------------------------
# Supplementary Note 2: the power system
# ---------------------------------------------------------------------------

def stacked_capacity(ax, capacities: pd.DataFrame, ylabel: str, *, order=None, labels=None,
                     rotation: float = 90, legend: bool = True) -> pd.DataFrame:
    """One stacked bar of installed capacity per row, technologies in the paper's colours.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    capacities : pandas.DataFrame
        One row per bar (country or zone), one column per technology.
    ylabel : str
        Y-axis label, with its unit.
    order : sequence, optional
        Row order. By default the bars are sorted by total, largest first.
    labels : sequence of str, optional
        Tick labels; by default the row index.
    rotation : float
        Rotation of the tick labels, in degrees.
    legend : bool
        Whether to draw the technology legend (top right, the emptiest corner of sorted bars).

    Returns
    -------
    pandas.DataFrame
        ``capacities`` in the order drawn.
    """
    capacities = capacities.loc[order if order is not None
                                else capacities.sum(axis=1).sort_values(ascending=False).index]
    bottoms = np.zeros(len(capacities))
    for technology in capacities.columns:
        ax.bar(range(len(capacities)), capacities[technology].values, bottom=bottoms, width=0.78,
               color=TECH_COLORS[technology], edgecolor="white", lw=0.2, label=technology, zorder=2)
        bottoms += capacities[technology].values
    ax.set_xticks(range(len(capacities)))
    ax.set_xticklabels(labels if labels is not None else list(capacities.index),
                       fontsize=FS["tick"], rotation=rotation,
                       ha="right" if 0 < rotation < 90 else "center")
    ax.set_xlim(-0.8, len(capacities) - 0.2)
    ax.set_ylabel(ylabel, fontsize=FS["label"])
    ax.tick_params(labelsize=FS["tick"])
    ax.grid(True, axis="y", color="0.93", lw=0.3)
    ax.set_axisbelow(True)
    if legend:
        ax.legend(fontsize=FS["tick"], frameon=False, loc="upper right", handlelength=1.0,
                  handletextpad=0.5, borderaxespad=0.2)
    return capacities


def figure_s17(capacities: pd.DataFrame, capacities_by_zone: pd.DataFrame):
    """S17: the LOW/2050 build-out per country (a, b) and per balancing zone (c, d).

    Parameters
    ----------
    capacities : pandas.DataFrame
        Installed capacity per modelled country (see
        :func:`~cordex2energy_analysis.power_system.installed_capacities_by_country`).
    capacities_by_zone : pandas.DataFrame
        The same summed into zones, in the paper's zone order.

    Returns
    -------
    matplotlib.figure.Figure
    """
    # a and b take the full width: 37 rotated country codes at 7 pt need ~2.5 mm each.
    fig = plt.figure(figsize=(WIDTH, 150 * MM))
    grid = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 0.95], hspace=0.62, wspace=0.22)
    ax_a, ax_b = fig.add_subplot(grid[0, :]), fig.add_subplot(grid[1, :])
    ax_c, ax_d = fig.add_subplot(grid[2, 0]), fig.add_subplot(grid[2, 1])

    stacked_capacity(ax_a, capacities[GENERATION_TECHNOLOGIES], "Installed capacity (GW)")
    stacked_capacity(ax_b, capacities[BATTERY_TECHNOLOGIES], "Energy capacity (GWh)")
    # c and d repeat the colours of a and b, so they carry no legend of their own.
    zone_ticks = [ZONE_LABELS[zone] for zone in ZONE_ORDER]
    stacked_capacity(ax_c, capacities_by_zone[GENERATION_TECHNOLOGIES], "Installed capacity (GW)",
                     order=ZONE_ORDER, labels=zone_ticks, rotation=30, legend=False)
    stacked_capacity(ax_d, capacities_by_zone[BATTERY_TECHNOLOGIES], "Energy capacity (GWh)",
                     order=ZONE_ORDER, labels=zone_ticks, rotation=30, legend=False)

    fit_width(fig)
    panel_letters(fig, [("a", ax_a), ("b", ax_b), ("c", ax_c), ("d", ax_d)])
    return fig


def figure_s18(totals: pd.DataFrame, added_by_zone: pd.DataFrame, reference_by_zone: pd.DataFrame):
    """S18: what the HIGH/2050 build-out adds over LOW/2050, Europe-wide (a) and by zone (b, c).

    Parameters
    ----------
    totals : pandas.DataFrame
        Europe-wide capacity per technology under each build-out, plus ``change %``
        (from :func:`~cordex2energy_analysis.power_system.compare_build_outs`).
    added_by_zone : pandas.DataFrame
        Capacity HIGH adds, per zone and technology.
    reference_by_zone : pandas.DataFrame
        The LOW build-out per zone and technology, the base of the percentage labels.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig = plt.figure(figsize=(WIDTH, 100 * MM))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.94], hspace=0.60, wspace=0.20)
    # a is two axes under one letter: generation is power (GW) and batteries are energy (GWh).
    top = grid[0, :].subgridspec(1, 2, width_ratios=[len(GENERATION_TECHNOLOGIES),
                                                     len(BATTERY_TECHNOLOGIES)], wspace=0.28)
    ax_a, ax_a_battery = fig.add_subplot(top[0, 0]), fig.add_subplot(top[0, 1])
    ax_b, ax_c = fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])

    scenarios = list(totals.columns[:2])
    for ax, technologies, ylabel in ((ax_a, GENERATION_TECHNOLOGIES, "Generation capacity (GW)"),
                                     (ax_a_battery, BATTERY_TECHNOLOGIES, "Battery capacity (GWh)")):
        rows = totals.loc[technologies]
        positions = np.arange(len(technologies))
        for k, scenario in enumerate(scenarios):
            ax.bar(positions + (k - 0.5) * 0.38, rows[scenario].values, 0.36,
                   color=SCENARIO_COLORS[k], edgecolor="white", lw=0.3, label=scenario, zorder=2)
        for i, technology in enumerate(technologies):
            ax.annotate(f"+{rows.loc[technology, 'change %']:.0f} %",
                        xy=(i, max(rows.loc[technology, scenarios])),
                        xytext=(0, 2), textcoords="offset points", ha="center", va="bottom",
                        fontsize=FS["tick"], fontweight="bold", color="0.25")
        ax.set_ylim(0, ax.get_ylim()[1] * 1.22)
        ax.set_xticks(positions)
        ax.set_xticklabels(technologies, fontsize=FS["tick"])
        ax.set_xlim(-0.6, len(technologies) - 0.4)
        ax.set_ylabel(ylabel, fontsize=FS["label"])
        ax.tick_params(labelsize=FS["tick"])
        ax.grid(True, axis="y", color="0.93", lw=0.3)
        ax.set_axisbelow(True)

    # b, c: ADDED capacity per zone, labelled with the zone's percentage growth.
    zone_ticks = [ZONE_LABELS[zone] for zone in ZONE_ORDER]
    for ax, technologies, ylabel in ((ax_b, GENERATION_TECHNOLOGIES, "Added capacity (GW)"),
                                     (ax_c, BATTERY_TECHNOLOGIES, "Added energy (GWh)")):
        added = added_by_zone[technologies].loc[ZONE_ORDER]
        bottoms = np.zeros(len(added))
        for technology in added.columns:
            ax.bar(range(len(added)), added[technology].values, bottom=bottoms, width=0.72,
                   color=TECH_COLORS[technology], edgecolor="white", lw=0.2, label=technology,
                   zorder=2)
            bottoms += added[technology].values
        reference_total = reference_by_zone[technologies].loc[ZONE_ORDER].sum(axis=1)
        for i, zone in enumerate(ZONE_ORDER):
            ax.annotate(f"+{bottoms[i] / reference_total.loc[zone] * 100:.0f} %",
                        xy=(i, bottoms[i]), xytext=(0, 2), textcoords="offset points",
                        ha="center", va="bottom", fontsize=FS["tick"], color="0.25")
        ax.set_ylim(0, bottoms.max() * 1.20)
        ax.set_xticks(range(len(added)))
        ax.set_xticklabels(zone_ticks, fontsize=FS["tick"], rotation=30, ha="right")
        ax.set_xlim(-0.7, len(added) - 0.3)
        ax.set_ylabel(ylabel, fontsize=FS["label"])
        ax.tick_params(labelsize=FS["tick"])
        ax.grid(True, axis="y", color="0.93", lw=0.3)
        ax.set_axisbelow(True)
        # The free corner differs: b's tallest bars are right of centre, c's left of centre.
        ax.legend(fontsize=FS["tick"], frameon=False, handlelength=1.0,
                  loc="upper left" if ax is ax_b else "upper right",
                  handletextpad=0.5, borderaxespad=0.2)

    fit_width(fig)
    legend_below(fig, [ax_a, ax_a_battery], *ax_a.get_legend_handles_labels(), ncol=2)
    panel_letters(fig, [("a", ax_a), ("b", ax_b), ("c", ax_c)])
    return fig


def figure_s19(baseload):
    """S19: the baseload demand the build-out serves, a typical year per balancing zone.

    Parameters
    ----------
    baseload : xarray.DataArray
        Hourly baseload (GW) per country, dimensions ``(hour_of_year, country)``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(WIDTH, 52 * MM))
    countries_in_file = {str(country) for country in baseload["country"].values}
    days = np.arange(baseload.sizes["hour_of_year"]) / 24
    for zone in ZONE_ORDER:
        members = [country for country in REALTIME_BALANCING_ZONES[zone]
                   if country in countries_in_file]
        if not members:
            continue
        zone_total = baseload.sel(country=members).sum("country")
        # One-week centred mean: the question is the seasonal shape, not the diurnal cycle.
        weekly = pd.Series(zone_total.values).rolling(24 * 7, center=True, min_periods=1).mean()
        ax.plot(days, weekly.values, lw=0.9, color=ZONE_COLORS[zone], label=ZONE_LABELS[zone])
    ax.set_xlabel("day of the typical year", fontsize=FS["label"])
    ax.set_ylabel("Baseload demand (GW)", fontsize=FS["label"])
    ax.set_xlim(0, 365)
    ax.grid(color="0.93", lw=0.3)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=FS["tick"])
    fit_width(fig)
    handles, labels = ax.get_legend_handles_labels()
    legend_below(fig, ax, handles, labels, ncol=6)
    return fig


def operation_figure(curves: dict, style, *, mode: str, ncols: int, ylabel: str, height_mm: float,
                     spread=(10, 90)):
    """S20-S22: an annual operation cycle per country or zone, one colour per warming level.

    Parameters
    ----------
    curves : dict
        ``{panel title key: {warming level: array (chain-years, 365)}}``, see
        :func:`~cordex2energy_analysis.runs.operation_curves`.
    style : PaperStyle
        Supplies the warming-level colours and order.
    mode : {"spaghetti", "band"}
        ``"spaghetti"`` draws every chain-year thinly (rasterised) with the mean on top;
        ``"band"`` draws the median and the ``spread`` percentile band.
    ncols : int
        Panels per row.
    ylabel : str
        Y-axis label of the first column.
    height_mm : float
        Figure height before the width is fitted.
    spread : tuple of float
        Lower and upper percentile of the band (``mode="band"``), across chain-years.

    Returns
    -------
    matplotlib.figure.Figure
    """
    keys = list(curves)
    nrows = int(np.ceil(len(keys) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(WIDTH, height_mm * MM), squeeze=False,
                             sharex=True, gridspec_kw={"hspace": 0.42, "wspace": 0.48})
    day = np.arange(365)
    for index, key in enumerate(keys):
        ax = axes[index // ncols, index % ncols]
        for period in style.period_names:
            block = curves[key].get(period)
            if block is None or not len(block):
                continue
            colour = style.gwl_colors[period]
            if mode == "spaghetti":
                # Thousands of thin lines: rasterised, so the PDF stays small; the mean stays vector.
                chain_years = LineCollection([np.column_stack([day, curve]) for curve in block],
                                             colors=colour, linewidths=0.2, alpha=0.05, zorder=1)
                chain_years.set_rasterized(True)
                ax.add_collection(chain_years, autolim=True)
                ax.plot(day, block.mean(axis=0), color=colour, lw=1.0, zorder=3, label=period)
            else:
                low, median, high = np.percentile(block, [spread[0], 50, spread[1]], axis=0)
                ax.fill_between(day, low, high, color=colour, alpha=0.18, lw=0, zorder=1)
                ax.plot(day, median, color=colour, lw=0.9, zorder=3, label=period)

        ax.set_title(COUNTRY_NAMES.get(key, key), fontsize=FS["title"], fontweight="bold")
        ax.set_xlim(0, 364)
        ax.set_xticks(MONTH_STARTS, minor=True)
        ax.set_xticks([MONTH_STARTS[month] for month in (0, 3, 6, 9)])
        ax.set_xticklabels(["Jan", "Apr", "Jul", "Oct"], fontsize=FS["tick"], ha="left")
        ax.tick_params(labelsize=FS["tick"])
        ax.grid(color="0.93", lw=0.3)
        ax.set_axisbelow(True)
        ax.autoscale_view()          # a LineCollection does not update the y limits on its own
        if index % ncols == 0:
            ax.set_ylabel(ylabel, fontsize=FS["label"])

    used = [axes[i // ncols, i % ncols] for i in range(len(keys))]
    for index in range(len(keys), nrows * ncols):
        axes[index // ncols, index % ncols].set_axis_off()
    fit_width(fig)
    legend_below(fig, [ax for ax in used if ax in list(axes[-1])], *gwl_handles(style), ncol=3,
                 title=GWL_TITLE)
    return fig
