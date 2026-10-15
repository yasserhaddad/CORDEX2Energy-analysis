"""Builders of the main-text figures (Figure 1, the pipeline schematic, is drawn outside Python).

Each ``figure_N`` function takes the tables it draws and returns a matplotlib figure at the
journal's artwork size; the notebook only loads the data, calls the builder and saves the result.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from .analysis import compute_sign_agreement
from .config import (DRIVER_DISPLAY, DURATION_BINS, MAIN_SEASONS, SEASON_MONTHS, SEASON_ORDER,
                     SOUTHERN_ZONES, SPREAD, ZONE_LABELS, ZONE_ORDER)
from .extremes_analysis import (attribution_change_by_model, compute_exceedance_probability,
                                mask_structural_zeros)
from .fields import country_field, demand_field
from .plotting_publication import (
    FS,
    GWL_TITLE,
    MM,
    QUANTITY_CMAP,
    SEASON_COUNT_BINS_SUB_DAILY,
    WIDTH,
    ZONE_COLORS,
    align_colorbar_labels,
    fit_width,
    legend_below,
    level_handles,
    mean_state_maps,
    panel_letters,
    panel_scenario_bars,
    panel_storage_stack,
    paired_change_pct,
    regime_figure,
    season_contribution_by_model,
    storage_handles,
    zone_tail_duration_figure,
)
from .power_system import REALTIME_BALANCING_ZONES
from .tables import DURATION_BIN_LABELS, duration_spectrum

#: Height of a one-row figure before its width is fitted, so Figures 6-8 share one panel height.
PANEL_ROW_MM = 43.2
#: Fixed, symmetric colour classes of the change maps (per cent; TWh per year for residual load).
SUPPLY_CHANGE_BINS = [-5, -2, -1, 1, 2, 5]
MAP_CHANGE_BINS = {
    "Solar generation": SUPPLY_CHANGE_BINS,
    "Wind generation": SUPPLY_CHANGE_BINS,
    "Hydro generation": [-20, -10, -5, -2, -1, 1, 2, 5, 10, 20],
    "Heating demand": [-50, -30, -20, -15, -10, -1, 1, 10, 15, 20, 30, 50],
    "Cooling demand": [-200, -150, -100, -50, -25, -1, 1, 25, 50, 100, 150, 200],
    "Residual load": [-10, -5, -2, -0.5, 0.5, 2, 5, 10],
}
#: Quantities whose change is drawn in TWh per year rather than in per cent.
ABSOLUTE_CHANGE_ROWS = ("Residual load",)
#: Short duration-class labels for ticks and table headers.
CLASS_LABELS = {"sub_daily": "≤1d", "multi_day": ">1d"}
EVENING_WINDOW, NIGHT_WINDOW = (15, 21.8), (-0.8, 6.0)


def mean_state_quantities(country_output: dict, demand_by_month: dict, period_windows: dict) -> list:
    """The six mapped quantities of Figures 2 and 3.

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
        ``(name, field, colormap, level centre)`` with ``field(period) -> DataArray (model,
        country)`` in TWh per year; the residual-load level is diverging, centred on 0.
    """
    def from_run(expression):
        """Field builder for a quantity computed from the run outputs."""
        return lambda period: country_field(country_output, period_windows, expression, period)

    def from_demand(variable):
        """Field builder for a demand variable."""
        return lambda period: demand_field(demand_by_month, period_windows, variable, period)

    hydro = from_run(lambda data: data["gen_hydro_ror"] + data["reservoir_dispatch_gwh"])
    return [
        ("Solar generation", from_run(lambda data: data["gen_pv"]), "plasma", None),
        ("Wind generation", from_run(lambda data: data["gen_wind_on"] + data["gen_wind_off"]),
         "Greens", None),
        ("Hydro generation", lambda period: mask_structural_zeros(hydro(period)),
         QUANTITY_CMAP["Hydro"], None),
        ("Heating demand", from_demand("heating"), QUANTITY_CMAP["Heating demand"], None),
        ("Cooling demand", from_demand("cooling"), QUANTITY_CMAP["Cooling demand"], None),
        ("Residual load", from_run(lambda data: data["residual_load"]), "BrBG_r", 0),
    ]


def figure_mean_state(quantities: list, context, height_mm: float = 125):
    """Figures 2 and 3: level at +1.5 °C and change at +2.0 and +3.0 °C, one row per quantity.

    Parameters
    ----------
    quantities : list of tuple
        Rows to draw, from :func:`mean_state_quantities` (three per figure).
    context : PaperContext
        Style, warming levels and the print-resolution country geometry.
    height_mm : float
        Height before the width is fitted; three rows of equal-area maps fill 180 mm.

    Returns
    -------
    figure : matplotlib.figure.Figure
    row_axes : list of matplotlib.axes.Axes
        The first-column axis of each row.
    """
    names = context.period_names
    fig, row_axes = mean_state_maps(
        quantities, context.countries_print, context.style, compute_sign_agreement,
        height_mm=height_mm, width_mm=180, change_bins=MAP_CHANGE_BINS,
        absolute_change_rows=ABSOLUTE_CHANGE_ROWS, change_labels={},
        column_titles=[context.reference, f"Δ {names[1]}", f"Δ {context.warmest}"])
    fig.get_layout_engine().set(h_pad=1 / 72, hspace=0.0)
    # Aligning the colorbar labels pushes them outward, so the width is fitted after each pass.
    for _ in range(2):
        align_colorbar_labels(fig, fig.change_colorbars)
        align_colorbar_labels(fig, fig.reference_colorbars)
        fit_width(fig, target_mm=179.5, max_height_mm=height_mm + 40)
    panel_letters(fig, list(zip("abc", row_axes)), inside=True)
    return fig, row_axes


def figure_tail_and_duration(run, context, storage_events):
    """Figure 5: balancing zones, southern exceedance curves, and events and severity by duration.

    Parameters
    ----------
    run : ResidualLoadRun
        The paper's run, with its threshold.
    context : PaperContext
        Style, windows and geometry.
    storage_events : pandas.DataFrame
        One row per event (:func:`~cordex2energy_analysis.runs.event_tables`).

    Returns
    -------
    matplotlib.figure.Figure
    """
    return zone_tail_duration_figure(
        countries=context.countries_print, residual_load_by_model=run.residual_load,
        period_windows=context.period_windows, zone_threshold=run.threshold,
        season_months=SEASON_MONTHS, style=context.style, southern_zones=SOUTHERN_ZONES,
        seasons=MAIN_SEASONS, spread=SPREAD, zone_members=REALTIME_BALANCING_ZONES,
        zone_colors=ZONE_COLORS,
        duration_spectrum=lambda value: duration_spectrum(storage_events, value),
        bin_labels=DURATION_BIN_LABELS, layout="beside", map_share=0.74)


def regime_titles(reference: str, warmest: str) -> dict:
    """Panel titles of the sub-daily (Figure 6) and multi-day (S31) regime figures.

    Parameters
    ----------
    reference, warmest : str
        The two warming levels the change panels compare.

    Returns
    -------
    dict
        ``{duration class: {panel letter: title}}``.
    """
    change = f"{reference} to {warmest}"
    return {
        "sub_daily": {"a": "Sub-daily extreme events",
                      "b": f"Change in events by season and zone, {change}",
                      "c": "Exceedance by hour of day, southern zones",
                      "d": f"Evening and overnight drivers, {change}"},
        "multi_day": {"a": "Multi-day extreme events",
                      "b": f"Change in events by season and zone, {change}",
                      "c": f"Driver decomposition by season, {change}"},
    }


def figure_sub_daily(tables: dict, southern_hours, context):
    """Figure 6: sub-daily events, their seasonal change, the diurnal cycle and its drivers.

    Parameters
    ----------
    tables : dict
        From :func:`~cordex2energy_analysis.runs.event_tables`.
    southern_hours : pandas.DataFrame
        Exceedance hours of the southern zones by hour of day
        (:func:`~cordex2energy_analysis.runs.pooled_exceedance_by_hour`).
    context : PaperContext
        Style and warming levels.

    Returns
    -------
    matplotlib.figure.Figure
    """
    reference, warmest = context.reference, context.warmest
    return regime_figure(
        "sub_daily",
        events_annual=tables["events_annual"], events_by_season=tables["events_by_season"],
        style=context.style, titles=regime_titles(reference, warmest)["sub_daily"],
        season_bins=SEASON_COUNT_BINS_SUB_DAILY, seasons=SEASON_ORDER,
        attribution=lambda zone, season: attribution_change_by_model(
            tables["evening_peak_components"], zone, season, reference, warmest,
            context.style.drivers),
        driver_seasons=list(MAIN_SEASONS), driver_zones=SOUTHERN_ZONES, diurnal=southern_hours,
        diurnal_windows=dict(evening=EVENING_WINDOW, night=NIGHT_WINDOW,
                             evening_label="evening", night_label="night"),
        driver_display=DRIVER_DISPLAY)


def figure_storage(storage_attenuation, severity_by_bin, context):
    """Figure 7: share of severity storage balances, and the energy it leaves, by event duration.

    Parameters
    ----------
    storage_attenuation : pandas.DataFrame
        From :func:`~cordex2energy_analysis.runs.event_tables`.
    severity_by_bin : pandas.DataFrame
        From :func:`~cordex2energy_analysis.tables.severity_by_duration`.
    context : PaperContext
        Style and warming levels.

    Returns
    -------
    matplotlib.figure.Figure
    """
    warmest = context.warmest
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, PANEL_ROW_MM * MM),
                             gridspec_kw={"wspace": 0.30, "width_ratios": (1.45, 0.75)})
    panel_storage_stack(axes[0], storage_attenuation, context.style,
                        group_by=("duration_class", list(DURATION_BINS),
                                  [CLASS_LABELS[duration_class] for duration_class in DURATION_BINS]),
                        where={"period": warmest}, level_label=warmest)
    positions = np.arange(len(DURATION_BIN_LABELS))
    block = severity_by_bin[severity_by_bin.period == warmest]
    balanced = [block[block.bin == label]["absorbed"].median() for label in DURATION_BIN_LABELS]
    unbalanced = [block[block.bin == label]["sev_actual"].median() for label in DURATION_BIN_LABELS]
    axes[1].bar(positions, balanced, 0.50, color="#4C9F70", edgecolor="white", lw=0.3,
                label="balanced by storage", zorder=2)
    axes[1].bar(positions, unbalanced, 0.50, bottom=balanced, color="#9E9E9E", edgecolor="white",
                lw=0.3, label="left unbalanced", zorder=2)
    day_boundary = DURATION_BIN_LABELS.index("12-24 h") + 0.5        # panel a's class split
    axes[1].axvline(day_boundary, color="0.35", lw=0.7, ls="--", zorder=1)
    axes[1].annotate("24 h", xy=(day_boundary, 1.0), xycoords=("data", "axes fraction"),
                     xytext=(2, -2), textcoords="offset points", ha="left", va="top",
                     fontsize=FS["tick"], color="0.35")
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(DURATION_BIN_LABELS, fontsize=FS["tick"], rotation=45, ha="right")
    axes[1].set_xlim(-0.6, len(DURATION_BIN_LABELS) - 0.4)
    axes[1].set_ylabel("severity (TWh yr$^{-1}$)", fontsize=FS["label"])
    axes[1].set_xlabel("event duration", fontsize=FS["label"])
    axes[1].tick_params(labelsize=FS["tick"])
    titles = ("Balanced share by zone and duration class", "Unbalanced severity, Europe total")
    for title, ax in zip(titles, axes):
        ax.set_title(title, fontsize=FS["title"], fontweight="bold")
    fit_width(fig)
    legend_below(fig, axes[0], *storage_handles(context.style), ncol=2)
    legend_below(fig, axes[1], *axes[1].get_legend_handles_labels(), ncol=1)
    panel_letters(fig, list(zip("ab", axes)))
    return fig


def figure_scenarios(events_by_scenario: dict, context):
    """Figure 8: sub-daily and multi-day events per zone under LOW/2050 and HIGH/2050.

    Parameters
    ----------
    events_by_scenario : dict
        ``{scenario label: events_annual table}``, both scored on the paper's threshold.
    context : PaperContext
        Style and warming levels.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, PANEL_ROW_MM * MM), gridspec_kw={"wspace": 0.30})
    for ax, duration_class in zip(axes, DURATION_BINS):
        panel_scenario_bars(ax, events_by_scenario, context.style, duration_class=duration_class,
                            ylabel="events yr$^{-1}$")
    for title, ax in zip(("Sub-daily extreme events", "Multi-day extreme events"), axes):
        ax.set_title(title, fontsize=FS["title"], fontweight="bold")
    fit_width(fig)
    # Both keys apply to both panels: one row, centred under the pair, 4 pt below the lowest ink.
    scenario_handles, scenario_labels = axes[0].get_legend_handles_labels()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    floor = min(ax.get_tightbbox(renderer).y0 for ax in axes) / fig.bbox.height
    legend_top = floor - 4.0 / 72 / fig.get_figheight()
    middle = 0.5 * (axes[0].get_position().x0 + axes[1].get_position().x1)
    fig.legend(scenario_handles, scenario_labels, loc="upper right", ncol=2, frameon=False,
               bbox_to_anchor=(middle - 0.01, legend_top), title="Supply scenario")
    fig.legend(*level_handles(context.style), loc="upper left", ncol=2, frameon=False,
               bbox_to_anchor=(middle + 0.01, legend_top), title=GWL_TITLE)
    panel_letters(fig, list(zip("ab", axes)))
    return fig


# ---------------------------------------------------------------------------
# Numbers behind the figures: what the text quotes, as tables
# ---------------------------------------------------------------------------

def threshold_table(threshold) -> pd.Series:
    """The CERRA P95 deficit threshold of each zone.

    Parameters
    ----------
    threshold : xarray.DataArray
        Threshold per zone (GWh per 3h step), e.g. ``run.threshold``.

    Returns
    -------
    pandas.Series
        Indexed by zone label, in GWh per 3h step.
    """
    zones = [zone for zone in ZONE_ORDER if zone in threshold["zone"].values]
    return pd.Series({ZONE_LABELS[zone]: float(threshold.sel(zone=zone)) for zone in zones},
                     name="P95 threshold (GWh per 3h step)")


def exceedance_ratio_table(run, context, zones=SOUTHERN_ZONES, seasons=MAIN_SEASONS) -> pd.DataFrame:
    """Probability of exceeding the threshold at two warming levels, and their ratio (Figure 5b).

    Parameters
    ----------
    run : ResidualLoadRun
        The run and its threshold.
    context : PaperContext
        Warming levels and windows.
    zones : sequence of str
        Zones, in row order.
    seasons : sequence of str
        Seasons, in row order.

    Returns
    -------
    pandas.DataFrame
        One row per (zone, season): ensemble-median exceedance probability (%) at both levels and
        their ratio, the factor printed in Figure 5b.
    """
    reference, warmest = context.reference, context.warmest
    rows = {}
    for zone in zones:
        threshold = float(run.threshold.sel(zone=zone))
        for season in seasons:
            probability = {period: compute_exceedance_probability(
                run.residual_load, context.period_windows, zone, SEASON_MONTHS[season], period,
                threshold) for period in (reference, warmest)}
            rows[(ZONE_LABELS[zone], season)] = {
                f"P(RL > P95) {reference} (%)": 100 * probability[reference],
                f"P(RL > P95) {warmest} (%)": 100 * probability[warmest],
                "ratio": (probability[warmest] / probability[reference]
                          if probability[reference] > 0 else np.nan)}
    return pd.DataFrame(rows).T


def duration_spectrum_table(storage_events: pd.DataFrame, context) -> pd.DataFrame:
    """Europe-wide events and severity per year in each duration bin (Figure 5c, d).

    Parameters
    ----------
    storage_events : pandas.DataFrame
        One row per event, from :func:`~cordex2energy_analysis.runs.event_tables`.
    context : PaperContext
        Warming levels.

    Returns
    -------
    pandas.DataFrame
        Rows are duration bins; columns ``(events yr-1 | severity TWh yr-1, warming level)``,
        each the median across chains of the zones' sum.
    """
    parts = {}
    for value, label in (("counts", "events yr-1"), ("severity", "severity TWh yr-1")):
        spectrum = duration_spectrum(storage_events, value)
        median = spectrum.groupby(["bin", "period"])["value"].median().unstack("period")
        parts[label] = median.loc[DURATION_BIN_LABELS, list(context.period_names)]
    return pd.concat(parts, axis=1)


def regime_change_table(events_annual: pd.DataFrame, events_by_season: pd.DataFrame, context,
                        duration_class: str) -> pd.DataFrame:
    """Events per year by zone, their paired change, and its seasonal split (Figure 6a, b; S31a, b).

    Parameters
    ----------
    events_annual, events_by_season : pandas.DataFrame
        From :func:`~cordex2energy_analysis.runs.event_tables`.
    context : PaperContext
        Style and warming levels.
    duration_class : {"sub_daily", "multi_day"}
        Event class.

    Returns
    -------
    pandas.DataFrame
        One row per zone: median events per year at both levels, the median per-chain change in
        per cent and in events per year with the number of chains agreeing on its sign (the
        figure's annotation), and the median per-chain change of each season in events per year
        (the heatmap).
    """
    style, reference, warmest = context.style, context.reference, context.warmest
    data = events_annual[events_annual.duration_class == duration_class]
    rows = {}
    for zone in style.zones:
        levels = {period: data[(data.zone == zone) & (data.period == period)]
                  .set_index("model")["freq_per_year"] for period in (reference, warmest)}
        percent, agreeing, _, _ = paired_change_pct(levels[reference], levels[warmest], style)
        shared = levels[reference].index.intersection(levels[warmest].index)
        change = levels[warmest].loc[shared] - levels[reference].loc[shared]
        row = {reference: levels[reference].median(), warmest: levels[warmest].median(),
               "Δ %": percent, "Δ events yr-1": change.median(),
               "n agree": f"{agreeing}/{style.n_models}"}
        for season in SEASON_ORDER:
            per_chain, no_events = season_contribution_by_model(
                events_by_season, events_annual, zone, season, duration_class, style,
                relative=False)
            row[f"Δ {season}"] = np.nan if no_events else per_chain.median()
        rows[ZONE_LABELS[zone]] = row
    table = pd.DataFrame(rows).T
    numeric = [column for column in table.columns if column != "n agree"]
    table[numeric] = table[numeric].astype(float)
    return table


def driver_change_table(evening_peak_components: pd.DataFrame, context, zones=SOUTHERN_ZONES,
                        seasons=MAIN_SEASONS) -> pd.DataFrame:
    """Each driver's contribution to the change in deficit-window residual load (Figure 6d).

    Parameters
    ----------
    evening_peak_components : pandas.DataFrame
        From :func:`~cordex2energy_analysis.runs.event_tables`.
    context : PaperContext
        Style and warming levels.
    zones, seasons : sequence of str
        Zones and seasons, in row order.

    Returns
    -------
    pandas.DataFrame
        One row per (zone, season): the median across chains of each driver's contribution and of
        the net change, in GWh per 3h step, with the interquartile range of the net change.
    """
    rows = {}
    for zone in zones:
        for season in seasons:
            per_chain = attribution_change_by_model(
                evening_peak_components, zone, season, context.reference, context.warmest,
                context.style.drivers)
            row = per_chain.median().to_dict()
            row["net P25"], row["net P75"] = per_chain["net"].quantile([0.25, 0.75])
            rows[(ZONE_LABELS[zone], season)] = row
    return pd.DataFrame(rows).T


def severity_balance_table(severity_by_bin: pd.DataFrame, period: str) -> pd.DataFrame:
    """Severity storage balanced, and what it left, per duration bin at one level (Figure 7b).

    Parameters
    ----------
    severity_by_bin : pandas.DataFrame
        From :func:`~cordex2energy_analysis.tables.severity_by_duration`.
    period : str
        Warming level.

    Returns
    -------
    pandas.DataFrame
        Rows are duration bins: median balanced and unbalanced severity (TWh per year), and the
        balanced share of their sum (%).
    """
    block = severity_by_bin[severity_by_bin.period == period]
    table = pd.DataFrame({
        "balanced (TWh yr-1)": block.groupby("bin")["absorbed"].median(),
        "unbalanced (TWh yr-1)": block.groupby("bin")["sev_actual"].median()}).loc[DURATION_BIN_LABELS]
    total = table.sum(axis=1)
    table["balanced share (%)"] = 100 * table["balanced (TWh yr-1)"] / total.where(total > 0)
    return table


def europe_event_totals(events_by_scenario: dict, context) -> pd.DataFrame:
    """Europe-wide events per year under each build-out (the sum over zones, Figure 8).

    Parameters
    ----------
    events_by_scenario : dict
        ``{scenario label: events_annual table}``.
    context : PaperContext
        Warming levels.

    Returns
    -------
    pandas.DataFrame
        Rows ``(scenario, duration class)``; columns the warming levels, each the median across
        chains of the zones' sum.
    """
    parts = {}
    for label, events in events_by_scenario.items():
        totals = events.groupby(["duration_class", "model", "period"])["freq_per_year"].sum()
        parts[label] = (totals.groupby(["duration_class", "period"]).median()
                        .unstack("period")[list(context.period_names)])
    return pd.concat(parts)


def show_table(title: str, table, digits: int = 1) -> None:
    """Print a titled table, rounded, so it reads the same in Jupyter and in a plain run.

    Parameters
    ----------
    title : str
        Printed above the table.
    table : pandas.DataFrame or pandas.Series
        The table.
    digits : int
        Decimals shown.
    """
    print(f"{title}\n{table.round(digits).to_string()}\n")
