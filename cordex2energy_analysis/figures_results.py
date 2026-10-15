"""Builders of the Supplementary Note 3 figures (S23-S33): additional results.

Each ``figure_sN`` function takes the tables it draws and returns a matplotlib figure at the
journal's artwork size; the notebook only loads the data, calls the builder and saves the result.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec

from .analysis import model_spread_bounds
from .config import (DRIVER_DISPLAY, DURATION_BINS, DURATION_LABELS, DURATION_TITLES, SEASON_MONTHS,
                     SEASON_ORDER, SPREAD, ZONE_LABELS, ZONE_ORDER)
from .extremes_analysis import attribution_change_by_model
from .fields import country_field
from .figures_main import EVENING_WINDOW, NIGHT_WINDOW, regime_titles
from .plotting_paper import _change_color
from .plotting_publication import (
    FS,
    GWL_TITLE,
    MM,
    SCENARIO_COLORS,
    SEASON_COUNT_BINS_MULTI_DAY,
    STORAGE_PARTS,
    WIDTH,
    MapRow,
    driver_handles,
    europe_change_handles,
    europe_change_panel,
    fit_width,
    gwl_handles,
    gwl_maps,
    legend_below,
    panel_driver_attribution,
    panel_hour_of_day,
    panel_letters,
    regime_figure,
    season_label,
    sign_agreement,
    storage_handles,
    tail_exceedance_figure,
)
from .tables import DURATION_BIN_LABELS

#: Seasonal residual-load change classes (TWh per season): half the annual edges.
SEASONAL_RESIDUAL_LOAD_BINS = [-5, -2, -1, -0.25, 0.25, 1, 2, 5]


def figure_s23(residual_load_output: dict, context):
    """S23: residual load per season, level at +1.5 °C and change at +2.0 and +3.0 °C.

    Parameters
    ----------
    residual_load_output : dict
        Per-country monthly residual load
        (:func:`~cordex2energy_analysis.fields.load_country_output` with ``("residual_load.nc",)``).
    context : PaperContext
        Style, windows and geometry.

    Returns
    -------
    matplotlib.figure.Figure
    """
    def row(season):
        """Map row of the seasonal residual load."""
        fields = {period: country_field(residual_load_output, context.period_windows,
                                        lambda data: data["residual_load"], period,
                                        SEASON_MONTHS[season])
                  for period in context.period_names}
        # The row label names the season, so the colorbar unit is plain TWh.
        return MapRow(label=season_label(season),
                      field=lambda period, row_fields=fields: row_fields[period],
                      kind="country", cmap="BrBG_r", unit="TWh", change="absolute",
                      bins=SEASONAL_RESIDUAL_LOAD_BINS, reference_center=0, change_label=None)

    fig, row_axes = gwl_maps([row(season) for season in SEASON_ORDER], context.countries_print,
                             context.style, sign_agreement, height_mm=162, share_abs_scale=True)
    panel_letters(fig, list(zip("abcd", row_axes)), inside=True)
    return fig


def europe_change_spans(change: dict, labels, columns) -> dict:
    """x-limits per column covering every chain, winter and summer sharing one.

    Parameters
    ----------
    change : dict
        ``{(column, label): DataFrame}`` from
        :func:`~cordex2energy_analysis.plotting_publication.europe_change_table`.
    labels : sequence of str
        Quantities drawn.
    columns : sequence of str
        ``Annual``, ``DJF``, ``JJA``.

    Returns
    -------
    dict
        ``{column: (low, high)}``, each spanning zero.
    """
    spans = {}
    for column in columns:
        low = min(change[column, label].twh.min() for label in labels)
        high = max(change[column, label].twh.max() for label in labels)
        spans[column] = (min(low, 0.0), max(high, 0.0))
    joint = (min(spans["DJF"][0], spans["JJA"][0]), max(spans["DJF"][1], spans["JJA"][1]))
    spans["DJF"] = spans["JJA"] = joint
    return spans


def figure_s24(zone_change: dict, labels, columns):
    """S24: Figure 4 resolved by balancing zone, one row per zone.

    Parameters
    ----------
    zone_change : dict
        ``{zone: table}``, each as
        :func:`~cordex2energy_analysis.plotting_publication.europe_change_table` restricted to
        the zone.
    labels : sequence of str
        Quantities, top to bottom.
    columns : sequence of str
        ``Annual``, ``DJF``, ``JJA``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    columns = list(columns)
    fig, axes = plt.subplots(len(ZONE_ORDER), len(columns), figsize=(WIDTH, 205 * MM),
                             sharey=True, gridspec_kw={"wspace": 0.10, "hspace": 0.42})
    for row, zone in enumerate(ZONE_ORDER):
        spans = europe_change_spans(zone_change[zone], labels, columns)
        for col, column in enumerate(columns):
            ax = axes[row, col]
            europe_change_panel(ax, zone_change[zone], labels, column, span=spans[column])
            if row == 0:
                ax.set_title(season_label(column), fontsize=FS["title"], fontweight="bold")
            if col == 0:
                ax.set_ylabel(ZONE_LABELS[zone], fontsize=FS["label"], fontweight="bold")
                ax.set_yticklabels(labels, fontsize=FS["tick"])
            if row == len(ZONE_ORDER) - 1:
                unit = "TWh yr$^{-1}$" if column == "Annual" else "TWh season$^{-1}$"
                ax.set_xlabel(f"change ({unit})", fontsize=FS["label"])
    fit_width(fig)
    legend_below(fig, list(axes[-1]), *europe_change_handles(), ncol=3)
    return fig


def label_below_row(fig, axes, text: str):
    """One axis label centred under a row of panels, 3 pt below their lowest ink.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure, drawn once to measure the row.
    axes : sequence of matplotlib.axes.Axes
        The row.
    text : str
        The label.
    """
    fig.canvas.draw()
    renderer, inverse = fig.canvas.get_renderer(), fig.transFigure.inverted()
    boxes = [ax.get_tightbbox(renderer).transformed(inverse) for ax in axes]
    fig.text(0.5 * (axes[0].get_position().x0 + axes[-1].get_position().x1),
             min(box.y0 for box in boxes) - 3.0 / 72 / fig.get_figheight(),
             text, ha="center", va="top", fontsize=FS["label"])


def figure_s25(reference_metrics: dict, chain_thresholds: dict, cerra_metrics, threshold, zones):
    """S25: CORDEX against CERRA in the reference period, as distributions (a) and p95s (b).

    Parameters
    ----------
    reference_metrics : dict
        ``{chain: Dataset}`` in the +1.5 °C window
        (:func:`~cordex2energy_analysis.runs.reference_period_thresholds`).
    chain_thresholds : dict
        ``{chain: DataArray}``, the p95 each chain would imply.
    cerra_metrics : xarray.Dataset
        CERRA system metrics (:func:`~cordex2energy_analysis.runs.cerra_system_metrics`).
    threshold : xarray.DataArray
        The paper's CERRA p95 per zone.
    zones : sequence of str
        Zones, one column each.

    Returns
    -------
    figure : matplotlib.figure.Figure
    bias : pandas.DataFrame
        Per zone: CERRA p95, median chain p95 and their difference in per cent.
    """
    models = list(reference_metrics)
    fig, axes = plt.subplots(2, len(zones), figsize=(WIDTH, 82 * MM), squeeze=False,
                             gridspec_kw={"hspace": 0.42, "wspace": 0.46,
                                          "height_ratios": [1.0, 0.72]})
    bias_rows = []
    for col, zone in enumerate(zones):
        ax = axes[0, col]
        cerra_values = cerra_metrics["residual_load"].sel(zone=zone).values
        cerra_values = cerra_values[np.isfinite(cerra_values)]
        edges = np.linspace(*np.percentile(cerra_values, [0.5, 99.5]), 90)
        centres = 0.5 * (edges[:-1] + edges[1:])
        per_chain = np.vstack([
            np.histogram(dataset["residual_load"].sel(zone=zone).values[
                np.isfinite(dataset["residual_load"].sel(zone=zone).values)],
                bins=edges, density=True)[0]
            for dataset in reference_metrics.values()])
        ax.fill_between(centres, per_chain.min(axis=0), per_chain.max(axis=0),
                        color="#4A90E2", alpha=0.30, lw=0, label=f"CORDEX ({len(models)} chains)")
        ax.plot(centres, np.median(per_chain, axis=0), color="#1F4E79", lw=0.9)
        ax.plot(centres, np.histogram(cerra_values, bins=edges, density=True)[0],
                color="black", lw=0.8, ls="--", label="CERRA")
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3))
        ax.yaxis.get_offset_text().set_fontsize(FS["tick"])
        ax.set_title(ZONE_LABELS[zone], fontsize=FS["title"], fontweight="bold")
        ax.tick_params(labelsize=FS["tick"])
        ax.grid(True, axis="x", color="0.93", lw=0.3)
        ax.set_axisbelow(True)
        if col == 0:
            ax.set_ylabel("Density", fontsize=FS["label"])

        ax = axes[1, col]
        thresholds = [float(chain_thresholds[model].sel(zone=zone)) for model in models]
        cerra_threshold = float(threshold.sel(zone=zone))
        box = ax.boxplot([thresholds], positions=[0], widths=0.5, patch_artist=True,
                         showfliers=False, medianprops=dict(color="black", lw=0.8))
        for patch in box["boxes"]:
            patch.set_facecolor("#4A90E2")
            patch.set_alpha(0.7)
            patch.set_edgecolor("0.25")
            patch.set_lw(0.4)
        for element in ("whiskers", "caps"):
            for line in box[element]:
                line.set_lw(0.5)
        ax.plot(np.zeros(len(thresholds)), thresholds, "o", ms=1.6, color="0.2", alpha=0.6, zorder=3)
        ax.axhline(cerra_threshold, color="black", ls="--", lw=0.7, zorder=4)
        bias = 100 * (np.median(thresholds) - cerra_threshold) / cerra_threshold
        ax.annotate(f"{bias:+.0f}%", xy=(0.55, cerra_threshold), xytext=(0, 3),
                    textcoords="offset points", fontsize=FS["tick"], fontweight="bold",
                    color=_change_color(bias))
        ax.set_xticks([])
        ax.set_xlim(-0.6, 1.1)
        ax.tick_params(labelsize=FS["tick"])
        ax.grid(True, axis="y", color="0.93", lw=0.3)
        ax.set_axisbelow(True)
        if col == 0:
            ax.set_ylabel("P95 per 3h step (GWh)", fontsize=FS["label"])
        bias_rows.append(dict(zone=ZONE_LABELS[zone], cerra=round(cerra_threshold, 1),
                              cordex_median=round(float(np.median(thresholds)), 1),
                              bias_pct=round(bias, 1)))

    fit_width(fig)
    label_below_row(fig, list(axes[0]), "residual load per 3h step (GWh)")
    panel_letters(fig, [("a", axes[0, 0]), ("b", axes[1, 0])])
    # One dashed entry: the same mark means CERRA in both rows.
    handles, labels = axes[0, 0].get_legend_handles_labels()
    legend_below(fig, list(axes[1]), handles, labels, ncol=2)
    return fig, pd.DataFrame(bias_rows)


def figure_s26(seasonal_thresholds: dict, zones):
    """S26: the season-specific CERRA p95 the paper does not use, per zone and season.

    Parameters
    ----------
    seasonal_thresholds : dict
        ``{zone: {season: threshold}}`` (GWh per 3h step).
    zones : sequence of str
        Zones, one row each.

    Returns
    -------
    matplotlib.figure.Figure
    """
    table = np.array([[float(seasonal_thresholds[str(zone)][season]) for season in SEASON_ORDER]
                      for zone in zones])
    fig, ax = plt.subplots(figsize=(WIDTH * 0.52, 52 * MM))
    image = ax.imshow(table, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(SEASON_ORDER)), [season_label(season) for season in SEASON_ORDER],
                  fontsize=FS["label"])
    ax.set_yticks(range(len(zones)), [ZONE_LABELS[zone] for zone in zones], fontsize=FS["label"])
    ax.tick_params(length=0)
    ax.grid(False)
    for row in range(table.shape[0]):
        for col in range(table.shape[1]):
            value = table[row, col]
            # Ink chosen against the cell: YlOrRd runs from near-white to deep red.
            shade = (value - table.min()) / max(table.max() - table.min(), 1e-9)
            ax.text(col, row, f"{value:.0f}", ha="center", va="center", fontsize=FS["tick"],
                    color="white" if shade > 0.6 else "0.15")
    bar = fig.colorbar(image, ax=ax, fraction=0.05, pad=0.03)
    bar.set_label("P95 per 3h step (GWh)", fontsize=FS["label"])
    bar.ax.tick_params(labelsize=FS["tick"])
    bar.outline.set_linewidth(0.4)
    fit_width(fig, target_mm=110)
    return fig


def figure_s27(high_run, paper_threshold, zones):
    """S27: where the paper's (LOW/2050) threshold falls inside HIGH/2050's CERRA distribution.

    Parameters
    ----------
    high_run : ResidualLoadRun
        The HIGH/2050 run with its own CERRA calibration (``cerra_deficit`` and ``threshold``).
    paper_threshold : xarray.DataArray
        The paper's threshold per zone.
    zones : sequence of str
        Zones, one panel each.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, len(zones), figsize=(WIDTH, 48 * MM), sharey=True,
                             gridspec_kw={"wspace": 0.26})
    for ax, zone in zip(axes, zones):
        values = high_run.cerra_deficit.sel(zone=zone).values
        positive = np.sort(values[np.isfinite(values)])
        survival = 1.0 - np.arange(len(positive)) / len(positive)      # P(RL > x | RL > 0)
        threshold_paper = float(paper_threshold.sel(zone=zone))
        threshold_own = float(high_run.threshold.sel(zone=zone))
        percentile = 100 * float((positive <= threshold_paper).mean())
        ax.plot(positive, survival, color="0.25", lw=0.8, zorder=3)
        ax.axvline(threshold_paper, color=SCENARIO_COLORS[0], ls="-", lw=0.8, zorder=4,
                   label="LOW 2050 P95")
        ax.axvline(threshold_own, color=SCENARIO_COLORS[1], ls="--", lw=0.8, zorder=4,
                   label="HIGH 2050 P95")
        ax.set_yscale("log")
        ax.set_ylim(1e-4, 1.05)
        ax.set_xlim(0, max(threshold_paper, threshold_own) * 1.5)
        ax.set_title(ZONE_LABELS[zone], fontsize=FS["title"], fontweight="bold")
        ax.tick_params(labelsize=FS["tick"])
        ax.grid(True, which="major", color="0.93", lw=0.3)
        ax.set_axisbelow(True)
        # ">p100" where the paper's threshold lies beyond every observed value.
        placement = ">p100" if threshold_paper > positive[-1] else f"p{percentile:.1f}"
        ax.annotate(f"{placement}\n×{threshold_paper / threshold_own:.2f}",
                    xy=(threshold_paper, 0.4), xytext=(2, 0), textcoords="offset points",
                    ha="left", va="top", fontsize=FS["tick"], fontweight="bold",
                    color=SCENARIO_COLORS[0])
        if zone == zones[0]:
            ax.set_ylabel("P(RL > x)", fontsize=FS["label"])
    fit_width(fig)
    label_below_row(fig, list(axes), "residual load per 3h step (GWh)")
    handles, labels = axes[0].get_legend_handles_labels()
    legend_below(fig, list(axes), handles, labels, ncol=2, gap_pt=14.0,
                 title="Supply scenario", title_fontsize=FS["tick"])
    return fig


def figure_s28(run, context, events_by_season):
    """S28: exceedance curves of the zonal residual load, every zone and season.

    Parameters
    ----------
    run : ResidualLoadRun
        The paper's run, with its threshold.
    context : PaperContext
        Style and windows.
    events_by_season : pandas.DataFrame
        Event hours per chain, level, zone, season and class.

    Returns
    -------
    matplotlib.figure.Figure
    """
    return tail_exceedance_figure(
        seasons=SEASON_ORDER, residual_load_by_model=run.residual_load,
        period_windows=context.period_windows, zone_threshold=run.threshold,
        zone_season_hours=events_by_season, season_months=SEASON_MONTHS,
        duration_labels=DURATION_LABELS, style=context.style, spread=SPREAD,
        duration_block=False, height_mm=92)


def figure_s29(zone_spectrum: dict, style):
    """S29: events (a) and their severity (b) by duration, one column per zone.

    Parameters
    ----------
    zone_spectrum : dict
        ``{"counts" | "severity": duration_spectrum(..., by=("model", "period", "zone"))}``.
    style : PaperStyle
        Warming-level colours.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(2, len(ZONE_ORDER), figsize=(WIDTH, 100 * MM), sharex=True,
                             gridspec_kw={"hspace": 0.20, "wspace": 0.42})
    positions = np.arange(len(DURATION_BIN_LABELS))
    for row, (key, ylabel) in enumerate((("counts", "events yr$^{-1}$"),
                                         ("severity", "severity (TWh yr$^{-1}$)"))):
        table = zone_spectrum[key]
        for col, zone in enumerate(ZONE_ORDER):
            ax = axes[row, col]
            block = table[table.zone == zone]
            for period in style.period_names:
                wide = block[block.period == period].pivot(index="model", columns="bin",
                                                          values="value")[DURATION_BIN_LABELS]
                ax.fill_between(positions, wide.quantile(0.25), wide.quantile(0.75),
                                color=style.gwl_colors[period], alpha=0.18, lw=0)
                ax.plot(positions, wide.median(), "o-", ms=2.2, lw=0.9, color=style.gwl_colors[period])
            ax.axvline(2.5, color="0.4", ls="--", lw=0.5)            # the 24 h class boundary
            ax.set_ylim(bottom=0)
            if row == 0:
                ax.set_title(ZONE_LABELS[zone], fontsize=FS["title"], fontweight="bold")
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=FS["label"])
            ax.tick_params(labelsize=FS["tick"])
            ax.set_xticks(positions)
            ax.set_xticklabels(DURATION_BIN_LABELS, rotation=90, fontsize=FS["tick"])
            ax.grid(True, color="0.93", lw=0.3)
            ax.set_axisbelow(True)
    fit_width(fig)
    legend_below(fig, list(axes[-1]), *gwl_handles(style), ncol=3, title=GWL_TITLE)
    panel_letters(fig, [("a", axes[0, 0]), ("b", axes[1, 0])])
    return fig


def figure_s30(hour_exceedance, evening_peak_components, context):
    """S30: Figure 6's second row for every zone: diurnal exceedance and its drivers.

    Parameters
    ----------
    hour_exceedance : pandas.DataFrame
        From :func:`~cordex2energy_analysis.runs.exceedance_by_hour`.
    evening_peak_components : pandas.DataFrame
        Driver anomalies over the deficit window.
    context : PaperContext
        Style and warming levels.

    Returns
    -------
    matplotlib.figure.Figure
    """
    style = context.style
    fig = plt.figure(figsize=(WIDTH, 198 * MM))
    grid = gridspec.GridSpec(len(ZONE_ORDER), 2, hspace=0.62, wspace=0.30, figure=fig)
    lettered, last_djf = [], None
    for row, zone in enumerate(ZONE_ORDER):
        # Summer stacked over winter, per-zone limits: the seasons and zones differ several-fold.
        cell = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=grid[row, 0], hspace=0.18)
        ax_jja, last_djf = fig.add_subplot(cell[0]), fig.add_subplot(cell[1])
        panel_hour_of_day(
            ax_jja, last_djf, hour_exceedance[hour_exceedance.zone == zone], style,
            evening=EVENING_WINDOW, night=NIGHT_WINDOW,
            evening_label="evening" if row == 0 else "", night_label="night" if row == 0 else "",
            ylabel="h yr$^{-1}$ > P95")
        ax_jja.set_title(f"{ZONE_LABELS[zone]} - exceedance by hour of day",
                         fontsize=FS["title"], fontweight="bold")
        if row < len(ZONE_ORDER) - 1:
            last_djf.set_xlabel("")

        ax_drivers = fig.add_subplot(grid[row, 1])
        panel_driver_attribution(
            ax_drivers,
            lambda zone_name, season: attribution_change_by_model(
                evening_peak_components, zone_name, season, context.reference, context.warmest,
                style.drivers),
            ["DJF", "JJA"], replace(style, zones=[zone]),
            ylabel="Δ per 3h step (GWh)", show_sign_convention=False,
            reserve_sign_gutter=False, show_zone_labels=False)
        ax_drivers.set_title("Evening and overnight drivers", fontsize=FS["title"],
                             fontweight="bold")
        lettered += [ax_jja, ax_drivers]

    fit_width(fig)
    gwl_handle_list, gwl_labels = gwl_handles(style)
    driver_handle_list, driver_labels = driver_handles(style)
    legend_below(fig, last_djf, gwl_handle_list, gwl_labels, ncol=len(gwl_labels), title=GWL_TITLE)
    legend_below(fig, lettered[-1], driver_handle_list,
                 [DRIVER_DISPLAY.get(name, name) for name in driver_labels], ncol=3)
    panel_letters(fig, list(zip("abcdefghijkl", lettered)))
    return fig


def figure_s31(tables: dict, context):
    """S31: the multi-day regime, by the same builder as Figure 6.

    Parameters
    ----------
    tables : dict
        From :func:`~cordex2energy_analysis.runs.event_tables`.
    context : PaperContext
        Style and warming levels.

    Returns
    -------
    matplotlib.figure.Figure
    """
    reference, warmest = context.reference, context.warmest
    titles = dict(regime_titles(reference, warmest)["multi_day"])
    titles["b"] = f"Change by season and zone, {reference} to {warmest}"   # fits beside the bins
    return regime_figure(
        "multi_day",
        events_annual=tables["events_annual"], events_by_season=tables["events_by_season"],
        style=context.style, titles=titles, season_bins=SEASON_COUNT_BINS_MULTI_DAY,
        seasons=SEASON_ORDER,
        attribution=lambda zone, season: attribution_change_by_model(
            tables["multi_day_events"], zone, season, reference, warmest, context.style.drivers,
            column_template="{season}_{driver}", net_column="{season}_RL"),
        driver_seasons=SEASON_ORDER, diurnal=None, driver_display=DRIVER_DISPLAY, letters="abc")


def panel_storage_by_level(ax, storage_attenuation, zone: str, period_names):
    """A zone's reservoir and battery shares of balanced severity, per warming level and class.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    storage_attenuation : pandas.DataFrame
        From :func:`~cordex2energy_analysis.runs.event_tables`.
    zone : str
        Balancing zone.
    period_names : sequence of str
        Warming levels.
    """
    rows = storage_attenuation[storage_attenuation.zone == zone]
    width = 0.80 / len(period_names)
    level_labels = [name.strip("+").replace("°C", "").strip() for name in period_names]
    for i, duration_class in enumerate(DURATION_BINS):
        for k, period in enumerate(period_names):
            cell = rows[(rows.duration_class == duration_class) & (rows.period == period)]
            bottom = 0.0
            for column, _label, color in STORAGE_PARTS:
                value = cell[column].median()
                ax.bar(i + (k - (len(period_names) - 1) / 2) * width, value, width * 0.88,
                       bottom=bottom, color=color, edgecolor="white", linewidth=0.4, zorder=2)
                bottom += value
        if i:
            ax.axvline(i - 0.5, color="0.85", lw=0.4, zorder=0)
    ax.set_ylim(0, 105)
    ax.set_xlim(-0.5, len(DURATION_BINS) - 0.5)
    ax.set_ylabel("share of severity\nbalanced by storage (%)", fontsize=FS["label"])
    ax.set_xticks([i + (k - (len(period_names) - 1) / 2) * width
                   for i in range(len(DURATION_BINS)) for k in range(len(period_names))])
    ax.set_xticklabels(level_labels * len(DURATION_BINS), fontsize=FS["tick"])
    ax.tick_params(axis="x", length=0, pad=1.0)
    ax.tick_params(axis="y", labelsize=FS["tick"])
    for i, duration_class in enumerate(DURATION_BINS):
        ax.annotate(DURATION_TITLES[duration_class], xy=(i, 0), xycoords=("data", "axes fraction"),
                    xytext=(0, -11.0), textcoords="offset points", ha="center", va="top",
                    fontsize=FS["tick"], annotation_clip=False)


def figure_s32(storage_attenuation, hydro_change: pd.Series, context, zone="Iberian_Peninsula"):
    """S32: Iberia's balanced share by warming level (a) and its reservoir inflow and dispatch (b).

    Parameters
    ----------
    storage_attenuation : pandas.DataFrame
        From :func:`~cordex2energy_analysis.runs.event_tables`.
    hydro_change : pandas.Series
        Per-chain per-cent change of inflow and dispatch by season
        (:func:`~cordex2energy_analysis.tables.paired_percent_change`).
    context : PaperContext
        Style and warming levels.
    zone : str
        The zone drawn.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig = plt.figure(figsize=(WIDTH, 58 * MM))
    grid = gridspec.GridSpec(1, 2, wspace=0.30, figure=fig)
    ax_a = fig.add_subplot(grid[0, 0])
    panel_storage_by_level(ax_a, storage_attenuation, zone, context.period_names)
    ax_a.set_title("Balanced share by warming level", fontsize=FS["title"], fontweight="bold")

    ax_b = fig.add_subplot(grid[0, 1])
    width = 0.36
    for k, (quantity, color) in enumerate((("inflow", "#3B7EA1"), ("dispatch", "#C8842B"))):
        offsets, medians, below, above = [], [], [], []
        for i, season in enumerate(SEASON_ORDER):
            values = hydro_change.xs((season, quantity), level=("season", "quantity")).values
            low, high = model_spread_bounds(values, context.style.spread, axis=0)
            median = float(np.median(values))
            offsets.append(i + (k - 0.5) * width)
            medians.append(median)
            below.append(median - low)
            above.append(high - median)
        ax_b.bar(offsets, medians, width * 0.88, color=color, edgecolor="white", linewidth=0.4,
                 zorder=2, label=quantity.capitalize())
        ax_b.errorbar(offsets, medians, yerr=[below, above], fmt="none", ecolor="0.25",
                      elinewidth=0.6, capsize=1.2, capthick=0.5, zorder=4)
    ax_b.axhline(0, color="0.25", lw=0.5, zorder=3)
    ax_b.set_xticks(range(len(SEASON_ORDER)))
    ax_b.set_xticklabels([season_label(season) for season in SEASON_ORDER], fontsize=FS["tick"])
    ax_b.set_ylabel(f"change from {context.reference} (%)", fontsize=FS["label"])
    ax_b.set_title(f"Reservoir inflow and dispatch at {context.warmest}", fontsize=FS["title"],
                   fontweight="bold")
    ax_b.tick_params(labelsize=FS["tick"])
    ax_b.grid(True, axis="y", color="0.93", lw=0.3)
    ax_b.set_axisbelow(True)
    fit_width(fig)
    legend_below(fig, [ax_a], *storage_handles(context.style), ncol=2)
    legend_below(fig, [ax_b], *ax_b.get_legend_handles_labels(), ncol=2)
    panel_letters(fig, [("a", ax_a), ("b", ax_b)])
    return fig


def panel_zone_season_counts(axes, events_by_season, style, seasons, duration_class: str):
    """Events per year by zone, one sub-axis per season, one bar per warming level.

    Parameters
    ----------
    axes : sequence of matplotlib.axes.Axes
        One per season.
    events_by_season : pandas.DataFrame
        Events per chain, level, zone, season and class.
    style : PaperStyle
        Zones, colours and warming levels.
    seasons : sequence of str
        Seasons, one per axis.
    duration_class : str
        Duration class drawn.
    """
    positions = np.arange(len(style.zones))
    median = events_by_season[events_by_season.duration_class == duration_class].groupby(
        ["season", "zone", "period"])["freq_per_year"].median()

    def value(season, zone, period):
        """Median of a cell, zero where the cell has no value."""
        found = median.get((season, zone, period), np.nan)
        return 0.0 if found is None or not np.isfinite(found) else float(found)

    for ax, season in zip(axes, seasons):
        top = 0.0
        for k, period in enumerate(style.period_names):
            values = [value(season, zone, period) for zone in style.zones]
            top = max(top, max(values))
            ax.bar(positions + (k - 1) * 0.26, values, 0.24, color=style.gwl_colors[period],
                   edgecolor="white", linewidth=0.3, zorder=2)
        for i, zone in enumerate(style.zones):
            if max(value(season, zone, period) for period in style.period_names) <= 1e-9:
                ax.annotate("no events", xy=(i, 0), xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", rotation=90, fontsize=FS["tick"], color="0.55")
        ax.set_ylim(0, max(top, 1.0) * 1.18)
        ax.set_xticks(positions)
        ax.set_xticklabels(style.labels(), fontsize=FS["tick"], rotation=90, va="top", ha="center")
        ax.set_xlim(-0.6, len(style.zones) - 0.4)
        ax.set_title(season_label(season), fontsize=FS["label"])
        ax.tick_params(labelsize=FS["tick"])
        ax.grid(True, axis="y", color="0.93", lw=0.3)
        ax.set_axisbelow(True)


def figure_s33(events_by_season_high, anomaly, survivor_zones, context):
    """S33: events that survive the HIGH/2050 build-out (a, b) and what drives them in winter (c, d).

    Parameters
    ----------
    events_by_season_high : pandas.DataFrame
        HIGH/2050 events per chain, level, zone, season and class, on the paper's threshold.
    anomaly : callable
        ``anomaly(block, zone, season) -> DataFrame``, per-chain driver anomalies during events
        (see :func:`~cordex2energy_analysis.tables.event_anomaly`).
    survivor_zones : sequence of str
        Zones with a winter event left, drawn in c and d.
    context : PaperContext
        Style.

    Returns
    -------
    matplotlib.figure.Figure
    """
    style = context.style
    fig = plt.figure(figsize=(WIDTH, 168 * MM))
    grid = gridspec.GridSpec(3, 1, height_ratios=[1.0, 1.0, 1.32], hspace=1.20, figure=fig)
    top = [fig.add_subplot(gridspec.GridSpecFromSubplotSpec(
        1, len(SEASON_ORDER), subplot_spec=grid[row], wspace=0.18)[i])
        for row in (0, 1) for i in range(len(SEASON_ORDER))]
    counts_a, counts_b = top[:len(SEASON_ORDER)], top[len(SEASON_ORDER):]

    # One y scale per row, so the seasons compare; multi-day keeps its own (five times rarer).
    for axes, duration_class in ((counts_a, "sub_daily"), (counts_b, "multi_day")):
        panel_zone_season_counts(axes, events_by_season_high, style, SEASON_ORDER, duration_class)
        shared = max(ax.get_ylim()[1] for ax in axes)
        for k, ax in enumerate(axes):
            ax.set_ylim(0, shared)
            if k:
                ax.set_yticklabels([])
        axes[0].set_ylabel(f"{DURATION_TITLES[duration_class]}\nevents yr$^{{-1}}$",
                           fontsize=FS["label"])

    bottom = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=grid[2], wspace=0.30)
    survivor_style = replace(style, zones=list(survivor_zones))
    ax_c = fig.add_subplot(bottom[0])
    panel_driver_attribution(ax_c, lambda zone, season: anomaly("weather", zone, season),
                             ["DJF"], survivor_style, ylabel="anomaly per 3h step (GWh)",
                             sublabels=[""])
    ax_d = fig.add_subplot(bottom[1])
    panel_driver_attribution(ax_d, lambda zone, season: anomaly("total", zone, season),
                             ["DJF"], survivor_style, ylabel="anomaly per 3h step (GWh)",
                             show_sign_convention=False, sublabels=[""], reserve_sign_gutter=False)
    for ax, title in ((ax_c, "Winter event anomaly,\nrelative to hour of day mean"),
                      (ax_d, "Winter event anomaly,\nrelative to winter mean")):
        ax.set_title(title, fontsize=FS["title"], fontweight="bold")

    fit_width(fig)
    # The warming-level key belongs to rows a and b: centred in the band between them and row c.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    band_low = min(ax.get_tightbbox(renderer).y0 for ax in counts_b) / fig.bbox.height
    band_high = max(ax.get_tightbbox(renderer).y1 for ax in (ax_c, ax_d)) / fig.bbox.height
    gwl_handle_list, gwl_labels = gwl_handles(style)
    fig.legend(gwl_handle_list, gwl_labels, loc="center",
               bbox_to_anchor=(0.5, 0.5 * (band_low + band_high)), ncol=3, title=GWL_TITLE,
               frameon=False, fontsize=FS["tick"], handlelength=1.0, handletextpad=0.5,
               columnspacing=1.4)
    driver_handle_list, driver_labels = driver_handles(style)
    legend_below(fig, [ax_c, ax_d], driver_handle_list,
                 [DRIVER_DISPLAY.get(name, name) for name in driver_labels], ncol=4)
    panel_letters(fig, [("a", counts_a[0]), ("b", counts_b[0]), ("c", ax_c), ("d", ax_d)])
    return fig
