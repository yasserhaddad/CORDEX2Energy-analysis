"""Builders of the Supplementary Note 1 figures (S1-S16) and the Figure 1 inset panels.

Each builder takes the data it draws and returns a matplotlib figure at the journal's artwork
size; the notebook loads the data, calls the builder and saves the result.
"""

from __future__ import annotations

import json

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from cordex2energy.conversion_energy import demand_coefficients

from .config import GWL_COLORS, REPOSITORY, SEASON_ORDER, WARMING_LEVELS
from .demand_validation import COOL_SECTORS, NO_JRC_ISO2, SECTOR_LABEL
from .pipeline_inputs import CLIMATE_UNITS, DRIVING_KEYS, GCM_DISPLAY, RCM_DISPLAY, climate_rows
from .plotting_paper import plot_country_choropleth, set_map_aspect
from .plotting_publication import (
    DRIVING_GCM_COLORS,
    DRIVING_GCM_ORDER,
    FS,
    GWL_TITLE,
    MAP_EXTENT_LAEA,
    MM,
    RCM_MARKERS,
    WIDTH,
    _draw_grid,
    _to_laea,
    fit_width,
    gwl_handles,
    gwl_maps,
    legend_below,
    panel_gcm_gmst,
    panel_letters,
    season_label,
    sign_agreement,
)

#: Short forms of the baseload specifications on the S10 axis.
SPEC_SHORT = {"FE": "per-country", "noFE": "shared"}
SECTOR_COLOR = {"res": "#C0392B", "srv": "#2E6F9E"}
#: S9 class edges: NRMSE (%) and interannual correlation, shared between the fuels.
NRMSE_BINS = [0, 5, 10, 20, 30, 50, 80]
CORR_BINS = [-0.4, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
NRMSE_CMAP = plt.get_cmap("magma_r", len(NRMSE_BINS) - 1)
CORR_CMAP = plt.get_cmap("YlGn", len(CORR_BINS) - 1)
#: Fill of the countries in the study but not in JRC-IDEES, and of those outside the study.
NO_JRC_COLOR = "#8F8F8F"
OUTSIDE_COLOR = "lightgrey"
#: States too small to read as a filled polygon, drawn as a circle.
SMALL_STATES = ("MT", "LU")
#: Row scaling of the conversion figures: per-country totals share the seasons' absolute scale,
#: capacity factors share every scale.
COUNTRY_SCALING = dict(share_abs_scale=list(range(1, 1 + len(SEASON_ORDER))))
GRID_SCALING = dict(share_abs_scale=True, share_change_scale=True)


# ---------------------------------------------------------------------------
# Figure 1 insets
# ---------------------------------------------------------------------------

def pipeline_panel(cooling, period: str, color: str, y_limits: tuple, *,
                   country_label: str = "Spain", window_years: int = 20):
    """One warming level of the cooling-demand inset: every chain thin, the median bold.

    Parameters
    ----------
    cooling : pandas.DataFrame
        From :func:`~cordex2energy_analysis.pipeline_inputs.annual_country_cooling`.
    period : str
        Warming level to draw.
    color : str
        The level's colour.
    y_limits : tuple of float
        ``(bottom, top)``, shared by the three levels.
    country_label : str
        Country named on the y axis.
    window_years : int
        Length of a warming-level window.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(WIDTH * 0.32, 38 * MM))
    block = cooling[cooling.period == period]
    for _, chain in block.groupby("model"):
        chain = chain.sort_values("year_in_window")
        ax.plot(chain["year_in_window"], chain["twh"], lw=0.5, color=color, alpha=0.45, zorder=2)
    median = block.groupby("year_in_window")["twh"].median()
    ax.plot(median.index, median.values, lw=1.4, color=color, zorder=3)

    ax.set_ylim(*y_limits)
    ax.set_xlim(1, window_years)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_xlabel("year within the 20-year window", fontsize=FS["label"])
    ax.set_ylabel(f"{country_label} cooling demand\n(TWh yr$^{{-1}}$)", fontsize=FS["label"])
    ax.set_title(period, fontsize=FS["title"], fontweight="bold", color=color)
    ax.tick_params(labelsize=FS["tick"])
    ax.grid(True, color="0.93", lw=0.3)
    ax.set_axisbelow(True)
    return fig


def pipeline_panels(cooling, period_names) -> dict:
    """The three cooling-demand insets of Figure 1, on one y scale, a third of the page each.

    Parameters
    ----------
    cooling : pandas.DataFrame
        From :func:`~cordex2energy_analysis.pipeline_inputs.annual_country_cooling`.
    period_names : sequence of str
        Warming-level labels, in the order of
        :data:`~cordex2energy_analysis.config.WARMING_LEVELS`.

    Returns
    -------
    dict
        ``{file suffix ("1p5", ...): Figure}``.
    """
    y_limits = (cooling["twh"].min() * 0.94, cooling["twh"].max() * 1.06)
    figures = {}
    for period, level, color in zip(period_names, WARMING_LEVELS, GWL_COLORS):
        fig = pipeline_panel(cooling, period, color, y_limits)
        fit_width(fig, target_mm=58)
        figures[f"{level:.1f}".replace(".", "p")] = fig
    return figures


# ---------------------------------------------------------------------------
# S1-S2: the chains
# ---------------------------------------------------------------------------

def fit_window_labels(ax, starts, ends, *, fig):
    """Widen the x axis so the rightmost window label ends inside it.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The window panel.
    starts, ends : sequence of int
        First and last years of the drawn windows.
    fig : matplotlib.figure.Figure
        The figure, drawn once to measure the label.

    Notes
    -----
    A label of ``L`` pixels in an axis of ``W`` pixels leaves ``D = S (1 - L/W)`` of the span ``S``
    for the data, so the span that fits both is ``D / (1 - L/W)``, with a 14 % margin.
    """
    fig.canvas.draw()
    probe = ax.text(0, 0, "8888-8888", fontsize=FS["tick"])
    label_px = probe.get_window_extent(fig.canvas.get_renderer()).width
    probe.remove()
    axis_px = ax.get_window_extent().width
    data_span = (max(ends) + 1.5) - (min(starts) - 4)
    span = data_span / max(1.0 - 1.14 * label_px / axis_px, 0.35)
    ax.set_xlim(min(starts) - 4, min(starts) - 4 + span)


def figure_s1(gmst, gcm_rows, period_windows: dict, style):
    """S1: the driving GCMs inside CMIP5 (a) and the warming-level windows they define (b).

    Parameters
    ----------
    gmst : xarray.DataArray
        CMIP5 GMST anomalies, from :func:`~cordex2energy_analysis.pipeline_inputs.load_cmip5_gmst`.
    gcm_rows : list of tuple
        ``(GCM_member, chain)``, from
        :func:`~cordex2energy_analysis.pipeline_inputs.driving_gcm_rows`.
    period_windows : dict
        ``{warming level: {chain: (first year, last year)}}``.
    style : PaperStyle
        Supplies the warming-level colours.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 62 * MM),
                             gridspec_kw={"wspace": 0.30, "width_ratios": (1.0, 1.0)})
    ax_a, ax_b = axes
    panel_gcm_gmst(ax_a, gmst, driving=DRIVING_KEYS, warming_levels=WARMING_LEVELS,
                   level_colors=dict(zip(WARMING_LEVELS, GWL_COLORS)))

    height = 0.26
    starts, ends = [], []
    for row, (gcm, model) in enumerate(gcm_rows):
        for k, period in enumerate(period_windows):
            window = period_windows[period].get(model)
            if window is None:
                continue
            start, end = int(window[0]), int(window[1])
            starts.append(start)
            ends.append(end)
            ax_b.barh(row + (k - 1) * height, end - start + 1, left=start, height=height * 0.86,
                      color=style.gwl_colors[period], edgecolor="white", linewidth=0.3, zorder=3)
            ax_b.text(end + 1.5, row + (k - 1) * height, f"{start}-{end}", va="center", ha="left",
                      fontsize=FS["tick"], color="0.35")
    ax_b.set_yticks(range(len(gcm_rows)))
    ax_b.set_yticklabels([GCM_DISPLAY[gcm] for gcm, _ in gcm_rows], fontsize=FS["label"])
    ax_b.invert_yaxis()
    ax_b.set_xlabel("year", fontsize=FS["label"])
    ax_b.set_xlim(min(starts) - 4, max(ends) + 1.5)
    ax_b.tick_params(labelsize=FS["tick"])
    ax_b.grid(True, axis="x", color="0.90", lw=0.3)
    ax_b.set_axisbelow(True)

    fit_width(fig)
    fit_window_labels(ax_b, starts, ends, fig=fig)

    gcm_handles = [plt.Line2D([], [], color="0.72", lw=1.0)]
    gcm_labels = ["other CMIP5 models"]
    for gcm in DRIVING_GCM_ORDER:
        if gcm in DRIVING_KEYS:
            gcm_handles.append(plt.Line2D([], [], color=DRIVING_GCM_COLORS[gcm], lw=1.4))
            gcm_labels.append(gcm)
    legend_below(fig, [ax_a], gcm_handles, gcm_labels, ncol=3)
    level_handles, level_labels = gwl_handles(style)
    legend_below(fig, [ax_b], level_handles, list(level_labels), ncol=3, title=GWL_TITLE)
    panel_letters(fig, [("a", ax_a), ("b", ax_b)])
    return fig


def figure_s2(seasonal_change, chain_metadata_path=REPOSITORY / "models.json"):
    """S2: each chain's change in precipitation (a) and runoff (b) against warming, per season.

    Parameters
    ----------
    seasonal_change : pandas.DataFrame
        From :func:`~cordex2energy_analysis.pipeline_inputs.european_seasonal_change`.
    chain_metadata_path : pathlib.Path
        ``models.json``, which names each chain's RCM and GCM.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(2, len(SEASON_ORDER), figsize=(WIDTH, 88 * MM),
                             gridspec_kw={"hspace": 0.42, "wspace": 0.10})
    with open(chain_metadata_path) as handle:
        chain_metadata = {chain["id"]: chain for chain in json.load(handle)["model_chains"]}
    cordex_to_cmip5 = {"CNRM-CERFACS-CNRM-CM5": "CNRM-CM5", "MOHC-HadGEM2-ES": "HadGEM2-ES",
                       "NCC-NorESM1-M": "NorESM1-M", "MPI-M-MPI-ESM-LR": "MPI-ESM-LR",
                       "ICHEC-EC-EARTH": "EC-EARTH"}
    for row, (column, label) in enumerate((("dP", "change in precipitation\n(mm d$^{-1}$)"),
                                           ("dR", "change in runoff\n(mm d$^{-1}$)"))):
        span = seasonal_change[column]
        pad = 0.08 * (span.max() - span.min())
        for col, season in enumerate(SEASON_ORDER):
            ax = axes[row, col]
            block = seasonal_change[seasonal_change.season == season]
            ax.axhline(0.0, color="0.45", lw=0.7, zorder=1)
            for _, point in block.iterrows():
                info = chain_metadata.get(point["model"], {})
                gcm = cordex_to_cmip5.get(info.get("gcm"), info.get("gcm"))
                ax.plot(point["dT"], point[column],
                        RCM_MARKERS.get(info.get("rcm"), "o"), ms=3.8,
                        color=DRIVING_GCM_COLORS.get(gcm, "0.5"), mec="white", mew=0.4, zorder=3)
            ax.set_ylim(span.min() - pad, span.max() + pad)
            ax.tick_params(labelsize=FS["tick"])
            ax.grid(color="0.93", lw=0.3)
            ax.set_axisbelow(True)
            if row == 0:
                ax.set_title(season_label(season), fontsize=FS["title"], fontweight="bold")
            if col:
                ax.set_yticklabels([])
            else:
                ax.set_ylabel(label, fontsize=FS["label"])
            if row == 1:
                ax.set_xlabel("ΔT (K)", fontsize=FS["label"])
    fit_width(fig)
    fig.align_ylabels(axes[:, 0])
    handles = [plt.Line2D([], [], color="0.45", marker=marker, ls="none", ms=4)
               for marker in RCM_MARKERS.values()]
    labels = [RCM_DISPLAY.get(name, name) for name in RCM_MARKERS]
    handles += [plt.Line2D([], [], color=DRIVING_GCM_COLORS[gcm], marker="s", ls="none", ms=3.4)
                for gcm in DRIVING_GCM_ORDER]
    labels += list(DRIVING_GCM_ORDER)
    legend_below(fig, list(axes.ravel()), handles, labels, ncol=5)
    panel_letters(fig, list(zip("ab", [axes[0, 0], axes[1, 0]])))
    return fig


# ---------------------------------------------------------------------------
# S3-S8: the climate variables
# ---------------------------------------------------------------------------

def climate_figure(grids: dict, countries_print, style, *, season=None, height_mm: float = 174):
    """S3-S7: the bias-corrected climate variables, annual or for one season.

    Parameters
    ----------
    grids : dict
        ``{warming level: Dataset}`` from
        :func:`~cordex2energy_analysis.pipeline_inputs.load_gwl_grids`.
    countries_print : geopandas.GeoDataFrame
        Country boundaries at print resolution.
    style : PaperStyle
        Figure style.
    season : str, optional
        Season to draw from seasonal grids.
    height_mm : float
        Figure height.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, row_axes = gwl_maps(climate_rows(grids, season=season), countries_print, style,
                             sign_agreement, height_mm=height_mm)
    panel_letters(fig, list(zip("abcde", row_axes)), inside=True)
    return fig


def figure_s8(fields, countries_print):
    """S8: runoff (a) beside the routed channel discharge (b), ensemble median at one level.

    Parameters
    ----------
    fields : xarray.Dataset
        Climate grids at one warming level, with ``mrro`` and ``Qrouted``.
    countries_print : geopandas.GeoDataFrame
        Country boundaries at print resolution.

    Returns
    -------
    matplotlib.figure.Figure

    Notes
    -----
    Runoff is on a linear scale clipped at its 99th percentile; discharge on a logarithmic scale
    from the median of its positive cells, below which it is left blank.
    """
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 66 * MM), gridspec_kw={"wspace": 0.06})

    for ax, (variable, cmap) in zip(axes, (("mrro", "PuBu"), ("Qrouted", "Blues"))):
        field = fields[variable].median("model")
        finite = field.values[np.isfinite(field.values)]
        positive = finite[finite > 0]
        if variable == "Qrouted":
            floor = max(float(np.percentile(positive, 50)), 1e-3)
            norm = mcolors.LogNorm(vmin=floor, vmax=float(np.nanmax(finite)))
            field = field.where(field >= floor)
            extend = "min"
        else:
            norm = mcolors.Normalize(vmin=0.0, vmax=float(np.percentile(positive, 99)))
            extend = "max"

        colormap = plt.get_cmap(cmap).copy()
        colormap.set_bad("white")
        mesh = _draw_grid(ax, field, crs=countries_print.crs, cmap=colormap, norm=norm)
        _, _, footprint = _to_laea(field, countries_print.crs)
        countries_print.clip(footprint).boundary.plot(ax=ax, color="0.35", linewidth=0.25)
        ax.set_xlim(*MAP_EXTENT_LAEA[:2])
        ax.set_ylim(*MAP_EXTENT_LAEA[2:])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        ax.set_aspect("equal")
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(f"{variable}  ({CLIMATE_UNITS[variable]})", fontsize=FS["title"], fontweight="bold")

        bar = fig.colorbar(mesh, ax=ax, fraction=0.030, pad=0.015, extend=extend)
        bar.ax.tick_params(labelsize=FS["tick"])
        bar.outline.set_linewidth(0.3)

    fit_width(fig)
    panel_letters(fig, list(zip("ab", list(axes))))
    return fig


# ---------------------------------------------------------------------------
# S9-S10: the demand model
# ---------------------------------------------------------------------------

def small_state_markers(ax, values_by_iso2, countries, cmap, norm):
    """Circles at the positions of the states too small to fill.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Map axes.
    values_by_iso2 : pandas.Series
        Mapped values, indexed by ISO2.
    countries : geopandas.GeoDataFrame
        Country polygons in the map projection.
    cmap : matplotlib.colors.Colormap
        Colour map of the values.
    norm : matplotlib.colors.Normalize
        Normalisation of the values.
    """
    for iso in SMALL_STATES:
        if iso not in values_by_iso2.index:
            continue
        point = countries.loc[countries["ISO2"] == iso, "geometry"].representative_point().iloc[0]
        ax.plot(point.x, point.y, "o", ms=3.4, color=cmap(norm(values_by_iso2[iso])),
                mec="0.25", mew=0.4, zorder=4)


def demand_map_panel(ax, values_by_iso2, countries, cmap, bins, title: str):
    """One classed choropleth of a demand-model score, gap-filled countries grey and hatched.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Map axes.
    values_by_iso2 : pandas.Series
        Score per country, indexed by ISO2.
    countries : geopandas.GeoDataFrame
        Country polygons in the map projection.
    cmap : matplotlib.colors.Colormap
        Classed colour map.
    bins : sequence of float
        Class edges.
    title : str
        Panel title.

    Raises
    ------
    ValueError
        If a value falls outside the class edges.
    """
    inside = (values_by_iso2 >= bins[0]) & (values_by_iso2 <= bins[-1])
    if not inside.all():
        raise ValueError(f"{title}: outside the classed scale {bins}: "
                         f"{values_by_iso2[~inside].round(2).to_dict()}")
    norm = mcolors.BoundaryNorm(bins, cmap.N)
    plot_country_choropleth(ax, values_by_iso2, countries, cmap=cmap, norm=norm, legend=False,
                            edgecolor="0.5", linewidth=0.25)
    small_state_markers(ax, values_by_iso2, countries, cmap, norm)
    countries[countries["ISO2"].isin(NO_JRC_ISO2.values())].plot(
        ax=ax, facecolor=NO_JRC_COLOR, edgecolor="0.25", hatch="////", linewidth=0.25, zorder=3)
    set_map_aspect(ax, countries)
    ax.set_title(title, fontsize=FS["title"], fontweight="bold")


def row_colorbar(fig, axes, cmap, bins, label: str):
    """One horizontal colorbar under a row of maps, ticked at the class edges.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure.
    axes : sequence of matplotlib.axes.Axes
        The row.
    cmap : matplotlib.colors.Colormap
        Classed colour map.
    bins : sequence of float
        Class edges.
    label : str
        Colorbar label.

    Returns
    -------
    matplotlib.colorbar.Colorbar
    """
    mappable = plt.cm.ScalarMappable(norm=mcolors.BoundaryNorm(bins, cmap.N), cmap=cmap)
    bar = fig.colorbar(mappable, ax=list(axes), orientation="horizontal", fraction=0.055,
                       pad=0.02, aspect=42, shrink=0.62, ticks=bins)
    bar.set_label(label, fontsize=FS["label"])
    bar.ax.tick_params(labelsize=FS["tick"], length=2)
    bar.outline.set_linewidth(0.4)
    return bar


def figure_s9(nrmse_by_iso2: dict, correlation_by_iso2: dict, countries):
    """S9: the demand model's NRMSE (a, b) and interannual correlation (c, d) per country.

    Parameters
    ----------
    nrmse_by_iso2, correlation_by_iso2 : dict
        ``{"heating": Series, "cooling": Series}`` from
        :func:`~cordex2energy_analysis.demand_validation.by_iso2`.
    countries : geopandas.GeoDataFrame
        Country polygons in the map projection.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 152 * MM),
                             gridspec_kw={"wspace": 0.02, "hspace": 0.55})
    (ax_a, ax_b), (ax_c, ax_d) = axes
    for ax, fuel in ((ax_a, "heating"), (ax_b, "cooling")):
        demand_map_panel(ax, nrmse_by_iso2[fuel], countries, NRMSE_CMAP, NRMSE_BINS,
                         f"{fuel.capitalize()}, NRMSE")
    for ax, fuel in ((ax_c, "heating"), (ax_d, "cooling")):
        demand_map_panel(ax, correlation_by_iso2[fuel], countries, CORR_CMAP, CORR_BINS,
                         f"{fuel.capitalize()}, correlation")
    row_colorbar(fig, (ax_a, ax_b), NRMSE_CMAP, NRMSE_BINS, "NRMSE (% of mean annual energy)")
    bar = row_colorbar(fig, (ax_c, ax_d), CORR_CMAP, CORR_BINS,
                       "correlation with JRC-IDEES, year to year")
    legend_below(fig, [ax_c, ax_d, bar.ax],
                 [Patch(facecolor=NO_JRC_COLOR, edgecolor="0.25", hatch="////", linewidth=0.3),
                  Patch(facecolor=OUTSIDE_COLOR, edgecolor="0.5", linewidth=0.3)],
                 ["in the study, not in JRC-IDEES",
                  "outside the study's country set"], ncol=2, gap_pt=2.0)
    panel_letters(fig, list(zip("abcd", [ax_a, ax_b, ax_c, ax_d])))
    fit_width(fig, target_mm=100)
    return fig


def figure_s10(theta: dict, candidates, observed_cdd):
    """S10: the fitted adoption share against HRE (a) and how sharply each curvature is pinned (b).

    Parameters
    ----------
    theta : dict
        ``{sector: curvature}`` of the production coefficients.
    candidates : pandas.DataFrame
        From :func:`~cordex2energy_analysis.demand_validation.curvature_candidates`.
    observed_cdd : pandas.Series
        Mean annual CDD of each fitted country over the fit window.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(WIDTH, 66 * MM),
                                     gridspec_kw={"wspace": 0.85, "width_ratios": (1.0, 1.18)})
    grid = np.linspace(0, 1200, 500)
    for sector in COOL_SECTORS:
        ax_a.plot(grid, 1 - np.exp(-theta[sector] * grid), color=SECTOR_COLOR[sector], lw=1.4,
                  label=f"{SECTOR_LABEL[sector]}, θ = {theta[sector]:.5f}")
        ax_a.plot(grid, 1 - np.exp(-demand_coefficients.HRE_THETA[sector] * grid), color=SECTOR_COLOR[sector],
                  lw=0.9, ls="--", alpha=0.75,
                  label=f"{SECTOR_LABEL[sector]}, HRE {demand_coefficients.HRE_THETA[sector]}")
    ax_a.plot(observed_cdd.values, np.full(len(observed_cdd), 0.02), "|", ms=4, mew=0.7,
              color="0.35", label="national CDD, CERRA 2000–2020 mean")
    ax_a.set_xlim(0, 1200)
    ax_a.set_ylim(0, 1.10)
    ax_a.set_xlabel("annual CDD (degree-days)", fontsize=FS["label"])
    ax_a.set_ylabel("adoption share  $s(\\mathrm{CDD})$", fontsize=FS["label"])
    ax_a.set_title("Fitted adoption share vs HRE", fontsize=FS["title"], fontweight="bold")
    ax_a.tick_params(labelsize=FS["tick"])
    ax_a.grid(color="0.93", lw=0.3)
    ax_a.set_axisbelow(True)

    order, labels = [], []
    for sector in COOL_SECTORS:
        block = candidates[candidates.sector == sector].sort_values("theta")
        for _, row in block.iterrows():
            order.append(row)
            labels.append(f"{row.group} - {SPEC_SHORT[row.spec]}")
    for position, row in enumerate(order):
        colour = SECTOR_COLOR[row.sector]
        ax_b.plot([row.band_lo, row.band_hi], [position, position], color=colour,
                  lw=2.4 if row.chosen else 1.0, alpha=1.0 if row.chosen else 0.45,
                  solid_capstyle="butt", zorder=2)
        ax_b.plot(row.theta, position, "o" if row.chosen else ".", ms=4.2 if row.chosen else 3.0,
                  color=colour, mec="white" if row.chosen else "none", mew=0.5, zorder=3)
        if row.chosen:
            ax_b.annotate("chosen", (row.band_hi, position), xytext=(3, 0), textcoords="offset points",
                          fontsize=FS["tick"], va="center", color=colour, fontweight="bold")
    ax_b.set_yticks(range(len(order)))
    ax_b.set_yticklabels(labels, fontsize=FS["tick"])
    ax_b.set_xscale("log")
    ax_b.set_xlabel("curvature θ, with its 1 %-RSS interval", fontsize=FS["label"])
    ax_b.set_title("Candidate curvatures and how sharply each is pinned", fontsize=FS["label"],
                   fontweight="bold")
    ax_b.tick_params(labelsize=FS["tick"])
    ax_b.grid(color="0.93", lw=0.3, axis="x")
    ax_b.set_axisbelow(True)
    ax_b.legend(handles=[plt.Line2D([], [], color=SECTOR_COLOR[sector], lw=2.0,
                                    label=SECTOR_LABEL[sector])
                         for sector in COOL_SECTORS],
                fontsize=FS["tick"], frameon=False, loc="lower right")
    panel_letters(fig, [("a", ax_a), ("b", ax_b)])
    # 178.5 mm: the saved tight bbox comes out ~1.7 mm wider than fit_width measures.
    fit_width(fig, target_mm=178.5)
    legend_below(fig, [ax_a], *ax_a.get_legend_handles_labels(), ncol=1)
    return fig


# ---------------------------------------------------------------------------
# S11-S16: the conversion steps
# ---------------------------------------------------------------------------

def conversion_figure(rows, countries_print, style, *, gridded: bool = False):
    """S11-S16: one conversion step, the annual row above the four seasons.

    Parameters
    ----------
    rows : list of MapRow
        From :func:`~cordex2energy_analysis.pipeline_inputs.conversion_rows`.
    countries_print : geopandas.GeoDataFrame
        Country boundaries at print resolution.
    style : PaperStyle
        Figure style.
    gridded : bool
        Capacity-factor rows, which share every scale; otherwise per-country totals, whose
        seasonal rows share one absolute scale and the annual row has its own.

    Returns
    -------
    matplotlib.figure.Figure
    """
    scaling = GRID_SCALING if gridded else COUNTRY_SCALING
    fig, row_axes = gwl_maps(rows, countries_print, style, sign_agreement, height_mm=196, **scaling)
    panel_letters(fig, list(zip("abcde", row_axes)), inside=True)
    return fig
