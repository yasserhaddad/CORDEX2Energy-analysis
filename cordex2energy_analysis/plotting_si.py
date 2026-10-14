"""Figure builders for the supplementary material.

Companion to :mod:`cordex2energy_analysis.plotting_paper` (the main text).  Same conventions — every
function returns a :class:`~matplotlib.figure.Figure`, and the shared presentation settings arrive as a
single :class:`~cordex2energy_analysis.plotting_paper.PaperStyle`.

Only the genuinely new builders live here.  Most SI figures call functions that already exist
(``plot_residual_load_kde``, ``plot_event_diurnal_profile``, ``plot_component_anomaly_distributions``,
``compute_unserved_energy_fraction`` …) — reuse those rather than reimplementing them.
"""

from __future__ import annotations

from typing import Callable, Mapping, Sequence

import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from .analysis import compute_sign_agreement
from .extremes_analysis import mask_structural_zeros
from .plotting import CMAPS
from .plotting_extremes import whis_for
from .plotting_paper import (
    ATTRIBUTION_SHADE_FLOOR,
    MAP_EXTENT,
    PaperStyle,
    _change_color,
    _hatch_disagreeing_countries,
    _symmetric_limit,
    plot_country_choropleth,
)

__all__ = [
    "plot_zone_map",
    "plot_gwl_windows",
    "plot_climate_gwl_maps",
    "plot_country_gwl_maps",
    "plot_tyndp_capacity",
    "plot_tyndp_baseload",
    "plot_cerra_vs_cordex",
    "plot_seasonal_operation",
    "plot_driver_attribution_spread",
    "plot_scenario_sensitivity",
    "plot_flexibility_counterfactual",
    "CLIMATE_CMAPS",
    "CLIMATE_UNITS",
    "ZONE_COLORS",
    "TECH_COLORS",
]

#: Sequential colormap per gridded variable — one hue, light -> dark, never a rainbow.  ``pv`` and
#: ``wind`` are taken from the project-wide :data:`cordex2energy_analysis.plotting.CMAPS` so the SI
#: capacity-factor maps (S5) look like every other PV/wind map in the project.
CLIMATE_CMAPS = {
    "tas": "inferno",
    "rsds": "plasma",
    "sfcWind": "GnBu",
    "pr": "YlGnBu",
    "mrro": "PuBu",
    "Qrouted": "GnBu",
    "pv": CMAPS["pv"],
    "wind": CMAPS["wind"],
}

#: Units, for the colorbar of the absolute (reference-level) panel.
CLIMATE_UNITS = {
    "tas": "K",
    "rsds": "W m$^{-2}$",
    "sfcWind": "m s$^{-1}$",
    "pr": "mm d$^{-1}$",
    "mrro": "kg m$^{-2}$",
    "Qrouted": "m$^3$ s$^{-1}$",
    "pv": "capacity factor",
    "wind": "capacity factor",
}

#: Variables whose absolute field is so channel-/peak-concentrated that a plain max saturates the
#: colour scale — the reference panel clips to a robust upper percentile instead.
_ROBUST_ABS_VMAX = {"Qrouted"}

#: One hue per balancing zone (S2).  Deliberately NOT the warming-level colours — these encode
#: identity, not magnitude, and reusing the GWL ramp here would suggest an order that does not exist.
ZONE_COLORS = {
    "Iberian_Peninsula": "#E6AB02",
    "South_East": "#D95F02",
    "Central_Western": "#7570B3",
    "Central_Eastern": "#A6761D",
    "Nordic": "#1B9E77",
    "British_Isles": "#E7298A",
}


# ---------------------------------------------------------------------------
# S1 — the ensemble and its GWL windows
# ---------------------------------------------------------------------------
def _driving_gcm(model: str) -> str:
    """The driving GCM of a chain name: everything after the leading RCM token."""
    return "_".join(model.split("_")[1:])


def plot_gwl_windows(period_defs: Mapping, models: Sequence[str], style: PaperStyle,
                     *, group_by_gcm: bool = True,
                     title: str = "S1 · The ensemble and its global-warming-level windows") -> Figure:
    """The three 20-year GWL windows on a common time axis.

    The windows are set by the *driving GCM*, so the two RCMs sharing a GCM produce identical bars.
    With ``group_by_gcm`` (the default) each GCM is drawn once — the timeline carries no duplication,
    and the RCM×GCM structure is left to the accompanying chain table.  Pass ``group_by_gcm=False`` to
    draw one row per chain.
    """
    periods = list(style.period_names)

    if group_by_gcm:
        # One representative chain per GCM; the windows are identical across its RCMs by construction.
        rows, seen = [], set()
        for model in models:
            gcm = _driving_gcm(model)
            if gcm not in seen:
                seen.add(gcm)
                rows.append((gcm, model))
        labels = [gcm.replace("_r1i1p1", "").replace("_r12i1p1", "") for gcm, _ in rows]
        keys = [model for _, model in rows]
    else:
        keys = list(models)
        labels = [m.replace("_r1i1p1", "").replace("_r12i1p1", "") for m in models]

    fig, ax = plt.subplots(figsize=(11, 0.5 * len(keys) + 2.2))

    for row, key in enumerate(keys):
        y = len(keys) - row - 1
        for k, period in enumerate(periods):
            window = period_defs[period].get(key)
            if window is None:
                continue
            start, end = int(window[0]), int(window[1])
            ax.barh(y + (k - 1) * 0.26, end - start + 1, left=start, height=0.24,
                    color=style.gwl_colors[period], edgecolor="white", lw=0.5,
                    label=period if row == 0 else None, zorder=3)
            ax.text(end + 1.5, y + (k - 1) * 0.26, f"{start}–{end}", va="center", ha="left",
                    fontsize=6.5, color="0.35")

    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels(list(reversed(labels)), fontsize=8)
    ax.set_xlabel("year", fontsize=9)
    ax.set_xlim(2000, 2100)
    ax.grid(axis="x", color="0.92", lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, frameon=False, ncol=len(periods), loc="lower right")
    ax.tick_params(labelsize=8)
    subtitle = "one row per driving GCM (the two RCMs share each GCM's windows)" if group_by_gcm else ""
    ax.set_title(f"{title}\n{subtitle}" if subtitle else title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# S2 — the balancing zones
# ---------------------------------------------------------------------------
def plot_zone_map(zone_members: Mapping[str, Sequence[str]], countries: gpd.GeoDataFrame,
                  style: PaperStyle,
                  *, title: str = "S2 · The six TYNDP balancing zones") -> Figure:
    """Countries coloured by balancing zone, with the dissolved zone outline drawn on top.

    Countries inside the map extent that belong to no zone are greyed and labelled as out of scope —
    the reader must be able to tell "not modelled" from "modelled and small".
    """
    zone_of = {iso: zone for zone, members in zone_members.items() for iso in members}
    frame = countries.copy()
    frame["zone"] = frame["ISO2"].map(zone_of)

    # A modelled country with no polygon would vanish from the map without a word — say so instead.
    orphans = sorted(set(zone_of) - set(frame["ISO2"]))
    if orphans:
        raise KeyError(f"zone members with no country geometry (they would be silently missing "
                       f"from the map): {orphans}")

    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    frame[frame["zone"].isna()].plot(ax=ax, facecolor="0.88", edgecolor="white", linewidth=0.4)
    for zone in style.zones:
        members = frame[frame["zone"] == zone]
        if members.empty:
            continue
        members.plot(ax=ax, facecolor=ZONE_COLORS[zone], edgecolor="white", linewidth=0.4,
                     alpha=0.85)
        # the zone outline: dissolve the member countries so the reader sees the zone as one object
        outline = members.dissolve()
        outline.boundary.plot(ax=ax, color="0.15", linewidth=1.3, zorder=4)
        # Anchor the label inside the zone's LARGEST landmass.  The representative point of the
        # dissolved multipolygon can fall in open water (the British Isles land in the Atlantic).
        parts = outline.geometry.iloc[0]
        largest = max(getattr(parts, "geoms", [parts]), key=lambda g: g.area)
        point = largest.representative_point()
        ax.annotate(style.zone_labels[zone], xy=(point.x, point.y), ha="center", va="center",
                    fontsize=9, fontweight="bold", color="0.1", zorder=6,
                    bbox=dict(fc="white", ec="0.6", lw=0.6, alpha=0.85, pad=1.8))

    ax.set_xlim(MAP_EXTENT[0], MAP_EXTENT[1])
    ax.set_ylim(MAP_EXTENT[2], MAP_EXTENT[3])
    ax.set_axis_off()
    ax.set_aspect("auto")                    # geopandas forces "equal", which squeezes the map
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.annotate("grey = in the map extent but not modelled", xy=(0.01, 0.01),
                xycoords="axes fraction", fontsize=8, color="0.45")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# S3 / S4 — the climate variables at each warming level
# ---------------------------------------------------------------------------
def plot_climate_gwl_maps(
    gwl_fields: Mapping[str, xr.Dataset],
    variables: Sequence[str],
    countries: gpd.GeoDataFrame,
    style: PaperStyle,
    *,
    season: str | None = None,
    clip_percentile: float = 98,
    title: str = "S3 · Bias-corrected EURO-CORDEX climate variables across warming levels",
) -> Figure:
    """Rows = climate variable; column 0 = the absolute field at the reference level, then one column
    per warmer level showing the change.

    **The change type is read from each variable's ``change_type`` attribute**, written by
    ``scripts/compute_climate_means_gwl.py``: ``tas`` is shown as an ABSOLUTE change (K), every other
    variable as a percentage.  A percentage change in Kelvin is meaningless, so the caller must not be
    able to get this wrong — hence the attribute rather than a keyword.

    Hatching carries the same meaning as everywhere else in the paper: fewer than
    ``style.robustness_threshold`` of the chains agree on the sign of the change.
    """
    periods = list(style.period_names)
    fig, axes = plt.subplots(len(variables), len(periods), squeeze=False, constrained_layout=True,
                             figsize=(4.4 * len(periods), 3.1 * len(variables)))

    # Draw to the DATA's own bounds, not MAP_EXTENT: the grid stops short of the extent on the east
    # and north, so drawing to MAP_EXTENT leaves a strip of bare (unclipped) country outlines past the
    # data. Clip the country overlay to the same bounds so nothing is drawn where there is no field.
    grid = gwl_fields[periods[0]][variables[0]]
    lon0, lon1 = float(grid["lon"].min()), float(grid["lon"].max())
    lat0, lat1 = float(grid["lat"].min()), float(grid["lat"].max())
    countries = countries.cx[lon0:lon1, lat0:lat1]

    for row, variable in enumerate(variables):
        source = gwl_fields[periods[0]][variable]
        change_type = source.attrs.get("change_type")
        if change_type not in ("absolute", "percent"):
            raise ValueError(
                f"{variable}: change_type is {change_type!r}; the GWL file must declare 'absolute' "
                "or 'percent' (a % change in Kelvin is meaningless)")

        def field(period, variable=variable):
            da = gwl_fields[period][variable]
            return da.sel(season=season) if season is not None and "season" in da.dims else da

        reference_per_chain = field(periods[0])
        reference = reference_per_chain.mean("model")

        # column 0 — the absolute field at the reference level, so the reader can judge whether a
        # large % change sits on a baseline worth caring about
        ax = axes[row, 0]
        abs_vmax = (float(np.nanpercentile(reference.values, 99))
                    if variable in _ROBUST_ABS_VMAX else None)   # channel-concentrated (Qrouted)
        mesh = ax.pcolormesh(reference["lon"], reference["lat"], reference.values,
                             cmap=CLIMATE_CMAPS.get(variable, "viridis"), shading="auto",
                             vmax=abs_vmax)
        countries.boundary.plot(ax=ax, color="0.35", linewidth=0.25)
        bar = fig.colorbar(mesh, ax=ax, shrink=0.72, pad=0.02,
                           extend="max" if abs_vmax is not None else "neither")
        bar.set_label(CLIMATE_UNITS.get(variable, ""), fontsize=8)
        bar.ax.tick_params(labelsize=7)
        bar.ax.yaxis.set_major_locator(MaxNLocator(nbins=5))    # PV's 0–0.2 range is otherwise dense

        # columns 1+ — the change: absolute for tas, percent for the rest
        changes, agreements = [], []
        for period in periods[1:]:
            per_chain = field(period) - reference_per_chain
            if change_type == "percent":
                per_chain = 100 * per_chain / reference_per_chain
            per_chain = per_chain.where(np.isfinite(per_chain))
            changes.append(per_chain.mean("model"))
            agreements.append(compute_sign_agreement(per_chain, model_dim="model"))

        limit = max(_symmetric_limit(c, clip_percentile) for c in changes)
        norm = mcolors.TwoSlopeNorm(vcenter=0, vmin=-limit, vmax=limit)
        for col, (change, agreement) in enumerate(zip(changes, agreements), start=1):
            ax = axes[row, col]
            ax.pcolormesh(change["lon"], change["lat"], change.values, cmap="RdBu_r", norm=norm,
                          shading="auto")
            countries.boundary.plot(ax=ax, color="0.35", linewidth=0.25)
            _hatch_disagreement(ax, agreement, change, style)

        mappable = plt.cm.ScalarMappable(norm=norm, cmap="RdBu_r")
        mappable.set_array([])
        bar = fig.colorbar(mappable, ax=[axes[row, c] for c in range(1, len(periods))],
                           fraction=0.03, pad=0.01, extend="both")
        bar.set_label(f"Δ {CLIMATE_UNITS.get(variable, '')}" if change_type == "absolute"
                      else "% change", fontsize=8)
        bar.ax.tick_params(labelsize=7)

        for col in range(len(periods)):
            ax = axes[row, col]
            ax.set_xlim(lon0, lon1)
            ax.set_ylim(lat0, lat1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("auto")            # geopandas forces "equal", which squeezes the panel
            for spine in ax.spines.values():
                spine.set_visible(False)

        long_name = source.attrs.get("long_name", variable)
        axes[row, 0].set_ylabel(f"{variable}\n{long_name}", fontsize=9, fontweight="bold")

    for col, period in enumerate(periods):
        axes[0, col].set_title(period if col == 0 else f"{period}  −  {periods[0]}",
                               fontsize=11, fontweight="bold")

    fig.suptitle(title + (f" — {season}" if season else ""), fontsize=13, fontweight="bold")
    return fig


def _hatch_disagreement(ax, agreement, field, style: PaperStyle) -> None:
    """The 'not robust' idiom of the main text, on a grid: < 2/3 of the chains agree on the sign."""
    mask = ((agreement.values < style.robustness_threshold)
            & np.isfinite(field.values)).astype(float)
    overlay = ax.contourf(agreement["lon"].values, agreement["lat"].values, mask,
                          levels=[0.5, 1.5], hatches=["///"], colors="none", zorder=3)
    overlay.set_facecolor("none")
    overlay.set_edgecolor("black")
    overlay.set_linewidth(0)


# ---------------------------------------------------------------------------
# S6 — country-level fields across warming levels (the choropleth analog of S3)
# ---------------------------------------------------------------------------
def plot_country_gwl_maps(
    fields: Mapping[str, Mapping[str, xr.DataArray]],
    countries: gpd.GeoDataFrame,
    style: PaperStyle,
    *,
    unit: str = "TWh / yr",
    abs_cmap: str = "Blues",
    title: str = "S6 · Country-level fields across warming levels",
) -> Figure:
    """Rows = quantity; column 0 = the absolute field at the reference level, then one column per
    warmer level showing the **% change** — the choropleth analog of :func:`plot_climate_gwl_maps`.

    ``fields`` maps a row label to ``{period: DataArray(model, country)}``.  The figure carries exactly
    two colorbars, each **common to all rows**: one absolute (column 0) and one diverging %-change
    (the warmer-level columns), both on a single shared scale so the rows are directly comparable.
    Change columns hatch countries where fewer than ``style.robustness_threshold`` chains agree on the
    sign; structural zeros (no fleet in a country) are greyed, never drawn as a real value.
    """
    periods = list(style.period_names)
    reference_period = periods[0]
    labels = list(fields)

    # Pre-reduce every row to the ensemble median (absolute reference + per-level % change +
    # sign agreement), so the shared colour scales can be set from all rows at once.
    references, changes_by_row, agreements_by_row = {}, {}, {}
    for label in labels:
        reference_per_chain = fields[label][reference_period]
        references[label] = mask_structural_zeros(reference_per_chain.median("model"))
        changes, agreements = [], []
        for period in periods[1:]:
            per_chain = 100 * (fields[label][period] - reference_per_chain) / reference_per_chain
            per_chain = per_chain.where(np.isfinite(per_chain))
            changes.append(mask_structural_zeros(per_chain.median("model")))
            agreements.append(compute_sign_agreement(per_chain, model_dim="model"))
        changes_by_row[label], agreements_by_row[label] = changes, agreements

    # One absolute scale and one %-change scale, shared across every row.
    abs_values = np.concatenate([r.values[np.isfinite(r.values)] for r in references.values()])
    abs_vmin, abs_vmax = float(np.nanmin(abs_values)), float(np.nanpercentile(abs_values, 98))
    change_values = np.concatenate([c.values[np.isfinite(c.values)]
                                    for cs in changes_by_row.values() for c in cs])
    limit = float(np.nanpercentile(np.abs(change_values), 98)) if change_values.size else 1.0

    # Explicit GridSpec with a dedicated bottom row for the two colorbars — so every map cell is the
    # same size (equal aspect) by construction.  Attaching colorbars to axis subsets instead lets the
    # layout engine reserve unequal space below column 0 vs the change columns, which distorts the
    # panels; a colorbar row avoids that entirely.
    nrows, ncols = len(labels), len(periods)
    fig = plt.figure(figsize=(4.4 * ncols, 3.3 * nrows + 1.0))
    grid = fig.add_gridspec(nrows + 1, ncols, height_ratios=[1] * nrows + [0.14],
                            left=0.06, right=0.99, top=0.92, bottom=0.07, hspace=0.12, wspace=0.05)
    axes = np.array([[fig.add_subplot(grid[r, c]) for c in range(ncols)] for r in range(nrows)],
                    dtype=object)

    for row, label in enumerate(labels):
        plot_country_choropleth(axes[row, 0], references[label].to_series(), countries,
                                cmap=abs_cmap, vmin=abs_vmin, vmax=abs_vmax, legend=False)
        for col, (change, agreement) in enumerate(
                zip(changes_by_row[label], agreements_by_row[label]), start=1):
            plot_country_choropleth(axes[row, col], change.to_series(), countries, cmap="RdBu_r",
                                    vcenter=0, vmin=-limit, vmax=limit, legend=False)
            _hatch_disagreeing_countries(axes[row, col], agreement, change, countries, style)
        axes[row, 0].text(-0.04, 0.5, label, transform=axes[row, 0].transAxes, rotation=90,
                          va="center", ha="center", fontsize=9.5, fontweight="bold")

    for col, period in enumerate(periods):
        axes[0, col].set_title(period if col == 0 else f"{period}  −  {reference_period}",
                               fontsize=11, fontweight="bold")

    # Two shared colorbars in the dedicated bottom row: absolute under column 0, %-change under the rest.
    abs_bar = fig.colorbar(
        plt.cm.ScalarMappable(norm=mcolors.Normalize(vmin=abs_vmin, vmax=abs_vmax), cmap=abs_cmap),
        cax=fig.add_subplot(grid[nrows, 0]), orientation="horizontal", extend="max")
    abs_bar.set_label(unit, fontsize=9)
    abs_bar.ax.tick_params(labelsize=8)
    abs_bar.ax.xaxis.set_major_locator(MaxNLocator(nbins=5))

    change_bar = fig.colorbar(
        plt.cm.ScalarMappable(norm=mcolors.TwoSlopeNorm(vcenter=0, vmin=-limit, vmax=limit),
                              cmap="RdBu_r"),
        cax=fig.add_subplot(grid[nrows, 1:]), orientation="horizontal", extend="both")
    change_bar.set_label(f"% change vs {reference_period}", fontsize=9)
    change_bar.ax.tick_params(labelsize=8)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    return fig


# ---------------------------------------------------------------------------
# S7 — the TYNDP scenario inputs
# ---------------------------------------------------------------------------
#: Colours of the TYNDP technologies.  Generation matches ``DRIVER_PALETTE`` in the main text so a
#: reader moving between the SI and Figs 3–4 sees the same technology in the same hue.
TECH_COLORS = {
    "solar PV": "#E6AB02",
    "wind onshore": "#7570B3",
    "wind offshore": "#1B7C3D",
    "battery (prosumer)": "#C79A6B",
    "battery (utility)": "#8B5A2B",
}


def plot_tyndp_capacity(
    generation_gw: pd.DataFrame,
    battery_gwh: pd.DataFrame,
    *,
    scenario_label: str = "supply LOW / 2050",
    title: str = "S7a · TYNDP-2024 installed capacity",
) -> Figure:
    """What the scenario installs: generation and storage per country.

    ``generation_gw``   index = ISO2, columns = solar PV / wind onshore / wind offshore, in **GW**.
    ``battery_gwh``     index = ISO2, columns = battery (prosumer) / (utility), in **GWh**.

    Generation and storage get separate panels *because their units differ*: power (GW) and energy
    (GWh) do not stack.  Putting them in one bar would be a units error dressed up as a total.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))

    def stacked(ax, frame, ylabel, panel_title):
        order = frame.sum(axis=1).sort_values(ascending=False).index
        frame = frame.loc[order]
        bottoms = np.zeros(len(frame))
        for column in frame.columns:
            ax.bar(range(len(frame)), frame[column].values, bottom=bottoms,
                   color=TECH_COLORS.get(column, "0.6"), edgecolor="white", lw=0.4, label=column)
            bottoms += frame[column].values
        ax.set_xticks(range(len(frame)))
        ax.set_xticklabels(frame.index, fontsize=6.5, rotation=90)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(panel_title, fontsize=10, fontweight="bold")
        ax.legend(fontsize=8, frameon=False)
        ax.tick_params(labelsize=8)

    stacked(axes[0], generation_gw, "installed capacity (GW)",
            "a · Generation capacity per country")
    stacked(axes[1], battery_gwh, "energy capacity (GWh)",
            "b · Battery storage per country")

    fig.suptitle(f"{title} — {scenario_label}", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def plot_tyndp_baseload(
    baseload_gw: xr.DataArray,
    zone_members: Mapping[str, Sequence[str]],
    style: PaperStyle,
    *,
    scenario_label: str = "demand DE / 2050",
    title: str = "S7b · TYNDP-2024 typical-year baseload, per balancing zone",
) -> Figure:
    """The typical-year non-thermal demand — the largest single term in the residual load — resolved
    **per balancing zone** rather than aggregated, so the reader sees where the demand sits and how
    its seasonal shape differs between zones (the northern winter peak vs the southern summer one).

    ``baseload_gw`` is ``(hour_of_year, country)`` in GW; countries are mapped to zones via
    ``zone_members`` and summed within each zone.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    countries_in = {str(c) for c in baseload_gw["country"].values}
    days = np.arange(baseload_gw.sizes["hour_of_year"]) / 24

    for zone in style.zones:
        members = [c for c in zone_members[zone] if c in countries_in]
        if not members:
            continue
        zone_total = baseload_gw.sel(country=members).sum("country")
        smooth = pd.Series(zone_total.values).rolling(24 * 7, center=True, min_periods=1).mean()
        ax.plot(days, smooth.values, lw=1.8, color=ZONE_COLORS[zone], label=style.zone_labels[zone])

    ax.set_xlabel("day of the typical year", fontsize=10)
    ax.set_ylabel("baseload demand (GW, 7-day mean)", fontsize=10)
    ax.set_xlim(0, 365)
    ax.legend(fontsize=9, frameon=False, ncol=2, title="balancing zone", title_fontsize=9)
    ax.grid(color="0.93", lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=9)
    fig.suptitle(f"{title} — {scenario_label}", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


# ---------------------------------------------------------------------------
# S23 / S24 — how the flexibility assets actually operate over the year
# ---------------------------------------------------------------------------
#: Day-of-year index of each month's first day (non-leap), for readable x-ticks.
_MONTH_STARTS = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
_MONTH_LABELS = list("JFMAMJJASOND")


def plot_seasonal_operation(
    curves: Mapping[str, Mapping[str, np.ndarray]],
    style: PaperStyle,
    *,
    mode: str = "band",
    ylabel: str = "",
    ncols: int = 5,
    spread: tuple = (10, 90),
    title: str = "",
    caption: str = "",
) -> Figure:
    """Day-of-year operation of a flexibility asset — one panel per country/zone, one colour per
    warming level.

    ``curves`` maps a panel label to ``{period: (n_curves, 365)}`` — one curve per chain-year, from
    :func:`~cordex2energy_analysis.extremes_analysis.compute_annual_operation_curves`.

    ``mode``:
      * ``"spaghetti"`` — every chain-year drawn thin, plus a bold mean. Readable **only** for a
        smooth, strongly seasonal quantity (reservoir storage), where the thin lines add a real
        inter-annual envelope.
      * ``"band"`` — median + inter-annual percentile band. Use for anything carrying day-to-day
        weather noise (dispatch, throughput): there, 540 spaghetti lines collapse into a solid block
        and whichever warming level is drawn last hides the other two.
    """
    if mode not in ("band", "spaghetti"):
        raise ValueError(f"mode must be 'band' or 'spaghetti', got {mode!r}")

    labels = list(curves)
    nrows = int(np.ceil(len(labels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 2.9 * nrows + 0.6),
                             squeeze=False, sharex=True)
    day = np.arange(365)

    for index, label in enumerate(labels):
        ax = axes[index // ncols, index % ncols]
        for period in style.period_names:
            block = curves[label].get(period)
            if block is None or not len(block):
                continue
            colour = style.gwl_colors[period]
            if mode == "spaghetti":
                for curve in block:
                    ax.plot(day, curve, color=colour, lw=0.4, alpha=0.05, zorder=1)
                ax.plot(day, block.mean(axis=0), color=colour, lw=2.2, zorder=3, label=period)
            else:
                low, median, high = np.percentile(block, [spread[0], 50, spread[1]], axis=0)
                ax.fill_between(day, low, high, color=colour, alpha=0.18, lw=0, zorder=1)
                ax.plot(day, median, color=colour, lw=1.9, zorder=3, label=period)

        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.set_xlim(0, 364)
        ax.set_xticks(_MONTH_STARTS)
        ax.set_xticklabels(_MONTH_LABELS, fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(color="0.93", lw=0.6)
        ax.set_axisbelow(True)
        if index % ncols == 0:
            ax.set_ylabel(ylabel, fontsize=8.5)

    for index in range(len(labels), nrows * ncols):
        axes[index // ncols, index % ncols].set_axis_off()

    spread_note = (f"thin = every chain-year, bold = mean" if mode == "spaghetti"
                   else f"line = median, band = P{spread[0]}–P{spread[1]} across chain-years")
    axes[0, -1].legend(fontsize=7.5, frameon=False)
    fig.suptitle(f"{title}\n{style.n_models} chains × 20 years per level · {spread_note}"
                 + (f"\n{caption}" if caption else ""),
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93 if not caption else 0.90])
    return fig


# ---------------------------------------------------------------------------
# S8 — does CORDEX reproduce CERRA in energy space?
# ---------------------------------------------------------------------------
def plot_cerra_vs_cordex(
    cordex_reference: Mapping[str, xr.Dataset],
    cerra: xr.Dataset,
    cordex_threshold,
    cerra_threshold,
    style: PaperStyle,
    *,
    title: str = "S8 · Does CORDEX reproduce CERRA in energy space? (reference period)",
) -> Figure:
    """The paper's most attackable assumption, tested.

    The p95 deficit threshold is calibrated on **CERRA** and applied to **CORDEX**.  If the two
    disagree on the reference-period residual load, every event count in the paper inherits that bias.
    So: per zone, the CORDEX 9-chain envelope of the residual-load distribution against CERRA's, and
    the threshold each of them implies.
    """
    fig, axes = plt.subplots(2, len(style.zones), figsize=(3.0 * len(style.zones), 6.4),
                             squeeze=False)

    bias_rows = []
    for col, zone in enumerate(style.zones):
        # (top) the distributions
        ax = axes[0, col]
        cerra_values = cerra["residual_load"].sel(zone=zone).values
        cerra_values = cerra_values[np.isfinite(cerra_values)]
        lo, hi = np.percentile(cerra_values, [0.5, 99.5])
        edges = np.linspace(lo, hi, 90)
        centres = 0.5 * (edges[:-1] + edges[1:])

        per_chain = []
        for model, dataset in cordex_reference.items():
            values = dataset["residual_load"].sel(zone=zone).values
            values = values[np.isfinite(values)]
            per_chain.append(np.histogram(values, bins=edges, density=True)[0])
        per_chain = np.vstack(per_chain)
        ax.fill_between(centres, per_chain.min(axis=0), per_chain.max(axis=0),
                        color="#4A90E2", alpha=0.30, lw=0, label="CORDEX (9 chains)")
        ax.plot(centres, np.median(per_chain, axis=0), color="#1F4E79", lw=1.5)
        ax.plot(centres, np.histogram(cerra_values, bins=edges, density=True)[0],
                color="black", lw=1.6, ls="--", label="CERRA")
        ax.set_yticks([])
        ax.set_title(style.zone_labels[zone], fontsize=10, fontweight="bold")
        ax.set_xlabel("residual load (GWh / 3 h)", fontsize=8)
        ax.tick_params(labelsize=7)
        if col == 0:
            ax.set_ylabel("density", fontsize=9)
            ax.legend(fontsize=7, frameon=False)

        # (bottom) the p95 threshold each dataset implies
        ax = axes[1, col]
        cordex_thr = [float(cordex_threshold[m].sel(zone=zone)) for m in cordex_reference]
        cerra_thr = float(cerra_threshold.sel(zone=zone))
        ax.boxplot([cordex_thr], positions=[0], widths=0.5, patch_artist=True, showfliers=False,
                   medianprops=dict(color="black", lw=1.3),
                   boxprops=dict(facecolor="#4A90E2", alpha=0.7))
        ax.plot(np.zeros(len(cordex_thr)), cordex_thr, "o", ms=3, color="0.2", alpha=0.6)
        ax.axhline(cerra_thr, color="black", ls="--", lw=1.5)
        bias = 100 * (np.median(cordex_thr) - cerra_thr) / cerra_thr
        ax.annotate(f"{bias:+.0f}%", xy=(0.5, cerra_thr), xytext=(4, 4),
                    textcoords="offset points", fontsize=9, fontweight="bold",
                    color=_change_color(bias))
        ax.set_xticks([]); ax.set_xlim(-0.6, 0.9)
        ax.tick_params(labelsize=7)
        if col == 0:
            ax.set_ylabel("p95 threshold\n(GWh / 3 h)", fontsize=9)
        bias_rows.append(dict(zone=zone, cerra=cerra_thr,
                              cordex_median=float(np.median(cordex_thr)), bias_pct=bias))

    axes[1, 0].annotate("dashed = CERRA (what the paper uses)", xy=(0.02, 0.02),
                        xycoords="axes fraction", fontsize=7, color="0.4")
    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.bias_table = pd.DataFrame(bias_rows)      # the notebook prints this — state it, do not hide it
    return fig


# ---------------------------------------------------------------------------
# S14 — the inter-model spread the stacked bars of Figs 3c/4c cannot show
# ---------------------------------------------------------------------------
def plot_driver_attribution_spread(
    attribution: Callable[[str, str], pd.DataFrame],
    seasons: Sequence[str],
    style: PaperStyle,
    *,
    title: str = "S14 · Per-driver attribution — the inter-model spread behind Figs 3c and 4c",
) -> Figure:
    """One panel per zone; within it, a box per driver showing all 9 chains.

    Figs 3c and 4c stack the **median** driver contributions and deliberately carry no error bars on
    the segments: a stacked segment sits on a floating baseline, and its neighbours' errors are
    correlated with its own, so a whisker on a segment would be read as a confidence interval it is
    not.  The honest spread belongs here instead — same quantity, unstacked, every chain shown.

    ``attribution(zone, season)`` is the same callable Figs 3c/4c are drawn with, so this figure
    cannot drift away from them.
    """
    whis = whis_for(style.spread)
    n_cols = 3
    n_rows = int(np.ceil(len(style.zones) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.0 * n_cols, 4.2 * n_rows), squeeze=False)

    # Season = shade, exactly as in the main-text attribution panels.
    season_alpha = dict(zip(seasons, np.linspace(1.0, ATTRIBUTION_SHADE_FLOOR, len(seasons))))
    width = 0.8 / len(seasons)

    for index, zone in enumerate(style.zones):
        ax = axes[index // n_cols, index % n_cols]
        for s_index, season in enumerate(seasons):
            offset = (s_index - (len(seasons) - 1) / 2) * width
            per_chain = attribution(zone, season)
            for d_index, driver in enumerate(style.drivers):
                values = per_chain[driver].values
                boxes = ax.boxplot([values], positions=[d_index + offset], widths=width * 0.85,
                                   patch_artist=True, showfliers=False, whis=whis,
                                   medianprops=dict(color="black", lw=1.2))
                for box in boxes["boxes"]:
                    box.set_facecolor(style.driver_palette[driver])
                    box.set_alpha(season_alpha[season])
                    box.set_edgecolor("0.3")
                    box.set_lw(0.7)
                ax.plot(np.full(len(values), d_index + offset), values, "o", ms=2.2,
                        color="0.15", alpha=0.5, zorder=5)

        ax.axhline(0, color="0.25", lw=0.9)
        ax.set_xticks(range(len(style.drivers)))
        ax.set_xticklabels(style.drivers, fontsize=8, rotation=30, ha="right")
        ax.set_title(style.zone_labels[zone], fontsize=11, fontweight="bold")
        ax.tick_params(labelsize=8)
        if index % n_cols == 0:
            ax.set_ylabel(f"contribution to Δ residual load\n({style.warmest_period} − "
                          f"{style.reference_period}, GWh / 3 h)", fontsize=9)

    for index in range(len(style.zones), n_rows * n_cols):
        axes[index // n_cols, index % n_cols].set_axis_off()

    handles = [plt.Line2D([], [], marker="s", ls="", markersize=9, alpha=season_alpha[s],
                          color="0.35", label=s) for s in seasons]
    axes[0, -1].legend(handles=handles, fontsize=8, frameon=False, title="season",
                       title_fontsize=8, loc="best")
    fig.suptitle(f"{title}\nPositive = deepens the deficit · boxes = {style.spread_label} · "
                 f"dots = the {style.n_models} chains", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


# ---------------------------------------------------------------------------
# S21 — how much of the deficit do batteries actually absorb?
# ---------------------------------------------------------------------------
def plot_flexibility_counterfactual(
    events_with: pd.DataFrame,
    events_without: pd.DataFrame,
    style: PaperStyle,
    *,
    title: str = "S21 · Flexibility counterfactual — events with and without batteries",
) -> Figure:
    """Event frequency per zone with batteries and without, at every warming level.

    **Both runs are scored against the same absolute threshold** (the CERRA p95 of the run the paper
    uses).  That is the only comparison that means anything: re-calibrating the threshold on the
    battery-free system would move the goalposts by exactly the effect being measured, and the
    batteries would appear to do nothing.
    """
    classes = ["sub_daily", "multi_day"]
    fig, axes = plt.subplots(len(classes), 1, figsize=(12, 4.4 * len(classes)), squeeze=False)

    whis = whis_for(style.spread)
    positions = np.arange(len(style.zones))
    width = 0.8 / (2 * len(style.period_names))

    for row, duration_class in enumerate(classes):
        ax = axes[row, 0]
        slot = 0
        for period in style.period_names:
            for label, events, hatch in [("with batteries", events_with, ""),
                                         ("no batteries", events_without, "///")]:
                data = events[(events.duration_class == duration_class)
                              & (events.period == period)]
                per_zone = [data[data.zone == z]["freq_per_year"].values for z in style.zones]
                offset = (slot - (2 * len(style.period_names) - 1) / 2) * width
                boxes = ax.boxplot(per_zone, positions=positions + offset, widths=width * 0.85,
                                   patch_artist=True, showfliers=False, whis=whis,
                                   medianprops=dict(color="black", lw=1.1))
                for box in boxes["boxes"]:
                    box.set_facecolor(style.gwl_colors[period])
                    box.set_alpha(0.8)
                    box.set_edgecolor("0.25")
                    box.set_lw(0.7)
                    box.set_hatch(hatch)
                slot += 1

        ax.set_xticks(positions)
        ax.set_xticklabels(style.labels(), fontsize=9, rotation=15)
        ax.set_ylabel(f"{duration_class} events per year", fontsize=9)
        ax.set_title(duration_class, fontsize=10, fontweight="bold")
        ax.tick_params(labelsize=8)

    handles = [plt.Line2D([], [], marker="s", ls="", markersize=10, color=style.gwl_colors[p],
                          label=p) for p in style.period_names]
    handles += [plt.Rectangle((0, 0), 1, 1, facecolor="0.85", edgecolor="0.25", label="with batteries"),
                plt.Rectangle((0, 0), 1, 1, facecolor="0.85", edgecolor="0.25", hatch="///",
                              label="no batteries")]
    axes[0, 0].legend(handles=handles, fontsize=8, frameon=False, ncol=2)

    fig.suptitle(f"{title}\nBoth runs scored against the same CERRA p95 threshold",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


# ---------------------------------------------------------------------------
# S20 — does the story survive a different capacity build-out?
# ---------------------------------------------------------------------------
#: Scenario colours for the sensitivity figure (identity, not magnitude).
_SCENARIO_COLORS = ["#4A90E2", "#D0021B", "#F5A623"]


def plot_scenario_sensitivity(
    events_by_scenario: Mapping[str, pd.DataFrame],
    style: PaperStyle,
    *,
    title: str = "S20 · Scenario sensitivity — do the conclusions survive a different build-out?",
) -> Figure:
    """Event frequency under each TYNDP build-out, shown **both** as absolute counts and as % change.

    "Is the southern surge an artifact of the LOW capacity build-out?" is the first question a referee
    asks.  ``events_by_scenario`` maps a scenario label to its ``events_annual`` table.

    The build-outs have very different baselines (e.g. HIGH/2040 Nordic multi-day ≈ 0.05 events/yr), so
    a %-change alone misleads — a tiny baseline manufactures a huge percentage.  The **left column**
    therefore shows absolute frequencies at the reference and warmest level (open bar = reference, solid
    = warmest, coloured by scenario), where those near-zero baselines are self-evident; the **right
    column** shows the %-change for the cases whose baseline is large enough to carry one.
    """
    scenarios = list(events_by_scenario)
    classes = ["sub_daily", "multi_day"]
    positions = np.arange(len(style.zones))
    fig, axes = plt.subplots(len(classes), 2, figsize=(17, 4.4 * len(classes)), squeeze=False,
                             gridspec_kw={"width_ratios": [1.4, 1]})

    def medians(events, duration_class, period):
        data = events[events.duration_class == duration_class]
        return np.array([data[(data.zone == z) & (data.period == period)]["freq_per_year"].median()
                         for z in style.zones])

    for row, duration_class in enumerate(classes):
        # ── left: absolute frequencies, reference (open) and warmest (solid) per scenario ──────────
        ax = axes[row, 0]
        width = 0.8 / len(scenarios)
        for k, scenario in enumerate(scenarios):
            colour = _SCENARIO_COLORS[k % 3]
            offset = (k - (len(scenarios) - 1) / 2) * width
            ref = medians(events_by_scenario[scenario], duration_class, style.reference_period)
            warm = medians(events_by_scenario[scenario], duration_class, style.warmest_period)
            ax.bar(positions + offset, ref, width * 0.9, facecolor="white", edgecolor=colour, lw=1.3,
                   label=f"{scenario} · {style.reference_period}")
            ax.bar(positions + offset, warm, width * 0.9, facecolor=colour, edgecolor=colour,
                   alpha=0.85, label=f"{scenario} · {style.warmest_period}")
        ax.set_xticks(positions)
        ax.set_xticklabels(style.labels(), fontsize=8, rotation=18)
        ax.set_ylabel(f"{duration_class} events per year", fontsize=9)
        ax.set_title(f"{duration_class} — absolute frequency (open = {style.reference_period}, "
                     f"solid = {style.warmest_period})", fontsize=10, fontweight="bold")
        ax.legend(fontsize=7, frameon=False, ncol=len(scenarios))
        ax.tick_params(labelsize=8)

        # ── right: % change, greyed where the reference baseline is too small to carry one ─────────
        ax = axes[row, 1]
        width = 0.8 / len(scenarios)
        baseline_floor = 0.5           # events/yr; below this a % change is dominated by noise
        for k, scenario in enumerate(scenarios):
            ref = medians(events_by_scenario[scenario], duration_class, style.reference_period)
            warm = medians(events_by_scenario[scenario], duration_class, style.warmest_period)
            change = np.where(ref >= baseline_floor, 100 * (warm - ref) / ref, np.nan)
            offset = (k - (len(scenarios) - 1) / 2) * width
            ax.bar(positions + offset, change, width * 0.9, label=scenario,
                   color=_SCENARIO_COLORS[k % 3], edgecolor="white", lw=0.5)
            for x, value, base in zip(positions + offset, change, ref):
                if np.isfinite(value):
                    ax.annotate(f"{value:+.0f}", xy=(x, value), xytext=(0, 3 if value >= 0 else -11),
                                textcoords="offset points", ha="center", fontsize=7.5,
                                fontweight="bold", color=_change_color(value))
        ax.axhline(0, color="0.25", lw=0.9)
        ax.set_xticks(positions)
        ax.set_xticklabels(style.labels(), fontsize=8, rotation=18)
        ax.set_ylabel(f"% change ({style.warmest_period} vs {style.reference_period})", fontsize=9)
        ax.set_title(f"{duration_class} — % change (baseline ≥ {baseline_floor:.1f}/yr only)",
                     fontsize=10, fontweight="bold")
        ax.legend(fontsize=8, frameon=False, title="build-out", title_fontsize=8)
        ax.tick_params(labelsize=8)

    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig
