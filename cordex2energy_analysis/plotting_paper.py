"""Figure builders for the residual-load extremes paper (main text + supplementary).

Every function here returns a :class:`~matplotlib.figure.Figure`; saving is left to the caller, so
the notebook stays a narrative and this module stays a library.

Shared context
--------------
The figures need the same dozen presentation settings (zone order, warming-level colours, driver
palette, robustness threshold, spread convention...).  Rather than thread them through every
signature, they are bundled once into a :class:`PaperStyle` and passed as a single argument.

Encoding invariants
-------------------
These are load-bearing — the paper's uncertainty language depends on them being applied uniformly:

* **Hatching means "not robust"** — everywhere (maps, heatmaps, attribution stacks): fewer than
  ``style.robustness_threshold`` of the chains agree on the *sign* of the change.
* **No error bars on stacked-bar segments.**  Each segment sits on the cumulative sum of the ones
  below it, so an interval drawn there straddles a floating baseline; and the component errors are
  correlated (they sum exactly to the net).  Instead the segments are hatched, and the inter-model
  spread is shown on the **net** marker only, which has a fixed zero baseline.
* **Boxes carry the median and the inter-model spread** (``style.spread``: the IQR, or the full
  range).  Every label describing it derives from ``style.spread_label`` — never hardcode it.  Per-chain dots are available on every box
  (``show_points``) and are kept where the ensemble spread is itself the message (Fig 2b); they are
  turned off in the regime figures, where 6 zones × 3 levels × 9 chains would be 162 dots.  Per-chain
  robustness is reported regardless — as the ``n/9`` agreement counts on the figures, and in S8/S14.
* **Maps are drawn in ETRS89-LAEA (EPSG:3035) at equal aspect** when the geometry is loaded with
  ``crs=MAP_CRS``: the EU's official equal-area projection, so a country's drawn area is its true
  area. The older lon/lat drawing, still used by the legacy working notebooks, forced
  ``set_aspect("auto")`` to fill its panel box, which stretched the map by whatever that box's
  shape was and inflated the north (a degree of longitude is 0.64 of a degree of latitude at 50 N).
  :func:`map_extent` and :func:`set_map_aspect` pick the right behaviour from the geometry's CRS.
* **A percentage change is only plotted where the baseline is meaningful** (see
  ``min_baseline_fraction``): a ratio taken off a near-zero base is noise and would hijack the
  colour scale.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import gridspec
from matplotlib.figure import Figure

from .extremes_analysis import (
    attribution_change_by_model,
    compute_exceedance_probability,
    compute_exceedance_survival,
    compute_survival_x_grid,
)
from .plotting_extremes import SPREAD_LABEL, model_spread_bounds, spread_qpair, whis_for

__all__ = [
    "PaperStyle",
    "MAP_EXTENT", "MAP_CRS", "MAP_EXTENT_LAEA", "map_extent", "set_map_aspect", "is_projected",
    "RIVER_CMAP",
    "load_country_geometry",
    "plot_country_choropleth",
    "panel_event_frequency",
    "panel_driver_attribution",
    "panel_storage_attenuation",
    "plot_mean_state_maps",
    "plot_tail_survival",
    "plot_sub_daily_regime",
    "plot_multi_day_regime",
    "plot_threshold_calibration",
    "plot_zone_season_change",
    "plot_significance",
    "plot_level_vs_persistence",
    "plot_cooccurrence",
    "plot_threshold_sensitivity",
]

# Colour of a rising / falling quantity, used for every annotated change in the paper.
UP_COLOR = "#B2182B"
DOWN_COLOR = "#2166AC"

#: Alpha per season, dark -> light, applied on top of the warming-level colour.
#: The bars and the season legend both read from this, so they cannot drift apart.
SEASON_SHADES = (1.0, 0.72, 0.46, 0.24)

#: Faintest shade allowed in the driver-attribution panel.  It is deliberately much gentler than
#: ``SEASON_SHADES``: the *summer* bars there carry the paper's headline signal (the cooling-demand
#: spike) and the "not robust" hatching, so fading them to 0.46 would make the most important bars
#: the least legible.  Seasons are still ordered dark -> light, just over a shorter range.
ATTRIBUTION_SHADE_FLOOR = 0.55


# ---------------------------------------------------------------------------
# Shared context
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PaperStyle:
    """Presentation settings shared by every figure in the paper."""

    zones: Sequence[str]                    # narrative order (southern -> northern -> the outlier)
    zone_labels: Mapping[str, str]          # zone key -> short display label
    period_names: Sequence[str]             # e.g. ["+1.5 °C", "+2.0 °C", "+3.0 °C"]
    gwl_colors: Mapping[str, str]
    driver_palette: Mapping[str, str]       # ordered: the stacking order of the attribution bars
    duration_palette: Mapping[str, str]
    season_order: Sequence[str]
    n_models: int
    robustness_threshold: int               # e.g. 6 of 9 chains must agree on the sign
    spread: str = "iqr"                     # "iqr" (P25-P75) | "minmax" | "tukey" (1.5xIQR)

    @property
    def reference_period(self) -> str:
        return self.period_names[0]

    @property
    def warmest_period(self) -> str:
        return self.period_names[-1]

    @property
    def drivers(self) -> list[str]:
        return list(self.driver_palette)

    @property
    def spread_percentiles(self) -> tuple[float, float]:
        """(low, high) percentiles for the inter-model band — e.g. (25.0, 75.0)."""
        return spread_qpair(self.spread)

    @property
    def spread_label(self) -> str:
        """How the spread is described on the figures — follows ``spread``, never hardcoded."""
        return SPREAD_LABEL[self.spread]

    def labels(self) -> list[str]:
        return [self.zone_labels[z] for z in self.zones]


def _change_color(pct: float) -> str:
    return UP_COLOR if pct > 0 else DOWN_COLOR


# ---------------------------------------------------------------------------
# Maps
# ---------------------------------------------------------------------------
MAP_EXTENT = (-12, 40, 33, 72)          # lon_min, lon_max, lat_min, lat_max

#: The publication maps' projection: ETRS89 Lambert azimuthal equal-area, as used by Eurostat, the
#: EEA and INSPIRE for pan-European statistical maps.
MAP_CRS = "EPSG:3035"
#: Its extent, in metres. The study's 37 countries span x 2636-6501 km, y 1430-5423 km. The box
#: leaves ~160 km of sea west of Portugal, takes in all of Lebanon east of Cyprus (to 6745 km), and
#: drops to 1300 km so Crete (south tip 1423 km) has sea below it. All of it stays inside the
#: climate grid's footprint (its west edge reaches x = 2437 km at Portugal's latitude, its east edge
#: ~6960 km and its south edge ~930 km in the eastern Mediterranean), so gridded maps fill it too.
MAP_EXTENT_LAEA = (2480e3, 6800e3, 1300e3, 5470e3)
#: Generous lon/lat window the geometry is clipped to BEFORE projecting: an azimuthal projection
#: blows up toward its antipode, so the far east of Russia and overseas territories must not be
#: carried into it.
_PROJECTION_WINDOW = (-30, 60, 25, 76)


def is_projected(countries) -> bool:
    return countries.crs is not None and countries.crs.is_projected


def map_extent(countries):
    """(x0, x1, y0, y1) of the map in the geometry's own coordinates."""
    return MAP_EXTENT_LAEA if is_projected(countries) else MAP_EXTENT


def set_map_aspect(ax, countries) -> None:
    """Equal aspect on a projected map (the point of an equal-area projection); the legacy
    lon/lat maps keep "auto". Call after EVERY geopandas draw, which resets it to "equal"."""
    ax.set_aspect("equal" if is_projected(countries) else "auto")

#: Blue ramp used for hydro, matching ``CMAPS`` in :mod:`cordex2energy_analysis.plotting`.
RIVER_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "river", ["#1c4e80", "#4ea3c8", "#cfe9f2", "#ffffff"]
)


def load_country_geometry(natural_earth_geojson, crs=None) -> gpd.GeoDataFrame:
    """Country polygons for the choropleths, keyed by the ISO2 code the energy data uses.

    Natural Earth (not NUTS): NUTS dropped the UK after Brexit and omits non-NUTS states, which
    leaves blank gaps on the map.  Natural Earth covers the whole extent, so a country without data
    is *greyed* rather than missing.

    Two coding quirks are handled here: ``ISO_A2`` is ``"-99"`` for several countries (France,
    Norway, Kosovo) and falls back to ``ISO_A2_EH``; and Natural Earth codes the United Kingdom
    ``"GB"`` while the c2e data uses ``"UK"``.

    ``crs=MAP_CRS`` returns the geometry clipped to a generous European window and projected; the
    default keeps the legacy lon/lat geometry.
    """
    ne = gpd.read_file(natural_earth_geojson)
    iso2 = ne["ISO_A2"].where(ne["ISO_A2"] != "-99", ne["ISO_A2_EH"]).replace({"GB": "UK"})
    # Northern Cyprus is a separate admin-0 unit in Natural Earth with ISO "-99" in BOTH code
    # fields, so it dissolved into an anonymous "-99" blob and was drawn grey beside a coloured
    # south. The study treats Cyprus as one country, so the two are merged here.
    iso2 = iso2.where(ne["ADM0_A3"] != "CYN", "CY")
    countries = (
        ne.assign(ISO2=iso2)[["ISO2", "geometry"]]
        .dissolve(by="ISO2")
        .reset_index()
        .cx[MAP_EXTENT[0]:MAP_EXTENT[1], MAP_EXTENT[2]:MAP_EXTENT[3]]
    )
    if crs is None:
        return countries
    from shapely.geometry import box
    clipped = countries.clip(box(_PROJECTION_WINDOW[0], _PROJECTION_WINDOW[2],
                                 _PROJECTION_WINDOW[1], _PROJECTION_WINDOW[3]))
    return clipped[~clipped.geometry.is_empty].to_crs(crs).reset_index(drop=True)


def plot_country_choropleth(
    ax,
    values_by_iso2: pd.Series,
    countries: gpd.GeoDataFrame,
    *,
    cmap="viridis",
    vcenter: float | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    norm=None,
    clabel: str = "",
    extend: str = "neither",
    legend: bool = True,
    edgecolor: str = "white",
    linewidth: float = 0.3,
):
    """Draw one country choropleth; countries without a value are greyed, not dropped.

    ``norm`` takes precedence over ``vcenter``/``vmin``/``vmax`` when given — that is how a
    discrete :class:`~matplotlib.colors.BoundaryNorm` is passed in for a binned scale.

    ``edgecolor``/``linewidth`` default to today's white hairline.  A darker border is needed on a
    binned scale whose neutral class is near-white (``NEUTRAL_CLASS_COLOR``): white borders on an
    almost-white fill make adjacent countries indistinguishable exactly where the map is saying
    "nothing happens here".
    """
    values = values_by_iso2.rename("val").rename_axis("ISO2").reset_index()
    merged = countries.merge(values, on="ISO2", how="left")

    kw = dict(vmin=vmin, vmax=vmax)
    if norm is not None:
        kw = dict(norm=norm)
    elif vcenter is not None:
        lo = vmin if vmin is not None else float(np.nanmin(merged["val"]))
        hi = vmax if vmax is not None else float(np.nanmax(merged["val"]))
        kw = dict(norm=mcolors.TwoSlopeNorm(vcenter=vcenter,
                                            vmin=min(lo, vcenter - 1e-6),
                                            vmax=max(hi, vcenter + 1e-6)))
    merged.plot(column="val", cmap=cmap, legend=legend, ax=ax,
                legend_kwds={"label": clabel, "shrink": 0.72, "extend": extend},
                missing_kwds={"color": "lightgrey"},
                edgecolor=edgecolor, linewidth=linewidth, **kw)
    x0, x1, y0, y1 = map_extent(countries)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_axis_off()
    set_map_aspect(ax, countries)   # geopandas resets the aspect on every draw (see docstring)
    return ax


def _hatch_disagreeing_countries(ax, agreement, field, countries, style: PaperStyle,
                                 min_magnitude: float | None = None) -> None:
    """Hatch countries where fewer than ``style.robustness_threshold`` chains agree on the sign.

    ``min_magnitude`` suppresses the hatch where ``|value|`` is below it.  Sign agreement on a
    negligible change is not informative — if a country moves by 0.3 % the chains will disagree on
    the sign roughly at random, and hatching it says "uncertain" when the truthful statement is
    "flat".  Measured on Figure 1, *every* hatched solar country and 9 of 11 hatched wind countries
    were of this kind, which is what made the hatching read as noise.
    """
    isos = [
        str(country)
        for country, n_agree, value in zip(agreement["country"].values, agreement.values, field.values)
        if n_agree < style.robustness_threshold and np.isfinite(value)
        and (min_magnitude is None or abs(value) >= min_magnitude)
    ]
    if isos:
        countries[countries["ISO2"].isin(isos)].plot(
            ax=ax, facecolor="none", edgecolor="black", hatch="///", linewidth=0.0, zorder=3)


def _symmetric_limit(values, clip_percentile: float) -> float:
    """Robust symmetric colour limit; the extend arrows carry whatever lies beyond it."""
    finite = np.abs(values.values[np.isfinite(values.values)])
    return max(float(np.nanpercentile(finite, clip_percentile)), 1e-6) if finite.size else 1.0


#: Colour of the "no material change" class on a binned change scale.  Deliberately the RdBu/RdBu_r
#: midpoint rather than a grey: near-zero already renders as this colour on the continuous scale, so
#: binning only makes it an explicit, labelled class — and it stays distinguishable from the
#: ``lightgrey`` used for countries with no data at all.
NEUTRAL_CLASS_COLOR = "#f7f7f7"


def _zero_class(edges: Sequence[float]) -> tuple[float, float]:
    """The (low, high) edges of the bin containing zero — the "no material change" class."""
    index = int(np.searchsorted(edges, 0.0, side="right")) - 1
    if not 0 <= index < len(edges) - 1:
        raise ValueError(f"change bin edges must straddle zero; got {list(edges)}")
    return float(edges[index]), float(edges[index + 1])


def _binned_norm(edges: Sequence[float], cmap_name: str, *, by: str = "midpoint"):
    """A discrete (BoundaryNorm, ListedColormap) pair for a set of change-bin edges.

    ``by="midpoint"`` (default) samples each bin's colour at its **midpoint** on a symmetric scale
    rather than by bin index, so an asymmetric edge set (cooling runs +1 → +200 % with nothing
    below) still maps magnitude to colour intensity sensibly.

    ``by="rank"`` samples by each bin's ORDINAL distance from the zero class, both sides normalised
    by the longer side. For a geometric edge set spanning two orders of magnitude (Figure 3b runs
    -1 to +8 events per year, most cells inside +/-0.25) midpoint sampling washes every small class
    out to near-white; rank sampling keeps each class a distinct step, and a side with fewer classes
    simply never reaches the darkest shade.

    The bin containing zero is overridden to :data:`NEUTRAL_CLASS_COLOR`, and the under/over colours
    carry the extend arrows.
    """
    edges = [float(e) for e in edges]
    base = plt.get_cmap(cmap_name)
    mids = [0.5 * (lo + hi) for lo, hi in zip(edges[:-1], edges[1:])]
    if by == "rank":
        zero = next(i for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])) if lo <= 0.0 <= hi)
        depth = max(zero, len(mids) - 1 - zero) or 1
        colors = [base(0.5 * (1.0 + (i - zero) / depth)) for i in range(len(mids))]
    else:
        span = max(abs(edges[0]), abs(edges[-1])) or 1.0
        colors = [base(0.5 * (1.0 + np.clip(m / span, -1.0, 1.0))) for m in mids]

    low, high = _zero_class(edges)
    colors[edges.index(low)] = mcolors.to_rgba(NEUTRAL_CLASS_COLOR)

    discrete = mcolors.ListedColormap(colors)
    discrete.set_under(base(0.0))
    discrete.set_over(base(1.0))
    return mcolors.BoundaryNorm(edges, discrete.N), discrete


def _baseline_relevance_mask(reference, min_baseline_fraction: float):
    """Where the baseline is too small for a percentage change to mean anything.

    A country whose +1.5 °C baseline is a negligible fraction of the row's typical magnitude yields
    an enormous but meaningless ratio (Nordic cooling demand reaches +1500 %), which would hijack the
    colour scale.  Those countries are greyed instead.
    """
    magnitude = np.abs(reference.values)
    magnitude = magnitude[np.isfinite(magnitude)]
    if not magnitude.size:
        return None
    floor = min_baseline_fraction * float(np.nanpercentile(magnitude, 90))
    return np.abs(reference) >= floor


# ---------------------------------------------------------------------------
# Panels shared by the two regime figures (Figs 3 & 4)
# ---------------------------------------------------------------------------
def panel_event_frequency(ax, events: pd.DataFrame, duration_class: str, style: PaperStyle,
                          ylabel: str, *, show_points: bool = True,
                          show_legend: bool = True) -> None:
    """Events per year by zone, one box per warming level.

    ``show_points`` overlays every chain as a dot — honest, but 162 dots in the regime figures, which
    therefore pass ``False``.  ``show_legend`` is off in the regime figures too: they carry a single
    figure-level warming-level legend instead (a per-axes one lands on the British Isles boxes, which
    hug the bottom of the axis).
    """
    whis = whis_for(style.spread)
    data = events[events.duration_class == duration_class]
    width, positions = 0.26, np.arange(len(style.zones))

    for k, period in enumerate(style.period_names):
        offset = (k - 1) * width
        per_zone = [data[(data.zone == z) & (data.period == period)]["freq_per_year"].values
                    for z in style.zones]
        boxes = ax.boxplot(per_zone, positions=positions + offset, widths=width * 0.86,
                           patch_artist=True, showfliers=False, whis=whis,
                           medianprops=dict(color="black", lw=1.3))
        for box in boxes["boxes"]:
            box.set_facecolor(style.gwl_colors[period])
            box.set_alpha(0.75)
            box.set_edgecolor("0.3")
            box.set_lw(0.7)
        if show_points:
            for x, values in zip(positions + offset, per_zone):
                ax.plot(np.full(len(values), x), values, "o", ms=2.2, color="0.15", alpha=0.5,
                        zorder=5)

    ax.set_ylim(0, ax.get_ylim()[1] * 1.24)          # headroom for the annotations
    for i, zone in enumerate(style.zones):
        ref = data[(data.zone == zone) & (data.period == style.reference_period)] \
            .set_index("model")["freq_per_year"]
        warm = data[(data.zone == zone) & (data.period == style.warmest_period)] \
            .set_index("model")["freq_per_year"]
        if ref.median() <= 0:
            continue
        delta = (warm - ref).reindex(ref.index).dropna()
        sign = np.sign(np.median(delta))
        n_agree = int((np.sign(delta) == sign).sum())
        pct = 100 * (warm.median() - ref.median()) / ref.median()
        ax.annotate(f"{pct:+.0f}%\n{n_agree}/{style.n_models}",
                    xy=(i, ax.get_ylim()[1]), xytext=(0, -2), textcoords="offset points",
                    ha="center", va="top", fontsize=8, fontweight="bold", color=_change_color(pct))

    ax.set_xticks(positions)
    ax.set_xticklabels(style.labels(), fontsize=8, rotation=18)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)
    if show_legend:
        ax.legend(handles=[plt.Line2D([], [], marker="s", ls="", color=style.gwl_colors[p], label=p)
                           for p in style.period_names],
                  fontsize=7.5, frameon=False, ncol=1, loc="lower right")


def panel_driver_attribution(ax, attribution: Callable[[str, str], pd.DataFrame],
                             seasons: Sequence[str], style: PaperStyle, title: str,
                             *, show_net_spread: bool = True) -> None:
    """Diverging stacked bars: each driver's contribution to the change in residual load.

    ``attribution(zone, season)`` must return a per-chain DataFrame with one column per driver plus
    ``"net"`` (see :func:`~cordex2energy_analysis.extremes_analysis.attribution_change_by_model`).

    On the deliberate absence of error bars on the segments, see the module docstring.
    """
    width = 0.8 / len(seasons)
    positions = np.arange(len(style.zones))

    # Season = SHADE.  Hue is already the driver and hatching is already "not robust", so shade is
    # the only channel left.  ``seasons`` arrives in calendar order, so earlier = darker (as in
    # Fig 4b) — but over a gentler range (see ATTRIBUTION_SHADE_FLOOR).
    season_alpha = dict(zip(seasons, np.linspace(1.0, ATTRIBUTION_SHADE_FLOOR, len(seasons))))

    for s_index, season in enumerate(seasons):
        offset = (s_index - (len(seasons) - 1) / 2) * width
        alpha = season_alpha[season]
        for i, zone in enumerate(style.zones):
            per_chain = attribution(zone, season)
            median = per_chain.median()
            top = bottom = 0.0
            for driver in style.drivers:
                value = median[driver]
                n_agree = int((np.sign(per_chain[driver]) == np.sign(value)).sum())
                base = top if value >= 0 else bottom
                ax.bar(i + offset, value, width * 0.9, bottom=base,
                       color=style.driver_palette[driver], alpha=alpha,
                       edgecolor="white", linewidth=0.4, zorder=2,
                       hatch=("///" if n_agree < style.robustness_threshold else None))
                if value >= 0:
                    top += value
                else:
                    bottom += value

            net = per_chain["net"]
            low, high = model_spread_bounds(net.values, style.spread, axis=0)
            spread = ([[net.median() - low], [high - net.median()]]
                      if show_net_spread else None)
            ax.errorbar(i + offset, net.median(), yerr=spread,
                        fmt="D", ms=4.5, color="black", ecolor="black",
                        elinewidth=1.1, capsize=2.5, zorder=6)
            if s_index == 0:
                ax.axvline(i + 0.5, color="0.9", lw=0.6, zorder=0)

    ax.axhline(0, color="0.25", lw=0.9, zorder=3)
    ax.set_xticks(positions)
    ax.set_xticklabels(style.labels(), fontsize=8, rotation=18)
    ax.set_ylabel("Δ residual load (GWh / 3 h)", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.tick_params(labelsize=8)
    _annotate_sign_convention(ax)

    # A season key. Fig 3 carries no season encoding anywhere else, so the shade must be keyed here
    # (Fig 4b keys the same shades). Neutral swatches: a coloured one would read as a driver.
    season_key = ax.legend(
        handles=[plt.Rectangle((0, 0), 1, 1, facecolor="0.35", alpha=season_alpha[s],
                               edgecolor="white") for s in seasons],
        labels=list(seasons), title="season (shade)", fontsize=7.5, title_fontsize=7.5,
        frameon=False, ncol=len(seasons), loc="upper left")
    ax.add_artist(season_key)          # keep it when the driver legend is added afterwards


def _annotate_sign_convention(ax, *, arrow_x: float = -0.115, label_x: float = -0.155,
                              labels=("deepens the deficit", "eases the deficit")) -> None:
    """Two arrows outside the y-axis saying what the sign of the change means.

    This belongs on the axis, not smuggled into the y-label.  Drawn outside the axes with
    ``clip_on=False``; the figures are saved with ``bbox_inches="tight"``, which expands the canvas
    to include them.

    ``arrow_x`` and ``label_x`` are axes fractions, so the same defaults put the arrows three times
    further from a full-width panel than from a half-width one.  Callers that draw at more than one
    width pass fractions computed from the axes' own size, which is what keeps the gutter a fixed
    number of millimetres instead of a fixed proportion.
    """
    for y_from, y_to, y_text, label, color in [
        (0.56, 0.98, 0.77, labels[0], UP_COLOR),
        (0.44, 0.02, 0.23, labels[1], DOWN_COLOR),
    ]:
        ax.annotate("", xy=(arrow_x, y_to), xytext=(arrow_x, y_from), xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.3),
                    annotation_clip=False)
        ax.text(label_x, y_text, label, transform=ax.transAxes, rotation=90,
                va="center", ha="center", fontsize=7.5, color=color, clip_on=False)


def panel_storage_attenuation(ax, attenuation: pd.DataFrame, events: pd.DataFrame,
                              duration_class: str, style: PaperStyle, *,
                              show_points: bool = False,
                              show_frequency_change: bool = False) -> None:
    """Severity absorbed by storage: the two Shapley shares and their total, annotated with the
    change in event frequency.

    Three boxes per zone — **reservoir | battery | combined** — so every quantity carries its own
    inter-model spread.  (The previous stacked bar showed a spread for the *total* only, and overlaid
    the per-chain combined values on a bar built from two separate medians: two different quantities
    in one place, whose bar-top need not equal ``median(combined)``.)

    The two shares still sum to the combined **exactly, per chain** — that is what the Shapley
    decomposition guarantees, and the notebook asserts it — it is simply stated rather than drawn.

    Pairing the attenuation with the change in event *frequency* is the point of the panel: storage
    can absorb most of the severity and still not prevent the exceedances.
    """
    whis = whis_for(style.spread)
    warm = attenuation[(attenuation.duration_class == duration_class)
                       & (attenuation.period == style.warmest_period)]
    frequency = events[events.duration_class == duration_class]

    parts = [("reservoir_attenuation_share_pct", "reservoir", "#08519C"),
             ("battery_attenuation_share_pct", "battery", "#8B5A2B"),
             ("combined_attenuation_pct", "combined", "0.62")]
    width, positions = 0.26, np.arange(len(style.zones))

    for i, zone in enumerate(style.zones):
        per_zone = warm[warm.zone == zone]
        for k, (column, _label, color) in enumerate(parts):
            position = i + (k - 1) * width
            values = per_zone[column].dropna().values
            if not len(values):
                continue
            boxes = ax.boxplot([values], positions=[position], widths=width * 0.86,
                               patch_artist=True, showfliers=False, whis=whis,
                               medianprops=dict(color="black", lw=1.2))
            for box in boxes["boxes"]:
                box.set_facecolor(color)
                box.set_alpha(0.8)
                box.set_edgecolor("0.3")
                box.set_lw(0.7)
            if show_points:
                ax.plot(np.full(len(values), position), values, "o", ms=2.2, color="0.15",
                        alpha=0.5, zorder=5)

        if show_frequency_change:
            ref = frequency[(frequency.zone == zone)
                            & (frequency.period == style.reference_period)]["freq_per_year"].median()
            hot = frequency[(frequency.zone == zone)
                            & (frequency.period == style.warmest_period)]["freq_per_year"].median()
            if ref > 0:
                pct = 100 * (hot - ref) / ref
                ax.annotate(f"{pct:+.0f}%", xy=(i, 116), ha="center", va="top", fontsize=9,
                            fontweight="bold", color=_change_color(pct))

    ax.set_xticks(positions)
    ax.set_xticklabels(style.labels(), fontsize=8, rotation=18)
    ax.set_ylabel("severity absorbed by storage (%)\nShapley shares", fontsize=9)
    ax.set_xlim(-0.5, len(style.zones) - 0.5)
    ax.set_ylim(0, 122 if show_frequency_change else 105)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=color, alpha=0.8) for _, _, color in parts],
              labels=[label for _, label, _ in parts],
              fontsize=7.5, frameon=False, loc="lower left", ncol=3)
    ax.tick_params(labelsize=8)


# ---------------------------------------------------------------------------
# Figure 1 — the mean state
# ---------------------------------------------------------------------------
def plot_mean_state_maps(
    quantities: Sequence[tuple],
    countries: gpd.GeoDataFrame,
    style: PaperStyle,
    sign_agreement: Callable,
    *,
    clip_percentile: float = 98,
    min_baseline_fraction: float = 0.02,
    unit: str = "TWh yr$^{-1}$",
    change_bins: Mapping[str, Sequence[float]] | None = None,
    absolute_change_rows: Sequence[str] = (),
    change_clabels: Mapping[str, str] | None = None,
    title: str = "Figure 1 · The mean state improves across warming levels",
) -> Figure:
    """Country maps: one row per energy variable, one column per warming level.

    ``quantities`` is a sequence of ``(row_label, field(period) -> DataArray(model, country),
    cmap, absolute_center)``.  ``absolute_center=0`` gives the first column a diverging, zero-centred
    scale (used for residual load, which changes sign between countries); ``None`` gives a sequential
    one.  Columns 2+ are the change vs the reference period, sharing one colorbar per row because
    each variable needs its own scale (heating ≈ −30 % while cooling runs into the hundreds).

    ``change_bins`` maps a row label to **fixed** bin edges for its change columns.  Without it the
    scale is continuous and auto-zoomed to ``clip_percentile`` of the row's own values, which
    misrepresents a row that carries no signal: solar moves by under 2 % in every country, so
    auto-zooming stretches pure noise across the full ramp and the map reads as structure.  Fixed
    edges say "flat" honestly, and the class containing zero is drawn in the colormap's own midpoint
    and labelled as no material change.

    ``absolute_change_rows`` names rows whose change columns are a **difference in the row's unit**
    rather than a percentage.  This is required for any quantity that changes sign between countries:
    residual load is negative wherever a country is in surplus, so dividing by it inverts the
    percentage — the United Kingdom tightens by +14.6 TWh yr⁻¹ off a −67.9 TWh yr⁻¹ base and reports
    as −18.7 %, rendering blue while it is the most tightening country in Europe.  A difference has
    no denominator, so the sign is always physical.  Such rows keep their own diverging colormap
    (so their change columns agree in direction with their column 0) and are never baseline-masked.
    """
    fig, axes = plt.subplots(len(quantities), len(style.period_names), figsize=(11.0, 15.5),
                             squeeze=False, constrained_layout=True)

    for row, (name, field, cmap, absolute_center) in enumerate(quantities):
        reference_per_chain = field(style.reference_period)          # (model, country)
        reference = reference_per_chain.median("model")

        # column 0 — absolute value at the reference warming level
        ax = axes[row, 0]
        if absolute_center is None:
            plot_country_choropleth(ax, reference.to_series().dropna(), countries,
                                    cmap=cmap, clabel=unit, legend=True)
        else:
            limit = _symmetric_limit(reference, clip_percentile)
            plot_country_choropleth(ax, reference.to_series().dropna(), countries,
                                    cmap=cmap, vcenter=absolute_center, vmin=-limit, vmax=limit,
                                    clabel=unit, extend="both", legend=True)

        # columns 1+ — the change vs the reference level
        absolute = name in absolute_change_rows
        edges = list((change_bins or {}).get(name, ())) or None

        # A difference has no denominator, so nothing needs greying; a ratio does.
        baseline_ok = None if absolute else _baseline_relevance_mask(reference,
                                                                     min_baseline_fraction)
        changes, agreements = [], []
        for period in style.period_names[1:]:
            delta = field(period) - reference_per_chain
            per_chain = delta if absolute else 100 * delta / reference_per_chain
            per_chain = per_chain.where(np.isfinite(per_chain))
            change = per_chain.median("model")
            if baseline_ok is not None:
                change = change.where(baseline_ok)
            changes.append(change)
            agreements.append(sign_agreement(per_chain, model_dim="model"))

        # An absolute-change row keeps its own diverging colormap so its change columns agree
        # in direction with its column 0 — which means its bar can run the OTHER way round
        # from the percentage rows.  `change_clabels` is how that gets said out loud.
        change_cmap = cmap if absolute else "RdBu"
        clabel = (change_clabels or {}).get(
            name, f"change ({unit})" if absolute else "% change")

        if edges is not None:
            norm, discrete_cmap = _binned_norm(edges, change_cmap)
            # Hatching is suppressed inside the class containing zero — see
            # `_hatch_disagreeing_countries`.
            neutral = max(abs(e) for e in _zero_class(edges))
            per_col_kw = dict(norm=norm, cmap=discrete_cmap)
            bar_kw = dict(mappable=plt.cm.ScalarMappable(norm=norm, cmap=discrete_cmap),
                          extend="both", ticks=edges, spacing="uniform")
        else:
            limit = max(_symmetric_limit(c, clip_percentile) for c in changes)
            data_max = max(float(np.nanmax(np.abs(c.values))) for c in changes)
            neutral = None
            per_col_kw = dict(cmap=change_cmap, vcenter=0, vmin=-limit, vmax=limit)
            bar_kw = dict(mappable=plt.cm.ScalarMappable(
                              norm=mcolors.TwoSlopeNorm(vcenter=0, vmin=-limit, vmax=limit),
                              cmap=change_cmap),
                          extend="both" if data_max > limit * 1.0001 else "neither")

        for col, (change, agreement) in enumerate(zip(changes, agreements), start=1):
            ax = axes[row, col]
            plot_country_choropleth(ax, change.to_series().dropna(), countries,
                                    legend=False, **per_col_kw)
            _hatch_disagreeing_countries(ax, agreement, change, countries, style,
                                         min_magnitude=neutral)
            set_map_aspect(ax, countries)   # the hatch layer is another geopandas draw

        mappable = bar_kw.pop("mappable")
        mappable.set_array([])
        bar = fig.colorbar(mappable, ax=[axes[row, c] for c in range(1, len(style.period_names))],
                           fraction=0.025, pad=0.01, **bar_kw)
        bar.set_label(clabel, fontsize=8)
        bar.ax.tick_params(labelsize=7)

        # row label: the map axes are switched off, so re-enable just enough to carry the ylabel
        axes[row, 0].set_ylabel(name, fontsize=11, fontweight="bold", labelpad=8)
        axes[row, 0].axis("on")
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])
        for spine in axes[row, 0].spines.values():
            spine.set_visible(False)

    for col, period in enumerate(style.period_names):
        axes[0, col].set_title(period if col == 0 else f"{period}  (% vs {style.reference_period})",
                               fontsize=12, fontweight="bold")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    return fig


# ---------------------------------------------------------------------------
# Figure 2 — the tail
# ---------------------------------------------------------------------------
def plot_tail_survival(
    residual_load_by_model: Mapping,
    period_defs: Mapping,
    zone_threshold,
    deficit_budget: pd.DataFrame,
    style: PaperStyle,
    season_months: Mapping[str, Sequence[int]],
    *,
    seasons: Sequence[str] = ("DJF", "JJA"),
    title: str = "Figure 2 · The deficit tail fattens even as the mean state improves",
) -> Figure:
    """Exceedance (survival) curves per zone and season, plus the deficit-budget panel.

    A *density* plot cannot show a change confined to the top few per cent of the distribution; on a
    log survival axis the warming levels fan out exactly where the events live.  The dashed line is
    the single absolute deficit threshold (one per zone, season-independent — a season-specific
    threshold would re-baseline each season to its own tail and erase the winter→summer migration
    this figure exists to show).

    **The seasons of a zone share one x-axis**, so their panels are directly comparable: the
    threshold sits at the same place in both, and a season whose distribution stops short of that
    line is visibly a season with no exceedance — which is why no annotation says so.
    """
    whis = whis_for(style.spread)
    fig = plt.figure(figsize=(16.5, 8.4))
    grid = gridspec.GridSpec(3, len(style.zones), height_ratios=[1, 1, 0.8],
                             hspace=0.35, wspace=0.3)

    for col, zone in enumerate(style.zones):
        threshold = float(zone_threshold.sel(zone=zone))
        # ONE grid per zone, pooled over the displayed seasons and always containing the threshold.
        x = compute_survival_x_grid(
            residual_load_by_model, period_defs, zone,
            [season_months[s] for s in seasons], style.period_names, threshold=threshold)

        for row, season in enumerate(seasons):
            ax = fig.add_subplot(grid[row, col])
            _, survival = compute_exceedance_survival(
                residual_load_by_model, period_defs, zone, season_months[season],
                style.period_names, spread=style.spread, x=x)

            ax.axvspan(threshold, x[-1], color="0.87", alpha=0.6, lw=0, zorder=0)   # the deficit tail
            for period in style.period_names:
                median, low, high = survival[period]
                ax.plot(x, median, color=style.gwl_colors[period], lw=1.7, label=period, zorder=3)
                ax.fill_between(x, low, high, color=style.gwl_colors[period], alpha=0.18, lw=0,
                                zorder=2)
            ax.axvline(threshold, color="0.2", ls="--", lw=1.1, zorder=4)
            ax.set_yscale("log")
            ax.set_ylim(2e-4, 1.05)
            ax.set_xlim(x[0], x[-1])                       # identical for every season of the zone
            ax.tick_params(labelsize=7)
            ax.grid(True, which="major", color="0.92", lw=0.6)

            # how much more often is the threshold crossed at the warmest level? (evaluated exactly
            # at the threshold, so it does not depend on the grid)
            p_ref = compute_exceedance_probability(
                residual_load_by_model, period_defs, zone, season_months[season],
                style.reference_period, threshold)
            p_warm = compute_exceedance_probability(
                residual_load_by_model, period_defs, zone, season_months[season],
                style.warmest_period, threshold)
            if p_ref > 0:
                # The threshold sits in the right half of most zones, so a label placed to its right
                # runs off the panel. Flip it to the inside.
                # Park it in the top-left of the shaded tail band. That band is >=28 % of the axis
                # in every zone, and no curve reaches P = 0.55 to the right of the threshold, so the
                # label can never collide with a curve nor run off the panel.
                ax.annotate(
                    f"×{p_warm / p_ref:.1f}", xy=(threshold, 0.55),
                    xytext=(5, 0), textcoords="offset points", ha="left", va="top",
                    fontsize=9, fontweight="bold", color=_change_color(p_warm - p_ref))

            if row == 0:
                ax.set_title(style.zone_labels[zone], fontsize=10, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"{season}\nP(RL > x)", fontsize=9, fontweight="bold")
            else:
                ax.set_yticklabels([])
            if row == len(seasons) - 1:                    # one shared x-axis: label the bottom only
                ax.set_xlabel("residual load (GWh / 3 h)", fontsize=8)
            else:
                ax.set_xticklabels([])
            if row == 0 and col == len(style.zones) - 1:
                ax.legend(fontsize=7, frameon=False, loc="lower left")

    # ── (b) the deficit budget, by timescale ───────────────────────────────────────────────
    ax_budget = fig.add_subplot(grid[2, 2:4])
    totals = deficit_budget.groupby(["model", "period"])[["sub_hours_yr", "multi_hours_yr"]] \
        .sum().reset_index()
    classes = [("sub_hours_yr", "sub-daily\n(≤ 24 h)", "sub_daily"),
               ("multi_hours_yr", "multi-day\n(> 24 h)", "multi_day")]
    width = 0.24
    changes: dict[int, tuple[float, str]] = {}

    for i, (column, label, duration_class) in enumerate(classes):
        for k, period in enumerate(style.period_names):
            position = i + (k - 1) * width
            values = totals[totals.period == period][column].values
            boxes = ax_budget.boxplot([values], positions=[position], widths=width * 0.85,
                                      patch_artist=True, showfliers=False, whis=whis,
                                      medianprops=dict(color="black", lw=1.3))
            for box in boxes["boxes"]:
                box.set_facecolor(style.gwl_colors[period])
                box.set_alpha(0.75)
                box.set_edgecolor("0.3")
                box.set_lw(0.7)
            ax_budget.plot(np.full(len(values), position), values, "o", ms=2.4,
                           color="0.15", alpha=0.55, zorder=5)      # every chain

        reference = totals[totals.period == style.reference_period][column].median()
        warmest = totals[totals.period == style.warmest_period][column].median()
        changes[i] = (100 * (warmest - reference) / reference, style.duration_palette[duration_class])

    top = totals[[c[0] for c in classes]].values.max() * 1.22
    ax_budget.set_ylim(0, top)
    for i, (change, color) in changes.items():          # both at one height, so they read as a pair
        ax_budget.annotate(f"{change:+.0f} %", xy=(i, top * 0.94), ha="center", va="top",
                           fontsize=10.5, fontweight="bold", color=color)

    ax_budget.set_xticks(range(len(classes)))
    ax_budget.set_xticklabels([c[1] for c in classes], fontsize=9)
    ax_budget.set_xlim(-0.5, len(classes) - 0.5)
    ax_budget.set_ylabel("deficit-hours per year\n(summed over zones)", fontsize=9)
    ax_budget.set_title("The sub-daily deficit burden grows; the multi-day burden does not",
                        fontsize=10, fontweight="bold")
    ax_budget.legend(handles=[plt.Line2D([], [], marker="s", ls="", color=style.gwl_colors[p],
                                         label=p) for p in style.period_names],
                     fontsize=7, frameon=False, loc="lower right")
    ax_budget.tick_params(labelsize=8)
    ax_budget.text(
        0.5, -0.38,
        "The multi-day burden is flat, not falling: the northern decline is offset by the new\n"
        "Mediterranean summer regime (Fig 4). It is the sub-daily burden that grows.",
        transform=ax_budget.transAxes, ha="center", va="top", fontsize=8, color="0.35")

    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.995)
    return fig


# ---------------------------------------------------------------------------
# Figures 3 & 4 — the two regimes (matched layouts)
# ---------------------------------------------------------------------------
def plot_sub_daily_regime(
    events: pd.DataFrame,
    attenuation: pd.DataFrame,
    diurnal_profiles: Mapping[str, list],
    evening_peak_components: pd.DataFrame,
    style: PaperStyle,
    *,
    southern_zones: Sequence[str] = ("Iberian_Peninsula", "South_East"),
    title: str = "Figure 3 · The sub-daily regime — short evening deficits surge in Southern Europe",
) -> Figure:
    """what changes → why they are short → why they grow → can flexibility stop it."""
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.6))

    panel_event_frequency(axes[0, 0], events, "sub_daily", style, "sub-daily events per year",
                          show_points=False, show_legend=False)
    axes[0, 0].set_title("a · Short evening deficits surge in the south",
                         fontsize=10, fontweight="bold")

    # (b) why they are short — the diurnal shape of residual load on stressed days
    ax_diurnal = axes[0, 1]
    for period in style.period_names:
        curves = [s for s in diurnal_profiles[period] if s.name.split("|")[1] in southern_zones]
        profiles = pd.concat(curves, axis=1)
        hours = profiles.index.values
        ax_diurnal.plot(hours, profiles.median(axis=1), color=style.gwl_colors[period], lw=2.0,
                        label=period, zorder=3)
        ax_diurnal.fill_between(hours,
                                *model_spread_bounds(profiles.values, style.spread, axis=1),
                                color=style.gwl_colors[period], alpha=0.16, lw=0, zorder=2)
    ax_diurnal.axhspan(1.0, ax_diurnal.get_ylim()[1], color="0.9", alpha=0.0)
    ax_diurnal.axhline(1.0, color="0.25", ls="--", lw=1.2, zorder=4)
    ax_diurnal.axvspan(15, 21, color="#F5A623", alpha=0.12, lw=0, zorder=0)
    ax_diurnal.annotate("evening peak\n15–21 h", xy=(18, ax_diurnal.get_ylim()[1]), xytext=(0, -6),
                        textcoords="offset points", ha="center", va="top", fontsize=8.5,
                        fontweight="bold", color="#8a5a00")
    ax_diurnal.annotate("threshold", xy=(0.5, 1.0), xytext=(2, 3), textcoords="offset points",
                        fontsize=8, color="0.25")
    ax_diurnal.set_xticks(range(0, 24, 3))
    ax_diurnal.set_xlabel("hour of day (UTC)", fontsize=9)
    ax_diurnal.set_ylabel("residual load ÷ threshold\n(on stressed days, southern zones)", fontsize=9)
    ax_diurnal.set_title("b · PV has faded by the evening demand peak", fontsize=10,
                         fontweight="bold")
    ax_diurnal.tick_params(labelsize=8)      # the warming levels are in the figure legend

    # (c) why they grow — attribution of the evening-peak change, within season
    panel_driver_attribution(
        axes[1, 0],
        lambda zone, season: attribution_change_by_model(
            evening_peak_components, zone, season,
            style.reference_period, style.warmest_period, style.drivers),
        ["DJF", "JJA"], style,
        "c · Summer demand drives the evening-peak deficit")
    _add_driver_legend(axes[1, 0], style)

    panel_storage_attenuation(axes[1, 1], attenuation, events, "sub_daily", style)
    axes[1, 1].set_title("d · Storage absorbs most of the severity — but the events still come",
                         fontsize=10, fontweight="bold")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    _add_gwl_legend(fig, style)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def plot_multi_day_regime(
    events: pd.DataFrame,
    attenuation: pd.DataFrame,
    multi_day_by_season: pd.DataFrame,
    multi_day_events: pd.DataFrame,
    style: PaperStyle,
    *,
    title: str = ("Figure 4 · The multi-day regime — winter deficits ease, except where wind fails "
                  "and where heat waves begin"),
) -> Figure:
    """Matched to Fig 3: what changes → when → why (within season) → can flexibility stop it."""
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.6))

    panel_event_frequency(axes[0, 0], events, "multi_day", style, "multi-day events per year",
                          show_points=False, show_legend=False)
    axes[0, 0].set_title("a · Long deficits ease in the north — but not everywhere",
                         fontsize=10, fontweight="bold")

    # (b) WHEN — the seasonal decomposition that exposes all three mechanisms at once
    ax_season = axes[0, 1]
    width, positions = 0.26, np.arange(len(style.zones))
    for k, period in enumerate(style.period_names):
        offset = (k - 1) * width
        bottoms = np.zeros(len(style.zones))
        for season, alpha in zip(style.season_order, SEASON_SHADES):
            values = np.array([
                multi_day_by_season[(multi_day_by_season.zone == z)
                                    & (multi_day_by_season.period == period)
                                    & (multi_day_by_season.season == season)]["md_events_yr"].median()
                for z in style.zones])
            ax_season.bar(positions + offset, values, width * 0.86, bottom=bottoms,
                          color=style.gwl_colors[period], alpha=alpha,
                          edgecolor="white", linewidth=0.4)
            bottoms += values
    ax_season.set_ylim(0, ax_season.get_ylim()[1] * 1.16)          # headroom for the legend
    ax_season.set_xticks(positions)
    ax_season.set_xticklabels(style.labels(), fontsize=8, rotation=18)
    ax_season.set_ylabel("multi-day events per year\n(stacked by season)", fontsize=9)
    ax_season.set_title("b · A new SUMMER multi-day regime in the Mediterranean",
                        fontsize=10, fontweight="bold")
    ax_season.tick_params(labelsize=8)

    # A real legend for the shading. The warming-level COLOURS are already in the figure-level
    # legend, so only the season SHADES need keying here; a neutral swatch keeps them from being
    # mistaken for a warming level.
    ax_season.legend(
        handles=[plt.Rectangle((0, 0), 1, 1, facecolor="0.35", alpha=alpha, edgecolor="white")
                 for alpha in SEASON_SHADES],
        labels=list(style.season_order), title="season (shade)",
        fontsize=7.5, title_fontsize=7.5, frameon=False, ncol=1, loc="upper left")

    # Callouts shelved: they collided with the bars. The seasonal decomposition speaks for itself,
    # and both numbers are already in the figure caption.
    # summer = [multi_day_by_season[(multi_day_by_season.zone == "South_East")
    #                               & (multi_day_by_season.period == p)
    #                               & (multi_day_by_season.season == "JJA")]["md_events_yr"].median()
    #           for p in style.period_names]
    # ax_season.annotate(
    #     f"South-East JJA:\n{summer[0]:.2f} → {summer[-1]:.2f} events/yr  "
    #     f"(~{summer[-1] / max(summer[0], 1e-9):.0f}×)",
    #     xy=(1 + width, 5.4), xytext=(28, 26), textcoords="offset points",
    #     fontsize=8.5, fontweight="bold", color=UP_COLOR,
    #     arrowprops=dict(arrowstyle="->", color=UP_COLOR, lw=1.2))
    # ax_season.annotate("British Isles:\nrises in every season\n(offshore-wind deficit)",
    #                    xy=(5, 4.9), xytext=(-96, 30), textcoords="offset points",
    #                    fontsize=8.5, fontweight="bold", color="#1B7C3D",
    #                    arrowprops=dict(arrowstyle="->", color="#1B7C3D", lw=1.2))

    # (c) WHY — attributed WITHIN season (pooling would confound the SE winter->summer shift)
    panel_driver_attribution(
        axes[1, 0],
        lambda zone, season: attribution_change_by_model(
            multi_day_events, zone, season,
            style.reference_period, style.warmest_period, style.drivers,
            column_template="{season}_{driver}", net_column="{season}_RL"),
        ["DJF", "JJA"], style,
        "c · Winter: heating demand falls · Summer: cooling demand rises")
    _add_driver_legend(axes[1, 0], style)

    panel_storage_attenuation(axes[1, 1], attenuation, events, "multi_day", style)
    axes[1, 1].set_title("d · A 4-hour battery cannot span a ~58-hour deficit",
                         fontsize=10, fontweight="bold")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    _add_gwl_legend(fig, style)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def _add_gwl_legend(fig, style: PaperStyle) -> None:
    """One warming-level legend for the whole figure.

    A per-axes legend has nowhere to go in these panels: the %-change annotations own the top, and
    the British Isles boxes hug the bottom-right.  And panels (a) and (b) of Fig 3 were carrying the
    same legend twice.
    """
    fig.legend(handles=[plt.Line2D([], [], marker="s", ls="", ms=8, color=style.gwl_colors[p],
                                   label=p) for p in style.period_names],
               loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=len(style.period_names),
               frameon=False, fontsize=9.5)


def _add_driver_legend(ax, style: PaperStyle) -> None:
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=style.driver_palette[d])
                       for d in style.drivers]
                      + [plt.Line2D([], [], marker="D", ls="", color="black")],
              labels=style.drivers + [f"net Δ ({style.spread_label})"],
              fontsize=6.8, frameon=False, ncol=4, loc="lower left")


# ---------------------------------------------------------------------------
# Supplementary figures
# ---------------------------------------------------------------------------
def plot_threshold_calibration(seasonal_thresholds: Mapping, style: PaperStyle) -> Figure:
    """S6 — the season-specific CERRA p95 levels (reference only; events use the annual level)."""
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    table = pd.DataFrame(
        {season: [float(seasonal_thresholds[str(z)][season]) for z in style.zones]
         for season in style.season_order},
        index=style.labels())
    sns.heatmap(table, annot=True, fmt=".0f", cmap="YlOrRd", ax=ax,
                cbar_kws={"label": "p95 threshold (GWh / 3 h)"})
    ax.set_title("S6 · Season-specific CERRA p95 deficit thresholds", fontsize=11, fontweight="bold")
    return fig


def plot_zone_season_change(change_by_class: Mapping[str, Mapping[str, tuple]],
                            style: PaperStyle) -> Figure:
    """S7 — zone × season % change per duration class.

    ``change_by_class[duration_class][period]`` is ``(median, low, high, n_agree)``, each a
    (zone, season) array — see
    :func:`~cordex2energy_analysis.extremes_analysis.compute_zone_season_change`.
    """
    classes = list(change_by_class)
    targets = style.period_names[1:]
    fig, axes = plt.subplots(len(classes), len(targets), figsize=(13, 8), squeeze=False)

    for row, duration_class in enumerate(classes):
        for col, target in enumerate(targets):
            median, low, high, n_agree = change_by_class[duration_class][target]
            ax = axes[row, col]
            limit = np.nanmax(np.abs(median)) if np.isfinite(median).any() else 1
            sns.heatmap(median, ax=ax, cmap="RdBu_r", center=0, vmin=-limit, vmax=limit,
                        xticklabels=list(style.season_order), yticklabels=style.labels(),
                        cbar_kws={"label": "% change in event count"},
                        linewidths=0.6, linecolor="white")
            for i in range(median.shape[0]):
                for j in range(median.shape[1]):
                    if not np.isfinite(median[i, j]):
                        # No events at the ABSOLUTE deficit level here. That is a result (this zone
                        # has no deficit problem in this season), not a gap — and it is exactly what
                        # a season-specific threshold would have hidden.
                        ax.text(j + 0.5, i + 0.5, "no\nevents", ha="center", va="center",
                                fontsize=7, style="italic", color="0.45")
                        continue
                    ax.text(j + 0.5, i + 0.38, f"{median[i, j]:+.0f}%", ha="center", va="center",
                            fontsize=8, fontweight="bold")
                    ax.text(j + 0.5, i + 0.68, f"[{low[i, j]:+.0f}, {high[i, j]:+.0f}]",
                            ha="center", va="center", fontsize=6, color="0.3")
                    if n_agree[i, j] < style.robustness_threshold:
                        ax.plot(j + 0.5, i + 0.5, marker="x", ms=17, mew=2.0, color="0.2",
                                alpha=0.55)
            ax.set_title(f"{duration_class} — {target} vs {style.reference_period}",
                         fontsize=10, fontweight="bold")
            ax.tick_params(labelsize=8)

    fig.suptitle(
        f"S7 · Zone × season change in event count  "
        f"(× = < {style.robustness_threshold}/{style.n_models} chains agree on the sign)",
        fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def plot_significance(significance: pd.DataFrame, duration_class: str, style: PaperStyle) -> Figure:
    """S8 — Wilcoxon signed-rank p-values and ensemble agreement, per zone × season."""
    comparisons = significance.comparison.unique()
    fig, axes = plt.subplots(len(comparisons), 2, figsize=(11.5, 3.4 * len(comparisons)),
                             squeeze=False)
    for row, comparison in enumerate(comparisons):
        data = significance[(significance.duration_class == duration_class)
                            & (significance.comparison == comparison)]
        agreement = data.pivot(index="zone", columns="season", values="agree") \
            .reindex(style.zones)[list(style.season_order)]
        pvalues = data.pivot(index="zone", columns="season", values="p") \
            .reindex(style.zones)[list(style.season_order)]

        sns.heatmap(agreement, ax=axes[row, 0], cmap="RdYlGn", vmin=0.5, vmax=1.0, annot=True,
                    fmt=".2f", yticklabels=style.labels(),
                    cbar_kws={"label": "fraction of chains agreeing"},
                    linewidths=0.6, linecolor="white")
        axes[row, 0].set_title(f"{duration_class} · {comparison} — ensemble agreement",
                               fontsize=10, fontweight="bold")
        sns.heatmap(pvalues, ax=axes[row, 1], cmap="viridis_r", vmin=0, vmax=0.2, annot=True,
                    fmt=".3f", yticklabels=style.labels(), cbar_kws={"label": "Wilcoxon p"},
                    linewidths=0.6, linecolor="white")
        axes[row, 1].set_title(f"{duration_class} · {comparison} — Wilcoxon p (bold border: p < 0.05)",
                               fontsize=10, fontweight="bold")
        for i in range(pvalues.shape[0]):
            for j in range(pvalues.shape[1]):
                if np.isfinite(pvalues.values[i, j]) and pvalues.values[i, j] < 0.05:
                    axes[row, 1].add_patch(plt.Rectangle((j, i), 1, 1, fill=False, ec="red", lw=2.2))
        for ax in axes[row]:
            ax.tick_params(labelsize=8)

    fig.suptitle(f"S8 · Robustness — {duration_class}", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def plot_level_vs_persistence(multi_day_events: pd.DataFrame, style: PaperStyle) -> Figure:
    """S11 — the multi-day decline is *fewer* events, not *shorter* ones."""
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
    positions = np.arange(len(style.zones))

    for k, (column, ylabel, ax) in enumerate(
            [("md_events_yr", "event count (per year)", axes[0]),
             ("md_meandur", "mean event duration (h)", axes[1])]):
        for j, period in enumerate(style.period_names):
            values = [multi_day_events[(multi_day_events.zone == z)
                                       & (multi_day_events.period == period)][column].median()
                      for z in style.zones]
            ax.bar(positions + (j - 1) * 0.27, values, 0.25, color=style.gwl_colors[period],
                   label=period if k == 0 else None, edgecolor="white", lw=0.5)
            for i, zone in enumerate(style.zones):
                per_chain = multi_day_events[(multi_day_events.zone == zone)
                                             & (multi_day_events.period == period)][column] \
                    .dropna().values
                ax.plot(np.full(len(per_chain), i + (j - 1) * 0.27), per_chain, "o", ms=2.2,
                        color="0.15", alpha=0.5)
        ax.set_xticks(positions)
        ax.set_xticklabels(style.labels(), fontsize=8, rotation=18)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.tick_params(labelsize=8)

    axes[0].set_title("a · Event COUNT changes strongly (a level effect)",
                      fontsize=10, fontweight="bold")
    axes[0].legend(fontsize=8, frameon=False)
    axes[1].set_title("b · Mean DURATION barely moves (no persistence effect)",
                      fontsize=10, fontweight="bold")
    reference = multi_day_events[multi_day_events.period == style.reference_period]["md_meandur"].median()
    warmest = multi_day_events[multi_day_events.period == style.warmest_period]["md_meandur"].median()
    axes[1].text(0.02, 0.96, f"pooled mean duration: {reference:.0f} h → {warmest:.0f} h",
                 transform=axes[1].transAxes, va="top", fontsize=9, color="0.3",
                 bbox=dict(fc="white", ec="0.85", pad=2.5))

    fig.suptitle("S11 · The multi-day decline is a level effect, not a persistence effect",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def plot_cooccurrence(
    cooccurrence: pd.DataFrame,
    period_defs: Mapping,
    style: PaperStyle,
    duration_class: str,
    *,
    window_steps: int,
    time_step_h: float = 3.0,
) -> list[Figure]:
    """S13 — zone-pair co-occurrence, returned as an explicit list of figures.

    :func:`~cordex2energy_analysis.plotting_extremes.plot_zone_cooccurrence_heatmaps` draws straight
    to pyplot and returns nothing, creating *several* figures (absolute levels, then the change vs
    the baseline).  Relying on ``plt.savefig`` to pick "the current figure" afterwards silently saves
    only one of them — and, if the plotter has already called ``plt.show()``, sometimes an empty one.
    Close first, then collect every figure it created.
    """
    from .plotting_extremes import plot_zone_cooccurrence_heatmaps

    plt.close("all")
    plot_zone_cooccurrence_heatmaps(
        cooccurrence, list(style.zones), period_defs,
        baseline_period=style.reference_period,
        zone_short=dict(style.zone_labels),
        window_steps=window_steps, time_step_h=time_step_h,
        stipple_agreement=True, robust_n=style.robustness_threshold, spread=style.spread,
    )
    figures = [plt.figure(n) for n in plt.get_fignums()]
    for figure in figures:
        figure.suptitle(f"S13 · Zone-pair co-occurrence — {duration_class}",
                        fontsize=12, fontweight="bold")
    return figures


def plot_threshold_sensitivity(sensitivity: pd.DataFrame, style: PaperStyle) -> Figure:
    """S14 — annual vs season-specific p95: the conclusions hold, but the baseline does not.

    ``sensitivity`` carries one row per (model, period, zone, season, scheme) with ``sub_daily`` and
    ``multi_day`` event rates — see
    :func:`~cordex2energy_analysis.extremes_analysis.compute_threshold_sensitivity`.
    """
    def change_and_baseline(duration_class: str, scheme: str, zone: str, season: str):
        rows = sensitivity[(sensitivity.scheme == scheme) & (sensitivity.zone == zone)
                           & (sensitivity.season == season)]
        ref = rows[rows.period == style.reference_period].set_index("model")[duration_class]
        warm = rows[rows.period == style.warmest_period].set_index("model")[duration_class]
        base, hot = ref.median(), warm.median()
        return (100 * (hot - base) / base) if base > 0 else np.nan, base

    classes = ["sub_daily", "multi_day"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.6), gridspec_kw={"width_ratios": [1, 1, 1.15]})

    for row, duration_class in enumerate(classes):
        matrices = {}
        for col, scheme in enumerate(["annual", "seasonal"]):
            matrix = np.full((len(style.zones), len(style.season_order)), np.nan)
            for i, zone in enumerate(style.zones):
                for j, season in enumerate(style.season_order):
                    matrix[i, j] = change_and_baseline(duration_class, scheme, zone, season)[0]
            matrices[scheme] = matrix
            limit = (np.nanmax(np.abs(matrix[np.isfinite(matrix)]))
                     if np.isfinite(matrix).any() else 1)
            sns.heatmap(matrix, ax=axes[row, col], cmap="RdBu_r", center=0, vmin=-limit, vmax=limit,
                        annot=True, fmt=".0f", xticklabels=list(style.season_order),
                        yticklabels=style.labels(), cbar_kws={"label": "% change"},
                        linewidths=0.6, linecolor="white", annot_kws={"fontsize": 8})
            axes[row, col].set_title(f"{duration_class} — {scheme} threshold",
                                     fontsize=10, fontweight="bold")
            axes[row, col].tick_params(labelsize=8)

        # right column: the reference-period events MANUFACTURED by re-baselining
        inflation = np.zeros((len(style.zones), len(style.season_order)))
        for i, zone in enumerate(style.zones):
            for j, season in enumerate(style.season_order):
                inflation[i, j] = (change_and_baseline(duration_class, "seasonal", zone, season)[1]
                                   - change_and_baseline(duration_class, "annual", zone, season)[1])
        sns.heatmap(inflation, ax=axes[row, 2], cmap="Purples", annot=True, fmt=".1f",
                    xticklabels=list(style.season_order), yticklabels=style.labels(),
                    cbar_kws={"label": "extra events/yr at +1.5 °C"},
                    linewidths=0.6, linecolor="white", annot_kws={"fontsize": 8})
        axes[row, 2].set_title(
            f"{duration_class} — events MANUFACTURED at the\nreference by the seasonal threshold",
            fontsize=10, fontweight="bold")
        axes[row, 2].tick_params(labelsize=8)

        populated = np.isfinite(matrices["annual"]) & np.isfinite(matrices["seasonal"])
        agree = int((np.sign(matrices["annual"][populated])
                     == np.sign(matrices["seasonal"][populated])).sum())
        print(f"{duration_class}: sign agrees in {agree}/{int(populated.sum())} populated cells")

    fig.suptitle("S14 · Threshold sensitivity — the conclusions hold, but a season-specific p95 "
                 "re-baselines each season to its own tail", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig
