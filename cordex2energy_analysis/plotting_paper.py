"""Shared figure style, country maps and change-class helpers of the publication figures.

The figures share a dozen presentation settings (zone order, warming-level colours, driver
palette, robustness threshold, spread convention), bundled into one :class:`PaperStyle`.

Encoding conventions applied by every figure:

* **Hatching means "not robust"**: fewer than ``style.robustness_threshold`` chains agree on the
  sign of the change.
* **No error bars on stacked-bar segments.** A segment sits on the cumulative sum of the ones
  below it and the component errors are correlated, so the inter-model spread is shown on the
  net marker only, which has a fixed zero baseline.
* **Boxes carry the median and the inter-model spread** (``style.spread``); every label
  describing the spread derives from ``style.spread_label``.
* **Maps are drawn in ETRS89-LAEA (EPSG:3035) at equal aspect** when the geometry is loaded with
  ``crs=MAP_CRS``, so a country's drawn area is its true area. Lon/lat geometry is drawn with
  ``"auto"`` aspect; :func:`map_extent` and :func:`set_map_aspect` choose from the geometry's CRS.
* **A percentage change is only drawn where the baseline is meaningful**
  (:func:`_baseline_relevance_mask`): a ratio taken off a near-zero base would dominate the
  colour scale.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analysis import SPREAD_LABEL, spread_qpair

__all__ = [
    "UP_COLOR",
    "DOWN_COLOR",
    "PaperStyle",
    "MAP_EXTENT",
    "MAP_CRS",
    "MAP_EXTENT_LAEA",
    "is_projected",
    "map_extent",
    "set_map_aspect",
    "load_country_geometry",
    "plot_country_choropleth",
    "NEUTRAL_CLASS_COLOR",
]

#: Colour of a rising and of a falling quantity, used for every annotated change.
UP_COLOR = "#B2182B"
DOWN_COLOR = "#2166AC"
#: Colour of the "no material change" class on a binned change scale: the RdBu midpoint, the colour
#: near-zero takes on the continuous scale, and distinct from the ``lightgrey`` of countries
#: without data.
NEUTRAL_CLASS_COLOR = "#f7f7f7"


# ---------------------------------------------------------------------------
# Shared context
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PaperStyle:
    """Presentation settings shared by every figure.

    Attributes
    ----------
    zones : sequence of str
        Balancing zones, in display order.
    zone_labels : mapping
        ``{zone: short display label}``.
    period_names : sequence of str
        Warming-level labels, coolest first (e.g. ``["+1.5 °C", "+2.0 °C", "+3.0 °C"]``).
    gwl_colors : mapping
        ``{warming level: colour}``.
    driver_palette : mapping
        ``{driver: colour}``, in the stacking order of the attribution bars.
    duration_palette : mapping
        ``{duration class: colour}``.
    season_order : sequence of str
        Seasons, in display order.
    n_models : int
        Number of chains.
    robustness_threshold : int
        Chains that must agree on the sign of a change for it to count as robust.
    spread : {"iqr", "minmax", "tukey"}
        Inter-model spread convention (see :func:`~cordex2energy_analysis.analysis.spread_qpair`).
    """

    zones: Sequence[str]
    zone_labels: Mapping[str, str]
    period_names: Sequence[str]
    gwl_colors: Mapping[str, str]
    driver_palette: Mapping[str, str]
    duration_palette: Mapping[str, str]
    season_order: Sequence[str]
    n_models: int
    robustness_threshold: int
    spread: str = "iqr"

    @property
    def reference_period(self) -> str:
        """The first (reference) warming level."""
        return self.period_names[0]

    @property
    def warmest_period(self) -> str:
        """The last (warmest) warming level."""
        return self.period_names[-1]

    @property
    def drivers(self) -> list[str]:
        """Driver names, in stacking order."""
        return list(self.driver_palette)

    @property
    def spread_percentiles(self) -> tuple[float, float]:
        """(low, high) box percentiles of the spread convention, e.g. ``(25.0, 75.0)``."""
        return spread_qpair(self.spread)

    @property
    def spread_label(self) -> str:
        """Description of the spread convention for figure labels."""
        return SPREAD_LABEL[self.spread]

    def labels(self) -> list[str]:
        """Display labels of :attr:`zones`, in order.

        Returns
        -------
        list of str
        """
        return [self.zone_labels[zone] for zone in self.zones]


def _change_color(change: float) -> str:
    """Colour of a change: :data:`UP_COLOR` if positive, else :data:`DOWN_COLOR`.

    Parameters
    ----------
    change : float
        Signed change.

    Returns
    -------
    str
    """
    return UP_COLOR if change > 0 else DOWN_COLOR


# ---------------------------------------------------------------------------
# Maps
# ---------------------------------------------------------------------------
#: Lon/lat map window: (lon_min, lon_max, lat_min, lat_max).
MAP_EXTENT = (-12, 40, 33, 72)

#: Projection of the maps: ETRS89 Lambert azimuthal equal-area, as used by Eurostat, the EEA and
#: INSPIRE for pan-European statistical maps.
MAP_CRS = "EPSG:3035"
#: Map window in :data:`MAP_CRS`, in metres: (x0, x1, y0, y1). It covers the study's 37 countries
#: (x 2636-6501 km, y 1430-5423 km) with sea margins west of Portugal and south of Crete, and stays
#: inside the climate grid's footprint, so gridded maps fill it too.
MAP_EXTENT_LAEA = (2480e3, 6800e3, 1300e3, 5470e3)
#: Lon/lat window (lon_min, lon_max, lat_min, lat_max) the geometry is clipped to before
#: projecting: an azimuthal projection diverges toward its antipode, so far-eastern Russia and
#: overseas territories are cut first.
_PROJECTION_WINDOW = (-30, 60, 25, 76)


def is_projected(countries) -> bool:
    """Whether a geometry is in a projected CRS.

    Parameters
    ----------
    countries : geopandas.GeoDataFrame
        Country polygons.

    Returns
    -------
    bool
    """
    return countries.crs is not None and countries.crs.is_projected


def map_extent(countries):
    """Map window in the geometry's own coordinates.

    Parameters
    ----------
    countries : geopandas.GeoDataFrame
        Country polygons.

    Returns
    -------
    tuple of float
        ``(x0, x1, y0, y1)``: :data:`MAP_EXTENT_LAEA` for projected geometry, else
        :data:`MAP_EXTENT`.
    """
    return MAP_EXTENT_LAEA if is_projected(countries) else MAP_EXTENT


def set_map_aspect(ax, countries) -> None:
    """Set the map aspect: equal for projected geometry, ``"auto"`` for lon/lat.

    Call after every geopandas draw, which resets the aspect to ``"equal"``.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Map axes.
    countries : geopandas.GeoDataFrame
        The geometry drawn.
    """
    ax.set_aspect("equal" if is_projected(countries) else "auto")


def load_country_geometry(natural_earth_geojson, crs=None) -> gpd.GeoDataFrame:
    """Country polygons for the choropleths, keyed by the ISO2 code the energy data uses.

    Parameters
    ----------
    natural_earth_geojson : path-like
        Natural Earth admin-0 countries.
    crs : str, optional
        Target CRS, e.g. :data:`MAP_CRS`. The geometry is then clipped to
        ``_PROJECTION_WINDOW`` and projected; by default it stays in lon/lat.

    Returns
    -------
    geopandas.GeoDataFrame
        Columns ``ISO2`` and ``geometry``, one row per country intersecting :data:`MAP_EXTENT`.

    Notes
    -----
    Natural Earth rather than NUTS, because NUTS omits the UK and non-NUTS states; a country
    without data is then greyed rather than missing. ``ISO_A2`` is ``"-99"`` for several
    countries (France, Norway, Kosovo) and falls back to ``ISO_A2_EH``; Natural Earth's ``"GB"``
    becomes ``"UK"`` as in the energy data; Northern Cyprus (``ADM0_A3 == "CYN"``, ``"-99"`` in
    both code fields) is merged into ``"CY"``.
    """
    natural_earth = gpd.read_file(natural_earth_geojson)
    iso2 = (natural_earth["ISO_A2"]
            .where(natural_earth["ISO_A2"] != "-99", natural_earth["ISO_A2_EH"])
            .replace({"GB": "UK"}))
    iso2 = iso2.where(natural_earth["ADM0_A3"] != "CYN", "CY")
    countries = (
        natural_earth.assign(ISO2=iso2)[["ISO2", "geometry"]]
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
    colorbar_label: str = "",
    extend: str = "neither",
    legend: bool = True,
    edgecolor: str = "white",
    linewidth: float = 0.3,
):
    """Draw one country choropleth; countries without a value are greyed, not dropped.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Map axes.
    values_by_iso2 : pandas.Series
        Value per country, indexed by ISO2.
    countries : geopandas.GeoDataFrame
        Country polygons with an ``ISO2`` column.
    cmap : str or matplotlib.colors.Colormap
        Colour map.
    vcenter : float, optional
        Centre of a two-slope (diverging) normalisation.
    vmin, vmax : float, optional
        Colour limits; with ``vcenter`` they default to the data range.
    norm : matplotlib.colors.Normalize, optional
        Takes precedence over ``vcenter``, ``vmin`` and ``vmax``; used to pass a
        :class:`~matplotlib.colors.BoundaryNorm` for a binned scale.
    colorbar_label : str
        Colorbar label.
    extend : str
        Colorbar extension (``"neither"``, ``"min"``, ``"max"``, ``"both"``).
    legend : bool
        Whether to draw the colorbar.
    edgecolor : str
        Country border colour. A darker border is needed on a binned scale whose neutral class
        is near-white (:data:`NEUTRAL_CLASS_COLOR`).
    linewidth : float
        Country border width.

    Returns
    -------
    matplotlib.axes.Axes
        ``ax``.
    """
    values = values_by_iso2.rename("val").rename_axis("ISO2").reset_index()
    merged = countries.merge(values, on="ISO2", how="left")

    color_kwargs = dict(vmin=vmin, vmax=vmax)
    if norm is not None:
        color_kwargs = dict(norm=norm)
    elif vcenter is not None:
        low = vmin if vmin is not None else float(np.nanmin(merged["val"]))
        high = vmax if vmax is not None else float(np.nanmax(merged["val"]))
        color_kwargs = dict(norm=mcolors.TwoSlopeNorm(vcenter=vcenter,
                                                      vmin=min(low, vcenter - 1e-6),
                                                      vmax=max(high, vcenter + 1e-6)))
    merged.plot(column="val", cmap=cmap, legend=legend, ax=ax,
                legend_kwds={"label": colorbar_label, "shrink": 0.72, "extend": extend},
                missing_kwds={"color": "lightgrey"},
                edgecolor=edgecolor, linewidth=linewidth, **color_kwargs)
    x0, x1, y0, y1 = map_extent(countries)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_axis_off()
    set_map_aspect(ax, countries)   # geopandas resets the aspect on every draw
    return ax


def _hatch_disagreeing_countries(ax, agreement, field, countries, style: PaperStyle,
                                 min_magnitude: float | None = None) -> None:
    """Hatch countries where fewer than ``style.robustness_threshold`` chains agree on the sign.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Map axes.
    agreement : xarray.DataArray
        Chains agreeing on the sign, on ``country`` (ISO2).
    field : xarray.DataArray
        The change drawn, aligned with ``agreement``; NaN countries are not hatched.
    countries : geopandas.GeoDataFrame
        Country polygons with an ``ISO2`` column.
    style : PaperStyle
        Supplies the robustness threshold.
    min_magnitude : float, optional
        No hatch where ``|field|`` is below this: sign agreement on a negligible change is
        close to random and would mark a flat country as uncertain.
    """
    iso_codes = [
        str(country)
        for country, n_agree, value in zip(agreement["country"].values, agreement.values, field.values)
        if n_agree < style.robustness_threshold and np.isfinite(value)
        and (min_magnitude is None or abs(value) >= min_magnitude)
    ]
    if iso_codes:
        countries[countries["ISO2"].isin(iso_codes)].plot(
            ax=ax, facecolor="none", edgecolor="black", hatch="///", linewidth=0.0, zorder=3)


def _symmetric_limit(values, clip_percentile: float) -> float:
    """Robust symmetric colour limit; the colorbar's extend arrows cover what lies beyond.

    Parameters
    ----------
    values : xarray.DataArray
        Field to scale.
    clip_percentile : float
        Percentile of ``|values|`` used as the limit.

    Returns
    -------
    float
        At least ``1e-6``; ``1.0`` if no value is finite.
    """
    finite = np.abs(values.values[np.isfinite(values.values)])
    return max(float(np.nanpercentile(finite, clip_percentile)), 1e-6) if finite.size else 1.0


def _zero_class(edges: Sequence[float]) -> tuple[float, float]:
    """Edges of the bin containing zero, the "no material change" class.

    Parameters
    ----------
    edges : sequence of float
        Increasing bin edges.

    Returns
    -------
    tuple of float
        ``(low, high)``.

    Raises
    ------
    ValueError
        If the edges do not straddle zero.
    """
    index = int(np.searchsorted(edges, 0.0, side="right")) - 1
    if not 0 <= index < len(edges) - 1:
        raise ValueError(f"change bin edges must straddle zero; got {list(edges)}")
    return float(edges[index]), float(edges[index + 1])


def _binned_norm(edges: Sequence[float], cmap_name: str, *, by: str = "midpoint"):
    """Discrete normalisation and colour map for a set of change-bin edges.

    Parameters
    ----------
    edges : sequence of float
        Increasing bin edges, straddling zero.
    cmap_name : str
        Diverging colour map to sample.
    by : {"midpoint", "rank"}
        ``"midpoint"`` samples each bin's colour at its midpoint on a scale symmetric about
        zero, so an asymmetric edge set still maps magnitude to intensity. ``"rank"`` samples by
        the bin's ordinal distance from the zero class, both sides normalised by the longer
        side, so a geometric edge set spanning orders of magnitude keeps every class a distinct
        step.

    Returns
    -------
    norm : matplotlib.colors.BoundaryNorm
    cmap : matplotlib.colors.ListedColormap
        One colour per bin; the bin containing zero is :data:`NEUTRAL_CLASS_COLOR`, and the
        under/over colours (the colour map's ends) carry the extend arrows.
    """
    edges = [float(edge) for edge in edges]
    base = plt.get_cmap(cmap_name)
    midpoints = [0.5 * (low + high) for low, high in zip(edges[:-1], edges[1:])]
    if by == "rank":
        zero = next(i for i, (low, high) in enumerate(zip(edges[:-1], edges[1:]))
                    if low <= 0.0 <= high)
        depth = max(zero, len(midpoints) - 1 - zero) or 1
        colors = [base(0.5 * (1.0 + (i - zero) / depth)) for i in range(len(midpoints))]
    else:
        span = max(abs(edges[0]), abs(edges[-1])) or 1.0
        colors = [base(0.5 * (1.0 + np.clip(midpoint / span, -1.0, 1.0))) for midpoint in midpoints]

    low, high = _zero_class(edges)
    colors[edges.index(low)] = mcolors.to_rgba(NEUTRAL_CLASS_COLOR)

    discrete = mcolors.ListedColormap(colors)
    discrete.set_under(base(0.0))
    discrete.set_over(base(1.0))
    return mcolors.BoundaryNorm(edges, discrete.N), discrete


def _baseline_relevance_mask(reference, min_baseline_fraction: float):
    """Where the baseline is large enough for a percentage change to be meaningful.

    Parameters
    ----------
    reference : xarray.DataArray
        Reference-level field.
    min_baseline_fraction : float
        Minimum ``|reference|`` as a fraction of the 90th percentile of the finite ``|reference|``.

    Returns
    -------
    xarray.DataArray or None
        Boolean mask, True where the baseline passes; ``None`` if no value is finite.

    Notes
    -----
    A country whose baseline is a negligible fraction of the field's typical magnitude yields a
    very large ratio that would dominate the colour scale; masked countries are greyed instead.
    """
    magnitude = np.abs(reference.values)
    magnitude = magnitude[np.isfinite(magnitude)]
    if not magnitude.size:
        return None
    floor = min_baseline_fraction * float(np.nanpercentile(magnitude, 90))
    return np.abs(reference) >= floor


def _annotate_sign_convention(ax, *, arrow_x: float = -0.115, label_x: float = -0.155,
                              labels=("deepens the deficit", "eases the deficit")) -> None:
    """Two arrows left of the y-axis stating what the sign of the change means.

    Drawn outside the axes with ``clip_on=False``; saving with ``bbox_inches="tight"`` expands
    the canvas to include them.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to annotate.
    arrow_x, label_x : float
        Horizontal positions of the arrows and of their labels, in axes fractions. Callers
        drawing at several widths pass fractions computed from the axes' size to keep a fixed
        gutter in millimetres.
    labels : tuple of str
        Labels of the upward (positive) and downward (negative) arrows.
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

