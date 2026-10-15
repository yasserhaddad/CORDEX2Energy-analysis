"""Publication figure style, panel builders and figure composers.

Every figure is built at its final print width (180 mm) rather than scaled down later, so the
type sizes chosen here are the sizes printed. The module provides

* the style: :data:`FS` type sizes, :func:`publication_rc`, colour constants;
* layout helpers: :func:`panel_letters`, :func:`legend_below`, :func:`align_colorbar_labels`,
  :func:`fit_width`;
* :func:`save`, which writes ``.pdf`` and ``.png`` and refuses a figure whose tight bounding box
  is wider than 180 mm or that carries type outside :data:`FONT_RANGE`;
* panel builders (``panel_*``) that draw into a given axes, and figure composers that return a
  :class:`~matplotlib.figure.Figure`.

Conventions shared by the builders: the ensemble statistic is the median across chains, the
spread is the inter-chain interquartile range unless stated otherwise, a change is warmest minus
reference warming level paired within each chain, and a value is hatched where fewer than
``style.robustness_threshold`` chains agree on the sign of the change. Composers set no figure
title; panel titles are kept.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Mapping, Sequence

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
from matplotlib import gridspec
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from .analysis import model_spread_bounds
from .plotting_paper import (
    PaperStyle,
    _annotate_sign_convention,
    _baseline_relevance_mask,
    _binned_norm,
    _change_color,
    _hatch_disagreeing_countries,
    _symmetric_limit,
    _zero_class,
    MAP_EXTENT_LAEA,
    is_projected,
    map_extent,
    plot_country_choropleth,
    set_map_aspect,
)

__all__ = [
    "MM",
    "WIDTH",
    "FONT_RANGE",
    "FS",
    "SEASON_NAME",
    "HYDRO_CMAP",
    "QUANTITY_CMAP",
    "SIGN_LABELS",
    "GWL_TITLE",
    "season_label",
    "CBAR_PAD",
    "CBAR_TICKS_SEQUENTIAL",
    "CBAR_TICKS_DIVERGING",
    "STORAGE_PARTS",
    "SCENARIO_COLORS",
    "publication_rc",
    "blend",
    "LETTER_INSIDE_PT",
    "LETTER_INSIDE_X_PT",
    "panel_letter",
    "panel_letters",
    "align_colorbar_labels",
    "legend_below",
    "fit_width",
    "save",
    "MapRow",
    "sign_agreement",
    "LAEA_RES_M",
    "gwl_maps",
    "mean_state_maps",
    "survival_axes",
    "paired_change_pct",
    "panel_zone_duration_hours",
    "tail_exceedance_figure",
    "SEASON_COUNT_BINS_SUB_DAILY",
    "SEASON_COUNT_BINS_MULTI_DAY",
    "season_contribution_by_model",
    "panel_season_contribution",
    "panel_event_frequency",
    "panel_hour_of_day",
    "panel_storage_stack",
    "panel_driver_attribution",
    "regime_figure",
    "panel_zone_map",
    "panel_southern_survival",
    "zone_tail_duration_figure",
    "LEVEL_ALPHA",
    "panel_scenario_bars",
    "level_handles",
    "gwl_handles",
    "storage_handles",
    "driver_handles",
    "DRIVING_GCM_ORDER",
    "DRIVING_GCM_COLORS",
    "RCM_MARKERS",
    "panel_gcm_gmst",
    "gcm_warming_figure",
    "model_characterisation_figure",
    "CATEGORY_COLOR",
    "QUANTITY_CATEGORY",
    "QUANTITY_COLOR",
    "QUANTITY_MEDIAN",
    "EUROPE_EXTENSIONS",
    "europe_change_table",
    "EUROPE_SUMMARY_COLUMNS",
    "europe_change_summary",
    "markdown_table",
    "europe_change_panel",
    "europe_change_handles",
    "europe_change_figure",
    "ZONE_COLORS",
    "TECH_COLORS",
]

#: Inches per millimetre.
MM = 1 / 25.4
#: Maximum (double-column) figure width, in inches.
WIDTH = 180 * MM
#: Permitted type sizes (pt), checked by :func:`save`.
FONT_RANGE = (7.0, 8.0)

#: Type sizes (pt) by role: tick labels, legends and annotations; axis labels; titles and panel
#: letters. Every size in the module is taken from here.
FS = {"tick": 7.0, "label": 7.5, "title": 8.0}

#: Display names of the season keys. The acronyms remain the data keys and are translated only
#: when drawn.
SEASON_NAME = {"DJF": "Winter", "MAM": "Spring", "JJA": "Summer", "SON": "Autumn"}


#: Reference-level colour ramps of the map rows. They avoid the red-blue axis, which is reserved
#: for the change columns (``RdBu_r``).
HYDRO_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "hydro_teal", ["#f2f9f7", "#a6dbd0", "#5ab4ac", "#1f8279", "#01524a"])
QUANTITY_CMAP = {"Hydro": HYDRO_CMAP, "Heating demand": "Purples",
                 "Cooling demand": "YlOrBr", "Residual load": "BrBG_r"}

#: Labels of the sign-convention arrows beside the driver-attribution panels.
SIGN_LABELS = ("deepens", "eases")

#: Title of every warming-level legend.
GWL_TITLE = "Global warming level"


def season_label(season) -> str:
    """Return the display name of a season key.

    Parameters
    ----------
    season : object
        A season key such as ``"DJF"``; any other value is returned as its string.

    Returns
    -------
    str
        ``SEASON_NAME[season]``, or ``str(season)`` if it is not a season key.
    """
    return SEASON_NAME.get(str(season), str(season))


#: Gap between a map and its colorbar, as a fraction of the parent axes width (matplotlib's
#: default). Shared by the reference and the change colorbars.
CBAR_PAD = 0.05
#: Number of ticks inside a reference (column 0) colorbar: 4 on a sequential scale, 5 on a
#: diverging one so that zero is marked. See :func:`_set_reference_ticks`.
CBAR_TICKS_SEQUENTIAL = 4
CBAR_TICKS_DIVERGING = 5

#: Storage components as ``(column, label, colour)``, in stacking order.
STORAGE_PARTS = (("reservoir_attenuation_share_pct", "reservoir", "#08519C"),
                 ("battery_attenuation_share_pct", "battery", "#8B5A2B"))

#: Colours of the supply build-out scenarios. Distinct from the warming-level colours.
SCENARIO_COLORS = ("#117733", "#882255", "#44AA99")

#: Base style of every publication figure: seaborn's "whitegrid" written out, so the figures do
#: not depend on seaborn. No tick marks on either axis; ``_RC`` then sets sizes and colours.
_BASE_STYLE = {
    "figure.facecolor": "white", "axes.labelcolor": ".15", "xtick.direction": "out",
    "ytick.direction": "out", "xtick.color": ".15", "ytick.color": ".15", "axes.axisbelow": True,
    "grid.linestyle": "-", "text.color": ".15", "lines.solid_capstyle": "round",
    "patch.edgecolor": "w", "patch.force_edgecolor": True,
    "xtick.top": False, "ytick.right": False, "axes.grid": True, "axes.facecolor": "white",
    "axes.edgecolor": ".8", "grid.color": ".93", "axes.spines.left": True,
    "axes.spines.bottom": True, "axes.spines.right": True, "axes.spines.top": True,
    "xtick.bottom": False, "ytick.left": False,
}

_RC = {
    "font.family": "sans-serif",
    # DejaVu Sans: always present with matplotlib, and it carries Greek, the superscript minus and
    # the multiplication sign.
    "font.sans-serif": ["DejaVu Sans", "Nimbus Sans", "Liberation Sans"],
    "font.size": 7.0,
    "axes.labelsize": 7.5,
    "axes.titlesize": 8.0,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
    "legend.fontsize": 7.0,
    "figure.titlesize": 8.0,
    "legend.title_fontsize": 7.0,
    # Type 42 (TrueType, embedded and subset); journal production systems reject Type 3 fonts.
    "pdf.fonttype": 42,
    "pdf.compression": 6,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    # Mathtext in the same family, so a unit superscript does not embed a second typeface.
    "mathtext.fontset": "custom",
    "mathtext.rm": "DejaVu Sans",
    "mathtext.it": "DejaVu Sans:italic",
    "mathtext.bf": "DejaVu Sans:bold",
    "mathtext.default": "regular",
    "axes.linewidth": 0.4,
    "axes.edgecolor": "0.3",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.color": "0.90",
    "grid.linewidth": 0.3,
    "lines.linewidth": 0.8,
    "lines.markersize": 2.5,
    "patch.linewidth": 0.3,
    "hatch.linewidth": 0.3,
    "xtick.major.width": 0.4,
    "ytick.major.width": 0.4,
    "xtick.major.size": 1.8,
    "ytick.major.size": 1.8,
    "legend.frameon": False,
    "legend.handlelength": 1.4,
    "legend.handletextpad": 0.5,
    "legend.columnspacing": 1.0,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": 400,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.01,
}


@contextmanager
def publication_rc(**overrides):
    """Apply the publication style for the duration of a ``with`` block.

    Parameters
    ----------
    **overrides
        rcParams applied on top of the publication style.

    Notes
    -----
    The style is ``_BASE_STYLE`` (seaborn "whitegrid" without tick marks) followed by ``_RC``
    (fonts, sizes, line widths, PDF font embedding). The previous rcParams are restored on exit.
    """
    with mpl.rc_context({**_BASE_STYLE, **_RC, **overrides}):
        yield


def blend(color, alpha: float, background="white") -> tuple:
    """Return the opaque colour equal to *color* drawn at *alpha* over *background*.

    Used where overlapping translucent fills would darken ambiguously, e.g. adjacent bars.

    Parameters
    ----------
    color : color
        Any matplotlib colour.
    alpha : float
        Opacity of *color*, in [0, 1].
    background : color, default "white"
        Colour underneath.

    Returns
    -------
    tuple of float
        RGB triple.
    """
    rgb = np.asarray(mcolors.to_rgb(color))
    back = np.asarray(mcolors.to_rgb(background))
    return tuple(alpha * rgb + (1 - alpha) * back)


#: Panel-letter rule shared by every figure: 8 pt bold lowercase; x at the left edge of the
#: panel's ink (y-label, tick labels, sign-convention arrows), shared by every letter of a column;
#: top of the letter at the top of the panel's ink (title, else axes), shared by every letter of a
#: row, raised where a tall y-label reaches the corner. With ``inside=True`` (map rows) the letter
#: goes inside the axes' top-left corner instead, ``LETTER_INSIDE_PT`` below the top ...
LETTER_INSIDE_PT = 1.0
#: ... and ``LETTER_INSIDE_X_PT`` in from the left edge.
LETTER_INSIDE_X_PT = 2.0


def _letter_anchor(fig: Figure, ax, renderer, to_figure,
                   inside: bool = False) -> tuple[float, float]:
    """Return the left and top of a panel's ink in figure coordinates.

    Parameters
    ----------
    fig : Figure
        Figure containing *ax*.
    ax : Axes
        The panel.
    renderer : RendererBase
        Renderer of *fig*.
    to_figure : Transform
        Display-to-figure transform (``fig.transFigure.inverted()``).
    inside : bool, default False
        Return the top as the axes top minus ``LETTER_INSIDE_PT`` rather than the ink's top.

    Returns
    -------
    tuple of float
        ``(left, top)``.

    Notes
    -----
    The title is hidden while the left edge is measured, because a centred title wider than its
    axes would otherwise move the left edge.
    """
    title = ax.title
    was_visible = title.get_visible()
    title.set_visible(False)
    fig.canvas.draw()
    left = ax.get_tightbbox(renderer).transformed(to_figure).x0
    title.set_visible(was_visible)
    fig.canvas.draw()
    if inside:
        return left, ax.get_position().y1 - LETTER_INSIDE_PT / 72 / fig.get_figheight()
    return left, ax.get_tightbbox(renderer).transformed(to_figure).y1


def panel_letter(fig: Figure, ax, letter: str, *, fontsize: float | None = None,
                 x: float | None = None, y: float | None = None, inside: bool = False):
    """Draw one bold panel letter at the position given by the panel-letter rule.

    Prefer :func:`panel_letters`, which also aligns the letters of a row and of a column.

    Parameters
    ----------
    fig : Figure
        Figure to draw on.
    ax : Axes
        Panel the letter belongs to.
    letter : str
        The letter.
    fontsize : float, optional
        Defaults to ``FS["title"]``.
    x, y : float, optional
        Position in figure coordinates; each one not given is measured from *ax*.
    inside : bool, default False
        Place the letter inside the axes' top-left corner (see ``LETTER_INSIDE_PT``).

    Returns
    -------
    Text
        The letter artist.

    Notes
    -----
    The letter is recorded in ``fig._panel_letters``; :func:`save` writes these into the PDF
    metadata so captions can be checked against the drawn panels.
    """
    if x is None or y is None:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        left, top = _letter_anchor(fig, ax, renderer, fig.transFigure.inverted(), inside)
        x = left if x is None else x
        y = top if y is None else y
    fig._panel_letters = sorted(set(getattr(fig, "_panel_letters", set())) | {letter})
    return fig.text(x, y, letter, ha="left", va="top", fontsize=fontsize or FS["title"],
                    fontweight="bold")


def _group(entries, key, tolerance):
    """Group *entries* whose *key* lies within *tolerance* of the first entry of a group.

    Parameters
    ----------
    entries : iterable
        Items to group.
    key : callable
        Maps an entry to a float.
    tolerance : float
        Maximum distance from a group's first (smallest) key.

    Returns
    -------
    list of list
        Groups in ascending key order.
    """
    groups: list[list] = []
    for entry in sorted(entries, key=key):
        if groups and abs(key(entry) - key(groups[-1][0])) <= tolerance:
            groups[-1].append(entry)
        else:
            groups.append([entry])
    return groups


def panel_letters(fig: Figure, pairs, *, column_tolerance: float = 0.06,
                  row_tolerance: float = 0.03, inside: bool = False,
                  fontsize: float | None = None):
    """Letter several panels: one x per column of panels, one top per row of panels.

    Parameters
    ----------
    fig : Figure
        Figure to draw on.
    pairs : sequence of (str, Axes)
        Letter and panel.
    column_tolerance : float, default 0.06
        Panels whose axes' left edges differ by at most this (figure fraction) form a column.
    row_tolerance : float, default 0.03
        Panels whose axes' top edges differ by at most this (figure fraction) form a row.
    inside : bool, default False
        Place letters inside the axes' top-left corner (map rows); see ``LETTER_INSIDE_PT``.
    fontsize : float, optional
        Defaults to ``FS["title"]``.

    Returns
    -------
    dict of str to Text
        The letter artists by letter.

    Notes
    -----
    Call after the final layout and size (e.g. after :func:`fit_width`): positions are measured on
    the rendered figure. A row's letters are raised where a rotated y-label reaches the axes top,
    leaving 2 pt between label and letter.
    """
    axes = [ax for _, ax in pairs]
    titles = [(ax.title, ax.title.get_visible()) for ax in axes]
    for title, _ in titles:
        title.set_visible(False)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    to_figure = fig.transFigure.inverted()
    lefts = [ax.get_tightbbox(renderer).transformed(to_figure).x0 for ax in axes]
    for title, was_visible in titles:
        title.set_visible(was_visible)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inset = LETTER_INSIDE_PT / 72 / fig.get_figheight()
    # Axes boxes from the rendered extent, not `get_position`: for an axes in a subfigure the
    # latter is relative to the subfigure, so side-by-side blocks would merge into one column.
    boxes = [ax.get_window_extent(renderer).transformed(to_figure) for ax in axes]
    tops = [box.y1 - inset if inside else ax.get_tightbbox(renderer).transformed(to_figure).y1
            for ax, box in zip(axes, boxes)]

    entries = [dict(letter=letter, ax=ax, x0=box.x0, y1=box.y1, left=left, top=top, lifted=False)
               for (letter, ax), box, left, top in zip(pairs, boxes, lefts, tops)]
    for column in _group(entries, lambda entry: entry["x0"], column_tolerance):
        for entry in column:
            entry["left"] = (min(other["x0"] for other in column)
                             + LETTER_INSIDE_X_PT / 72 / fig.get_figwidth()
                             if inside else min(other["left"] for other in column))
    letter_height = 1.15 * (fontsize or FS["title"]) / 72 / fig.get_figheight()
    for entry in entries:
        label = entry["ax"].yaxis.label
        if not inside and label.get_visible() and label.get_text().strip():
            label_top = label.get_window_extent(renderer).transformed(to_figure).y1
            gap = 2.0 / 72 / fig.get_figheight()
            if label_top > entry["top"] - letter_height - gap:
                entry["top"], entry["lifted"] = (max(entry["top"], label_top + letter_height + gap),
                                                 True)
    for row in _group(entries, lambda entry: entry["y1"], row_tolerance):
        pick = max if (not inside or any(other["lifted"] for other in row)) else min
        top = pick(other["top"] for other in row)
        for entry in row:
            entry["top"] = top
    return {entry["letter"]: panel_letter(fig, entry["ax"], entry["letter"], x=entry["left"],
                                          y=entry["top"], inside=inside, fontsize=fontsize)
            for entry in entries}


def align_colorbar_labels(fig: Figure, bars, *, tolerance: float = 0.02):
    """Align the labels of colorbars in the same column on one vertical line.

    Parameters
    ----------
    fig : Figure
        Figure containing the colorbars.
    bars : sequence of Colorbar
        Colorbars to align; those without a label are ignored.
    tolerance : float, default 0.02
        Colorbars whose left edges differ by at most this (figure fraction) form a column.

    Notes
    -----
    Each label is placed 3 pt right of the widest drawn tick label of its column. The offset is
    stored in points from the bar's edge, so it survives a later resize; call this after the
    final layout.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    entries = [(bar, bar.ax.get_window_extent(renderer)) for bar in bars
               if bar.ax.yaxis.label.get_text().strip()]
    width_px = fig.get_figwidth() * fig.dpi
    for column in _group(entries, lambda entry: entry[1].x0 / width_px, tolerance):
        def tick_right(bar):
            """Return the right edge (px) of the bar's drawn tick labels, else of the bar."""
            # Only ticks inside the bar's range: the locator also emits clipped, undrawn labels.
            low, high = sorted(bar.ax.get_ylim())
            return max((tick.get_window_extent(renderer).x1 for tick, location in
                        zip(bar.ax.yaxis.get_ticklabels(), bar.ax.yaxis.get_ticklocs())
                        if tick.get_visible() and tick.get_text()
                        and low - 1e-9 <= location <= high + 1e-9),
                       default=bar.ax.get_window_extent(renderer).x1)
        target = max(tick_right(bar) for bar, _ in column) + 3.0 * fig.dpi / 72
        # The label's anchor-to-edge distance depends on its alignment, so the offset is corrected
        # iteratively against the measured label position.
        offsets = {}
        for bar, extent in column:
            offsets[bar] = (target - extent.x1) * 72 / fig.dpi
        for _ in range(3):
            for bar, _ in column:
                shift = mtransforms.ScaledTranslation(offsets[bar] / 72, 0, fig.dpi_scale_trans)
                bar.ax.yaxis.set_label_coords(1.0, 0.5, transform=bar.ax.transAxes + shift)
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            for bar, _ in column:
                error = target - bar.ax.yaxis.label.get_window_extent(renderer).x0
                offsets[bar] += error * 72 / fig.dpi
            target = max(tick_right(bar) for bar, _ in column) + 3.0 * fig.dpi / 72
        for bar, _ in column:
            shift = mtransforms.ScaledTranslation(offsets[bar] / 72, 0, fig.dpi_scale_trans)
            bar.ax.yaxis.set_label_coords(1.0, 0.5, transform=bar.ax.transAxes + shift)
    fig.canvas.draw()


def legend_below(fig: Figure, axes, handles, labels, *, gap_pt: float = 6.0, ncol: int = 4,
                 fontsize: float = 7.0, **kwargs):
    """Place a figure legend centred under *axes*, a fixed gap below their lowest ink.

    Parameters
    ----------
    fig : Figure
        Figure to draw on.
    axes : Axes or list or tuple of Axes
        Panels the legend belongs to.
    handles, labels : sequence
        Legend entries.
    gap_pt : float, default 6.0
        Gap (pt) between the lowest tight bounding box of *axes* and the legend's top.
    ncol : int, default 4
        Number of legend columns.
    fontsize : float, default 7.0
        Legend font size.
    **kwargs
        Passed to :meth:`Figure.legend`.

    Returns
    -------
    Legend

    Notes
    -----
    The vertical position is measured off the tight bounding boxes (tick and axis labels
    included); the horizontal centre off the axes frames, since y labels hang off the left side
    only and would shift the centre left.
    """
    axes = list(axes) if isinstance(axes, (list, tuple)) else [axes]
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    to_figure = fig.transFigure.inverted()
    boxes = [ax.get_tightbbox(renderer).transformed(to_figure) for ax in axes]
    frames = [ax.get_position() for ax in axes]
    centre = 0.5 * (min(frame.x0 for frame in frames) + max(frame.x1 for frame in frames))
    top = min(box.y0 for box in boxes) - gap_pt / 72 / fig.get_figheight()
    return fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(centre, top),
                      ncol=ncol, frameon=False, fontsize=fontsize, **kwargs)


def fit_width(fig: Figure, *, target_mm: float = 180.0, max_height_mm: float | None = None,
              tolerance_mm: float = 0.6, max_iter: int = 5):
    """Resize the figure until its rendered (tight) width equals *target_mm*.

    Point sizes are absolute, so resizing the canvas changes the panels but not the type; the
    figure is then placed at print size without rescaling. Fixed text margins do not scale with
    the canvas, hence the iteration.

    Parameters
    ----------
    fig : Figure
        Figure to resize, in place.
    target_mm : float, default 180.0
        Target width of the tight bounding box.
    max_height_mm : float, optional
        Do not grow the figure beyond this height.
    tolerance_mm : float, default 0.6
        Stop once within this distance of the target.
    max_iter : int, default 5
        Maximum number of resizes.

    Returns
    -------
    Figure
        *fig*.
    """
    for _ in range(max_iter):
        fig.canvas.draw()
        width_mm = fig.get_tightbbox(fig.canvas.get_renderer()).width / MM
        if abs(width_mm - target_mm) <= tolerance_mm:
            break
        scale = target_mm / width_mm
        width, height = fig.get_size_inches()
        if max_height_mm is not None and height * scale / MM > max_height_mm:
            scale = min(scale, max_height_mm * MM / height)
        fig.set_size_inches(width * scale, height * scale)
    return fig


def save(fig: Figure, stem: str, figs_dir: Path, *, dpi: int = 400,
         max_width_mm: float = 180.0, font_range: tuple = FONT_RANGE, check_fonts: bool = True):
    """Write ``{stem}.pdf`` and ``{stem}.png`` after checking the artwork rules.

    Parameters
    ----------
    fig : Figure
        Figure to save (with ``bbox_inches="tight"``).
    stem : str
        File name without suffix.
    figs_dir : path-like
        Output directory; created if missing.
    dpi : int, default 400
        Resolution of the PNG.
    max_width_mm : float, default 180.0
        Maximum width of the tight bounding box (0.5 mm tolerance).
    font_range : tuple of float, default FONT_RANGE
        Permitted ``(min, max)`` type size (pt) of every non-empty text.
    check_fonts : bool, default True
        Check type sizes.

    Returns
    -------
    list of Path
        The PDF and PNG paths.

    Raises
    ------
    AssertionError
        If the width or a type size is out of range, or the PDF header is invalid.

    Notes
    -----
    The panel letters drawn through :func:`panel_letter` are written into the PDF ``Keywords``
    metadata as ``panels=<letters>``. A one-line summary (size, first panel height, file sizes)
    is printed.
    """
    figs_dir = Path(figs_dir)
    figs_dir.mkdir(parents=True, exist_ok=True)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    tight = fig.get_tightbbox(renderer)
    width_mm = tight.width / MM
    assert width_mm <= max_width_mm + 0.5, (
        f"{stem}: saved width {width_mm:.1f} mm exceeds the {max_width_mm:.0f} mm limit. "
        f"Note this is the TIGHT bbox — artists drawn outside the axes widen it.")

    if check_fonts:
        smallest, largest = font_range
        bad = sorted({round(float(text.get_fontsize()), 2) for text in fig.findobj(mpl.text.Text)
                      if text.get_text().strip()
                      and not (smallest - 1e-6 <= text.get_fontsize() <= largest + 1e-6)})
        assert not bad, f"{stem}: type sizes outside {smallest}-{largest} pt: {bad}"

    letters = "".join(getattr(fig, "_panel_letters", []))
    paths = []
    for suffix, savefig_kwargs in ((".pdf", dict(metadata={"Keywords": f"panels={letters}"})),
                                   (".png", dict(dpi=dpi))):
        path = figs_dir / f"{stem}{suffix}"
        fig.savefig(path, bbox_inches="tight", **savefig_kwargs)
        paths.append(path)

    with open(paths[0], "rb") as handle:
        assert handle.read(5) == b"%PDF-", f"{stem}: PDF did not write a valid header"

    sizes = "  ".join(f"{path.suffix[1:]} {path.stat().st_size / 1024:.0f} kB" for path in paths)
    # The first panel's height is reported because `fit_width` rescales whole figures, so equal
    # figsizes do not imply equal panel heights across figures.
    panel_mm = (fig.axes[0].get_position().height * fig.get_figheight() / MM) if fig.axes else 0.0
    print(f"  {stem:<28} {width_mm:5.1f} x {tight.height / MM:5.1f} mm   "
          f"panel {panel_mm:4.1f} mm   {sizes}")
    return paths


# ══════════════════════════════════════════════════════════════════════════════════════════
# Warming-level maps
# ══════════════════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class MapRow:
    """One row (quantity) of a warming-level map figure drawn by :func:`gwl_maps`.

    Attributes
    ----------
    label : str
        Row label, written left of the first map.
    field : callable
        ``field(period) -> DataArray`` with a ``model`` dimension plus either ``country``
        (``kind="country"``) or ``lat``/``lon`` (``kind="grid"``). Any season is fixed by the
        caller.
    cmap : str or Colormap
        Colour ramp of the reference-level column.
    kind : {"country", "grid"}, default "country"
        Country choropleth or gridded field.
    unit : str, default ""
        Unit of the reference-level colorbar.
    change : {"percent", "absolute"}, default "percent"
        Change columns as per cent of each chain's reference or as a difference in ``unit``.
    bins : sequence of float, optional
        Discrete change classes; ``None`` gives a continuous ramp.
    reference_center : float, optional
        Centre of a diverging reference-level scale; ``None`` gives a sequential scale.
    change_cmap : str, optional
        Change ramp; defaults to ``RdBu_r`` (red = more of the quantity).
    change_label : str, optional
        Change colorbar label; defaults to ``"% change"`` or ``"change (<unit>)"``.
    robust_abs_vmax : bool, default False
        Clip the reference-level scale of a gridded row at its 99th percentile.
    """

    label: str
    field: Callable[[str], "object"]
    cmap: str
    kind: str = "country"
    unit: str = ""
    change: str = "percent"
    bins: Sequence[float] | None = None
    reference_center: float | None = None
    change_cmap: str | None = None
    change_label: str | None = None
    robust_abs_vmax: bool = False


def sign_agreement(per_chain, model_dim: str = "model"):
    """Count the chains whose change has the sign of the ensemble-mean change.

    A vectorised equivalent of :func:`analysis.compute_sign_agreement` (identical results, much
    faster on large grids).

    Parameters
    ----------
    per_chain : DataArray
        Per-chain change; NaN entries do not count.
    model_dim : str, default "model"
        Chain dimension.

    Returns
    -------
    DataArray
        Agreement count, with *model_dim* removed.
    """
    import xarray as xr

    axis = per_chain.dims.index(model_dim)
    values = per_chain.values
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(values, axis=axis, keepdims=True)
        count = ((np.sign(values) == np.sign(mean)) & ~np.isnan(values)).sum(axis=axis)
    dims = [d for d in per_chain.dims if d != model_dim]
    return xr.DataArray(count, dims=dims,
                        coords={d: per_chain.coords[d] for d in dims if d in per_chain.coords})


#: Resolution (m) of the equal-area raster a lon/lat grid is resampled onto; finer than the
#: 0.1 degree source cells (7-11 km), so no two source cells merge into one pixel.
LAEA_RES_M = 5000.0
#: Cache of :func:`_laea_index` results, keyed by grid and CRS.
_RASTER_INDEX = {}


def _laea_index(lon, lat, crs):
    """Build the nearest-neighbour lookup from an equal-area raster to a regular lon/lat grid.

    Each target pixel (``LAEA_RES_M`` spacing over ``MAP_EXTENT_LAEA``) is mapped back to the
    source cell containing it. Results are cached in ``_RASTER_INDEX``.

    Parameters
    ----------
    lon, lat : ndarray
        Regular, ascending cell-centre coordinates of the source grid (degrees).
    crs : CRS or str
        Projected CRS of the target raster.

    Returns
    -------
    row, column : ndarray of int
        Source indices of every target pixel.
    valid : ndarray of bool
        Target pixels that fall inside the source grid.
    extent : tuple of float
        ``(x0, x1, y0, y1)`` of the target raster, for ``imshow``.
    footprint : Polygon
        Outline of the source grid in *crs*.
    """
    import pyproj
    from shapely.geometry import Polygon

    key = (float(lon[0]), float(lon[-1]), lon.size, float(lat[0]), float(lat[-1]), lat.size,
           str(crs))
    if key in _RASTER_INDEX:
        return _RASTER_INDEX[key]
    lon_step, lat_step = float(lon[1] - lon[0]), float(lat[1] - lat[0])
    lon_west, lon_east = lon[0] - lon_step / 2, lon[-1] + lon_step / 2
    lat_south, lat_north = lat[0] - lat_step / 2, lat[-1] + lat_step / 2
    ring_lon = np.r_[np.linspace(lon_west, lon_east, 400), np.full(400, lon_east),
                     np.linspace(lon_east, lon_west, 400), np.full(400, lon_west)]
    ring_lat = np.r_[np.full(400, lat_south), np.linspace(lat_south, lat_north, 400),
                     np.full(400, lat_north), np.linspace(lat_north, lat_south, 400)]
    to_map = pyproj.Transformer.from_crs(4326, crs, always_xy=True)
    ring_x, ring_y = to_map.transform(ring_lon, ring_lat)
    footprint = Polygon(zip(ring_x, ring_y))
    x0, x1, y0, y1 = MAP_EXTENT_LAEA
    xs = np.arange(x0 + LAEA_RES_M / 2, x1, LAEA_RES_M)
    ys = np.arange(y0 + LAEA_RES_M / 2, y1, LAEA_RES_M)
    grid_x, grid_y = np.meshgrid(xs, ys)
    to_lonlat = pyproj.Transformer.from_crs(crs, 4326, always_xy=True)
    source_lon, source_lat = to_lonlat.transform(grid_x, grid_y)
    column = np.rint((source_lon - lon[0]) / lon_step).astype(int)
    row = np.rint((source_lat - lat[0]) / lat_step).astype(int)
    valid = (row >= 0) & (row < lat.size) & (column >= 0) & (column < lon.size)
    extent = (xs[0] - LAEA_RES_M / 2, xs[-1] + LAEA_RES_M / 2,
              ys[0] - LAEA_RES_M / 2, ys[-1] + LAEA_RES_M / 2)
    _RASTER_INDEX[key] = (row, column, valid, extent, footprint)
    return _RASTER_INDEX[key]


def _to_laea(field, crs):
    """Resample a (lat, lon) field onto the equal-area raster by nearest neighbour.

    Parameters
    ----------
    field : DataArray
        Field on a regular ``lat``/``lon`` grid.
    crs : CRS or str
        Projected CRS of the raster.

    Returns
    -------
    values : ndarray
        Resampled values, NaN outside the source grid.
    extent : tuple of float
        ``(x0, x1, y0, y1)`` of the raster.
    footprint : Polygon
        Outline of the source grid in *crs*.
    """
    row, column, valid, extent, footprint = _laea_index(field["lon"].values,
                                                        field["lat"].values, crs)
    out = np.full(valid.shape, np.nan)
    out[valid] = np.asarray(field.values, dtype=float)[row[valid], column[valid]]
    return out, extent, footprint


def _draw_grid(ax, field, *, crs=None, **kwargs):
    """Draw a gridded field as an image.

    Parameters
    ----------
    ax : Axes
        Target axes.
    field : DataArray
        Field on ``lat``/``lon`` coordinates.
    crs : CRS or str, optional
        Resample onto the equal-area raster in this CRS (see :func:`_to_laea`). Otherwise a
        regular, south-to-north grid is drawn with ``imshow`` in lon/lat and an irregular one with
        a rasterised ``pcolormesh``.
    **kwargs
        Passed to the drawing call (``cmap``, ``norm``, ``vmax`` ...).

    Returns
    -------
    AxesImage or QuadMesh
    """
    if crs is not None:
        values, extent, _ = _to_laea(field, crs)
        return ax.imshow(values, extent=extent, origin="lower", interpolation="nearest", **kwargs)
    lon, lat = field["lon"].values, field["lat"].values
    lon_steps, lat_steps = np.diff(lon), np.diff(lat)
    regular = (lon_steps.size and lat_steps.size
               and float(lon_steps.max() - lon_steps.min()) < 1e-9
               and float(lat_steps.max() - lat_steps.min()) < 1e-9
               and lat_steps[0] > 0)
    if not regular:
        return ax.pcolormesh(lon, lat, field.values, shading="auto", rasterized=True, **kwargs)
    lon_step, lat_step = float(lon_steps[0]), float(lat_steps[0])
    extent = (lon[0] - lon_step / 2, lon[-1] + lon_step / 2,
              lat[0] - lat_step / 2, lat[-1] + lat_step / 2)
    return ax.imshow(field.values, extent=extent, origin="lower", aspect="auto",
                     interpolation="nearest", **kwargs)


def _set_reference_ticks(bar, target: int, *, symmetric: bool):
    """Set *target* round-valued ticks that lie inside a colorbar's range.

    ``MaxNLocator`` also emits ticks outside the data range, which are clipped from the drawing,
    so ``nbins`` is searched for the value giving *target* ticks within the range.

    Parameters
    ----------
    bar : Colorbar
        Colorbar to set the ticks of.
    target : int
        Wanted number of ticks.
    symmetric : bool
        Use a symmetric locator (diverging scale).

    Returns
    -------
    list of float
        The ticks set. If no ``nbins`` gives exactly *target*, the closest set is used and a
        message is printed.
    """
    low, high = float(bar.norm.vmin), float(bar.norm.vmax)
    best = None
    for nbins in range(2, 13):
        locator = MaxNLocator(nbins=nbins, symmetric=symmetric)
        ticks = [tick for tick in locator.tick_values(low, high) if low <= tick <= high]
        if len(ticks) == target:
            bar.set_ticks(ticks)
            return ticks
        if best is None or abs(len(ticks) - target) < abs(len(best) - target):
            best = ticks
    print(f"  [ticks] wanted {target} inside [{low:g}, {high:g}] but the closest round set has "
          f"{len(best)}: {best}")
    bar.set_ticks(best)
    return best


def _reference_colorbar(fig: Figure, ax, norm, row: "MapRow", extend: str, mappable=None):
    """Draw the colorbar of a reference-level (column 0) map.

    Parameters
    ----------
    fig : Figure
        Figure; the bar is appended to ``fig.reference_colorbars``.
    ax : Axes
        Map the colorbar belongs to.
    norm : Normalize
        Colour normalisation; ignored if *mappable* is given.
    row : MapRow
        Supplies the colour ramp, the unit label and whether the scale is diverging.
    extend : str
        Colorbar ``extend``.
    mappable : ScalarMappable, optional
        The drawn image; built from *norm* and ``row.cmap`` if omitted.

    Returns
    -------
    Colorbar
        With ``CBAR_TICKS_DIVERGING`` ticks on a diverging scale, ``CBAR_TICKS_SEQUENTIAL``
        otherwise.
    """
    if mappable is None:
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=row.cmap)
        mappable.set_array([])
    bar = fig.colorbar(mappable, ax=ax, shrink=0.72, fraction=0.15, pad=CBAR_PAD, extend=extend)
    bar.set_label(row.unit, fontsize=FS["tick"])
    bar.ax.tick_params(labelsize=FS["tick"])
    bar.outline.set_linewidth(0.3)
    diverging = row.reference_center is not None
    _set_reference_ticks(bar, CBAR_TICKS_DIVERGING if diverging else CBAR_TICKS_SEQUENTIAL,
                         symmetric=diverging)
    fig.reference_colorbars.append(bar)
    return bar


def _hatch_grid(ax, agreement, field, style: PaperStyle, *, crs=None) -> None:
    """Hatch grid cells where fewer than ``style.robustness_threshold`` chains agree on the sign.

    Parameters
    ----------
    ax : Axes
        Target map.
    agreement : DataArray
        Sign-agreement count on ``lat``/``lon``.
    field : DataArray
        The drawn change; cells where it is NaN are not hatched.
    style : PaperStyle
        Supplies ``robustness_threshold``.
    crs : CRS or str, optional
        Equal-area CRS the map is drawn in (see :func:`_draw_grid`).
    """
    mask = ((agreement.values < style.robustness_threshold)
            & np.isfinite(field.values)).astype(float)
    if crs is not None:
        mask, extent, _ = _to_laea(agreement.copy(data=mask), crs)
        xs = np.linspace(extent[0], extent[1], mask.shape[1])
        ys = np.linspace(extent[2], extent[3], mask.shape[0])
        coordinates = (xs, ys, np.nan_to_num(mask))
    else:
        coordinates = (agreement["lon"].values, agreement["lat"].values, mask)
    overlay = ax.contourf(*coordinates, levels=[0.5, 1.5], hatches=["///"], colors="none",
                          zorder=3)
    overlay.set_facecolor("none")
    overlay.set_edgecolor("black")
    overlay.set_linewidth(0)


def gwl_maps(
    rows: Sequence[MapRow],
    countries,
    style: PaperStyle,
    sign_agreement: Callable,
    *,
    height_mm: float = 230.0,
    width_mm: float = 180.0,
    fit_maps: bool = True,
    clip_percentile: float = 98,
    min_baseline_fraction: float = 0.002,
    edgecolor: str = "0.35",
    linewidth: float = 0.25,
    share_abs_scale: bool | Sequence[int] = False,
    share_change_scale: bool | Sequence[int] = False,
    figure=None,
    column_titles: Sequence[str] | None = None,
) -> tuple[Figure, list]:
    """Draw a grid of maps: one row per quantity, one column per warming level.

    Column 0 shows the ensemble median at the reference level; the other columns show the
    ensemble median of the per-chain change from the reference level, hatched where fewer than
    ``style.robustness_threshold`` chains agree on its sign. Every row has its own reference and
    change colorbar. Rows can mix country choropleths and gridded fields.

    Parameters
    ----------
    rows : sequence of MapRow
        One entry per row.
    countries : GeoDataFrame
        Country polygons with an ``ISO2`` column; projected (equal-area) or lon/lat.
    style : PaperStyle
        Supplies the warming levels (``period_names``) and the robustness threshold.
    sign_agreement : callable
        ``sign_agreement(per_chain, model_dim="model")`` returning the agreement count, e.g.
        this module's :func:`sign_agreement`.
    height_mm, width_mm : float, default 230.0, 180.0
        Initial figure size.
    fit_maps : bool, default True
        On projected maps, narrow the figure until the equal-aspect maps fill their cells.
    clip_percentile : float, default 98
        Percentile of absolute values used for symmetric, unbinned colour limits.
    min_baseline_fraction : float, default 0.002
        Per-cent changes of country rows are masked where the reference is below this fraction
        of the row's 90th-percentile magnitude.
    edgecolor, linewidth : str, float, default "0.35", 0.25
        Country borders.
    share_abs_scale : bool or sequence of int, default False
        Put the reference-level column of all rows (True) or of the listed rows on one scale.
    share_change_scale : bool or sequence of int, default False
        Same for the change columns of unbinned rows.
    figure : Figure or SubFigure, optional
        Draw into this (sub)figure instead of a new one; the caller then owns the size and
        *fit_maps* is ignored.
    column_titles : sequence of str, optional
        One title per column, replacing ``"<level>"`` / ``"<level>\\nchange vs <reference>"``.

    Returns
    -------
    fig : Figure
        The figure; ``fig.reference_colorbars`` and ``fig.change_colorbars`` list the colorbars.
    row_axes : list of Axes
        The column-0 axes of every row, which carry the row labels.

    Notes
    -----
    The change ramp is ``RdBu_r`` unless the row sets ``change_cmap``: red means more of the
    quantity. Per-cent changes are ``100 * (warm - reference) / reference`` per chain, then the
    median. Unbinned change scales are symmetric at the ``clip_percentile`` of absolute values.
    """
    n_rows, n_cols = len(rows), len(style.period_names)
    if figure is None:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(width_mm * MM, height_mm * MM),
                                 squeeze=False, constrained_layout=True)
    else:
        fig, fit_maps = figure, False
        axes = fig.subplots(n_rows, n_cols, squeeze=False)
    row_axes = []
    fig.reference_colorbars, fig.change_colorbars = [], []
    border = dict(edgecolor=edgecolor, linewidth=linewidth)

    # All rows' reference and change fields are computed before drawing, so that shared scales
    # can see every row.
    references_per_chain = [row.field(style.reference_period) for row in rows]
    references = [r.median("model") for r in references_per_chain]
    shared_limit = shared_vmin = shared_vmax = None
    shared_rows = (set(range(len(rows))) if share_abs_scale is True
                   else set() if share_abs_scale is False else set(share_abs_scale))
    if shared_rows:
        finite = np.concatenate([references[i].values[np.isfinite(references[i].values)].ravel()
                                 for i in sorted(shared_rows)])
        shared_vmin, shared_vmax = float(np.nanmin(finite)), float(np.nanmax(finite))
        shared_limit = max(float(np.nanpercentile(np.abs(finite), clip_percentile)), 1e-6)

    all_changes, all_agreements = [], []
    for row_index, row in enumerate(rows):
        reference_per_chain = references_per_chain[row_index]
        reference = references[row_index]
        absolute = row.change == "absolute"
        # Per-cent changes on a near-zero country baseline are masked; gridded rows are not.
        baseline_ok = (None if absolute or row.kind == "grid"
                       else _baseline_relevance_mask(reference, min_baseline_fraction))
        changes, agreements = [], []
        for period in style.period_names[1:]:
            delta = row.field(period) - reference_per_chain
            per_chain = delta if absolute else 100 * delta / reference_per_chain
            per_chain = per_chain.where(np.isfinite(per_chain))
            change = per_chain.median("model")
            if baseline_ok is not None:
                change = change.where(baseline_ok)
            changes.append(change)
            agreements.append(sign_agreement(per_chain, model_dim="model"))
        all_changes.append(changes)
        all_agreements.append(agreements)

    change_shared_rows = (set(range(len(rows))) if share_change_scale is True
                          else set() if share_change_scale is False else set(share_change_scale))
    shared_change_limit = (
        max(_symmetric_limit(c, clip_percentile)
            for i in sorted(change_shared_rows) for c in all_changes[i])
        if change_shared_rows else None)

    for row_index, row in enumerate(rows):
        reference = references[row_index]
        absolute = row.change == "absolute"
        edges = list(row.bins) if row.bins else None
        is_grid = row.kind == "grid"
        changes, agreements = all_changes[row_index], all_agreements[row_index]

        # Changes get their own diverging ramp (red = more of the quantity), never the row's
        # sequential reference ramp.
        change_cmap = row.change_cmap or "RdBu_r"
        change_label = row.change_label or (f"change ({row.unit})" if absolute else "% change")

        if edges is not None:
            norm, discrete_cmap = _binned_norm(edges, change_cmap)
            neutral = max(abs(edge) for edge in _zero_class(edges))
            change_kwargs = dict(norm=norm, cmap=discrete_cmap)
            colorbar_kwargs = dict(mappable=plt.cm.ScalarMappable(norm=norm, cmap=discrete_cmap),
                                   extend="both", ticks=edges, spacing="uniform")
        else:
            limit = (shared_change_limit if row_index in change_shared_rows
                     else max(_symmetric_limit(change, clip_percentile) for change in changes))
            data_max = max(float(np.nanmax(np.abs(change.values))) for change in changes)
            neutral = None
            norm = mcolors.TwoSlopeNorm(vcenter=0, vmin=-limit, vmax=limit)
            change_kwargs = dict(cmap=change_cmap, vcenter=0, vmin=-limit, vmax=limit)
            colorbar_kwargs = dict(mappable=plt.cm.ScalarMappable(norm=norm, cmap=change_cmap),
                                   extend="both" if data_max > limit * 1.0001 else "neither")

        if is_grid:
            # Country outlines are clipped to the grid's footprint (projected) or bounds (lon/lat),
            # so none are drawn beyond the field.
            crs = countries.crs if is_projected(countries) else None
            if crs is not None:
                _, _, footprint = _to_laea(reference, crs)
                clipped = countries.clip(footprint)
            else:
                lon0, lon1 = float(reference["lon"].min()), float(reference["lon"].max())
                lat0, lat1 = float(reference["lat"].min()), float(reference["lat"].max())
                clipped = countries.cx[lon0:lon1, lat0:lat1]

            ax = axes[row_index, 0]
            abs_vmax = (float(np.nanpercentile(reference.values, 99))
                        if row.robust_abs_vmax else None)
            if row_index in shared_rows and abs_vmax is None:
                abs_vmax = shared_vmax
            image = _draw_grid(ax, reference, crs=crs, cmap=row.cmap, vmax=abs_vmax)
            clipped.boundary.plot(ax=ax, color=edgecolor, linewidth=linewidth)
            _reference_colorbar(fig, ax, None, row,
                                "max" if abs_vmax is not None else "neither", mappable=image)

            for column, (change, agreement) in enumerate(zip(changes, agreements), start=1):
                ax = axes[row_index, column]
                _draw_grid(ax, change, crs=crs,
                           **({"norm": change_kwargs["norm"], "cmap": change_kwargs["cmap"]}
                              if edges is not None else {"cmap": change_cmap, "norm": norm}))
                clipped.boundary.plot(ax=ax, color=edgecolor, linewidth=linewidth)
                _hatch_grid(ax, agreement, change, style, crs=crs)

            for column in range(n_cols):
                ax = axes[row_index, column]
                if crs is not None:
                    ax.set_xlim(*MAP_EXTENT_LAEA[:2])
                    ax.set_ylim(*MAP_EXTENT_LAEA[2:])
                else:
                    ax.set_xlim(lon0, lon1)
                    ax.set_ylim(lat0, lat1)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.grid(False)
                set_map_aspect(ax, countries)
                for spine in ax.spines.values():
                    spine.set_visible(False)
        else:
            ax = axes[row_index, 0]
            # The norm is built here and shared by map and colorbar, rather than letting geopandas
            # build the colorbar, so that its ticks and padding can be controlled.
            if row.reference_center is None:
                finite = reference.values[np.isfinite(reference.values)]
                shared_here = row_index in shared_rows
                vmin = shared_vmin if shared_here else float(np.nanmin(finite))
                vmax = shared_vmax if shared_here else float(np.nanmax(finite))
                if vmax <= vmin:
                    vmax = vmin + 1e-6
                reference_norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
                reference_extend = "neither"
            else:
                limit = (shared_limit if row_index in shared_rows
                         else _symmetric_limit(reference, clip_percentile))
                reference_norm = mcolors.TwoSlopeNorm(vcenter=row.reference_center, vmin=-limit,
                                                      vmax=limit)
                reference_extend = "both"
            plot_country_choropleth(ax, reference.to_series().dropna(), countries,
                                    cmap=row.cmap, norm=reference_norm, legend=False, **border)
            _reference_colorbar(fig, ax, reference_norm, row, reference_extend)

            for column, (change, agreement) in enumerate(zip(changes, agreements), start=1):
                ax = axes[row_index, column]
                plot_country_choropleth(ax, change.to_series().dropna(), countries, legend=False,
                                        **border, **change_kwargs)
                _hatch_disagreeing_countries(ax, agreement, change, countries, style,
                                             min_magnitude=neutral)
                set_map_aspect(ax, countries)

        mappable = colorbar_kwargs.pop("mappable")
        mappable.set_array([])
        # matplotlib scales `pad` by the parent's width, and this bar spans all change columns
        # while the reference bar spans one; rescale so both sit the same distance from the maps.
        span = (axes[row_index, n_cols - 1].get_position().x1
                - axes[row_index, 1].get_position().x0)
        pad = CBAR_PAD * axes[row_index, 0].get_position().width / span
        bar = fig.colorbar(mappable, ax=[axes[row_index, column] for column in range(1, n_cols)],
                           fraction=0.030, pad=pad, **colorbar_kwargs)
        bar.set_label(change_label, fontsize=FS["tick"])
        bar.ax.tick_params(labelsize=FS["tick"])
        bar.outline.set_linewidth(0.3)
        fig.change_colorbars.append(bar)

        # The map axes are switched off, so re-enable just enough to carry the row label.
        axes[row_index, 0].set_ylabel(row.label, fontsize=FS["title"], fontweight="bold",
                                      labelpad=3)
        axes[row_index, 0].axis("on")
        axes[row_index, 0].set_xticks([])
        axes[row_index, 0].set_yticks([])
        axes[row_index, 0].grid(False)
        for spine in axes[row_index, 0].spines.values():
            spine.set_visible(False)
        row_axes.append(axes[row_index, 0])

    # Column titles state only the comparison; units are on each row's colorbar. Two lines, as
    # one line overruns the neighbouring title at map width.
    for column, period in enumerate(style.period_names):
        title = (column_titles[column] if column_titles is not None
                 else period if column == 0 else f"{period}\nchange vs {style.reference_period}")
        axes[0, column].set_title(title, fontsize=FS["title"], fontweight="bold")

    # Equal-aspect maps in a tall grid are height-limited and letterboxed in wider cells: narrow
    # the figure by the measured slack (cell width minus drawn map width) until it is gone.
    if fit_maps and is_projected(countries):
        for _ in range(4):
            fig.canvas.draw()
            probe = axes[0, n_cols - 1]
            slack = probe.get_position(original=True).width - probe.get_position().width
            slack_in = slack * fig.get_figwidth()
            if slack_in < 0.5 * MM:
                break
            fig.set_size_inches(fig.get_figwidth() - 0.95 * n_cols * slack_in, fig.get_figheight())
    return fig, row_axes


def mean_state_maps(
    quantities: Sequence[tuple],
    countries,
    style: PaperStyle,
    sign_agreement: Callable,
    *,
    unit: str = "TWh yr$^{-1}$",
    change_bins: Mapping[str, Sequence[float]] | None = None,
    absolute_change_rows: Sequence[str] = (),
    change_labels: Mapping[str, str] | None = None,
    **kwargs,
) -> tuple[Figure, list]:
    """Draw country-level warming-level maps from ``(label, field, cmap, reference_center)`` tuples.

    A wrapper around :func:`gwl_maps` with every row of ``kind="country"``.

    Parameters
    ----------
    quantities : sequence of tuple
        ``(label, field, cmap, reference_center)`` per row; see :class:`MapRow`.
    countries : GeoDataFrame
        Country polygons with an ``ISO2`` column.
    style : PaperStyle
        Warming levels and robustness threshold.
    sign_agreement : callable
        Sign-agreement counter, see :func:`gwl_maps`.
    unit : str, default "TWh yr$^{-1}$"
        Unit of every row.
    change_bins : mapping of str to sequence of float, optional
        Discrete change classes per row label.
    absolute_change_rows : sequence of str, default ()
        Rows whose change is drawn as a difference in ``unit`` rather than per cent.
    change_labels : mapping of str to str, optional
        Change colorbar label per row label.
    **kwargs
        Passed to :func:`gwl_maps`.

    Returns
    -------
    fig : Figure
    row_axes : list of Axes
        See :func:`gwl_maps`.
    """
    rows = [
        MapRow(label=name, field=field, cmap=cmap, kind="country", unit=unit,
               change="absolute" if name in absolute_change_rows else "percent",
               bins=(change_bins or {}).get(name), reference_center=absolute_center,
               change_label=(change_labels or {}).get(name))
        for name, field, cmap, absolute_center in quantities
    ]
    return gwl_maps(rows, countries, style, sign_agreement, **kwargs)


# ══════════════════════════════════════════════════════════════════════════════════════════
# Exceedance tails, event time and event regimes
# ══════════════════════════════════════════════════════════════════════════════════════════
def survival_axes(ax, survival, x, threshold, style: PaperStyle, *, ratio: float | None = None,
                  show_legend: bool = False):
    """Draw exceedance curves P(residual load > x), one per warming level, on a log y axis.

    Parameters
    ----------
    ax : Axes
        Target axes.
    survival : mapping
        ``{period: (median, low, high)}`` arrays over *x*; the band is the inter-chain spread.
    x : ndarray
        Residual-load grid.
    threshold : float
        Event threshold; the region above it is shaded and marked by a dashed line.
    style : PaperStyle
        Warming levels and their colours.
    ratio : float, optional
        Ratio of the warmest- to reference-level exceedance probability, printed as ``×ratio``
        at the right edge (red above 1, blue below).
    show_legend : bool, default False
        Draw a warming-level legend in the axes.
    """
    ax.axvspan(threshold, x[-1], color="0.88", alpha=0.6, lw=0, zorder=0)
    for period in style.period_names:
        median, low, high = survival[period]
        ax.plot(x, median, color=style.gwl_colors[period], lw=0.9, label=period, zorder=3)
        ax.fill_between(x, low, high, color=style.gwl_colors[period], alpha=0.18, lw=0, zorder=2)
    ax.axvline(threshold, color="0.2", ls="--", lw=0.6, zorder=4)
    ax.set_yscale("log")
    ax.set_ylim(2e-4, 1.05)
    ax.set_xlim(x[0], x[-1])
    ax.grid(True, which="major", color="0.92", lw=0.3)
    if ratio is not None and np.isfinite(ratio):
        # Right-aligned at the axes edge rather than at the threshold: the shaded band can be
        # too narrow to hold the label, but its right end is always inside it.
        ax.annotate(f"×{ratio:.1f}", xy=(0.98, 0.55), xycoords=("axes fraction", "data"),
                    ha="right", va="top", fontsize=FS["tick"], fontweight="bold",
                    color=_change_color(ratio - 1))
    if show_legend:
        ax.legend(fontsize=FS["tick"], frameon=False, loc="lower left", handlelength=1.0)


def paired_change_pct(reference: pd.Series, warmest: pd.Series, style: PaperStyle,
                      *, floor: float = 0.0):
    """Return the median per-cent change across chains, paired within each chain.

    Each chain's own change ``100 * (warmest - reference) / reference`` is computed first, then
    the median, so the result is consistent with the sign-agreement count.

    Parameters
    ----------
    reference, warmest : Series
        Per-chain values at the reference and warmest warming level, indexed by chain. Only
        chains present (non-NaN) in both are used.
    style : PaperStyle
        Supplies ``robustness_threshold``.
    floor : float, default 0.0
        Chains whose reference is at or below this are dropped.

    Returns
    -------
    median : float
        Median per-cent change; NaN if fewer than ``style.robustness_threshold`` chains remain.
    agreeing : int
        Chains whose change has the sign of the median (0 if the median is NaN).
    used, dropped : int
        Chains used and chains dropped by *floor*.
    """
    common = reference.dropna().index.intersection(warmest.dropna().index)
    reference, warmest = reference.loc[common], warmest.loc[common]
    usable = reference > floor
    used, dropped = int(usable.sum()), int((~usable).sum())
    if used < style.robustness_threshold:
        return np.nan, 0, used, dropped
    change = 100 * (warmest[usable] - reference[usable]) / reference[usable]
    median = float(change.median())
    return median, int((np.sign(change) == np.sign(median)).sum()), used, dropped


def panel_zone_duration_hours(axes, table: pd.DataFrame, style: PaperStyle, seasons,
                              duration_labels, *, day_classes=("multi_day",),
                              spread: bool = True):
    """Draw the time per year spent in events, per zone, one axes per (duration class, season).

    Axes are filled class-major: all seasons of the first class, then all seasons of the next.
    Axes of one duration class share their y limit.

    Parameters
    ----------
    axes : sequence of Axes
        ``len(duration_labels) * len(seasons)`` axes, or fewer (the remaining classes are skipped).
    table : DataFrame
        One row per (model, period, zone, season, duration_class) with an ``hours_yr`` column.
    style : PaperStyle
        Zones, warming levels and their colours.
    seasons : sequence of str
        Season keys, in axes order within a class.
    duration_labels : mapping of str to str
        Duration class to display label (its first line is used in the axes titles).
    day_classes : sequence of str, default ("multi_day",)
        Classes drawn in days per year instead of hours per year.
    spread : bool, default True
        Draw the inter-chain interquartile range as a bar with the median as a rule; otherwise
        a plain median bar.

    Notes
    -----
    A missing (NaN) cell is drawn as zero, and a zone with no events at any warming level is
    labelled "no events".
    """
    classes = list(duration_labels)
    positions = np.arange(len(style.zones))
    grouped = table.groupby(["season", "duration_class", "zone", "period"])["hours_yr"]
    quartiles = {statistic: grouped.quantile(quantile)
                 for statistic, quantile in (("low", 0.25), ("median", 0.5), ("high", 0.75))}

    def value(statistic, season, duration_class, zone, period):
        """Return one quartile of a cell, zero where the cell has no events.

        The burden table stores NaN for cells without events (mean duration of an empty set);
        zero lets the "no events" label fire. Day classes are divided by 24.
        """
        found = quartiles[statistic].get((season, duration_class, zone, period), np.nan)
        found = 0.0 if found is None or not np.isfinite(found) else float(found)
        return found / 24.0 if duration_class in day_classes else found

    layout = [(classes[index // len(seasons)], seasons[index % len(seasons)])
              for index in range(len(axes))]
    tops = {}
    for duration_class, season in layout:
        tops[duration_class] = max(
            tops.get(duration_class, 0.0),
            max(value("high" if spread else "median", season, duration_class, zone, period)
                for zone in style.zones for period in style.period_names))

    width = 0.26
    for ax, (duration_class, season) in zip(axes, layout):
        for k, period in enumerate(style.period_names):
            offset = (k - 1) * width
            for i, zone in enumerate(style.zones):
                median = value("median", season, duration_class, zone, period)
                if spread:
                    low = value("low", season, duration_class, zone, period)
                    high = value("high", season, duration_class, zone, period)
                    ax.bar(i + offset, high - low, width * 0.62, bottom=low,
                           color=blend(style.gwl_colors[period], 0.55), edgecolor="none", zorder=2)
                    ax.plot([i + offset - width * 0.42, i + offset + width * 0.42],
                            [median, median], color=style.gwl_colors[period], lw=1.0,
                            solid_capstyle="butt", zorder=3)
                else:
                    ax.bar(i + offset, median, width * 0.92, color=style.gwl_colors[period],
                           edgecolor="white", linewidth=0.3, zorder=2)

        for i, zone in enumerate(style.zones):
            if max(value("median", season, duration_class, zone, period)
                   for period in style.period_names) <= 1e-9:
                ax.annotate("no events", xy=(i, 0), xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", rotation=90, fontsize=FS["tick"], color="0.55")

        ax.set_ylim(0, max(tops[duration_class], 1e-9) * 1.18)
        ax.set_xticks(positions)
        # Vertical labels: a 30-degree label is wider than a zone slot on these narrow axes.
        ax.set_xticklabels(style.labels(), fontsize=FS["tick"], rotation=90, va="top", ha="center")
        ax.set_xlim(-0.6, len(style.zones) - 0.4)
        ax.set_title(f"{duration_labels[duration_class].splitlines()[0]} \u00b7 {season_label(season)}",
                     fontsize=FS["label"])
        ax.tick_params(labelsize=FS["tick"])
        ax.grid(True, axis="y", color="0.93", lw=0.3)
        ax.set_axisbelow(True)

    # One y label per class (classes differ in unit), on the class's first axes.
    for index, (duration_class, _season) in enumerate(layout):
        if index % len(seasons) == 0:
            unit = "days" if duration_class in day_classes else "hours"
            axes[index].set_ylabel(f"time in events ({unit[0]} yr$^{{-1}}$)", fontsize=FS["label"])


def tail_exceedance_figure(
    *,
    seasons: Sequence[str],
    residual_load_by_model: Mapping,
    period_windows: Mapping,
    zone_threshold,
    zone_season_hours: pd.DataFrame,
    season_months: Mapping[str, Sequence[int]],
    duration_labels: Mapping[str, str],
    style: PaperStyle,
    spread: str = "tukey",
    block_title: str = "Time per year inside extreme events, by duration class and zone",
    height_mm: float = 136.0,
    duration_block: bool = True,
) -> Figure:
    """Draw exceedance curves per (season, zone) above the time spent in events per zone.

    Panel a is a grid of :func:`survival_axes`, one row per season and one column per zone;
    panel b is :func:`panel_zone_duration_hours` for two duration classes and two seasons.

    Parameters
    ----------
    seasons : sequence of str
        Season keys, one survival row each (two when *duration_block* is True).
    residual_load_by_model : mapping
        Per-chain residual-load series, as taken by :mod:`extremes_analysis`.
    period_windows : mapping
        Warming-level windows per chain.
    zone_threshold : DataArray
        Event threshold per ``zone``.
    zone_season_hours : DataFrame
        Event-time table for :func:`panel_zone_duration_hours`.
    season_months : mapping of str to sequence of int
        Months of each season.
    duration_labels : mapping of str to str
        Duration class to display label.
    style : PaperStyle
        Zones, warming levels and colours.
    spread : str, default "tukey"
        Spread convention of the survival bands.
    block_title : str
        Title centred above panel b.
    height_mm : float, default 136.0
        Initial figure height.
    duration_block : bool, default True
        Draw panel b. Without it the survival grid fills the figure and no letters are drawn.

    Returns
    -------
    Figure

    Notes
    -----
    Rows are placed with explicit GridSpec bounds because the gap between the blocks must hold
    the survival tick labels, the shared x label and panel b's title.
    """
    from .extremes_analysis import (compute_exceedance_probability, compute_exceedance_survival,
                                    compute_survival_x_grid)

    zones = list(style.zones)
    seasons = list(seasons)
    fig = plt.figure(figsize=(WIDTH, height_mm * MM))
    top = gridspec.GridSpec(len(seasons), len(zones), figure=fig, left=0.055, right=0.995,
                            top=0.955, bottom=0.575 if duration_block else 0.255,
                            hspace=0.13, wspace=0.24)
    bottom = (gridspec.GridSpec(1, 4, figure=fig, left=0.055, right=0.995, top=0.375, bottom=0.150,
                                wspace=0.46) if duration_block else None)
    first_survival, last_row = None, []

    for column, zone in enumerate(zones):
        threshold = float(zone_threshold.sel(zone=zone))
        x = compute_survival_x_grid(residual_load_by_model, period_windows, zone,
                                    [season_months[s] for s in seasons], style.period_names,
                                    threshold=threshold)
        for row, season in enumerate(seasons):
            ax = fig.add_subplot(top[row, column])
            _, survival = compute_exceedance_survival(
                residual_load_by_model, period_windows, zone, season_months[season],
                style.period_names, spread=spread, grid=x)
            probability_reference = compute_exceedance_probability(
                residual_load_by_model, period_windows, zone, season_months[season],
                style.reference_period, threshold)
            probability_warmest = compute_exceedance_probability(
                residual_load_by_model, period_windows, zone, season_months[season],
                style.warmest_period, threshold)
            survival_axes(ax, survival, x, threshold, style,
                          ratio=(probability_warmest / probability_reference
                                 if probability_reference > 0 else np.nan),
                          show_legend=False)
            ax.tick_params(labelsize=FS["tick"])
            if row == 0:
                ax.set_title(style.zone_labels[zone], fontsize=FS["title"], fontweight="bold")
                if column == 0:
                    first_survival = ax
            else:
                last_row.append(ax)
            if column == 0:
                ax.set_ylabel(f"{season_label(season)}\nP(RL > x)", fontsize=FS["label"])
            else:
                ax.set_yticklabels([])
            if row != len(seasons) - 1:
                ax.set_xticklabels([])

    # One shared x label for the survival block; positioned vertically after fit_width below.
    span = [min(ax.get_position().x0 for ax in last_row),
            max(ax.get_position().x1 for ax in last_row)]
    x_label_artist = fig.text(sum(span) / 2, 0.50, "residual load per 3h step (GWh)", ha="center",
                              va="top", fontsize=FS["label"])

    if duration_block:
        duration_axes = [fig.add_subplot(bottom[i]) for i in range(4)]
        panel_zone_duration_hours(duration_axes, zone_season_hours, style, seasons,
                                  duration_labels)

    handles, labels = gwl_handles(style)
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.005),
               ncol=len(labels), frameon=False, fontsize=FS["tick"], title=GWL_TITLE)

    fit_width(fig)
    # Everything below is measured on the final size, hence after fit_width.
    fig.canvas.draw()
    to_figure = fig.transFigure.inverted()
    renderer = fig.canvas.get_renderer()

    if duration_block:
        # Panels a and b share one letter x: the leftmost ink of both blocks.
        left_x = min(min(ax.get_tightbbox(renderer).transformed(to_figure).x0 for ax in block)
                     for block in ([first_survival], duration_axes))

        boxes = [ax.get_tightbbox(renderer).transformed(to_figure) for ax in duration_axes]
        title_y = max(box.y1 for box in boxes) + 0.010
        block_text = fig.text(0.5 * (duration_axes[0].get_position().x0
                                     + duration_axes[-1].get_position().x1),
                              title_y, block_title, ha="center", va="bottom",
                              fontsize=FS["title"], fontweight="bold")
        fig.canvas.draw()
        block_top = block_text.get_window_extent(fig.canvas.get_renderer()).transformed(
            to_figure).y1
        # Letter b's top is at the top of its block title (a figure text, measured here).
        # `panel_letter` rather than `fig.text`, so that `save` records the letter.
        panel_letter(fig, duration_axes[0], "b", x=left_x, y=block_top)
        panel_letter(fig, first_survival, "a", x=left_x)

    # The shared x label sits 4 pt below the lowest survival tick labels.
    floor = min(ax.get_tightbbox(renderer).transformed(to_figure).y0 for ax in last_row)
    x_label_artist.set_position((x_label_artist.get_position()[0],
                                 floor - 4.0 / 72 / fig.get_figheight()))
    return fig


#: Class edges (change in events per year) of :func:`panel_season_contribution`, one set per
#: duration class since their ranges differ by an order of magnitude. Classes roughly double in
#: width; the panel colours by class rank, so wide outer classes do not wash out the inner ones.
SEASON_COUNT_BINS_SUB_DAILY = [-1, -0.5, -0.25, -0.05, 0.05, 0.25, 0.5, 1, 2, 4, 8]
SEASON_COUNT_BINS_MULTI_DAY = [-0.8, -0.4, -0.2, -0.1, -0.025, 0.025, 0.1, 0.2, 0.4, 0.8]


def season_contribution_by_model(seasonal: pd.DataFrame, annual: pd.DataFrame, zone: str,
                                 season: str, duration_class: str, style: PaperStyle,
                                 *, relative: bool = True) -> tuple[pd.Series, bool]:
    """Return each chain's contribution of one season to a zone's change in event frequency.

    Parameters
    ----------
    seasonal : DataFrame
        Columns ``model, period, zone, season, duration_class, freq_per_year``.
    annual : DataFrame
        Columns ``model, period, zone, duration_class, freq_per_year``.
    zone, season, duration_class : str
        The cell.
    style : PaperStyle
        Supplies the reference and warmest warming levels.
    relative : bool, default True
        If True, ``100 * (freq[season, warmest] - freq[season, reference]) /
        freq[annual, reference]``: points of the zone's annual per-cent change (chains with a zero
        annual reference are dropped). If False, the change in events per year.

    Returns
    -------
    values : Series
        Per-chain contribution, indexed by chain; only chains present in all three inputs.
    no_events : bool
        True if the cell has no events at either level (or no chains), as distinct from a cell
        with too few chains.

    Notes
    -----
    Each event is assigned whole to one season, so a chain's four relative contributions sum to
    its annual per-cent change; the four medians across chains do not.
    """
    rows = seasonal[(seasonal.zone == zone) & (seasonal.duration_class == duration_class)
                    & (seasonal.season == season)]
    reference = rows[rows.period == style.reference_period].set_index("model")["freq_per_year"]
    warmest = rows[rows.period == style.warmest_period].set_index("model")["freq_per_year"]
    base = annual[(annual.zone == zone) & (annual.duration_class == duration_class)
                  & (annual.period == style.reference_period)].set_index("model")["freq_per_year"]

    common = reference.dropna().index.intersection(warmest.dropna().index).intersection(
        base.dropna().index)
    if not len(common):
        return pd.Series(dtype=float), True
    reference, warmest, base = reference[common], warmest[common], base[common]
    no_events = bool((reference.abs().max() == 0) and (warmest.abs().max() == 0))
    if not relative:
        return warmest - reference, no_events
    usable = base > 0
    return 100 * (warmest[usable] - reference[usable]) / base[usable], no_events


def panel_season_contribution(ax, seasonal: pd.DataFrame, annual: pd.DataFrame, style: PaperStyle,
                              seasons, duration_class: str, *,
                              edges=SEASON_COUNT_BINS_SUB_DAILY, cmap: str = "RdBu_r",
                              relative: bool = False, show_colorbar: bool = True,
                              cbar_label="events yr$^{-1}$"):
    """Draw a seasons-by-zones heatmap of the median change in event frequency per season.

    Each cell is the median across chains of :func:`season_contribution_by_model`, coloured in
    discrete classes and printed. Cells where fewer than ``style.robustness_threshold`` chains
    agree on the sign are hatched; cells with no events read "none", cells with too few chains
    "n/a" with the chain count. Values smaller than half the smallest class edge print as zero
    and are never hatched.

    Parameters
    ----------
    ax : Axes
        Target axes.
    seasonal, annual : DataFrame
        Event-frequency tables, see :func:`season_contribution_by_model`.
    style : PaperStyle
        Zones, robustness threshold and number of chains.
    seasons : sequence of str
        Season keys, one row each.
    duration_class : str
        Duration class drawn.
    edges : sequence of float, default SEASON_COUNT_BINS_SUB_DAILY
        Class edges; colours are assigned by class rank.
    cmap : str, default "RdBu_r"
        Colour ramp (red = more events).
    relative : bool, default False
        Draw relative contributions instead of events per year.
    show_colorbar : bool, default True
        Draw a vertical colorbar in an inset right of the axes, leaving the axes width unchanged.
    cbar_label : str, default "events yr$^{-1}$"
        Colorbar label.

    Returns
    -------
    AxesImage
        The heatmap.
    """
    row_items = list(seasons)
    column_items = list(style.zones)
    values = np.full((len(row_items), len(column_items)), np.nan)
    agreement = np.zeros(values.shape, dtype=int)
    no_events = np.zeros(values.shape, dtype=bool)
    used = np.zeros(values.shape, dtype=int)
    neutral = min(abs(edge) for edge in edges) / 2

    for row, season in enumerate(row_items):
        for column, zone in enumerate(column_items):
            per_chain, empty = season_contribution_by_model(seasonal, annual, zone, season,
                                                            duration_class, style,
                                                            relative=relative)
            no_events[row, column] = empty
            used[row, column] = len(per_chain)
            if empty or len(per_chain) < style.robustness_threshold:
                continue
            value = float(per_chain.median())
            values[row, column] = value
            # A near-zero median means no change, so it is never hatched for sign disagreement.
            agreement[row, column] = (len(per_chain) if abs(value) < neutral
                                      else int((np.sign(per_chain) == np.sign(value)).sum()))

    norm, discrete = _binned_norm(edges, cmap, by="rank")
    discrete = discrete.copy()
    discrete.set_bad("white")
    image = ax.imshow(np.ma.masked_invalid(values), cmap=discrete, norm=norm, aspect="auto",
                      origin="upper",
                      extent=(-0.5, len(column_items) - 0.5, len(row_items) - 0.5, -0.5))

    digits = 1 if max(abs(edge) for edge in edges) >= 1 else 2
    for row in range(len(row_items)):
        for column in range(len(column_items)):
            if no_events[row, column]:
                ax.annotate("none", xy=(column, row), ha="center", va="center", fontsize=FS["tick"],
                            color="0.55")
                continue
            value = values[row, column]
            if not np.isfinite(value):
                ax.annotate(f"n/a\n{used[row, column]}/{style.n_models}", xy=(column, row),
                            ha="center", va="center", fontsize=FS["tick"], color="0.55")
                continue
            # White text on dark classes, dark text elsewhere, by the class colour's luminance.
            red, green, blue, _ = discrete(norm(value))
            light = 0.299 * red + 0.587 * green + 0.114 * blue > 0.55
            ax.annotate(f"{value:+.{digits}f}" if abs(value) >= neutral else f"{0.0:.{digits}f}",
                        xy=(column, row), ha="center", va="center", fontsize=FS["tick"],
                        color="0.15" if light else "white")
            if agreement[row, column] < style.robustness_threshold:
                ax.add_patch(plt.Rectangle((column - 0.5, row - 0.5), 1, 1, fill=False,
                                           hatch="///", edgecolor="0.45", lw=0.0, zorder=3))

    # No grid: it would run through the cell centres (the tick positions); borders are drawn below.
    ax.grid(False)
    ax.set_xticks(range(len(column_items)))
    ax.set_xticklabels(style.labels(), fontsize=FS["tick"], rotation=30, ha="right",
                       rotation_mode="anchor")
    ax.set_yticks(range(len(row_items)))
    ax.set_yticklabels([season_label(season) for season in row_items], fontsize=FS["tick"])
    ax.tick_params(length=0, labelsize=FS["tick"])
    for spine in ax.spines.values():
        spine.set_linewidth(0.4)
        spine.set_color("0.5")
    # White cell borders.
    for column in range(len(column_items) - 1):
        ax.axvline(column + 0.5, color="white", lw=0.8, zorder=2)
    for row in range(len(row_items) - 1):
        ax.axhline(row + 0.5, color="white", lw=0.8, zorder=2)

    if show_colorbar:
        # Vertical (the class edges do not fit horizontally), in an inset axes so the heatmap
        # keeps its width.
        bar = ax.figure.colorbar(image, cax=ax.inset_axes([1.03, 0.0, 0.035, 1.0]),
                                 orientation="vertical", extend="both", ticks=edges,
                                 spacing="uniform")
        # Each edge at its own precision ("-0.025", "0.4", "8").
        bar.ax.set_yticklabels([f"{edge:g}" for edge in edges])
        bar.set_label(cbar_label, fontsize=FS["tick"], labelpad=2.0)
        bar.ax.tick_params(labelsize=FS["tick"], length=1.5, width=0.4, pad=1.0)
        bar.outline.set_linewidth(0.4)
    return image


def panel_event_frequency(ax, events: pd.DataFrame, duration_class: str | None, style: PaperStyle,
                          ylabel: str, *, value_column: str = "freq_per_year",
                          yscale: str = "linear",
                          categories: Sequence[str] | None = None,
                          category_column: str = "zone",
                          category_labels: Sequence[str] | None = None,
                          rotation: float = 30,
                          annotations: Mapping[str, str] | None = None,
                          annotation_color: str | None = None) -> None:
    """Draw one quantity per category as box plots across chains, one box per warming level.

    Above each category the paired median per-cent change from the reference to the warmest
    level (:func:`paired_change_pct`) and the number of chains agreeing on its sign are printed.

    Parameters
    ----------
    ax : Axes
        Target axes.
    events : DataFrame
        Columns ``model, period, <category_column>, <value_column>`` and, if *duration_class* is
        given, ``duration_class``.
    duration_class : str or None
        Keep only this duration class; ``None`` keeps all rows.
    style : PaperStyle
        Zones, warming levels, colours, number of chains.
    ylabel : str
        Y-axis label.
    value_column : str, default "freq_per_year"
        Quantity drawn.
    yscale : str, default "linear"
        Y scale; on ``"log"`` the lower limit is set from the smallest positive value.
    categories : sequence of str, optional
        Categories on the x axis; defaults to ``style.zones``.
    category_column : str, default "zone"
        Column holding the categories.
    category_labels : sequence of str, optional
        Tick labels; default zone labels for zones, else the categories.
    rotation : float, default 30
        Tick-label rotation (degrees); 60 or more centres the labels.
    annotations : mapping of str to str, optional
        Text per category replacing the change annotation; categories missing from it are left
        blank.
    annotation_color : str, optional
        Annotation colour; by default red for an increase and blue otherwise, or dark grey
        for *annotations*.

    Notes
    -----
    Box = interquartile range across chains, whiskers = the most extreme chain within 1.5 IQR,
    line = median, no fliers. Box fills are the warming-level colours blended at 0.75.
    """
    # A fixed Tukey whisker, not `style.spread`: that convention is for bands and error bars,
    # and its IQR form would collapse the whiskers onto the box edges.
    whis = 1.5
    data = events if duration_class is None else events[events.duration_class == duration_class]
    categories = list(categories) if categories is not None else list(style.zones)
    labels = list(category_labels) if category_labels is not None else (
        style.labels() if category_column == "zone" else list(categories))
    width, positions = 0.26, np.arange(len(categories))
    if yscale != "linear":
        ax.set_yscale(yscale)

    for k, period in enumerate(style.period_names):
        per_category = [data[(data[category_column] == c) & (data.period == period)]
                        [value_column].values for c in categories]
        boxes = ax.boxplot(per_category, positions=positions + (k - 1) * width, widths=width * 0.86,
                           patch_artist=True, showfliers=False, whis=whis,
                           medianprops=dict(color="black", lw=0.7),
                           whiskerprops=dict(lw=0.4), capprops=dict(lw=0.4))
        for box in boxes["boxes"]:
            box.set_facecolor(blend(style.gwl_colors[period], 0.75))
            box.set_edgecolor("0.3")
            box.set_lw(0.4)

    # Headroom for the annotation row; a log axis also needs a positive lower limit.
    if yscale == "log":
        low, high = ax.get_ylim()
        finite = np.concatenate([data[data.period == p][value_column].values
                                 for p in style.period_names])
        finite = finite[np.isfinite(finite) & (finite > 0)]
        ax.set_ylim(max(low, finite.min() / 2.2) if finite.size else low, high * 3.2)
    else:
        ax.set_ylim(0, ax.get_ylim()[1] * 1.26)

    for i, category in enumerate(categories):
        rows = data[data[category_column] == category]
        if annotations is not None:
            text, color = annotations.get(category), annotation_color or "0.25"
            if text is None:
                continue
        else:
            reference = (rows[rows.period == style.reference_period]
                         .set_index("model")[value_column])
            warmest = rows[rows.period == style.warmest_period].set_index("model")[value_column]
            percent, n_agree, _, _ = paired_change_pct(reference, warmest, style)
            if not np.isfinite(percent):
                continue
            text = f"{percent:+.0f}%\n{n_agree}/{style.n_models}"
            color = annotation_color or _change_color(percent)
        # In axes fraction, so the placement is the same on linear and log axes.
        ax.annotate(text, xy=(i, 1.0), xycoords=("data", "axes fraction"),
                    xytext=(0, -1), textcoords="offset points",
                    ha="center", va="top", fontsize=FS["tick"], fontweight="bold", color=color)

    ax.set_xticks(positions)
    # Steep rotations are centred under the tick; shallow ones are right-aligned to it.
    steep = dict(ha="center", va="top") if rotation >= 60 else {}
    ax.set_xticklabels(labels, fontsize=FS["tick"], rotation=rotation,
                       **({"ha": "center" if rotation == 0 else "right"} if not steep else steep))
    # The zone axis keeps matplotlib's default margins.
    if category_column != "zone":
        ax.set_xlim(-0.5, len(categories) - 0.5)
    ax.set_ylabel(ylabel, fontsize=FS["label"])
    ax.tick_params(labelsize=FS["tick"])


def panel_hour_of_day(ax_jja, ax_djf, southern_hours: pd.DataFrame, style: PaperStyle,
                      *, evening=(15, 21.8), night=(-0.8, 6.0), band_color="#F5A623",
                      evening_label="evening", night_label="night",
                      ylabel="time above P95 (h yr$^{-1}$)"):
    """Draw threshold-exceedance hours per year by hour of day, summer above winter.

    Exceedance hours are counted per time step against the zone threshold, independently of the
    event detection. Lines are the median across chains, bands the inter-chain spread
    (``style.spread_percentiles``).

    Parameters
    ----------
    ax_jja, ax_djf : Axes
        Upper (summer) and lower (winter) axes; the lower carries the x and y labels.
    southern_hours : DataFrame
        Columns ``season, period, model, hour, hours_yr``.
    style : PaperStyle
        Warming levels, colours and spread convention.
    evening, night : tuple of float, default (15, 21.8), (-0.8, 6.0)
        Hour ranges shaded and labelled; together one window wrapping midnight.
    band_color : str, default "#F5A623"
        Shading colour.
    evening_label, night_label : str
        Labels of the two shaded ranges, written in the upper axes.
    ylabel : str, default "time above P95 (h yr$^{-1}$)"
        One y label, centred on the pair of axes.
    """
    low_percentile, high_percentile = style.spread_percentiles
    for ax, season in ((ax_jja, "JJA"), (ax_djf, "DJF")):
        data = southern_hours[southern_hours.season == season]
        for period in style.period_names:
            wide = data[data.period == period].pivot(index="model", columns="hour",
                                                     values="hours_yr")
            hours = wide.columns.values
            ax.plot(hours, wide.median(axis=0).values, "-o", ms=1.8,
                    color=style.gwl_colors[period], lw=0.9, zorder=3)
            ax.fill_between(hours, np.percentile(wide.values, low_percentile, axis=0),
                            np.percentile(wide.values, high_percentile, axis=0),
                            color=style.gwl_colors[period], alpha=0.15, lw=0, zorder=2)
        ax.axvspan(*evening, color=band_color, alpha=0.16, lw=0, zorder=0)
        ax.axvspan(*night, color=band_color, alpha=0.16, lw=0, zorder=0)
        ax.set_xlim(-0.8, 21.8)
        ax.set_xticks(range(0, 24, 3))
        ax.set_ylim(0, ax.get_ylim()[1] * 1.30)
        # Season name at the top centre: the gap between the two shaded windows, above the
        # midday trough of the curves.
        ax.annotate(season_label(season), xy=(0.50, 0.95), xycoords="axes fraction",
                    ha="center", va="top",
                    fontsize=FS["tick"], fontweight="bold", color="0.25")
        ax.tick_params(labelsize=FS["tick"])

    for label, window in ((evening_label, evening), (night_label, night)):
        ax_jja.annotate(label, xy=(float(np.mean(window)), ax_jja.get_ylim()[1]),
                        xytext=(0, -1), textcoords="offset points", ha="center", va="top",
                        fontsize=FS["tick"], fontweight="bold", color="#8a5a00")
    ax_djf.set_xlabel("hour of day", fontsize=FS["label"])
    ax_djf.set_ylabel(ylabel, fontsize=FS["label"])
    ax_jja.set_xticklabels([])

    # The y label is placed clear of the wider of the two axes' tick labels (measured), and
    # centred on the pair.
    figure = ax_jja.figure
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    inset = max(
        (ax.get_window_extent().x0
         - min(tick.get_window_extent(renderer).x0 for tick in ax.get_yticklabels()
               if tick.get_visible() and tick.get_text().strip()))
        / ax.get_window_extent().width
        for ax in (ax_jja, ax_djf))
    # Centre of the pair in the lower axes' fraction.
    lower, upper = ax_djf.get_position(), ax_jja.get_position()
    top = (upper.y1 - lower.y0) / lower.height
    ax_djf.yaxis.set_label_coords(-(inset + 0.02), 0.5 * top)


def _grouped_zone_axis(ax, style: PaperStyle, sublabels: Sequence[str], width: float,
                       *, zone_offset_pt: float | None = None, show_zones: bool = True) -> None:
    """Draw a two-level x axis: a sublabel under each bar of a group, the zone under the group.

    Sublabels are rotated to vertical if the widest does not fit its bar's pitch; zone names are
    horizontal if the longest fits 92 % of a zone slot, else rotated 30 degrees.

    Parameters
    ----------
    ax : Axes
        Axes with zones at integer x positions.
    style : PaperStyle
        Zones and zone labels.
    sublabels : sequence of str
        Labels of the bars within a zone group.
    width : float
        Bar pitch within a group, in data units.
    zone_offset_pt : float, optional
        Vertical offset (pt) of the zone names from the axes bottom; measured below the
        sublabels if omitted.
    show_zones : bool, default True
        Draw the zone names.
    """
    n = len(sublabels)
    ax.set_xticks([i + (s - (n - 1) / 2) * width
                   for i in range(len(style.zones)) for s in range(n)])
    ax.set_xticklabels(list(sublabels) * len(style.zones), fontsize=FS["tick"])
    ax.tick_params(axis="x", length=0, pad=1.0)
    ax.tick_params(axis="y", labelsize=FS["tick"])

    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    pitch_px = ax.get_window_extent().width / len(style.zones) * width / 1.0
    labels = [tick for tick in ax.get_xticklabels() if tick.get_text().strip()]
    if labels and max(tick.get_window_extent(renderer).width for tick in labels) > 1.0 * pitch_px:
        for tick in ax.get_xticklabels():
            tick.set_rotation(90)
            tick.set_va("top")
        ax.figure.canvas.draw()
    if zone_offset_pt is None:
        # 3 pt below the lowest sublabel; with blank sublabels, 5 pt below the axes.
        floor = ax.get_window_extent().y0
        drops = [floor - tick.get_window_extent(renderer).y0 for tick in labels]
        zone_offset_pt = (-(max(drops) * 72 / ax.figure.dpi + 3.0) if drops else -5.0)
    slot_mm = (ax.get_window_extent().width / len(style.zones)) / ax.figure.dpi * 25.4
    probe = ax.figure.text(0, 0, max(style.labels(), key=len), fontsize=FS["tick"])
    widest_mm = probe.get_window_extent(renderer).width / ax.figure.dpi * 25.4
    probe.remove()
    rotation, ha = (0, "center") if widest_mm <= slot_mm * 0.92 else (30, "right")

    if not show_zones:
        return
    for i, label in enumerate(style.labels()):
        ax.annotate(label, xy=(i, 0), xycoords=("data", "axes fraction"),
                    xytext=(0, zone_offset_pt), textcoords="offset points", ha=ha, va="top",
                    fontsize=FS["tick"], rotation=rotation, annotation_clip=False)


def panel_storage_stack(ax, attenuation: pd.DataFrame, style: PaperStyle,
                        *, group_by: tuple[str, Sequence[str], Sequence[str]] | None = None,
                        where: Mapping[str, object] | None = None,
                        ylabel="share of severity\nbalanced by storage (%)", show_legend: bool = False,
                        level_label: str | None = None, group_span: float = 0.80,
                        spread: bool = True):
    """Draw storage attenuation as stacked bars per zone: reservoir plus battery share.

    The two shares are the Shapley split of the severity removed by storage, so the stack height
    is the combined attenuation. Segment heights are medians across chains.

    Parameters
    ----------
    ax : Axes
        Target axes.
    attenuation : DataFrame
        One row per chain and cell with a ``zone`` column and the columns of ``STORAGE_PARTS``.
    style : PaperStyle
        Zones and spread convention.
    group_by : tuple (str, sequence, sequence of str), optional
        ``(column, values, sublabels)``: draw one stack per value of *column* within each zone,
        labelled by *sublabels*. One stack per zone if omitted.
    where : mapping of str to object, optional
        Keep only rows where each column equals the given value.
    ylabel : str
        Y-axis label.
    show_legend : bool, default False
        Draw the component legend in the axes.
    level_label : str, optional
        Text in the axes' top-left corner naming the fixed selection (e.g. the warming level).
    group_span : float, default 0.80
        Fraction of a zone slot taken by a group of stacks.
    spread : bool, default True
        Draw the inter-chain spread (``style.spread``) of the combined share as an error bar on
        each stack.
    """
    rows = attenuation
    for key, value in (where or {}).items():
        rows = rows[rows[key] == value]
    column, values, sublabels = group_by if group_by else (None, [None], None)
    grouped = len(values) > 1
    positions = np.arange(len(style.zones))
    width = group_span / len(values) if grouped else 0.62

    for i, zone in enumerate(style.zones):
        for k, value in enumerate(values):
            per_cell = rows[rows.zone == zone]
            if column is not None:
                per_cell = per_cell[per_cell[column] == value]
            offset = (k - (len(values) - 1) / 2) * width if grouped else 0.0
            bottom = 0.0
            for name, _label, color in STORAGE_PARTS:
                height = per_cell[name].median()
                if not np.isfinite(height):
                    continue
                ax.bar(i + offset, height, width * (0.88 if grouped else 1.0), bottom=bottom,
                       color=color, edgecolor="white", linewidth=0.4, zorder=2)
                bottom += height
            if spread and bottom > 0:
                total = per_cell[[name for name, _, _ in STORAGE_PARTS]].sum(axis=1)
                low, high = model_spread_bounds(total.values, style.spread, axis=0)
                median = float(total.median())
                ax.errorbar(i + offset, median, yerr=[[max(median - low, 0.0)],
                                                      [max(high - median, 0.0)]],
                            fmt="none", ecolor="0.25", elinewidth=0.6, capsize=1.2, capthick=0.5,
                            zorder=4)
        if grouped and i:
            ax.axvline(i - 0.5, color="0.85", lw=0.4, zorder=0)

    ax.set_ylim(0, 105)
    ax.set_ylabel(ylabel, fontsize=FS["label"])
    if grouped:
        ax.set_xlim(-0.5, len(style.zones) - 0.5)
        _grouped_zone_axis(ax, style, list(sublabels), width)
    else:
        ax.set_xticks(positions)
        ax.set_xticklabels(style.labels(), fontsize=FS["tick"], rotation=30, ha="right")
        ax.set_xlim(-0.6, len(style.zones) - 0.4)
        ax.tick_params(labelsize=FS["tick"])
    if level_label:
        ax.annotate(level_label, xy=(0.005, 0.98), xycoords="axes fraction", ha="left", va="top",
                    fontsize=FS["tick"], fontweight="bold", color="0.25")
    if show_legend:
        handles, labels = storage_handles(style)
        ax.legend(handles=handles, labels=labels, fontsize=FS["tick"], frameon=False, loc="lower left",
                  ncol=3, handlelength=1.0)


def panel_driver_attribution(ax, attribution: Callable[[str, str], pd.DataFrame],
                             seasons: Sequence[str], style: PaperStyle,
                             *, ylabel="Δ per 3h step (GWh)", show_net_spread: bool = True,
                             group_span: float = 0.80,
                             show_sign_convention: bool = True,
                             sublabels: Sequence[str] | None = None,
                             reserve_sign_gutter: bool = True,
                             show_zone_labels: bool = True) -> None:
    """Draw each driver's contribution to the change in residual load as diverging stacked bars.

    One stack per (zone, season): positive contributions stacked upward from zero, negative ones
    downward, each segment the median across chains and hatched where fewer than
    ``style.robustness_threshold`` chains agree on its sign. A black diamond marks the median net
    change, with the inter-chain spread (``style.spread``) as error bar.

    Parameters
    ----------
    ax : Axes
        Target axes.
    attribution : callable
        ``attribution(zone, season) -> DataFrame`` indexed by chain, with one column per driver
        in ``style.drivers`` and a ``net`` column.
    seasons : sequence of str
        Season keys, one stack each within a zone group.
    style : PaperStyle
        Zones, drivers (stacking order) and their colours.
    ylabel : str, default "Δ per 3h step (GWh)"
        Y-axis label.
    show_net_spread : bool, default True
        Draw the error bar of the net change.
    group_span : float, default 0.80
        Fraction of a zone slot taken by its stacks.
    show_sign_convention : bool, default True
        Draw the "deepens"/"eases" arrows left of the axis.
    sublabels : sequence of str, optional
        Text under each stack; defaults to the season names.
    reserve_sign_gutter : bool, default True
        Keep the y label at the position it has with the arrows even when they are not drawn,
        so stacked panels align.
    show_zone_labels : bool, default True
        Draw the zone names under the groups.
    """
    width = group_span / len(seasons)

    for season_index, season in enumerate(seasons):
        offset = (season_index - (len(seasons) - 1) / 2) * width
        for i, zone in enumerate(style.zones):
            per_chain = attribution(zone, season)
            median = per_chain.median()
            top = bottom = 0.0
            for driver in style.drivers:
                value = median[driver]
                n_agree = int((np.sign(per_chain[driver]) == np.sign(value)).sum())
                base = top if value >= 0 else bottom
                ax.bar(i + offset, value, width * 0.88, bottom=base,
                       color=style.driver_palette[driver],
                       edgecolor="white", linewidth=0.25, zorder=2,
                       hatch=("///" if n_agree < style.robustness_threshold else None))
                if value >= 0:
                    top += value
                else:
                    bottom += value

            net = per_chain["net"]
            low, high = model_spread_bounds(net.values, style.spread, axis=0)
            spread = ([[net.median() - low], [high - net.median()]] if show_net_spread else None)
            ax.errorbar(i + offset, net.median(), yerr=spread, fmt="D", ms=2.0, color="black",
                        ecolor="black", elinewidth=0.6, capsize=1.2, capthick=0.5, zorder=6)
            if season_index == 0 and i:
                ax.axvline(i - 0.5, color="0.85", lw=0.4, zorder=0)

    ax.axhline(0, color="0.25", lw=0.5, zorder=3)
    ax.set_xlim(-0.5, len(style.zones) - 0.5)

    _grouped_zone_axis(ax, style,
                       [season_label(season) for season in seasons]
                       if sublabels is None else sublabels, width,
                       show_zones=show_zone_labels)

    ax.set_ylabel(ylabel, fontsize=FS["label"])
    # The gutter left of the axis is laid out in millimetres from the widest tick label (the
    # inset): arrows, their labels, then the y label. Millimetres rather than axes fractions so
    # it does not scale with the panel width.
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    extent = ax.get_window_extent()
    width_mm = extent.width / ax.figure.dpi * 25.4
    ticks = [tick for tick in ax.get_yticklabels()
             if tick.get_visible() and tick.get_text().strip()]
    inset_mm = (max((extent.x0 - tick.get_window_extent(renderer).x0 for tick in ticks),
                    default=0.0)
                / ax.figure.dpi * 25.4)
    if show_sign_convention:
        _annotate_sign_convention(ax, arrow_x=-(inset_mm + 1.5) / width_mm,
                                  label_x=-(inset_mm + 3.7) / width_mm,
                                  labels=SIGN_LABELS)
    if reserve_sign_gutter or show_sign_convention:
        ax.yaxis.set_label_coords(-(inset_mm + 7.2) / width_mm, 0.5)
    else:
        ax.yaxis.set_label_coords(-(inset_mm + 1.6) / width_mm, 0.5)
    for text in ax.texts:
        if text.get_text() in SIGN_LABELS:
            text.set_fontsize(FS["tick"])


def regime_figure(regime: str, *, events_annual: pd.DataFrame, events_by_season: pd.DataFrame,
                  style: PaperStyle, titles: Mapping[str, str], season_bins: Sequence[float],
                  seasons: Sequence[str], attribution: Callable[[str, str], pd.DataFrame],
                  driver_seasons: Sequence[str], driver_zones: Sequence[str] | None = None,
                  diurnal: pd.DataFrame | None = None,
                  diurnal_windows: Mapping[str, object] | None = None,
                  driver_display: Mapping[str, str] | None = None,
                  letters: str = "abcd", height_mm: float = 128) -> Figure:
    """Draw the figure of one event-duration regime: frequency, seasons and drivers.

    Panel a: event frequency per zone (:func:`panel_event_frequency`); b: per-season change
    (:func:`panel_season_contribution`); c: exceedance by hour of day
    (:func:`panel_hour_of_day`), only if *diurnal* is given; last: driver attribution
    (:func:`panel_driver_attribution`), across the whole bottom row when there is no panel c.

    Parameters
    ----------
    regime : str
        Duration class, ``"sub_daily"`` or ``"multi_day"``.
    events_annual, events_by_season : DataFrame
        Event-frequency tables, see :func:`season_contribution_by_model`.
    style : PaperStyle
        Zones, warming levels, colours.
    titles : mapping of str to str
        Panel titles by letter (``"a"`` to ``"d"``).
    season_bins : sequence of float
        Class edges of panel b.
    seasons : sequence of str
        Seasons of panel b.
    attribution : callable
        ``attribution(zone, season)`` for the driver panel.
    driver_seasons : sequence of str
        Seasons of the driver panel.
    driver_zones : sequence of str, optional
        Zones of the driver panel; defaults to ``style.zones``.
    diurnal : DataFrame, optional
        Hour-of-day exceedance table for panel c; omitted for the multi-day regime.
    diurnal_windows : mapping, optional
        Keyword arguments of :func:`panel_hour_of_day`.
    driver_display : mapping of str to str, optional
        Display names of drivers in the legend.
    letters : str, default "abcd"
        Panel letters, in panel order.
    height_mm : float, default 128
        Initial figure height.

    Returns
    -------
    Figure
    """
    is_sub = diurnal is not None
    fig = plt.figure(figsize=(WIDTH, height_mm * MM))
    # hspace holds panel a's rotated zone names, the warming-level legend and the bottom titles.
    grid = gridspec.GridSpec(2, 2, hspace=0.78, wspace=0.30, figure=fig)

    ax_a = fig.add_subplot(grid[0, 0])
    # Black annotations: the red/blue change colours would compete with the warming-level boxes.
    panel_event_frequency(ax_a, events_annual, regime, style,
                          f"{'sub-daily' if regime == 'sub_daily' else 'multi-day'} events yr$^{{-1}}$",
                          annotation_color="black")
    ax_a.set_title(titles["a"], fontsize=FS["title"], fontweight="bold")

    ax_b = fig.add_subplot(grid[0, 1])
    panel_season_contribution(ax_b, events_by_season, events_annual, style, seasons, regime,
                              edges=season_bins)
    ax_b.set_title(titles["b"], fontsize=FS["title"], fontweight="bold")

    if is_sub:
        # Nested, equal-width bottom row, independent of the top row's widths.
        bottom = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=grid[1, :], wspace=0.40)
        cell = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=bottom[0], hspace=0.18)
        ax_c = fig.add_subplot(cell[0])
        panel_hour_of_day(ax_c, fig.add_subplot(cell[1]), diurnal, style,
                          **(diurnal_windows or {}))
        ax_c.set_title(titles["c"], fontsize=FS["title"], fontweight="bold")
        ax_drivers = fig.add_subplot(bottom[1])
        drivers_letter = "d"
    else:
        ax_c = ax_drivers = fig.add_subplot(grid[1, :])
        drivers_letter = "c"

    driver_style = replace(style, zones=list(driver_zones)) if driver_zones else style
    panel_driver_attribution(ax_drivers, attribution, list(driver_seasons), driver_style)
    ax_drivers.set_title(titles[drivers_letter], fontsize=FS["title"], fontweight="bold")

    fit_width(fig)
    # Legends are placed after fit_width, as they are measured off the final layout: the
    # warming-level legend under panel a, the driver legend under the driver panel.
    gwl_legend_handles, gwl_legend_labels = gwl_handles(style)
    legend_below(fig, [ax_a], gwl_legend_handles, gwl_legend_labels, ncol=len(gwl_legend_labels),
                 title=GWL_TITLE)
    driver_legend_handles, driver_legend_labels = driver_handles(style)
    legend_below(fig, ax_drivers, driver_legend_handles,
                 [(driver_display or {}).get(name, name) for name in driver_legend_labels], ncol=4)
    lettered = [ax_a, ax_b, ax_c] + ([ax_drivers] if is_sub else [])
    panel_letters(fig, list(zip(letters, lettered)))
    return fig


def panel_zone_map(ax, countries, style, zone_members, zone_colors):
    """Draw a map of the balancing zones with a two-column legend below it.

    Parameters
    ----------
    ax : Axes
        Target axes.
    countries : GeoDataFrame
        Country polygons with an ``ISO2`` column.
    style : PaperStyle
        Zones (drawing and legend order) and zone labels.
    zone_members : mapping of str to iterable of str
        ISO2 codes of each zone's countries; other countries are drawn grey.
    zone_colors : mapping of str to color
        Fill colour of each zone.
    """
    zone_by_iso2 = {iso: zone for zone, members in zone_members.items()
                    for iso in members}
    frame = countries.copy()
    frame["zone"] = frame["ISO2"].map(zone_by_iso2)
    frame[frame["zone"].isna()].plot(ax=ax, color="0.90", edgecolor="0.35", linewidth=0.25)
    for zone in style.zones:
        subset = frame[frame["zone"] == zone]
        if len(subset):
            subset.plot(ax=ax, color=zone_colors[zone], edgecolor="white", linewidth=0.3)
    x0, x1, y0, y1 = map_extent(countries)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_axis_off()
    set_map_aspect(ax, countries)
    # Anchored by its top just below the axes, so the legend grows downward, away from the map.
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=zone_colors[zone])
                       for zone in style.zones],
              labels=style.labels(), fontsize=FS["tick"], frameon=False, ncol=2,
              loc="upper center", bbox_to_anchor=(0.5, -0.01), handlelength=1.0,
              handletextpad=0.5, columnspacing=1.0)


def panel_southern_survival(axes, residual_load_by_model, period_windows, zone_threshold,
                            season_months, style, zones, seasons, spread):
    """Draw a (season x zone) grid of exceedance curves, as :func:`survival_axes`.

    Parameters
    ----------
    axes : sequence of sequence of Axes
        ``axes[row][column]``, one row per season and one column per zone.
    residual_load_by_model, period_windows, zone_threshold, season_months
        See :func:`tail_exceedance_figure`.
    style : PaperStyle
        Warming levels, zone labels and colours.
    zones : sequence of str
        Zones, one column each.
    seasons : sequence of str
        Seasons, one row each; the last row carries the x label.
    spread : str
        Spread convention of the bands.
    """
    # Imported locally, as in tail_exceedance_figure.
    from .extremes_analysis import (compute_exceedance_probability, compute_exceedance_survival,
                                    compute_survival_x_grid)
    for column, zone in enumerate(zones):
        threshold = float(zone_threshold.sel(zone=zone))
        x = compute_survival_x_grid(residual_load_by_model, period_windows, zone,
                                    [season_months[s] for s in seasons], style.period_names,
                                    threshold=threshold)
        for row, season in enumerate(seasons):
            ax = axes[row][column]
            _, survival = compute_exceedance_survival(
                residual_load_by_model, period_windows, zone, season_months[season], style.period_names,
                spread=spread, grid=x)
            probability_reference = compute_exceedance_probability(
                residual_load_by_model, period_windows, zone, season_months[season],
                style.reference_period, threshold)
            probability_warmest = compute_exceedance_probability(
                residual_load_by_model, period_windows, zone, season_months[season],
                style.warmest_period, threshold)
            survival_axes(ax, survival, x, threshold, style,
                          ratio=(probability_warmest / probability_reference
                                 if probability_reference > 0 else np.nan),
                          show_legend=False)
            ax.tick_params(labelsize=FS["tick"])
            if row == 0:
                ax.set_title(style.zone_labels[zone], fontsize=FS["title"], fontweight="bold")
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("residual load per 3h step (GWh)", fontsize=FS["label"])
            if column == 0:
                ax.set_ylabel(f"{season_label(season)}\nP(RL > x)", fontsize=FS["label"])
            else:
                ax.set_yticklabels([])


def zone_tail_duration_figure(*, countries, residual_load_by_model, period_windows,
                              zone_threshold, season_months, style, southern_zones,
                              seasons, spread, zone_members, zone_colors, duration_spectrum,
                              bin_labels, layout="beside", map_share=0.74):
    """Draw the zone map, southern exceedance curves and the event-duration spectrum.

    Panel a: balancing zones (:func:`panel_zone_map`); b: exceedance curves of the southern zones
    (:func:`panel_southern_survival`); c and d: event frequency and severity per duration bin,
    median across chains with the inter-chain spread (``style.spread``) as a band and a dashed
    line at 24 h.

    Parameters
    ----------
    countries : GeoDataFrame
        Country polygons with an ``ISO2`` column.
    residual_load_by_model, period_windows, zone_threshold, season_months
        See :func:`tail_exceedance_figure`.
    style : PaperStyle
        Zones, warming levels, colours, spread convention.
    southern_zones : sequence of str
        Zones of panel b (two).
    seasons : sequence of str
        Seasons of panel b (two).
    spread : str
        Spread convention of panel b's bands.
    zone_members, zone_colors : mapping
        See :func:`panel_zone_map`.
    duration_spectrum : callable
        ``duration_spectrum(value)`` for ``value`` in ``("counts", "severity")``, returning a
        DataFrame with columns ``period, bin, value`` (one row per chain).
    bin_labels : sequence of str
        Duration-bin labels in order; must contain ``"12-24 h"``.
    layout : {"beside", "spanning"}, default "beside"
        ``"beside"``: map and panel b side by side above c and d. ``"spanning"``: the map takes
        the full height on the left, with b above c and d on the right.
    map_share : float, default 0.74
        Width ratio of the map column, out of a pair summing to 2.

    Returns
    -------
    Figure
    """
    if layout == "beside":
        fig = plt.figure(figsize=(WIDTH, 100 * MM))
        outer = gridspec.GridSpec(2, 1, height_ratios=(1.0, 0.66), hspace=0.55, figure=fig)
        top = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0], wspace=0.24,
                                               width_ratios=(map_share, 2.0 - map_share))
        map_cell, survival_spec = top[0], top[1]
        bottom_spec = outer[1]
    else:
        fig = plt.figure(figsize=(WIDTH, 104 * MM))
        outer = gridspec.GridSpec(1, 2, wspace=0.24,
                                  width_ratios=(map_share, 2.0 - map_share), figure=fig)
        right = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[1], hspace=0.62,
                                                 height_ratios=(1.0, 0.90))
        map_cell, survival_spec = outer[0], right[0]
        bottom_spec = right[1]

    ax_a = fig.add_subplot(map_cell)
    panel_zone_map(ax_a, countries, style, zone_members, zone_colors)
    ax_a.set_title("Balancing zones", fontsize=FS["title"], fontweight="bold")

    survival_cell = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=survival_spec,
                                                     hspace=0.22, wspace=0.10)
    survival = [[fig.add_subplot(survival_cell[row, column]) for column in range(2)]
                for row in range(2)]
    panel_southern_survival(survival, residual_load_by_model, period_windows, zone_threshold,
                            season_months, style, southern_zones, seasons, spread)
    survival[0][0].set_title(style.zone_labels[southern_zones[0]], fontsize=FS["title"],
                             fontweight="bold")

    # Nested and equal-width, so map_share does not reach panels c and d.
    bottom_cell = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=bottom_spec, wspace=0.30)
    bottom = [fig.add_subplot(bottom_cell[0]), fig.add_subplot(bottom_cell[1])]
    positions = np.arange(len(bin_labels))
    boundary = bin_labels.index("12-24 h") + 0.5
    for ax, value, label in zip(bottom, ("counts", "severity"),
                                ("events yr$^{-1}$", "severity (TWh yr$^{-1}$)")):
        spectrum = duration_spectrum(value)
        for period in style.period_names:
            block = spectrum[spectrum.period == period]
            median = [block[block.bin == duration_bin]["value"].median()
                      for duration_bin in bin_labels]
            bounds = [model_spread_bounds(block[block.bin == duration_bin]["value"].values,
                                          style.spread)
                      for duration_bin in bin_labels]
            low = [bound_low for bound_low, _ in bounds]
            high = [bound_high for _, bound_high in bounds]
            ax.fill_between(positions, low, high, color=style.gwl_colors[period], alpha=0.13, lw=0)
            ax.plot(positions, median, "-o", ms=2.2, lw=1.0, color=style.gwl_colors[period],
                    label=period)
        ax.axvline(boundary, color="0.35", lw=0.7, ls="--", zorder=1)
        ax.annotate("24 h", xy=(boundary, 1.0), xycoords=("data", "axes fraction"),
                    xytext=(2, -2), textcoords="offset points", ha="left", va="top",
                    fontsize=FS["tick"], color="0.35")
        ax.set_xticks(positions)
        ax.set_xticklabels(bin_labels, fontsize=FS["tick"], rotation=45, ha="right")
        ax.set_ylabel(label, fontsize=FS["label"])
        ax.set_xlabel("event duration", fontsize=FS["label"])
        ax.tick_params(labelsize=FS["tick"])
    bottom[0].set_title("Frequency of events by duration", fontsize=FS["title"], fontweight="bold")
    bottom[1].set_title("Severity by duration", fontsize=FS["title"], fontweight="bold")

    fit_width(fig)
    legend_below(fig, bottom, *gwl_handles(style), ncol=3, title=GWL_TITLE)
    panel_letters(fig, [("a", ax_a), ("b", survival[0][0]),
                        ("c", bottom[0]), ("d", bottom[1])])
    return fig


#: Blend strength of the reference and warmest level in :func:`panel_scenario_bars`.
LEVEL_ALPHA = (0.35, 0.90)


def panel_scenario_bars(ax, by_scenario: Mapping[str, pd.DataFrame], style: PaperStyle,
                        *, value_column: str = "freq_per_year", duration_class: str | None = None,
                        ylabel="", headroom: float = 1.08):
    """Draw reference and warmest-level medians per zone and scenario as grouped bars.

    Each zone has one touching pair of bars per scenario, in the scenario colour
    (``SCENARIO_COLORS``): the reference level pale, the warmest level solid (``LEVEL_ALPHA``).
    Medians across chains only, without spread; missing values are drawn as zero.

    Parameters
    ----------
    ax : Axes
        Target axes.
    by_scenario : mapping of str to DataFrame
        See :func:`panel_dumbbell`.
    style : PaperStyle
        Zones and warming levels.
    value_column : str, default "freq_per_year"
        Quantity drawn.
    duration_class : str, optional
        Keep only this duration class.
    ylabel : str, default ""
        Y-axis label.
    headroom : float, default 1.08
        Factor applied to the upper y limit.
    """
    scenarios = list(by_scenario)
    positions = np.arange(len(style.zones))
    bar_width = 0.17
    pair_gap = 0.09                              # between the scenario pairs of a zone
    pair_centres = [(k - (len(scenarios) - 1) / 2) * (2 * bar_width + pair_gap)
                    for k in range(len(scenarios))]

    for k, (label, table) in enumerate(by_scenario.items()):
        color = SCENARIO_COLORS[k % len(SCENARIO_COLORS)]
        data = table if duration_class is None else table[table.duration_class == duration_class]
        for j, period in enumerate((style.reference_period, style.warmest_period)):
            values = np.array([data[(data.zone == zone) & (data.period == period)][value_column]
                               .median() for zone in style.zones])
            x = positions + pair_centres[k] + (j - 0.5) * bar_width
            ax.bar(x, np.nan_to_num(values), bar_width, color=blend(color, LEVEL_ALPHA[j]),
                   edgecolor=color, linewidth=0.4, zorder=2,
                   label=label if j == 1 else None)

    ax.set_ylim(0, ax.get_ylim()[1] * headroom)
    ax.set_xticks(positions)
    ax.set_xticklabels(style.labels(), fontsize=FS["tick"], rotation=30, ha="right")
    ax.set_xlim(-0.6, len(style.zones) - 0.4)
    ax.set_ylabel(ylabel, fontsize=FS["label"])
    ax.tick_params(labelsize=FS["tick"])
    ax.grid(True, axis="y", color="0.92", lw=0.3)
    ax.grid(False, axis="x")
    ax.set_axisbelow(True)


def level_handles(style: PaperStyle, color="0.35"):
    """Return legend entries for the pale/solid warming levels of :func:`panel_scenario_bars`.

    Parameters
    ----------
    style : PaperStyle
        Supplies the reference and warmest level names.
    color : color, default "0.35"
        Neutral colour the two patches are drawn in.

    Returns
    -------
    handles : list of Patch
    labels : list of str
    """
    return ([mpatches.Patch(facecolor=blend(color, alpha), edgecolor=color, linewidth=0.4)
             for alpha in LEVEL_ALPHA],
            [style.reference_period, style.warmest_period])


# ══════════════════════════════════════════════════════════════════════════════════════════
# Shared legend handles
# ══════════════════════════════════════════════════════════════════════════════════════════
def gwl_handles(style: PaperStyle):
    """Return legend entries for the warming levels.

    Parameters
    ----------
    style : PaperStyle
        Warming levels and their colours.

    Returns
    -------
    handles : list of Line2D
        Square markers.
    labels : list of str
        Warming-level names.
    """
    return ([plt.Line2D([], [], marker="s", ls="", ms=3.5, color=style.gwl_colors[period])
             for period in style.period_names], list(style.period_names))


def storage_handles(style: PaperStyle):
    """Return legend entries for the storage components (``STORAGE_PARTS``).

    Parameters
    ----------
    style : PaperStyle
        Unused; kept for a uniform signature with the other handle builders.

    Returns
    -------
    handles : list of Rectangle
    labels : list of str
    """
    return ([plt.Rectangle((0, 0), 1, 1, color=color) for _, _, color in STORAGE_PARTS],
            [label for _, label, _ in STORAGE_PARTS])


def driver_handles(style: PaperStyle):
    """Return legend entries for the attribution drivers and the net change marker.

    Parameters
    ----------
    style : PaperStyle
        Drivers and their colours.

    Returns
    -------
    handles : list of Artist
        One patch per driver, then a black diamond.
    labels : list of str
        Driver names, then "net Δ".
    """
    handles = [plt.Rectangle((0, 0), 1, 1, color=style.driver_palette[driver])
               for driver in style.drivers]
    handles.append(plt.Line2D([], [], marker="D", ls="", ms=2.5, color="black"))
    return handles, list(style.drivers) + ["net \u0394"]


# ══════════════════════════════════════════════════════════════════════════════════════════
# Climate-model characterisation
# ══════════════════════════════════════════════════════════════════════════════════════════
#: The GCMs driving the chains, in legend and drawing order.
DRIVING_GCM_ORDER = ["CNRM-CM5", "EC-EARTH", "HadGEM2-ES", "MPI-ESM-LR", "NorESM1-M"]
#: Colours of the driving GCMs (Okabe-Ito, colourblind-safe, unordered).
DRIVING_GCM_COLORS = {"CNRM-CM5": "#0072B2", "EC-EARTH": "#009E73", "HadGEM2-ES": "#D55E00",
                      "MPI-ESM-LR": "#CC79A7", "NorESM1-M": "#E69F00"}
#: Marker of each RCM.
RCM_MARKERS = {"CNRM-ALADIN63": "o", "MOHC-HadREM3-GA7-05": "^"}


def panel_gcm_gmst(ax, gmst, *, driving, warming_levels, level_colors) -> None:
    """Draw the global-mean temperature anomaly of every CMIP5 model, driving GCMs highlighted.

    Other models are thin grey lines underneath; driving GCMs are coloured
    (``DRIVING_GCM_COLORS``); warming levels are dashed horizontal lines.

    Parameters
    ----------
    ax : Axes
        Target axes.
    gmst : DataArray
        Anomaly (K, relative to 1861-1900) with dimensions ``model`` and ``year``.
    driving : mapping of str to str
        Driving GCM name to its ``model`` key in *gmst*.
    warming_levels : sequence of float
        Warming levels (K).
    level_colors : mapping of float to color
        Colour of each warming level.
    """
    years = gmst["year"].values
    background = [m for m in gmst["model"].values if m not in set(driving.values())]
    for model in background:
        ax.plot(years, gmst.sel(model=model).values, lw=0.35, color="0.72", alpha=0.55, zorder=1)
    for gcm in DRIVING_GCM_ORDER:
        if gcm not in driving:
            continue
        ax.plot(years, gmst.sel(model=driving[gcm]).values, lw=1.1,
                color=DRIVING_GCM_COLORS[gcm], label=gcm, zorder=3)
    for level in warming_levels:
        ax.axhline(level, color=level_colors[level], lw=0.6, ls="--", zorder=2)
        ax.annotate(f"+{level:.1f} °C", xy=(years[0], level), xytext=(2, 1.5),
                    textcoords="offset points", fontsize=FS["tick"], color=level_colors[level],
                    ha="left", va="bottom", fontweight="bold")
    ax.set_xlim(years[0], years[-1])
    ax.set_xlabel("year", fontsize=FS["label"])
    ax.set_ylabel("GSAT anomaly vs 1861–1900 (K)", fontsize=FS["label"])
    ax.tick_params(labelsize=FS["tick"])
    ax.grid(color="0.93", lw=0.3)
    ax.set_axisbelow(True)


def gcm_warming_figure(gmst, crossing, *, driving, warming_levels, level_colors,
                       height_mm: float = 44) -> Figure:
    """Draw the driving GCMs within the CMIP5 ensemble: warming curves and level-crossing years.

    Panel a: :func:`panel_gcm_gmst`. Panel b: per warming level, the first year of every run's
    20-year window as a grey dot (seeded vertical jitter), each driving GCM as a coloured tick and
    the ensemble median as a dashed tick.

    Parameters
    ----------
    gmst : DataArray
        Anomaly with dimensions ``model`` and ``year``.
    crossing : DataFrame
        Columns ``model, ensemble, warming_level, start_year`` for the whole ensemble.
    driving : mapping of str to str
        Driving GCM name to its ``model`` key in *gmst* and *crossing*.
    warming_levels : sequence of float
        Warming levels (K), one row of panel b each.
    level_colors : mapping of float to color
        Colour of each warming level.
    height_mm : float, default 44
        Initial figure height.

    Returns
    -------
    Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, height_mm * MM),
                             gridspec_kw={"wspace": 0.26, "width_ratios": (1.12, 0.88)})
    ax_a, ax_b = axes
    panel_gcm_gmst(ax_a, gmst, driving=driving, warming_levels=warming_levels,
                   level_colors=level_colors)

    random_state = np.random.RandomState(0)
    for row, level in enumerate(warming_levels):
        block = crossing[crossing["warming_level"] == level]
        others = block[~block["model"].isin(driving)]
        ax_b.scatter(others["start_year"], row + random_state.uniform(-0.16, 0.16, len(others)),
                     s=4, color="0.72", alpha=0.7, lw=0, zorder=1)
        for gcm in DRIVING_GCM_ORDER:
            hit = block[block["model"] == gcm]
            if hit.empty:
                continue
            ax_b.plot([float(hit["start_year"].iloc[0])] * 2, [row - 0.30, row + 0.30],
                      lw=1.3, color=DRIVING_GCM_COLORS[gcm], solid_capstyle="butt", zorder=3)
        median = float(block["start_year"].median())
        ax_b.plot([median] * 2, [row - 0.38, row + 0.38], lw=0.8, color="0.25", ls="--", zorder=2)
    ax_b.set_yticks(range(len(warming_levels)))
    ax_b.set_yticklabels([f"+{level:.1f} °C" for level in warming_levels], fontsize=FS["label"])
    ax_b.set_ylim(len(warming_levels) - 0.55, -0.45)
    ax_b.set_xlabel("first year of the 20-year window", fontsize=FS["label"])
    ax_b.tick_params(labelsize=FS["tick"])
    ax_b.grid(True, axis="x", color="0.93", lw=0.3)
    ax_b.set_axisbelow(True)

    fit_width(fig)
    handles = [plt.Line2D([], [], color="0.72", lw=1.0, label="other CMIP5 models")]
    handles += [plt.Line2D([], [], color=DRIVING_GCM_COLORS[gcm], lw=1.4, label=gcm)
                for gcm in DRIVING_GCM_ORDER if gcm in driving]
    legend_below(fig, list(axes), handles, [handle.get_label() for handle in handles], ncol=6)
    panel_letters(fig, [("a", ax_a), ("b", ax_b)])
    return fig


def model_characterisation_figure(table, *, rcm_labels, warming_levels, show_drivers: bool,
                                  height_mm: float = 56, width_mm: float = 110,
                                  y_column: str = "pr_mm_per_day",
                                  y_label: str = "change in precipitation\n(mm d$^{-1}$)",
                                  show_gcm: bool = True,
                                  quadrant_labels: tuple[str, str] = ("wetter", "drier")) -> Figure:
    """Draw each model's change in European temperature against a second variable.

    The change is taken within each model between the lowest and highest of *warming_levels*.
    Chains (``source == "rcm"``) are coloured by driving GCM with one marker per RCM; CMIP5 GCMs
    (``source == "gcm"``) are a grey cloud. Axis limits follow the data.

    Parameters
    ----------
    table : DataFrame
        Columns ``source, model, gcm, rcm, warming_level, tas_degC, <y_column>`` and, for GCM
        rows, ``driving``.
    rcm_labels : mapping of str to str
        Legend label of each RCM in ``RCM_MARKERS``.
    warming_levels : sequence of float
        Warming levels; the lowest and highest are compared.
    show_drivers : bool
        Draw the driving GCMs as hollow circles in their colour instead of within the grey cloud.
    height_mm, width_mm : float, default 56, 110
        Figure size; the width is the target of :func:`fit_width`.
    y_column : str, default "pr_mm_per_day"
        Variable on the y axis.
    y_label : str
        Y-axis label.
    show_gcm : bool, default True
        Draw the CMIP5 GCMs (and the vertical zero line); disable for variables the GCM
        ensemble does not fully cover.
    quadrant_labels : tuple of str, default ("wetter", "drier")
        Labels of the upper and lower half, in the right-hand corners.

    Returns
    -------
    Figure
    """
    fig, ax = plt.subplots(figsize=(width_mm * MM, height_mm * MM))
    first, last = min(warming_levels), max(warming_levels)

    def delta(block):
        """Return per-model changes ``dT`` and ``dY`` between the first and last level."""
        wide = block.pivot_table(index="model", columns="warming_level",
                                 values=["tas_degC", y_column])
        return pd.DataFrame({"dT": wide[("tas_degC", last)] - wide[("tas_degC", first)],
                             "dY": wide[(y_column, last)] - wide[(y_column, first)]})

    gcm = table[table["source"] == "gcm"]
    rcm = table[table["source"] == "rcm"]
    rcm_delta = delta(rcm)
    gcm_delta = delta(gcm) if show_gcm else None
    driving = (set(gcm[gcm["driving"].fillna(False)]["model"])
               if show_gcm and "driving" in gcm else set())

    ax.axhline(0.0, color="0.45", lw=0.7, zorder=1)
    # The vertical zero line only with the GCM cloud, which reaches near zero; the chains alone
    # sit far from it and the line would stretch the axis.
    if show_gcm:
        ax.axvline(0.0, color="0.45", lw=0.7, zorder=1)

    if show_gcm:
        background = gcm_delta.drop(index=[m for m in driving if m in gcm_delta.index]) \
            if show_drivers else gcm_delta
        ax.plot(background["dT"], background["dY"], "o", ms=2.6, color="0.70", mec="none",
                alpha=0.75, zorder=2)

        if show_drivers:
            for model in sorted(driving & set(gcm_delta.index)):
                row = gcm[gcm["model"] == model].iloc[0]
                ax.plot(gcm_delta.loc[model, "dT"], gcm_delta.loc[model, "dY"], "o", ms=5.0,
                        mfc="none", mec=DRIVING_GCM_COLORS[row["gcm"]], mew=1.2, zorder=3)

    for model, row in rcm_delta.iterrows():
        info = rcm[rcm["model"] == model].iloc[0]
        ax.plot(row["dT"], row["dY"], RCM_MARKERS.get(info["rcm"], "o"), ms=4.4,
                color=DRIVING_GCM_COLORS[info["gcm"]], mec="white", mew=0.5, zorder=4)

    for label, (x_fraction, y_fraction), vertical_alignment in (
            (quadrant_labels[0], (0.985, 0.97), "top"),
            (quadrant_labels[1], (0.985, 0.03), "bottom")):
        ax.annotate(label, xy=(x_fraction, y_fraction), xycoords="axes fraction", ha="right",
                    va=vertical_alignment, fontsize=FS["tick"], color="0.45", style="italic")

    ax.set_xlabel(f"change in temperature, {first} to {last} (K)", fontsize=FS["label"])
    ax.set_ylabel(y_label, fontsize=FS["label"])
    ax.tick_params(labelsize=FS["tick"])
    ax.grid(color="0.93", lw=0.3)
    ax.set_axisbelow(True)

    fit_width(fig, target_mm=width_mm)
    handles = [plt.Line2D([], [], color="0.70", marker="o", ls="none", ms=3,
                          label="CMIP5 GCMs")] if show_gcm else []
    if show_drivers and show_gcm:
        handles.append(plt.Line2D([], [], color="0.35", marker="o", ls="none", ms=5,
                                  mfc="none", mew=1.2, label="driving GCM"))
    handles += [plt.Line2D([], [], color="0.45", marker=marker, ls="none", ms=4,
                           label=rcm_labels.get(rcm_name, rcm_name))
                for rcm_name, marker in RCM_MARKERS.items()]
    handles += [plt.Line2D([], [], color=DRIVING_GCM_COLORS[gcm_name], marker="s", ls="none",
                           ms=3.4, label=gcm_name) for gcm_name in DRIVING_GCM_ORDER]
    # ncol=5 keeps the marker-type entries apart from the GCM colours (legends fill column-major).
    legend_below(fig, ax, handles, [handle.get_label() for handle in handles], ncol=5)
    return fig


# ══════════════════════════════════════════════════════════════════════════════════════════
# Europe-wide change: table and figure
# ══════════════════════════════════════════════════════════════════════════════════════════
#: Colours per quantity category as ``(component box, category-total box, median line)``:
#: demand orange, supply blue, residual load ("net") grey.
CATEGORY_COLOR = {"demand": ("#fdb863", "#e08214", "#7f3b08"),
                  "supply": ("#92c5de", "#4393c3", "#053061"),
                  "net": ("#d9d9d9", "#969696", "#252525")}
#: Category of each quantity and whether it is a component (0) or a category total (1).
QUANTITY_CATEGORY = {"Heating demand": ("demand", 0), "Cooling demand": ("demand", 0),
                     "Total demand": ("demand", 1), "Solar": ("supply", 0), "Wind": ("supply", 0),
                     "Hydro": ("supply", 0), "Total supply": ("supply", 1),
                     "Residual load": ("net", 1)}
#: Box and median-line colour of each quantity.
QUANTITY_COLOR = {quantity: CATEGORY_COLOR[category][kind]
                  for quantity, (category, kind) in QUANTITY_CATEGORY.items()}
QUANTITY_MEDIAN = {quantity: CATEGORY_COLOR[category][2]
                   for quantity, (category, _) in QUANTITY_CATEGORY.items()}
#: Styles of the min-max extensions in :func:`europe_change_panel`: "outline" (white, edged in
#: the box colour) or "tint" (a paler shade of the box colour).
EUROPE_EXTENSIONS = ("outline", "tint")


def europe_change_table(rows, columns, reference: str, warmest: str, *,
                        members=None, dimension: str = "country") -> dict:
    """Compute the per-chain Europe-wide change of every quantity and time scale.

    Parameters
    ----------
    rows : sequence of (str, callable)
        ``(label, builder)`` with ``builder(period, months)`` returning a per-chain field; every
        dimension other than ``model`` is summed.
    columns : mapping of str to sequence of int or None
        Time scale name to its months (``None`` for the whole year).
    reference, warmest : str
        Warming levels compared.
    members : iterable of str, optional
        Restrict every field to these entries of *dimension* before summing (e.g. a balancing
        zone); entries a field lacks are skipped.
    dimension : str, default "country"
        Dimension *members* refers to.

    Returns
    -------
    dict
        ``{(column, label): DataFrame}`` indexed by chain (chains present at both levels), with
        columns ``ref`` (reference total), ``twh`` (warmest minus reference) and ``pct``
        (``100 * twh / |ref|``, so the sign follows the change even for a negative reference).
    """
    out = {}
    for column, months in columns.items():
        for label, builder in rows:
            fields = {}
            for period in (reference, warmest):
                field = builder(period, months)
                if members is not None:
                    present = [member for member in members
                               if member in set(map(str, field[dimension].values))]
                    field = field.sel({dimension: present})
                fields[period] = field.sum([name for name in field.dims if name != "model"],
                                           skipna=True)
            shared = sorted(set(map(str, fields[reference]["model"].values))
                            & set(map(str, fields[warmest]["model"].values)))
            reference_total = fields[reference].sel(model=shared).to_series()
            change = fields[warmest].sel(model=shared).to_series() - reference_total
            out[column, label] = pd.DataFrame({"ref": reference_total, "twh": change,
                                               "pct": 100 * change / reference_total.abs()})
    return out


#: Statistics columns of :func:`europe_change_summary`, in order.
EUROPE_SUMMARY_COLUMNS = ("median", "P25", "P75", "min", "max", "sign agreement")


def europe_change_summary(change: dict, labels, columns, *, value: str = "twh",
                          negligible: float = 0.05) -> dict:
    """Summarise the per-chain changes of :func:`europe_change_table` across chains.

    Parameters
    ----------
    change : dict
        Output of :func:`europe_change_table`.
    labels : sequence of str
        Quantities, one row each.
    columns : iterable of str
        Time scales, one table each.
    value : {"twh", "pct"}, default "twh"
        Column summarised.
    negligible : float, default 0.05
        Below this absolute median the sign agreement is reported as a dash.

    Returns
    -------
    dict of str to DataFrame
        Per time scale, one row per quantity with the columns of ``EUROPE_SUMMARY_COLUMNS``:
        median, quartiles, min, max and the sign agreement as ``"agreeing/total"``.
    """
    out = {}
    for column in columns:
        rows = {}
        for label in labels:
            series = change[column, label][value]
            median = series.median()
            agree = int((np.sign(series) == np.sign(median)).sum())
            agreement = "\u2013" if abs(median) < negligible else f"{agree}/{len(series)}"
            rows[label] = {"median": median, "P25": series.quantile(0.25),
                           "P75": series.quantile(0.75), "min": series.min(), "max": series.max(),
                           "sign agreement": agreement}
        out[column] = pd.DataFrame.from_dict(rows, orient="index")[list(EUROPE_SUMMARY_COLUMNS)]
    return out


def markdown_table(frame: pd.DataFrame, *, decimals: int = 1, index_label: str = "") -> str:
    """Format a DataFrame as a GitHub/Pandoc pipe table.

    Numbers are written signed, with a true minus sign (zero unsigned); other values as text.

    Parameters
    ----------
    frame : DataFrame
        Table to format.
    decimals : int, default 1
        Decimal places of numbers.
    index_label : str, default ""
        Header of a single-level index column.

    Returns
    -------
    str
        The table. A MultiIndex gives one column per level, with a repeated value blanked
        below its first row.
    """
    def cell(value):
        """Format one table cell."""
        if isinstance(value, (int, float, np.floating, np.integer)) and not isinstance(value, bool):
            text = f"{value:+.{decimals}f}"
            return "0.0" if float(text) == 0 else text.replace("-", "\u2212")
        return str(value)

    levels = frame.index.nlevels
    names = list(frame.index.names) if levels > 1 else [index_label]
    header = [name or "" for name in names] + [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join([":---"] * levels + ["---:"] * len(frame.columns)) + "|"]
    previous = (None,) * levels
    for key, row in frame.iterrows():
        key = key if isinstance(key, tuple) else (key,)
        shown = [str(part) if part != previous[i] or any(key[j] != previous[j] for j in range(i))
                 else "" for i, part in enumerate(key)]
        previous = key
        lines.append("| " + " | ".join(shown + [cell(v) for v in row.values]) + " |")
    return "\n".join(lines)


def _europe_change_spans(change, labels, columns, value, share_seasons=True):
    """Return x limits per time scale covering every chain and zero.

    Parameters
    ----------
    change : dict
        Output of :func:`europe_change_table`.
    labels : sequence of str
        Quantities.
    columns : sequence of str
        Time scales.
    value : {"twh", "pct"}
        Column drawn.
    share_seasons : bool, default True
        In TWh, give all time scales other than ``"Annual"`` one joint span.

    Returns
    -------
    dict of str to (float, float)
    """
    spans = {}
    for column in columns:
        low = min(change[column, label][value].min() for label in labels)
        high = max(change[column, label][value].max() for label in labels)
        spans[column] = (min(low, 0.0), max(high, 0.0))
    seasonal = [column for column in columns if column != "Annual"]
    if share_seasons and value == "twh" and len(seasonal) > 1:
        joint = (min(spans[column][0] for column in seasonal),
                 max(spans[column][1] for column in seasonal))
        spans.update({column: joint for column in seasonal})
    return spans


def europe_change_panel(ax, change, labels, column, *, value="twh", span=None,
                        extension="tint", height=0.34):
    """Draw one time scale of the Europe-wide change as horizontal boxes, one per quantity.

    Each box spans the interquartile range across chains, extended to the minimum and maximum,
    with the median as a line; colours follow ``QUANTITY_COLOR`` and ``QUANTITY_MEDIAN``.

    Parameters
    ----------
    ax : Axes
        Target axes.
    change : dict
        Output of :func:`europe_change_table`.
    labels : sequence of str
        Quantities, top to bottom.
    column : str
        Time scale drawn.
    value : {"twh", "pct"}, default "twh"
        Column drawn.
    span : tuple of float, optional
        X range to show, padded by 8 % on each side.
    extension : {"tint", "outline"}, default "tint"
        Style of the min-max extensions, see ``EUROPE_EXTENSIONS``.
    height : float, default 0.34
        Box height, in row units.

    Raises
    ------
    ValueError
        If *extension* is not in ``EUROPE_EXTENSIONS``.
    """
    if extension not in EUROPE_EXTENSIONS:
        raise ValueError(f"extension={extension!r}: expected one of {EUROPE_EXTENSIONS}")
    positions = np.arange(len(labels))[::-1]
    for y, label in zip(positions, labels):
        series = change[column, label][value]
        minimum, q25, median, q75, maximum = (series.min(), series.quantile(0.25),
                                              series.median(), series.quantile(0.75),
                                              series.max())
        color = QUANTITY_COLOR.get(label, "0.6")
        outline = extension == "outline"
        for left, right in ((minimum, q25), (q75, maximum)):
            ax.barh(y, right - left, left=left, height=height, zorder=1,
                    facecolor="white" if outline else blend(color, 0.35),
                    edgecolor=color if outline else "none", linewidth=0.8 if outline else 0)
        ax.barh(y, q75 - q25, left=q25, height=height, facecolor=color, zorder=2,
                edgecolor=color if outline else "none", linewidth=0.8 if outline else 0)
        ax.plot([median, median], [y - height / 2, y + height / 2],
                color=QUANTITY_MEDIAN.get(label, "0.2"), lw=1.4, solid_capstyle="butt", zorder=3)
    ax.axvline(0, color="0.3", lw=0.5, zorder=0)
    for name in ("Total demand", "Total supply"):
        if name in labels:
            ax.axhline(positions[labels.index(name)] - 0.5, color="0.80", lw=0.4, zorder=0)
    if span is not None:
        pad = 0.08 * (span[1] - span[0])
        ax.set_xlim(span[0] - pad, span[1] + pad)
    ax.set_yticks(positions)
    ax.set_ylim(-0.6, len(labels) - 0.4)
    ax.tick_params(labelsize=FS["tick"])
    ax.grid(True, axis="x", color="0.92", lw=0.3)
    ax.grid(False, axis="y")
    ax.set_axisbelow(True)


def europe_change_handles(extension="tint"):
    """Return legend entries for :func:`europe_change_panel`.

    The demand and supply category colours, then the interquartile range, min-max extension and
    median in neutral grey. The residual load has no entry (its grey would read as a statistic).

    Parameters
    ----------
    extension : {"tint", "outline"}, default "tint"
        Style of the min-max extension entry.

    Returns
    -------
    handles : list of Artist
    labels : list of str
    """
    grey = "0.60"
    handles = [mpatches.Patch(facecolor=CATEGORY_COLOR["demand"][1], edgecolor="none"),
               mpatches.Patch(facecolor=CATEGORY_COLOR["supply"][1], edgecolor="none"),
               mpatches.Patch(facecolor=grey, edgecolor=grey, linewidth=0.8),
               (mpatches.Patch(facecolor="white", edgecolor=grey, linewidth=0.8)
                if extension == "outline" else
                mpatches.Patch(facecolor=blend(grey, 0.35), edgecolor="none")),
               plt.Line2D([], [], color="0.15", lw=1.4)]
    labels = ["demand (total darker)", "supply (total darker)",
              "interquartile range", "min to max", "median"]
    return handles, labels


def europe_change_figure(change: dict, labels, columns, *, value: str = "twh",
                         height_mm: float = 66, extension: str = "tint",
                         bar_height: float = 0.34) -> Figure:
    """Draw the Europe-wide change: one panel per time scale, one row per quantity.

    Panels are :func:`europe_change_panel`, lettered a, b, c. X limits cover every chain; in TWh
    the seasonal panels share one range and ``"Annual"`` keeps its own.

    Parameters
    ----------
    change : dict
        Output of :func:`europe_change_table`.
    labels : sequence of str
        Quantities, top to bottom.
    columns : iterable of str
        Time scales, one panel each (season keys are shown by name).
    value : {"twh", "pct"}, default "twh"
        Column drawn; the unit is TWh per year for ``"Annual"``, TWh per season otherwise.
    height_mm : float, default 66
        Initial figure height.
    extension : {"tint", "outline"}, default "tint"
        Style of the min-max extensions.
    bar_height : float, default 0.34
        Box height, in row units.

    Returns
    -------
    Figure
    """
    columns = list(columns)
    fig, axes = plt.subplots(1, len(columns), figsize=(WIDTH, height_mm * MM), sharey=True,
                             gridspec_kw={"wspace": 0.10})
    spans = _europe_change_spans(change, labels, columns, value)
    for ax, column in zip(axes, columns):
        europe_change_panel(ax, change, labels, column, value=value, span=spans[column],
                            extension=extension, height=bar_height)
        unit = ("%" if value == "pct" else
                "TWh yr$^{-1}$" if column == "Annual" else "TWh season$^{-1}$")
        ax.set_title(season_label(column), fontsize=FS["title"], fontweight="bold")
        ax.set_xlabel(f"change ({unit})", fontsize=FS["label"])
    axes[0].set_yticklabels(labels, fontsize=FS["tick"])
    fit_width(fig)
    legend_below(fig, list(axes), *europe_change_handles(extension), ncol=3)
    panel_letters(fig, list(zip("abc", axes)))
    return fig


#: Colour of each balancing zone; categorical, distinct from the warming-level colours.
ZONE_COLORS = {
    "Iberian_Peninsula": "#E6AB02",
    "South_East": "#D95F02",
    "Central_Western": "#7570B3",
    "Central_Eastern": "#A6761D",
    "Nordic": "#1B9E77",
    "British_Isles": "#E7298A",
}


#: Colours of the TYNDP scenario technologies; generation technologies match the driver palette.
TECH_COLORS = {
    "solar PV": "#E6AB02",
    "wind onshore": "#7570B3",
    "wind offshore": "#1B7C3D",
    "battery (prosumer)": "#C79A6B",
    "battery (utility)": "#8B5A2B",
}
