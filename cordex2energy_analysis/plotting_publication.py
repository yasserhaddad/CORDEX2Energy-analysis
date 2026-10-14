"""Publication-grade rendering of the paper's five main figures.

This module exists because the working figures in ``paper_main_figures_v2.ipynb`` are sized for a
screen, not a page: every figsize there is a hardcoded inch tuple 2.0-2.4x the 180 mm journal
ceiling, so their 6.8-13 pt type renders at ~3-5 pt once scaled down.  Rebuilding at the target
width is the only honest fix — you cannot shrink your way to a legible figure.

Three invariants are enforced here rather than hoped for, because each is silent when it breaks:

* **width <= 180 mm**, measured on the *tight* bounding box.  ``bbox_inches="tight"`` EXPANDS the
  canvas to include artists drawn outside the axes (the sign-convention arrows, the figure-level
  legends), so checking ``fig.get_figwidth()`` would pass a figure that is saved too wide.
* **every glyph between 5 and 7 pt.**
* **the file parses** once written.

Output is **PDF + PNG**.  PDF over EPS on measurement, not taste: on this paper's own Natural Earth
geometry PDF came out 2.7-5.3x smaller (Figure 1: 3.7 MB vs 10.4 MB) because matplotlib writes EPS
as uncompressed ASCII PostScript while PDF Flate-compresses and subsets fonts.  The decisive point
is not size though — **EPS cannot represent transparency**.  Season is encoded *as alpha* in the
driver-attribution and multi-day-season panels, so an EPS export would collapse four seasons into
one shade without erroring.  PDF renders alpha natively, so those encodings survive untouched.

Every builder returns a :class:`~matplotlib.figure.Figure` and sets **no figure title** — the
caption lives in the notebook markdown, ready to paste into the manuscript.  Panel titles stay.
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

from .plotting_extremes import model_spread_bounds
from .plotting_paper import (
    PaperStyle,
    _annotate_sign_convention,
    _baseline_relevance_mask,
    _binned_norm,
    _change_color,
    _hatch_disagreeing_countries,
    _symmetric_limit,
    _zero_class,
    MAP_CRS,
    MAP_EXTENT_LAEA,
    is_projected,
    map_extent,
    plot_country_choropleth,
    set_map_aspect,
)

__all__ = [
    "MM", "WIDTH", "COL", "FONT_RANGE",
    "publication_rc", "blend", "panel_letter", "save", "CBAR_PAD", "align_colorbar_labels",
    "MapRow", "gwl_maps", "mean_state_maps", "sign_agreement", "survival_axes",
    "tail_exceedance_figure", "panel_zone_duration_hours", "panel_season_contribution",
    "season_contribution_by_model", "SEASON_CONTRIBUTION_BINS",
    "SEASON_COUNT_BINS_SUB_DAILY", "SEASON_COUNT_BINS_MULTI_DAY",
    "panel_event_frequency", "panel_hour_of_day",
    "panel_storage_stack", "panel_driver_attribution", "panel_dumbbell",
    "panel_scenario_bars", "level_handles", "LEVEL_ALPHA",
    "regime_figure", "zone_tail_duration_figure",
    "panel_binned_attenuation", "panel_letters", "fit_width", "legend_below", "FS",
    "SEASON_NAME", "season_label", "GWL_TITLE",
    "QUANTITY_CMAP", "HYDRO_CMAP", "QUANTITY_COLOR", "europe_change_table",
    "europe_change_figure", "europe_change_panel", "europe_change_handles",
    "CATEGORY_COLOR", "QUANTITY_MEDIAN", "EUROPE_EXTENSIONS",
    "europe_change_summary", "markdown_table", "EUROPE_SUMMARY_COLUMNS",
    "gwl_handles", "driver_handles", "storage_handles", "SCENARIO_COLORS", "paired_change_pct",
]

MM = 1 / 25.4
WIDTH = 180 * MM            #: double-column / maximum figure width (in)
COL = 89 * MM               #: single-column width (in), for reference
FONT_RANGE = (7.0, 8.0)     #: the permitted type-size band

#: Type sizes by ROLE, the only place a size is chosen. Raised from Nature's 5-7 pt after internal
#: review found the figures unreadable at print size: every tick, legend and annotation is now 7 pt,
#: axis labels 7.5, panel titles and letters 8. Literal sizes elsewhere would drift from this.
FS = {"tick": 7.0, "label": 7.5, "title": 8.0}

#: Season names for DISPLAY only. The acronyms are also the data keys (``.season == "DJF"``,
#: ``season_months[season]``, ``"{season}_{driver}"`` columns), so they are translated at the point
#: of drawing and never renamed in the data. Captions give the acronym once in parentheses.
SEASON_NAME = {"DJF": "Winter", "MAM": "Spring", "JJA": "Summer", "SON": "Autumn"}


#: Reference-level (+1.5 °C) ramps for the rows whose natural colour collided with the change
#: columns. Every change column is RdBu_r, so a Blues ramp (hydro AND heating used it) or a YlOrRd
#: one (cooling) sat right next to a colorbar of the same hue meaning something else. These are
#: moved off the red-blue axis: heating purple, cooling ochre-brown, hydro teal -- the latter
#: close to its old blue, as asked, but no longer a shade the change bar also uses.
HYDRO_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "hydro_teal", ["#f2f9f7", "#a6dbd0", "#5ab4ac", "#1f8279", "#01524a"])
QUANTITY_CMAP = {"Hydro": HYDRO_CMAP, "Heating demand": "Purples",
                 "Cooling demand": "YlOrBr", "Residual load": "BrBG_r"}

#: The sign-convention labels beside the driver panels' arrows.
SIGN_LABELS = ("deepens", "eases")

#: Title of every warming-level key, so a reader never has to infer what "+1.5 °C" refers to.
GWL_TITLE = "Global warming level"


def season_label(season) -> str:
    """Display name of a season key; anything that is not a season key passes through unchanged."""
    return SEASON_NAME.get(str(season), str(season))

#: Gap between a map and its colorbar, as a fraction of the parent axes.  One constant so the
#: reference bar and the change bar sit the same distance out; 0.05 is matplotlib's own
#: constrained-layout default, which the reference bar was already getting implicitly.
CBAR_PAD = 0.05
#: Ticks drawn INSIDE a reference (column 0) colorbar: 4 on a sequential scale, 5 on a diverging one
#: so the zero is marked.  See :func:`_set_reference_ticks` for why a plain locator will not do it.
CBAR_TICKS_SEQUENTIAL = 4
CBAR_TICKS_DIVERGING = 5

#: Storage components.  Reservoir first because it is the larger share in every zone.
STORAGE_PARTS = (("reservoir_attenuation_share_pct", "reservoir", "#08519C"),
                 ("battery_attenuation_share_pct", "battery", "#8B5A2B"))

#: Build-out colours for Figure 5.  Deliberately NOT the GWL ramp: blue/amber/red mean "warming
#: level" in every other figure of this paper, and reusing them for "scenario" here is exactly the
#: collision that made the old S20 hard to read.
SCENARIO_COLORS = ("#117733", "#882255", "#44AA99")

_RC = {
    "font.family": "sans-serif",
    # DejaVu Sans throughout.  It is matplotlib's own default, is always present, and has by far
    # the widest glyph coverage of the sans faces on this machine (Greek, the superscript minus and
    # the multiplication sign all resolve from it).  Note it sets wider than Helvetica at the same
    # point size, so the dense 5-7 pt panels have less slack than they did on Nimbus Sans.
    "font.sans-serif": ["DejaVu Sans", "Nimbus Sans", "Liberation Sans"],
    "font.size": 7.0,
    "axes.labelsize": 7.5,
    "axes.titlesize": 8.0,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
    "legend.fontsize": 7.0,
    "figure.titlesize": 8.0,
    "legend.title_fontsize": 7.0,
    # Type 42 = TrueType, embedded and subset.  The default (3) is a Type 3 bitmap-ish font that
    # most production systems reject.
    "pdf.fonttype": 42,
    "pdf.compression": 6,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    # Point mathtext at the SAME family rather than a stix fallback: with "stixsans" the only
    # mathtext in the paper (the yr^-1 superscript on Figure 1's colorbars) embedded a second
    # typeface for one superscript.  DejaVu Sans carries Greek, so Delta resolves from it too.
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
    """Apply the publication rc block for the duration of the block, then restore.

    A context manager rather than a global setting on purpose: ``plotting.py`` calls
    ``sns.set_style("whitegrid")`` at **import time**, so anything that imports the sibling plotting
    modules silently re-styles the session.  Scoping the rc means a figure built inside this block
    cannot be contaminated by an import that happens to run later in the notebook.
    """
    with mpl.rc_context({**_RC, **overrides}):
        yield


def blend(color, alpha: float, bg="white") -> tuple:
    """The solid RGB equal to *color* drawn at *alpha* over *bg*.

    PDF renders transparency natively, so this is a readability tool, not a correctness one: use it
    where overlapping translucent fills would darken ambiguously (stacked bars), and leave genuine
    alpha alone where the overlap is the point (uncertainty bands).
    """
    rgb = np.asarray(mcolors.to_rgb(color))
    back = np.asarray(mcolors.to_rgb(bg))
    return tuple(alpha * rgb + (1 - alpha) * back)


#: THE panel-letter rule, for every figure in the paper:
#: * 8 pt bold lowercase (the title size);
#: * horizontally at the left edge of the panel's ink -- its y-label, tick labels and any
#:   sign-convention arrows -- with every letter in a COLUMN on that column's leftmost edge;
#: * vertically with the letter's TOP at the top of the panel's ink -- the title if it has one, the
#:   axes (or its offset text) if not -- with every letter in a ROW on that row's highest top; a
#:   y-label tall enough to reach that corner lifts its row until the letters clear it.
#: The one exception is ``inside=True`` for map rows: equal-area maps carry empty sky above 70 N, so
#: a letter above the box floats in the gap between rows; there it goes inside the map's top-left
#: corner (empty Atlantic), clear of the rotated row label.
LETTER_INSIDE_PT = 1.0
#: ... and this far in from the map's left edge, so the letter sits in the map's empty corner.
LETTER_INSIDE_X_PT = 2.0


def _letter_anchor(fig: Figure, ax, renderer, inv, inside: bool = False) -> tuple[float, float]:
    """(left edge of the panel's ink, top of the panel's ink) in figure coordinates.

    The title is hidden while the left edge is measured: a centred title is usually wider than its
    axes, and a tight bbox that included it would drag the letter left of the y-label.
    """
    title = ax.title
    was_visible = title.get_visible()
    title.set_visible(False)
    fig.canvas.draw()
    left = ax.get_tightbbox(renderer).transformed(inv).x0
    title.set_visible(was_visible)
    fig.canvas.draw()
    if inside:
        return left, ax.get_position().y1 - LETTER_INSIDE_PT / 72 / fig.get_figheight()
    return left, ax.get_tightbbox(renderer).transformed(inv).y1


def panel_letter(fig: Figure, ax, letter: str, *, fontsize: float | None = None,
                 x: float | None = None, y: float | None = None, inside: bool = False):
    """Draw one bold panel letter by the rule above (``LETTER_INSIDE_PT``); ``x``/``y`` override.

    Prefer :func:`panel_letters`, which also aligns rows and columns.
    """
    if x is None or y is None:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        left, top = _letter_anchor(fig, ax, renderer, fig.transFigure.inverted(), inside)
        x = left if x is None else x
        y = top if y is None else y
    # Recorded so :func:`save` can stamp the shipped PDF with the panels it actually letters. A
    # caption that describes a panel the figure does not draw is otherwise invisible until a
    # reader finds it, and it has already happened twice in this paper.
    fig._panel_letters = sorted(set(getattr(fig, "_panel_letters", set())) | {letter})
    return fig.text(x, y, letter, ha="left", va="top", fontsize=fontsize or FS["title"],
                    fontweight="bold")


def _group(entries, key, tol):
    groups: list[list] = []
    for entry in sorted(entries, key=key):
        if groups and abs(key(entry) - key(groups[-1][0])) <= tol:
            groups[-1].append(entry)
        else:
            groups.append([entry])
    return groups


def panel_letters(fig: Figure, pairs, *, tol: float = 0.06, row_tol: float = 0.03,
                  inside: bool = False, fontsize: float | None = None):
    """Letter several panels by the rule above: one x per column, one top per row.

    Columns are panels whose axes start at the same x, rows panels whose axes end at the same top
    (within ``tol`` / ``row_tol`` of the figure). Measured in two redraws for all panels together.
    """
    axes = [ax for _, ax in pairs]
    titles = [(ax.title, ax.title.get_visible()) for ax in axes]
    for title, _ in titles:
        title.set_visible(False)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    lefts = [ax.get_tightbbox(renderer).transformed(inv).x0 for ax in axes]
    for title, was_visible in titles:
        title.set_visible(was_visible)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inset = LETTER_INSIDE_PT / 72 / fig.get_figheight()
    # Axes boxes in FIGURE coordinates, from the rendered extent: `get_position` is relative to the
    # parent, which for an axes inside a subfigure is the subfigure, so two side-by-side blocks
    # would report the same x0 and be lettered as one column.
    boxes = [ax.get_window_extent(renderer).transformed(inv) for ax in axes]
    tops = [box.y1 - inset if inside else ax.get_tightbbox(renderer).transformed(inv).y1
            for ax, box in zip(axes, boxes)]

    entries = [dict(letter=letter, ax=ax, x0=box.x0, y1=box.y1, left=left, top=top, lifted=False)
               for (letter, ax), box, left, top in zip(pairs, boxes, lefts, tops)]
    for column in _group(entries, lambda e: e["x0"], tol):
        for e in column:
            # Maps: inside the map's own top-left corner (empty sea), clear of the row label,
            # instead of at the label's left edge where it sat too close to it.
            e["left"] = (min(c["x0"] for c in column) + LETTER_INSIDE_X_PT / 72 / fig.get_figwidth()
                         if inside else min(c["left"] for c in column))
    # A long rotated y-label can reach the axes top, right where the letter goes (S17's
    # "Energy capacity (GWh)"): that row's letters then rise until they clear it.
    letter_h = 1.15 * (fontsize or FS["title"]) / 72 / fig.get_figheight()
    for e in entries:
        label = e["ax"].yaxis.label
        if not inside and label.get_visible() and label.get_text().strip():
            label_top = label.get_window_extent(renderer).transformed(inv).y1
            # 2 pt of air between the label's end and the letter: flush, they read as one word.
            gap = 2.0 / 72 / fig.get_figheight()
            if label_top > e["top"] - letter_h - gap:
                e["top"], e["lifted"] = max(e["top"], label_top + letter_h + gap), True
    for row in _group(entries, lambda e: e["y1"], row_tol):
        pick = max if (not inside or any(r["lifted"] for r in row)) else min
        top = pick(r["top"] for r in row)
        for e in row:
            e["top"] = top
    return {e["letter"]: panel_letter(fig, e["ax"], e["letter"], x=e["left"], y=e["top"],
                                      inside=inside, fontsize=fontsize)
            for e in entries}


def align_colorbar_labels(fig: Figure, bars, *, tol: float = 0.02):
    """Put the labels of colorbars that share a column on ONE vertical line.

    Each label sits just right of its own tick labels, so "-200" and "5" put "% change" at two
    different distances from the bar and the column reads ragged. Call it last, after any resize:
    the offset is in points from the bar, so it survives a later resize.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    entries = [(bar, bar.ax.get_window_extent(renderer)) for bar in bars
               if bar.ax.yaxis.label.get_text().strip()]
    width_px = fig.get_figwidth() * fig.dpi
    for column in _group(entries, lambda e: e[1].x0 / width_px, tol):
        def tick_right(bar):
            # Only the tick labels actually drawn: the locator also emits labels beyond the bar's
            # range, clipped from the drawing but still measurable.
            lo, hi = sorted(bar.ax.get_ylim())
            return max((t.get_window_extent(renderer).x1 for t, v in
                        zip(bar.ax.yaxis.get_ticklabels(), bar.ax.yaxis.get_ticklocs())
                        if t.get_visible() and t.get_text() and lo - 1e-9 <= v <= hi + 1e-9),
                       default=bar.ax.get_window_extent(renderer).x1)
        target = max(tick_right(bar) for bar, _ in column) + 3.0 * fig.dpi / 72
        # Anchored a fixed number of POINTS right of the bar's edge (not an axes fraction), so the
        # alignment survives any later resize of the figure; the offset is corrected by what is
        # measured, since the label's anchor-to-edge distance depends on its alignment.
        offsets = {}
        for bar, box in column:
            offsets[bar] = (target - box.x1) * 72 / fig.dpi
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
    """Place a legend centred under *axes*, a fixed gap beneath their lowest rendered element.

    Anchoring to a guessed negative y put the legend a long way below the panel whenever the
    rotated zone labels were short, and on top of them when they were long.  Measuring the tight
    bbox of the axes it belongs to puts it the same distance below every time.

    The two directions are measured off DIFFERENT boxes on purpose.  Vertically the tight bbox is
    what matters: the legend has to clear the tick labels and the x label, whose depth varies.
    Horizontally the tight bbox is the wrong reference, because the y label and tick labels hang
    off the left side only; centring on it pushes the legend left of the plotting area it belongs
    to, which is the frame a reader actually sees.  So the centre comes from the axes position.
    """
    axes = list(axes) if isinstance(axes, (list, tuple)) else [axes]
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    boxes = [ax.get_tightbbox(renderer).transformed(inv) for ax in axes]
    frames = [ax.get_position() for ax in axes]
    centre = 0.5 * (min(b.x0 for b in frames) + max(b.x1 for b in frames))
    top = min(b.y0 for b in boxes) - gap_pt / 72 / fig.get_figheight()
    return fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(centre, top),
                      ncol=ncol, frameon=False, fontsize=fontsize, **kwargs)


def fit_width(fig: Figure, *, target_mm: float = 180.0, max_height_mm: float | None = None,
              tol_mm: float = 0.6, max_iter: int = 5):
    """Grow the figure until its RENDERED width is the target, so no rescaling happens later.

    A figure whose tight bbox comes out at 154 mm is not "safely under 180" — it will be placed at
    the column width and scaled UP by 1.17x, taking 7 pt type to 8 pt and quietly breaking the size
    rule this module enforces.  Point sizes are absolute while the layout is relative, so scaling
    the canvas grows the panels and leaves the type where it is; the fixed text margins do not
    scale, which is why this iterates instead of applying one ratio.
    """
    for _ in range(max_iter):
        fig.canvas.draw()
        width_mm = fig.get_tightbbox(fig.canvas.get_renderer()).width / MM
        if abs(width_mm - target_mm) <= tol_mm:
            break
        scale = target_mm / width_mm
        w, h = fig.get_size_inches()
        if max_height_mm is not None and h * scale / MM > max_height_mm:
            scale = min(scale, max_height_mm * MM / h)
        fig.set_size_inches(w * scale, h * scale)
    return fig


def save(fig: Figure, stem: str, figs_dir: Path, *, dpi: int = 400,
         max_width_mm: float = 180.0, font_range: tuple = FONT_RANGE, check_fonts: bool = True):
    """Write ``{stem}.pdf`` and ``{stem}.png``, refusing anything that breaks the artwork rules."""
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
        lo, hi = font_range
        bad = sorted({round(float(t.get_fontsize()), 2) for t in fig.findobj(mpl.text.Text)
                      if t.get_text().strip() and not (lo - 1e-6 <= t.get_fontsize() <= hi + 1e-6)})
        assert not bad, f"{stem}: type sizes outside {lo}-{hi} pt: {bad}"

    # The panels this figure letters, stamped into the PDF itself so a caption can be checked
    # against the artwork rather than against the source that built it. Figures 2, 3 and 4 share
    # one builder in a distant cell, so any "which code cell drew this" heuristic reads the wrong
    # figure — which is how a caption describing a storage panel survived its removal.
    letters = "".join(getattr(fig, "_panel_letters", []))
    paths = []
    for suffix, kwargs in ((".pdf", dict(metadata={"Keywords": f"panels={letters}"})),
                           (".png", dict(dpi=dpi))):
        path = figs_dir / f"{stem}{suffix}"
        fig.savefig(path, bbox_inches="tight", **kwargs)
        paths.append(path)

    with open(paths[0], "rb") as handle:
        assert handle.read(5) == b"%PDF-", f"{stem}: PDF did not write a valid header"

    sizes = "  ".join(f"{p.suffix[1:]} {p.stat().st_size / 1024:.0f} kB" for p in paths)
    # The first panel's own height, not just the canvas: `fit_width` scales the whole figure to
    # land on 180 mm, so two figures given the same figsize can still end up with panels of
    # different heights, and matching panel heights across figures is what a reader sees.
    panel_mm = (fig.axes[0].get_position().height * fig.get_figheight() / MM) if fig.axes else 0.0
    print(f"  {stem:<28} {width_mm:5.1f} x {tight.height / MM:5.1f} mm   "
          f"panel {panel_mm:4.1f} mm   {sizes}")
    return paths


# ══════════════════════════════════════════════════════════════════════════════════════════
# Figure 1 — the mean state
# ══════════════════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class MapRow:
    """One row of a GWL map figure.

    ``field(period) -> DataArray`` always carries a ``model`` dimension, plus either ``country``
    (``kind="country"``, drawn as a choropleth) or ``lat``/``lon`` (``kind="grid"``, drawn as a
    pcolormesh).  It is a function of the warming level ALONE — a season is closed over by the
    caller — so one row definition serves the annual and the seasonal figures unchanged.
    """

    label: str
    field: Callable[[str], "object"]
    cmap: str
    kind: str = "country"
    unit: str = ""
    change: str = "percent"                 # "percent" | "absolute"
    bins: Sequence[float] | None = None     # discrete change classes; None = continuous ramp
    abs_center: float | None = None         # diverging column 0 (residual load), else sequential
    change_cmap: str | None = None          # None -> RdBu_r (red = increase)
    clabel: str | None = None
    robust_abs_vmax: bool = False           # clip column 0 at p99 (channel-concentrated Qrouted)


def sign_agreement(per_chain, model_dim: str = "model"):
    """How many chains agree on the sign of the change, as a fully vectorised reduction.

    Bit-identical to :func:`analysis.compute_sign_agreement` (verified on random fields with and
    without NaNs) but ~114x faster, because that one goes through ``xr.apply_ufunc(vectorize=True)``
    and therefore loops in Python over every cell.  On a 426x510 grid it takes 7 s per field, and a
    six-row supplementary figure needs twelve of them: 84 s of the runtime was hatching alone.
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


#: Target resolution of the equal-area raster a lon/lat grid is resampled onto. 0.1 deg is 7-11 km
#: across the domain, so 5 km never merges two source cells into one pixel.
LAEA_RES_M = 5000.0
_RASTER_INDEX = {}


def _laea_index(lon, lat, crs):
    """Nearest-source-cell lookup from a regular raster in *crs* back onto a regular lon/lat grid.

    Resampling by INVERSE lookup (each target pixel asks which source cell it lies in) is exact for
    a nearest-neighbour picture and needs no scipy; cached per grid, since every panel of a figure
    shares one. Returns (row, col, valid, extent, footprint).
    """
    import pyproj
    from shapely.geometry import Polygon

    key = (float(lon[0]), float(lon[-1]), lon.size, float(lat[0]), float(lat[-1]), lat.size, str(crs))
    if key in _RASTER_INDEX:
        return _RASTER_INDEX[key]
    dlon, dlat = float(lon[1] - lon[0]), float(lat[1] - lat[0])
    lo0, lo1 = lon[0] - dlon / 2, lon[-1] + dlon / 2
    la0, la1 = lat[0] - dlat / 2, lat[-1] + dlat / 2
    ring_lon = np.r_[np.linspace(lo0, lo1, 400), np.full(400, lo1),
                     np.linspace(lo1, lo0, 400), np.full(400, lo0)]
    ring_lat = np.r_[np.full(400, la0), np.linspace(la0, la1, 400),
                     np.full(400, la1), np.linspace(la1, la0, 400)]
    to_map = pyproj.Transformer.from_crs(4326, crs, always_xy=True)
    rx, ry = to_map.transform(ring_lon, ring_lat)
    footprint = Polygon(zip(rx, ry))
    x0, x1, y0, y1 = MAP_EXTENT_LAEA
    xs = np.arange(x0 + LAEA_RES_M / 2, x1, LAEA_RES_M)
    ys = np.arange(y0 + LAEA_RES_M / 2, y1, LAEA_RES_M)
    grid_x, grid_y = np.meshgrid(xs, ys)
    to_lonlat = pyproj.Transformer.from_crs(crs, 4326, always_xy=True)
    src_lon, src_lat = to_lonlat.transform(grid_x, grid_y)
    col = np.rint((src_lon - lon[0]) / dlon).astype(int)
    row = np.rint((src_lat - lat[0]) / dlat).astype(int)
    valid = (row >= 0) & (row < lat.size) & (col >= 0) & (col < lon.size)
    extent = (xs[0] - LAEA_RES_M / 2, xs[-1] + LAEA_RES_M / 2,
              ys[0] - LAEA_RES_M / 2, ys[-1] + LAEA_RES_M / 2)
    _RASTER_INDEX[key] = (row, col, valid, extent, footprint)
    return _RASTER_INDEX[key]


def _to_laea(field, crs):
    """A (lat, lon) field resampled onto the equal-area raster: (values, extent, footprint)."""
    row, col, valid, extent, footprint = _laea_index(field["lon"].values, field["lat"].values, crs)
    out = np.full(valid.shape, np.nan)
    out[valid] = np.asarray(field.values, dtype=float)[row[valid], col[valid]]
    return out, extent, footprint


def _draw_grid(ax, field, *, crs=None, **kwargs):
    """Draw a gridded field, preferring ``imshow`` on a regular grid.

    ``pcolormesh`` draws one quad per cell: on the paper's 426x510 grid that is 217k polygons per
    panel and ~3.9 M for a six-row figure, redrawn on every layout pass.  The GWL products are on an
    exactly regular 0.1 deg lat/lon grid (spacing constant to 1.4e-14 deg), where ``imshow`` is not
    an approximation but the same picture drawn as one resampled image.  The pcolormesh branch is
    kept so an irregular grid still renders correctly rather than silently distorting.
    """
    if crs is not None:
        values, extent, _ = _to_laea(field, crs)
        return ax.imshow(values, extent=extent, origin="lower", interpolation="nearest", **kwargs)
    lon, lat = field["lon"].values, field["lat"].values
    dlon, dlat = np.diff(lon), np.diff(lat)
    regular = (dlon.size and dlat.size
               and float(dlon.max() - dlon.min()) < 1e-9
               and float(dlat.max() - dlat.min()) < 1e-9
               and dlat[0] > 0)
    if not regular:
        return ax.pcolormesh(lon, lat, field.values, shading="auto", rasterized=True, **kwargs)
    dx, dy = float(dlon[0]), float(dlat[0])
    extent = (lon[0] - dx / 2, lon[-1] + dx / 2, lat[0] - dy / 2, lat[-1] + dy / 2)
    return ax.imshow(field.values, extent=extent, origin="lower", aspect="auto",
                     interpolation="nearest", **kwargs)


def _set_reference_ticks(bar, target: int, *, symmetric: bool):
    """Put exactly *target* ticks INSIDE the bar's value range, at round numbers.

    Setting a locator alone is not enough.  ``MaxNLocator`` places ticks on round values over the
    requested interval, and the ones that fall outside the data range are clipped from the drawing
    but still exist as labelled ticks: solar spans ~30-430 TWh/yr, so ``nbins=3`` yields
    0/150/300/450 of which only two are ever drawn.  A count taken from the tick labels therefore
    passes while the figure shows half as many.  Search ``nbins`` for the value that lands the
    wanted number of ticks *within* the range, and pin them explicitly.
    """
    lo, hi = float(bar.norm.vmin), float(bar.norm.vmax)
    best = None
    for nbins in range(2, 13):
        ticks = [t for t in MaxNLocator(nbins=nbins, symmetric=symmetric).tick_values(lo, hi)
                 if lo <= t <= hi]
        if len(ticks) == target:
            bar.set_ticks(ticks)
            return ticks
        if best is None or abs(len(ticks) - target) < abs(len(best) - target):
            best = ticks
    # Not always reachable: over a range whose ends are both round (0 to 40, 0 to 1) the nice-number
    # steps jump straight from 3 in-range ticks to 5, so 4 does not exist. Every row in the paper
    # currently lands the target, so say so out loud if that ever stops being true rather than
    # quietly drawing a different number.
    print(f"  [ticks] wanted {target} inside [{lo:g}, {hi:g}] but the closest round set has "
          f"{len(best)}: {best}")
    bar.set_ticks(best)
    return best


def _reference_colorbar(fig: Figure, ax, norm, row: "MapRow", extend: str, mappable=None):
    """The column-0 colorbar: one construction for both row types, one tick rule.

    A diverging reference scale (residual load) gets one tick more than a sequential one, because
    it needs the zero marked and a symmetric locator delivers it.
    """
    if mappable is None:
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=row.cmap)
        mappable.set_array([])
    bar = fig.colorbar(mappable, ax=ax, shrink=0.72, fraction=0.15, pad=CBAR_PAD, extend=extend)
    bar.set_label(row.unit, fontsize=FS["tick"])
    bar.ax.tick_params(labelsize=FS["tick"])
    bar.outline.set_linewidth(0.3)
    diverging = row.abs_center is not None
    _set_reference_ticks(bar, CBAR_TICKS_DIVERGING if diverging else CBAR_TICKS_SEQUENTIAL,
                         symmetric=diverging)
    # Recorded on the figure so a notebook can assert the rendered tick counts rather than trust
    # the locator: MaxNLocator picks round values, so `nbins` is a target and not a guarantee.
    fig.reference_colorbars.append(bar)
    return bar


def _hatch_grid(ax, agreement, field, style: PaperStyle, *, crs=None) -> None:
    """The "not robust" idiom on a grid: fewer than ``robustness_threshold`` chains agree on sign."""
    mask = ((agreement.values < style.robustness_threshold)
            & np.isfinite(field.values)).astype(float)
    if crs is not None:
        mask, extent, _ = _to_laea(agreement.copy(data=mask), crs)
        xs = np.linspace(extent[0], extent[1], mask.shape[1])
        ys = np.linspace(extent[2], extent[3], mask.shape[0])
        coords = (xs, ys, np.nan_to_num(mask))
    else:
        coords = (agreement["lon"].values, agreement["lat"].values, mask)
    overlay = ax.contourf(*coords, levels=[0.5, 1.5], hatches=["///"], colors="none", zorder=3)
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
    """One row per quantity, one column per warming level; column 0 is the reference level.

    ``figure``: draw into this (sub)figure instead of a new one -- for side-by-side blocks; the
    caller then owns the size, so ``fit_maps`` is skipped. ``column_titles``: one string per column
    in place of the default two-line "<level> / change vs <reference>".

    Handles **both** map types in one figure, which the supplementary generation figure needs:
    solar and wind arrive as gridded capacity factors while hydro arrives as country totals, and
    neither of the two existing builders could draw them together.

    Three conventions are unified here rather than inherited, because a mixed figure cannot carry
    two of each: the ensemble is reduced with the **median** (``plot_climate_gwl_maps`` used the
    mean), the change ramp is **RdBu** so blue is an increase (it used ``RdBu_r``), and every row
    owns its colorbar so rows in different units stay legible side by side.

    Binning is per row, and deliberately so: discrete classes exist to stop a country polygon with
    a 0.3 % change reading as structure, which is a polygon problem.  Imposing bin edges on a smooth
    field would invent boundaries that are not in the data, so gridded rows pass ``bins=None``.

    ``share_abs_scale`` puts column 0 of every row on ONE scale.  Off by default, because the rows
    of a mixed figure carry different quantities in different units.  On for the seasonal figures,
    whose rows are the four seasons of the SAME quantity: auto-scaling each row there gave DJF a
    +-48 TWh bar against MAM's +-75, so a country could render identically in two rows while holding
    different values, which defeats the point of reading a seasonal cycle down a column.

    It also takes a SEQUENCE of row indices, for a figure whose rows are not all in one unit.  The
    conversion figures put an annual row (TWh per year) above the four seasons (TWh per season);
    the seasons must share so the cycle reads, and the annual row must not join them because it is
    a different quantity, four times the size by construction.

    ``share_change_scale`` does the same for the CHANGE columns, and takes the same forms.  Rows
    with ``bins`` already share a ramp, since the edges are fixed; rows without them -- the gridded
    ones -- otherwise get a symmetric limit computed from their own data, which on the solar figure
    gave five rows carrying +-3, +-11, +-7, +-3.5 and +-4.5 %, so one red meant 2 % in one row and
    8 % in another.  Sharing makes the winter signal dominate and the summer one go faint; that is
    the seasons differing by a factor of three, which is the comparison a shared scale is for.

    No figure title: the caption lives in the notebook markdown.
    """
    n_rows, n_cols = len(rows), len(style.period_names)
    # `width_mm` below 180 for a tall grid of equal-area maps: Europe is nearly square in true
    # proportions, so at full width each map is height-limited and letterboxed inside a wide cell.
    if figure is None:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(width_mm * MM, height_mm * MM),
                                 squeeze=False, constrained_layout=True)
    else:
        fig, fit_maps = figure, False
        axes = fig.subplots(n_rows, n_cols, squeeze=False)
    row_axes = []
    fig.reference_colorbars, fig.change_colorbars = [], []
    border = dict(edgecolor=edgecolor, linewidth=linewidth)

    # Computed up front so a shared scale can see every row before the first is drawn (and so the
    # reference field is read once per row rather than twice).
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

    # Every row's change fields BEFORE any row is drawn: a row cannot be put on a scale it shares
    # with the others until the others exist.  Nothing is computed twice -- the per-chain median and
    # the sign agreement are the expensive part and each still runs exactly once per row.
    all_changes, all_agreements = [], []
    for row_index, row in enumerate(rows):
        reference_per_chain = references_per_chain[row_index]
        reference = references[row_index]
        absolute = row.change == "absolute"
        # A near-zero denominator manufactures a huge percentage.  Only meaningful for country
        # polygons: on a grid the neighbouring cells carry the context that makes it readable.
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

        # A change is always diverging, so it needs its OWN map: reusing the row's sequential
        # column-0 cmap painted tas's warming in `inferno`, a one-sided ramp on a signed quantity.
        #
        # RdBu_r for every row: RED = MORE OF THE QUANTITY THIS ROW PLOTS. One rule the reader
        # learns once, and the same one Figures 3b/4b and the supplementary level grid already use
        # for changes. It replaces a default of RdBu (blue = increase) with per-row overrides, under
        # which red meant opposite things on different rows of one figure.
        change_cmap = row.change_cmap or "RdBu_r"
        clabel = row.clabel or (f"change ({row.unit})" if absolute else "% change")

        if edges is not None:
            norm, discrete_cmap = _binned_norm(edges, change_cmap)
            neutral = max(abs(e) for e in _zero_class(edges))
            change_kw = dict(norm=norm, cmap=discrete_cmap)
            bar_kw = dict(mappable=plt.cm.ScalarMappable(norm=norm, cmap=discrete_cmap),
                          extend="both", ticks=edges, spacing="uniform")
        else:
            limit = (shared_change_limit if row_index in change_shared_rows
                     else max(_symmetric_limit(c, clip_percentile) for c in changes))
            data_max = max(float(np.nanmax(np.abs(c.values))) for c in changes)
            neutral = None
            norm = mcolors.TwoSlopeNorm(vcenter=0, vmin=-limit, vmax=limit)
            change_kw = dict(cmap=change_cmap, vcenter=0, vmin=-limit, vmax=limit)
            bar_kw = dict(mappable=plt.cm.ScalarMappable(norm=norm, cmap=change_cmap),
                          extend="both" if data_max > limit * 1.0001 else "neither")

        if is_grid:
            # Draw to the DATA's bounds, not MAP_EXTENT: the grid stops short on the east and north,
            # so drawing to MAP_EXTENT leaves a strip of bare country outlines past the field.
            crs = countries.crs if is_projected(countries) else None
            if crs is not None:
                # Outlines only where the field exists: clipped to the grid's own footprint.
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
            # Drawn as an image, not a mesh: left as vector quads the PDF came out at 88 MB and
            # every layout pass redrew 3.9 M polygons. Text, boundaries and colorbars stay vector.
            mesh = _draw_grid(ax, reference, crs=crs, cmap=row.cmap, vmax=abs_vmax)
            clipped.boundary.plot(ax=ax, color=edgecolor, linewidth=linewidth)
            _reference_colorbar(fig, ax, None, row,
                                "max" if abs_vmax is not None else "neither", mappable=mesh)

            for col, (change, agreement) in enumerate(zip(changes, agreements), start=1):
                ax = axes[row_index, col]
                _draw_grid(ax, change, crs=crs,
                           **({"norm": change_kw["norm"], "cmap": change_kw["cmap"]}
                              if edges is not None else {"cmap": change_cmap, "norm": norm}))
                clipped.boundary.plot(ax=ax, color=edgecolor, linewidth=linewidth)
                _hatch_grid(ax, agreement, change, style, crs=crs)

            for col in range(n_cols):
                ax = axes[row_index, col]
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
            # The reference colorbar is built HERE rather than by geopandas. Passing legend=True
            # hands the bar to geopandas, which forwards legend_kwds to fig.colorbar and leaves
            # everything else at matplotlib's defaults, so there is no handle to set a tick locator
            # on and no way to control the padding. Constructing the norm explicitly and colouring
            # from it also makes the map and its bar share one object rather than agree by luck.
            if row.abs_center is None:
                finite = reference.values[np.isfinite(reference.values)]
                shared_here = row_index in shared_rows
                vmin = shared_vmin if shared_here else float(np.nanmin(finite))
                vmax = shared_vmax if shared_here else float(np.nanmax(finite))
                if vmax <= vmin:
                    vmax = vmin + 1e-6
                norm0, extend0 = mcolors.Normalize(vmin=vmin, vmax=vmax), "neither"
            else:
                limit = (shared_limit if row_index in shared_rows
                         else _symmetric_limit(reference, clip_percentile))
                norm0 = mcolors.TwoSlopeNorm(vcenter=row.abs_center, vmin=-limit, vmax=limit)
                extend0 = "both"
            plot_country_choropleth(ax, reference.to_series().dropna(), countries,
                                    cmap=row.cmap, norm=norm0, legend=False, **border)
            _reference_colorbar(fig, ax, norm0, row, extend0)

            for col, (change, agreement) in enumerate(zip(changes, agreements), start=1):
                ax = axes[row_index, col]
                plot_country_choropleth(ax, change.to_series().dropna(), countries, legend=False,
                                        **border, **change_kw)
                _hatch_disagreeing_countries(ax, agreement, change, countries, style,
                                             min_magnitude=neutral)
                set_map_aspect(ax, countries)

        mappable = bar_kw.pop("mappable")
        mappable.set_array([])
        # matplotlib scales `pad` by the width of the PARENT bbox, and this bar spans every change
        # column while the reference bar has one. Passing the same number therefore puts this bar
        # ~2x further out (measured: 0.0265 of figure width against 0.0130). Scale it by the width
        # ratio so both bars sit the same distance from their maps, which is what "the same
        # padding" means to a reader.
        span = (axes[row_index, n_cols - 1].get_position().x1
                - axes[row_index, 1].get_position().x0)
        pad = CBAR_PAD * axes[row_index, 0].get_position().width / span
        bar = fig.colorbar(mappable, ax=[axes[row_index, c] for c in range(1, n_cols)],
                           fraction=0.030, pad=pad, **bar_kw)
        bar.set_label(clabel, fontsize=FS["tick"])
        bar.ax.tick_params(labelsize=FS["tick"])
        bar.outline.set_linewidth(0.3)
        fig.change_colorbars.append(bar)

        # The map axes are switched off, so re-enable just enough to carry the row label.
        axes[row_index, 0].set_ylabel(row.label, fontsize=FS["title"], fontweight="bold", labelpad=3)
        axes[row_index, 0].axis("on")
        axes[row_index, 0].set_xticks([])
        axes[row_index, 0].set_yticks([])
        axes[row_index, 0].grid(False)
        for spine in axes[row_index, 0].spines.values():
            spine.set_visible(False)
        row_axes.append(axes[row_index, 0])

    # The change columns are NOT all percentages: an absolute row is a difference, which the working
    # figure mislabelled by hardcoding "% vs" in this title. Each row's own colorbar already carries
    # its unit, so the column title states only the comparison.
    for col, period in enumerate(style.period_names):
        # Two lines: at 8 pt on a ~35 mm map the one-line form ran into its neighbour's title.
        title = (column_titles[col] if column_titles is not None
                 else period if col == 0 else f"{period}\nchange vs {style.reference_period}")
        axes[0, col].set_title(title, fontsize=FS["title"], fontweight="bold")

    # Equal-area maps are drawn at equal aspect, so in a tall grid each map is limited by its
    # HEIGHT and sits letterboxed in a wider cell. Narrow the figure by exactly that slack --
    # measured as a column's original cell width minus its drawn map width -- so the maps keep
    # their size and only the white gaps go. Constrained layout re-flows on each draw.
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
    change_clabels: Mapping[str, str] | None = None,
    **kwargs,
) -> tuple[Figure, list]:
    """Figure 1: country maps only, in the ``(label, field, cmap, abs_center)`` tuple form.

    A thin adapter over :func:`gwl_maps` so the main-figures notebook is untouched by the
    supplementary rewrite that generalised the builder.
    """
    rows = [
        MapRow(label=name, field=field, cmap=cmap, kind="country", unit=unit,
               change="absolute" if name in absolute_change_rows else "percent",
               bins=(change_bins or {}).get(name), abs_center=absolute_center,
               # NO per-row change ramp. `absolute_change_rows` names the rows whose change is in
               # physical units rather than per cent — a statement about the UNIT, which it had
               # also been quietly making about the COLOUR. Because residual load was the only
               # absolute row, it alone got RdBu_r while the five per-cent rows got RdBu, so red
               # meant "increase" on one row of Figure 1 and "decrease" on the other five.
               clabel=(change_clabels or {}).get(name))
        for name, field, cmap, absolute_center in quantities
    ]
    return gwl_maps(rows, countries, style, sign_agreement, **kwargs)


# ══════════════════════════════════════════════════════════════════════════════════════════
# Figure 2 — the tail, and where the exceedance hours sit
# ══════════════════════════════════════════════════════════════════════════════════════════
def survival_axes(ax, survival, x, threshold, style: PaperStyle, *, ratio: float | None = None,
                  show_legend: bool = False):
    """One exceedance (survival) curve panel: P(RL > x) per warming level, log-y."""
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
        # Anchored to the RIGHT EDGE of the axis, not to the threshold. The shaded band is only
        # ~14 % of the panel in the Nordic zone (against ~50 % in Central-West), so a label placed
        # just right of the threshold ran into the frame there. Right-aligning puts it inside the
        # band in every zone without cropping any data or padding the axis with empty space, and
        # the band IS the region the ratio refers to, so the meaning is unchanged.
        ax.annotate(f"×{ratio:.1f}", xy=(0.98, 0.55), xycoords=("axes fraction", "data"),
                    ha="right", va="top", fontsize=FS["tick"], fontweight="bold",
                    color=_change_color(ratio - 1))
    if show_legend:
        ax.legend(fontsize=FS["tick"], frameon=False, loc="lower left", handlelength=1.0)


def paired_change_pct(reference: pd.Series, warmest: pd.Series, style: PaperStyle,
                      *, floor: float = 0.0):
    """(median per-cent change across chains, chains agreeing on its sign, chains used, dropped).

    **Paired by chain**: each chain's own change first, then the median.  The unpaired alternative,
    the change between the two ensemble medians, is what a reader can read straight off the drawn
    median lines, and that is its only advantage.  It can disagree in SIGN with the agreement count
    printed beside it, because that count is inherently paired — measured on this paper's run,
    Central-West sub-daily printed -3 % while the paired median and 5 of 9 chains said +7 %.  It is
    also not robust to the median chain changing identity between warming levels, the failure that
    turned +44 into +22 TWh/yr for the Europe residual-load total.  Figure 1's maps and Table 1
    already pair, so this is the estimator the whole paper uses.

    Chains whose OWN reference is at or below ``floor`` are dropped: a per-cent change on a
    near-zero baseline is noise, not signal.  The counts come back so the caller can refuse to
    annotate at all rather than pass off a four-chain median as an ensemble result — which is
    exactly what Figure 5b's Iberia multi-day would otherwise do.
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
    """Event burden per zone, ONE AXIS PER (duration class, season), grouped BY CLASS FIRST.

    Replaces the pair of panels that showed exceedance hours per zone and event hours per duration
    class.  The first of those restated what the survival panel above it already says with its
    threshold ratio; the second aggregated over zones, which hid the fact that the duration split is
    overwhelmingly zonal — South-East carries 581 h/yr of multi-day winter deficit against the
    British Isles' 43.

    Three things changed together, and they only work together:

    * **Class-major ordering.** The two sub-daily axes sit side by side and the two multi-day ones
      beside each other, so the eye compares seasons within a class rather than classes within a
      season.  The class split is the paper's subject; season is the variable inside it.
    * **One y limit per duration class.** Four independent scales made every sub-panel look alike
      and hid that multi-day winter is an order of magnitude above everything else.  Sharing within
      the class is what makes the two seasons comparable; sharing ACROSS classes is what the
      original four-scale version was avoiding, and it is still avoided.
    * **Days for the long class.** ``day_classes`` names the classes whose axis is divided by 24.
      A multi-day burden of 581 hours is 24 days; the hours reading invites the reader to compare
      it with a sub-daily 122 that is on a different scale anyway.

    ``spread`` draws the inter-quartile range across chains as a thin bar with the median as a
    highlighted rule, rather than a bare median bar.  This panel reported medians only, which made
    it the one place in the paper where a level was drawn with no indication of ensemble range.

    ``table`` carries one row per (model, period, zone, season, duration_class) with an ``hours_yr``
    column; the quartiles across chains are taken here.
    """
    classes = list(duration_labels)
    positions = np.arange(len(style.zones))
    grouped = table.groupby(["season", "duration_class", "zone", "period"])["hours_yr"]
    quartiles = {stat: grouped.quantile(q) for stat, q in (("lo", 0.25), ("mid", 0.5), ("hi", 0.75))}

    def value(stat, season, duration_class, zone, period):
        """Zero hours where a cell carries no events.

        The burden table stores NaN there, not zero, because `hours_yr` is frequency x mean
        duration and the mean duration of an empty set is undefined.  Left as NaN the bar would
        simply not be drawn AND the "no events" test below would never fire, so the cell would come
        out blank and indistinguishable from missing data — which is the one reading this panel
        must not allow.
        """
        got = quartiles[stat].get((season, duration_class, zone, period), np.nan)
        got = 0.0 if got is None or not np.isfinite(got) else float(got)
        return got / 24.0 if duration_class in day_classes else got

    # Class-major: axis index runs class-then-season, so classes[index // len(seasons)].
    layout = [(classes[index // len(seasons)], seasons[index % len(seasons)])
              for index in range(len(axes))]
    tops = {}
    for duration_class, season in layout:
        tops[duration_class] = max(
            tops.get(duration_class, 0.0),
            max(value("hi" if spread else "mid", season, duration_class, z, p)
                for z in style.zones for p in style.period_names))

    width = 0.26
    for ax, (duration_class, season) in zip(axes, layout):
        for k, period in enumerate(style.period_names):
            offset = (k - 1) * width
            for i, zone in enumerate(style.zones):
                mid = value("mid", season, duration_class, zone, period)
                if spread:
                    low = value("lo", season, duration_class, zone, period)
                    high = value("hi", season, duration_class, zone, period)
                    ax.bar(i + offset, high - low, width * 0.62, bottom=low,
                           color=blend(style.gwl_colors[period], 0.55), edgecolor="none", zorder=2)
                    ax.plot([i + offset - width * 0.42, i + offset + width * 0.42], [mid, mid],
                            color=style.gwl_colors[period], lw=1.0, solid_capstyle="butt", zorder=3)
                else:
                    ax.bar(i + offset, mid, width * 0.92, color=style.gwl_colors[period],
                           edgecolor="white", linewidth=0.3, zorder=2)

        # A zone with no events of this class in this season is a finding, not a gap.
        for i, zone in enumerate(style.zones):
            if max(value("mid", season, duration_class, zone, p)
                   for p in style.period_names) <= 1e-9:
                ax.annotate("no events", xy=(i, 0), xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", rotation=90, fontsize=FS["tick"], color="0.55")

        ax.set_ylim(0, max(tops[duration_class], 1e-9) * 1.18)
        ax.set_xticks(positions)
        # Vertical, not 30 degrees: these sub-axes are ~40 mm wide, so a 30-degree label spans
        # 9.2 mm of a 6.7 mm slot and neighbours collide. Upright costs height, not width.
        ax.set_xticklabels(style.labels(), fontsize=FS["tick"], rotation=90, va="top", ha="center")
        ax.set_xlim(-0.6, len(style.zones) - 0.4)
        ax.set_title(f"{duration_labels[duration_class].splitlines()[0]} \u00b7 {season_label(season)}",
                     fontsize=FS["label"])
        ax.tick_params(labelsize=FS["tick"])
        ax.grid(True, axis="y", color="0.93", lw=0.3)
        ax.set_axisbelow(True)

    # One y label per CLASS, on that class's first axis, because the two classes are now in
    # different units.
    for index, (duration_class, _season) in enumerate(layout):
        if index % len(seasons) == 0:
            unit = "days" if duration_class in day_classes else "hours"
            axes[index].set_ylabel(f"time in events ({unit[0]} yr$^{{-1}}$)", fontsize=FS["label"])


def tail_exceedance_figure(
    *,
    seasons: Sequence[str],
    residual_load_by_model: Mapping,
    period_defs: Mapping,
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
    """Survival grid on top, the two exceedance blocks below — Figure 2, for any season pair.

    Parameterised by ``seasons`` so the main figure (DJF/JJA) and its supplementary companion
    (MAM/SON) are the SAME code.  Duplicating the builder into the SI notebook would let the two
    drift apart figure by figure, which is exactly what the SI is meant to rule out.

    Rows are placed with explicit GridSpec bounds rather than `hspace`, because the gap between the
    survival block and the exceedance blocks has to hold three stacked things: the survival tick
    labels, one shared x-label, and the exceedance block titles. Tuning a single `hspace` to fit all
    three put the block titles straight through the x-labels.

    ``duration_block=False`` draws the survival grid ALONE, which is what the all-season
    supplementary version wants: the lower block's GridSpec is hard-coded to four axes, and
    `panel_zone_duration_hours` fills them class-major, so with four seasons every axis lands in the
    sub-daily class and the multi-day one is silently dropped. The figure then carries a single
    block and therefore no panel letters.
    """
    from .extremes_analysis import (compute_exceedance_probability, compute_exceedance_survival,
                                    compute_survival_x_grid)

    zones = list(style.zones)
    seasons = list(seasons)
    fig = plt.figure(figsize=(WIDTH, height_mm * MM))
    # With no lower block the survival grid takes that height back, stopping just above the shared
    # x-label and the warming-level legend instead of above the exceedance axes.
    top = gridspec.GridSpec(len(seasons), len(zones), figure=fig, left=0.055, right=0.995,
                            top=0.955, bottom=0.575 if duration_block else 0.255,
                            hspace=0.13, wspace=0.24)
    bottom = (gridspec.GridSpec(1, 4, figure=fig, left=0.055, right=0.995, top=0.375, bottom=0.150,
                                wspace=0.46) if duration_block else None)
    first_survival, last_row = None, []

    for col, zone in enumerate(zones):
        threshold = float(zone_threshold.sel(zone=zone))
        x = compute_survival_x_grid(residual_load_by_model, period_defs, zone,
                                    [season_months[s] for s in seasons], style.period_names,
                                    threshold=threshold)
        for row, season in enumerate(seasons):
            ax = fig.add_subplot(top[row, col])
            _, survival = compute_exceedance_survival(
                residual_load_by_model, period_defs, zone, season_months[season],
                style.period_names, spread=spread, x=x)
            p_ref = compute_exceedance_probability(residual_load_by_model, period_defs, zone,
                                                   season_months[season], style.reference_period,
                                                   threshold)
            p_warm = compute_exceedance_probability(residual_load_by_model, period_defs, zone,
                                                    season_months[season], style.warmest_period,
                                                    threshold)
            # No per-axes legend: the figure-level warming-level key below panel c already says
            # this, and repeating it inside a 26 mm panel spends the only clear corner it has.
            survival_axes(ax, survival, x, threshold, style,
                          ratio=(p_warm / p_ref if p_ref > 0 else np.nan), show_legend=False)
            ax.tick_params(labelsize=FS["tick"])
            if row == 0:
                ax.set_title(style.zone_labels[zone], fontsize=FS["title"], fontweight="bold")
                if col == 0:
                    first_survival = ax
            else:
                last_row.append(ax)
            if col == 0:
                ax.set_ylabel(f"{season_label(season)}\nP(RL > x)", fontsize=FS["label"])
            else:
                ax.set_yticklabels([])
            if row != len(seasons) - 1:
                ax.set_xticklabels([])

    # ONE x-label for the whole block. Six copies of "residual load (GWh / 3 h)" under six 26 mm
    # columns overlap each other, and they all say the same thing.
    span = [min(a.get_position().x0 for a in last_row), max(a.get_position().x1 for a in last_row)]
    x_label_artist = fig.text(sum(span) / 2, 0.50, "residual load per 3h step (GWh)", ha="center",
                              va="top", fontsize=FS["label"])

    if duration_block:
        ax_b = [fig.add_subplot(bottom[i]) for i in range(4)]
        panel_zone_duration_hours(ax_b, zone_season_hours, style, seasons, duration_labels)

    handles, labels = gwl_handles(style)
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.005),
               ncol=len(labels), frameon=False, fontsize=FS["tick"], title=GWL_TITLE)

    fit_width(fig)
    fig.canvas.draw()
    inv = fig.transFigure.inverted()
    renderer = fig.canvas.get_renderer()

    if duration_block:
        # a and b share the figure's left margin, so they share one letter x: they carry y-labels of
        # different widths ("DJF / P(RL > x)" against "event hours per year"), and anchoring each to
        # its own tight bbox left them visibly ragged.
        left_x = min(min(a.get_tightbbox(renderer).transformed(inv).x0 for a in axs)
                     for axs in ([first_survival], ax_b))

        # b is now FOUR axes under one block title, so the title is placed against the set.
        boxes = [a.get_tightbbox(renderer).transformed(inv) for a in ax_b]
        title_y = max(b.y1 for b in boxes) + 0.010
        block_text = fig.text(0.5 * (ax_b[0].get_position().x0 + ax_b[-1].get_position().x1),
                              title_y, block_title, ha="center", va="bottom",
                              fontsize=FS["title"], fontweight="bold")
        fig.canvas.draw()
        block_top = block_text.get_window_extent(fig.canvas.get_renderer()).transformed(inv).y1
        # Through `panel_letter` rather than a raw `fig.text`, even though both coordinates are
        # already known: that is the function which records the letter for `save` to stamp into the
        # PDF, so a bypass here left the shipped file claiming this figure has only a panel a.
        # Same rule as `panel_letters`: top of the letter on the top of the panel's ink, which for
        # b is its block title (a figure text, so measured here rather than by the helper).
        panel_letter(fig, ax_b[0], "b", x=left_x, y=block_top)
        panel_letter(fig, first_survival, "a", x=left_x)

    # The shared x-label sits a fixed 4 pt below the survival row's own tick labels. It was pinned
    # to a guessed figure coordinate before, which left it floating far under the axis.
    floor = min(a.get_tightbbox(renderer).transformed(inv).y0 for a in last_row)
    x_label_artist.set_position((x_label_artist.get_position()[0],
                                 floor - 4.0 / 72 / fig.get_figheight()))
    return fig


# ══════════════════════════════════════════════════════════════════════════════════════════
# Figures 3 and 4 — the two regimes
# ══════════════════════════════════════════════════════════════════════════════════════════
#: Change classes for :func:`panel_season_contribution`, shared by Figures 3 and 4 so a cell in one
#: means what it means in the other.  Symmetric about zero and open at both ends; on this run the
#: grid runs -10.2 to +39.7 points, so only the top class saturates.
SEASON_CONTRIBUTION_BINS = [-20, -10, -5, -2, 2, 5, 10, 20]
#: Cell classes for Figure 3b and S27b, in CHANGE IN EVENTS PER YEAR. One set per duration class:
#: measured over the 24 cells of each grid, sub-daily runs -1.0 to +7.9 and multi-day -0.80 to
#: +0.70, so a shared ramp would leave every multi-day cell in the neutral class. The top classes
#: were widened after review: at a top edge of 2 (sub-daily) and 0.4 (multi-day) four and three
#: cells sat in the overflow colour, indistinguishable from each other. Each class doubles, so the
#: whole range is covered; the panel colours by class RANK (see `_binned_norm`) so doubling the
#: span does not wash the 17 near-zero cells out to white.
SEASON_COUNT_BINS_SUB_DAILY = [-1, -0.5, -0.25, -0.05, 0.05, 0.25, 0.5, 1, 2, 4, 8]
SEASON_COUNT_BINS_MULTI_DAY = [-0.8, -0.4, -0.2, -0.1, -0.025, 0.025, 0.1, 0.2, 0.4, 0.8]


def season_contribution_by_model(seasonal: pd.DataFrame, annual: pd.DataFrame, zone: str,
                                 season: str, duration_class: str, style: PaperStyle,
                                 *, relative: bool = True) -> tuple[pd.Series, bool]:
    """PER-CHAIN contribution of one season to a zone's change, in points of its own baseline.

    ``100 x (freq[season, warmest] - freq[season, reference]) / freq[annual, reference]``, so a
    value reads as "this season contributed N points of the zone's per-cent change".  Paired inside
    the chain, matching every other change in this paper.

    Why not the plain change in events per year, which is the same unit as panel a: across the
    sub-daily grid those span **158x** (South-East JJA +7.90 against Central-East DJF +0.10), so one
    colour ramp leaves every small zone indistinguishable from no change, and Figures 3 and 4 could
    not share an encoding.  In points both grids span about 46, so one ramp serves both.

    Because the four seasonal counts sum to the annual count exactly — see
    ``compute_event_season_metrics``, which assigns each detected event whole to one season — a
    CHAIN's four contributions sum to that chain's own annual per-cent change exactly.  The four
    MEDIANS do not, for the same reason the four median shares did not, so a reader must not be
    invited to add a row.

    ``relative=False`` returns the plain change in events per year instead, which is what Figures 3b
    and 4b now draw.  The relative form was adopted when ONE colour ramp had to serve both figures:
    in events per year the sub-daily grid spans 8.8x (South-East JJA +7.90 against Central-East DJF
    +0.05) while the multi-day grid spans only -0.80 to +0.70, so a shared ramp left every multi-day
    cell indistinguishable from no change.  Giving each figure its own bins dissolves that — the two
    panels already sit in different figures, whose panel a axes are 0-30 and 0-7 events per year, so
    a reader is not being invited to compare cell colours across them.  What the counts buy is that
    the panel is in the same unit as panel a above it, which the relative form never was.

    Returns ``(values, has_no_events)``.  The flag is separate from an empty Series on
    purpose: a zone with no events of this class in this season is a different statement from one
    the robustness floor rejected, and collapsing the two is how a panel comes to show blank cells
    that a reader reads as missing data.
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
    """Which season each zone's change comes from: SEASONS AS ROWS, zones as columns.

    Replaces a two-season SHARE dumbbell that could not do this panel's job.  A share cannot show a
    rise — the British Isles' multi-day winter count goes 0.80 to 1.20 per year while its winter
    share FALLS 37.5 to 34.7 %, because the annual total grows faster — and two seasons hid where
    the growth actually is: of the four zones whose multi-day events increase, only the South-East
    grows in summer, while the British Isles grow across winter, spring and autumn and Central-West
    grows in autumn alone.

    The grid is seasons-by-zones rather than zones-by-seasons, and its cells are the change in
    EVENTS PER YEAR rather than points of the zone's own baseline.  Both were the other way round
    when one colour ramp had to serve Figures 3b and 4b together; ``season_contribution_by_model``
    carries why that constraint is gone.  What the current orientation buys is that the panel reads
    down a season and across the zones, matching how the caption argues, and that the cells are in
    panel a's unit.  Zone names still sit horizontal: "Central-West" is 11.5 mm at 5 pt against a
    ~13 mm column.

    Discrete classes rather than a continuous ramp, the same argument as Figure 1's maps: a cell
    changing by 0.05 events per year must not render as structure.  ``RdBu_r`` so red is a
    tightening, the convention Figure 1 already carries for residual load.  ``edges`` is per figure
    because the two classes differ by an order of magnitude in range, which is exactly the reason
    the relative form existed.
    """
    rows_items = list(seasons)
    column_items = list(style.zones)
    values = np.full((len(rows_items), len(column_items)), np.nan)
    agreement = np.zeros(values.shape, dtype=int)
    no_events = np.zeros(values.shape, dtype=bool)
    used = np.zeros(values.shape, dtype=int)
    neutral = min(abs(edge) for edge in edges) / 2

    for row, season in enumerate(rows_items):
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
            # A cell that rounds to zero is "no net change", not a disagreement: comparing signs
            # against a zero median counts the chains that happen to be exactly zero and would
            # stipple a cell whose message is that nothing happened.
            agreement[row, column] = (len(per_chain) if abs(value) < neutral
                                      else int((np.sign(per_chain) == np.sign(value)).sum()))

    norm, discrete = _binned_norm(edges, cmap, by="rank")
    discrete = discrete.copy()
    discrete.set_bad("white")
    image = ax.imshow(np.ma.masked_invalid(values), cmap=discrete, norm=norm, aspect="auto",
                      origin="upper",
                      extent=(-0.5, len(column_items) - 0.5, len(rows_items) - 0.5, -0.5))

    digits = 1 if max(abs(edge) for edge in edges) >= 1 else 2
    for row in range(len(rows_items)):
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
            # White on the saturated ends, ink everywhere else, decided on the class's own
            # luminance rather than on a threshold in data units that a new edge set would break.
            red, green, blue, _ = discrete(norm(value))
            light = 0.299 * red + 0.587 * green + 0.114 * blue > 0.55
            ax.annotate(f"{value:+.{digits}f}" if abs(value) >= neutral else f"{0.0:.{digits}f}",
                        xy=(column, row), ha="center", va="center", fontsize=FS["tick"],
                        color="0.15" if light else "white")
            if agreement[row, column] < style.robustness_threshold:
                ax.add_patch(plt.Rectangle((column - 0.5, row - 0.5), 1, 1, fill=False,
                                           hatch="///", edgecolor="0.45", lw=0.0, zorder=3))

    # The rc grid draws at the TICK positions, which here are the cell CENTRES, so it ran a hairline
    # straight through every printed value. The cell borders below are drawn explicitly instead.
    ax.grid(False)
    ax.set_xticks(range(len(column_items)))
    # Rotated as in panel a: at 7 pt six zone names no longer fit side by side in the heatmap width.
    ax.set_xticklabels(style.labels(), fontsize=FS["tick"], rotation=30, ha="right",
                       rotation_mode="anchor")
    ax.set_yticks(range(len(rows_items)))
    ax.set_yticklabels([season_label(r) for r in rows_items], fontsize=FS["tick"])
    ax.tick_params(length=0, labelsize=FS["tick"])
    for spine in ax.spines.values():
        spine.set_linewidth(0.4)
        spine.set_color("0.5")
    # Cell borders in the page ground, so the grid reads as cells rather than as a continuous field.
    for column in range(len(column_items) - 1):
        ax.axvline(column + 0.5, color="white", lw=0.8, zorder=2)
    for row in range(len(rows_items) - 1):
        ax.axhline(row + 0.5, color="white", lw=0.8, zorder=2)

    if show_colorbar:
        # VERTICAL, beside the grid: ten or eleven class edges at 7 pt do not fit across the
        # heatmap's width, and a horizontal bar ran them into one another ("-1.00-0.50-0.25").
        # In its own inset axis, outside the heatmap: `colorbar(ax=...)` steals the width it needs
        # from the panel, and panels a and b must stay the same width (asserted in the notebook).
        bar = ax.figure.colorbar(image, cax=ax.inset_axes([1.03, 0.0, 0.035, 1.0]),
                                 orientation="vertical", extend="both", ticks=edges,
                                 spacing="uniform")
        # Each edge at its own precision ("-0.025", "0.4", "8"), not one shared number of places.
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
    """One quantity by category, one box per warming level, annotated with the change and agreement.

    ``value_column`` is a parameter because the SAME panel draws Figures 3a/4a (event counts) and
    Figure 5b (``total_annual_severity_gwh``, the unabsorbed severity left after storage has done
    what it can).  The two must be drawn by one builder: they carry the same encoding, the same
    spread convention and the same change estimator, and a second copy would be free to drift from
    ``paired_change_pct`` — the whole point of which is that ONE estimator is used everywhere (the
    agreement count printed beside a percentage is inherently paired, so an unpaired percentage can
    contradict its own annotation).

    ``categories``/``category_column`` generalise the x axis off the zones.  Figure 5b puts the two
    DURATION CLASSES there and aggregates the zones away, because its subject is the split between
    short and long events rather than where they are.

    ``annotation_color`` overrides the red-up/blue-down ramp the change label otherwise takes from
    ``_change_color``.  Figures 3a and 4a set it to black: that ramp is blue-to-red and so is the
    warming-level ramp of the boxes underneath it, so the two colour systems were competing inside
    one panel for a distinction the ``+``/``-`` sign already carries.

    ``annotations`` replaces the default paired-change label with caller-supplied text, one entry
    per category.  Figure 5b needs it: its multi-day change is a null under every estimator
    (paired median -8.4 %, unpaired +0.7 %, paired mean +0.5 %, unpaired mean -0.9 %, per-chain
    changes running -33 to +45 %), so the default label described the median rather than the
    ensemble and contradicted its own visibly flat boxes.  What that panel annotates instead is the
    class's share of the total, which every estimator and all nine chains agree on.

    ``yscale="log"`` for quantities whose categories span more than about a decade.  The annotated
    change is what carries the magnitude of the growth on a log axis, which is why it is anchored in
    axes fraction rather than data coordinates below.
    """
    # 1.5, NOT whis_for(style.spread). `style.spread` is the convention for shaded BANDS and error
    # bars, which is the interquartile range; a box plot is a different object whose parts already
    # carry their own definitions — box = IQR, whisker = the most extreme chain within 1.5xIQR of
    # it. Feeding the band convention in here would have set whis=(25, 75) and collapsed every
    # whisker onto the edge of its own box.
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

    # Headroom for the annotation row. On a log axis a multiplicative headroom is the equivalent of
    # the linear 1.26x, and the floor has to come off zero because log(0) is undefined.
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
            ref = rows[rows.period == style.reference_period].set_index("model")[value_column]
            warm = rows[rows.period == style.warmest_period].set_index("model")[value_column]
            pct, n_agree, _, _ = paired_change_pct(ref, warm, style)
            if not np.isfinite(pct):
                continue
            text = f"{pct:+.0f}%\n{n_agree}/{style.n_models}"
            color = annotation_color or _change_color(pct)
        # Axes fraction, not the data-space top: identical placement on both scales, and it cannot
        # be thrown by a later limit change.
        ax.annotate(text, xy=(i, 1.0), xycoords=("data", "axes fraction"),
                    xytext=(0, -1), textcoords="offset points",
                    ha="center", va="top", fontsize=FS["tick"], fontweight="bold", color=color)

    ax.set_xticks(positions)
    # A steep rotation must be centred, not right-aligned: at 90 degrees "right" hangs the label off
    # the tick by half its length. The 4-panel option sets need 90 because six zone names in a 40 mm
    # panel is a 6.7 mm slot against "Central-West" at 9.2 mm rotated 30 degrees.
    # `va` is only passed for the steep case: matplotlib's default for an x tick label is already
    # "top", and naming it explicitly for the 30-degree case would change Figures 3a and 4a, which
    # nothing asked for.
    steep = dict(ha="center", va="top") if rotation >= 60 else {}
    ax.set_xticklabels(labels, fontsize=FS["tick"], rotation=rotation,
                       **({"ha": "center" if rotation == 0 else "right"} if not steep else steep))
    # Only when the caller replaced the zone axis. Figures 3a and 4a never set a limit and take
    # matplotlib's own margins; imposing one here would move them.
    if category_column != "zone":
        ax.set_xlim(-0.5, len(categories) - 0.5)
    ax.set_ylabel(ylabel, fontsize=FS["label"])
    ax.tick_params(labelsize=FS["tick"])


def panel_hour_of_day(ax_jja, ax_djf, southern_hours: pd.DataFrame, style: PaperStyle,
                      *, evening=(15, 21.8), night=(-0.8, 6.0), band_color="#F5A623",
                      evening_label="evening", night_label="night",
                      ylabel="time above P95 (h yr$^{-1}$)"):
    """Exceedance hours by hour of day, JJA above DJF, southern zones pooled.

    EXCEEDANCE hours, counted timestep by timestep against the zone threshold with no detector
    involved.  Not the event hours of :func:`panel_zone_duration_hours`, which is why the two carry
    different y-labels: assigning an hour of day to an event's hours would need per-timestep event
    membership, which the duration-stratified table does not carry.

    The shaded region is one window, 15:00 through 06:00, drawn as two spans only because it wraps
    midnight on a 0-21 axis.  Both halves therefore take the same colour: they are the same
    statement.  In southern JJA at +3.0 °C the exceedance runs 39 h/yr at 18:00 and 32 at 15:00 (the
    evening ramp once PV fades) and stays at 17 and 13 h/yr at 00:00 and 03:00, against ~0 between
    06:00 and 12:00.

    The working figure's "PV covers midday" annotation is dropped: the flat 06-12 stretch says it,
    and the label was competing with the curve it pointed at.
    """
    qlo, qhi = style.spread_percentiles
    for ax, season in ((ax_jja, "JJA"), (ax_djf, "DJF")):
        data = southern_hours[southern_hours.season == season]
        for period in style.period_names:
            wide = data[data.period == period].pivot(index="model", columns="hour",
                                                     values="hours_yr")
            hours = wide.columns.values
            ax.plot(hours, wide.median(axis=0).values, "-o", ms=1.8,
                    color=style.gwl_colors[period], lw=0.9, zorder=3)
            ax.fill_between(hours, np.percentile(wide.values, qlo, axis=0),
                            np.percentile(wide.values, qhi, axis=0),
                            color=style.gwl_colors[period], alpha=0.15, lw=0, zorder=2)
        ax.axvspan(*evening, color=band_color, alpha=0.16, lw=0, zorder=0)
        ax.axvspan(*night, color=band_color, alpha=0.16, lw=0, zorder=0)
        ax.set_xlim(-0.8, 21.8)
        ax.set_xticks(range(0, 24, 3))
        ax.set_ylim(0, ax.get_ylim()[1] * 1.30)
        # Season goes inside, at the TOP OF THE MIDDAY GAP. As a second rotated line on the y-label
        # it doubled the label's width, and these two axes are only ~25 mm tall. The bottom-left
        # corner it used to sit in is not free: the +3.0 °C JJA curve starts at ~17 h/yr at hour 0
        # with its spread band below, and the label ran into it. This spot is empty by construction
        # rather than by luck — the two shaded bands take axes fraction 0-0.30 and 0.70-1.00, so
        # x = 0.50 is in the white gap between them and clear of their labels, and the midday trough
        # under it is the flattest stretch of both curves.
        # In-panel labels at tick size: at the 8 pt title size they filled a quarter of a 25 mm axes.
        ax.annotate(season_label(season), xy=(0.50, 0.95), xycoords="axes fraction",
                    ha="center", va="top",
                    fontsize=FS["tick"], fontweight="bold", color="0.25")
        ax.tick_params(labelsize=FS["tick"])

    for label, window in ((evening_label, evening), (night_label, night)):
        ax_jja.annotate(label, xy=(float(np.mean(window)), ax_jja.get_ylim()[1]),
                        xytext=(0, -1), textcoords="offset points", ha="center", va="top",
                        fontsize=FS["tick"], fontweight="bold", color="#8a5a00")
    ax_djf.set_xlabel("hour of day", fontsize=FS["label"])
    # ONE y-label for the stacked pair, centred across both: at 7.5 pt two copies of it overran
    # each other in the ~25 mm the axes are tall. Same quantity, same unit, so one says it.
    ax_djf.set_ylabel(ylabel, fontsize=FS["label"])
    ax_jja.set_xticklabels([])

    # ONE x for both y-labels. DJF reaches three digits where JJA reaches two, so its tick labels
    # are wider and matplotlib pushes its y-label further left; the two then read as misaligned
    # rather than as two scales. Measured, not hardcoded: a future threshold or turbine change that
    # takes DJF to four digits would silently re-break a fixed offset.
    figure = ax_jja.figure
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    inset = max(
        (ax.get_window_extent().x0
         - min(t.get_window_extent(renderer).x0 for t in ax.get_yticklabels()
               if t.get_visible() and t.get_text().strip()))
        / ax.get_window_extent().width
        for ax in (ax_jja, ax_djf))
    # Centre of the pair, in the lower axes' own fraction: it spans 0 .. 1 + gap + the upper height.
    low, high = ax_djf.get_position(), ax_jja.get_position()
    top = (high.y1 - low.y0) / low.height
    ax_djf.yaxis.set_label_coords(-(inset + 0.02), 0.5 * top)


def _grouped_zone_axis(ax, style: PaperStyle, sublabels: Sequence[str], width: float,
                       *, zone_offset_pt: float | None = None, show_zones: bool = True) -> None:
    """Two-level x axis: *sublabels* under each bar of a group, the zone name under the group.

    Shared by ``panel_driver_attribution`` (season under each stack) and ``panel_storage_stack``
    (warming level under each stack).  One implementation because the zone row is MEASURED, not
    hardcoded: zone names go horizontal where they fit and rotate where they do not, since the same
    builder draws a full-width four-stack panel (a zone slot is ~23 mm) and a half-width one
    (~10 mm), and "Central-West" is 11.5 mm at 5 pt.
    """
    n = len(sublabels)
    ax.set_xticks([i + (s - (n - 1) / 2) * width
                   for i in range(len(style.zones)) for s in range(n)])
    ax.set_xticklabels(list(sublabels) * len(style.zones), fontsize=FS["tick"])
    ax.tick_params(axis="x", length=0, pad=1.0)
    ax.tick_params(axis="y", labelsize=FS["tick"])

    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    # Sublabels go vertical when they do not fit their stack: four season NAMES under each of six
    # zones ("Winter Spring Summer Autumn" at 7 pt) ran into one another; two do fit.
    pitch_px = ax.get_window_extent().width / len(style.zones) * width / 1.0
    labels = [t for t in ax.get_xticklabels() if t.get_text().strip()]
    if labels and max(t.get_window_extent(renderer).width for t in labels) > 1.0 * pitch_px:
        for t in ax.get_xticklabels():
            t.set_rotation(90)
            t.set_va("top")
        ax.figure.canvas.draw()
    if zone_offset_pt is None:
        # Measured under the sublabel row, whatever its orientation. A panel drawing ONE season has
        # nothing in that row -- the name is already in the title -- and passes blanks, so the
        # zone names then sit close under the axis instead of a whole empty line below it.
        floor = ax.get_window_extent().y0
        drops = [floor - t.get_window_extent(renderer).y0 for t in labels]
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
    """Storage attenuation as ONE stacked bar per zone: reservoir + battery = combined.

    This is an identity, not an approximation.  The Shapley split in ``_flex_event_metrics`` gives
    ``phi_bat + phi_res = sev_no_both - sev_actual``, so dividing through by ``sev_no_both``,
    ``reservoir_share + battery_share == combined_attenuation_pct`` exactly.  The working figure
    drew all three as separate boxes, which meant the third box was the sum of the other two —
    eighteen boxes carrying twelve numbers.  Stacking says the same thing in six bars and makes the
    decomposition visible instead of implied.

    ONE grouping parameter, not one per groupable column.  ``group_by`` is
    ``(column, values, sublabels)`` and ``where`` fixes every other selector, so the same builder
    draws Figure 5a (duration class inside each zone, at a fixed warming level) and the
    supplementary mechanism figure (warming level inside each zone, at a fixed duration class).
    A ``duration_class=`` argument beside a ``periods=`` argument would be two mutually exclusive
    keywords for one axis, and the second caller to want a third column would add a third.

    Whatever the grouping, HUE STAYS ON THE COMPONENT and the group is a labelled position, the
    convention ``panel_driver_attribution`` uses for season.  That is deliberate: bars of identical
    height and identical colour ARE an invariance, and shading them by the group would spend the
    encoding that carries the Shapley decomposition on a variable that mostly does not move.  Where
    it does move — Iberia across warming levels — the step down is legible precisely because its
    neighbours are flat.

    ``level_label`` states the fixed selector inside the axes.  Figure 5a needs it: the panel beside
    it draws all three warming levels, so without it a reader has no way to know this one is a
    level rather than a change.

    ``spread`` draws the inter-quartile range across chains ON THE TOTAL ONLY.  A stacked bar cannot
    carry spread per segment — two overlapping whiskers inside one bar destroy the decomposition
    that is the whole reason for stacking — but the combined attenuation is a quantity in its own
    right, and it is the number the caption quotes.  Reservoir and battery quartiles are printed by
    the notebook's table instead.
    """
    rows = attenuation
    for key, value in (where or {}).items():
        rows = rows[rows[key] == value]
    column, values, sublabels = group_by if group_by else (None, [None], None)
    grouped = len(values) > 1
    positions = np.arange(len(style.zones))
    # 0.62 of a whole slot for a single bar; grouped, the same 0.80 group span as the driver panel,
    # so a zone group reads as a group and the gap between zones survives.
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
    # The top-left corner is empty by construction rather than by luck: the tallest stack in any
    # zone reaches 95 % and the axis runs to 105.
    if level_label:
        ax.annotate(level_label, xy=(0.005, 0.98), xycoords="axes fraction", ha="left", va="top",
                    fontsize=FS["tick"], fontweight="bold", color="0.25")
    # Off by default: every bar runs from the floor to 36-95 %, so there is no free corner inside
    # this panel and an in-axes legend lands on the data. The composer places it at figure level.
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
    """Diverging stacked bars: each driver's contribution to the change in residual load.

    Season is a LABELLED POSITION, not a shade.  It was a shade while there were two of them and
    hue was already spent on the driver, but four shades of seven hues is twenty-eight fills a
    reader has to tell apart, and the two encodings interfere: a pale Demand bar in SON and a
    saturated Hydro RoR bar in DJF are the same ink.  Each stack now carries its season name
    underneath and every driver keeps its own colour at full strength, with the zone name on a
    second row below and a real gap between zone groups.

    ``group_span`` is the fraction of a zone's slot the stacks occupy; the remainder is the gap
    that separates one zone from the next.

    ``sublabels`` overrides the text written under each stack, which defaults to ``seasons``.  A
    panel drawing a single season passes blanks: repeating "DJF" under every stack of a panel whose
    title already says winter is one word printed once per zone for no reader.

    ``reserve_sign_gutter`` keeps the y-label clear of the sign-convention arrows even on panels
    that do not draw them -- see below.
    """
    width = group_span / len(seasons)
    positions = np.arange(len(style.zones))

    for s_index, season in enumerate(seasons):
        offset = (s_index - (len(seasons) - 1) / 2) * width
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
            if s_index == 0 and i:
                ax.axvline(i - 0.5, color="0.85", lw=0.4, zorder=0)

    ax.axhline(0, color="0.25", lw=0.5, zorder=3)
    ax.set_xlim(-0.5, len(style.zones) - 0.5)

    # ── Two-level x axis: the season under its own stack, the zone under its group ──────
    # No season legend any more: a label under the bar it names needs no key, and the key used to
    # sit inside the axes where it competed with the tallest stack.
    _grouped_zone_axis(ax, style,
                       [season_label(x) for x in seasons] if sublabels is None else sublabels, width,
                       show_zones=show_zone_labels)

    ax.set_ylabel(ylabel, fontsize=FS["label"])
    # The gutter is measured from the tick labels outward: first the widest tick label (the inset),
    # then fixed millimetre steps to the arrows, their labels and the y-label. Those steps were a
    # flat 7.1 / 9.6 / 12.4 mm from the axis when ticks were 5 pt; at 7 pt the tick labels alone
    # grew past the arrows. As plain axes fractions they would scale with the panel, so Figure 4's
    # full-width version pushed "GWh" 28 mm into the margin -- hence millimetres.
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    extent = ax.get_window_extent()
    width_mm = extent.width / ax.figure.dpi * 25.4
    ticks = [t for t in ax.get_yticklabels() if t.get_visible() and t.get_text().strip()]
    inset_mm = (max((extent.x0 - t.get_window_extent(renderer).x0 for t in ticks), default=0.0)
                / ax.figure.dpi * 25.4)
    # `show_sign_convention=False` is for a figure that stacks several of these panels: the arrows
    # say what the sign means, which is a property of the quantity and not of the row, so six copies
    # down one gutter is six times the ink for one statement. The y-label keeps the SAME offset
    # either way, so the stacked panels stay aligned with the one that does carry the arrows.
    if show_sign_convention:
        # "deepens" / "eases", not "... the deficit": at 7 pt each full phrase is ~25 mm of rotated
        # text and these axes are 30-40 mm tall, so the two ran into each other at the zero line.
        # The arrows' colours and every caption carry "the deficit".
        _annotate_sign_convention(ax, arrow_x=-(inset_mm + 1.5) / width_mm,
                                  label_x=-(inset_mm + 3.7) / width_mm,
                                  labels=SIGN_LABELS)
    # ...but that alignment is only worth paying for down a SHARED gutter. Panels set side by side
    # share none, so a panel without arrows reserving room for them pushes its y-label -- and the
    # panel letter, which `_letter_anchor` pins to the axis's leftmost extent -- into the margin.
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
    """One duration regime: counts, the seasonal decomposition, and what drives the change.

    Lives here rather than in a notebook because it draws BOTH regimes and they now sit in
    different documents — the sub-daily figure in the main text, the multi-day one in the
    supplement. Two copies of a builder that must stay comparable is how they stop being
    comparable.

    ``diurnal`` of ``None`` drops the hour-of-day panel and gives the driver panel the full bottom
    row.  A multi-day event spans whole days, so an hour-of-day profile of one would be
    meaningless; that is why only the sub-daily call passes a table.
    """
    is_sub = diurnal is not None
    fig = plt.figure(figsize=(WIDTH, height_mm * MM))
    # hspace holds, between the rows: panel a's rotated zone names, the warming-level key under
    # panel a, and the bottom row's titles. At 7-8 pt type 0.52 put the key inside panel c.
    grid = gridspec.GridSpec(2, 2, hspace=0.78, wspace=0.30, figure=fig)

    ax_a = fig.add_subplot(grid[0, 0])
    # Black, not the red-up/blue-down ramp: that ramp is blue-to-red and so is the warming-level
    # ramp of the boxes underneath it, so two colour systems were competing inside one panel for a
    # distinction the +/- sign already carries.
    panel_event_frequency(ax_a, events_annual, regime, style,
                          f"{'sub-daily' if regime == 'sub_daily' else 'multi-day'} events yr$^{{-1}}$",
                          annotation_color="black")
    ax_a.set_title(titles["a"], fontsize=FS["title"], fontweight="bold")

    # b is the same panel in both figures, so the two read in the same order: which season each
    # zone's change comes from. ALL FOUR seasons, because two hid the answer — of the four zones
    # whose multi-day events increase, only the South-East grows in summer, while the British Isles
    # grow across winter, spring and autumn and Central-West grows in autumn alone.
    ax_b = fig.add_subplot(grid[0, 1])
    panel_season_contribution(ax_b, events_by_season, events_annual, style, seasons, regime,
                              edges=season_bins)
    ax_b.set_title(titles["b"], fontsize=FS["title"], fontweight="bold")

    if is_sub:
        # All four panels the same width. An earlier version split this row 1.30/0.70 because the
        # driver panel had just been cut to two zones and looked empty beside the diurnal one; the
        # cure was worse than the complaint, since c and d are then two different sizes. The row is
        # still NESTED so a future ratio on the outer grid cannot reach panel b's heatmap.
        bottom = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=grid[1, :], wspace=0.40)
        cell = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=bottom[0], hspace=0.18)
        ax_c = fig.add_subplot(cell[0])
        panel_hour_of_day(ax_c, fig.add_subplot(cell[1]), diurnal, style,
                          **(diurnal_windows or {}))
        ax_c.set_title(titles["c"], fontsize=FS["title"], fontweight="bold")
        ax_drivers = fig.add_subplot(bottom[1])
        drivers_letter = "d"
    else:
        # The full bottom row, not one cell of it: four seasons over both columns is 5.8 mm per
        # stack against 5.1 mm for two seasons in half the width, so the panel gets LESS dense, and
        # it fills the quadrant this figure would otherwise leave empty.
        ax_c = ax_drivers = fig.add_subplot(grid[1, :])
        drivers_letter = "c"

    driver_style = replace(style, zones=list(driver_zones)) if driver_zones else style
    panel_driver_attribution(ax_drivers, attribution, list(driver_seasons), driver_style)
    ax_drivers.set_title(titles[drivers_letter], fontsize=FS["title"], fontweight="bold")

    fit_width(fig)
    # The warming-level key sits BETWEEN the rows rather than above the figure: it belongs to panel
    # a, and at the top of the page it read as a title for the whole figure. `legend_below` measures
    # off the top row's tight bbox, so it clears the rotated zone labels whatever they happen to be.
    gwl_h, gwl_l = gwl_handles(style)
    # Under panel a alone, the panel it belongs to; centred under a AND b it read as b's key.
    legend_below(fig, [ax_a], gwl_h, gwl_l, ncol=len(gwl_l), title=GWL_TITLE)
    # The driver key is measured against the panel it belongs to, so it sits the same short distance
    # below it whatever the rotated zone labels happen to be.
    drv_h, drv_l = driver_handles(style)
    legend_below(fig, ax_drivers, drv_h,
                 [(driver_display or {}).get(name, name) for name in drv_l], ncol=4)
    lettered = [ax_a, ax_b, ax_c] + ([ax_drivers] if is_sub else [])
    panel_letters(fig, list(zip(letters, lettered)))
    return fig


def panel_zone_map(ax, countries, style, zone_members, zone_colors):
    """The six balancing zones, as S4 draws them."""
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
    # `loc` names the part of the legend that lands ON the anchor, so "lower center" at y=-0.13
    # put the legend's BOTTOM 5.9 mm below the axes and let it grow upward, back over southern
    # Europe. "upper center" anchors its top there instead, so it grows downward into free space.
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=zone_colors[z]) for z in style.zones],
              labels=style.labels(), fontsize=FS["tick"], frameon=False, ncol=2,
              loc="upper center", bbox_to_anchor=(0.5, -0.01), handlelength=1.0,
              handletextpad=0.5, columnspacing=1.0)


def panel_southern_survival(axes, residual_load_by_model, period_defs, zone_threshold,
                            season_months, style, zones, seasons, spread):
    """2a trimmed to the two southern zones — the same call sequence tail_exceedance_figure makes."""
    # Imported here rather than at module scope for the same reason tail_exceedance_figure does it:
    # extremes_analysis imports from this module, so a top-level import would close the cycle.
    from .extremes_analysis import (compute_exceedance_probability, compute_exceedance_survival,
                                    compute_survival_x_grid)
    for column, zone in enumerate(zones):
        threshold = float(zone_threshold.sel(zone=zone))
        x = compute_survival_x_grid(residual_load_by_model, period_defs, zone,
                                    [season_months[s] for s in seasons], style.period_names,
                                    threshold=threshold)
        for row, season in enumerate(seasons):
            ax = axes[row][column]
            _, survival = compute_exceedance_survival(
                residual_load_by_model, period_defs, zone, season_months[season], style.period_names,
                spread=spread, x=x)
            p_ref = compute_exceedance_probability(residual_load_by_model, period_defs, zone,
                                                   season_months[season], style.reference_period, threshold)
            p_warm = compute_exceedance_probability(residual_load_by_model, period_defs, zone,
                                                    season_months[season], style.warmest_period, threshold)
            survival_axes(ax, survival, x, threshold, style,
                          ratio=(p_warm / p_ref if p_ref > 0 else np.nan), show_legend=False)
            ax.tick_params(labelsize=FS["tick"])
            if row == 0:
                ax.set_title(style.zone_labels[zone], fontsize=FS["title"], fontweight="bold")
                # The row below carries the axis; repeating it here puts a tick row in the gap.
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("residual load per 3h step (GWh)", fontsize=FS["label"])
            if column == 0:
                ax.set_ylabel(f"{season_label(season)}\nP(RL > x)", fontsize=FS["label"])
            else:
                ax.set_yticklabels([])


def zone_tail_duration_figure(*, countries, residual_load_by_model, period_defs,
                              zone_threshold, season_months, style, southern_zones,
                              seasons, spread, zone_members, zone_colors, duration_spectrum,
                              bin_labels, layout="beside", map_share=0.74):
    """Figure 2 as four panels, in one of two arrangements and at any map width.

    The two differ ONLY in where the map sits, because that is the whole question: Europe over
    -12 to 40 degrees of longitude and 33 to 72 of latitude wants a cell about 0.85 as wide as it
    is tall (a degree of longitude is ~0.64 of a degree of latitude at 50 N), and `set_aspect("auto")`
    stretches it to whatever cell it is given. `map_share` is that cell's share of a ratio pair
    summing to 2.0, so it is the one knob the sweep below turns; everything else is fixed.

    Widening helps BOTH arrangements, which is the opposite of what "the map looks squeezed" first
    suggests: as first rendered A sat at 0.69 and B at 0.50 against a true 0.85, so both were too
    NARROW, not too wide. What widening costs differs though — in A the map takes width from the
    survival block alone, in B from the survival block and from c and d together, because they share
    the right-hand column.

    Both nest the bottom row in its own equal-column sub-gridspec. The first version put
    `width_ratios` on the whole 2x2 grid, which applied the top row's 0.78/1.22 split to the bottom
    row as well and left panel c at ~60 mm against panel d at ~94 mm — two panels of the same kind,
    meant to be read together, at a 1.57x size difference.
    """
    if layout == "beside":
        fig = plt.figure(figsize=(WIDTH, 100 * MM))
        outer = gridspec.GridSpec(2, 1, height_ratios=(1.0, 0.66), hspace=0.55, figure=fig)
        top = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0], wspace=0.24,
                                               width_ratios=(map_share, 2.0 - map_share))
        map_cell, survival_spec = top[0], top[1]
        bottom_spec = outer[1]
    else:                                          # "spanning": the map takes the full height
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
    survival = [[fig.add_subplot(survival_cell[r, c]) for c in range(2)] for r in range(2)]
    panel_southern_survival(survival, residual_load_by_model, period_defs, zone_threshold,
                            season_months, style, southern_zones, seasons, spread)
    survival[0][0].set_title(style.zone_labels[southern_zones[0]], fontsize=FS["title"], fontweight="bold")

    # Nested and equal-columned, so the ratio above cannot reach the two spectrum panels.
    bottom_cell = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=bottom_spec, wspace=0.30)
    bottom = [fig.add_subplot(bottom_cell[0]), fig.add_subplot(bottom_cell[1])]
    positions = np.arange(len(bin_labels))
    boundary = bin_labels.index("12-24 h") + 0.5
    for ax, value, label in zip(bottom, ("counts", "severity"),
                                ("events yr$^{-1}$", "severity (TWh yr$^{-1}$)")):
        spectrum = duration_spectrum(value)
        for period in style.period_names:
            block = spectrum[spectrum.period == period]
            median = [block[block.bin == b]["value"].median() for b in bin_labels]
            # Through model_spread_bounds rather than hardcoded quartiles, so this band answers to
            # `style.spread` like every other band. Hardcoded, it would have silently ignored a
            # change of convention and left these two panels on a different one from the rest.
            bounds = [model_spread_bounds(block[block.bin == b]["value"].values, style.spread)
                      for b in bin_labels]
            low = [lo for lo, _ in bounds]
            high = [hi for _, hi in bounds]
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
    # The convention the rest of the paper now uses: the title names the quantity and the y-label
    # carries the unit. "Where the events are" / "Where the energy is" were message titles left
    # over from before Figures 3-6 were converted.
    bottom[0].set_title("Frequency of events by duration", fontsize=FS["title"], fontweight="bold")
    bottom[1].set_title("Severity by duration", fontsize=FS["title"], fontweight="bold")

    fit_width(fig)
    legend_below(fig, bottom, *gwl_handles(style), ncol=3, title=GWL_TITLE)
    panel_letters(fig, [("a", ax_a), ("b", survival[0][0]),
                        ("c", bottom[0]), ("d", bottom[1])])
    return fig


# 0.50 is in the list because the first sweep straddled the optimum without landing on it: at 0.42
# the map came out 15 % too narrow and at 0.58 17 % too wide. The map's HEIGHT is fixed by the
# layout (44.9 mm in A, 103.3 mm in B), so true shape needs 0.85 x that height in width — about
# 38 mm in A, and about 88 mm in B, which is half the page and out of reach here.


# ══════════════════════════════════════════════════════════════════════════════════════════
# Figure 5 — scenario sensitivity
# ══════════════════════════════════════════════════════════════════════════════════════════
def panel_dumbbell(ax, by_scenario: Mapping[str, pd.DataFrame], style: PaperStyle,
                   *, value_column: str = "freq_per_year", duration_class: str | None = None,
                   ylabel="", headroom: float = 1.12):
    """Level at the reference and warmest level, as a dot-to-dot per zone and build-out.

    Replaces a pair of bars drawn at the SAME x with no zorder, where the "hollow" part was an
    artifact of draw order rather than an encoding.  A dumbbell states level and change in one
    mark, needs no fill, and stays readable at 180 mm where four bars per zone would not.

    ``value_column`` chooses the quantity and ``duration_class`` of ``None`` skips the class filter,
    so Figure 6's three panels — an annual energy and two event counts — are one builder.

    NO SPREAD AND NO ANNOTATION, both measured decisions rather than omissions.  A 1.5xIQR whisker
    drawn on the marker runs straight through the other end of its own dumbbell (Iberia's LOW
    sub-daily spread spans 9 to 41 events per year across an open circle at 13.1, so a reader cannot
    tell which marker it belongs to), and giving each warming level its own whisker does not fit:
    four positions in a zone slot leave 1.53 mm of pitch against a 1.06 mm marker plus a 0.85 mm
    cap.  The agreement counts and the warming-sensitivity ratios live in the caption instead.
    """
    scenarios = list(by_scenario)
    positions = np.arange(len(style.zones))
    span = 0.7 / max(len(scenarios), 1)

    for k, (label, table) in enumerate(by_scenario.items()):
        color = SCENARIO_COLORS[k % len(SCENARIO_COLORS)]
        offset = (k - (len(scenarios) - 1) / 2) * span
        data = table if duration_class is None else table[table.duration_class == duration_class]
        ref, warm = [np.array([data[(data.zone == z) & (data.period == p)][value_column].median()
                               for z in style.zones])
                     for p in (style.reference_period, style.warmest_period)]
        for x, low, high in zip(positions + offset, ref, warm):
            if not (np.isfinite(low) and np.isfinite(high)):
                continue
            ax.annotate("", xy=(x, high), xytext=(x, low),
                        arrowprops=dict(arrowstyle="-|>,head_width=0.13,head_length=0.3",
                                        color=color, lw=0.8, shrinkA=0.8, shrinkB=0.0),
                        zorder=2)
        ax.plot(positions + offset, ref, "o", ms=3.0, mfc="white", mec=color, mew=0.8, zorder=3,
                label=label)
        ax.plot(positions + offset, warm, "o", ms=3.0, color=color, zorder=3)

    ax.set_ylim(min(0, ax.get_ylim()[0]), ax.get_ylim()[1] * headroom)
    ax.set_xticks(positions)
    ax.set_xticklabels(style.labels(), fontsize=FS["tick"], rotation=30, ha="right")
    ax.set_xlim(-0.6, len(style.zones) - 0.4)
    ax.set_ylabel(ylabel, fontsize=FS["label"])
    ax.tick_params(labelsize=FS["tick"])


#: Fill strength of the two warming levels in :func:`panel_scenario_bars`: the reference level is
#: the scenario colour washed out, the warmest level the colour itself.
LEVEL_ALPHA = (0.35, 0.90)


def panel_scenario_bars(ax, by_scenario: Mapping[str, pd.DataFrame], style: PaperStyle,
                        *, value_column: str = "freq_per_year", duration_class: str | None = None,
                        ylabel="", headroom: float = 1.08):
    """Level at the reference and warmest level per zone and build-out, as grouped bars.

    Replaces :func:`panel_dumbbell` in Figure 5 after internal review: with open and filled dots
    joined by an arrow it was "not obvious which green and purple dots belonged together". Here each
    zone carries two PAIRS of bars -- one pair per build-out, in the scenario colour -- and within a
    pair the washed-out bar is the reference level and the solid one the warmest. The gaps are the
    encoding: bars of one pair touch, the two pairs of a zone are a small gap apart, and zones are
    separated by a gap more than twice that, so the eye groups by zone first, then scenario.

    Filled with blended solid colours rather than alpha, so nothing darkens where bars meet.
    Medians only; the agreement counts and spread stay in the caption and Table 3, as before.
    """
    scenarios = list(by_scenario)
    positions = np.arange(len(style.zones))
    bar = 0.17                                   # one bar
    pair_gap = 0.09                              # between the two build-outs of a zone
    pair_centres = [(k - (len(scenarios) - 1) / 2) * (2 * bar + pair_gap)
                    for k in range(len(scenarios))]

    for k, (label, table) in enumerate(by_scenario.items()):
        color = SCENARIO_COLORS[k % len(SCENARIO_COLORS)]
        data = table if duration_class is None else table[table.duration_class == duration_class]
        for j, period in enumerate((style.reference_period, style.warmest_period)):
            values = np.array([data[(data.zone == z) & (data.period == period)][value_column].median()
                               for z in style.zones])
            x = positions + pair_centres[k] + (j - 0.5) * bar
            ax.bar(x, np.nan_to_num(values), bar, color=blend(color, LEVEL_ALPHA[j]),
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
    """Legend handles for the washed-out/solid warming-level encoding of `panel_scenario_bars`."""
    return ([mpatches.Patch(facecolor=blend(color, a), edgecolor=color, linewidth=0.4)
             for a in LEVEL_ALPHA],
            [style.reference_period, style.warmest_period])


def panel_binned_attenuation(ax, per_event: pd.DataFrame, zone: str | None, style: PaperStyle,
                             *, duration_class: str | None = "sub_daily", n_bins: int = 5,
                             bin_column: str = "sev_no_storage", log_x: bool = True,
                             ylabel="share of severity balanced (%)", spread: bool = True,
                             xlabel="event severity without storage (GWh)") -> pd.DataFrame:
    """Median attenuation against the counterfactual severity it faced, one line per warming level.

    The falsification test for the Iberian result.  "Absorption degrades under warming" has an
    obvious rival explanation — the events simply got bigger, and storage of a fixed size absorbs a
    smaller share of a larger event — which no aggregate can distinguish.  Conditioning on the
    counterfactual severity does: if the curve shifts DOWN at fixed event size, the degradation is
    in the asset rather than in the load.

    Bins are quantiles of the pooled ``bin_column``, so each carries the same number of events and
    each zone is binned on its own scale (Iberia's events are an order of magnitude larger than
    Central-East's).  Returns the drawn table so the caller can assert on it rather than read it off
    the picture.

    ``zone`` or ``duration_class`` of ``None`` skips that filter, and ``bin_column="duration_h"``
    turns the panel into attenuation against event LENGTH rather than event size — which answers a
    different question with the same machinery: whether storage's capability depends on the
    timescale the paper splits events by.
    """
    rows = per_event
    if zone is not None:
        rows = rows[rows.zone == zone]
    if duration_class is not None:
        rows = rows[rows.duration_class == duration_class]
    rows = rows.copy()
    edges = np.percentile(rows[bin_column].dropna(), np.linspace(0, 100, n_bins + 1))
    rows["bin"] = pd.cut(rows[bin_column], np.unique(edges), include_lowest=True)
    drawn = rows.groupby(["bin", "period"], observed=True)["atten"].median().unstack()
    centres = np.array([interval.mid for interval in drawn.index])

    # The band is the spread of EVENTS inside each bin, not of chains: the question this panel asks
    # is whether the curve moves at fixed event size, and a chain-level band would be reporting on
    # how many events each chain contributed to a bin rather than on the attenuation itself.
    band = {stat: rows.groupby(["bin", "period"], observed=True)["atten"].quantile(q).unstack()
            for stat, q in (("lo", 0.25), ("hi", 0.75))} if spread else None
    for period in style.period_names:
        if period not in drawn:
            continue
        if band is not None and period in band["lo"]:
            ax.fill_between(centres, band["lo"][period].values, band["hi"][period].values,
                            color=style.gwl_colors[period], alpha=0.13, lw=0, zorder=2)
        ax.plot(centres, drawn[period].values, "-o", ms=2.0, lw=0.9,
                color=style.gwl_colors[period], label=period, zorder=3)
    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel, fontsize=FS["label"])
    ax.set_ylabel(ylabel, fontsize=FS["label"])
    ax.tick_params(labelsize=FS["tick"])
    return drawn


# ══════════════════════════════════════════════════════════════════════════════════════════
# Shared legend handles — figure-level, so no panel has to give up space to them
# ══════════════════════════════════════════════════════════════════════════════════════════
def gwl_handles(style: PaperStyle):
    return ([plt.Line2D([], [], marker="s", ls="", ms=3.5, color=style.gwl_colors[p])
             for p in style.period_names], list(style.period_names))


def storage_handles(style: PaperStyle):
    """Just the two segments. The stack IS the total, so a separate key said it twice."""
    return ([plt.Rectangle((0, 0), 1, 1, color=color) for _, _, color in STORAGE_PARTS],
            [label for _, label, _ in STORAGE_PARTS])


def driver_handles(style: PaperStyle):
    handles = [plt.Rectangle((0, 0), 1, 1, color=style.driver_palette[d]) for d in style.drivers]
    handles.append(plt.Line2D([], [], marker="D", ls="", ms=2.5, color="black"))
    return handles, list(style.drivers) + ["net \u0394"]


# ══════════════════════════════════════════════════════════════════════════════════════════
# S4 and S5 — characterising the climate input
# ══════════════════════════════════════════════════════════════════════════════════════════
#: The five GCMs driving this paper's chains, in a fixed order so S4 and S5 colour them alike.
DRIVING_GCM_ORDER = ["CNRM-CM5", "EC-EARTH", "HadGEM2-ES", "MPI-ESM-LR", "NorESM1-M"]
#: Qualitative, colourblind-safe (Okabe-Ito). NOT the warming-level ramp: these name models, not
#: an ordered quantity, so a sequential ramp would imply a ranking that does not exist.
DRIVING_GCM_COLORS = {"CNRM-CM5": "#0072B2", "EC-EARTH": "#009E73", "HadGEM2-ES": "#D55E00",
                      "MPI-ESM-LR": "#CC79A7", "NorESM1-M": "#E69F00"}
RCM_MARKERS = {"CNRM-ALADIN63": "o", "MOHC-HadREM3-GA7-05": "^"}


def panel_gcm_gmst(ax, gmst, *, driving, warming_levels, level_colors) -> None:
    """Global-mean warming of every CMIP5 model, with the chains' five drivers picked out.

    Every model thin and grey, the five over the top. Grey is drawn first so the coloured lines are
    never buried and the ensemble reads as context rather than as data of its own. The paper's
    warming levels are horizontal rules, because they are what a companion panel turns into dates.
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
    """S4: where the five driving GCMs sit in the CMIP5 ensemble.

    ``gmst`` is the cached anomaly table (``model`` x ``year``), ``crossing`` a frame of
    ``model, ensemble, warming_level, start_year`` over the whole ensemble, and ``driving`` maps
    each driving GCM to the ``model`` key it has in both.

    Two panels because the question has two currencies: **a** is how much each model warms, **b**
    is when it crosses the levels this paper is stated at — which is what actually selects the
    20-year window every result is computed over.
    """
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, height_mm * MM),
                             gridspec_kw={"wspace": 0.26, "width_ratios": (1.12, 0.88)})
    ax_a, ax_b = axes
    panel_gcm_gmst(ax_a, gmst, driving=driving, warming_levels=warming_levels,
                   level_colors=level_colors)

    # b — one strip per level: every run's crossing year as a dot, the five as taller ticks. A
    # strip rather than a histogram because n is small (90 runs) and the individual positions are
    # the point; binning them would hide exactly the comparison being made.
    rng = np.random.RandomState(0)
    for row, level in enumerate(warming_levels):
        block = crossing[crossing["warming_level"] == level]
        others = block[~block["model"].isin(driving)]
        # A little vertical jitter, seeded, so overlapping years stay countable.
        ax_b.scatter(others["start_year"], row + rng.uniform(-0.16, 0.16, len(others)),
                     s=4, color="0.72", alpha=0.7, lw=0, zorder=1)
        for gcm in DRIVING_GCM_ORDER:
            hit = block[block["model"] == gcm]
            if hit.empty:
                continue
            ax_b.plot([float(hit["start_year"].iloc[0])] * 2, [row - 0.30, row + 0.30],
                      lw=1.3, color=DRIVING_GCM_COLORS[gcm], solid_capstyle="butt", zorder=3)
        median = float(block["start_year"].median())
        ax_b.plot([median] * 2, [row - 0.38, row + 0.38], lw=0.8, color="0.25", ls="--", zorder=2)
        # The run count per level lives in the caption, not on the panel: it is the same number for
        # all three rows here, so printing it three times was three labels carrying one fact.
    ax_b.set_yticks(range(len(warming_levels)))
    ax_b.set_yticklabels([f"+{level:.1f} °C" for level in warming_levels], fontsize=FS["label"])
    ax_b.set_ylim(len(warming_levels) - 0.55, -0.45)
    ax_b.set_xlabel("first year of the 20-year window", fontsize=FS["label"])
    ax_b.tick_params(labelsize=FS["tick"])
    ax_b.grid(True, axis="x", color="0.93", lw=0.3)
    ax_b.set_axisbelow(True)

    fit_width(fig)
    handles = [plt.Line2D([], [], color="0.72", lw=1.0, label="other CMIP5 models")]
    handles += [plt.Line2D([], [], color=DRIVING_GCM_COLORS[g], lw=1.4, label=g)
                for g in DRIVING_GCM_ORDER if g in driving]
    legend_below(fig, list(axes), handles, [h.get_label() for h in handles], ncol=6)
    panel_letters(fig, [("a", ax_a), ("b", ax_b)])
    return fig


def model_characterisation_figure(table, *, rcm_labels, warming_levels, show_drivers: bool,
                                  height_mm: float = 56, width_mm: float = 110,
                                  y_column: str = "pr_mm_per_day",
                                  y_label: str = "change in precipitation\n(mm d$^{-1}$)",
                                  show_gcm: bool = True,
                                  quadrant_labels: tuple[str, str] = ("wetter", "drier")) -> Figure:
    """S5: the change in mean European temperature and a hydrological variable between two levels.

    One quadrant panel. ``table`` is the cached per-model table; the change is taken between the
    lowest and highest ``warming_levels`` **within each model**, using that model's own windows.

    A CHANGE rather than the absolute state, because the absolute state of a bias-corrected chain
    is the reanalysis by construction — plotting it shows only that the correction worked. What the
    ensemble actually contributes to the paper is the response, which is this.

    ``show_drivers`` produces the S5a/S5b pair: with it, the five GCMs that drive the chains are
    drawn HOLLOW in their own colour and the chains FILLED in the same colour, so a chain can be
    read against its driver; without it, every GCM is grey and only the chains carry colour.

    ``y_column`` / ``y_label`` / ``quadrant_labels`` swap the vertical variable, and ``show_gcm``
    drops the ensemble cloud entirely.  Together they give S5c, the runoff version: HadGEM2-ES and
    EC-EARTH archive no runoff anywhere in CMIP5, so a grey cloud there would be missing the drivers
    of three of the nine chains -- better no ensemble than a partial one presented as the ensemble.
    The x axis stays the domain-mean temperature change, so a chain sits at the SAME x in all three
    panels and only its height changes.
    """
    fig, ax = plt.subplots(figsize=(width_mm * MM, height_mm * MM))
    first, last = min(warming_levels), max(warming_levels)

    def delta(block):
        """(dT, dY) per model between the two levels, keyed by model."""
        wide = block.pivot_table(index="model", columns="warming_level",
                                 values=["tas_degC", y_column])
        return pd.DataFrame({"dT": wide[("tas_degC", last)] - wide[("tas_degC", first)],
                             "dY": wide[(y_column, last)] - wide[(y_column, first)]})

    gcm = table[table["source"] == "gcm"]
    rcm = table[table["source"] == "rcm"]
    rcm_delta = delta(rcm)
    # Not merely unplotted: for a y variable the ensemble does not carry, pivoting the GCM block
    # gives an all-NaN column, so it is not computed at all rather than computed and discarded.
    gcm_delta = delta(gcm) if show_gcm else None
    driving = (set(gcm[gcm["driving"].fillna(False)]["model"])
               if show_gcm and "driving" in gcm else set())

    # The quadrant rules. Both are real zeros; only dP straddles its own, because +1.5 -> +3.0 C of
    # GLOBAL warming necessarily warms Europe, so no model can land left of dT = 0. The limits
    # therefore follow the data instead of being forced symmetric, which would leave half the panel
    # provably empty and squeeze the spread into a sixth of the axis.
    ax.axhline(0.0, color="0.45", lw=0.7, zorder=1)
    # The vertical zero is drawn only when the ensemble is: it separates nothing either way -- no
    # model cools between the two levels -- and it earns its place in a/b by anchoring the left end
    # of a cloud that reaches +0.45 K. With the chains alone (+1.30..+1.93 K) it does the opposite,
    # stretching the axis back to zero and leaving two thirds of the panel empty.
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

    # Name the quadrants rather than leaving the reader to work out the sign convention. Placed in
    # the corners of the axes, so they never collide with the cloud.
    for label, (fx, fy), va in ((quadrant_labels[0], (0.985, 0.97), "top"),
                                (quadrant_labels[1], (0.985, 0.03), "bottom")):
        ax.annotate(label, xy=(fx, fy), xycoords="axes fraction", ha="right", va=va,
                    fontsize=FS["tick"], color="0.45", style="italic")

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
    handles += [plt.Line2D([], [], color=DRIVING_GCM_COLORS[g], marker="s", ls="none", ms=3.4,
                           label=g) for g in DRIVING_GCM_ORDER]
    # ncol=5 so the key breaks after the marker-TYPE entries and the five colours occupy their own
    # run: matplotlib fills a legend column-major, and at ncol=4 an RCM shape landed directly under
    # "driving GCM" as though it were a kind of GCM.
    legend_below(fig, ax, handles, [h.get_label() for h in handles], ncol=5)
    return fig


# ══════════════════════════════════════════════════════════════════════════════════════════
# Figure 6 and Supplementary Fig. S31 — the Europe-wide change, demand against supply
# ══════════════════════════════════════════════════════════════════════════════════════════
def _ramp_shade(cmap, at: float):
    return mcolors.to_hex(plt.get_cmap(cmap)(at) if isinstance(cmap, str) else cmap(at))


#: Colour by CATEGORY, so a reader groups the rows by what they are before reading any label:
#: demand warm (PuOr's orange arm), supply cool (RdBu's blue arm), the residual load grey. Per
#: category (component IQR, total IQR, median): the box is light for components and mid for the
#: category total, and the median takes the category's darkest shade on every row. Not Figure 7's
#: green/purple (supply SCENARIOS there) and not used for warming levels anywhere in this figure.
CATEGORY_COLOR = {"demand": ("#fdb863", "#e08214", "#7f3b08"),
                  "supply": ("#92c5de", "#4393c3", "#053061"),
                  "net": ("#d9d9d9", "#969696", "#252525")}
QUANTITY_CATEGORY = {"Heating demand": ("demand", 0), "Cooling demand": ("demand", 0),
                     "Total demand": ("demand", 1), "Solar": ("supply", 0), "Wind": ("supply", 0),
                     "Hydro": ("supply", 0), "Total supply": ("supply", 1),
                     "Residual load": ("net", 1)}
QUANTITY_COLOR = {q: CATEGORY_COLOR[c][k] for q, (c, k) in QUANTITY_CATEGORY.items()}
QUANTITY_MEDIAN = {q: CATEGORY_COLOR[c][2] for q, (c, _) in QUANTITY_CATEGORY.items()}
#: How the min-max extensions of a Figure 3 box are drawn: "outline" (white, edged in the IQR
#: colour) or "tint" (a paler shade of the IQR colour).
EUROPE_EXTENSIONS = ("outline", "tint")


def europe_change_table(rows, columns, reference: str, warmest: str, *,
                        members=None, dim: str = "country") -> dict:
    """Per-chain Europe-wide change for every (column, row).

    ``rows``: ``[(label, builder)]`` with ``builder(period, months)`` returning a per-chain field
    whose every dimension except ``model`` is summed to the Europe total (countries or zones).
    ``columns``: ``{name: months or None}``. Returns ``{(column, label): DataFrame}`` indexed by
    chain with ``ref``, ``twh`` (change) and ``pct`` (change over ``|ref|``: the residual load is a
    SURPLUS annually, and a plain ratio would flip the sign of every per cent, as in Table 1).

    ``members``: restrict every field to these entries of ``dim`` before summing (a balancing zone);
    entries a field does not carry are skipped, as Table 1 does.
    """
    out = {}
    for column, months in columns.items():
        for label, builder in rows:
            fields = {}
            for period in (reference, warmest):
                field = builder(period, months)
                if members is not None:
                    present = [m for m in members if m in set(map(str, field[dim].values))]
                    field = field.sel({dim: present})
                fields[period] = field.sum([d for d in field.dims if d != "model"], skipna=True)
            shared = sorted(set(map(str, fields[reference]["model"].values))
                            & set(map(str, fields[warmest]["model"].values)))
            ref = fields[reference].sel(model=shared).to_series()
            change = fields[warmest].sel(model=shared).to_series() - ref
            out[column, label] = pd.DataFrame({"ref": ref, "twh": change,
                                               "pct": 100 * change / ref.abs()})
    return out


#: Statistics columns of Table 1 / Supplementary Table S1, in order.
EUROPE_SUMMARY_COLUMNS = ("median", "P25", "P75", "min", "max", "sign agreement")


def europe_change_summary(change: dict, labels, columns, *, value: str = "twh",
                          negligible: float = 0.05) -> dict:
    """Table 1 (and Supplementary Table S1 per zone): ``{column: DataFrame}``, one table per time
    scale, one row per quantity, with the statistics Figure 3 draws -- median, quartiles, min and max
    across the chains -- plus how many chains share the median's sign (``"9/9"``). Where the median
    is below ``negligible`` (winter cooling) its sign means nothing, and the agreement is a dash."""
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
    """A DataFrame as a GitHub/Pandoc pipe table: signed numbers with a true minus sign, text as is.
    A MultiIndex is written as one column per level, each repeated value blanked after its first row."""
    def cell(value):
        if isinstance(value, (int, float, np.floating, np.integer)) and not isinstance(value, bool):
            text = f"{value:+.{decimals}f}"
            return "0.0" if float(text) == 0 else text.replace("-", "\u2212")
        return str(value)

    levels = frame.index.nlevels
    names = list(frame.index.names) if levels > 1 else [index_label]
    header = [n or "" for n in names] + [str(c) for c in frame.columns]
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join([":---"] * levels + ["---:"] * len(frame.columns)) + "|"]
    previous = (None,) * levels
    for key, row in frame.iterrows():
        key = key if isinstance(key, tuple) else (key,)
        shown = [str(k) if k != previous[i] or any(key[j] != previous[j] for j in range(i)) else ""
                 for i, k in enumerate(key)]
        previous = key
        lines.append("| " + " | ".join(shown + [cell(v) for v in row.values]) + " |")
    return "\n".join(lines)


def _europe_change_spans(change, labels, columns, value, share_seasons=True):
    """x-limits per column covering every chain (the min-max extensions), seasons shared in TWh."""
    spans = {}
    for column in columns:
        lo = min(change[column, l][value].min() for l in labels)
        hi = max(change[column, l][value].max() for l in labels)
        spans[column] = (min(lo, 0.0), max(hi, 0.0))
    seasonal = [c for c in columns if c != "Annual"]
    if share_seasons and value == "twh" and len(seasonal) > 1:
        joint = (min(spans[c][0] for c in seasonal), max(spans[c][1] for c in seasonal))
        spans.update({c: joint for c in seasonal})
    return spans


def europe_change_panel(ax, change, labels, column, *, value="twh", span=None,
                        extension="tint", height=0.34):
    """One column of the Europe-wide figure: per quantity one box across the nine chains -- the
    interquartile range filled, extended to the minimum and maximum, the median a line across it.

    ``extension``: ``"outline"`` draws the two extensions white with an edge in the IQR colour,
    ``"tint"`` fills them with a paler shade of it (see ``EUROPE_EXTENSIONS``).
    """
    if extension not in EUROPE_EXTENSIONS:
        raise ValueError(f"extension={extension!r}: expected one of {EUROPE_EXTENSIONS}")
    positions = np.arange(len(labels))[::-1]
    for y, label in zip(positions, labels):
        series = change[column, label][value]
        lo, q25, med, q75, hi = (series.min(), series.quantile(0.25), series.median(),
                                 series.quantile(0.75), series.max())
        color = QUANTITY_COLOR.get(label, "0.6")
        outline = extension == "outline"
        for left, right in ((lo, q25), (q75, hi)):
            ax.barh(y, right - left, left=left, height=height, zorder=1,
                    facecolor="white" if outline else blend(color, 0.35),
                    edgecolor=color if outline else "none", linewidth=0.8 if outline else 0)
        ax.barh(y, q75 - q25, left=q25, height=height, facecolor=color, zorder=2,
                edgecolor=color if outline else "none", linewidth=0.8 if outline else 0)
        ax.plot([med, med], [y - height / 2, y + height / 2], color=QUANTITY_MEDIAN.get(label, "0.2"),
                lw=1.4, solid_capstyle="butt", zorder=3)
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
    """Legend: the two categories (total shade), then the three statistics in neutral grey.

    No residual-load entry: its grey would be indistinguishable from the statistics' grey, and it is
    the one grey row, labelled on the axis (the caption names its colour).
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
    """Figure 3: one column per period, one row per quantity, Europe-wide. See europe_change_panel.

    ``value="twh"`` or ``"pct"``. In TWh the seasonal columns SHARE an x-limit (both are TWh per
    season); the annual column keeps its own. Limits cover every chain, padded off the frame.
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
