"""Plotting functions for TYNDP extremes analysis (EVT metrics, attribution, distributions).

Period specification
--------------------
All plotting functions accept a ``period_defs`` argument in one of two forms:

Fixed (same time window for every model):
    {"Reference": ("1991", "2020"), "Mid-Century": ("2036", "2065"), ...}

Per-model (different window per model, e.g. Global Warming Levels):
    {"1.5°C": {"model_A": ("2010", "2039"), "model_B": ("2018", "2047"), ...}, ...}

Use :func:`resolve_model_periods` to obtain the relevant slice for a given model
regardless of which form is supplied.  Plotting functions that work directly on
xarray data (e.g. :func:`plot_residual_load_kde`) call this internally so that
GWL-based analyses require no extra handling from the caller.

Functions that operate on a pre-aggregated DataFrame (spaghetti, violin, attribution)
are period-type-agnostic: as long as the DataFrame has a ``Period`` column whose
values match the keys of ``period_defs``, both fixed and per-model workflows produce
identical output.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from scipy.stats import gaussian_kde

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

FixedPeriodSpec = dict[str, tuple[str, str]]
PerModelPeriodSpec = dict[str, dict[str, tuple[str, str]]]
PeriodDefs = Union[FixedPeriodSpec, PerModelPeriodSpec]

# Default season configuration (DJF/MAM/JJA/SON — matches notebook convention)
_DEFAULT_SEASON_MAP: dict[int, str] = {
    12: "DJF", 1: "DJF", 2: "DJF",
    3: "MAM", 4: "MAM", 5: "MAM",
    6: "JJA", 7: "JJA", 8: "JJA",
    9: "SON", 10: "SON", 11: "SON",
}
_DEFAULT_SEASON_ORDER: list[str] = ["DJF", "MAM", "JJA", "SON"]

# Colours consistent with the notebook's ``period_colors`` dict
_DEFAULT_PERIOD_COLORS: dict[str, str] = {
    "Reference": "#4A90E2",
    "Mid-Century": "#F5A623",
    "End-Century": "#E24A4A",
}


# ---------------------------------------------------------------------------
# Period resolution utilities
# ---------------------------------------------------------------------------


def resolve_model_periods(
    period_defs: PeriodDefs,
    model: str,
) -> dict[str, tuple[str, str] | None]:
    """Resolve period definitions for a specific model.

    Parameters
    ----------
    period_defs :
        Fixed periods ``{name: (start, end)}`` or per-model periods
        ``{name: {model: (start, end)}}``.
    model :
        Model identifier string (e.g. ``"CNRM-ALADIN63_CNRM-CERFACS-CNRM-CM5_r1i1p1"``).

    Returns
    -------
    dict
        Mapping of period name → ``(start_year, end_year)`` string tuple, or
        ``None`` if the period is not available for that model (only possible for
        per-model specs where a model has not reached a given warming level).
    """
    resolved: dict[str, tuple[str, str] | None] = {}
    for name, spec in period_defs.items():
        if isinstance(spec, dict):
            resolved[name] = spec.get(model)
        else:
            resolved[name] = spec
    return resolved


def _is_per_model(period_defs: PeriodDefs) -> bool:
    """Return True when period_defs uses per-model (GWL-style) specifications."""
    first = next(iter(period_defs.values()), None)
    return isinstance(first, dict)


def _period_colors_with_defaults(
    period_names: list[str],
    period_colors: dict[str, str] | None,
) -> dict[str, str]:
    """Return period_colors, filling in defaults from colorblind palette for unknown names."""
    palette = sns.color_palette("colorblind", max(len(period_names), 1))
    defaults = dict(zip(period_names, palette))
    defaults.update(_DEFAULT_PERIOD_COLORS)
    if period_colors is not None:
        defaults.update(period_colors)
    return defaults


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _kde_envelope(
    data_2d,
    x_grid: np.ndarray,
    bw: float = 0.12,
    threshold: float | None = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[None, None, None]:
    """Compute KDE for each model row; return (median, q10, q90) across models.

    Parameters
    ----------
    data_2d :
        Shape ``(n_models, n_timesteps)`` ndarray **or** a list of 1-D arrays of
        potentially different lengths (required for per-model period windows).
    x_grid :
        Evaluation grid.
    bw :
        KDE bandwidth (scalar passed to ``scipy.stats.gaussian_kde``).
    threshold :
        Only values above this level are included in the KDE.
        ``None`` includes the full distribution (no lower bound beyond finiteness).

    Returns
    -------
    ``(median, q10, q90)`` each of shape ``(len(x_grid),)``, or
    ``(None, None, None)`` when fewer than 2 members have enough samples.
    """
    kdes = []
    for row in data_2d:
        mask = np.isfinite(row) if threshold is None else np.isfinite(row) & (row > threshold)
        pos = row[mask]
        if len(pos) < 30:
            continue
        try:
            kdes.append(gaussian_kde(pos, bw_method=bw)(x_grid))
        except Exception:
            pass
    if len(kdes) < 2:
        return None, None, None
    arr = np.stack(kdes)
    return (
        np.median(arr, axis=0),
        np.percentile(arr, 10, axis=0),
        np.percentile(arr, 90, axis=0),
    )


def _collect_data_for_kde(
    ds_zonal: xr.Dataset,
    zone: str,
    period_name: str,
    period_defs: PeriodDefs,
    models: list[str],
    season_map: dict[int, str],
    season: str,
    rl_var: str,
    per_model: bool,
    threshold: float | None = 0.0,
) -> tuple[list | np.ndarray, np.ndarray]:
    """Return (data_for_kde, flat_filtered_samples) for one zone/period/season.

    *flat_filtered_samples* contains only finite values above *threshold*
    (or all finite values when *threshold* is ``None``).
    """
    def _filter(arr: np.ndarray) -> np.ndarray:
        mask = np.isfinite(arr) if threshold is None else np.isfinite(arr) & (arr > threshold)
        return arr[mask]

    if per_model:
        series_list: list[np.ndarray] = []
        all_vals: list[float] = []
        for model in models:
            window = resolve_model_periods(period_defs, model).get(period_name)
            if window is None:
                continue
            t0, t1 = window
            da = ds_zonal.sel(model=model, zone=zone, time=slice(t0, t1))[rl_var]
            months = da.time.dt.month.values
            smask = np.array([season_map.get(m, "") for m in months]) == season
            arr = da.values[smask]
            series_list.append(arr)
            all_vals.extend(_filter(arr).tolist())
        return series_list, np.asarray(all_vals)
    else:
        t0, t1 = period_defs[period_name]  # type: ignore[misc]
        da = ds_zonal.sel(zone=zone, time=slice(t0, t1))[rl_var]
        months = da.time.dt.month.values
        smask = np.array([season_map.get(m, "") for m in months]) == season
        data_2d = da.values[:, smask]
        return data_2d, _filter(data_2d.ravel())


# ---------------------------------------------------------------------------
# Public plotting functions
# ---------------------------------------------------------------------------


def plot_residual_load_kde(
    ds_residual_load_zonal: xr.Dataset,
    zones: list[str],
    period_defs: PeriodDefs,
    period_colors: dict[str, str] | None = None,
    models: list[str] | None = None,
    season_map: dict[int, str] | None = None,
    season_order: list[str] | None = None,
    rl_var: str = "residual_load",
    bw: float = 0.12,
    figsize: tuple = (18, 3.5),
    path_savefig: Path | str | None = None,
    threshold: float | None = 0.0,
) -> None:
    """Plot ensemble KDE envelopes of residual load per zone and season.

    One figure per zone, one sub-panel per season. Each panel shows three
    overlaid KDE curves (one per period): the solid line is the ensemble median
    and the shaded band spans the 10th–90th model percentile.

    Supports both fixed time periods (same window for all models) and per-model
    periods such as Global Warming Levels; see module docstring.

    Parameters
    ----------
    ds_residual_load_zonal :
        Dataset with dims ``(model, zone, time)`` and variable *rl_var*.
    zones :
        Ordered list of zone names to plot (one figure each).
    period_defs :
        Fixed or per-model period specification.
    period_colors :
        ``{period_name: colour}``. Falls back to :data:`_DEFAULT_PERIOD_COLORS`
        and then the colorblind palette for names not in the default dict.
    models :
        Model names. Inferred from the dataset ``model`` coordinate if None.
    season_map :
        ``{month_int: season_label}`` mapping. Defaults to DJF/MAM/JJA/SON.
    season_order :
        Ordered season labels. Defaults to ``["DJF", "MAM", "JJA", "SON"]``.
    rl_var :
        Variable name in the dataset (default: ``"residual_load"``).
    bw :
        KDE bandwidth (default: 0.12, consistent with notebook).
    figsize :
        Figure size for each zone panel.
    path_savefig :
        If a directory path, saves ``{zone}_kde.png`` inside it. If a file path
        template, inserts the zone name before the suffix.
    threshold :
        Lower bound for values included in the KDE.
        ``0.0`` (default) reproduces the original behaviour of plotting only
        positive residual load.  ``None`` plots the full distribution.
        Any other value plots only values strictly above that level.
    """
    if models is None:
        models = list(ds_residual_load_zonal.model.values)
    if season_map is None:
        season_map = _DEFAULT_SEASON_MAP
    if season_order is None:
        season_order = _DEFAULT_SEASON_ORDER

    period_names = list(period_defs.keys())
    colors = _period_colors_with_defaults(period_names, period_colors)
    per_model = _is_per_model(period_defs)

    if threshold is None:
        _title_suffix = "full distribution"
        _xlabel = "Residual load (GWh)"
    elif threshold == 0.0:
        _title_suffix = "positive residual load"
        _xlabel = "Positive residual load (GWh)"
    else:
        _title_suffix = f"residual load > {threshold:.0f} GWh"
        _xlabel = f"Residual load above {threshold:.0f} GWh"

    for zone in zones:
        fig, axes = plt.subplots(1, len(season_order), figsize=figsize)
        fig.suptitle(
            f"{zone.replace('_', ' ')} — KDE of {_title_suffix}"
            f" (n={len(models)} models)",
            fontsize=12,
        )

        for ax, season in zip(axes, season_order):
            # Collect all filtered values across periods to build a shared x-axis.
            all_vals: list[float] = []
            for period_name in period_names:
                _, vals = _collect_data_for_kde(
                    ds_residual_load_zonal, zone, period_name, period_defs,
                    models, season_map, season, rl_var, per_model,
                    threshold=threshold,
                )
                all_vals.extend(vals.tolist())

            if len(all_vals) < 50:
                ax.set_title(season, fontsize=11)
                ax.set_visible(False)
                continue

            x_lo = np.percentile(all_vals, 0.5) if threshold is None else (threshold or 0.0)
            x_grid = np.linspace(x_lo, np.percentile(all_vals, 99.5), 300)

            for period_name in period_names:
                color = colors[period_name]
                data_2d, _ = _collect_data_for_kde(
                    ds_residual_load_zonal, zone, period_name, period_defs,
                    models, season_map, season, rl_var, per_model,
                    threshold=threshold,
                )
                med, q10, q90 = _kde_envelope(data_2d, x_grid, bw=bw, threshold=threshold)
                if med is None:
                    continue
                ax.fill_between(x_grid, q10, q90, alpha=0.18, color=color)
                ax.plot(x_grid, med, color=color, lw=2, label=period_name)

            ax.set_title(season, fontsize=11)
            ax.set_xlabel(_xlabel, fontsize=9)
            ax.set_ylabel("Density" if season == season_order[0] else "", fontsize=9)
            ax.tick_params(labelsize=8)

        axes[-1].legend(fontsize=8, title="Period", title_fontsize=8)
        sns.despine()
        plt.tight_layout()

        if path_savefig is not None:
            p = Path(path_savefig)
            if p.suffix:
                dest = p.parent / f"{p.stem}_{zone}{p.suffix}"
            else:
                p.mkdir(parents=True, exist_ok=True)
                dest = p / f"{zone}_kde.png"
            dest.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(dest, dpi=150, bbox_inches="tight")

        plt.show()


def plot_evt_threshold_heatmap(
    evt_thresholds: dict,
    zones: list[str],
    models: list[str],
    reference_period: list[str] | tuple[str, str],
    figsize: tuple | None = None,
    path_savefig: Path | str | None = None,
) -> None:
    """Plot EVT threshold heatmap: absolute values and % deviation from zone median.

    Two-panel figure: the top panel shows raw GWh/3h thresholds; the bottom panel
    shows each model's deviation from the zone median so inter-model spread is
    visible irrespective of the absolute zone level.

    Parameters
    ----------
    evt_thresholds :
        ``{model: xr.DataArray}`` with a ``zone`` coordinate.
    zones :
        Ordered list of zone names (columns of the heatmap).
    models :
        Ordered list of model names (rows of the heatmap).
    reference_period :
        Two-element sequence ``[start_year, end_year]`` for the title annotation.
    figsize :
        Figure size. Auto-sized when None.
    path_savefig :
        Path to save the figure.
    """
    thr_matrix = pd.DataFrame(
        {
            model: {
                str(z): float(evt_thresholds[model].sel(zone=z).values)
                for z in zones
            }
            for model in models
        }
    ).T
    thr_matrix.index.name = "Model"
    thr_matrix.columns.name = "Zone"

    zone_medians = thr_matrix.median(axis=0)
    thr_pct_dev = (thr_matrix - zone_medians) / zone_medians * 100

    if figsize is None:
        figsize = (
            max(8, len(zones) * 1.4),
            max(5, len(models) * 0.65) * 2 + 1.5,
        )

    fig, axes = plt.subplots(2, 1, figsize=figsize, gridspec_kw={"hspace": 0.45})

    sns.heatmap(
        thr_matrix,
        annot=True,
        fmt=".0f",
        cmap="YlOrRd",
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": "EVT threshold (GWh / 3h)"},
        ax=axes[0],
    )
    axes[0].set_title(
        f"EVT thresholds — absolute values (GWh / 3h)\n"
        f"reference period {reference_period[0]}–{reference_period[1]}",
        fontsize=11,
    )
    axes[0].set_xlabel("Balancing zone", fontsize=10)
    axes[0].set_ylabel("Climate model chain", fontsize=10)
    axes[0].tick_params(axis="x", rotation=20, labelsize=9)
    axes[0].tick_params(axis="y", rotation=0, labelsize=8)

    abs_limit = max(abs(thr_pct_dev.values.min()), abs(thr_pct_dev.values.max()), 5)
    annot_abs = thr_matrix.map(lambda v: f"{v:.0f}")

    sns.heatmap(
        thr_pct_dev,
        annot=annot_abs,
        fmt="",
        cmap="RdBu_r",
        center=0,
        vmin=-abs_limit,
        vmax=abs_limit,
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": "Deviation from zone median (%)"},
        ax=axes[1],
    )
    axes[1].set_title(
        "EVT thresholds — deviation from zone median (%)\n"
        "(colour = relative spread; annotation = absolute GWh value)",
        fontsize=11,
    )
    axes[1].set_xlabel("Balancing zone", fontsize=10)
    axes[1].set_ylabel("Climate model chain", fontsize=10)
    axes[1].tick_params(axis="x", rotation=20, labelsize=9)
    axes[1].tick_params(axis="y", rotation=0, labelsize=8)

    fig.suptitle("Inter-model EVT threshold spread", fontsize=13, fontweight="bold", y=1.01)

    if path_savefig is not None:
        p = Path(path_savefig)
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(p, dpi=300, bbox_inches="tight")

    plt.show()


def plot_spaghetti_iqr(
    df_ens: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    period_defs: PeriodDefs,
    zones: list[str],
    models: list[str],
    period_colors: dict[str, str] | None = None,
    figsize: tuple = (15, 8),
    sharey: bool = False,
    path_savefig: Path | str | None = None,
) -> None:
    """2×3 grid of spaghetti + IQR plots, one panel per zone.

    Individual model trajectories are drawn in grey (thin); the ensemble median
    is overlaid in steelblue (bold) with a 25–75th and 10–90th percentile band.
    A dashed reference line marks the median of the first (reference) period.

    Works with any period ordering and both fixed and per-model period specs —
    the latter must be pre-aggregated into *df_ens* (missing model–period
    combinations are handled as NaN and silently skipped in the spaghetti lines).

    Parameters
    ----------
    df_ens :
        DataFrame with at least columns ``["Zone", "Period", "model", metric]``.
    metric :
        Column name to plot.
    ylabel :
        Y-axis label.
    title :
        Figure suptitle.
    period_defs :
        Period specification, used only to determine period ordering and labels.
    zones :
        Ordered list of zone names (must match 2×3 = 6 zones).
    models :
        Ordered list of model names.
    period_colors :
        ``{period_name: colour}``. Not used in the current steelblue style but
        reserved for future coloured-median extensions.
    figsize :
        Figure size.
    sharey :
        Share the y-axis across all panels.
    path_savefig :
        Path to save the figure.
    """
    period_order = list(period_defs.keys())
    n_periods = len(period_order)
    x_pos = list(range(n_periods))
    tick_rotation = 15 if max((len(p) for p in period_order), default=0) > 8 else 0

    fig, axes = plt.subplots(2, 3, figsize=figsize, sharey=sharey, squeeze=False)
    axes_flat = axes.flatten()
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)

    for ax, zone in zip(axes_flat, zones):
        df_z = df_ens[df_ens["Zone"] == zone]

        for model in models:
            df_zm = df_z[df_z["model"] == model].set_index("Period")
            vals = [
                df_zm.loc[p, metric] if p in df_zm.index else np.nan
                for p in period_order
            ]
            ax.plot(x_pos, vals, color="0.65", alpha=0.5, lw=0.9, zorder=1)

        medians, q25s, q75s, q10s, q90s = [], [], [], [], []
        for p in period_order:
            vals_p = df_z[df_z["Period"] == p][metric].dropna().values
            if len(vals_p):
                medians.append(np.median(vals_p))
                q25s.append(np.percentile(vals_p, 25))
                q75s.append(np.percentile(vals_p, 75))
                q10s.append(np.percentile(vals_p, 10))
                q90s.append(np.percentile(vals_p, 90))
            else:
                medians.append(np.nan)
                q25s.append(np.nan)
                q75s.append(np.nan)
                q10s.append(np.nan)
                q90s.append(np.nan)

        x = np.arange(n_periods)
        ax.fill_between(x, q10s, q90s, alpha=0.12, color="steelblue", zorder=2)
        ax.fill_between(
            x, q25s, q75s, alpha=0.28, color="steelblue", zorder=2, label="IQR (Q25–Q75)"
        )
        ax.plot(
            x, medians, color="steelblue", lw=2.5, marker="o", ms=6, zorder=3,
            label="Ensemble median",
        )
        if not np.isnan(medians[0]):
            ax.axhline(medians[0], color="steelblue", lw=0.7, ls="--", alpha=0.4)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(
            period_order, fontsize=8, rotation=tick_rotation,
            ha="right" if tick_rotation else "center",
        )
        ax.set_title(zone.replace("_", " "), fontsize=10)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(labelsize=8)
        sns.despine(ax=ax)

    axes_flat[0].legend(fontsize=8)
    plt.tight_layout()

    if path_savefig is not None:
        p = Path(path_savefig)
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(p, dpi=300, bbox_inches="tight")

    plt.show()


def plot_violin_grid(
    df_ens: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    period_defs: PeriodDefs,
    zones: list[str],
    period_colors: dict[str, str] | None = None,
    figsize: tuple = (15, 8),
    path_savefig: Path | str | None = None,
) -> None:
    """2×3 grid of violin plots with individual model points, one panel per zone.

    Each violin is coloured by period; raw model values are overlaid as a strip
    plot so the actual sample size and spread are always visible.

    Parameters
    ----------
    df_ens :
        DataFrame with columns ``["Zone", "Period", "model", metric]``.
    metric :
        Column name to plot.
    ylabel :
        Y-axis label.
    title :
        Figure suptitle.
    period_defs :
        Period specification, used to determine ordering and colours.
    zones :
        Ordered list of zone names (must match 2×3 = 6 zones).
    period_colors :
        ``{period_name: colour}``. Falls back to colorblind palette if None.
    figsize :
        Figure size.
    path_savefig :
        Path to save the figure.
    """
    period_order = list(period_defs.keys())
    colors = _period_colors_with_defaults(period_order, period_colors)
    vpal = [colors[p] for p in period_order]

    fig, axes = plt.subplots(2, 3, figsize=figsize, sharey=False, squeeze=False)
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)

    for ax, zone in zip(axes.flatten(), zones):
        df_z = df_ens[df_ens["Zone"] == zone][["Period", metric]].copy()
        df_z["Period"] = pd.Categorical(df_z["Period"], categories=period_order, ordered=True)

        sns.violinplot(
            data=df_z,
            x="Period",
            y=metric,
            hue="Period",
            order=period_order,
            palette=vpal,
            ax=ax,
            inner="quartile",
            cut=0,
            linewidth=0.8,
            saturation=0.85,
            legend=False,
        )
        sns.stripplot(
            data=df_z,
            x="Period",
            y=metric,
            hue="Period",
            order=period_order,
            ax=ax,
            palette=["0.2"] * len(period_order),
            size=4,
            alpha=0.7,
            jitter=True,
            zorder=5,
            legend=False,
        )

        ax.set_title(zone.replace("_", " "), fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(axis="x", labelsize=8)
        ax.tick_params(axis="y", labelsize=8)
        sns.despine(ax=ax)

    plt.tight_layout()

    if path_savefig is not None:
        p = Path(path_savefig)
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(p, dpi=300, bbox_inches="tight")

    plt.show()


def plot_attribution_stacked_bars(
    df_ens_extreme: pd.DataFrame,
    zones: list[str],
    period_defs: PeriodDefs,
    driver_cols: dict[str, str],
    driver_palette: dict[str, str],
    supply_scenario: str | None = None,
    supply_year: int | None = None,
    figsize: tuple = (14, 5),
    path_savefig: Path | str | None = None,
) -> None:
    """Stacked bar charts of ensemble-median driver attribution, one figure per period.

    Each bar group shows the median percentage-point attribution across models for
    each driver, stacked to 100% for visual comparison across zones.

    Parameters
    ----------
    df_ens_extreme :
        DataFrame with columns ``["Zone", "Period", "model", *driver_cols.values()]``.
    zones :
        Ordered list of zone names.
    period_defs :
        Period specification, used for ordering and title labels.
    driver_cols :
        ``{driver_display_label: column_name_in_df}`` mapping.
    driver_palette :
        ``{driver_display_label: colour}``.
    supply_scenario, supply_year :
        Optional subtitle annotation (e.g. ``"HIGH"``, ``2050``).
    figsize :
        Figure size per period.
    path_savefig :
        Directory or path template for saving; one file per period.
    """
    period_order = list(period_defs.keys())
    x_base = np.arange(len(zones))
    subtitle = f" ({supply_scenario} {supply_year})" if supply_scenario else ""

    for period_name in period_order:
        df_p = df_ens_extreme[df_ens_extreme["Period"] == period_name]

        fig, ax = plt.subplots(figsize=figsize)
        bottoms = np.zeros(len(zones))

        for driver, col in driver_cols.items():
            medians = [df_p[df_p["Zone"] == z][col].median() for z in zones]
            ax.bar(
                x_base,
                medians,
                bottom=bottoms,
                color=driver_palette[driver],
                label=driver,
                width=0.55,
                alpha=0.85,
            )
            bottoms += np.array(medians)

        ax.set_xticks(x_base)
        ax.set_xticklabels([z.replace("_", "\n") for z in zones], fontsize=9)
        ax.set_ylabel("Driver attribution share (%)", fontsize=10)
        ax.set_ylim(0, 110)
        ax.set_title(
            f"EVT extreme event driver attribution — {period_name}\n"
            f"ensemble median{subtitle}",
            fontsize=11,
        )
        ax.legend(loc="upper right", fontsize=9, ncol=3, title="Driver")
        sns.despine()
        plt.tight_layout()

        if path_savefig is not None:
            p = Path(path_savefig)
            safe = period_name.replace(" ", "_").replace("°", "deg").replace(".", "p")
            if p.suffix:
                dest = p.parent / f"{p.stem}_{safe}{p.suffix}"
            else:
                p.mkdir(parents=True, exist_ok=True)
                dest = p / f"attribution_{safe}.png"
            dest.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(dest, dpi=150, bbox_inches="tight")

        plt.show()


def plot_attribution_change_bars(
    df_ens_extreme: pd.DataFrame,
    zones: list[str],
    period_defs: PeriodDefs,
    driver_cols: dict[str, str],
    reference_period_name: str | None = None,
    figsize: tuple = (16, 6),
    path_savefig: Path | str | None = None,
) -> None:
    """Horizontal bar charts: change in driver attribution share vs a reference period.

    One row per non-reference period, one column per driver. Each bar shows the
    ensemble-median change in percentage points for each zone. Blue = decrease,
    red = increase.

    Parameters
    ----------
    df_ens_extreme :
        DataFrame with columns ``["Zone", "Period", "model", *driver_cols.values()]``.
    zones :
        Ordered list of zone names.
    period_defs :
        Period specification. All periods except *reference_period_name* become rows.
    driver_cols :
        ``{driver_display_label: column_name_in_df}`` mapping.
    reference_period_name :
        Period name to use as the baseline for change computation.  Defaults to
        the first key in *period_defs*, so it works for both fixed calendar periods
        (``"Reference"``) and Global Warming Level windows (``"+1.5°C"``).
    figsize :
        Figure size.
    path_savefig :
        Path to save the figure.
    """
    period_order = list(period_defs.keys())
    if reference_period_name is None:
        reference_period_name = period_order[0]
    compare_periods = [p for p in period_order if p != reference_period_name]
    n_compare = len(compare_periods)
    n_drivers = len(driver_cols)

    if n_compare == 0:
        raise ValueError(
            f"No comparison periods found after removing '{reference_period_name}'."
        )

    fig, axes = plt.subplots(
        n_compare, n_drivers,
        figsize=figsize,
        sharey=True,
        sharex=True,
        squeeze=False,
    )

    ref_by_zone = (
        df_ens_extreme[df_ens_extreme["Period"] == reference_period_name]
        .groupby("Zone")
    )

    for row_idx, period_name in enumerate(compare_periods):
        tgt_by_zone = (
            df_ens_extreme[df_ens_extreme["Period"] == period_name]
            .groupby("Zone")
        )
        for col_idx, (driver, col) in enumerate(driver_cols.items()):
            ax = axes[row_idx, col_idx]
            ref_vals = ref_by_zone[col].median().reindex(zones)
            tgt_vals = tgt_by_zone[col].median().reindex(zones)
            delta = tgt_vals - ref_vals

            bar_colors = ["#2166AC" if v <= 0 else "#D6604D" for v in delta.values]
            ax.barh(range(len(zones)), delta.values, color=bar_colors, alpha=0.85)
            ax.axvline(0, color="black", lw=0.7)

            if row_idx == 0:
                ax.set_title(driver, fontsize=9)

            ax.set_xlabel("Δ share (pp)", fontsize=8)
            ax.tick_params(labelsize=8)
            sns.despine(ax=ax)

            if col_idx == 0:
                ax.set_yticks(range(len(zones)))
                ax.set_yticklabels([z.replace("_", "\n") for z in zones], fontsize=7)
                ax.set_ylabel(
                    f"{period_name} − {reference_period_name}", fontsize=8
                )

    fig.suptitle(
        f"Change in driver attribution share vs {reference_period_name}"
        f"\n(ensemble median percentage points)",
        fontsize=11,
    )
    plt.tight_layout()

    if path_savefig is not None:
        p = Path(path_savefig)
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(p, dpi=300, bbox_inches="tight")

    plt.show()


# ---------------------------------------------------------------------------
# Seasonal exceedance distribution
# ---------------------------------------------------------------------------


def plot_seasonal_exceedance_distribution(
    df_seasonal,
    zones,
    season_order: list,
    period_defs,
    period_colors: dict | None = None,
    figsize: tuple = (15, 8),
    supply_scenario: str = "",
    supply_year="",
) -> None:
    """Bar chart of seasonal EVT exceedance fractions — ensemble median ± IQR.

    Parameters
    ----------
    df_seasonal :
        Output of ``compute_seasonal_exceedance_fractions``.
    zones :
        Ordered zone values (6 zones → 2×3 subplot grid).
    season_order :
        Season names in display order.
    period_defs :
        Period definitions (used to extract ordered period names).
    period_colors :
        ``{period_name: hex_color}``. Defaults to the library colour cycle.
    figsize :
        Figure size in inches.
    supply_scenario, supply_year :
        Injected into the figure title for provenance.
    """
    period_names = list(period_defs.keys())
    n_periods = len(period_names)
    period_colors = _period_colors_with_defaults(period_names, period_colors)

    x = np.arange(len(season_order))
    bar_width = 0.8 / n_periods

    fig, axes = plt.subplots(2, 3, figsize=figsize, sharey=True)
    axes_flat = list(axes.flat)

    for ax, zone in zip(axes_flat, zones):
        for pi, period_name in enumerate(period_names):
            sub = df_seasonal[
                (df_seasonal["zone"] == str(zone)) & (df_seasonal["period"] == period_name)
            ]
            medians = np.array([sub[s].median() for s in season_order])
            lo = np.array([sub[s].quantile(0.25) for s in season_order])
            hi = np.array([sub[s].quantile(0.75) for s in season_order])
            offsets = x + (pi - n_periods / 2 + 0.5) * bar_width
            ax.bar(offsets, medians, width=bar_width * 0.9,
                   color=period_colors[period_name], alpha=0.85, label=period_name)
            ax.errorbar(offsets, medians, yerr=[medians - lo, hi - medians],
                        fmt="none", color="k", capsize=2.5, linewidth=0.8)

        ax.set_title(str(zone).replace("_", " "), fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(season_order)
        ax.set_ylabel("% of exceedances")
        ax.set_ylim(0, 85)
        ax.axhline(25, color="grey", linestyle="--", linewidth=0.6, alpha=0.6)
        ax.grid(axis="y", linewidth=0.4)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=n_periods, bbox_to_anchor=(0.5, -0.03))
    suffix = f"{supply_scenario} {supply_year}".strip()
    title = "Seasonal distribution of EVT exceedances — ensemble median ± IQR"
    title += f"\n(model-calibrated thresholds{' · ' + suffix if suffix else ''} · dashed line = uniform 25%)"
    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Seasonal-conditional EVT metric bars
# ---------------------------------------------------------------------------


def plot_seasonal_conditional_metrics(
    df_seasonal_evt,
    metrics: list,
    zones,
    period_defs,
    season_order: list,
    period_colors: dict | None = None,
    figsize: tuple = (15, 8),
    supply_scenario: str = "",
    supply_year="",
    sharey: bool = False
) -> None:
    """Grouped boxplots of seasonal-conditional EVT metrics — one box per (season, period).

    One figure is produced per metric. Each box shows the inter-model spread
    (median line, IQR box, whiskers at 5th/95th percentile).

    Parameters
    ----------
    df_seasonal_evt :
        Output of ``compute_seasonal_evt_metrics`` collected into a DataFrame.
    metrics :
        List of ``(column_name, y_label, title_suffix)`` tuples.
    zones :
        Ordered zone values (6 zones → 2×3 grid per figure).
    period_defs :
        Period definitions (used to extract ordered period names).
    season_order :
        Season names in display order.
    period_colors :
        ``{period_name: hex_color}``.
    figsize :
        Figure size for each metric figure.
    supply_scenario, supply_year :
        Injected into figure titles.
    """
    period_names = list(period_defs.keys())
    n_periods = len(period_names)
    period_colors = _period_colors_with_defaults(period_names, period_colors)

    n_seasons = len(season_order)
    box_w = 0.7 / n_periods
    # x position for each (season, period) box
    season_x = np.arange(n_seasons)

    for metric, ylabel, title_suffix in metrics:
        fig, axes = plt.subplots(2, 3, figsize=figsize, sharey=sharey)
        axes_flat = list(axes.flat)

        for ax, zone in zip(axes_flat, zones):
            sub_zone = df_seasonal_evt[df_seasonal_evt["zone"] == str(zone)]

            for pi, period_name in enumerate(period_names):
                sub_period = sub_zone[sub_zone["Period"] == period_name]
                color = period_colors[period_name]
                offsets = season_x + (pi - n_periods / 2 + 0.5) * box_w

                data_per_season = [
                    sub_period[sub_period["season"] == s][metric].dropna().values
                    for s in season_order
                ]

                ax.boxplot(
                    data_per_season,
                    positions=offsets,
                    widths=box_w * 0.85,
                    patch_artist=True,
                    whis=[5, 95],
                    showfliers=False,
                    medianprops=dict(color="white", linewidth=1.8),
                    boxprops=dict(facecolor=color, alpha=0.85, linewidth=0.6),
                    whiskerprops=dict(color=color, linewidth=0.9, linestyle="--"),
                    capprops=dict(color=color, linewidth=0.9),
                )
                # invisible proxy for legend
                ax.plot([], [], color=color, linewidth=5, alpha=0.85, label=period_name)

            ax.set_title(str(zone).replace("_", " "), fontsize=11)
            ax.set_xticks(season_x)
            ax.set_xticklabels(season_order)
            ax.set_ylabel(ylabel)
            ax.grid(axis="y", linewidth=0.4)

        handles, labels = axes_flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=n_periods,
                   bbox_to_anchor=(0.5, -0.03))
        suptitle = f"Seasonal-conditional EVT — {title_suffix} — inter-model spread"
        suffix = f"{supply_scenario} {supply_year}".strip()
        if suffix:
            suptitle += f"\n(season-specific thresholds · {suffix} · whiskers = 5th/95th pct)"
        else:
            suptitle += "\n(season-specific thresholds · whiskers = 5th/95th pct)"
        fig.suptitle(suptitle, fontsize=13)
        plt.tight_layout()
        plt.show()



# ---------------------------------------------------------------------------
# Seasonal attribution stacked bars
# ---------------------------------------------------------------------------


def plot_seasonal_attribution(
    df_seasonal_evt,
    zones,
    season_order: list,
    period_defs,
    driver_cols: dict,
    driver_palette: dict,
    figsize: tuple = (15, 8),
    supply_scenario: str = "",
    supply_year="",
) -> None:
    """Stacked-bar climate driver attribution by season — one figure per season.

    Parameters
    ----------
    df_seasonal_evt :
        Output of ``compute_seasonal_evt_metrics`` collected into a DataFrame.
    zones :
        Ordered zone values (6 zones → 2×3 grid per figure).
    season_order :
        Season names (one figure per season).
    period_defs :
        Period definitions.
    driver_cols :
        ``{driver_label: column_name}`` mapping into ``df_seasonal_evt``.
    driver_palette :
        ``{driver_label: hex_color}``.
    figsize :
        Figure size for each season figure.
    supply_scenario, supply_year :
        Injected into figure titles.
    """
    import matplotlib.patches as mpatches

    period_names = list(period_defs.keys())
    n_periods = len(period_names)
    drivers = list(driver_cols.keys())

    for season in season_order:
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes_flat = list(axes.flat)

        for ax, zone in zip(axes_flat, zones):
            sub_zone = df_seasonal_evt[
                (df_seasonal_evt["zone"] == str(zone)) & (df_seasonal_evt["season"] == season)
            ]
            has_events = False
            for pi, period_name in enumerate(period_names):
                sub = sub_zone[sub_zone["Period"] == period_name]
                med_freq = sub["Frequency"].median()
                if pd.isna(med_freq) or med_freq == 0:
                    continue
                has_events = True
                bottom = 0.0
                for driver in drivers:
                    col = driver_cols[driver]
                    val = sub[col].median(skipna=True)
                    if pd.isna(val) or val <= 0:
                        continue
                    ax.bar(pi, val, bottom=bottom, width=0.65,
                           color=driver_palette[driver], alpha=0.9)
                    bottom += val

            if not has_events:
                ax.text(0.5, 0.5, "No events\nin this season",
                        ha="center", va="center", transform=ax.transAxes,
                        color="grey", fontsize=10, style="italic")

            ax.set_title(str(zone).replace("_", " "), fontsize=10)
            ax.set_xticks(range(n_periods))
            ax.set_xticklabels(period_names, fontsize=9, rotation=20, ha="right")
            ax.set_ylabel("Driver share (%)")
            ax.set_ylim(0, 108)
            ax.axhline(100, color="k", linewidth=0.5, linestyle="--", alpha=0.5)
            ax.grid(axis="y", linewidth=0.3)

        legend_handles = [mpatches.Patch(facecolor=driver_palette[d], label=d) for d in drivers]
        fig.legend(handles=legend_handles, loc="lower center", ncol=len(drivers),
                   bbox_to_anchor=(0.5, -0.04), fontsize=9)
        suptitle = f"Seasonal-conditional attribution — {season}"
        suffix = f"{supply_scenario} {supply_year}".strip()
        if suffix:
            suptitle += f"\n(season-specific thresholds · {suffix})"
        fig.suptitle(suptitle, fontsize=12)
        plt.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------
# Zone co-occurrence heatmaps
# ---------------------------------------------------------------------------

_ZONE_SHORT: dict = {
    "British_Isles": "BI", "Central_Eastern": "CE",
    "Central_Western": "CW", "Iberian_Peninsula": "IP",
    "Nordic": "NO", "South_East": "SE",
}


def _render_cooccurrence_heatmaps(
    df_coo, zones_str, labels, period_names, baseline_period,
    cmap_abs, cmap_diff, figsize_per_panel, suffix, season_tag,
):
    """Core heatmap renderer for a single season slice (or full period)."""
    median_mats: dict = {}
    for period_name in period_names:
        sub = df_coo[df_coo["period"] == period_name]
        median_mats[period_name] = (
            sub.groupby(["zone_a", "zone_b"])["co_occurrence_pct"]
            .median()
            .unstack()
            .reindex(index=zones_str, columns=zones_str)
        )

    w, h = figsize_per_panel
    n_periods = len(period_names)
    fig, axes = plt.subplots(1, n_periods, figsize=(w * n_periods, h), sharey=True)
    if n_periods == 1:
        axes = [axes]
    for ax, period_name in zip(axes, period_names):
        sns.heatmap(
            median_mats[period_name], ax=ax, vmin=0, vmax=100, cmap=cmap_abs,
            annot=True, fmt=".0f", linewidths=0.5,
            xticklabels=labels, yticklabels=labels,
            cbar=(ax == axes[-1]),
        )
        ax.set_title(period_name, fontsize=12, pad=8)
        ax.set_xlabel("Zone B (has event within ±72 h)", fontsize=9)
        ax.set_ylabel("Zone A (event origin)" if ax == axes[0] else "", fontsize=9)

    title1 = f"Compound event co-occurrence (%) — ensemble median{season_tag}"
    title1 += "\nFraction of Zone A exceedances co-occurring with Zone B exceedance within ±72 h"
    if suffix:
        title1 += f"\n({suffix})"
    fig.suptitle(title1, fontsize=12, y=1.04)
    plt.tight_layout()
    plt.show()

    diff_periods = [p for p in period_names if p != baseline_period]
    n_diff = len(diff_periods)
    if n_diff == 0:
        return

    abs_max = max(
        (median_mats[p] - median_mats[baseline_period]).abs().max().max()
        for p in diff_periods
    )
    abs_max = np.ceil(abs_max / 5) * 5

    fig, axes = plt.subplots(1, n_diff, figsize=(w * n_diff, h), sharey=True)
    if n_diff == 1:
        axes = [axes]
    for ax, period_name in zip(axes, diff_periods):
        diff_mat = median_mats[period_name] - median_mats[baseline_period]
        sns.heatmap(
            diff_mat, ax=ax, vmin=-abs_max, vmax=abs_max, cmap=cmap_diff, center=0,
            annot=True, fmt=".0f", linewidths=0.5,
            xticklabels=labels, yticklabels=labels,
            cbar=(ax == axes[-1]),
        )
        ax.set_title(f"{period_name} − {baseline_period}", fontsize=12, pad=8)
        ax.set_xlabel("Zone B", fontsize=9)
        ax.set_ylabel("Zone A (event origin)" if ax == axes[0] else "", fontsize=9)

    title2 = (
        f"Change in compound co-occurrence (pp) vs {baseline_period} — ensemble median{season_tag}"
        "\nPositive = more frequent co-occurrence at higher warming"
    )
    if suffix:
        title2 += f" · ({suffix})"
    fig.suptitle(title2, fontsize=12, y=1.04)
    plt.tight_layout()
    plt.show()


def plot_zone_cooccurrence_heatmaps(
    df_coo,
    zones,
    period_defs,
    baseline_period: str | None = None,
    cmap_abs: str = "YlOrRd",
    cmap_diff: str = "RdBu_r",
    figsize_per_panel: tuple = (5.5, 5.5),
    zone_short: dict | None = None,
    season_order: list | None = None,
    supply_scenario: str = "",
    supply_year="",
) -> None:
    """Heatmap figures for ensemble-median co-occurrence and change vs baseline.

    When ``df_coo`` contains a ``season`` column (i.e. it was produced with
    ``compute_zone_pair_cooccurrence(..., season_map=...)``), one pair of figures
    is rendered per season in ``season_order`` order.  Otherwise a single pair
    of figures is produced for the full period.

    Parameters
    ----------
    df_coo :
        Output of ``compute_zone_pair_cooccurrence``.
    zones :
        Ordered zone values.
    period_defs :
        Period definitions (used to extract ordered period names).
    baseline_period :
        Period name used as the baseline for the difference figure.
        Defaults to the first period in ``period_defs``.
    cmap_abs, cmap_diff :
        Colormaps for the absolute and difference figures.
    figsize_per_panel :
        Width × height in inches for a single heatmap panel.
    zone_short :
        Custom ``{zone_name: short_label}`` mapping.
    season_order :
        Ordered list of season names used when ``df_coo`` is seasonal.
        If ``None``, seasons are taken in the order they appear in the data.
    supply_scenario, supply_year :
        Injected into figure titles.
    """
    period_names = list(period_defs.keys())
    if baseline_period is None:
        baseline_period = period_names[0]

    zs = zone_short or _ZONE_SHORT
    labels = [zs.get(str(z), str(z)) for z in zones]
    zones_str = [str(z) for z in zones]
    suffix = f"{supply_scenario} {supply_year}".strip()

    if "season" in df_coo.columns:
        seasons = season_order or list(df_coo["season"].unique())
        for season in seasons:
            _render_cooccurrence_heatmaps(
                df_coo[df_coo["season"] == season],
                zones_str, labels, period_names, baseline_period,
                cmap_abs, cmap_diff, figsize_per_panel, suffix,
                season_tag=f" — {season}",
            )
    else:
        _render_cooccurrence_heatmaps(
            df_coo, zones_str, labels, period_names, baseline_period,
            cmap_abs, cmap_diff, figsize_per_panel, suffix, season_tag="",
        )


def _render_cooccurrence_boxplots(
    df_coo, zones_str, labels, period_names, colors, figsize, zs, suffix, season_tag,
):
    """Core boxplot renderer for a single season slice (or full period)."""
    n_periods = len(period_names)
    n_zones = len(zones_str)

    fig, axes = plt.subplots(1, n_zones, figsize=figsize, sharey=True, squeeze=False)
    axes = axes[0]

    group_width = 0.8
    box_width = group_width / n_periods

    for col_idx, zone_a in enumerate(zones_str):
        ax = axes[col_idx]
        df_a = df_coo[df_coo["zone_a"] == zone_a]
        dest_zones = [z for z in zones_str if z != zone_a]
        n_dest = len(dest_zones)

        for p_idx, period_name in enumerate(period_names):
            color = colors.get(period_name, f"C{p_idx}")
            df_p = df_a[df_a["period"] == period_name]
            positions, data = [], []
            for d_idx, zone_b in enumerate(dest_zones):
                vals = df_p[df_p["zone_b"] == zone_b]["co_occurrence_pct"].dropna().values
                if len(vals) == 0:
                    continue
                offset = (p_idx - (n_periods - 1) / 2) * box_width
                positions.append(d_idx + offset)
                data.append(vals)
            if not data:
                continue
            ax.boxplot(
                data, positions=positions, widths=box_width * 0.85,
                patch_artist=True, whis=[5, 95], showfliers=False,
                boxprops=dict(facecolor=color, alpha=0.85, linewidth=0.6),
                medianprops=dict(color="white", linewidth=1.8),
                whiskerprops=dict(linewidth=0.8),
                capprops=dict(linewidth=0.8),
            )

        ax.set_xticks(range(n_dest))
        ax.set_xticklabels(
            [zs.get(z, z) for z in dest_zones], fontsize=8, rotation=30, ha="right"
        )
        ax.set_title(f"Origin: {labels[col_idx]}", fontsize=9, pad=6)
        ax.set_ylim(0, 100)
        ax.set_xlabel("Zone B", fontsize=8)
        if col_idx == 0:
            ax.set_ylabel("Co-occurrence (%)", fontsize=9)
        ax.axhline(0, color="black", lw=0.5)
        sns.despine(ax=ax)

    proxy = [
        plt.Rectangle((0, 0), 1, 1, fc=colors.get(p, f"C{i}"), alpha=0.85)
        for i, p in enumerate(period_names)
    ]
    axes[-1].legend(proxy, period_names, fontsize=8, title="Period",
                    title_fontsize=8, loc="upper right")

    title = (
        f"Compound event co-occurrence (%) — inter-model spread{season_tag}"
        "\nbox = Q25–Q75 · whiskers = 5th/95th pct · one value per model"
    )
    if suffix:
        title += f" · ({suffix})"
    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.show()


def plot_zone_cooccurrence_boxplots(
    df_coo,
    zones,
    period_defs,
    period_colors: dict | None = None,
    zone_short: dict | None = None,
    season_order: list | None = None,
    figsize: tuple = (16, 10),
    supply_scenario: str = "",
    supply_year="",
) -> None:
    """Inter-model spread of zone-pair co-occurrence as boxplots.

    One subplot per origin zone (Zone A). Within each subplot the x-axis is
    the destination zone (Zone B) and each group of boxes represents a GWL
    period.  The full model distribution is shown (box = Q25–Q75, whiskers
    5th–95th pct).

    When ``df_coo`` contains a ``season`` column, one figure is rendered per
    season in ``season_order`` order.

    Parameters
    ----------
    df_coo :
        Output of ``compute_zone_pair_cooccurrence``.
    zones :
        Ordered zone values.
    period_defs :
        Period definitions.
    period_colors :
        ``{period_name: colour}`` mapping.  Falls back to the tab10 palette.
    zone_short :
        Custom ``{zone_name: short_label}`` mapping.
    season_order :
        Ordered season names used when ``df_coo`` is seasonal.
    figsize :
        Overall figure size.
    supply_scenario, supply_year :
        Injected into figure title.
    """
    period_names = list(period_defs.keys())
    n_periods = len(period_names)
    zones_str = [str(z) for z in zones]

    zs = zone_short or _ZONE_SHORT
    labels = [zs.get(z, z) for z in zones_str]

    colors = period_colors or {
        p: plt.cm.tab10(i / max(n_periods - 1, 1)) for i, p in enumerate(period_names)
    }
    suffix = f"{supply_scenario} {supply_year}".strip()

    if "season" in df_coo.columns:
        seasons = season_order or list(df_coo["season"].unique())
        for season in seasons:
            _render_cooccurrence_boxplots(
                df_coo[df_coo["season"] == season],
                zones_str, labels, period_names, colors, figsize, zs, suffix,
                season_tag=f" — {season}",
            )
    else:
        _render_cooccurrence_boxplots(
            df_coo, zones_str, labels, period_names, colors, figsize, zs, suffix,
            season_tag="",
        )


# ---------------------------------------------------------------------------
# Duration-severity power-law scaling
# ---------------------------------------------------------------------------


def plot_duration_severity_scaling(
    df_ds,
    zones,
    period_defs,
    period_colors: dict | None = None,
    figsize: tuple = (16, 10),
    min_events: int = 10,
    supply_scenario: str = "",
    supply_year="",
) -> None:
    """Log-log scatter + power-law fits of individual EVT event duration vs severity.

    Parameters
    ----------
    df_ds :
        Output of ``extract_evt_duration_severity_records``.
    zones :
        Ordered zone values (6 zones → 2×3 subplot grid).
    period_defs :
        Period definitions (used to extract ordered period names).
    period_colors :
        ``{period_name: hex_color}``.
    figsize :
        Figure size in inches.
    min_events :
        Minimum number of valid events required to fit a power law.
    supply_scenario, supply_year :
        Injected into the figure title.
    """
    from scipy.stats import linregress

    period_names = list(period_defs.keys())
    period_colors = _period_colors_with_defaults(period_names, period_colors)

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes_flat = list(axes.flat)
    fit_summary: list = []

    for ax, zone in zip(axes_flat, zones):
        for period_name in period_names:
            color = period_colors[period_name]
            sub = df_ds[(df_ds["zone"] == str(zone)) & (df_ds["period"] == period_name)]
            d = sub["duration_h"].values.astype(float)
            s = sub["severity_gwh"].values.astype(float)
            valid = (d > 0) & (s > 0)
            if valid.sum() < min_events:
                continue
            ax.scatter(d[valid], s[valid], color=color, alpha=0.12, s=5, rasterized=True)
            slope, intercept, r, *_ = linregress(np.log(d[valid]), np.log(s[valid]))
            d_fit = np.linspace(d[valid].min(), d[valid].max(), 200)
            ax.plot(d_fit, np.exp(intercept) * d_fit ** slope, color=color, linewidth=2.2,
                    label=f"{period_name}  b={slope:.2f} R²={r**2:.2f}")
            fit_summary.append({
                "zone": str(zone), "period": period_name,
                "b_exponent": slope, "R2": r ** 2, "n_events": int(valid.sum()),
            })

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(str(zone).replace("_", " "), fontsize=11)
        ax.set_xlabel("Duration (h)")
        ax.set_ylabel("Severity above threshold (GWh)")
        ax.legend(fontsize=7.5, framealpha=0.8)
        ax.grid(True, which="both", linewidth=0.35, alpha=0.6)

    suffix = f"{supply_scenario} {supply_year}".strip()
    suptitle = "Duration–severity scaling of EVT events — all models pooled  (S = a · D^b)"
    suptitle += f"\n(log-log scale · model-calibrated thresholds{' · ' + suffix if suffix else ''})"
    fig.suptitle(suptitle, fontsize=13)
    plt.tight_layout()
    plt.show()

    if fit_summary:
        import pandas as _pd
        print("\nPower-law fit summary (pooled across all models):")
        df_fits = _pd.DataFrame(fit_summary).set_index(["zone", "period"])
        print(df_fits.round(3).to_string())


# ---------------------------------------------------------------------------
# Seasonal attribution change bars
# ---------------------------------------------------------------------------


def plot_seasonal_attribution_change(
    df_seasonal_evt,
    zones,
    season_order: list,
    period_defs,
    driver_cols: dict,
    reference_period_name: str | None = None,
    share_x_across_seasons: bool = True,
    figsize: tuple = (16, 6),
    supply_scenario: str = "",
    supply_year="",
) -> None:
    """Change in seasonal driver attribution share vs a reference period.

    Produces one figure per season. Layout mirrors ``plot_attribution_change_bars``:
    rows = comparison periods, columns = drivers, horizontal bars = zones.
    Blue = driver share decreased, red = increased (ensemble-median Δ pp).

    Parameters
    ----------
    df_seasonal_evt :
        Output of ``compute_seasonal_evt_metrics`` collected into a DataFrame.
        Must contain columns ``zone``, ``season``, ``Period``, and the driver
        share columns listed in *driver_cols*.
    zones :
        Ordered zone values.
    season_order :
        Season names (one figure per season).
    period_defs :
        Period definitions.
    driver_cols :
        ``{driver_display_label: column_name_in_df}`` mapping.
    reference_period_name :
        Baseline period for computing deltas.  Defaults to the first key in
        *period_defs* (e.g. ``"+1.5°C"`` in GWL mode, ``"Reference"`` otherwise).
    share_x_across_seasons :
        If True (default), all season figures use the same symmetric x-axis
        limits, computed from the global maximum |Δ| across all seasons.
    figsize :
        Figure size for each season figure.
    supply_scenario, supply_year :
        Injected into figure titles.
    """
    period_order = list(period_defs.keys())
    if reference_period_name is None:
        reference_period_name = period_order[0]
    compare_periods = [p for p in period_order if p != reference_period_name]
    n_compare = len(compare_periods)
    n_drivers = len(driver_cols)
    zones_str = [str(z) for z in zones]

    if n_compare == 0:
        raise ValueError(
            f"No comparison periods found after removing '{reference_period_name}'."
        )

    suffix = f"{supply_scenario} {supply_year}".strip()

    # Pre-compute global x-limit if shared across seasons
    if share_x_across_seasons:
        medians_all = (
            df_seasonal_evt
            .groupby(["season", "Period", "zone"])[[*driver_cols.values()]]
            .median()
        )
        global_deltas = []
        for season in season_order:
            for period_name in compare_periods:
                for col in driver_cols.values():
                    try:
                        ref = medians_all.loc[(season, reference_period_name)][col].reindex(zones_str)
                        tgt = medians_all.loc[(season, period_name)][col].reindex(zones_str)
                        global_deltas.extend((tgt - ref).dropna().tolist())
                    except KeyError:
                        pass
        global_x_bound = max((abs(v) for v in global_deltas), default=1.0) * 1.1

    for season in season_order:
        sub_season = df_seasonal_evt[df_seasonal_evt["season"] == season]

        # Ensemble-median attribution per (period, zone) for this season
        medians = (
            sub_season
            .groupby(["Period", "zone"])[[*driver_cols.values()]]
            .median()
        )

        # Compute per-season symmetric x-limit (used when share_x_across_seasons=False)
        all_deltas = []
        for period_name in compare_periods:
            for col in driver_cols.values():
                try:
                    ref_vals = medians.loc[reference_period_name][col].reindex(zones_str)
                    tgt_vals = medians.loc[period_name][col].reindex(zones_str)
                    all_deltas.extend((tgt_vals - ref_vals).dropna().tolist())
                except KeyError:
                    pass
        x_bound = max((abs(v) for v in all_deltas), default=1.0) * 1.1

        fig, axes = plt.subplots(
            n_compare, n_drivers,
            figsize=figsize,
            sharey=True,
            sharex=True,
            squeeze=False,
        )

        for row_idx, period_name in enumerate(compare_periods):
            for col_idx, (driver, col) in enumerate(driver_cols.items()):
                ax = axes[row_idx, col_idx]

                try:
                    ref_vals = medians.loc[reference_period_name][col].reindex(zones_str)
                    tgt_vals = medians.loc[period_name][col].reindex(zones_str)
                except KeyError:
                    ax.text(0.5, 0.5, "no data", ha="center", va="center",
                            transform=ax.transAxes, color="grey", fontsize=8)
                    continue

                delta = tgt_vals - ref_vals
                bar_colors = ["#2166AC" if v <= 0 else "#D6604D" for v in delta.values]
                ax.barh(range(len(zones_str)), delta.values, color=bar_colors, alpha=0.85)
                ax.axvline(0, color="black", lw=0.7)

                if row_idx == 0:
                    ax.set_title(driver, fontsize=9)
                ax.set_xlabel("Δ share (pp)", fontsize=8)
                ax.tick_params(labelsize=8)
                sns.despine(ax=ax)

                ax.set_yticks(range(len(zones_str)))
                ax.set_yticklabels(
                    [z.replace("_", "\n") for z in zones_str], fontsize=7
                )
                if col_idx == 0:
                    ax.set_ylabel(f"{period_name} − {reference_period_name}", fontsize=8)

        bound = global_x_bound if share_x_across_seasons else x_bound
        axes[0, 0].set_xlim(-bound, bound)

        suptitle = (
            f"Change in seasonal driver attribution — {season} — vs {reference_period_name}"
            f"\n(ensemble-median Δ percentage points"
        )
        if suffix:
            suptitle += f" · season-specific thresholds · {suffix})"
        else:
            suptitle += " · season-specific thresholds)"
        fig.suptitle(suptitle, fontsize=11)
        plt.tight_layout()
        plt.show()


def plot_component_anomaly_distributions(
    df_ens_extreme: pd.DataFrame,
    zones: list[str],
    period_defs: PeriodDefs,
    component_cols: dict[str, str],
    component_palette: dict[str, str],
    supply_scenario: str | None = None,
    supply_year: int | None = None,
    period_palette: "dict[str, str] | None" = None,
    figsize: tuple = (16, 9),
    path_savefig: "Path | str | None" = None,
) -> None:
    """Box + strip plots of mean signed driver anomaly per event.

    Produces two sets of figures:

    1. **Per-period figures** (one per warming level / period): 2×3 zone grid,
       all components shown for that period, with individual model dots.
    2. **Period-comparison figure** (one figure): 2×3 zone grid, boxplots grouped
       by period so that shifts across warming levels are directly visible per zone
       and component.

    Demand anomaly is positive (excess demand above baseline); generation and
    flexibility anomalies are negative (shortfall / under-dispatch below baseline).

    Parameters
    ----------
    df_ens_extreme :
        DataFrame with columns ``["Zone", "Period", "model", *component_cols.values()]``.
    zones :
        Ordered list of 6 zone names (fills 2×3 grid).
    period_defs :
        Period specification; controls figure ordering.
    component_cols :
        ``{display_label: column_name_in_df}`` mapping for components to plot.
    component_palette :
        ``{display_label: colour}`` for per-period figures.
    supply_scenario, supply_year :
        Optional annotation injected into figure titles.
    period_palette :
        ``{period_name: colour}`` for the comparison figure.  Defaults to a
        sequential plasma palette from cool (reference) to warm (end-of-century).
    figsize :
        Figure size (used for both per-period and comparison figures).
    path_savefig :
        Directory or path template; files are saved per period plus one
        ``…_period_comparison`` file when provided.
    """
    import matplotlib.colors as mcolors
    import matplotlib.patches as mpatches

    period_order = list(period_defs.keys())
    comp_labels = list(component_cols.keys())
    subtitle = f" ({supply_scenario} {supply_year})" if supply_scenario else ""

    if period_palette is None:
        _pal_rgba = plt.cm.plasma(np.linspace(0.15, 0.85, len(period_order)))
        period_palette = {p: mcolors.to_hex(c) for p, c in zip(period_order, _pal_rgba)}

    for period_name in period_order:
        df_p = df_ens_extreme[df_ens_extreme["Period"] == period_name]

        fig, axes = plt.subplots(2, 3, figsize=figsize, sharey=False, squeeze=False)
        fig.suptitle(
            f"Driver anomaly per EVT event — {period_name}{subtitle}\n"
            f"(GWh; demand +, generation/flexibility −; boxes = IQR, whiskers = 10–90th pct)",
            fontsize=11,
            fontweight="bold",
            y=1.01,
        )

        for ax, zone in zip(axes.flatten(), zones):
            df_z = df_p[df_p["Zone"] == zone]

            # Build long-format table for seaborn
            melt_rows = []
            for label, col in component_cols.items():
                if col not in df_z.columns:
                    continue
                for val in df_z[col].dropna():
                    melt_rows.append({"Component": label, "Value": float(val)})

            if not melt_rows:
                ax.set_visible(False)
                continue

            df_melt = pd.DataFrame(melt_rows)
            present = [lbl for lbl in comp_labels if lbl in df_melt["Component"].unique()]
            palette_list = [component_palette.get(lbl, "#888888") for lbl in present]

            sns.boxplot(
                data=df_melt,
                x="Component",
                y="Value",
                hue="Component",
                order=present,
                palette=palette_list,
                ax=ax,
                width=0.5,
                linewidth=0.8,
                fliersize=0,
                legend=False,
                whis=(10, 90),
            )
            sns.stripplot(
                data=df_melt,
                x="Component",
                y="Value",
                hue="Component",
                order=present,
                palette=["0.25"] * len(present),
                ax=ax,
                size=4,
                alpha=0.65,
                jitter=True,
                zorder=5,
                legend=False,
            )

            ax.axhline(0, color="0.6", lw=0.6, ls="--")
            ax.set_title(zone.replace("_", " "), fontsize=10)
            ax.set_xlabel("")
            ax.set_ylabel("Avg GWh anomaly per event (+ demand, − supply)", fontsize=8)
            ax.tick_params(axis="x", labelsize=7, rotation=40)
            ax.tick_params(axis="y", labelsize=8)
            sns.despine(ax=ax)

        handles = [
            mpatches.Patch(
                facecolor=component_palette.get(lbl, "#888888"), label=lbl, alpha=0.85
            )
            for lbl in comp_labels
        ]
        fig.legend(
            handles=handles,
            loc="lower center",
            ncol=min(len(comp_labels), 4),
            fontsize=8,
            framealpha=0.9,
            title="Driver / Component",
            title_fontsize=8,
        )
        plt.tight_layout(rect=[0, 0.08, 1, 1])

        if path_savefig is not None:
            p = Path(path_savefig)
            safe = period_name.replace(" ", "_").replace("°", "deg").replace(".", "p")
            if p.suffix:
                dest = p.parent / f"{p.stem}_{safe}{p.suffix}"
            else:
                p.mkdir(parents=True, exist_ok=True)
                dest = p / f"component_anomaly_{safe}.png"
            dest.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(dest, dpi=150, bbox_inches="tight")

        plt.show()

    # ── Period-comparison figure ───────────────────────────────────────────────
    # One 2×3 zone grid; within each zone, boxplots are grouped by period so that
    # the shift in each driver across warming levels is directly visible.
    period_pal_list = [period_palette[p] for p in period_order]

    fig_cmp, axes_cmp = plt.subplots(2, 3, figsize=figsize, sharey=False, squeeze=False)
    fig_cmp.suptitle(
        f"Driver anomaly per EVT event — period comparison{subtitle}\n"
        f"(GWh; demand +, generation/flexibility −; boxes = IQR, whiskers = 10–90th pct)",
        fontsize=11,
        fontweight="bold",
        y=1.01,
    )

    for ax, zone in zip(axes_cmp.flatten(), zones):
        melt_rows = []
        for period_name in period_order:
            df_zp = df_ens_extreme[
                (df_ens_extreme["Zone"] == zone) & (df_ens_extreme["Period"] == period_name)
            ]
            for lbl, col in component_cols.items():
                if col not in df_zp.columns:
                    continue
                for val in df_zp[col].dropna():
                    melt_rows.append({"Component": lbl, "Period": period_name, "Value": float(val)})

        if not melt_rows:
            ax.set_visible(False)
            continue

        df_melt = pd.DataFrame(melt_rows)
        present = [lbl for lbl in comp_labels if lbl in df_melt["Component"].unique()]

        sns.boxplot(
            data=df_melt,
            x="Component",
            y="Value",
            hue="Period",
            hue_order=period_order,
            order=present,
            palette=period_pal_list,
            ax=ax,
            width=0.65,
            linewidth=0.7,
            fliersize=0,
            whis=(10, 90),
        )

        ax.axhline(0, color="0.6", lw=0.6, ls="--")
        ax.set_title(zone.replace("_", " "), fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("Avg GWh anomaly per event (+ demand, − supply)", fontsize=8)
        ax.tick_params(axis="x", labelsize=7, rotation=40)
        ax.tick_params(axis="y", labelsize=8)
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        sns.despine(ax=ax)

    period_handles = [
        mpatches.Patch(facecolor=period_palette[p], label=p, alpha=0.85)
        for p in period_order
    ]
    fig_cmp.legend(
        handles=period_handles,
        loc="lower center",
        ncol=len(period_order),
        fontsize=8,
        framealpha=0.9,
        title="Period / Warming level",
        title_fontsize=8,
    )
    plt.tight_layout(rect=[0, 0.06, 1, 1])

    if path_savefig is not None:
        p = Path(path_savefig)
        if p.suffix:
            dest = p.parent / f"{p.stem}_period_comparison{p.suffix}"
        else:
            p.mkdir(parents=True, exist_ok=True)
            dest = p / "component_anomaly_period_comparison.png"
        dest.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(dest, dpi=150, bbox_inches="tight")

    plt.show()


def plot_seasonal_component_anomaly_distributions(
    df_seasonal_evt,
    zones: list,
    season_order: list,
    period_defs: PeriodDefs,
    component_cols: dict[str, str],
    component_palette: dict[str, str],
    supply_scenario: str | None = None,
    supply_year: "int | None" = None,
    period_palette: "dict[str, str] | None" = None,
    season_palette: "dict[str, str] | None" = None,
    figsize: tuple = (16, 9),
    path_savefig: "Path | str | None" = None,
) -> None:
    """Seasonal breakdown of signed driver anomalies per EVT event.

    Produces two sets of figures:

    1. **Per-period figures** (one per warming level): 2×3 zone grid, within each
       zone boxplots with ``x=Component, hue=Season``.  Shows how driver anomalies
       vary by season for a given warming level.
    2. **Per-season figures** (one per season): 2×3 zone grid, within each zone
       boxplots with ``x=Component, hue=Period``.  Shows how each driver's anomaly
       shifts with warming level within each season.

    Input DataFrame (``df_seasonal_evt``) must have lowercase columns ``"zone"``,
    ``"season"``, ``"Period"``, and ``"model"``, as returned by
    ``compute_seasonal_evt_metrics``.

    Sign convention: demand anomaly positive (excess demand); generation and
    flexibility anomalies negative (shortfall / under-dispatch).

    Parameters
    ----------
    df_seasonal_evt :
        Output of ``compute_seasonal_evt_metrics`` collected into a DataFrame.
    zones :
        Ordered list of zone values (fills 2×3 grid).
    season_order :
        Season labels in display order, e.g. ``["DJF", "MAM", "JJA", "SON"]``.
    period_defs :
        Period definitions (controls period ordering).
    component_cols :
        ``{display_label: column_name}`` mapping for the components to plot.
    component_palette :
        ``{display_label: colour}`` used in per-period figures.
    supply_scenario, supply_year :
        Optional annotation in figure titles.
    period_palette :
        ``{period_name: colour}`` for the per-season figures.  Defaults to a
        sequential plasma palette (cool → warm).
    season_palette :
        ``{season_label: colour}`` for the per-period figures.  Defaults to a
        sequential viridis palette across the four seasons.
    figsize :
        Figure size for every panel.
    path_savefig :
        Save directory; files are named ``component_anomaly_seasonal_{period}.png``
        and ``component_anomaly_seasonal_season_{season}.png``.
    """
    import matplotlib.colors as mcolors
    import matplotlib.patches as mpatches

    period_order = list(period_defs.keys())
    comp_labels = list(component_cols.keys())
    subtitle = f" ({supply_scenario} {supply_year})" if supply_scenario else ""

    if period_palette is None:
        _rgba = plt.cm.plasma(np.linspace(0.15, 0.85, len(period_order)))
        period_palette = {p: mcolors.to_hex(c) for p, c in zip(period_order, _rgba)}

    if season_palette is None:
        _s_rgba = plt.cm.viridis(np.linspace(0.1, 0.9, len(season_order)))
        season_palette = {s: mcolors.to_hex(c) for s, c in zip(season_order, _s_rgba)}

    def _build_melt(df_sub, group_col: str) -> "tuple[pd.DataFrame, list[str]]":
        """Long-format rows from df_sub; group_col values become the hue column."""
        rows = []
        for grp_val in df_sub[group_col].unique():
            df_g = df_sub[df_sub[group_col] == grp_val]
            for lbl, col in component_cols.items():
                if col not in df_g.columns:
                    continue
                for val in df_g[col].dropna():
                    rows.append({"Component": lbl, "Group": grp_val, "Value": float(val)})
        df_melt = pd.DataFrame(rows)
        present = [lbl for lbl in comp_labels if lbl in df_melt["Component"].unique()] if rows else []
        return df_melt, present

    def _draw_zone_ax(ax, df_melt, present, hue_order, pal_list):
        if df_melt.empty or not present:
            ax.set_visible(False)
            return
        sns.boxplot(
            data=df_melt, x="Component", y="Value", hue="Group",
            hue_order=hue_order, order=present, palette=pal_list,
            ax=ax, width=0.65, linewidth=0.7, fliersize=0, whis=(10, 90),
        )
        ax.axhline(0, color="0.6", lw=0.6, ls="--")
        ax.set_xlabel("")
        ax.set_ylabel("Avg GWh anomaly per event (+ demand, − supply)", fontsize=8)
        ax.tick_params(axis="x", labelsize=7, rotation=40)
        ax.tick_params(axis="y", labelsize=8)
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        sns.despine(ax=ax)

    def _save(fig, stem: str) -> None:
        if path_savefig is None:
            return
        p = Path(path_savefig)
        safe = stem.replace(" ", "_").replace("°", "deg").replace(".", "p")
        if p.suffix:
            dest = p.parent / f"{p.stem}_{safe}{p.suffix}"
        else:
            p.mkdir(parents=True, exist_ok=True)
            dest = p / f"{safe}.png"
        dest.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(dest, dpi=150, bbox_inches="tight")

    # ── Figure set 1: per period, hue = season ────────────────────────────────
    for period_name in period_order:
        df_p = df_seasonal_evt[df_seasonal_evt["Period"] == period_name]

        fig, axes = plt.subplots(2, 3, figsize=figsize, sharey=False, squeeze=False)
        fig.suptitle(
            f"Seasonal driver anomaly per EVT event — {period_name}{subtitle}\n"
            f"(GWh; demand +, generation/flexibility −; boxes = IQR, whiskers = 10–90th pct)",
            fontsize=11, fontweight="bold", y=1.01,
        )

        for ax, zone in zip(axes.flatten(), zones):
            df_z = df_p[df_p["zone"] == str(zone)]
            df_melt, present = _build_melt(df_z, "season")
            # Restrict hue_order to seasons that are actually present
            present_seasons = [s for s in season_order if s in df_melt["Group"].unique()]
            pal = [season_palette[s] for s in present_seasons]
            ax.set_title(str(zone).replace("_", " "), fontsize=10)
            _draw_zone_ax(ax, df_melt, present, present_seasons, pal)

        season_handles = [
            mpatches.Patch(facecolor=season_palette[s], label=s, alpha=0.85)
            for s in season_order
        ]
        fig.legend(
            handles=season_handles, loc="lower center", ncol=len(season_order),
            fontsize=8, framealpha=0.9, title="Season", title_fontsize=8,
        )
        plt.tight_layout(rect=[0, 0.06, 1, 1])
        _save(fig, f"component_anomaly_seasonal_{period_name}")
        plt.show()

    # ── Figure set 2: per season, hue = period ────────────────────────────────
    for season in season_order:
        df_s = df_seasonal_evt[df_seasonal_evt["season"] == season]

        fig, axes = plt.subplots(2, 3, figsize=figsize, sharey=False, squeeze=False)
        fig.suptitle(
            f"Driver anomaly per EVT event — {season} — period comparison{subtitle}\n"
            f"(GWh; demand +, generation/flexibility −; boxes = IQR, whiskers = 10–90th pct)",
            fontsize=11, fontweight="bold", y=1.01,
        )

        for ax, zone in zip(axes.flatten(), zones):
            df_z = df_s[df_s["zone"] == str(zone)]
            df_melt, present = _build_melt(df_z, "Period")
            present_periods = [p for p in period_order if p in df_melt["Group"].unique()]
            pal = [period_palette[p] for p in present_periods]
            ax.set_title(str(zone).replace("_", " "), fontsize=10)
            _draw_zone_ax(ax, df_melt, present, present_periods, pal)

        period_handles = [
            mpatches.Patch(facecolor=period_palette[p], label=p, alpha=0.85)
            for p in period_order
        ]
        fig.legend(
            handles=period_handles, loc="lower center", ncol=len(period_order),
            fontsize=8, framealpha=0.9, title="Period / Warming level", title_fontsize=8,
        )
        plt.tight_layout(rect=[0, 0.06, 1, 1])
        _save(fig, f"component_anomaly_seasonal_season_{season}")
        plt.show()
