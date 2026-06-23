from pathlib import Path
from typing import Optional

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import regionmask
import seaborn as sns
import xarray as xr
import geopandas as gpd

from .analysis import (
    compute_gmst_anomalies,
    compute_percentage_change,
    compute_sign_agreement,
)

sns.set_style("whitegrid", {"grid.color": ".93"})

FONTSIZE_LEGEND = 18
FONTSIZE_TICKS = 16
FONTSIZE_LABELS = 19
FONTSIZE_TITLE = 22

COMPONENTS_SIZE_CONFIG_DEFAULT = {
    "FONTSIZE_LEGEND": 18,
    "FONTSIZE_TICKS": 16,
    "FONTSIZE_LABELS": 19,
    "FONTSIZE_TITLE": 22,
}

COLORBLIND_PALETTE = sns.color_palette("colorblind")
BLUE = COLORBLIND_PALETTE[0]
ORANGE = COLORBLIND_PALETTE[1]
GREEN = COLORBLIND_PALETTE[2]
RED = COLORBLIND_PALETTE[3]
PURPLE = COLORBLIND_PALETTE[4]
BROWN = COLORBLIND_PALETTE[5]
PINK = COLORBLIND_PALETTE[6]
GREY = COLORBLIND_PALETTE[7]
YELLOW = COLORBLIND_PALETTE[8]
SKY_BLUE = COLORBLIND_PALETTE[9]

river_cmap = mcolors.LinearSegmentedColormap.from_list(
    "river", ["#0b1c2d", "#1c4e80", "#4ea3c8", "#cfe9f2", "#ffffff"]
)

river_cmap = mcolors.LinearSegmentedColormap.from_list(
    "river", ["#1c4e80", "#4ea3c8", "#cfe9f2", "#ffffff"]
)

CMAPS = {
    "pv": "plasma",
    "wind": "Greens",
    "hydropower": "PuBu",
    "hydropower_ror": river_cmap,
    "hydropower_reservoir": river_cmap,
    "HDD": "Blues",
    "CDD": "YlOrRd",
}


def cm_to_inch(cm: float) -> float:
    """
    Convert centimeters to inches for matplotlib figsize.

    Parameters
    ----------
    cm : float
        Length in centimeters.

    Returns
    -------
    float
        Length in inches.
    """
    return cm / 2.54


def parse_figsize(
    figsize: tuple[float, float] | tuple[str, str] | tuple[float | str, float | str],
) -> tuple[float, float]:
    """
    Parse figsize input and convert to matplotlib-compatible format.

    Accepts figsize as:
    - Tuple of numbers: (width, height) in inches
    - Tuple of strings with 'cm': ('20cm', '15cm') converts to inches
    - Mixed formats: (20, '15cm') or ('20cm', 15)

    Parameters
    ----------
    figsize : tuple of float or str
        Figure size as (width, height). Values can be numbers (inches) or
        strings with 'cm' suffix (e.g., '20cm').

    Returns
    -------
    tuple of float
        Figure size in inches as (width, height).

    Examples
    --------
    >>> parse_figsize((10, 8))
    (10, 8)
    >>> parse_figsize(('20cm', '15cm'))
    (7.874015748031496, 5.905511811023622)
    >>> parse_figsize((10, '15cm'))
    (10, 5.905511811023622)
    """
    if not isinstance(figsize, (tuple, list)) or len(figsize) != 2:
        raise ValueError("figsize must be a tuple or list of length 2")

    width, height = figsize

    # Parse width
    if isinstance(width, str):
        if width.lower().endswith("cm"):
            width = cm_to_inch(float(width[:-2]))
        else:
            width = float(width)
    elif not isinstance(width, (int, float)):
        raise ValueError(f"Invalid width type: {type(width)}")

    # Parse height
    if isinstance(height, str):
        if height.lower().endswith("cm"):
            height = cm_to_inch(float(height[:-2]))
        else:
            height = float(height)
    elif not isinstance(height, (int, float)):
        raise ValueError(f"Invalid height type: {type(height)}")

    return (float(width), float(height))


def plot_gmst_anomalies(
    scenarios: list[str],
    subset_gcms: list[Path],
    var: str,
    time_dim: str,
    reference_period: slice,
    figsize=(18, 5),
    components_size_config: dict = None,
    path_savefig: Path = None,
):
    """
    Plot annual GMST anomalies for all GCMs in subset per scenario.

    Parameters
    ----------
    scenarios : list of str
        List of scenario names.
    subset_gcms : list of pathlib.Path
        List of file paths to GCM NetCDF files.
    var : str
        Variable name in the NetCDF files to plot.
    time_dim : str
        Name of the time dimension (e.g., "year").
    reference_period : slice
        Reference period (e.g., slice("1871", "1900")) for anomaly calculation.
    color_palette : list
        List of colors for plotting each GCM.
    figsize : tuple, optional
        Figure size.
    path_savefig : pathlib.Path, optional
        Path to directory where the figure will be saved.

    Returns
    -------
    None
    """

    fig, axes = plt.subplots(1, len(scenarios), figsize=figsize, sharey=True)
    lines = []
    labels = []

    for i, scenario in enumerate(scenarios):
        ax = axes[i]
        j = 0
        for file in subset_gcms:
            if scenario in file.name:
                ds = xr.open_dataset(file).sel(year=slice(reference_period.start, None))
                years, anomalies = compute_gmst_anomalies(
                    ds, var, reference_period, time_dim
                )
                gcm = file.stem.split("_")[2]
                (line,) = ax.plot(
                    years,
                    anomalies.values,
                    label=gcm,
                    alpha=1,
                    color=COLORBLIND_PALETTE[j],
                )
                if gcm not in labels:
                    lines.append(line)
                    labels.append(gcm)
                ds.close()
                j += 1
        ax.set_xlabel("Year", fontsize=FONTSIZE_LABELS)
        if i == 0:
            ax.set_ylabel(
                f"Annual GMST anomaly relative to {reference_period.start}-{reference_period.stop} (K)",
                fontsize=components_size_config.get("FONTSIZE_LABELS", FONTSIZE_LABELS),
            )
        ax.set_title(
            f"{scenario.upper()}",
            fontsize=components_size_config.get("FONTSIZE_TITLE", FONTSIZE_TITLE),
        )
        ax.tick_params(
            axis="both",
            labelsize=components_size_config.get("FONTSIZE_TICKS", FONTSIZE_TICKS),
        )

    fig.suptitle(
        "Annual GMST anomalies for all GCMs in subset (per scenario)",
        fontsize=components_size_config.get("FONTSIZE_TITLE", FONTSIZE_TITLE),
    )
    fig.legend(
        lines,
        labels,
        loc="lower center",
        ncol=len(labels),
        bbox_to_anchor=(0.5, -0.05),
        fontsize=components_size_config.get("FONTSIZE_LEGEND", FONTSIZE_LEGEND),
    )
    plt.tight_layout()
    if path_savefig is not None:
        path_savefig.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path_savefig, dpi=300, bbox_inches="tight")
    plt.show()


def plot_gmst_anomalies_subset_vs_all(
    scenarios: list[str],
    subset_gcms: list[Path],
    all_other_gcms_data: list[Path],
    var: str,
    time_dim: str,
    reference_period: slice,
    figsize=(18, 5),
    path_savefig: Path = None,
    separate_subset: bool = True,
    components_size_config: dict = None,
):
    """
    Plot annual GMST anomalies for all GCMs in subset and other GCMs per scenario.

    Parameters
    ----------
    scenarios : list of str
        List of scenario names.
    subset_gcms : list of pathlib.Path
        List of file paths to subset GCM NetCDF files.
    all_other_gcms_data : list of pathlib.Path
        List of file paths to other GCM NetCDF files.
    var : str
        Variable name in the NetCDF files to plot.
    time_dim : str
        Name of the time dimension (e.g., "year").
    reference_period : slice
        Reference period (e.g., slice("1871", "1900")) for anomaly calculation.
    color_palette : list, optional
        List of colors for plotting each GCM.
    figsize : tuple, optional
        Figure size.
    path_savefig : pathlib.Path, optional
        Path to directory where the figure will be saved.
    separate_subset : bool, optional
        If True, plot each subset GCM with a different color and label.
        If False, bundle all subset GCMs together with the same color and label "Subset GCMs".

    Returns
    -------
    None
    """
    annual_anomalies = {sc: [] for sc in scenarios}
    labels_dict = {sc: [] for sc in scenarios}
    colors_dict = {sc: [] for sc in scenarios}
    alphas_dict = {sc: [] for sc in scenarios}

    for sc in scenarios:
        # Other GCMs
        for file in [f for f in all_other_gcms_data if sc in f.name]:
            ds = (
                xr.open_dataset(file)
                .sortby(time_dim)
                .sel({time_dim: slice(reference_period.start, None)})
            )
            years, anomalies = compute_gmst_anomalies(
                ds, var, reference_period, time_dim
            )
            annual_anomalies[sc].append((years, anomalies.values))
            labels_dict[sc].append("All GCMs")
            colors_dict[sc].append(GREY)
            alphas_dict[sc].append(0.1)
            ds.close()
        # Subset GCMs
        subset_files = [f for f in subset_gcms if sc in f.name]
        for j, file in enumerate(subset_files):
            ds = (
                xr.open_dataset(file)
                .sortby(time_dim)
                .sel({time_dim: slice(reference_period.start, None)})
            )
            years, anomalies = compute_gmst_anomalies(
                ds, var, reference_period, time_dim
            )
            gcm = file.stem.split("_")[2]
            annual_anomalies[sc].append((years, anomalies.values))
            labels_dict[sc].append(gcm if separate_subset else "Subset GCMs")
            colors_dict[sc].append(
                COLORBLIND_PALETTE[j % len(COLORBLIND_PALETTE)]
                if separate_subset
                else COLORBLIND_PALETTE[0]
            )
            alphas_dict[sc].append(1.0 if separate_subset else 0.8)
            ds.close()

    fig, axes = plt.subplots(1, len(scenarios), figsize=figsize, sharey=True)
    if len(scenarios) == 1:
        axes = [axes]
    all_handles = []
    all_labels = []

    for i, sc in enumerate(scenarios):
        ax = axes[i]
        for (years, anomalies), label, color, alpha in zip(
            annual_anomalies[sc], labels_dict[sc], colors_dict[sc], alphas_dict[sc]
        ):
            (handle,) = ax.plot(years, anomalies, color=color, alpha=alpha, label=label)
            if label not in all_labels:
                all_handles.append(handle)
                all_labels.append(label)
        ax.set_xlabel(
            "Year",
            fontsize=components_size_config.get("FONTSIZE_LABELS", FONTSIZE_LABELS),
        )
        if i == 0:
            ax.set_ylabel(
                f"Annual GMST anomaly relative to {reference_period.start}-{reference_period.stop} (K)",
                fontsize=components_size_config.get("FONTSIZE_LABELS", FONTSIZE_LABELS),
            )
        ax.set_title(
            sc.upper(),
            fontsize=components_size_config.get("FONTSIZE_TITLE", FONTSIZE_TITLE),
        )
        ax.tick_params(
            axis="both",
            labelsize=components_size_config.get("FONTSIZE_TICKS", FONTSIZE_TICKS),
        )

    fig.suptitle(
        "Annual GMST anomalies for selected and other GCMs (per scenario)",
        fontsize=components_size_config.get("FONTSIZE_TITLE", FONTSIZE_TITLE),
    )
    fig.legend(
        all_handles,
        all_labels,
        loc="lower center",
        ncol=len(all_labels),
        bbox_to_anchor=(0.5, -0.05),
        fontsize=components_size_config.get("FONTSIZE_LEGEND", FONTSIZE_LEGEND),
    )
    plt.tight_layout()
    if path_savefig is not None:
        path_savefig.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path_savefig, dpi=300, bbox_inches="tight")
    plt.show()


def apply_land_mask(da: xr.DataArray) -> xr.DataArray:
    """
    Apply a land mask to the input DataArray.

    Parameters
    ----------
    da : xr.DataArray
        Input DataArray with 'lat' and 'lon' dimensions.

    Returns
    -------
    xr.DataArray
        DataArray with NaN values where the land mask is False.
    """
    # Create a land mask using Natural Earth land polygons
    # Higher resolution options: land_50 (medium), land_10 (high)
    # land_10 is the highest resolution available in regionmask
    land = regionmask.defined_regions.natural_earth_v5_1_2.land_10

    # Example: mask mean_ref (repeat for other datasets as needed)
    # This will create a mask with 1 for land, NaN for ocean
    land_mask = land.mask(da)
    return da.where(~np.isnan(land_mask))


def create_shared_colorbar(
    fig,
    axes,
    idxs,
    im,
    label,
    orientation="horizontal",
    shrink=0.3,
    aspect=70,
    width_ratio=0.6,
):
    """
    Create a shared colorbar for a group of subplots.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure object.
    axes : array-like
        List or array of axes.
    idxs : list of int
        Indices of axes to share the colorbar.
    im : matplotlib.image.AxesImage or QuadMesh
        The mappable object (from imshow, pcolormesh, etc.).
    label : str
        Colorbar label.
    orientation : str, optional
        Orientation of the colorbar ("horizontal" or "vertical").
    fontsize : int, optional
        Font size for the colorbar label and ticks.
    """

    from matplotlib.transforms import Bbox

    selected_axes = [axes[i] for i in idxs]
    # Get position that spans all selected axes
    bboxes = [ax.get_position() for ax in selected_axes]
    bbox = Bbox.union(bboxes)
    if orientation == "horizontal":
        cbar_height = 0.015
        cbar_pad = 0.02
        cbar_width = bbox.width * width_ratio
        cbar_x0 = bbox.x0 + (bbox.width - cbar_width) / 2
        cbar_ax = fig.add_axes(
            [cbar_x0, bbox.y0 - cbar_pad - cbar_height, cbar_width, cbar_height]
        )
    else:
        # Place colorbar to the right of the axes, not too far
        cbar_width = 0.02
        cbar_pad = 0.01
        cbar_ax = fig.add_axes([bbox.x1 + cbar_pad, bbox.y0, cbar_width, bbox.height])
    cbar = fig.colorbar(
        im, cax=cbar_ax, orientation=orientation, aspect=aspect, shrink=shrink
    )
    cbar.set_label(label, fontsize=FONTSIZE_LABELS)
    cbar.ax.tick_params(labelsize=FONTSIZE_TICKS)


def compute_vmin_vmax(
    ds: xr.Dataset,
    var_names: list[str],
    quantiles: tuple = None,
    center_zero: bool = False,
):
    """
    Compute a single vmin/vmax across all specified variables.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to compute vmin and vmax values for.
    var_names : list of str
        List of variable names to compute limits for.
    quantiles : tuple, optional
        Quantiles to use for min/max (e.g., (0.025, 0.975)). If None, use min/max.
    center_zero : bool, optional
        If True, make vmin/vmax symmetric around zero.

    Returns
    -------
    tuple
        (vmin, vmax) for all variables combined.
    """
    vals = []
    for v in var_names:
        if v in ds:
            vals.append(ds[v].values)
    if not vals:
        return None, None
    vals = np.concatenate([np.ravel(x) for x in vals])
    if quantiles is not None:
        vmin = float(np.nanquantile(vals, quantiles[0]))
        vmax = float(np.nanquantile(vals, quantiles[1]))
    else:
        vmin = float(np.nanmin(vals))
        vmax = float(np.nanmax(vals))
    if center_zero:
        absmax = max(abs(vmin), abs(vmax))
        vmin, vmax = -absmax, absmax
    return vmin, vmax


def plot_ensemble_mean(
    ds: xr.Dataset,
    var_titles: dict = None,
    var_clabel: dict = None,
    components_size_config: dict = None,
    log_scale_vars: list = None,
    simplify_colorbar: bool = False,
    figsize: tuple = None,
    country_border_width: float = 1.0,
    orientation: str = "horizontal",
    title: str = "Ensemble mean",
    path_savefig: Path = None,
):
    """
    Plot ensemble mean for each variable in the dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing variables with dimensions 'lat', 'lon', and 'model'.
    var_titles : dict, optional
        Dictionary mapping variable names to plot titles. If None, uses default titles.
    var_clabel : dict, optional
        Dictionary mapping variable names to colorbar labels. If None, uses default labels.
    components_size_config : dict, optional
        Dictionary with size configuration for plot components (fontsizes, colorbar params).
        If None, uses default sizes.
    log_scale_vars : list, optional
        List of variable names whose colorbar ticks should be transformed back from log10.
    simplify_colorbar : bool, optional
        If True, replace colorbar ticks with "Less" and "More" labels (default: False).
    figsize : tuple, optional
        Figure size. If None, automatically sized based on number of variables.
    country_border_width : float, optional
        Width of country borders (default: 1.0).
    orientation : str, optional
        Orientation of subplots: "horizontal" (nrows=1) or "vertical" (ncols=1). Default: "horizontal".
    title : str, optional
        Plot title (default: "Ensemble mean").
    path_savefig : pathlib.Path, optional
        Path to save the figure.

    Returns
    -------
    None
    """
    if var_titles is None:
        var_titles = {
            "pv": "Solar Power Potential",
            "wind": "Wind Power Potential",
            "hydropower_ror": "Run-of-river Hydropower Potential",
            "hydropower_reservoir": "Reservoir Hydropower Inflows",
            "HDD": "Heating Degree Days",
            "CDD": "Cooling Degree Days",
        }
    if var_clabel is None:
        var_clabel = {
            "pv": "Capacity Factor (-)",
            "wind": "Capacity Factor (-)",
            "hydropower_ror": "Generation (MWh)",
            "hydropower_reservoir": "Weekly Inflows (m³/s)",
            "HDD": "Degree-Days",
            "CDD": "Degree-Days",
        }
    if components_size_config is None:
        components_size_config = COMPONENTS_SIZE_CONFIG_DEFAULT
    if log_scale_vars is None:
        log_scale_vars = []

    var_names = list(ds.data_vars.keys())
    n_vars = len(var_names)

    if figsize is None:
        if orientation == "horizontal":
            figsize = (4 * n_vars, 6)
        else:
            figsize = (6, 4 * n_vars)
    figsize = parse_figsize(figsize)

    # Extract configuration values
    fontsize_legend = components_size_config.get("FONTSIZE_LEGEND", FONTSIZE_LEGEND)
    fontsize_ticks = components_size_config.get("FONTSIZE_TICKS", FONTSIZE_TICKS)
    fontsize_labels = components_size_config.get("FONTSIZE_LABELS", FONTSIZE_LABELS)
    fontsize_title = components_size_config.get("FONTSIZE_TITLE", FONTSIZE_TITLE)
    cbar_shrink = components_size_config.get("CBAR_SHRINK", 0.75)
    cbar_aspect = components_size_config.get("CBAR_ASPECT", 15)
    cbar_pad = components_size_config.get("CBAR_PAD", 0.03)

    min_lat, max_lat = ds.lat.min().item(), ds.lat.max().item()
    central_lon = (ds.lon.min().item() + ds.lon.max().item()) / 2

    if orientation == "horizontal":
        nrows, ncols = 1, n_vars
    else:
        nrows, ncols = n_vars, 1

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        subplot_kw={
            "projection": ccrs.Mercator(
                central_longitude=central_lon,
                min_latitude=min_lat,
                max_latitude=max_lat,
            )
        },
    )

    if n_vars == 1:
        axes = [axes]

    ens_mean = ds.mean(dim="model")

    for i, var in enumerate(var_names):
        ax = axes[i]

        # Prepare normalization and limits
        norm = None
        vmin_val = None
        vmax_val = None

        # Determine vmin/vmax based on variable type
        if var in ["hydropower_ror", "hydropower_reservoir"]:
            vmin_val = float(ens_mean[var].min().values)
            # vmin_val = float(ens_mean[var].quantile(0.02).values)
            vmax_val = float(ens_mean[var].quantile(0.985).values)
            # vmax_val = float(ens_mean[var].max().values)

        elif var in ["HDD", "CDD"]:
            vmin_val = float(ens_mean[var].min().values)
            vmax_val = float(ens_mean[var].quantile(0.98).values)
        else:
            vmin_val = float(ens_mean[var].min().values)
            vmax_val = float(ens_mean[var].max().values)
            # vmin_val = float(ens_mean[var].quantile(0.02).values)
            vmax_val = float(ens_mean[var].quantile(0.99).values)

        # For log scale variables, adjust vmin to highlight higher numbers (>= 1 in original space)
        if var in log_scale_vars:
            # Set vmin to at least 0 in log space (10^-1 = 0.1 in original space)
            vmin_val = max(vmin_val, -1)

        # Prepare colormap
        cmap = CMAPS.get(var, "viridis")

        im = ens_mean[var].plot(
            ax=ax,
            cmap=cmap,
            transform=ccrs.PlateCarree(),
            add_colorbar=True,
            norm=norm,
            vmin=vmin_val,
            vmax=vmax_val,
            cbar_kwargs={
                "shrink": cbar_shrink,
                "aspect": cbar_aspect,
                "pad": cbar_pad,
                "orientation": "horizontal",
            },
        )
        cbar = im.colorbar
        # Apply log scale transformation if needed
        if var in log_scale_vars:
            # Get actual data range to avoid extending beyond it
            data_min = max(ens_mean[var].min().values, -1)
            data_max = ens_mean[var].max().values
            # data_min = ens_mean[var].quantile(0.05).values if var in ["hydropower_ror", "hydropower_reservoir"] else data_min
            data_max = (
                ens_mean[var].quantile(0.995).values
                if var in ["hydropower_ror", "hydropower_reservoir"]
                else data_max
            )
            ticks = cbar.get_ticks()
            # Filter ticks to only include integer powers within data range
            ticks = ticks[(ticks >= data_min) & (ticks <= data_max)]
            ticks = ticks[
                np.abs(ticks - np.round(ticks)) < 1e-10
            ]  # Keep only integer values
            cbar.set_ticks(ticks)
            cbar.set_ticklabels([f"{10**val:.0f}" for val in ticks])

        cbar.set_label(var_clabel.get(var, ""), fontsize=fontsize_labels - 1)
        cbar.ax.tick_params(labelsize=fontsize_ticks)

        # Simplify colorbar if requested
        if simplify_colorbar:
            cbar.set_ticks([cbar.vmin, cbar.vmax])
            cbar.set_ticklabels(["Less", "More"])
            cbar.ax.tick_params(length=0)

        ax.coastlines()
        ax.add_feature(
            cfeature.BORDERS,
            edgecolor="black",
            linewidth=country_border_width,
            zorder=6,
        )
        ax.set_title(var_titles.get(var, var), fontsize=fontsize_labels)
        ax.set_xlabel("")
        ax.set_ylabel("")

    if title is not None:
        plt.suptitle(title, fontsize=fontsize_title, y=0.9)
    plt.tight_layout()
    if path_savefig is not None:
        path_savefig.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path_savefig, dpi=300, bbox_inches="tight")
    plt.show()


def plot_ensemble_mean_seasonal(
    ds: xr.Dataset,
    seasons: list,
    season_labels: dict = None,
    var_titles: dict = None,
    var_clabel: dict = None,
    components_size_config: dict = None,
    log_scale_vars: list = None,
    simplify_colorbar: bool = False,
    figsize: tuple = None,
    country_border_width: float = 1.0,
    title: str = None,
    path_savefig: Path = None,
    transpose: bool = False,
):
    """
    Plot ensemble mean maps for multiple variables and seasons in a grid.

    By default (``transpose=False``) rows correspond to variables and columns to
    seasons. Each row shares a common colorbar on the right.

    When ``transpose=True`` the layout is flipped: rows correspond to seasons
    and columns to variables. Each column shares a common colorbar at the bottom.
    This compact layout is especially suited for a small number of seasons (e.g.
    the two energy half-year seasons).

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with dimensions ``(model, lat, lon)`` and a ``season`` coordinate,
        i.e. the result of ``mean_ref.sel(season=...)``. Must be sliceable via
        ``ds.sel(season=s)`` for each ``s`` in ``seasons``.
    seasons : list
        Ordered list of season values to plot (one column each in default layout,
        one row each in transposed layout).
    season_labels : dict, optional
        Mapping from season value to display string.
        Defaults to the season value itself.
    var_titles : dict, optional
        Mapping from variable name to label string.
    var_clabel : dict, optional
        Mapping from variable name to colorbar label string.
    components_size_config : dict, optional
        Font-size and colorbar configuration dict (same keys as used elsewhere).
        Extra key ``CBAR_HEIGHT`` (float, default 0.05) controls the height of
        the colorbar row in the transposed layout.
    log_scale_vars : list, optional
        Variable names for which colorbar ticks are transformed back from log10.
    simplify_colorbar : bool, optional
        Replace colorbar ticks with "Less"/"More" when True.
    figsize : tuple, optional
        Figure size in inches (or centimeter strings). Auto-sized if None.
    country_border_width : float, optional
        Width of country border lines.
    title : str, optional
        Overall figure suptitle.
    path_savefig : Path, optional
        If provided, save the figure to this path at 300 dpi.
    transpose : bool, optional
        When True, plot seasons as rows and variables as columns (default False).
    """
    if var_titles is None:
        var_titles = {
            "pv": "Solar Power Potential",
            "wind": "Wind Power Potential",
            "hydropower_ror": "Run-of-river Hydropower",
            "hydropower_reservoir": "Reservoir Hydropower Inflows",
            "HDD": "Heating Degree Days",
            "CDD": "Cooling Degree Days",
        }
    if var_clabel is None:
        var_clabel = {
            "pv": "Capacity Factor (-)",
            "wind": "Capacity Factor (-)",
            "hydropower_ror": "Generation (MWh)",
            "hydropower_reservoir": "Weekly Inflows (m³/s)",
            "HDD": "Degree-Days",
            "CDD": "Degree-Days",
        }
    if components_size_config is None:
        components_size_config = COMPONENTS_SIZE_CONFIG_DEFAULT
    if log_scale_vars is None:
        log_scale_vars = []
    if season_labels is None:
        season_labels = {s: str(s) for s in seasons}

    from matplotlib.gridspec import GridSpec

    var_names = list(ds.data_vars.keys())
    n_vars = len(var_names)
    n_seasons = len(seasons)

    fontsize_ticks = components_size_config.get("FONTSIZE_TICKS", FONTSIZE_TICKS)
    fontsize_labels = components_size_config.get("FONTSIZE_LABELS", FONTSIZE_LABELS)
    fontsize_title = components_size_config.get("FONTSIZE_TITLE", FONTSIZE_TITLE)
    cbar_width = components_size_config.get("CBAR_WIDTH", 0.03)
    cbar_pad = components_size_config.get("CBAR_PAD", 0.02)
    hspace = components_size_config.get("HSPACE", 0.02)
    wspace = components_size_config.get("WSPACE", 0.02)

    min_lat = ds.lat.min().item()
    max_lat = ds.lat.max().item()
    central_lon = (ds.lon.min().item() + ds.lon.max().item()) / 2

    proj = ccrs.Mercator(
        central_longitude=central_lon,
        min_latitude=min_lat,
        max_latitude=max_lat,
    )

    fig = plt.figure(figsize=parse_figsize(figsize) if figsize is not None else None)

    if not transpose:
        # ── Default layout: rows = variables, columns = seasons ──────────────
        if figsize is None:
            fig.set_size_inches(4 * n_seasons + 1, 3.5 * n_vars)

        # GridSpec: n_seasons map columns + 1 narrow colorbar column
        gs = GridSpec(
            n_vars,
            n_seasons + 1,
            figure=fig,
            width_ratios=[1.0] * n_seasons + [cbar_width],
            hspace=hspace,
            wspace=wspace,
        )

        map_axes = np.empty((n_vars, n_seasons), dtype=object)
        cbar_axes = []
        for row in range(n_vars):
            for col in range(n_seasons):
                map_axes[row, col] = fig.add_subplot(gs[row, col], projection=proj)
            cbar_axes.append(fig.add_subplot(gs[row, n_seasons]))

        for row_idx, var in enumerate(var_names):
            season_means = xr.concat(
                [ds.sel(season=s).mean(dim="model")[var] for s in seasons],
                dim="season",
            )
            if var in ["hydropower_ror", "hydropower_reservoir"]:
                vmin_val = float(season_means.min().values)
                vmax_val = float(season_means.quantile(0.985).values)
            elif var in ["HDD", "CDD"]:
                vmin_val = float(season_means.min().values)
                vmax_val = float(season_means.quantile(0.98).values)
            else:
                vmin_val = float(season_means.min().values)
                vmax_val = float(season_means.quantile(0.99).values)

            if var in log_scale_vars:
                vmin_val = max(vmin_val, -1)

            cmap = CMAPS.get(var, "viridis")
            last_im = None

            for col_idx, season in enumerate(seasons):
                ax = map_axes[row_idx, col_idx]
                ens_mean_s = ds.sel(season=season).mean(dim="model")

                last_im = ens_mean_s[var].plot(
                    ax=ax,
                    cmap=cmap,
                    transform=ccrs.PlateCarree(),
                    add_colorbar=False,
                    vmin=vmin_val,
                    vmax=vmax_val,
                )

                ax.coastlines(linewidth=0.5)
                ax.add_feature(
                    cfeature.BORDERS,
                    edgecolor="black",
                    linewidth=country_border_width,
                    zorder=6,
                )
                ax.set_title(
                    season_labels.get(season, str(season)),
                    fontsize=fontsize_labels,
                    pad=4,
                )
                ax.set_xlabel("")
                ax.set_ylabel("")

            cbar = fig.colorbar(last_im, cax=cbar_axes[row_idx], orientation="vertical")
            cbar.set_label(var_clabel.get(var, ""), fontsize=fontsize_labels - 1)
            cbar.ax.tick_params(labelsize=fontsize_ticks)

            if simplify_colorbar:
                cbar.set_ticks([vmin_val, vmax_val])
                cbar.set_ticklabels(["Less", "More"])
                cbar.ax.tick_params(length=0)

            if var in log_scale_vars:
                ticks = cbar.get_ticks()
                data_min = max(season_means.min().values, -1)
                data_max = season_means.quantile(0.995).values
                ticks = ticks[(ticks >= data_min) & (ticks <= data_max)]
                ticks = ticks[np.abs(ticks - np.round(ticks)) < 1e-10]
                cbar.set_ticks(ticks)
                cbar.set_ticklabels([f"{10**val:.0f}" for val in ticks])

            # Row label: variable title to the left of the first column
            map_axes[row_idx, 0].text(
                -0.08,
                0.5,
                var_titles.get(var, var),
                transform=map_axes[row_idx, 0].transAxes,
                rotation=90,
                va="center",
                ha="right",
                fontsize=fontsize_labels,
                fontweight="bold",
                clip_on=False,
            )

    else:
        # ── Transposed layout: rows = seasons, columns = variables ────────────
        # Each variable column shares a common horizontal colorbar at the bottom.
        cbar_height = components_size_config.get("CBAR_HEIGHT", 0.05)

        if figsize is None:
            fig.set_size_inches(4 * n_vars, 3.5 * n_seasons + 1)

        gs = GridSpec(
            n_seasons + 1,
            n_vars,
            figure=fig,
            height_ratios=[1.0] * n_seasons + [cbar_height],
            hspace=hspace,
            wspace=wspace,
        )

        map_axes = np.empty((n_seasons, n_vars), dtype=object)
        cbar_axes = []
        for col in range(n_vars):
            for row in range(n_seasons):
                map_axes[row, col] = fig.add_subplot(gs[row, col], projection=proj)
            cbar_axes.append(fig.add_subplot(gs[n_seasons, col]))

        for col_idx, var in enumerate(var_names):
            season_means = xr.concat(
                [ds.sel(season=s).mean(dim="model")[var] for s in seasons],
                dim="season",
            )
            if var in ["hydropower_ror", "hydropower_reservoir"]:
                vmin_val = float(season_means.min().values)
                vmax_val = float(season_means.quantile(0.985).values)
            elif var in ["HDD", "CDD"]:
                vmin_val = float(season_means.min().values)
                vmax_val = float(season_means.quantile(0.98).values)
            else:
                vmin_val = float(season_means.min().values)
                vmax_val = float(season_means.quantile(0.99).values)

            if var in log_scale_vars:
                vmin_val = max(vmin_val, -1)

            cmap = CMAPS.get(var, "viridis")
            last_im = None

            for row_idx, season in enumerate(seasons):
                ax = map_axes[row_idx, col_idx]
                ens_mean_s = ds.sel(season=season).mean(dim="model")

                last_im = ens_mean_s[var].plot(
                    ax=ax,
                    cmap=cmap,
                    transform=ccrs.PlateCarree(),
                    add_colorbar=False,
                    vmin=vmin_val,
                    vmax=vmax_val,
                )

                ax.coastlines(linewidth=0.5)
                ax.add_feature(
                    cfeature.BORDERS,
                    edgecolor="black",
                    linewidth=country_border_width,
                    zorder=6,
                )
                ax.set_xlabel("")
                ax.set_ylabel("")

                # Season label to the left of the first column only
                if col_idx == 0:
                    ax.text(
                        -0.08,
                        0.5,
                        season_labels.get(season, str(season)),
                        transform=ax.transAxes,
                        rotation=90,
                        va="center",
                        ha="right",
                        fontsize=fontsize_labels,
                        fontweight="bold",
                        clip_on=False,
                    )

                # Variable title at the top of the first season row only;
                # clear the auto-title that xarray sets for all other rows.
                if row_idx == 0:
                    ax.set_title(
                        var_titles.get(var, var),
                        fontsize=fontsize_labels,
                        pad=4,
                    )
                else:
                    ax.set_title("")

            # Horizontal colorbar at the bottom of each variable column
            cbar = fig.colorbar(
                last_im,
                cax=cbar_axes[col_idx],
                orientation="horizontal",
            )
            cbar.set_label(var_clabel.get(var, ""), fontsize=fontsize_labels - 1)
            cbar.ax.tick_params(labelsize=fontsize_ticks)

            if simplify_colorbar:
                cbar.set_ticks([vmin_val, vmax_val])
                cbar.set_ticklabels(["Less", "More"])
                cbar.ax.tick_params(length=0)

            if var in log_scale_vars:
                ticks = cbar.get_ticks()
                data_min = max(season_means.min().values, -1)
                data_max = season_means.quantile(0.995).values
                ticks = ticks[(ticks >= data_min) & (ticks <= data_max)]
                ticks = ticks[np.abs(ticks - np.round(ticks)) < 1e-10]
                cbar.set_ticks(ticks)
                cbar.set_ticklabels([f"{10**val:.0f}" for val in ticks])

    if title is not None:
        fig.suptitle(
            title,
            fontsize=fontsize_title,
            y=components_size_config.get("SUPTITLE_Y", 0.97),
        )
    if path_savefig is not None:
        path_savefig.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path_savefig, dpi=300, bbox_inches="tight")
    plt.show()


def plot_ensemble_mean_change_seasonal(
    ref_ds: xr.Dataset,
    target_ds: xr.Dataset,
    seasons: list,
    season_labels: dict = None,
    var_titles: dict = None,
    var_clabel: dict = None,
    components_size_config: dict = None,
    absolute_change: bool = False,
    simplify_colorbar: bool = False,
    mask_insignificant: bool = False,
    significance_threshold=None,
    vmin: dict | float = None,
    vmax: dict | float = None,
    figsize: tuple = None,
    country_border_width: float = 0.5,
    title: str = None,
    path_savefig: Path = None,
    transpose: bool = False,
):
    """
    Plot ensemble mean change maps for multiple variables and seasons in a grid.

    By default (``transpose=False``) rows correspond to variables and columns to
    seasons, with a vertical colorbar per row. When ``transpose=True`` rows
    correspond to seasons and columns to variables, with a horizontal colorbar
    per column. Grid cells where fewer than 2/3 of models agree on the sign of
    change are hatched.

    Parameters
    ----------
    ref_ds : xr.Dataset
        Reference-period dataset with dimensions ``(model, season, lat, lon)``.
    target_ds : xr.Dataset
        Future-period dataset with the same structure as ``ref_ds``.
    seasons : list
        Ordered list of season values to plot.
    season_labels : dict, optional
        Mapping from season value to display string.
    var_titles : dict, optional
        Mapping from variable name to label string.
    var_clabel : dict, optional
        Mapping from variable name to colorbar label. Defaults to
        "% change" or "Absolute change" depending on ``absolute_change``.
    components_size_config : dict, optional
        Font-size and colorbar configuration dict. Recognised keys:
        ``FONTSIZE_TICKS``, ``FONTSIZE_LABELS``, ``FONTSIZE_TITLE``,
        ``CBAR_WIDTH``, ``CBAR_HEIGHT``, ``CBAR_PAD``, ``HSPACE``, ``WSPACE``,
        ``SUPTITLE_Y``.
    absolute_change : bool, optional
        If True, plot absolute change instead of percentage change.
    simplify_colorbar : bool, optional
        Replace colorbar ticks with "Less" / "More" labels.
    mask_insignificant : bool, optional
        Mask grid points where the ensemble-mean absolute change is below
        ``significance_threshold``.
    significance_threshold : dict or float, optional
        Absolute-change threshold per variable (dict) or global (float).
        Applied only when ``mask_insignificant=True``. Defaults to 0.01.
    vmin : dict or float, optional
        Override the auto-computed colorbar minimum.
    vmax : dict or float, optional
        Override the auto-computed colorbar maximum.
    figsize : tuple, optional
        Figure size in inches. Auto-sized when None.
    country_border_width : float, optional
        Width of country border lines (default: 0.5).
    title : str, optional
        Overall figure suptitle.
    path_savefig : Path, optional
        If provided, the figure is saved here at 300 dpi.
    transpose : bool, optional
        When True, plot seasons as rows and variables as columns (default False).
    """
    if var_titles is None:
        var_titles = {
            "pv": "Solar Power Potential",
            "wind": "Wind Power Potential",
            "hydropower_ror": "Run-of-river Hydropower",
            "hydropower_reservoir": "Reservoir Hydropower Inflows",
            "HDD": "Heating Degree Days",
            "CDD": "Cooling Degree Days",
        }
    if var_clabel is None:
        default_label = "Absolute change" if absolute_change else "% change"
        var_clabel = {var: default_label for var in ref_ds.data_vars}
    if components_size_config is None:
        components_size_config = COMPONENTS_SIZE_CONFIG_DEFAULT
    if season_labels is None:
        season_labels = {s: str(s) for s in seasons}

    # Significance threshold as a per-variable dict
    if mask_insignificant and significance_threshold is None:
        significance_threshold = {var: 0.01 for var in ref_ds.data_vars}
    elif mask_insignificant and isinstance(significance_threshold, (int, float)):
        significance_threshold = {
            var: significance_threshold for var in ref_ds.data_vars
        }

    from matplotlib.gridspec import GridSpec

    var_names = list(ref_ds.data_vars.keys())
    n_vars = len(var_names)
    n_seasons = len(seasons)

    fontsize_ticks = components_size_config.get("FONTSIZE_TICKS", FONTSIZE_TICKS)
    fontsize_labels = components_size_config.get("FONTSIZE_LABELS", FONTSIZE_LABELS)
    fontsize_title = components_size_config.get("FONTSIZE_TITLE", FONTSIZE_TITLE)
    cbar_width = components_size_config.get("CBAR_WIDTH", 0.03)
    hspace = components_size_config.get("HSPACE", 0.02)
    wspace = components_size_config.get("WSPACE", 0.02)

    min_lat = ref_ds.lat.min().item()
    max_lat = ref_ds.lat.max().item()
    central_lon = (ref_ds.lon.min().item() + ref_ds.lon.max().item()) / 2

    proj = ccrs.Mercator(
        central_longitude=central_lon,
        min_latitude=min_lat,
        max_latitude=max_lat,
    )

    fig = plt.figure(figsize=parse_figsize(figsize) if figsize is not None else None)

    if not transpose:
        if figsize is None:
            fig.set_size_inches(4 * n_seasons + 1, 3.5 * n_vars)
        gs = GridSpec(
            n_vars,
            n_seasons + 1,
            figure=fig,
            width_ratios=[1.0] * n_seasons + [cbar_width],
            hspace=hspace,
            wspace=wspace,
        )
        map_axes = np.empty((n_vars, n_seasons), dtype=object)
        cbar_axes = []
        for row in range(n_vars):
            for col in range(n_seasons):
                map_axes[row, col] = fig.add_subplot(gs[row, col], projection=proj)
            cbar_axes.append(fig.add_subplot(gs[row, n_seasons]))
    else:
        cbar_height = components_size_config.get("CBAR_HEIGHT", 0.05)
        if figsize is None:
            fig.set_size_inches(4 * n_vars, 3.5 * n_seasons + 1)
        gs = GridSpec(
            n_seasons + 1,
            n_vars,
            figure=fig,
            height_ratios=[1.0] * n_seasons + [cbar_height],
            hspace=hspace,
            wspace=wspace,
        )
        map_axes = np.empty((n_seasons, n_vars), dtype=object)
        cbar_axes = []
        for col in range(n_vars):
            for row in range(n_seasons):
                map_axes[row, col] = fig.add_subplot(gs[row, col], projection=proj)
            cbar_axes.append(fig.add_subplot(gs[n_seasons, col]))

    # ── Pre-compute change and sign-agreement for every season ────────────────
    all_ens_means = {}
    all_agree_masks = {var: {} for var in var_names}
    for season in seasons:
        ref_s = ref_ds.sel(season=season)
        tgt_s = target_ds.sel(season=season)

        if absolute_change:
            change_s = tgt_s - ref_s
        else:
            change_s = compute_percentage_change(ref_s, tgt_s, round=2)

        ens_mean_s = change_s.mean(dim="model")

        if mask_insignificant:
            abs_change_s = tgt_s - ref_s
            for _var in var_names:
                _threshold = significance_threshold.get(_var, 0.01)
                insig = np.abs(abs_change_s[_var].mean(dim="model")) < _threshold
                ens_mean_s[_var] = ens_mean_s[_var].where(~insig)

        all_ens_means[season] = ens_mean_s

        sign_agree = compute_sign_agreement(change_s, model_dim="model")
        for _var in var_names:
            n_models_per_point = (~np.isnan(change_s[_var])).sum(dim="model")
            threshold_agree = 2 * n_models_per_point / 3
            agree_mask = ~(sign_agree[_var] >= threshold_agree)
            all_agree_masks[_var][season] = agree_mask.where(
                ~np.isnan(ens_mean_s[_var])
            )

    # ── Auto vmin/vmax ─────────────────────────────────────────────────────────
    combined_ens_mean = xr.concat(
        [all_ens_means[s] for s in seasons], dim="season"
    ).mean(dim="season")

    pv_wind_vmin, pv_wind_vmax = compute_vmin_vmax(
        combined_ens_mean, ["pv", "wind"], quantiles=None, center_zero=True
    )
    hydropower_ror_vmin, hydropower_ror_vmax = compute_vmin_vmax(
        combined_ens_mean, ["hydropower_ror"], quantiles=(0.01, 0.99), center_zero=True
    )
    if absolute_change:
        hydropower_ror_vmin, hydropower_ror_vmax = pv_wind_vmin, pv_wind_vmax
    hydropower_reservoir_vmin, hydropower_reservoir_vmax = compute_vmin_vmax(
        combined_ens_mean,
        ["hydropower_reservoir"],
        quantiles=(0.01, 0.99),
        center_zero=True,
    )
    hdd_vmin, hdd_vmax = compute_vmin_vmax(
        combined_ens_mean, ["HDD"], quantiles=(0.05, 0.95), center_zero=True
    )
    cdd_vmin, cdd_vmax = compute_vmin_vmax(
        combined_ens_mean, ["CDD"], quantiles=(0.05, 0.95), center_zero=True
    )
    _auto_vmin = {
        "pv": pv_wind_vmin,
        "wind": pv_wind_vmin,
        "hydropower_ror": hydropower_ror_vmin,
        "hydropower_reservoir": hydropower_reservoir_vmin,
        "HDD": hdd_vmin,
        "CDD": cdd_vmin,
    }
    _auto_vmax = {
        "pv": pv_wind_vmax,
        "wind": pv_wind_vmax,
        "hydropower_ror": hydropower_ror_vmax,
        "hydropower_reservoir": hydropower_reservoir_vmax,
        "HDD": hdd_vmax,
        "CDD": cdd_vmax,
    }

    print(_auto_vmin)
    print(_auto_vmax)

    # ── Draw maps ─────────────────────────────────────────────────────────────
    def _draw_map(ax, var, season, vmin_val, vmax_val, cmap):
        """Draw one change map panel and return the image handle."""
        ens_mean_s = all_ens_means[season]
        agree_mask = all_agree_masks[var][season]

        im = ens_mean_s[var].plot(
            ax=ax,
            cmap=cmap,
            vmin=vmin_val,
            vmax=vmax_val,
            transform=ccrs.PlateCarree(),
            add_colorbar=False,
        )
        hatch_overlay = ax.contourf(
            agree_mask["lon"],
            agree_mask["lat"],
            agree_mask,
            levels=[0.5, 1.5],
            hatches=["///"],
            colors="none",
            transform=ccrs.PlateCarree(),
        )
        hatch_overlay.set_facecolor("none")
        hatch_overlay.set_edgecolor("black")
        hatch_overlay.set_linewidth(0)
        ax.coastlines(linewidth=0.5)
        ax.add_feature(
            cfeature.BORDERS,
            edgecolor="black",
            linewidth=country_border_width,
            zorder=6,
        )
        ax.set_xlabel("")
        ax.set_ylabel("")
        return im

    for var_idx, var in enumerate(var_names):
        if vmin is not None:
            vmin_val = vmin[var] if isinstance(vmin, dict) else vmin
        else:
            auto_v = _auto_vmin.get(var)
            if auto_v is not None:
                vmin_val = auto_v
            else:
                _all_vals = xr.concat(
                    [all_ens_means[s][var] for s in seasons], dim="season"
                )
                vmin_val = -float(np.nanquantile(np.abs(_all_vals.values), 0.99))

        if vmax is not None:
            vmax_val = vmax[var] if isinstance(vmax, dict) else vmax
        else:
            auto_v = _auto_vmax.get(var)
            if auto_v is not None:
                vmax_val = auto_v
            else:
                _all_vals = xr.concat(
                    [all_ens_means[s][var] for s in seasons], dim="season"
                )
                vmax_val = float(np.nanquantile(np.abs(_all_vals.values), 0.99))

        cmap = "seismic_r" if var.startswith("hydropower") else "RdBu"

        last_im = None

        if not transpose:
            # rows = variables, columns = seasons
            for col_idx, season in enumerate(seasons):
                ax = map_axes[var_idx, col_idx]
                last_im = _draw_map(ax, var, season, vmin_val, vmax_val, cmap)
                ax.set_title(
                    season_labels.get(season, str(season)),
                    fontsize=fontsize_labels,
                    pad=4,
                )

            # Vertical colorbar on the right
            cbar = fig.colorbar(last_im, cax=cbar_axes[var_idx], orientation="vertical")
            cbar.set_label(var_clabel.get(var, ""), fontsize=fontsize_labels - 1)
            cbar.ax.tick_params(labelsize=fontsize_ticks)
            if simplify_colorbar:
                cbar.set_ticks([vmin_val, vmax_val])
                cbar.set_ticklabels(["Less", "More"])
                cbar.ax.tick_params(length=0)

            # Variable row label on the left
            map_axes[var_idx, 0].text(
                -0.08,
                0.5,
                var_titles.get(var, var),
                transform=map_axes[var_idx, 0].transAxes,
                rotation=90,
                va="center",
                ha="right",
                fontsize=fontsize_labels,
                fontweight="bold",
                clip_on=False,
            )

        else:
            # rows = seasons, columns = variables
            for row_idx, season in enumerate(seasons):
                ax = map_axes[row_idx, var_idx]
                last_im = _draw_map(ax, var, season, vmin_val, vmax_val, cmap)

                # Variable title: top row only; clear xarray auto-title for others
                if row_idx == 0:
                    ax.set_title(
                        var_titles.get(var, var),
                        fontsize=fontsize_labels,
                        pad=4,
                    )
                else:
                    ax.set_title("")

                # Season label: left of first column only
                if var_idx == 0:
                    ax.text(
                        -0.08,
                        0.5,
                        season_labels.get(season, str(season)),
                        transform=ax.transAxes,
                        rotation=90,
                        va="center",
                        ha="right",
                        fontsize=fontsize_labels,
                        fontweight="bold",
                        clip_on=False,
                    )

            # Horizontal colorbar at the bottom of each variable column
            cbar = fig.colorbar(
                last_im, cax=cbar_axes[var_idx], orientation="horizontal"
            )
            cbar.set_label(var_clabel.get(var, ""), fontsize=fontsize_labels - 1)
            cbar.ax.tick_params(labelsize=fontsize_ticks)
            if simplify_colorbar:
                cbar.set_ticks([vmin_val, vmax_val])
                cbar.set_ticklabels(["Less", "More"])
                cbar.ax.tick_params(length=0)

    if title is not None:
        fig.suptitle(
            title,
            fontsize=fontsize_title,
            y=components_size_config.get("SUPTITLE_Y", 0.97),
        )
    if path_savefig is not None:
        path_savefig.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path_savefig, dpi=300, bbox_inches="tight")
    plt.show()


def plot_ensemble_mean_change(
    ref_dataset: xr.Dataset,
    target_dataset: xr.Dataset,
    var_titles: dict = None,
    var_clabel: dict = None,
    components_size_config: dict = None,
    absolute_change: bool = False,
    simplify_colorbar: bool = False,
    mask_insignificant: bool = False,
    significance_threshold: dict[str, float] | float = None,
    figsize: tuple = None,
    country_border_width: float = 0.5,
    orientation: str = "horizontal",
    title: str = "Ensemble mean percentage change",
    cmap: str = "RdBu",
    path_savefig: Path = None,
):
    """
    Plot ensemble mean percentage change for each dataset (variable), hatching where less than 2/3 of models agree on sign.

    Parameters
    ----------
    ref_dataset : xr.Dataset
        Dataset of variables averaged over the reference period, containing the following coordinates
        'lat', 'lon', and 'model'.
    target_dataset : xr.Dataset
        Dataset of variables averaged over the target period, containing the following coordinates
        'lat', 'lon', and 'model'.
    var_titles : dict, optional
        Dictionary mapping variable names to plot titles. If None, uses default titles.
    var_clabel : dict, optional
        Dictionary mapping variable names to colorbar labels. If None, uses default labels
        ("% relative to reference" for percentage change, "Change relative to reference" for absolute change).
    components_size_config : dict, optional
        Dictionary with size configuration for plot components (fontsizes, colorbar params).
        If None, uses default sizes.
    absolute_change : bool, optional
        If True, compute absolute change instead of percentage change.
    simplify_colorbar : bool, optional
        If True, replace colorbar ticks with "Less" and "More" labels (default: False).
    mask_insignificant : bool, optional
        If True, mask out percentage changes where absolute change is below significance_threshold.
        This helps avoid showing misleadingly high percentages from small absolute changes (default: False).
    significance_threshold : dict or float, optional
        Threshold for absolute change to be considered significant. Can be:
        - dict mapping variable names to thresholds: {"pv": 0.01, "wind": 0.01}
        - single float value applied to all variables
        Only used when mask_insignificant is True. If None, defaults to 0.01 for all variables.
    figsize : tuple, optional
        Figure size. If None, automatically sized based on number of variables.
    country_border_width : float, optional
        Width of country borders (default: 0.5).
    orientation : str, optional
        Orientation of subplots: "horizontal" (nrows=1) or "vertical" (ncols=1). Default: "horizontal".
    title : str, optional
        Plot title (default: "Ensemble mean percentage change").
    cmap : str, optional
        Colormap (default: "RdBu_r").
    path_savefig : pathlib.Path, optional
        Path to save the figure.

    Returns
    -------
    None
    """
    if var_titles is None:
        var_titles = {
            "pv": "Solar Power Potential",
            "wind": "Wind Power Potential",
            "hydropower_ror": "Run-of-river Hydropower Potential",
            "hydropower_reservoir": "Reservoir Hydropower Inflows",
            "HDD": "Heating Degree Days",
            "CDD": "Cooling Degree Days",
        }

    if components_size_config is None:
        components_size_config = COMPONENTS_SIZE_CONFIG_DEFAULT

    # Process significance threshold
    if mask_insignificant and significance_threshold is None:
        # Default thresholds
        significance_threshold = {var: 0.01 for var in ref_dataset.data_vars.keys()}
    elif mask_insignificant and isinstance(significance_threshold, (int, float)):
        # Convert single value to dict
        significance_threshold = {
            var: significance_threshold for var in ref_dataset.data_vars.keys()
        }

    var_names = list(ref_dataset.data_vars.keys())
    n_vars = len(var_names)

    if figsize is None:
        if orientation == "horizontal":
            figsize = (4 * n_vars, 6)
        else:
            figsize = (6, 4 * n_vars)
    figsize = parse_figsize(figsize)

    # Extract configuration values
    fontsize_legend = components_size_config.get("FONTSIZE_LEGEND", FONTSIZE_LEGEND)
    fontsize_ticks = components_size_config.get("FONTSIZE_TICKS", FONTSIZE_TICKS)
    fontsize_labels = components_size_config.get("FONTSIZE_LABELS", FONTSIZE_LABELS)
    fontsize_title = components_size_config.get("FONTSIZE_TITLE", FONTSIZE_TITLE)
    cbar_shrink = components_size_config.get("CBAR_SHRINK", 0.7)
    cbar_aspect = components_size_config.get("CBAR_ASPECT", 20)
    cbar_pad = components_size_config.get("CBAR_PAD", 0.03)

    min_lat, max_lat = ref_dataset.lat.min().item(), ref_dataset.lat.max().item()
    central_lon = (ref_dataset.lon.min().item() + ref_dataset.lon.max().item()) / 2

    if orientation == "horizontal":
        nrows, ncols = 1, n_vars
    else:
        nrows, ncols = n_vars, 1

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        subplot_kw={
            "projection": ccrs.Mercator(
                central_longitude=central_lon,
                min_latitude=min_lat,
                max_latitude=max_lat,
            )
        },
    )

    if all(ref_dataset.model.values != target_dataset.model.values):
        raise ValueError(
            "Reference and target datasets must have the same 'model' coordinate."
        )

    if n_vars == 1:
        axes = [axes]

    # Compute percentage change for each model
    if absolute_change:
        change = target_dataset - ref_dataset
    else:
        change = compute_percentage_change(ref_dataset, target_dataset, round=2)
    # Ensemble mean
    ens_mean = change.mean(dim="model")

    # Agreement mask: sign agreement across models
    sign_agree = compute_sign_agreement(change, model_dim="model")

    # Compute agreement mask with variable-specific model counts
    agree_mask = {}
    for var in var_names:
        # Count non-NaN models for each grid point
        n_models_per_point = (~np.isnan(change[var])).sum(dim="model")
        # Threshold is 2/3 of available models at each point
        threshold = 2 * n_models_per_point / 3
        # Mask where agreement is less than threshold
        agree_mask[var] = ~(sign_agree[var] >= threshold)
        # Place NaNs where original data is NaN
        agree_mask[var] = agree_mask[var].where(~np.isnan(ens_mean[var]))

    agree_mask = xr.Dataset(agree_mask)

    # Determine common vmin/vmax for PV/Wind and HDD/CDD groups
    pv_wind_vmin, pv_wind_vmax = compute_vmin_vmax(
        ens_mean, ["pv", "wind"], quantiles=None, center_zero=True
    )
    hydropower_ror_vmin, hydropower_ror_vmax = compute_vmin_vmax(
        ens_mean, ["hydropower_ror"], quantiles=(0.01, 0.99), center_zero=True
    )
    if absolute_change:
        hydropower_ror_vmin, hydropower_ror_vmax = pv_wind_vmin, pv_wind_vmax
    hydropower_reservoir_vmin, hydropower_reservoir_vmax = compute_vmin_vmax(
        ens_mean, ["hydropower_reservoir"], quantiles=(0.01, 0.99), center_zero=True
    )
    hdd_vmin, hdd_vmax = compute_vmin_vmax(
        ens_mean, ["HDD"], quantiles=(0.05, 0.95), center_zero=True
    )
    cdd_vmin, cdd_vmax = compute_vmin_vmax(
        ens_mean, ["CDD"], quantiles=(0.05, 0.95), center_zero=True
    )

    # Mask insignificant changes if requested
    if mask_insignificant:
        abs_change = target_dataset - ref_dataset
        for var in var_names:
            threshold = significance_threshold.get(var, 0.01)
            # Create mask where absolute change is below threshold
            insignificant_mask = np.abs(abs_change[var].mean(dim="model")) < threshold
            # Apply mask to percentage change
            change[var] = change[var].where(~insignificant_mask)
        # Ensemble mean
        ens_mean = change.mean(dim="model")

    for i, var in enumerate(var_names):
        ax = axes[i]
        im = ens_mean[var].plot(
            ax=ax,
            cmap=cmap if not var.startswith("hydropower") else "seismic_r",
            vmin=pv_wind_vmin
            if var in ["pv", "wind"]
            else (
                hydropower_ror_vmin
                if var == "hydropower_ror"
                else (
                    hydropower_reservoir_vmin
                    if var == "hydropower_reservoir"
                    else (hdd_vmin if var == "HDD" else cdd_vmin)
                )
            ),
            vmax=pv_wind_vmax
            if var in ["pv", "wind"]
            else (
                hydropower_ror_vmax
                if var == "hydropower_ror"
                else (
                    hydropower_reservoir_vmax
                    if var == "hydropower_reservoir"
                    else (hdd_vmax if var == "HDD" else cdd_vmax)
                )
            ),
            transform=ccrs.PlateCarree(),
            add_colorbar=True,
            cbar_kwargs={
                "shrink": cbar_shrink,
                "aspect": cbar_aspect,
                "pad": cbar_pad,
                "orientation": "horizontal",
            },
        )
        cbar = im.colorbar
        # Set colorbar label from var_clabel dict or use default
        if var_clabel and var in var_clabel:
            label = var_clabel[var]
        else:
            label = (
                "% relative to reference"
                if not absolute_change
                else "Change relative to reference"
            )
        cbar.set_label(label, fontsize=fontsize_labels - 1)
        cbar.ax.tick_params(labelsize=fontsize_ticks)

        # Simplify colorbar if requested
        if simplify_colorbar:
            cbar.set_ticks([cbar.vmin, cbar.vmax])
            cbar.set_ticklabels(["Less", "More"])
            cbar.ax.tick_params(length=0)

        # Add hatching where less than 2/3 agree using contourf
        hatch_overlay = ax.contourf(
            agree_mask[var]["lon"],
            agree_mask[var]["lat"],
            agree_mask[var],
            levels=[0.5, 1.5],  # one filled band where mask == 1
            hatches=["///"],
            colors="none",
            transform=ccrs.PlateCarree(),
        )

        hatch_overlay.set_facecolor("none")  # keep fill transparent
        # hatch_overlay.set_edgecolor((0, 0, 0, 0.8))  # RGBA, alpha=0.5
        hatch_overlay.set_edgecolor("black")  # hatches black
        hatch_overlay.set_linewidth(0)  # polygon outlines invisible
        ax.coastlines()
        ax.add_feature(
            cfeature.BORDERS, edgecolor="black", linewidth=country_border_width
        )
        ax.set_title(var_titles[var], fontsize=fontsize_labels)
        ax.set_xlabel("")
        ax.set_ylabel("")

    if title is not None:
        plt.suptitle(title, fontsize=fontsize_title, y=0.9)
    plt.tight_layout()
    if path_savefig is not None:
        path_savefig.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path_savefig, dpi=300, bbox_inches="tight")
    plt.show()


def plot_yearly_timeseries_ensemble_mean_range(
    ds: xr.Dataset,
    selected_regions: list[str],
    var_titles: dict = None,
    var_ylabel: dict = None,
    components_size_config: dict = None,
    time_dim: str = "time",
    model_dim: str = "model",
    region_dim: str = "mask",
    rolling: int = 5,
    figsize: tuple = None,
    color_palette: list = None,
    simplify_yaxis: bool = False,
    orientation: str = "horizontal",
    title: str = "Regional mean (rolling) timeseries",
    path_savefig: Path = None,
):
    """
    Plot ensemble mean and range (min/max) timeseries for specified regions.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing the data variable and region mask.
    selected_regions : list of str
        List of region names to plot (must match region_dim values).
    var_titles : dict, optional
        Dictionary mapping variable names to plot titles. If None, uses default titles.
    var_ylabel : dict, optional
        Dictionary mapping variable names to y-axis labels. If None, uses default labels.
    components_size_config : dict, optional
        Dictionary with size configuration for plot components (fontsizes).
        If None, uses default sizes.
    time_dim : str, optional
        Name of the time dimension.
    model_dim : str, optional
        Name of the model dimension.
    region_dim : str, optional
        Name of the region mask dimension.
    rolling : int, optional
        Window size for rolling mean.
    figsize : tuple, optional
        Figure size. If None, automatically sized based on number of variables.
    color_palette : list, optional
        List of colors to use for different regions. If None, uses default colorblind palette.
    simplify_yaxis : bool, optional
        If True, replace y-axis ticks with "Less" and "More" labels (default: False).
    orientation : str, optional
        Orientation of subplots: "horizontal" (nrows=1) or "vertical" (ncols=1). Default: "horizontal".
    title : str, optional
        Plot title.
    path_savefig : pathlib.Path, optional
        Path to save the figure.

    Returns
    -------
    None
    """
    if var_titles is None:
        var_titles = {
            "pv": "Solar Power Potential",
            "wind": "Wind Power Potential",
            "hydropower_ror": "Run-of-river Hydropower Potential",
            "hydropower_reservoir": "Reservoir Hydropower Inflows",
            "HDD": "Heating Degree Days",
            "CDD": "Cooling Degree Days",
        }
    if var_ylabel is None:
        var_ylabel = {
            "pv": "Capacity Factor (-)",
            "wind": "Capacity Factor (-)",
            "hydropower_ror": "Generation (MWh)",
            "hydropower_reservoir": "Weekly Inflows (m³/s)",
            "HDD": "Degree-Days",
            "CDD": "Degree-Days",
        }
    if components_size_config is None:
        components_size_config = COMPONENTS_SIZE_CONFIG_DEFAULT
    if color_palette is None:
        color_palette = COLORBLIND_PALETTE

    var_names = list(ds.data_vars.keys())
    n_vars = len(var_names)

    if figsize is None:
        if orientation == "horizontal":
            figsize = (4 * n_vars, 6)
        else:
            figsize = (6, 4 * n_vars)
    figsize = parse_figsize(figsize)

    # Extract configuration values
    fontsize_legend = components_size_config.get("FONTSIZE_LEGEND", FONTSIZE_LEGEND)
    fontsize_ticks = components_size_config.get("FONTSIZE_TICKS", FONTSIZE_TICKS)
    fontsize_labels = components_size_config.get("FONTSIZE_LABELS", FONTSIZE_LABELS)
    fontsize_title = components_size_config.get("FONTSIZE_TITLE", FONTSIZE_TITLE)
    bbox_to_anchor = components_size_config.get("LEGEND_BBOX_TO_ANCHOR", (0.5, -0.35))
    hspace = components_size_config.get("HSPACE", 0.45)
    wspace = components_size_config.get("WSPACE", 0.3)
    xtick_years = components_size_config.get("XTICK_YEARS", None)
    highlight_linewidth = components_size_config.get("HIGHLIGHT_LINEWIDTH", 1)
    other_models_linewidth = components_size_config.get("OTHER_MODELS_LINEWIDTH", 1)
    other_models_alpha = components_size_config.get("OTHER_MODELS_ALPHA", 0.45)

    if orientation == "horizontal":
        nrows, ncols = 1, n_vars
    else:
        nrows, ncols = n_vars, 1

    # Add spacing when orientation is vertical/horizontal to align with ensemble mean plots
    gridspec_kw = (
        {"hspace": hspace} if orientation == "vertical" else {"wspace": wspace}
    )

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        gridspec_kw=gridspec_kw,
        constrained_layout=True,
    )

    if n_vars == 1:
        axes = [axes]

    ds_region = ds.sel({region_dim: selected_regions})
    if rolling is not None:
        # Apply rolling mean if specified
        ds_region = ds_region.rolling({time_dim: rolling}, center=True).mean()
    mean_ds_region = ds_region.mean(dim=model_dim)
    min_ds_region = ds_region.min(dim=model_dim)
    max_ds_region = ds_region.max(dim=model_dim)
    wind_solar_vmin, wind_solar_vmax = compute_vmin_vmax(
        ds_region, ["pv", "wind"], quantiles=None, center_zero=False
    )

    for i, var in enumerate(var_names):
        ax = axes[i] if n_vars > 1 else axes
        for j, region in enumerate(selected_regions):
            mean_da = mean_ds_region[var].sel({region_dim: region})
            min_da = min_ds_region[var].sel({region_dim: region})
            max_da = max_ds_region[var].sel({region_dim: region})
            color = color_palette[j % len(color_palette)]
            ax.plot(mean_da[time_dim], mean_da, label=f"{region} (mean)", color=color)
            ax.fill_between(
                mean_da[time_dim].values,
                min_da.values,
                max_da.values,
                alpha=0.3,
                color=color,
                label=f"{region} (range)",
            )
            if var in ["pv", "wind"]:
                ax.set_ylim(wind_solar_vmin - 0.02, wind_solar_vmax + 0.02)

        ax.set_ylabel(var_ylabel.get(var, ""), fontsize=fontsize_labels)
        ax.set_title(f"{var_titles.get(var, var)}", fontsize=fontsize_labels)
        ax.tick_params(axis="both", labelsize=fontsize_ticks)

        # Simplify y-axis if requested
        if simplify_yaxis:
            ax.set_yticks([])

        # Set specific years for x-axis ticks
        if xtick_years is not None:
            import pandas as pd
            from matplotlib.dates import DateFormatter

            # Convert years to datetime objects for positioning
            tick_dates = [pd.Timestamp(f"{year}-01-01") for year in xtick_years]
            ax.set_xticks(tick_dates)
            ax.xaxis.set_major_formatter(DateFormatter("%Y"))

    handles, labels = ax.get_legend_handles_labels()
    fig = ax.get_figure()
    # Adjust legend position when vertical to compensate for increased subplot spacing
    legend_bbox = (
        bbox_to_anchor
        if orientation == "horizontal"
        else (bbox_to_anchor[0], bbox_to_anchor[1] + 0.15)
    )
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(selected_regions) + 1,
        bbox_to_anchor=legend_bbox,
        fontsize=fontsize_legend,
    )
    if title is not None:
        plt.suptitle(title, fontsize=fontsize_title)
    plt.tight_layout()
    if path_savefig is not None:
        path_savefig.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path_savefig, dpi=300, bbox_inches="tight")
    plt.show()


def plot_yearly_timeseries_ensemble(
    ds: xr.Dataset,
    selected_regions: list[str],
    var_titles: dict = None,
    var_ylabel: dict = None,
    components_size_config: dict = None,
    time_dim: str = "time",
    model_dim: str = "model",
    region_dim: str = "mask",
    rolling: int = 5,
    figsize: tuple = None,
    color_palette: list = None,
    simplify_yaxis: bool = False,
    plot_mean: bool = False,
    highlight_models: list[str] = None,
    highlight_colors: dict[str, str] | list[str] = None,
    highlight_labels: dict[str, str] | list[str] = None,
    orientation: str = "horizontal",
    title: str = "Regional mean (rolling) timeseries",
    path_savefig: Path = None,
):
    """
    Plot ensemble timeseries for specified regions, with options to plot mean and highlight
    specific models.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing the data variable and region mask.
    selected_regions : list of str
        List of region names to plot (must match region_dim values).
    var_titles : dict, optional
        Dictionary mapping variable names to plot titles. If None, uses default titles.
    var_ylabel : dict, optional
        Dictionary mapping variable names to y-axis labels. If None, uses default labels.
    components_size_config : dict, optional
        Dictionary with size configuration for plot components (fontsizes).
        If None, uses default sizes.
    time_dim : str, optional
        Name of the time dimension.
    model_dim : str, optional
        Name of the model dimension.
    region_dim : str, optional
        Name of the region mask dimension.
    rolling : int, optional
        Window size for rolling mean.
    figsize : tuple, optional
        Figure size. If None, automatically sized based on number of variables.
    color_palette : list, optional
        List of colors to use for different regions. If None, uses default colorblind palette.
    simplify_yaxis : bool, optional
        If True, replace y-axis ticks with "Less" and "More" labels (default: False).
    plot_mean : bool, optional
        If True, plot the ensemble mean with thicker lines (linewidth 3). Default is False.
    highlight_models : list of str, optional
        List of model names to highlight with bold lines (linewidth 3). If None, no models are highlighted.
    highlight_colors : dict or list, optional
        Colors for highlighted models. Can be:
        - dict mapping model names to colors: {"model1": "red", "model2": "blue"}
        - list of colors in same order as highlight_models
        If None, cycles through colorblind palette starting from index 3.
    highlight_labels : dict or list, optional
        Custom labels for highlighted models. Can be:
        - dict mapping model names to labels: {"model1": "Custom Label 1", "model2": "Custom Label 2"}
        - list of labels in same order as highlight_models
        If None, uses the model names directly.
    orientation : str, optional
        Orientation of subplots: "horizontal" (nrows=1) or "vertical" (ncols=1). Default: "horizontal".
    title : str, optional
        Plot title.
    path_savefig : pathlib.Path, optional
        Path to save the figure.

    Returns
    -------
    None
    """
    if var_titles is None:
        var_titles = {
            "pv": "Solar Power Potential",
            "wind": "Wind Power Potential",
            "hydropower_ror": "Run-of-river Hydropower Potential",
            "hydropower_reservoir": "Reservoir Hydropower Inflows",
            "HDD": "Heating Degree Days",
            "CDD": "Cooling Degree Days",
        }
    if var_ylabel is None:
        var_ylabel = {
            "pv": "Capacity Factor (-)",
            "wind": "Capacity Factor (-)",
            "hydropower_ror": "Generation (MWh)",
            "hydropower_reservoir": "Weekly Inflows (m³/s)",
            "HDD": "Degree-Days",
            "CDD": "Degree-Days",
        }
    if components_size_config is None:
        components_size_config = COMPONENTS_SIZE_CONFIG_DEFAULT
    if color_palette is None:
        color_palette = COLORBLIND_PALETTE

    # Process highlight_models and highlight_colors
    if highlight_models is None:
        highlight_models = []
    if highlight_colors is None:
        # Default colors cycling through palette
        highlight_colors = {
            model: COLORBLIND_PALETTE[(i + 3) % len(COLORBLIND_PALETTE)]
            for i, model in enumerate(highlight_models)
        }
    elif isinstance(highlight_colors, list):
        # Convert list to dict
        highlight_colors = {
            model: color for model, color in zip(highlight_models, highlight_colors)
        }

    # Process highlight_labels
    if highlight_labels is None:
        # Use model names as default labels
        highlight_labels = {model: model for model in highlight_models}
    elif isinstance(highlight_labels, list):
        # Convert list to dict
        highlight_labels = {
            model: label for model, label in zip(highlight_models, highlight_labels)
        }

    var_names = list(ds.data_vars.keys())
    n_vars = len(var_names)

    if figsize is None:
        if orientation == "horizontal":
            figsize = (4 * n_vars, 6)
        else:
            figsize = (6, 4 * n_vars)
    figsize = parse_figsize(figsize)

    # Extract configuration values
    fontsize_legend = components_size_config.get("FONTSIZE_LEGEND", FONTSIZE_LEGEND)
    fontsize_ticks = components_size_config.get("FONTSIZE_TICKS", FONTSIZE_TICKS)
    fontsize_labels = components_size_config.get("FONTSIZE_LABELS", FONTSIZE_LABELS)
    fontsize_title = components_size_config.get("FONTSIZE_TITLE", FONTSIZE_TITLE)
    bbox_to_anchor = components_size_config.get("LEGEND_BBOX_TO_ANCHOR", (0.5, -0.35))
    hspace = components_size_config.get("HSPACE", 0.45)
    wspace = components_size_config.get("WSPACE", 0.3)
    xtick_years = components_size_config.get("XTICK_YEARS", None)
    highlight_linewidth = components_size_config.get("HIGHLIGHT_LINEWIDTH", 1)
    other_models_linewidth = components_size_config.get("OTHER_MODELS_LINEWIDTH", 1)
    other_models_alpha = components_size_config.get("OTHER_MODELS_ALPHA", 0.45)

    if orientation == "horizontal":
        nrows, ncols = 1, n_vars
    else:
        nrows, ncols = n_vars, 1

    # Add spacing when orientation is vertical/horizontal to align with ensemble mean plots
    gridspec_kw = (
        {"hspace": hspace} if orientation == "vertical" else {"wspace": wspace}
    )

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        gridspec_kw=gridspec_kw,
        constrained_layout=True,
    )

    if n_vars == 1:
        axes = [axes]

    ds_region = ds.sel({region_dim: selected_regions})
    if rolling is not None:
        # Apply rolling mean if specified
        ds_region = ds_region.rolling({time_dim: rolling}, center=True).mean()

    mean_ds_region = ds_region.mean(dim=model_dim)
    wind_solar_vmin, wind_solar_vmax = compute_vmin_vmax(
        ds_region, ["pv", "wind", "hydropower_ror"], quantiles=None, center_zero=False
    )

    for i, var in enumerate(var_names):
        ax = axes[i] if n_vars > 1 else axes
        for j, region in enumerate(selected_regions):
            if plot_mean:
                mean_da = mean_ds_region[var].sel({region_dim: region})
                ax.plot(
                    mean_da[time_dim],
                    mean_da,
                    label=f"{region} (mean)" if len(selected_regions) > 1 else "Mean",
                    color=color_palette[j % len(color_palette)],
                    linewidth=2,
                )
            for model in ds_region[model_dim].values:
                da = ds_region[var].sel({region_dim: region, model_dim: model})

                # Determine line properties based on whether this model should be highlighted
                if model in highlight_models:
                    line_color = highlight_colors.get(
                        model, color_palette[j % len(color_palette)]
                    )
                    line_width = highlight_linewidth
                    line_alpha = 1.0
                else:
                    line_color = color_palette[j % len(color_palette)]
                    line_width = other_models_linewidth
                    line_alpha = other_models_alpha

                ax.plot(
                    da[time_dim],
                    da,
                    label=f"{region} - {model}",
                    alpha=line_alpha,
                    color=line_color,
                    linewidth=line_width,
                )

            if var in ["pv", "wind", "hydropower_ror"]:
                offset = max((wind_solar_vmin / 10), (wind_solar_vmax / 10))
                ax.set_ylim(wind_solar_vmin - offset, wind_solar_vmax + offset)

        ax.set_ylabel(var_ylabel.get(var, ""), fontsize=fontsize_labels - 2)
        ax.set_title(f"{var_titles.get(var, var)}", fontsize=fontsize_labels)
        ax.tick_params(axis="both", labelsize=fontsize_ticks)

        # Simplify y-axis if requested
        if simplify_yaxis:
            ax.set_yticks([])

        # Set specific years for x-axis ticks
        if xtick_years is not None:
            import pandas as pd
            from matplotlib.dates import DateFormatter

            # Convert years to datetime objects for positioning
            tick_dates = [pd.Timestamp(f"{year}-01-01") for year in xtick_years]
            ax.set_xticks(tick_dates)
            ax.xaxis.set_major_formatter(DateFormatter("%Y"))

    # Create custom legend with one entry per region (mean) and one for individual models
    custom_handles = []
    custom_labels = []
    for j, region in enumerate(selected_regions):
        color = color_palette[j % len(color_palette)]
        if plot_mean:
            # Create a line handle with the region's color
            line_handle = plt.Line2D(
                [0],
                [0],
                color=color,
                linewidth=2,
                label=f"{region} (mean)" if len(selected_regions) > 1 else "Mean",
            )
            custom_handles.append(line_handle)
            custom_labels.append(
                f"{region} (mean)" if len(selected_regions) > 1 else "Mean"
            )

        # Add entries for highlighted models if any
        if highlight_models:
            for model in highlight_models:
                model_color = highlight_colors.get(model, color)
                model_label = highlight_labels.get(model, model)
                highlighted_handle = plt.Line2D(
                    [0],
                    [0],
                    color=model_color,
                    linewidth=highlight_linewidth,
                    label=f"{region} - {model_label}"
                    if len(selected_regions) > 1
                    else model_label,
                )
                custom_handles.append(highlighted_handle)
                custom_labels.append(
                    f"{region} - {model_label}"
                    if len(selected_regions) > 1
                    else model_label
                )

        # Add a single entry for other models
        model_handle = plt.Line2D(
            [0],
            [0],
            color=color,
            linewidth=other_models_linewidth,
            alpha=other_models_alpha,
            label=f"{region} (other models)"
            if len(selected_regions) > 1
            else "Other models",
        )
        custom_handles.append(model_handle)
        custom_labels.append(
            f"{region} (other models)" if len(selected_regions) > 1 else "Other models"
        )

    fig = ax.get_figure()
    # ncol_legend = len(custom_labels) if not plot_mean and highlight_models else len(selected_regions) + len(highlight_models)
    # ncol_legend = len(selected_regions) if orientation == "horizontal" else len(custom_labels)
    ncol_legend = len(custom_labels)
    # Adjust legend position when vertical to compensate for increased subplot spacing
    legend_bbox = (
        bbox_to_anchor
        if orientation == "horizontal"
        else (bbox_to_anchor[0], bbox_to_anchor[1] + 0.15)
    )
    fig.legend(
        custom_handles,
        custom_labels,
        loc="lower center",
        ncol=ncol_legend,
        bbox_to_anchor=legend_bbox,
        fontsize=fontsize_legend,
    )
    if title is not None:
        plt.suptitle(title, fontsize=fontsize_title)
    plt.tight_layout()
    if path_savefig is not None:
        path_savefig.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path_savefig, dpi=300, bbox_inches="tight")
    plt.show()


def plot_interval_strip_regions(
    ds: xr.Dataset,
    selected_regions: list[str],
    time_dim: str = "time",
    model_dim: str = "model",
    region_dim: str = "mask",
    window_length: int = 20,
    figsize=(10, 5),
    title: str = "Gradient",
    path_savefig: Path = None,
):
    return


def plot_percentage_change_spread(
    ref_dataset: xr.Dataset,
    target_dataset: xr.Dataset,
    variables: list[str],
    valid_models: list[str],
    selected_model: str = None,
    color: str = None,
    components_size_config: dict = None,
    bar_alpha: float = 0.5,
    ax: Optional[plt.Axes] = None,
    figsize: tuple = None,
    title: str = None,
    show_plot: bool = True,
    path_savefig: Path = None,
):
    """
    Plot horizontal spread bars showing percentage change range across models.

    Parameters
    ----------
    ref_dataset : xr.Dataset
        Dataset of variables averaged over the reference period, containing the following coordinates
        'lat', 'lon', and 'model'.
    target_dataset : xr.Dataset
        Dataset of variables averaged over the target period, containing the following coordinates
        'lat', 'lon', and 'model'.
    variables : list of str
        List of variable names to plot.
    valid_models : list of str
        List of valid model names to include in the calculation.
    selected_model : str, optional
        Model name to highlight with a marker. If None, uses ensemble mean.
    color : str, optional
        Color for bars and markers. If None, uses default colorblind palette.
    components_size_config : dict, optional
        Dictionary with size configuration for plot components (e.g., marker size). If None, uses default sizes.
    bar_alpha : float, optional
        Transparency of the spread bars (default: 0.5).
    figsize : tuple, optional
        Figure size. If None, automatically sized based on number of variables.
    title : str, optional
        Plot title.
    path_savefig : pathlib.Path, optional
        Path to save the figure.

    Returns
    -------
    None
    """
    if color is None:
        color = BLUE
    if components_size_config is None:
        components_size_config = COMPONENTS_SIZE_CONFIG_DEFAULT

    components_size_config["MARKER_SIZE"] = components_size_config.get("MARKER_SIZE", 8)
    components_size_config["BAR_HEIGHT"] = components_size_config.get("BAR_HEIGHT", 0.3)
    components_size_config["BBOX_TO_ANCHOR"] = components_size_config.get(
        "BBOX_TO_ANCHOR", (0.5, -0.5)
    )

    var_titles = {
        "pv": "Solar Power",
        "wind": "Wind Power",
        "hydropower": "Water Availability",
        "HDD": "Heating Degree Days",
        "CDD": "Cooling Degree Days",
    }

    n_vars = len(variables)
    if figsize is None:
        figsize = (8, max(3, n_vars * 1.5))
    if ax is None:
        figsize = parse_figsize(figsize)
        _, ax = plt.subplots(n_vars, 1, figsize=figsize, sharex=True)
        show_plot = True
    else:
        show_plot = False

    if n_vars == 1:
        ax = [ax]

    # Compute all percentage changes first to determine centered xlim
    all_pct_changes = []
    for var in variables:
        for model in valid_models:
            ref_val = ref_dataset[var].sel(model=model).mean(dim=["lat", "lon"]).values
            mid_val = (
                target_dataset[var].sel(model=model).mean(dim=["lat", "lon"]).values
            )
            if not (np.isnan(ref_val) or np.isnan(mid_val)):
                pct = (mid_val - ref_val) / ref_val * 100
                all_pct_changes.append(pct)

    max_abs = np.max(np.abs(all_pct_changes)) if all_pct_changes else 10

    for i, var in enumerate(variables):
        ax_sel = ax[i]

        # Compute percentage change for each model
        pct_change = []
        for model in valid_models:
            ref_val = ref_dataset[var].sel(model=model).mean(dim=["lat", "lon"]).values
            mid_val = (
                target_dataset[var].sel(model=model).mean(dim=["lat", "lon"]).values
            )
            if np.isnan(ref_val) or np.isnan(mid_val):
                continue
            pct = (mid_val - ref_val) / ref_val * 100
            pct_change.append(pct)

        if not pct_change:
            continue

        # Compute min, max, and mean/selected
        pct_min = np.min(pct_change)
        pct_max = np.max(pct_change)

        if selected_model and selected_model in valid_models:
            # Use selected model value
            ref_val = (
                ref_dataset[var]
                .sel(model=selected_model)
                .mean(dim=["lat", "lon"])
                .values
            )
            mid_val = (
                target_dataset[var]
                .sel(model=selected_model)
                .mean(dim=["lat", "lon"])
                .values
            )
            pct_mean = (mid_val - ref_val) / ref_val * 100
            label = selected_model
        else:
            pct_mean = np.mean(pct_change)
            label = "Mean"

        # Plot horizontal spread bar from min to max
        ax_sel.barh(
            0,
            pct_max - pct_min,
            left=pct_min,
            height=components_size_config["BAR_HEIGHT"],
            alpha=bar_alpha,
            color=color,
            align="center",
        )
        ax_sel.plot(
            pct_mean,
            0,
            "o",
            color=color,
            alpha=1,
            markersize=components_size_config["MARKER_SIZE"],
            label=label,
        )

        ax_sel.set_xlim(-max_abs * 1.1, max_abs * 1.1)
        ax_sel.set_ylim(-0.5, 0.5)
        ax_sel.set_yticks([])
        ax_sel.set_title(
            var_titles.get(var, var),
            fontsize=components_size_config["FONTSIZE_TITLE"] - 2,
        )
        ax_sel.set_xlabel(
            "Percentage change (%)" if i == n_vars - 1 else "",
            fontsize=components_size_config["FONTSIZE_LABELS"],
        )
        ax_sel.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax_sel.grid(True, alpha=0.3, axis="x")
        ax_sel.tick_params(axis="x", labelsize=components_size_config["FONTSIZE_TICKS"])

    # Add common legend outside of all axes
    handles, labels = ax[0].get_legend_handles_labels()
    fig = ax[0].get_figure()
    fig.legend(
        handles,
        labels,
        fontsize=components_size_config["FONTSIZE_LEGEND"],
        loc="lower center",
        bbox_to_anchor=components_size_config["BBOX_TO_ANCHOR"],
    )

    if title is not None:
        plt.suptitle(title, fontsize=components_size_config["FONTSIZE_TITLE"], y=0.98)

    if path_savefig is not None:
        path_savefig.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path_savefig, dpi=300, bbox_inches="tight")
    if show_plot:
        plt.tight_layout()
        plt.show()


def plot_nodes_choropleth(
    gdf_nodes: gpd.GeoDataFrame,
    data: "xr.DataArray | xr.Dataset",
    var: Optional[str] = None,
    time_agg: Optional[str] = "mean",
    node_dim: str = "node",
    cmap: Optional[str] = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    ax: Optional[plt.Axes] = None,
    figsize: tuple = (14, 8),
    title: Optional[str] = None,
    label: Optional[str] = None,
    add_countries: bool = True,
    add_coastlines: bool = True,
    edgecolor: str = "white",
    linewidth: float = 0.4,
    projection=None,
    path_savefig: Optional[Path] = None,
) -> plt.Axes:
    """
    Plot node polygons coloured by values from a grouped xarray DataArray/Dataset.

    The ``data`` argument must contain a dimension named ``node_dim`` whose
    coordinate values are the integer node indices used when building the
    regionmask mask (i.e. ``numbers=range(len(gdf_nodes))``).
    This is the natural output of::

        ds.groupby(mask_nodes).mean(["lat", "lon"])

    If ``data`` also has a ``time`` dimension, it is reduced first via
    ``time_agg`` (default ``"mean"``).

    Parameters
    ----------
    gdf_nodes : geopandas.GeoDataFrame
        GeoDataFrame with one row per node.  Row positions must correspond
        to the integer node numbers in the ``node_dim`` coordinate.
    data : xr.DataArray or xr.Dataset
        Array / dataset with at least a ``node_dim`` dimension.
        If a Dataset, ``var`` must name the variable to plot.
        If a ``time`` dimension is present, ``time_agg`` is applied first.
    var : str, optional
        Variable to select when ``data`` is a :class:`xr.Dataset`.
    time_agg : {'mean', 'sum', 'min', 'max', 'median'} or None, optional
        Temporal aggregation applied before plotting.  Pass ``None`` to skip
        (the array must already be scalar in time).
    node_dim : str, optional
        Name of the node dimension.  Default ``'node'``.
    cmap : str, optional
        Matplotlib colormap name.
    vmin, vmax : float, optional
        Colour scale limits.  Inferred from data if not provided.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.  A new figure is created when not provided.
    figsize : tuple, optional
        Figure size used only when ``ax`` is ``None``.
    title : str, optional
        Axes title.
    label : str, optional
        Colorbar label.  Defaults to the variable name.
    add_countries : bool, optional
        Overlay country borders.
    add_coastlines : bool, optional
        Overlay coastlines.
    edgecolor : str, optional
        Colour of polygon edges.
    linewidth : float, optional
        Width of polygon edges.
    projection : cartopy.crs.CRS, optional
        Map projection.  Defaults to ``ccrs.PlateCarree()``.
    path_savefig : pathlib.Path, optional
        If provided, the figure is saved here.

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.cm as mcm
    import matplotlib.colors as mcolors_local
    import pandas as pd

    # --- resolve DataArray --------------------------------------------------
    if isinstance(data, xr.Dataset):
        if var is None:
            var = list(data.data_vars)[0]
        da = data[var]
    else:
        da = data
        if var is None and da.name is not None:
            var = da.name

    # --- temporal aggregation -----------------------------------------------
    if "time" in da.dims and time_agg is not None:
        da = getattr(da, time_agg)(dim="time")

    # guard: if time is still present, take first step
    if "time" in da.dims:
        da = da.isel(time=0)

    # --- build value series indexed by node number --------------------------
    node_coords = da[node_dim].values.astype(int)
    node_values = da.values  # shape: (n_nodes,) after time reduction

    value_series = pd.Series(node_values, index=node_coords)

    # --- merge into GeoDataFrame --------------------------------------------
    gdf = gdf_nodes.copy()
    gdf["_plot_value"] = gdf.index.map(value_series)

    # --- colour mapping -----------------------------------------------------
    v_min = vmin if vmin is not None else float(np.nanmin(node_values))
    v_max = vmax if vmax is not None else float(np.nanmax(node_values))
    norm = mcolors_local.Normalize(vmin=v_min, vmax=v_max)
    cmap_obj = mcm.get_cmap(cmap)

    # --- projection / axes --------------------------------------------------
    if projection is None:
        projection = ccrs.PlateCarree()

    created_fig = ax is None
    if created_fig:
        figsize = parse_figsize(figsize)
        fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": projection})

    # --- draw polygons ------------------------------------------------------
    for idx, row in gdf.iterrows():
        val = row["_plot_value"]
        color = cmap_obj(norm(val)) if not np.isnan(val) else "lightgrey"
        geom = row.geometry
        ax.add_geometries(
            [geom],
            crs=ccrs.PlateCarree(),
            facecolor=color,
            edgecolor=edgecolor,
            linewidth=linewidth,
        )

    # --- map extent from node bounds ----------------------------------------
    minx, miny, maxx, maxy = gdf.total_bounds
    pad_x = (maxx - minx) * 0.05
    pad_y = (maxy - miny) * 0.05
    ax.set_extent(
        [minx - pad_x, maxx + pad_x, miny - pad_y, maxy + pad_y],
        crs=ccrs.PlateCarree(),
    )

    # --- optional map features ----------------------------------------------
    if add_coastlines:
        ax.coastlines(linewidth=0.6, color="0.3")
    if add_countries:
        ax.add_feature(cfeature.BORDERS, edgecolor="0.3", linewidth=0.6, zorder=5)

    # --- colorbar -----------------------------------------------------------
    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation="vertical", shrink=0.6, pad=0.02)
    cbar.set_label(
        label if label is not None else (var or ""), fontsize=FONTSIZE_LABELS
    )
    cbar.ax.tick_params(labelsize=FONTSIZE_TICKS)

    # --- title / layout -----------------------------------------------------
    if title is not None:
        ax.set_title(title, fontsize=FONTSIZE_TITLE)

    if created_fig:
        plt.tight_layout()
        if path_savefig is not None:
            path_savefig.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(path_savefig, dpi=300, bbox_inches="tight")
        plt.show()

    return ax


def plot_unmet_demand_overview(
    ds_rl,
    var_name: str = "residual_load",
    zone_dim: str = "zone",
    n_cols: int = 3,
    cmap: str = "YlOrBr",
    vmax: float = None,
    height_per_zone: float = 0.22,
    panel_min_height: float = 1.8,
    cbar_label="3-Hourly Unmet Energy Demand (GW)",
    suptitle: str = None,
):
    """
    Yearly-grid heatmap of unmet energy demand.

    Layout: one panel per calendar year, arranged in *n_cols* columns filled
    top-to-bottom before moving to the next column.
    Within each panel, zones occupy the y-axis and time-steps the x-axis.
    Only positive residual-load values are shown (surplus is rendered as 0).
    """
    from matplotlib.colors import Normalize

    if zone_dim not in ds_rl.dims:
        zone_dim = "country"

    zones = ds_rl[zone_dim].values
    n_zones = len(zones)

    rl_pos = ds_rl[var_name].clip(min=0)
    years = sorted(np.unique(ds_rl.time.dt.year.values))
    n_years = len(years)
    n_rows_grid = int(np.ceil(n_years / n_cols))

    if vmax is None:
        vmax = float(rl_pos.quantile(0.999))
    norm = Normalize(vmin=0, vmax=vmax)
    cmap_obj = plt.get_cmap(cmap)

    panel_h = max(panel_min_height, n_zones * height_per_zone)
    fig_w = 22
    fig_h = n_rows_grid * panel_h

    fig, axes = plt.subplots(
        n_rows_grid,
        n_cols,
        figsize=(fig_w, fig_h),
        constrained_layout=False,
    )
    axes = np.atleast_2d(axes)

    month_abbrs = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]

    for idx, yr in enumerate(years):
        # Column-major: fill down each column before moving to the next
        ri = idx % n_rows_grid
        ci = idx // n_rows_grid
        ax = axes[ri, ci]

        # How many rows are populated in this column?
        col_size = min(n_rows_grid, n_years - ci * n_rows_grid)
        is_bottom_of_col = ri == col_size - 1
        is_first_col = ci == 0

        rl_year = rl_pos.sel(time=rl_pos.time.dt.year == yr)
        data = rl_year.transpose(zone_dim, "time").values  # (n_zones, n_time)

        ax.imshow(
            data,
            aspect="auto",
            norm=norm,
            cmap=cmap_obj,
            interpolation="nearest",
            origin="upper",
        )

        # y-axis: zone labels only in the first column
        if is_first_col:
            ax.set_yticks(np.arange(n_zones))
            ax.set_yticklabels(zones, fontsize=5)
            ax.tick_params(axis="y", which="major", length=1, pad=1)
        else:
            ax.set_yticks([])

        # thin white grid lines between zone rows (all panels)
        ax.set_yticks(np.arange(n_zones) - 0.5, minor=True)
        ax.yaxis.grid(which="minor", color="white", linewidth=0.3)
        ax.tick_params(axis="y", which="minor", length=0)

        # x-axis: month ticks only on the bottom panel of each column
        times = pd.DatetimeIndex(rl_year.time.values)
        month_ticks, month_labels = [], []
        for m in range(1, 13):
            pos = np.where(times.month == m)[0]
            if len(pos):
                month_ticks.append(int(pos[0]))
                month_labels.append(month_abbrs[m - 1])

        if is_bottom_of_col:
            ax.set_xticks(month_ticks)
            ax.set_xticklabels(month_labels, fontsize=6, rotation=45, ha="right")
            ax.tick_params(axis="x", length=2, pad=1)
        else:
            ax.set_xticks([])

        ax.set_title(str(yr), fontsize=8, loc="left", pad=2, fontweight="bold")
        ax.set_xlim(-0.5, data.shape[1] - 0.5)

    # hide unused panels (bottom of last column when n_years % n_rows_grid != 0)
    for idx in range(n_years, n_rows_grid * n_cols):
        ri = idx % n_rows_grid
        ci = idx // n_rows_grid
        axes[ri, ci].set_visible(False)

    # shared horizontal colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(
        sm,
        ax=axes.ravel().tolist(),
        orientation="horizontal",
        fraction=0.012,
        pad=0.01,
        shrink=0.35,
        aspect=40,
    )
    cbar.set_label(cbar_label, fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    fig.suptitle(
        suptitle if suptitle is not None else "Unmet Energy Demand by Balancing Zone",
        fontsize=11,
        y=0.89,
    )
    return fig


def plot_energy_extreme_metrics_dashboard(df_compiled):
    """Generates a 6-panel comprehensive visual dashboard comparing three periods

    (Reference, Mid-century, End-century) across all balancing zones, including
    the long-term structural evolution of climate drivers.
    """
    # Reset index to make it compatible with Seaborn plotting
    df_plot = df_compiled.reset_index()

    # Clean up period names for visual appeal in the legend
    df_plot["Period"] = df_plot["Period"].str.replace(r"^\d+_", "", regex=True)

    # Set up styling for academic/professional reporting
    sns.set_theme(style="whitegrid")

    # GridSpec with 4 rows: [A/B] [C/D] [era-legend] [E/F]
    # The dedicated legend row (index 2) is kept invisible to reserve space.
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(16, 20))
    gs = GridSpec(
        4, 2,
        figure=fig,
        height_ratios=[1, 1, 0.05, 1.02],
        hspace=0.3,
        wspace=0.3,
        top=0.93,
        bottom=0.10,
        left=0.08,
        right=0.97,
    )
    _ax00 = fig.add_subplot(gs[0, 0])
    _ax01 = fig.add_subplot(gs[0, 1])
    _ax10 = fig.add_subplot(gs[1, 0])
    _ax11 = fig.add_subplot(gs[1, 1])
    ax_era_row = fig.add_subplot(gs[2, :])
    ax_era_row.set_visible(False)
    _ax20 = fig.add_subplot(gs[3, 0])
    _ax21 = fig.add_subplot(gs[3, 1])
    axes = np.array([[_ax00, _ax01], [_ax10, _ax11], [_ax20, _ax21]])

    fig.suptitle(
        "Evolution of Extreme Grid Events & Climate Drivers Across Climate Eras",
        fontsize=18,
        weight="bold",
        y=0.97,
    )

    palette = ["#4A90E2", "#F5A623", "#E24A4A"]  # Ref, Mid, End clean colors

    # =====================================================================
    # --- ROW 1: CORE GRID STRESS ---
    # =====================================================================

    # --- Panel A: Frequency ---
    sns.barplot(
        data=df_plot,
        x="Zone",
        y="Frequency",
        hue="Period",
        ax=axes[0, 0],
        palette=palette,
    )
    axes[0, 0].set_title(
        "A. Annual Event Frequency (Count)", fontsize=13, weight="semibold"
    )
    axes[0, 0].set_ylabel("Number of Events / Year")
    axes[0, 0].get_legend().remove()

    # --- Panel B: Duration (Average vs Max) ---
    sns.barplot(
        data=df_plot,
        x="Zone",
        y="Max_Duration_Hours",
        hue="Period",
        ax=axes[0, 1],
        palette=palette,
        alpha=0.3,
    )
    sns.barplot(
        data=df_plot,
        x="Zone",
        y="Avg_Duration_Hours",
        hue="Period",
        ax=axes[0, 1],
        palette=palette,
    )
    axes[0, 1].set_title(
        "B. Duration Stress (Bars: Average, Shaded: Absolute Maximum)",
        fontsize=13,
        weight="semibold",
    )
    axes[0, 1].set_ylabel("Duration (Hours)")
    axes[0, 1].get_legend().remove()

    # =====================================================================
    # --- ROW 2: ENERGY & SYSTEM INTERACTION ---
    # =====================================================================

    # --- Panel C: Average Energy Severity ---
    sns.barplot(
        data=df_plot,
        x="Zone",
        y="Avg_Energy_Severity_MWh",
        hue="Period",
        ax=axes[1, 0],
        palette=palette,
    )
    axes[1, 0].set_title(
        "C. Average Energy Severity per Event (MWh)",
        fontsize=13,
        weight="semibold",
    )
    axes[1, 0].set_ylabel("Energy Deficit / Event (MWh)")
    axes[1, 0].get_legend().remove()

    # --- Panel D: Cross-Zonal Interconnection Traffic (PCIT) ---
    sns.barplot(
        data=df_plot,
        x="Zone",
        y="Avg_PCIT_During_Events_MW",
        hue="Period",
        ax=axes[1, 1],
        palette=palette,
    )
    axes[1, 1].set_title(
        "D. Coincident Cross-Border Interconnection Traffic (PCIT)",
        fontsize=13,
        weight="semibold",
    )
    axes[1, 1].set_ylabel("Average PCIT Volume (MW)")
    axes[1, 1].get_legend().remove()

    # =====================================================================
    # --- ROW 3: CLIMATE DRIVER COMPOSITION SHARE EVOLUTION ---
    # =====================================================================

    # Define color scheme for the individual structural driving forces
    driver_colors = {
        "Demand Spikes": "#D95F02",  # Warm Orange for demand
        "Solar PV Drought": "#E6AB02",  # Golden Yellow for Solar PV
        "Onshore Wind Drought": "#7570B3",  # Muted Purple for Onshore Wind
        "Offshore Wind Drought": "#1B7C3D",  # Dark Green for Offshore Wind
        "Hydro RoR Drought": "#3182BD",  # Water Blue for Hydro Run-of-River
    }

    # Map your dictionary data columns to clean, presentable legend names
    driver_mapping = {
        "Demand_Driven_Share_%": "Demand Spikes",
        "PV_Driven_Share_%": "Solar PV Drought",
        "Wind_Onshore_Driven_Share_%": "Onshore Wind Drought",
        "Wind_Offshore_Driven_Share_%": "Offshore Wind Drought",
        "Hydro_RoR_Driven_Share_%": "Hydro RoR Drought",
    }

    # Order data: Ref first, then Mid, then End within each Zone
    df_plot["_period_sort"] = df_plot["Period"].map(
        lambda p: 0 if p.lower().startswith("ref") else (1 if p.lower().startswith("mid") else 2)
    )
    df_drivers = df_plot.sort_values(by=["Zone", "_period_sort"]).drop(columns=["_period_sort"]).copy()

    # Create distinct labels for each individual bar (e.g., "Zone X\nReference")
    bar_positions = np.arange(len(df_drivers))
    tick_labels = [
        f"{row['Zone']}\n({row['Period'][:3]})"
        for _, row in df_drivers.iterrows()
    ]

    # Initialize a zero base array to systematically construct stacked heights
    bottom_features_left = np.zeros(len(df_drivers))
    bottom_features_right = np.zeros(len(df_drivers))

    # --- Split Driver Analysis into Two Complementary Views for Visual Clarity ---

    # Panel E: Demand vs. Collective Renewable Supply Droughts
    total_renewables = (
        df_drivers["PV_Driven_Share_%"]
        + df_drivers["Wind_Onshore_Driven_Share_%"]
        + df_drivers["Wind_Offshore_Driven_Share_%"]
        + df_drivers["Hydro_RoR_Driven_Share_%"]
    )

    axes[2, 0].bar(
        bar_positions,
        df_drivers["Demand_Driven_Share_%"],
        label="Demand Spikes",
        color=driver_colors["Demand Spikes"],
        width=0.75,
    )
    axes[2, 0].bar(
        bar_positions,
        total_renewables,
        bottom=df_drivers["Demand_Driven_Share_%"],
        label="Supply Droughts (All RES)",
        color="#7F7F7F",
        width=0.75,
    )

    axes[2, 0].set_title(
        "E. Core Structural Pivot (Demand vs. Supply-Driven Shares)",
        fontsize=13,
        weight="semibold",
    )
    axes[2, 0].set_ylabel("Percentage Component Share (%)")
    axes[2, 0].set_ylim(0, 100)
    axes[2, 0].set_xticks(bar_positions)
    axes[2, 0].set_xticklabels(tick_labels, rotation=45, fontsize=9, ha="right")
    axes[2, 0].legend(bbox_to_anchor=(0.5, -0.25), loc="upper center", frameon=True, fontsize=9, ncol=2)

    # Panel F: Granular Asset Breakdown of the Crisis Mass
    for col_key, clean_label in driver_mapping.items():
        axes[2, 1].bar(
            bar_positions,
            df_drivers[col_key],
            bottom=bottom_features_right,
            label=clean_label,
            color=driver_colors[clean_label],
            width=0.75,
        )
        bottom_features_right += df_drivers[col_key].values

    axes[2, 1].set_title(
        "F. Granular Climate Drivers (Individual Asset Contribution to Deficit)",
        fontsize=13,
        weight="semibold",
    )
    axes[2, 1].set_ylabel("Percentage Component Share (%)")
    axes[2, 1].set_ylim(0, 100)
    axes[2, 1].set_xticks(bar_positions)
    axes[2, 1].set_xticklabels(tick_labels, rotation=45, fontsize=9, ha="right")

    # Dedup and build a clean legend for panel F
    handles_f, labels_f = axes[2, 1].get_legend_handles_labels()
    axes[2, 1].legend(
        handles_f, labels_f, bbox_to_anchor=(0.5, -0.25), loc="upper center", frameon=True, fontsize=9, ncol=3
    )

    # =====================================================================
    # --- CLEAN DESIGN LAYOUT MANAGEMENT ---
    # =====================================================================

    # Rotate X labels on Row 1 and Row 2 for baseline cleanliness
    for row in range(2):
        for col in range(2):
            axes[row, col].set_xticklabels(
                axes[row, col].get_xticklabels(), rotation=15, ha="right"
            )
            axes[row, col].set_xlabel("")

    # Global Era Legend: anchored to the dedicated invisible legend row
    handles_era, labels_era = axes[1, 1].get_legend_handles_labels()
    handles_era = handles_era[:3]
    labels_era = labels_era[:3]

    era_pos = ax_era_row.get_position()
    fig.legend(
        handles_era,
        labels_era,
        title="Climate Period Evaluation (Fixed 2050 Infrastructure)",
        title_fontsize=12,
        loc="center",
        frameon=True,
        shadow=False,
        ncol=3,
        fontsize=11,
        bbox_to_anchor=(0.5, (era_pos.y0 + era_pos.y1) / 2),
        bbox_transform=fig.transFigure,
    )
    return fig