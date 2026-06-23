from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import xarray as xr
import yaml


def load_warming_level_periods(
    yml_path: Union[str, Path],
    warming_levels: list[float] | None = None,
    gcms: Union[list[str], dict[str, str], None] = None,
) -> dict[tuple[str, str], dict[str, dict[float, dict | None]]]:
    """Load warming-level periods from a pre-computed YAML file.

    The YAML format is the one produced by Mathias Hauser's ``warming_levels.py``
    script (e.g. ``cmip5_warming_levels_all_ens_1850_1900.yml``).  Each section
    key encodes the level as an integer with one implicit decimal place:
    ``warming_level_15`` → 1.5 °C, ``warming_level_061`` → 0.61 °C.

    Parameters
    ----------
    yml_path : str | Path
        Path to the YAML file.
    warming_levels : list[float] | None
        Levels to extract (e.g. ``[1.5, 2.0, 3.0]``).  ``None`` returns all
        levels present in the file.
    gcms : list[str] | dict[str, str] | None
        Models to include.

        - ``None`` — return all models.
        - ``list[str]`` — include all ensemble members for the listed models.
        - ``dict[str, str]`` — ``{model: ensemble}`` mapping; keeps only the
          specified ensemble member per model (e.g.
          ``{"CNRM-CM5": "r1i1p1", "EC-EARTH": "r12i1p1"}``).

    Returns
    -------
    dict
        Nested dict keyed by ``(model, ensemble)``::

            {
                (model, ensemble): {
                    scenario: {
                        warming_level: {
                            "start_year": int,
                            "end_year": int,
                            "central_year": int,
                        }
                    }
                }
            }

        Missing combinations (level not reached, model absent from the file)
        are simply not present in the dict; use ``.get()`` to handle them.
    """
    def _suffix_to_float(suffix: str) -> float:
        # "15" → 1.5,  "20" → 2.0,  "061" → 0.61
        return int(suffix) / 10 ** (len(suffix) - 1)

    with open(Path(yml_path)) as fh:
        data = yaml.safe_load(fh)

    # Map float level → YAML section key for every level in the file
    available: dict[float, str] = {}
    for key in data:
        if key.startswith("warming_level_"):
            suffix = key[len("warming_level_"):]
            available[_suffix_to_float(suffix)] = key

    keys_to_read = (
        {wl: available[wl] for wl in warming_levels if wl in available}
        if warming_levels is not None
        else available
    )

    # Normalise gcms argument into (model_set | None, ens_map | None)
    if gcms is None:
        model_set, ens_map = None, None
    elif isinstance(gcms, dict):
        model_set, ens_map = set(gcms.keys()), gcms
    else:
        model_set, ens_map = set(gcms), None

    result: dict[tuple[str, str], dict[str, dict[float, dict]]] = {}
    for wl, key in keys_to_read.items():
        for entry in data.get(key, []):
            model, ens = entry["model"], entry["ensemble"]
            if model_set is not None and model not in model_set:
                continue
            if ens_map is not None and ens_map.get(model) != ens:
                continue
            sy = entry["start_year"]
            me_key = (model, ens)
            result.setdefault(me_key, {}).setdefault(entry["exp"], {})[wl] = {
                "start_year": sy,
                "end_year": entry["end_year"],
                "central_year": sy + 10,
            }

    return result


def compute_gmst_anomalies(
    ds, var, reference_period, time_dim="time"
) -> tuple[np.ndarray, xr.DataArray]:
    mean = ds[var].mean(dim=["lat", "lon"])
    years = ds[time_dim].values if time_dim in ds else np.arange(len(mean))
    ref_mean = mean.sel(year=reference_period).mean()
    anomalies = mean - ref_mean
    return years, anomalies


def compute_percentage_change(
    ref: xr.Dataset, target: xr.Dataset, round: int = 2, epsilon: float = 0
) -> xr.Dataset:
    """
    Compute the percentage change between a reference dataset and a target dataset.

    Parameters
    ----------
    ref : xr.Dataset
        Reference dataset.
    target : xr.Dataset
        Target dataset to compare against the reference.
    round : int, optional
        Number of decimal places to round the result, default is 2.
    epsilon : float, optional
        Small value to avoid division by zero, default is 0.

    Returns
    -------
    xr.DataArray
        Percentage change between target and reference datasets, calculated as
        100 * (target - ref) / ref.

    """
    return (100 * (target - ref) / (ref + epsilon)).round(round)


def groupby_season(
    ds: xr.Dataset | xr.DataArray,
    time_dim: str = "time",
    season_type: str = "meteorological",
) -> xr.Dataset | xr.DataArray:
    """
    Group the dataset by seasons.

    Parameters
    ----------
    ds : xr.Dataset | xr.DataArray
        Input dataset containing the variable of interest.
    time_dim : str, optional
        Name of the time dimension, default is "time".
    season_type : str, optional
        Type of seasonal breakdown to use, default is "meteorological".
        Options:
        - "meteorological": Winter (DJF), Spring (MAM), Summer (JJA), Fall (SON)
        - "energy": Winter (ONDJFM), Summer (AMJJAS)

    Returns
    -------
    xr.Dataset | xr.DataArray
        Dataset grouped by seasons.

    """
    # Ensure time is a datetime type
    if not np.issubdtype(ds[time_dim].dtype, np.datetime64):
        ds = ds.copy()
        ds[time_dim] = xr.decode_cf(ds[[time_dim]])[time_dim]

    if season_type == "meteorological":
        # seasons = xr.where(
        #     months.isin([12, 1, 2]), "Winter",
        #     xr.where(
        #         months.isin([3, 4, 5]), "Spring",
        #         xr.where(
        #             months.isin([6, 7, 8]), "Summer",
        #             "Fall"
        #         )
        #     )
        # )
        seasons = ds[time_dim].dt.season
        seasons = [
            "Winter"
            if s == "DJF"
            else "Spring"
            if s == "MAM"
            else "Summer"
            if s == "JJA"
            else "Fall"
            for s in seasons.values
        ]
    elif season_type == "energy":
        months = ds[time_dim].dt.month
        seasons = xr.where(months.isin([10, 11, 12, 1, 2, 3]), "Winter", "Summer")
    else:
        raise ValueError(
            f"Unknown season_type: {season_type}. Must be 'meteorological' or 'energy'."
        )

    ds = ds.assign_coords(season=seasons)
    return ds.groupby("season")


def compute_seasonal_statistics(
    ds: xr.Dataset | xr.DataArray,
    time_dim: str = "time",
    method: str = "mean",
    season_type: str = "meteorological",
) -> xr.Dataset | xr.DataArray:
    """
    Compute seasonal statistics for each year in the dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset containing the variable of interest.
    time_dim : str, optional
        Name of the time dimension, default is "time".
    method : str, optional
        Statistical method to use, default is "mean".
    season_type : str, optional
        Type of seasonal breakdown to use, default is "meteorological".
        Options:
        - "meteorological": Winter (DJF), Spring (MAM), Summer (JJA), Fall (SON)
        - "energy": Winter (ONDJFM), Summer (AMJJAS)

    Returns
    -------
    xr.DataArray
        Seasonal averages with dimensions ('year', 'season', ...).
    """
    # Ensure time is a datetime type
    if not np.issubdtype(ds[time_dim].dtype, np.datetime64):
        ds = ds.copy()
        ds[time_dim] = xr.decode_cf(ds[[time_dim]])[time_dim]

    if method not in ["mean", "median", "sum"]:
        raise ValueError(f"Unknown method: {method}")

    if season_type not in ["meteorological", "energy"]:
        raise ValueError(
            f"Unknown season_type: {season_type}. Must be 'meteorological' or 'energy'."
        )

    if season_type == "meteorological":
        # Use standard meteorological seasons (DJF, MAM, JJA, SON)
        ds_resampled = ds.resample({time_dim: "QS-DEC"})
        if method == "sum":
            ds_resampled = ds_resampled.sum(dim=time_dim)
        elif method == "median":
            ds_resampled = ds_resampled.median(dim=time_dim)
        else:
            ds_resampled = ds_resampled.mean(dim=time_dim)
        ds_resampled = ds_resampled.assign_coords(
            year=(time_dim, ds_resampled[time_dim].dt.year.data),
            season=(
                time_dim,
                [
                    "Winter"
                    if s == "DJF"
                    else "Spring"
                    if s == "MAM"
                    else "Summer"
                    if s == "JJA"
                    else "Fall"
                    for s in ds_resampled[time_dim].dt.season.data
                ],
            ),
        )
        return ds_resampled

    else:  # energy seasons
        # Define energy seasons: Winter (ONDJFM), Summer (AMJJAS)
        # Create a copy to avoid modifying the original
        ds_copy = ds.copy()
        months = ds_copy[time_dim].dt.month

        # Assign energy seasons: 10-3 = Winter, 4-9 = Summer
        # Winter: October (10), November (11), December (12), January (1), February (2), March (3)
        # Summer: April (4), May (5), June (6), July (7), August (8), September (9)
        energy_season = xr.where((months >= 10) | (months <= 3), "Winter", "Summer")

        # For year assignment in energy seasons, Oct-Dec belong to the next year's winter
        year = ds_copy[time_dim].dt.year.copy()
        year = xr.where((months >= 10), year + 1, year)

        # Add season and year as coordinates
        ds_copy = ds_copy.assign_coords(energy_season=energy_season, energy_year=year)

        # Create a combined year-season grouping variable
        # Use numpy char operations for string concatenation
        year_str = year.values.astype(str)
        season_str = energy_season.values.astype(str)
        year_season_values = np.char.add(np.char.add(year_str, "_"), season_str)
        year_season = xr.DataArray(
            year_season_values, dims=time_dim, coords={time_dim: ds_copy[time_dim]}
        )
        ds_copy = ds_copy.assign_coords(year_season=year_season)

        # Group by the combined coordinate
        ds_grouped = ds_copy.groupby("year_season")

        if method == "sum":
            ds_result = ds_grouped.sum(dim=time_dim)
        elif method == "median":
            ds_result = ds_grouped.median(dim=time_dim, skipna=True)
        else:
            ds_result = ds_grouped.mean(dim=time_dim, skipna=True)

        # Extract year and season from the combined coordinate
        year_season_values = ds_result["year_season"].values
        years = np.array([int(ys.split("_")[0]) for ys in year_season_values])
        seasons = np.array([ys.split("_")[1] for ys in year_season_values])

        # Create a multi-index with year and season
        ds_result = ds_result.assign_coords(
            year=("year_season", years), season=("year_season", seasons)
        )

        # Reshape to have year and season as separate dimensions
        # Use set_index and unstack to pivot the data
        ds_result = ds_result.set_index(year_season=["year", "season"])
        ds_result = ds_result.unstack("year_season")

        return ds_result


def compute_regional_values(
    ds: xr.Dataset | xr.DataArray,
    regions_mask: xr.DataArray,
    regions_list: list[str],
    method: str = "mean",
) -> xr.Dataset | xr.DataArray:
    """
    Compute regional averages for the given dataset using the provided regions mask.

    Parameters
    ----------
    ds : xr.Dataset | xr.DataArray
        Input dataset containing the variable of interest.
    regions_mask : xr.DataArray
        Boolean mask indicating the regions over which to compute the averages.
    regions_list : list[str]
        List of region names corresponding to the regions mask.
    method : str, optional
        Aggregation method to use, default is "mean".

    Returns
    -------
    xr.Dataset | xr.DataArray
        Regional averages with dimensions ('region', ...).
    """
    if method not in ["mean", "median", "sum"]:
        raise ValueError(f"Unknown method: {method}")
    ds_regional = ds.groupby(regions_mask)
    if method == "median":
        ds_regional = ds_regional.median(dim=("lat", "lon"))
    elif method == "sum":
        ds_regional = ds_regional.sum(dim=("lat", "lon"))
    else:
        ds_regional = ds_regional.mean(dim=("lat", "lon"))
    ds_regional["mask"] = regions_list
    return ds_regional


def compute_sign_agreement(
    ds: xr.Dataset | xr.DataArray, model_dim: str = "model"
) -> xr.Dataset | xr.DataArray:
    """
    Compute the number of models that agree in sign with the multi-model mean.

    Parameters
    ----------
    ds : xr.Dataset | xr.DataArray
        Input data array with a model dimension.
    model_dim : str, optional
        Name of the model dimension, default is "model".

    Returns
    -------
    xr.Dataset | xr.DataArray
        Array with the count of models agreeing in sign with the multi-model mean.
    """
    return xr.apply_ufunc(
        lambda x: np.sum((np.sign(x) == np.sign(np.nanmean(x))) & ~np.isnan(x)),
        ds,
        input_core_dims=[[model_dim]],
        vectorize=True,
    )


def ensure_daily(
    ds: xr.Dataset | xr.DataArray,
    method: str = "mean",
) -> xr.Dataset | xr.DataArray:
    """Resample to daily if data is at sub-daily resolution.

    Parameters
    ----------
    ds : xr.Dataset | xr.DataArray
        Input dataset with a ``time`` dimension.
    method : str
        'mean' for capacity-factor-like variables (PV, wind, hydro);
        'sum'  for accumulation variables (HDD, CDD).

    Returns
    -------
    xr.Dataset | xr.DataArray
        Dataset resampled to daily cadence, or the original if already daily.
    """
    if len(ds.time) < 2:
        return ds
    dt_hours = float(
        (ds.time.values[1] - ds.time.values[0]) / np.timedelta64(1, "h")
    )
    if dt_hours < 24:
        return (
            ds.resample(time="D").sum(dim="time")
            if method == "sum"
            else ds.resample(time="D").mean(dim="time")
        )
    return ds


def compute_doy_threshold(
    ds_daily: xr.Dataset | xr.DataArray,
    reference_period: list[str],
    percentile: float = 10.0,
    window: int = 15,
) -> xr.Dataset | xr.DataArray:
    """Per-DOY P<percentile> threshold with a \u00b1window-day centred moving window.

    For each calendar DOY d (1\u2013365) all daily values within [d\u2212window, d+window]
    (wrapping around the year boundary) are pooled across all years of the
    reference period, giving (2\u00b7window+1)\u00b7n_years \u2248 930 samples per bin for the
    default settings.  The p<percentile> quantile of that pool is the threshold.

    Leap days (DOY 366) are mapped to DOY 365 before pooling.

    Parameters
    ----------
    ds_daily : xr.Dataset | xr.DataArray
        Daily-resolution dataset covering at least *reference_period*.
    reference_period : list[str]
        Two-element list ``[start_year, end_year]`` used to slice the data.
    percentile : float
        Percentile to compute (0\u2013100).  Default 10.
    window : int
        Half-width in days of the centred DOY window.  Default 15.

    Returns
    -------
    xr.Dataset | xr.DataArray
        Same structure as input but with the ``time`` dimension replaced by
        ``doy`` (integer 1\u2013365).
    """
    q = percentile / 100.0
    ds_ref = ds_daily.sel(time=slice(*reference_period))

    # Map DOY 366 (leap days) to 365 so we stay within a 1-365 range
    doy_arr = np.clip(ds_ref.time.dt.dayofyear.values, 1, 365)

    thresholds = []
    for doy in range(1, 366):
        # Centred window with wrap-around; set avoids duplicates at the extremes
        window_doys = {
            (doy - window + offset - 1) % 365 + 1
            for offset in range(2 * window + 1)
        }
        mask = np.isin(doy_arr, list(window_doys))
        pooled = ds_ref.isel(time=mask)
        thr = pooled.quantile(q, dim="time", skipna=True).drop_vars("quantile")
        thresholds.append(thr)

    doy_coord = xr.DataArray(np.arange(1, 366, dtype=int), dims="doy", name="doy")
    return xr.concat(thresholds, dim=doy_coord)


def compute_exceedance_frequency(
    ds_daily: xr.Dataset | xr.DataArray,
    threshold_doy: xr.Dataset | xr.DataArray,
    period: list[str],
) -> xr.Dataset | xr.DataArray:
    """Fraction of days in *period* where the daily value falls below the
    per-DOY reference threshold.

    For supply variables (PV, wind, hydro) this counts low-generation days
    (values below the P10 baseline).  For demand variables (HDD, CDD) pass
    a P90 threshold and interpret the result as the fraction of days that
    exceed the historical P90.

    Parameters
    ----------
    ds_daily : xr.Dataset | xr.DataArray
        Daily-resolution dataset covering at least *period*.
    threshold_doy : xr.Dataset | xr.DataArray
        Per-DOY threshold as returned by :func:`compute_doy_threshold`,
        with a ``doy`` dimension (integer 1\u2013365).
    period : list[str]
        Two-element list ``[start_year, end_year]`` defining the target period.

    Returns
    -------
    xr.Dataset | xr.DataArray
        No time dimension; one fraction value per (model, lat, lon).
    """
    ds_period = ds_daily.sel(time=slice(*period))

    # Build a time-indexed DOY array, clipping leap days to 365
    doy_idx = xr.DataArray(
        np.clip(ds_period.time.dt.dayofyear.values, 1, 365),
        coords={"time": ds_period.time},
        dims="time",
    )

    # Vectorised lookup along the doy dimension \u2192 result has time dimension
    thr = threshold_doy.sel(doy=doy_idx)

    below = ds_period < thr
    return below.mean(dim="time", skipna=True)
