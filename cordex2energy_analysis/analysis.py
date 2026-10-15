"""Warming-level windows, seasonal statistics, sign agreement and inter-model spread.

The warming-level windows come from a pre-computed YAML of crossing years per CMIP5 run; a
window can be fixed for every chain or resolved per chain (:func:`resolve_model_periods`).
The inter-model spread conventions (:data:`SPREAD_PERCENTILES`) are shared by every box and
shaded band of the figures.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import xarray as xr
import yaml

#: Fixed periods ``{name: (first year, last year)}``, the same for every chain.
FixedPeriodSpec = dict[str, tuple[str, str]]
#: Per-chain periods ``{name: {chain: (first year, last year)}}``.
PerModelPeriodSpec = dict[str, dict[str, tuple[str, str]]]
#: Either form of period definition.
PeriodDefs = Union[FixedPeriodSpec, PerModelPeriodSpec]

#: (low, high) percentiles of the **box** for each spread convention. Under ``"tukey"`` the box is
#: still the IQR; it is the whisker that follows the 1.5 x IQR rule (see
#: :func:`model_spread_bounds`).
SPREAD_PERCENTILES = {"iqr": (25.0, 75.0), "minmax": (0.0, 100.0), "tukey": (25.0, 75.0)}
#: How each spread convention is described on the figures.
SPREAD_LABEL = {"iqr": "IQR (P25–P75)", "minmax": "min–max", "tukey": "1.5×IQR (Tukey)"}


# ---------------------------------------------------------------------------
# Warming-level windows
# ---------------------------------------------------------------------------

def load_warming_level_periods(
    yaml_path: Union[str, Path],
    warming_levels: list[float] | None = None,
    gcms: Union[list[str], dict[str, str], None] = None,
) -> dict[tuple[str, str], dict[str, dict[float, dict | None]]]:
    """Load warming-level periods from a pre-computed YAML file.

    Parameters
    ----------
    yaml_path : str or pathlib.Path
        YAML file in the format written by M. Hauser's ``warming_levels.py`` (e.g.
        ``cmip5_warming_levels_all_ens_1861_1900.yml``). Each section key encodes the level as an
        integer with one implicit decimal place: ``warming_level_15`` is 1.5 °C,
        ``warming_level_061`` is 0.61 °C.
    warming_levels : list of float, optional
        Levels to extract (e.g. ``[1.5, 2.0, 3.0]``). ``None`` returns every level in the file.
    gcms : list of str or dict, optional
        Models to include. ``None`` keeps every model; a list keeps every ensemble member of the
        listed models; a ``{model: ensemble}`` dict keeps only that member of each model (e.g.
        ``{"CNRM-CM5": "r1i1p1", "EC-EARTH": "r12i1p1"}``).

    Returns
    -------
    dict
        ``{(model, ensemble): {scenario: {warming_level: {"start_year": int, "end_year": int,
        "central_year": int}}}}``. Combinations absent from the file (level not reached, model
        not listed) are absent from the dict.
    """
    def suffix_to_level(suffix: str) -> float:
        """Warming level of a YAML key suffix, e.g. "15" -> 1.5 and "061" -> 0.61."""
        return int(suffix) / 10 ** (len(suffix) - 1)

    with open(Path(yaml_path)) as handle:
        sections = yaml.safe_load(handle)

    available: dict[float, str] = {}
    for key in sections:
        if key.startswith("warming_level_"):
            available[suffix_to_level(key[len("warming_level_"):])] = key

    keys_to_read = (
        {level: available[level] for level in warming_levels if level in available}
        if warming_levels is not None
        else available
    )

    if gcms is None:
        model_set, member_by_model = None, None
    elif isinstance(gcms, dict):
        model_set, member_by_model = set(gcms.keys()), gcms
    else:
        model_set, member_by_model = set(gcms), None

    result: dict[tuple[str, str], dict[str, dict[float, dict]]] = {}
    for level, key in keys_to_read.items():
        for entry in sections.get(key, []):
            model, ensemble = entry["model"], entry["ensemble"]
            if model_set is not None and model not in model_set:
                continue
            if member_by_model is not None and member_by_model.get(model) != ensemble:
                continue
            start_year = entry["start_year"]
            result.setdefault((model, ensemble), {}).setdefault(entry["exp"], {})[level] = {
                "start_year": start_year,
                "end_year": entry["end_year"],
                "central_year": start_year + 10,
            }

    return result


def resolve_model_periods(
    period_windows: PeriodDefs,
    model: str,
) -> dict[str, tuple[str, str] | None]:
    """Resolve period definitions for one chain.

    Parameters
    ----------
    period_windows : dict
        Fixed periods ``{name: (start, end)}`` or per-chain periods
        ``{name: {chain: (start, end)}}``; the two forms may be mixed.
    model : str
        Chain identifier (e.g. ``"CNRM-ALADIN63_CNRM-CERFACS-CNRM-CM5_r1i1p1"``).

    Returns
    -------
    dict
        ``{name: (start, end)}``; ``None`` where a per-chain period has no entry for ``model``
        (the chain does not reach that warming level).
    """
    resolved: dict[str, tuple[str, str] | None] = {}
    for name, spec in period_windows.items():
        if isinstance(spec, dict):
            resolved[name] = spec.get(model)
        else:
            resolved[name] = spec
    return resolved


# ---------------------------------------------------------------------------
# Seasonal statistics and sign agreement
# ---------------------------------------------------------------------------

def compute_seasonal_statistics(
    data: xr.Dataset | xr.DataArray,
    time_dim: str = "time",
    method: str = "mean",
    season_type: str = "meteorological",
) -> xr.Dataset | xr.DataArray:
    """Seasonal statistic of every season of every year.

    Parameters
    ----------
    data : xarray.Dataset or xarray.DataArray
        Input with a time dimension; a non-datetime time coordinate is CF-decoded first.
    time_dim : str
        Name of the time dimension.
    method : {"mean", "median", "sum"}
        Statistic taken within each season.
    season_type : {"meteorological", "energy"}
        ``"meteorological"``: Winter (DJF), Spring (MAM), Summer (JJA), Fall (SON).
        ``"energy"``: Winter (ONDJFM) and Summer (AMJJAS); October-December count towards the
        following year's winter.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Meteorological seasons: along ``time_dim`` (one step per season, starting in December)
        with ``year`` and ``season`` coordinates. Energy seasons: dimensions ``year`` and
        ``season`` in place of ``time_dim``.

    Raises
    ------
    ValueError
        If ``method`` or ``season_type`` is not one of the options.
    """
    if not np.issubdtype(data[time_dim].dtype, np.datetime64):
        data = data.copy()
        data[time_dim] = xr.decode_cf(data[[time_dim]])[time_dim]

    if method not in ["mean", "median", "sum"]:
        raise ValueError(f"Unknown method: {method}")

    if season_type not in ["meteorological", "energy"]:
        raise ValueError(
            f"Unknown season_type: {season_type}. Must be 'meteorological' or 'energy'."
        )

    if season_type == "meteorological":
        resampled = data.resample({time_dim: "QS-DEC"})
        if method == "sum":
            resampled = resampled.sum(dim=time_dim)
        elif method == "median":
            resampled = resampled.median(dim=time_dim)
        else:
            resampled = resampled.mean(dim=time_dim)
        resampled = resampled.assign_coords(
            year=(time_dim, resampled[time_dim].dt.year.data),
            season=(
                time_dim,
                [
                    "Winter"
                    if season == "DJF"
                    else "Spring"
                    if season == "MAM"
                    else "Summer"
                    if season == "JJA"
                    else "Fall"
                    for season in resampled[time_dim].dt.season.data
                ],
            ),
        )
        return resampled

    labelled = data.copy()
    months = labelled[time_dim].dt.month
    energy_season = xr.where((months >= 10) | (months <= 3), "Winter", "Summer")
    # October-December belong to the following year's winter.
    year = labelled[time_dim].dt.year.copy()
    year = xr.where((months >= 10), year + 1, year)
    labelled = labelled.assign_coords(energy_season=energy_season, energy_year=year)

    year_season_values = np.char.add(np.char.add(year.values.astype(str), "_"),
                                     energy_season.values.astype(str))
    year_season = xr.DataArray(
        year_season_values, dims=time_dim, coords={time_dim: labelled[time_dim]}
    )
    labelled = labelled.assign_coords(year_season=year_season)
    grouped = labelled.groupby("year_season")

    if method == "sum":
        result = grouped.sum(dim=time_dim)
    elif method == "median":
        result = grouped.median(dim=time_dim, skipna=True)
    else:
        result = grouped.mean(dim=time_dim, skipna=True)

    year_season_values = result["year_season"].values
    years = np.array([int(label.split("_")[0]) for label in year_season_values])
    seasons = np.array([label.split("_")[1] for label in year_season_values])
    result = result.assign_coords(
        year=("year_season", years), season=("year_season", seasons)
    )
    result = result.set_index(year_season=["year", "season"])
    return result.unstack("year_season")


def compute_sign_agreement(
    data: xr.Dataset | xr.DataArray, model_dim: str = "model"
) -> xr.Dataset | xr.DataArray:
    """Number of models whose sign matches that of the multi-model mean.

    Parameters
    ----------
    data : xarray.Dataset or xarray.DataArray
        Input with a model dimension.
    model_dim : str
        Name of the model dimension.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Count of non-NaN models agreeing in sign with the NaN-ignoring mean, without
        ``model_dim``.
    """
    return xr.apply_ufunc(
        lambda values: np.sum((np.sign(values) == np.sign(np.nanmean(values)))
                              & ~np.isnan(values)),
        data,
        input_core_dims=[[model_dim]],
        vectorize=True,
    )


# ---------------------------------------------------------------------------
# Inter-model spread
# ---------------------------------------------------------------------------

def spread_qpair(spread: str = "iqr") -> tuple[float, float]:
    """(low, high) box percentiles of a spread convention.

    Parameters
    ----------
    spread : {"iqr", "minmax", "tukey"}
        Spread convention.

    Returns
    -------
    tuple of float
        ``(25, 75)`` for ``"iqr"`` and ``"tukey"``, ``(0, 100)`` for ``"minmax"``.

    Raises
    ------
    ValueError
        If ``spread`` is not a known convention.

    Notes
    -----
    These are the percentiles of the box only. Under ``"tukey"`` the whisker is not a
    percentile, so bands and whiskers must use :func:`model_spread_bounds`.
    """
    try:
        return SPREAD_PERCENTILES[spread]
    except KeyError as exc:
        raise ValueError(
            f"spread must be one of {list(SPREAD_PERCENTILES)}; got {spread!r}"
        ) from exc


def model_spread_bounds(values, spread: str = "iqr", axis=0):
    """Low and high inter-model spread bounds along an axis.

    Parameters
    ----------
    values : array_like
        Values with the models along ``axis``; NaNs are ignored.
    spread : {"iqr", "minmax", "tukey"}
        Spread convention.
    axis : int
        Model axis.

    Returns
    -------
    low, high : numpy.ndarray or float
        The bounds, with ``axis`` removed. For ``"iqr"`` and ``"minmax"`` the percentiles of
        :func:`spread_qpair`; for ``"tukey"`` the Tukey whiskers, i.e. the most extreme values
        inside ``[Q1 - 1.5 IQR, Q3 + 1.5 IQR]`` (not the fences themselves, so a band never
        extends past the data).
    """
    if spread != "tukey":
        low_percentile, high_percentile = spread_qpair(spread)
        return (np.nanpercentile(values, low_percentile, axis=axis),
                np.nanpercentile(values, high_percentile, axis=axis))

    array = np.asarray(values, dtype=float)
    q1 = np.nanpercentile(array, 25.0, axis=axis, keepdims=True)
    q3 = np.nanpercentile(array, 75.0, axis=axis, keepdims=True)
    fence = 1.5 * (q3 - q1)
    with np.errstate(invalid="ignore"):
        low = np.nanmin(np.where(array >= q1 - fence, array, np.nan), axis=axis)
        high = np.nanmax(np.where(array <= q3 + fence, array, np.nan), axis=axis)
    return low, high
