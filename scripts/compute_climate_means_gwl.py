"""Per-chain warming-level means of the bias-corrected EURO-CORDEX climate variables.

For each warming level, every chain's bias-corrected field is averaged over that chain's own
20-year window, annually or per season, and the chains are stacked on a ``model`` dimension.
Output: ``<processed>/energy/analysis/<aggregation>/gwl/mean_gwl_<level x 10>_climate[_<season
type>].nc``. The energy counterpart (PV and wind capacity factors) is
``compute_energy_means_gwl.py``.

Variables:

    tas      near-surface air temperature    (heating and cooling demand)
    rsds     surface downwelling shortwave   (PV)
    sfcWind  near-surface wind speed         (wind power)
    pr       precipitation                   (hydropower)
    mrro     total runoff                    (hydropower)
    Qrouted  routed river discharge (mRM)    (hydropower)

``tas``, ``rsds`` and ``sfcWind`` are read from the daily store where it exists and from the
3-hourly store otherwise; the two give the same period mean to float32 rounding. This holds for
the mean only: a statistic that is not linear in time must read the 3-hourly store.

Each variable carries a ``change_type`` attribute, ``"absolute"`` (``tas``) or ``"percent"``,
which the figures use to express the change between levels.

``mrro`` is on a different grid from the other variables (cell centres offset by half a cell), so
every period mean is interpolated onto the 3-hourly ``tas`` grid of the first chain, and the
variables are merged with ``join="exact"``.

Examples
--------
    python compute_climate_means_gwl.py --dry_run    # one chain, small extent, first level
    python compute_climate_means_gwl.py
    python compute_climate_means_gwl.py --aggregation seasonal
"""
import argparse
import glob
import json
from pathlib import Path

import dask
import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar

from cordex2energy.utils.utils_xarray import sel_extent
from cordex2energy_analysis.analysis import compute_seasonal_statistics
from cordex2energy_analysis.extremes_analysis import resolve_gwl_period_definitions

# ── Variable spec ──────────────────────────────────────────────────────────────
# "resolution": bias-corrected stores to try, in order of preference. "source": "routed" for the
# mRM discharge, read by open_routed_runoff. "change_type": how the change between levels is
# expressed.
VARIABLES = {
    "tas":     {"resolution": ["day", "3hr"], "change_type": "absolute", "long_name": "Near-surface air temperature"},
    "rsds":    {"resolution": ["day", "3hr"], "change_type": "percent",  "long_name": "Surface downwelling shortwave radiation"},
    "sfcWind": {"resolution": ["day", "3hr"], "change_type": "percent",  "long_name": "Near-surface wind speed"},
    "pr":      {"resolution": ["day"],        "change_type": "percent",  "long_name": "Precipitation"},
    "mrro":    {"resolution": ["day"],        "change_type": "percent",  "long_name": "Total runoff"},
    # River-channel discharge: the bias-corrected mrro passed through the mRM routing model.
    "Qrouted": {"source": "routed",           "change_type": "percent",  "long_name": "Routed runoff (mRM)"},
}

# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Per-model GWL gridded means of the bias-corrected climate variables")
parser.add_argument("--paths_file", type=str, default=str(Path(__file__).resolve().parent.parent / "paths.json"))
parser.add_argument("--models_json", type=str, default=str(Path(__file__).resolve().parent.parent / "models.json"))
parser.add_argument("--scenario", type=str, default="rcp85", choices=["rcp26", "rcp45", "rcp85"])
parser.add_argument("--aggregation", type=str, default="annual", choices=["annual", "seasonal"])
parser.add_argument("--season_type", type=str, default="meteorological", choices=["meteorological", "energy"])
parser.add_argument("--warming_levels", type=float, nargs="+", default=[1.5, 2.0, 3.0])
parser.add_argument("--variables", type=str, nargs="+", default=list(VARIABLES),
                    choices=list(VARIABLES))
parser.add_argument("--dry_run", action="store_true", help="1 model, small extent, first GWL only — for validation")
args = parser.parse_args()

with open(args.paths_file) as handle:
    paths = json.load(handle)
path_data_processed = Path(paths["path_data_processed"])
climate_scenario = args.scenario
aggregation = args.aggregation
season_type = args.season_type
warming_levels = args.warming_levels[:1] if args.dry_run else args.warming_levels

dask.config.set(num_workers=12, threads_per_worker=4)
dask.config.set({"array.chunk-size": "500MiB"})

path_cordex = path_data_processed / "cordex_processed"
extent = (
    {"lat_min": 45, "lat_max": 50, "lon_min": 5, "lon_max": 12}
    if args.dry_run
    else {"lat_min": 30, "lat_max": 72.6, "lon_min": -12, "lon_max": 39}   # matches the energy means
)

# ── Discover the chains ────────────────────────────────────────────────────────
chain_dirs = sorted(folder for folder in path_cordex.glob(f"*_{climate_scenario}")
                    if folder.is_dir())
models = [folder.name[: -len(f"_{climate_scenario}")] for folder in chain_dirs]
if args.dry_run:
    chain_dirs, models = chain_dirs[:1], models[:1]
print(f"Chains ({len(models)}):")
for model in models:
    print(f"  {model}")

# ── Per-model GWL windows ──────────────────────────────────────────────────────
period_windows, _, _ = resolve_gwl_period_definitions(
    warming_levels=warming_levels,
    models=models,
    chain_meta_path=args.models_json,
    gwl_yaml_path=path_data_processed / "cmip5_warming_levels_all_ens_1861_1900.yml",
    climate_scenario=climate_scenario,
)
print("\nGWL windows resolved:")
for label, windows in period_windows.items():
    print(f"  {label}: {len(windows)}/{len(models)} chains")


#: Store each (chain, variable) was read from: "day", "3hr" or "routed".
USED_STORE: dict[tuple[str, str], str] = {}

#: Root of the mRM routed-runoff output (per-chain folders of 5-year sub-folders).
PATH_ROUTED = Path("/net/land/landclim/pseubert/out_rcm/hist")


def open_routed_runoff(model: str) -> xr.DataArray:
    """One chain's routed river discharge (mRM), on the standard calendar, cropped to the extent.

    Parameters
    ----------
    model : str
        Chain identifier.

    Returns
    -------
    xarray.DataArray
        Daily ``Qrouted``, the 5-year stores concatenated along time without duplicates.

    Notes
    -----
    Stores named ``Qrouted_*_new.zarr`` are preferred; any ``Qrouted_*.zarr`` is the fallback.
    """
    base = PATH_ROUTED / f"{model}_{climate_scenario}"
    stores = sorted(glob.glob(str(base / "*" / "Qrouted_*_new.zarr"))
                    or glob.glob(str(base / "*" / "Qrouted_*.zarr")))
    if not stores:
        raise FileNotFoundError(f"no Qrouted zarr under {base}")
    parts = [sel_extent(xr.open_zarr(store, chunks="auto").sortby("lat"), extent=extent)["Qrouted"]
             for store in stores]
    discharge = xr.concat(parts, dim="time").sortby("time").drop_duplicates("time")
    discharge = discharge.convert_calendar("standard", align_on="year")
    USED_STORE[(model, "Qrouted")] = "routed"
    return discharge


def open_variable(model: str, variable: str) -> xr.DataArray:
    """One chain's variable, on the standard calendar, cropped to the extent.

    Parameters
    ----------
    model : str
        Chain identifier.
    variable : str
        Key of :data:`VARIABLES`.

    Returns
    -------
    xarray.DataArray
        The variable from the first existing store of its ``"resolution"`` list (or the routed
        discharge), converted from the stores' no-leap calendar to the standard one.

    Raises
    ------
    FileNotFoundError
        If no store exists.
    """
    if VARIABLES[variable].get("source") == "routed":
        return open_routed_runoff(model)
    base = path_cordex / f"{model}_{climate_scenario}" / "bias_corrected"
    tried = [base / resolution / f"{variable}.zarr"
             for resolution in VARIABLES[variable]["resolution"]]
    store = next((candidate for candidate in tried if candidate.exists()), None)
    if store is None:
        raise FileNotFoundError(
            f"no store for {variable} / {model} — tried {[str(path) for path in tried]}")
    USED_STORE[(model, variable)] = store.parent.name
    dataset = xr.open_zarr(store, chunks="auto")
    dataset = dataset.convert_calendar("standard", align_on="year").sortby("lat")
    return sel_extent(dataset, extent=extent)[variable]


# One reference grid for every variable: tas/rsds/sfcWind/pr already share it, mrro does not.
_reference_grid = sel_extent(xr.open_zarr(path_cordex / f"{models[0]}_{climate_scenario}"
                                          / "bias_corrected" / "3hr" / "tas.zarr", chunks="auto"),
                             extent=extent)
REF_LAT, REF_LON = _reference_grid["lat"], _reference_grid["lon"]
print(f"\nReference grid: lat[{REF_LAT.size}] lon[{REF_LON.size}]")


def to_reference_grid(field: xr.DataArray) -> xr.DataArray:
    """Put a period-mean field on the reference grid.

    Parameters
    ----------
    field : xarray.DataArray
        Field on ``(lat, lon)``.

    Returns
    -------
    xarray.DataArray
        The field with the reference coordinates assigned if its grid already matches (to
        floating-point tolerance), else linearly interpolated onto the reference grid.
    """
    same = (field.sizes.get("lat") == REF_LAT.size and field.sizes.get("lon") == REF_LON.size
            and np.allclose(field["lat"].values, REF_LAT.values)
            and np.allclose(field["lon"].values, REF_LON.values))
    if same:
        return field.assign_coords(lat=REF_LAT, lon=REF_LON)   # drop float noise, keep the values
    return field.interp(lat=REF_LAT, lon=REF_LON, method="linear")


def temporal_mean(series: xr.DataArray, window) -> xr.DataArray:
    """Climatological mean over one warming-level window.

    Parameters
    ----------
    series : xarray.DataArray
        Time series of one chain.
    window : tuple of str
        ``(first year, last year)``.

    Returns
    -------
    xarray.DataArray
        With ``--aggregation annual``, the mean of the yearly means (every year weighs the same);
        with ``seasonal``, one mean per season (see
        :func:`~cordex2energy_analysis.analysis.compute_seasonal_statistics`).
    """
    series = series.sel(time=slice(*window))
    if aggregation == "seasonal":
        seasonal = compute_seasonal_statistics(series.to_dataset(), time_dim="time", method="mean",
                                               season_type=season_type)
        if season_type == "meteorological":
            return seasonal.groupby("season").mean(dim="time", skipna=True).compute()
        return seasonal.mean(dim="year", skipna=True).compute()
    return series.resample(time="YE").mean("time", skipna=True).mean("time", skipna=True).compute()


# ── Compute + save one file per GWL ────────────────────────────────────────────
season_suffix = f"_{season_type}" if aggregation == "seasonal" else ""
out_dir = path_data_processed / "energy" / "analysis" / aggregation / "gwl"
out_dir.mkdir(parents=True, exist_ok=True)

for warming_level in warming_levels:
    label = f"+{warming_level:.1f}°C"
    tag = f"{int(round(warming_level * 10))}"
    print(f"\n=== GWL {label} ({aggregation}"
          f"{', ' + season_type if aggregation == 'seasonal' else ''}) ===")

    per_variable = []
    for variable in args.variables:
        source = VARIABLES[variable].get("resolution", VARIABLES[variable].get("source", "?"))
        print(f"  {variable} ({source}) …")
        per_model = []
        with ProgressBar(dt=10):
            # Per chain, not concatenate-then-slice: each chain has its own window, and
            # concatenating first would align time coordinates across chains.
            for model in models:
                window = period_windows[label].get(model)
                if window is None:
                    print(f"    [skip {model}: no {label} window]")
                    continue
                chain_mean = to_reference_grid(temporal_mean(open_variable(model, variable), window))
                per_model.append(chain_mean.expand_dims(model=[model]))

        merged = xr.concat(per_model, dim="model")
        if isinstance(merged, xr.Dataset):
            merged = merged[variable]
        merged.attrs.update(
            change_type=VARIABLES[variable]["change_type"],
            long_name=VARIABLES[variable]["long_name"],
            native_resolution=str(VARIABLES[variable].get("resolution",
                                                          VARIABLES[variable].get("source"))),
        )
        per_variable.append(merged.rename(variable))

        n_chains = merged.sizes["model"]
        if not args.dry_run:
            assert n_chains == len(models), \
                f"{variable}: only {n_chains}/{len(models)} chains — a store is missing"
        stores = [USED_STORE[(model, variable)] for model in models
                  if (model, variable) in USED_STORE]
        detail = (f"{stores.count('routed')} routed" if "routed" in stores
                  else f"{stores.count('day')} day / {stores.count('3hr')} 3hr")
        print(f"    -> {n_chains} chains ({detail}), mean = {float(merged.mean()):.3f}")

    # join="exact": a grid mismatch raises instead of padding with NaN.
    output = xr.merge(per_variable, join="exact")
    output.attrs.update(
        description="Per-model GWL-window means of the bias-corrected EURO-CORDEX climate variables",
        warming_level=label,
        scenario=climate_scenario,
    )
    out_path = out_dir / f"mean_gwl_{tag}_climate{season_suffix}.nc"
    if out_path.exists():
        out_path.unlink()
    output.to_netcdf(out_path)
    print(f"  Saved: {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")

print("\nDone.")
