"""Per-chain warming-level means of the gridded PV and onshore-wind capacity factors.

For each warming level, every chain's capacity factors are averaged over that chain's own 20-year
window, annually or per season. Output: one file per level,
``<processed>/energy/analysis/<aggregation>/gwl/mean_gwl_<level x 10>_<turbine>[_<season type>].nc``,
with variables ``pv`` and ``wind`` on ``(model, lat, lon[, season])``. Wind uses ``--turbine``, by
default the paper's SWT120/3600. Hydropower and demand are per-country quantities and are not gridded here.

Examples
--------
    python compute_energy_means_gwl.py --aggregation seasonal --season_type meteorological
    python compute_energy_means_gwl.py --dry_run    # one chain, small extent, first level
"""
import argparse
import json
from pathlib import Path

import dask
import xarray as xr
from dask.diagnostics import ProgressBar

from cordex2energy.utils.constants import EUROPE_TURBINES
from cordex2energy.utils.utils_xarray import sel_extent
from cordex2energy_analysis.analysis import compute_seasonal_statistics
from cordex2energy_analysis.extremes_analysis import resolve_gwl_period_definitions

# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Compute per-model GWL-period gridded CF means")
parser.add_argument("--paths_file", type=str, default=str(Path(__file__).resolve().parent.parent / "paths.json"))
parser.add_argument("--models_json", type=str, default=str(Path(__file__).resolve().parent.parent / "models.json"))
parser.add_argument("--scenario", type=str, default="rcp85", choices=["rcp26", "rcp45", "rcp85"])
parser.add_argument("--aggregation", type=str, default="annual", choices=["annual", "seasonal"])
parser.add_argument("--season_type", type=str, default="meteorological", choices=["meteorological", "energy"])
parser.add_argument("--turbine", type=str, default="SWT120/3600", choices=EUROPE_TURBINES,
                    help="Wind turbine of the wind capacity factors (default: the paper's).")
parser.add_argument("--warming_levels", type=float, nargs="+", default=[1.5, 2.0, 3.0])
parser.add_argument("--dry_run", action="store_true", help="1 model, small extent, first GWL only — for validation")
args = parser.parse_args()

with open(args.paths_file) as handle:
    paths = json.load(handle)
path_data_processed = Path(paths["path_data_processed"])
climate_scenario = args.scenario
aggregation = args.aggregation
season_type = args.season_type
warming_levels = args.warming_levels[:1] if args.dry_run else args.warming_levels
selected_turbine = args.turbine

dask.config.set(num_workers=12, threads_per_worker=4)
dask.config.set({"array.chunk-size": "500MiB"})

path_energy_europe = path_data_processed / "energy" / "europe"
extent = (
    {"lat_min": 45, "lat_max": 50, "lon_min": 5, "lon_max": 12}
    if args.dry_run
    else {"lat_min": 30, "lat_max": 72.6, "lon_min": -12, "lon_max": 39}
)

#: The turbine name as it appears in the wind file names.
turbine_tag = selected_turbine.replace("/", "_")


def _pv_chain(stem: str) -> str:
    """Chain of a PV store, from its stem ``pv_<chain>_<scenario>``.

    Parameters
    ----------
    stem : str
        Store file stem.

    Returns
    -------
    str
    """
    return stem.split("pv_", 1)[1].rsplit("_", 1)[0]


def _wind_chain(stem: str) -> str:
    """Chain of a wind store, from its stem ``wind_power_<chain>_<scenario>_<turbine tag>``.

    Parameters
    ----------
    stem : str
        Store file stem.

    Returns
    -------
    str
    """
    return stem.split("wind_power_", 1)[1].split(f"_{turbine_tag}")[0].rsplit("_", 1)[0]


# ── Load gridded CF zarrs (PV, onshore wind) concatenated on model ─────────────
print("Loading PV capacity factors …")
path_pv_zarr = sorted((path_energy_europe / "pv").glob(f"*/*{climate_scenario}*.zarr"))
if args.dry_run:
    path_pv_zarr = path_pv_zarr[:1]
data_var_pv = list(xr.open_zarr(path_pv_zarr[0], chunks="auto").data_vars)[0]
pv = sel_extent(
    xr.concat([xr.open_zarr(path, chunks="auto") for path in path_pv_zarr],
              dim="model").rename({data_var_pv: "pv"}),
    extent=extent,
)
pv["model"] = [_pv_chain(path.stem) for path in path_pv_zarr]

print("Loading wind capacity factors …")
path_wind_zarr = sorted((path_energy_europe / "wind_power")
                        .glob(f"*/*{climate_scenario}*{turbine_tag}*.zarr"))
if args.dry_run:
    path_wind_zarr = path_wind_zarr[:1]
data_var_wind = list(xr.open_zarr(path_wind_zarr[0], chunks="auto").data_vars)[0]
wind = sel_extent(
    xr.concat([xr.open_zarr(path, chunks="auto") for path in path_wind_zarr],
              dim="model").rename({data_var_wind: "wind"}),
    extent=extent,
)
wind["model"] = [_wind_chain(path.stem) for path in path_wind_zarr]

models = list(pv["model"].values)
print(f"Models ({len(models)}): {models}")

# ── Per-model GWL windows ──────────────────────────────────────────────────────
period_windows, _, _ = resolve_gwl_period_definitions(
    warming_levels=warming_levels,
    models=models,
    chain_meta_path=args.models_json,
    gwl_yaml_path=path_data_processed / "cmip5_warming_levels_all_ens_1861_1900.yml",
    climate_scenario=climate_scenario,
)
print("GWL windows resolved:")
for label, windows in period_windows.items():
    print(f"  {label}: {len(windows)}/{len(models)} models")


def compute_temporal_mean(data, time_period, aggregation, season_type):
    """Climatological mean over one period.

    Parameters
    ----------
    data : xarray.Dataset
        Time series of one chain.
    time_period : tuple of str
        ``(first year, last year)``.
    aggregation : {"annual", "seasonal"}
        One mean, or one mean per season.
    season_type : {"meteorological", "energy"}
        Season definition (see
        :func:`~cordex2energy_analysis.analysis.compute_seasonal_statistics`).

    Returns
    -------
    xarray.Dataset
        The annual mean is the mean of the yearly means when the series is sub-annual.
    """
    period = data.sel(time=slice(*time_period))
    if aggregation == "seasonal":
        seasonal = compute_seasonal_statistics(period, time_dim="time", method="mean",
                                               season_type=season_type)
        if season_type == "meteorological":
            return seasonal.groupby("season").mean(dim="time", skipna=True).compute()
        return seasonal.mean(dim="year", skipna=True).compute()
    span = period.time[-1].dt.year.values - period.time[0].dt.year.values + 1
    if len(period.time) > span:
        period = period.resample(time="YE").mean(dim="time", skipna=True)
    return period.mean(dim="time", skipna=True).compute()


def gwl_ensemble_mean(data, label):
    """Every chain's mean over its own window at one warming level.

    Parameters
    ----------
    data : xarray.Dataset
        Time series with a ``model`` dimension.
    label : str
        Warming-level label, a key of ``period_windows``.

    Returns
    -------
    xarray.Dataset
        Chain means stacked on ``model``; chains without a window are skipped.
    """
    per_model = []
    for model in data["model"].values:
        window = period_windows[label].get(str(model))
        if window is None:
            print(f"    [skip {model}: no {label} window]")
            continue
        chain_mean = compute_temporal_mean(data.sel(model=model), window, aggregation, season_type)
        per_model.append(chain_mean.expand_dims(model=[model]))
    return xr.concat(per_model, dim="model")


# ── Compute + save one file per GWL ────────────────────────────────────────────
season_suffix = f"_{season_type}" if aggregation == "seasonal" else ""
out_dir = path_data_processed / "energy" / "analysis" / aggregation / "gwl"
out_dir.mkdir(parents=True, exist_ok=True)

for warming_level in warming_levels:
    label = f"+{warming_level:.1f}°C"
    tag = f"{int(round(warming_level * 10))}"  # 1.5 -> 15
    print(f"\nComputing GWL {label} (aggregation={aggregation}{', ' + season_type if aggregation == 'seasonal' else ''}) …")
    with ProgressBar(dt=5):
        mean_pv = gwl_ensemble_mean(pv, label)
        mean_wind = gwl_ensemble_mean(wind, label)
    output = xr.merge([mean_pv, mean_wind])
    out_path = out_dir / f"mean_gwl_{tag}_{turbine_tag}{season_suffix}.nc"
    if out_path.exists():
        out_path.unlink()
    output.to_netcdf(out_path)
    print(f"  Saved: {out_path}")

print("\nDone.")
