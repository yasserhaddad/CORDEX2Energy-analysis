"""Compute per-model GWL-period gridded means of PV and onshore-wind capacity factors.

Adapted from ``compute_energy_means.py``: instead of a single fixed calendar period
applied to every model, this slices **each model's own 20-year Global Warming Level
window** (+1.5 / +2.0 / +3.0 °C, resolved per driving GCM) before averaging. Outputs one
NetCDF per GWL with dims ``(model, lat, lon[, season])`` and variables ``pv``, ``wind`` —
consumed by ``notebooks/analyse_energy_gwl.ipynb`` (Part 1, View A gridded maps).

Only PV + onshore wind are handled here (the gridded resource). Hydro and demand come from
the c2e country-level data in the notebook.

Example:
    python compute_energy_means_gwl.py --aggregation seasonal --season_type meteorological
    python compute_energy_means_gwl.py --dry_run          # 1 model, small extent, +1.5 only
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
parser.add_argument("--warming_levels", type=float, nargs="+", default=[1.5, 2.0, 3.0])
parser.add_argument("--dry_run", action="store_true", help="1 model, small extent, first GWL only — for validation")
args = parser.parse_args()

paths = json.load(open(args.paths_file))
path_data_processed = Path(paths["path_data_processed"])
climate_scenario = args.scenario
aggregation = args.aggregation
season_type = args.season_type
warming_levels = args.warming_levels[:1] if args.dry_run else args.warming_levels
selected_turbine = EUROPE_TURBINES[0]  # only one turbine for now, but could be extended to multiple

dask.config.set(num_workers=12, threads_per_worker=4)
dask.config.set({"array.chunk-size": "500MiB"})

path_energy_europe = path_data_processed / "energy" / "europe"
extent = (
    {"lat_min": 45, "lat_max": 50, "lon_min": 5, "lon_max": 12}
    if args.dry_run
    else {"lat_min": 30, "lat_max": 72.6, "lon_min": -12, "lon_max": 39}
)


_turb = selected_turbine.replace("/", "_")


def _pv_chain(stem: str) -> str:
    # 'pv_<chain>_<rcp>' -> '<chain>'
    return stem.split("pv_", 1)[1].rsplit("_", 1)[0]


def _wind_chain(stem: str) -> str:
    # 'wind_power_<chain>_<rcp>_<turbine_fn>' -> '<chain>'  (turbine_fn contains '_')
    return stem.split("wind_power_", 1)[1].split(f"_{_turb}")[0].rsplit("_", 1)[0]


# ── Load gridded CF zarrs (PV, onshore wind) concatenated on model ─────────────
print("Loading PV capacity factors …")
path_pv_zarr = sorted((path_energy_europe / "pv").glob(f"*/*{climate_scenario}*.zarr"))
if args.dry_run:
    path_pv_zarr = path_pv_zarr[:1]
data_var_pv = list(xr.open_zarr(path_pv_zarr[0], chunks="auto").data_vars)[0]
ds_pv = sel_extent(
    xr.concat([xr.open_zarr(p, chunks="auto") for p in path_pv_zarr], dim="model").rename({data_var_pv: "pv"}),
    extent=extent,
)
ds_pv["model"] = [_pv_chain(p.stem) for p in path_pv_zarr]

print("Loading wind capacity factors …")
path_wind_zarr = sorted((path_energy_europe / "wind_power").glob(f"*/*{climate_scenario}*{_turb}*.zarr"))
if args.dry_run:
    path_wind_zarr = path_wind_zarr[:1]
data_var_wind = list(xr.open_zarr(path_wind_zarr[0], chunks="auto").data_vars)[0]
ds_wind = sel_extent(
    xr.concat([xr.open_zarr(p, chunks="auto") for p in path_wind_zarr], dim="model").rename({data_var_wind: "wind"}),
    extent=extent,
)
ds_wind["model"] = [_wind_chain(p.stem) for p in path_wind_zarr]

models = list(ds_pv["model"].values)
print(f"Models ({len(models)}): {models}")

# ── Per-model GWL windows ──────────────────────────────────────────────────────
period_defs, _, _ = resolve_gwl_period_definitions(
    warming_levels=warming_levels,
    models=models,
    chain_meta_path=args.models_json,
    gwl_yaml_path=path_data_processed / "cmip5_warming_levels_all_ens_1861_1900.yml",
    climate_scenario=climate_scenario,
)
print("GWL windows resolved:")
for lbl, w in period_defs.items():
    print(f"  {lbl}: {len(w)}/{len(models)} models")


def compute_temporal_mean(ds, time_period, aggregation, season_type):
    """Single climatological mean over ``time_period``; annual or per-season (period scope)."""
    ds_period = ds.sel(time=slice(*time_period))
    if aggregation == "seasonal":
        ds_seasonal = compute_seasonal_statistics(ds_period, time_dim="time", method="mean", season_type=season_type)
        if season_type == "meteorological":
            return ds_seasonal.groupby("season").mean(dim="time", skipna=True).compute()
        return ds_seasonal.mean(dim="year", skipna=True).compute()
    # annual — reduce to yearly then average over years
    span = ds_period.time[-1].dt.year.values - ds_period.time[0].dt.year.values + 1
    if len(ds_period.time) > span:
        ds_period = ds_period.resample(time="YE").mean(dim="time", skipna=True)
    return ds_period.mean(dim="time", skipna=True).compute()


def gwl_ensemble_mean(ds, label):
    """Per-model GWL-window mean, concatenated back on the model dim."""
    per_model = []
    for m in ds["model"].values:
        win = period_defs[label].get(str(m))
        if win is None:
            print(f"    [skip {m}: no {label} window]")
            continue
        mean_m = compute_temporal_mean(ds.sel(model=m), win, aggregation, season_type)
        per_model.append(mean_m.expand_dims(model=[m]))
    return xr.concat(per_model, dim="model")


# ── Compute + save one file per GWL ────────────────────────────────────────────
season_suffix = f"_{season_type}" if aggregation == "seasonal" else ""
out_dir = path_data_processed / "energy" / "analysis" / aggregation / "gwl"
out_dir.mkdir(parents=True, exist_ok=True)

for wl in warming_levels:
    label = f"+{wl:.1f}°C"
    tag = f"{int(round(wl * 10))}"  # 1.5 -> 15
    print(f"\nComputing GWL {label} (aggregation={aggregation}{', ' + season_type if aggregation == 'seasonal' else ''}) …")
    with ProgressBar(dt=5):
        mean_pv = gwl_ensemble_mean(ds_pv, label)
        mean_wind = gwl_ensemble_mean(ds_wind, label)
    ds_out = xr.merge([mean_pv, mean_wind])
    out_path = out_dir / f"mean_gwl_{tag}{season_suffix}.nc"
    if out_path.exists():
        out_path.unlink()
    ds_out.to_netcdf(out_path)
    print(f"  Saved: {out_path}")

print("\nDone.")
