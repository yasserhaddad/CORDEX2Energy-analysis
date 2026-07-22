"""Compute per-model GWL-period gridded means of the BIAS-CORRECTED EURO-CORDEX climate variables.

The sibling script ``compute_energy_means_gwl.py`` does this for the derived energy fields (PV and
wind capacity factors).  This one does it for the *climate* variables those fields are built from, so
the supplementary material can show the underlying signal.

Why a script and not a notebook cell: ``cordex_processed`` is ~53 TB, and a single bias-corrected
3-hourly variable is 166-270 GB **per model** (0.1 deg, lat 426 x lon 710, 1991-2100).  A 9-model x
3-GWL x 20-year mean is hundreds of GB of reads per variable.  The output is ~23 MB (annual) /
~94 MB (seasonal) per GWL, which a notebook then loads instantly.

Variables — the five that drive every conversion in the pipeline:

    tas      3-hourly   near-surface air temperature   -> heating / cooling demand
    rsds     3-hourly   surface downwelling shortwave  -> PV
    sfcWind  3-hourly   near-surface wind speed        -> wind power
    pr       daily      precipitation                  -> hydro
    mrro     daily      total runoff                   -> hydro

``tas``, ``rsds`` and ``sfcWind`` exist in both a daily and a 3-hourly store.  We read the **daily** one
where it exists and fall back to 3-hourly only for the two chains that lack it (MOHC-HadREM3-GA7-05 x
MPI-ESM-LR and x NorESM1-M), so all nine chains are always present — the script asserts this.

That substitution is safe *because we only take a mean*: verified on a chain that has both stores, the
period-mean field is identical to float32 rounding (max |diff| = 2e-4 K for tas, 1e-4 W m-2 for rsds,
2e-6 m s-1 for sfcWind).  It is worth ~8x less I/O (daily tas is 24 GB/model, 3-hourly is 191 GB).
NOTE: this equivalence holds for the MEAN only.  Any future statistic that is not linear in time
(a quantile, an exceedance count, a diurnal cycle) MUST read the 3-hourly store.

Each variable carries a ``change_type`` attribute (``"absolute"`` or ``"percent"``) that the plotting
notebook MUST honour: a percentage change in Kelvin is meaningless, so ``tas`` is absolute.

``mrro`` is stored on a DIFFERENT grid from the others (lat 20.90-71.80, lon -51.70-72.70, cell centres
offset by half a cell — zero coordinates in common with tas/rsds/sfcWind/pr).  Merging naively would
silently produce a union grid on which the variables never overlap.  Every field is therefore
interpolated onto one reference grid after the temporal mean (cheap: the climatology is 2-D), and the
final merge uses ``join="exact"`` so any future grid mismatch fails loudly instead of padding with NaN.

Example:
    python compute_climate_means_gwl.py --dry_run                 # 1 model, small extent, +1.5 only
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
# resolution: which bias_corrected store to read.  change_type: how the notebook must display the
# change vs the reference GWL (a % change in Kelvin is meaningless).
# "resolution": stores to try, in order of preference (daily first: 8x less I/O, identical mean).
VARIABLES = {
    "tas":     {"resolution": ["day", "3hr"], "change_type": "absolute", "long_name": "Near-surface air temperature"},
    "rsds":    {"resolution": ["day", "3hr"], "change_type": "percent",  "long_name": "Surface downwelling shortwave radiation"},
    "sfcWind": {"resolution": ["day", "3hr"], "change_type": "percent",  "long_name": "Near-surface wind speed"},
    "pr":      {"resolution": ["day"],        "change_type": "percent",  "long_name": "Precipitation"},
    "mrro":    {"resolution": ["day"],        "change_type": "percent",  "long_name": "Total runoff"},
    # Routed runoff: the bias-corrected mrro passed through the mRM routing model — river-channel
    # discharge, not grid-cell runoff. Lives in a separate tree (5-year folders, global 0.1 deg daily),
    # so it is read by open_routed_runoff, not from cordex_processed/bias_corrected.
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

paths = json.load(open(args.paths_file))
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
chain_dirs = sorted(p for p in path_cordex.glob(f"*_{climate_scenario}") if p.is_dir())
models = [p.name[: -len(f"_{climate_scenario}")] for p in chain_dirs]
if args.dry_run:
    chain_dirs, models = chain_dirs[:1], models[:1]
print(f"Chains ({len(models)}):")
for m in models:
    print(f"  {m}")

# ── Per-model GWL windows ──────────────────────────────────────────────────────
period_defs, _, _ = resolve_gwl_period_definitions(
    warming_levels=warming_levels,
    models=models,
    chain_meta_path=args.models_json,
    gwl_yaml_path=path_data_processed / "cmip5_warming_levels_all_ens_1861_1900.yml",
    climate_scenario=climate_scenario,
)
print("\nGWL windows resolved:")
for label, windows in period_defs.items():
    print(f"  {label}: {len(windows)}/{len(models)} chains")


USED_STORE: dict[tuple[str, str], str] = {}   # (model, variable) -> "day" | "3hr" | "routed"

# Routed runoff lives outside cordex_processed, in per-chain 5-year folders.
PATH_ROUTED = Path("/net/land/landclim/pseubert/out_rcm/hist")


def open_routed_runoff(model: str) -> xr.DataArray:
    """One chain's routed runoff (mRM): global daily river discharge, concatenated across the 5-year
    folders and cropped to the extent.

    Not in cordex_processed — the mRM output lives in its own tree, split into 5-year subfolders, on a
    global 0.1 deg grid.  Cropping on open keeps only the European window (the field is otherwise
    global).  The '_new' suffix matches preprocess_cordex.py; fall back to any Qrouted zarr otherwise.
    """
    base = PATH_ROUTED / f"{model}_{climate_scenario}"
    stores = sorted(glob.glob(str(base / "*" / "Qrouted_*_new.zarr"))
                    or glob.glob(str(base / "*" / "Qrouted_*.zarr")))
    if not stores:
        raise FileNotFoundError(f"no Qrouted zarr under {base}")
    parts = [sel_extent(xr.open_zarr(s, chunks="auto").sortby("lat"), extent=extent)["Qrouted"]
             for s in stores]
    da = xr.concat(parts, dim="time").sortby("time").drop_duplicates("time")
    da = da.convert_calendar("standard", align_on="year")
    USED_STORE[(model, "Qrouted")] = "routed"
    return da


def open_variable(model: str, variable: str) -> xr.DataArray:
    """One chain's variable, on the standard calendar, cropped to the extent.

    The bias-corrected stores are on a 365-day (no-leap) calendar; converting to standard here means
    the GWL window slices and the season grouping behave like every other dataset in the analysis.
    """
    if VARIABLES[variable].get("source") == "routed":
        return open_routed_runoff(model)
    base = path_cordex / f"{model}_{climate_scenario}" / "bias_corrected"
    tried = [base / res / f"{variable}.zarr" for res in VARIABLES[variable]["resolution"]]
    store = next((s for s in tried if s.exists()), None)
    if store is None:
        raise FileNotFoundError(
            f"no store for {variable} / {model} — tried {[str(t) for t in tried]}")
    USED_STORE[(model, variable)] = store.parent.name
    ds = xr.open_zarr(store, chunks="auto")
    ds = ds.convert_calendar("standard", align_on="year").sortby("lat")
    return sel_extent(ds, extent=extent)[variable]


# One reference grid for every variable.  tas/rsds/sfcWind/pr already share it; mrro does not.
_ref = sel_extent(xr.open_zarr(path_cordex / f"{models[0]}_{climate_scenario}"
                               / "bias_corrected" / "3hr" / "tas.zarr", chunks="auto"),
                  extent=extent)
REF_LAT, REF_LON = _ref["lat"], _ref["lon"]
print(f"\nReference grid: lat[{REF_LAT.size}] lon[{REF_LON.size}]")


def to_reference_grid(da: xr.DataArray) -> xr.DataArray:
    """Put a period-mean field on the reference grid (a no-op for the four that already are)."""
    same = (da.sizes.get("lat") == REF_LAT.size and da.sizes.get("lon") == REF_LON.size
            and np.allclose(da["lat"].values, REF_LAT.values)
            and np.allclose(da["lon"].values, REF_LON.values))
    if same:
        return da.assign_coords(lat=REF_LAT, lon=REF_LON)   # kill float noise, keep the values
    return da.interp(lat=REF_LAT, lon=REF_LON, method="linear")


def temporal_mean(da: xr.DataArray, window) -> xr.DataArray:
    """Climatological mean over one GWL window — annual, or one value per season."""
    da = da.sel(time=slice(*window))
    if aggregation == "seasonal":
        seasonal = compute_seasonal_statistics(da.to_dataset(), time_dim="time", method="mean",
                                               season_type=season_type)
        if season_type == "meteorological":
            return seasonal.groupby("season").mean(dim="time", skipna=True).compute()
        return seasonal.mean(dim="year", skipna=True).compute()
    # annual: reduce to yearly means first so every year weighs the same, then average the years
    return da.resample(time="YE").mean("time", skipna=True).mean("time", skipna=True).compute()


# ── Compute + save one file per GWL ────────────────────────────────────────────
season_suffix = f"_{season_type}" if aggregation == "seasonal" else ""
out_dir = path_data_processed / "energy" / "analysis" / aggregation / "gwl"
out_dir.mkdir(parents=True, exist_ok=True)

for wl in warming_levels:
    label = f"+{wl:.1f}°C"
    tag = f"{int(round(wl * 10))}"
    print(f"\n=== GWL {label} ({aggregation}"
          f"{', ' + season_type if aggregation == 'seasonal' else ''}) ===")

    per_variable = []
    for variable in args.variables:
        source = VARIABLES[variable].get("resolution", VARIABLES[variable].get("source", "?"))
        print(f"  {variable} ({source}) …")
        per_model = []
        with ProgressBar(dt=10):
            # Deliberately per-model rather than concat-then-slice: each chain has its own GWL window,
            # and concatenating first would force an alignment of time coords across chains that the
            # calendar conversion does not guarantee.
            for model in models:
                window = period_defs[label].get(model)
                if window is None:
                    print(f"    [skip {model}: no {label} window]")
                    continue
                mean_m = to_reference_grid(temporal_mean(open_variable(model, variable), window))
                per_model.append(mean_m.expand_dims(model=[model]))

        merged = xr.concat(per_model, dim="model")
        if isinstance(merged, xr.Dataset):
            merged = merged[variable]
        merged.attrs.update(
            change_type=VARIABLES[variable]["change_type"],       # the notebook MUST honour this
            long_name=VARIABLES[variable]["long_name"],
            native_resolution=str(VARIABLES[variable].get("resolution",
                                                          VARIABLES[variable].get("source"))),
        )
        per_variable.append(merged.rename(variable))

        n = merged.sizes["model"]
        if not args.dry_run:
            assert n == len(models), f"{variable}: only {n}/{len(models)} chains — a store is missing"
        stores = [USED_STORE[(m, variable)] for m in models if (m, variable) in USED_STORE]
        detail = (f"{stores.count('routed')} routed" if "routed" in stores
                  else f"{stores.count('day')} day / {stores.count('3hr')} 3hr")
        print(f"    -> {n} chains ({detail}), mean = {float(merged.mean()):.3f}")

    # join="exact": a grid mismatch must raise, not silently NaN-pad (see the mrro note above)
    ds_out = xr.merge(per_variable, join="exact")
    ds_out.attrs.update(
        description="Per-model GWL-window means of the bias-corrected EURO-CORDEX climate variables",
        warming_level=label,
        scenario=climate_scenario,
    )
    out_path = out_dir / f"mean_gwl_{tag}_climate{season_suffix}.nc"
    if out_path.exists():
        out_path.unlink()
    ds_out.to_netcdf(out_path)
    print(f"  Saved: {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")

print("\nDone.")
