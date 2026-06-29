import argparse
import json
from pathlib import Path

import dask
import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar

from cordex2energy.utils.constants import (
    CH_TURBINES,
    END_CENTURY_PERIOD,
    EUROPE_TURBINES,
    EXTENT_CH,
    MID_CENTURY_PERIOD,
    REFERENCE_PERIOD,
)
from cordex2energy.utils.utils_xarray import sel_extent
from cordex2energy_analysis.analysis import (
    compute_doy_threshold,
    compute_exceedance_frequency,
    ensure_daily,
)

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Compute per-DOY percentile thresholds and exceedance frequencies for energy data"
)
parser.add_argument(
    "--paths_file",
    type=str,
    default="../../paths_so4.json",
    help="Path to the JSON file containing data paths (default: ../../paths_so4.json)",
)
parser.add_argument(
    "--scenario",
    type=str,
    default="rcp85",
    choices=["rcp26", "rcp45", "rcp85"],
    help="Climate scenario to process (default: rcp85)",
)
parser.add_argument(
    "--percentile",
    type=float,
    default=10.0,
    help=(
        "Percentile threshold for extremes (default: 10.0). "
        "Use 10 for low-supply extremes (PV, wind, hydro); "
        "90 for high-demand extremes (HDD, CDD)."
    ),
)
parser.add_argument(
    "--window",
    type=int,
    default=15,
    help=(
        "Half-width in days of the centred DOY window used to pool reference samples "
        "(default: 15). Each DOY threshold is estimated from all values within "
        "[DOY-window, DOY+window] across all reference years, giving "
        "(2*window+1)*n_ref_years ≈ 930 samples per bin."
    ),
)

args = parser.parse_args()

paths = json.load(open(args.paths_file))
path_data_processed = Path(paths["path_data_processed"])

dask.config.set(num_workers=12, threads_per_worker=4)
dask.config.set({"array.chunk-size": "500MiB"})

climate_scenario = args.scenario
percentile = args.percentile
window = args.window
selected_turbine_europe = EUROPE_TURBINES[0]
selected_turbine_ch = CH_TURBINES[0]

path_energy_europe = path_data_processed / "energy" / "europe"
path_energy_ch = path_data_processed / "energy" / "ch"
extent = {"lat_min": 30, "lat_max": 72.6, "lon_min": -12, "lon_max": 39}


# ── Data loading (mirrors compute_energy_means.py exactly) ───────────────────

# PV
path_pv = path_energy_europe / "pv"
path_pv_zarr = sorted(path_pv.glob(f"*/*{climate_scenario}*.zarr"))
data_var_pv = list(xr.open_zarr(path_pv_zarr[0], chunks="auto").data_vars.keys())[0]
ds_pv = sel_extent(
    xr.concat(
        [xr.open_zarr(path, chunks="auto") for path in path_pv_zarr], dim="model"
    ).rename({data_var_pv: "pv"}),
    extent=extent,
)
ds_pv["model"] = [path.stem.split("pv_")[1] for path in path_pv_zarr]

# Wind (Europe)
path_wind = path_energy_europe / "wind_power"
path_wind_zarr = sorted(
    path_wind.glob(
        f"*/*{climate_scenario}*{selected_turbine_europe.replace('/', '_')}*.zarr"
    )
)
data_var_wind = list(xr.open_zarr(path_wind_zarr[0], chunks="auto").data_vars.keys())[0]
ds_wind = sel_extent(
    xr.concat(
        [xr.open_zarr(path, chunks="auto") for path in path_wind_zarr], dim="model"
    ).rename({data_var_wind: "wind"}),
    extent=extent,
)
ds_wind["model"] = [
    path.stem.split("wind_power_")[1].split(
        f"_{selected_turbine_europe.replace('/', '_')}"
    )[0]
    for path in path_wind_zarr
]

# Wind (CH)
path_wind_ch = path_energy_ch / "wind_power"
path_wind_ch_zarr = sorted(path_wind_ch.glob(f"*{climate_scenario}*.zarr"))
ds_wind_ch = xr.concat(
    [
        xr.open_zarr(path, chunks="auto")[[selected_turbine_ch.replace("/", "_")]]
        for path in path_wind_ch_zarr
    ],
    dim="model",
)
ds_wind_ch["model"] = [path.stem.split("wind_power_")[1] for path in path_wind_ch_zarr]
data_var_wind_ch = list(ds_wind_ch.data_vars.keys())[0]
ds_wind_ch = ds_wind_ch.rename({data_var_wind_ch: "wind"})

# Hydropower run-of-river
path_hydropower_ror = path_energy_europe / "hydropower"
path_hydropower_ror_zarr = sorted(
    path_hydropower_ror.glob(f"*/hydropower_ror_*{climate_scenario}*.zarr")
)
ds_hydropower_ror = sel_extent(
    xr.concat(
        [
            xr.open_zarr(path, chunks="auto")
            .convert_calendar("standard", align_on="year")
            .sortby("lat", ascending=True)
            for path in path_hydropower_ror_zarr
        ],
        dim="model",
    ),
    extent=extent,
)
ds_hydropower_ror["lat"] = ds_pv["lat"]
ds_hydropower_ror["lon"] = ds_pv["lon"]
ds_hydropower_ror["model"] = [
    path.stem.split("hydropower_ror_")[1] for path in path_hydropower_ror_zarr
]

# Hydropower reservoir
path_hydropower_reservoir = path_energy_europe / "hydropower"
path_hydropower_reservoir_zarr = sorted(
    path_hydropower_reservoir.glob(f"*/hydropower_reservoir_*{climate_scenario}*.zarr")
)
ds_hydropower_reservoir = sel_extent(
    xr.concat(
        [
            xr.open_zarr(path, chunks="auto")
            .convert_calendar("standard", align_on="year")
            .sortby("lat", ascending=True)
            for path in path_hydropower_reservoir_zarr
        ],
        dim="model",
    ),
    extent=extent,
)
ds_hydropower_reservoir["lat"] = ds_pv["lat"]
ds_hydropower_reservoir["lon"] = ds_pv["lon"]
ds_hydropower_reservoir["model"] = [
    path.stem.split("hydropower_reservoir_")[1] for path in path_hydropower_reservoir_zarr
]

# Demand (HDD / CDD)
path_demand = path_energy_europe / "demand"
paths_demand_zarr = sorted(path_demand.glob(f"*{climate_scenario}.zarr"))
ds_cdd_hdd = sel_extent(
    xr.concat(
        [
            xr.open_zarr(path, chunks="auto").isel(band=0).drop_duplicates(dim="time")
            for path in paths_demand_zarr
        ],
        dim="model",
    ),
    extent=extent,
)
ds_cdd_hdd["model"] = [path.stem for path in paths_demand_zarr]


# ── Output path ───────────────────────────────────────────────────────────────
out_dir = path_data_processed / "energy" / "analysis" / "extremes"
out_dir.mkdir(parents=True, exist_ok=True)

p_int = int(percentile)
percentile_label = f"p{p_int:02d}" if percentile == p_int else f"p{percentile}"


# ── Resample to daily ─────────────────────────────────────────────────────────
print("Resampling to daily resolution...")
ds_pv_daily = ensure_daily(ds_pv, method="mean")
ds_wind_daily = ensure_daily(ds_wind, method="mean")
ds_wind_ch_daily = ensure_daily(ds_wind_ch, method="mean")
ds_hydropower_ror_daily = ensure_daily(ds_hydropower_ror, method="mean")
ds_hydropower_reservoir_daily = ensure_daily(ds_hydropower_reservoir, method="mean")
ds_cdd_hdd_daily = ensure_daily(ds_cdd_hdd, method="sum")


# ── Reference DOY thresholds ──────────────────────────────────────────────────
print(
    f"Computing reference DOY thresholds "
    f"(P{percentile}, window=±{window} days, period={REFERENCE_PERIOD})..."
)
with ProgressBar(dt=1):
    threshold_pv = compute_doy_threshold(
        ds_pv_daily, REFERENCE_PERIOD, percentile, window
    ).compute()
    threshold_wind = compute_doy_threshold(
        ds_wind_daily, REFERENCE_PERIOD, percentile, window
    ).compute()
    threshold_hydropower_ror = compute_doy_threshold(
        ds_hydropower_ror_daily, REFERENCE_PERIOD, percentile, window
    ).compute()
    threshold_hydropower_reservoir = compute_doy_threshold(
        ds_hydropower_reservoir_daily, REFERENCE_PERIOD, percentile, window
    ).compute()
    threshold_cdd_hdd = compute_doy_threshold(
        ds_cdd_hdd_daily, REFERENCE_PERIOD, percentile, window
    ).compute()
    threshold_wind_ch = compute_doy_threshold(
        ds_wind_ch_daily, REFERENCE_PERIOD, percentile, window
    ).compute()

threshold_europe = xr.merge(
    [
        threshold_pv,
        threshold_wind,
        threshold_hydropower_ror,
        threshold_hydropower_reservoir,
        threshold_cdd_hdd,
    ]
)
path_threshold = out_dir / f"{percentile_label}_threshold_doy_{climate_scenario}.nc"
if path_threshold.exists():
    path_threshold.unlink()
threshold_europe.to_netcdf(path_threshold)
print(f"  Saved: {path_threshold}")

threshold_ch = xr.merge(
    [
        sel_extent(
            xr.merge(
                [
                    threshold_pv,
                    threshold_hydropower_ror,
                    threshold_hydropower_reservoir,
                    threshold_cdd_hdd,
                ]
            )[["pv", "hydropower_ror", "hydropower_reservoir", "HDD", "CDD"]],
            extent=EXTENT_CH,
        ),
        threshold_wind_ch,
    ]
)[["pv", "wind", "hydropower_ror", "hydropower_reservoir", "HDD", "CDD"]]
path_threshold_ch = out_dir / f"{percentile_label}_threshold_doy_{climate_scenario}_ch.nc"
if path_threshold_ch.exists():
    path_threshold_ch.unlink()
threshold_ch.to_netcdf(path_threshold_ch)
print(f"  Saved: {path_threshold_ch}")


# ── Mid-century exceedance frequencies ───────────────────────────────────────
print(f"Computing mid-century exceedance frequencies ({MID_CENTURY_PERIOD})...")
with ProgressBar(dt=1):
    exc_mid_pv = compute_exceedance_frequency(
        ds_pv_daily, threshold_pv, MID_CENTURY_PERIOD
    ).compute()
    exc_mid_wind = compute_exceedance_frequency(
        ds_wind_daily, threshold_wind, MID_CENTURY_PERIOD
    ).compute()
    exc_mid_hydropower_ror = compute_exceedance_frequency(
        ds_hydropower_ror_daily, threshold_hydropower_ror, MID_CENTURY_PERIOD
    ).compute()
    exc_mid_hydropower_reservoir = compute_exceedance_frequency(
        ds_hydropower_reservoir_daily, threshold_hydropower_reservoir, MID_CENTURY_PERIOD
    ).compute()
    exc_mid_cdd_hdd = compute_exceedance_frequency(
        ds_cdd_hdd_daily, threshold_cdd_hdd, MID_CENTURY_PERIOD
    ).compute()
    exc_mid_wind_ch = compute_exceedance_frequency(
        ds_wind_ch_daily, threshold_wind_ch, MID_CENTURY_PERIOD
    ).compute()

exc_mid_europe = xr.merge(
    [
        exc_mid_pv,
        exc_mid_wind,
        exc_mid_hydropower_ror,
        exc_mid_hydropower_reservoir,
        exc_mid_cdd_hdd,
    ]
)
path_exc_mid = (
    out_dir / f"exceedance_freq_mid_century_{percentile_label}_{climate_scenario}.nc"
)
if path_exc_mid.exists():
    path_exc_mid.unlink()
exc_mid_europe.to_netcdf(path_exc_mid)
print(f"  Saved: {path_exc_mid}")

exc_mid_ch = xr.merge(
    [
        sel_extent(
            xr.merge(
                [
                    exc_mid_pv,
                    exc_mid_hydropower_ror,
                    exc_mid_hydropower_reservoir,
                    exc_mid_cdd_hdd,
                ]
            )[["pv", "hydropower_ror", "hydropower_reservoir", "HDD", "CDD"]],
            extent=EXTENT_CH,
        ),
        exc_mid_wind_ch,
    ]
)[["pv", "wind", "hydropower_ror", "hydropower_reservoir", "HDD", "CDD"]]
path_exc_mid_ch = (
    out_dir / f"exceedance_freq_mid_century_{percentile_label}_{climate_scenario}_ch.nc"
)
if path_exc_mid_ch.exists():
    path_exc_mid_ch.unlink()
exc_mid_ch.to_netcdf(path_exc_mid_ch)
print(f"  Saved: {path_exc_mid_ch}")


# ── End-of-century exceedance frequencies ────────────────────────────────────
print(f"Computing end-of-century exceedance frequencies ({END_CENTURY_PERIOD})...")
with ProgressBar(dt=1):
    exc_eoc_pv = compute_exceedance_frequency(
        ds_pv_daily, threshold_pv, END_CENTURY_PERIOD
    ).compute()
    exc_eoc_wind = compute_exceedance_frequency(
        ds_wind_daily, threshold_wind, END_CENTURY_PERIOD
    ).compute()
    exc_eoc_hydropower_ror = compute_exceedance_frequency(
        ds_hydropower_ror_daily, threshold_hydropower_ror, END_CENTURY_PERIOD
    ).compute()
    exc_eoc_hydropower_reservoir = compute_exceedance_frequency(
        ds_hydropower_reservoir_daily, threshold_hydropower_reservoir, END_CENTURY_PERIOD
    ).compute()
    exc_eoc_cdd_hdd = compute_exceedance_frequency(
        ds_cdd_hdd_daily, threshold_cdd_hdd, END_CENTURY_PERIOD
    ).compute()
    exc_eoc_wind_ch = compute_exceedance_frequency(
        ds_wind_ch_daily, threshold_wind_ch, END_CENTURY_PERIOD
    ).compute()

exc_eoc_europe = xr.merge(
    [
        exc_eoc_pv,
        exc_eoc_wind,
        exc_eoc_hydropower_ror,
        exc_eoc_hydropower_reservoir,
        exc_eoc_cdd_hdd,
    ]
)
path_exc_eoc = (
    out_dir / f"exceedance_freq_end_of_century_{percentile_label}_{climate_scenario}.nc"
)
if path_exc_eoc.exists():
    path_exc_eoc.unlink()
exc_eoc_europe.to_netcdf(path_exc_eoc)
print(f"  Saved: {path_exc_eoc}")

exc_eoc_ch = xr.merge(
    [
        sel_extent(
            xr.merge(
                [
                    exc_eoc_pv,
                    exc_eoc_hydropower_ror,
                    exc_eoc_hydropower_reservoir,
                    exc_eoc_cdd_hdd,
                ]
            )[["pv", "hydropower_ror", "hydropower_reservoir", "HDD", "CDD"]],
            extent=EXTENT_CH,
        ),
        exc_eoc_wind_ch,
    ]
)[["pv", "wind", "hydropower_ror", "hydropower_reservoir", "HDD", "CDD"]]
path_exc_eoc_ch = (
    out_dir / f"exceedance_freq_end_of_century_{percentile_label}_{climate_scenario}_ch.nc"
)
if path_exc_eoc_ch.exists():
    path_exc_eoc_ch.unlink()
exc_eoc_ch.to_netcdf(path_exc_eoc_ch)
print(f"  Saved: {path_exc_eoc_ch}")


print(f"\nDone! Results saved for scenario {climate_scenario}")
print(f"  percentile={percentile}, window=±{window} days")
print(f"  Output directory: {out_dir}")