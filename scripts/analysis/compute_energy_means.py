import argparse
import json
from pathlib import Path

import dask
import xarray as xr
from dask.diagnostics import ProgressBar

from cordex2energy.utils.constants import (
    EUROPE_TURBINES,
    CH_TURBINES,
    EXTENT_CH,
    REFERENCE_PERIOD,
    MID_CENTURY_PERIOD,
)
from cordex2energy.utils.utils_xarray import sel_extent
from cordex2energy_analysis.analysis import compute_seasonal_statistics

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Compute temporal means for energy data"
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
    "--aggregation",
    type=str,
    default="annual",
    choices=["annual", "seasonal"],
    help="How to aggregate time within each output step: annual (full-year) or seasonal (default: annual)",
)
parser.add_argument(
    "--time_resolution",
    type=str,
    default="period",
    choices=["period", "yearly"],
    help="Output time scope: period (single mean over entire period) or yearly (one value per year) (default: period)",
)
parser.add_argument(
    "--season_type",
    type=str,
    default="energy",
    choices=["meteorological", "energy"],
    help="Type of seasonal breakdown (only used if aggregation=seasonal): meteorological (DJF, MAM, JJA, SON) or energy (ONDJFM, AMJJAS) (default: energy)",
)

args = parser.parse_args()

paths = json.load(open(args.paths_file))

path_data_processed = Path(paths["path_data_processed"])

dask.config.set(num_workers=12, threads_per_worker=4)
dask.config.set({"array.chunk-size": "500MiB"})

climate_scenario = args.scenario
aggregation = args.aggregation
time_resolution = args.time_resolution
season_type = args.season_type
selected_turbine_europe = EUROPE_TURBINES[0]
selected_turbine_ch = CH_TURBINES[0]

path_energy_europe = path_data_processed / "energy" / "europe"
path_energy_ch = path_data_processed / "energy" / "ch"
extent = {"lat_min": 30, "lat_max": 72.6, "lon_min": -12, "lon_max": 39}

# Load PV
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
# ds_pv = ds_pv.resample(time="YE").sum(dim="time")

# Load Wind
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

# Load hydropower
path_hydropower_ror = path_energy_europe / "hydropower"
path_hydropower_ror_zarr = sorted(path_hydropower_ror.glob(f"*/hydropower_ror_*{climate_scenario}*.zarr"))
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

# Load hydropower reservoir
path_hydropower_reservoir = path_energy_europe / "hydropower"
path_hydropower_reservoir_zarr = sorted(path_hydropower_reservoir.glob(f"*/hydropower_reservoir_*{climate_scenario}*.zarr"))
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

# Load demand
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


# Helper function to compute temporal means
def compute_temporal_mean(ds, time_period, aggregation, time_resolution, season_type="meteorological", method="mean"):
    """
    Compute temporal aggregates based on the specified aggregation type and time resolution.

    Parameters
    ----------
    aggregation : str
        How to aggregate within each time step: "annual" (full-year) or "seasonal".
    time_resolution : str
        Output time scope: "period" (single climatological mean over entire period)
        or "yearly" (one value per year, with year — and optionally season — as coordinates).
    method : str
        How to reduce values within each aggregation window: "mean" or "sum".
        Use "sum" for accumulation quantities (e.g. CDD, HDD).
    """
    ds_period = ds.sel(time=slice(*time_period))

    if aggregation == "seasonal":
        # compute_seasonal_statistics handles both meteorological (resample QS-DEC) and
        # energy (groupby year_season) paths and returns (year, season) coordinates.
        ds_seasonal = compute_seasonal_statistics(
            ds_period, time_dim="time", method=method, season_type=season_type
        )
        if time_resolution == "yearly":
            return ds_seasonal.compute()
        else:  # period — average over years to get a single climatological value per season
            if season_type == "meteorological":
                return ds_seasonal.groupby("season").mean(dim="time", skipna=True).compute()
            else:  # energy: output already has explicit 'year' dimension
                return ds_seasonal.mean(dim="year", skipna=True).compute()

    else:  # annual — reduce to one value per year (or one value over the whole period)
        # Ensure yearly resolution first (resample if data is finer than yearly)
        time_span = ds_period.time[-1].dt.year.values - ds_period.time[0].dt.year.values + 1
        if len(ds_period.time) > time_span:
            ds_period = (
                ds_period.resample(time="YE").sum(dim="time")
                if method == "sum"
                else ds_period.resample(time="YE").mean(dim="time", skipna=True)
            )
        if time_resolution == "yearly":
            return ds_period.compute()
        else:  # period — single climatological mean over all years
            return ds_period.mean(dim="time", skipna=True).compute()


# Build output path components from arguments
season_suffix = f"_{season_type}" if aggregation == "seasonal" else ""
out_dir = path_data_processed / "energy" / "analysis" / aggregation / time_resolution

# Analysis
print(f"Computing reference period means (aggregation={aggregation}, time_resolution={time_resolution})")
if aggregation == "seasonal":
    print(f"Using {season_type} seasons")

with ProgressBar(dt=1):
    mean_ref_pv = compute_temporal_mean(ds_pv, REFERENCE_PERIOD, aggregation, time_resolution, season_type)
    mean_ref_wind = compute_temporal_mean(ds_wind, REFERENCE_PERIOD, aggregation, time_resolution, season_type)
    mean_ref_hydropower_ror = compute_temporal_mean(ds_hydropower_ror, REFERENCE_PERIOD, aggregation, time_resolution, season_type)
    mean_ref_hydropower_reservoir = compute_temporal_mean(ds_hydropower_reservoir, REFERENCE_PERIOD, aggregation, time_resolution, season_type)
    mean_ref_cdd_hdd = compute_temporal_mean(ds_cdd_hdd, REFERENCE_PERIOD, aggregation, time_resolution, season_type, method="sum")
    mean_ref_wind_ch = compute_temporal_mean(ds_wind_ch, REFERENCE_PERIOD, aggregation, time_resolution, season_type)

mean_ref = xr.merge([mean_ref_pv, mean_ref_wind, mean_ref_hydropower_ror, mean_ref_hydropower_reservoir, mean_ref_cdd_hdd])
path_mean_ref = out_dir / f"mean_reference_period{season_suffix}.nc"
path_mean_ref.parent.mkdir(parents=True, exist_ok=True)
if path_mean_ref.exists():
    path_mean_ref.unlink()
mean_ref.to_netcdf(path_mean_ref)


# Build CH reference using in-memory components — avoids a fragile save/reload roundtrip
# that can introduce coordinate floating-point drift and silent NaN via outer-join merge.
mean_ref_ch = xr.merge(
    [
        sel_extent(
            xr.merge([mean_ref_pv, mean_ref_hydropower_ror, mean_ref_hydropower_reservoir, mean_ref_cdd_hdd])[
                ["pv", "hydropower_ror", "hydropower_reservoir", "HDD", "CDD"]
            ],
            extent=EXTENT_CH,
        ),
        mean_ref_wind_ch,
    ]
)[["pv", "wind", "hydropower_ror", "hydropower_reservoir", "HDD", "CDD"]]
path_mean_ref_ch = out_dir / f"mean_reference_period_ch{season_suffix}.nc"
path_mean_ref_ch.parent.mkdir(parents=True, exist_ok=True)
if path_mean_ref_ch.exists():
    path_mean_ref_ch.unlink()
mean_ref_ch.to_netcdf(path_mean_ref_ch)


print(f"Computing mid-century means (aggregation={aggregation}, time_resolution={time_resolution})")
if aggregation == "seasonal":
    print(f"Using {season_type} seasons")

with ProgressBar(dt=1):
    mean_mid_century_pv = compute_temporal_mean(ds_pv, MID_CENTURY_PERIOD, aggregation, time_resolution, season_type)
    mean_mid_century_wind = compute_temporal_mean(ds_wind, MID_CENTURY_PERIOD, aggregation, time_resolution, season_type)
    mean_mid_century_hydropower_ror = compute_temporal_mean(ds_hydropower_ror, MID_CENTURY_PERIOD, aggregation, time_resolution, season_type)
    mean_mid_century_hydropower_reservoir = compute_temporal_mean(ds_hydropower_reservoir, MID_CENTURY_PERIOD, aggregation, time_resolution, season_type)
    mean_mid_century_cdd_hdd = compute_temporal_mean(ds_cdd_hdd, MID_CENTURY_PERIOD, aggregation, time_resolution, season_type, method="sum")
    mean_mid_century_wind_ch = compute_temporal_mean(ds_wind_ch, MID_CENTURY_PERIOD, aggregation, time_resolution, season_type)

mean_mid_century = xr.merge(
    [
        mean_mid_century_pv,
        mean_mid_century_wind,
        mean_mid_century_hydropower_ror,
        mean_mid_century_hydropower_reservoir,
        mean_mid_century_cdd_hdd,
    ]
)
path_mean_mid_century = out_dir / f"mean_mid_century{season_suffix}.nc"
path_mean_mid_century.parent.mkdir(parents=True, exist_ok=True)
if path_mean_mid_century.exists():
    path_mean_mid_century.unlink()
mean_mid_century.to_netcdf(path_mean_mid_century)

# Build CH mid-century using in-memory components — same pattern as reference above.
mean_mid_century_ch = xr.merge(
    [
        sel_extent(
            xr.merge([mean_mid_century_pv, mean_mid_century_hydropower_ror, mean_mid_century_hydropower_reservoir, mean_mid_century_cdd_hdd])[
                ["pv", "hydropower_ror", "hydropower_reservoir", "HDD", "CDD"]
            ],
            extent=EXTENT_CH,
        ),
        mean_mid_century_wind_ch,
    ]
)[["pv", "wind", "hydropower_ror", "hydropower_reservoir", "HDD", "CDD"]]
path_mean_mid_century_ch = out_dir / f"mean_mid_century_ch{season_suffix}.nc"
path_mean_mid_century_ch.parent.mkdir(parents=True, exist_ok=True)
if path_mean_mid_century_ch.exists():
    path_mean_mid_century_ch.unlink()
mean_mid_century_ch.to_netcdf(path_mean_mid_century_ch)

print(f"Done! Results saved for scenario {climate_scenario}")
print(f"  aggregation={aggregation}, time_resolution={time_resolution}" + (f", season_type={season_type}" if aggregation == "seasonal" else ""))
