import json
import sys
from pathlib import Path

import dask
import regionmask
import xarray as xr
import rioxarray as rxr
import numpy as np
import geopandas as gpd
from dask.diagnostics import ProgressBar

from cordex2energy.utils.constants import (
    EXTENT_CH,
    EXTENT_BC,
)
from cordex2energy.utils.utils_xarray import sel_extent
from cordex2energy.conversion_energy.aggregate_data import create_regions_mask
from cordex2energy_analysis.analysis import compute_seasonal_statistics, compute_regional_values
from cordex2energy.conversion_energy.demand import (
    process_population_data,
    normalize_population_data,
)
from cordex2energy.data_processing.load_data import ClimateModelData

paths = json.load(open("../../paths_so4.json"))

path_data_processed = Path(paths["path_data_processed"])

dask.config.set(num_workers=12, threads_per_worker=4)
dask.config.set({"array.chunk-size": "500MiB"})

climate_scenario = "rcp85"

path_cordex_processed = path_data_processed / "cordex_processed"

# Climate variables and their temporal resolutions and aggregation methods
CLIMATE_VARS = {
    "tas":     {"temp_res": "3hr", "agg_yearly": "mean",  "agg_seasonal": "mean"},
    "pr":      {"temp_res": "day", "agg_yearly": "sum",   "agg_seasonal": "sum"},
    "rsds":    {"temp_res": "3hr", "agg_yearly": "mean",  "agg_seasonal": "mean"},
    "sfcWind": {"temp_res": "3hr", "agg_yearly": "mean",  "agg_seasonal": "mean"},
    "clt":     {"temp_res": "day", "agg_yearly": "mean",  "agg_seasonal": "mean"},
}

# Load data for each climate variable across all models for the given scenario
ds_climate_vars = {}
for var, meta in CLIMATE_VARS.items():
    temp_res = meta["temp_res"]
    paths_var_zarr = sorted(
        path_cordex_processed.glob(
            f"*_{climate_scenario}/bias_corrected/{temp_res}/{var}.zarr"
        )
    )
    if not paths_var_zarr:
        print(f"Warning: no zarr files found for variable '{var}', skipping.")
        continue

    print(f"Loading {var} ({len(paths_var_zarr)} models) ...")
    ds_var = sel_extent(
        xr.concat(
            [
                xr.open_zarr(path, chunks="auto")
                .convert_calendar("standard", align_on="year")
                .sortby("lat", ascending=True)
                for path in paths_var_zarr
            ],
            dim="model",
        ),
        extent=EXTENT_CH,
    )
    ds_var["model"] = [
        path.parts[-4].rsplit("_", 1)[0] for path in paths_var_zarr
    ]
    ds_climate_vars[var] = ds_var


# Use one of the loaded variables as a spatial reference for lat/lon alignment
_ref_var = next(iter(ds_climate_vars))
ds_ref = ds_climate_vars[_ref_var]


# Load population data for demand-weighted spatial aggregation
extent_pop = {
    "lat_min": EXTENT_BC["lat_min"] - 1,
    "lat_max": EXTENT_BC["lat_max"] + 1,
    "lon_min": EXTENT_BC["lon_min"] - 1,
    "lon_max": EXTENT_BC["lon_max"] + 1,
}

ds_tas_bc = ClimateModelData(
    path_cordex_processed
    / f"{ds_ref.model.values[0]}_{climate_scenario}"
    / "bias_corrected"
    / "3hr"
    / "tas.zarr",
    extent=EXTENT_BC,
    chunks="auto",
)

da_worldpop_coarse = process_population_data(
    sel_extent(
        rxr.open_rasterio(
            path_data_processed / "pop" / "ppp_2020_1km_Aggregated.tif"
        ).rename({"x": "lon", "y": "lat"}),
        extent=extent_pop,
    ),
    ds_tas_bc.ds_data,
    coarsen=True,
)

countries = regionmask.defined_regions.natural_earth_v5_0_0.countries_10
mask_countries_worldpop = countries.mask(da_worldpop_coarse, flag="names")
da_worldpop_coarse = normalize_population_data(
    da_worldpop_coarse, mask_countries_worldpop
).where(da_worldpop_coarse > 0, np.nan)

da_worldpop_coarse = da_worldpop_coarse.chunk(
    {
        "lat": ds_tas_bc.ds_data.chunksizes["lat"][0],
        "lon": ds_tas_bc.ds_data.chunksizes["lon"][0],
    }
)
da_worldpop_coarse = sel_extent(da_worldpop_coarse, extent=EXTENT_CH)

# Create Switzerland country mask
mask_switzerland = sel_extent(
    (mask_countries_worldpop.cf == "Switzerland"), extent=EXTENT_CH
)

# Load regional shapefile for Switzerland
gdf_regions_ch = gpd.read_file(
    path_data_processed / "maps" / "regions_ch" / "PRODREG.shp"
).to_crs("EPSG:4326")
_, regions_mask = create_regions_mask(
    gdf_regions_ch, names="ProdregN_3", da=ds_ref[_ref_var]
)

# ---------------------------------------------------------------------------
# Yearly statistics
# ---------------------------------------------------------------------------
print("Computing yearly timeseries of climate variables (by region)")

ds_yearly_regional_list = []
ds_yearly_ch_list = []

for var, meta in CLIMATE_VARS.items():
    if var not in ds_climate_vars:
        continue
    ds_var = ds_climate_vars[var]
    agg = meta["agg_yearly"]

    # Resample sub-daily data to daily means before yearly aggregation
    if meta["temp_res"] == "3hr":
        ds_var = ds_var.resample(time="D").mean(dim="time")

    print(f"  {var} (yearly {agg}) ...")
    with ProgressBar(dt=1):
        if agg == "sum":
            ds_resampled = ds_var.resample(time="YE").sum(dim="time")
        else:
            ds_resampled = ds_var.resample(time="YE").mean(dim="time")

        ds_regional = compute_regional_values(
            ds_resampled,
            regions_mask,
            gdf_regions_ch["ProdregN_3"].values,
            method=agg,
        ).compute()
        ds_yearly_regional_list.append(ds_regional)

        # Switzerland-wide
        if agg == "sum":
            ds_ch = (
                ds_var.where(mask_switzerland)
                .resample(time="YE")
                .sum(dim="time")
                .sum(dim=["lat", "lon"])
                .compute()
            )
        else:
            ds_ch = (
                ds_var.where(mask_switzerland)
                .resample(time="YE")
                .mean(dim="time")
                .mean(dim=["lat", "lon"])
                .compute()
            )
        ds_ch = ds_ch.expand_dims({"mask": ["Switzerland"]})
        ds_yearly_ch_list.append(ds_ch)

ds_yearly_regional = xr.merge(ds_yearly_regional_list)
ds_yearly_ch = xr.merge(ds_yearly_ch_list)
ds_yearly_combined = xr.concat([ds_yearly_regional, ds_yearly_ch], dim="mask")

path_out_yearly = path_data_processed / "energy" / "ch" / "statistics" / "regional_stats_climate_yearly.nc"
path_out_yearly.parent.mkdir(parents=True, exist_ok=True)
if path_out_yearly.exists():
    path_out_yearly.unlink()
ds_yearly_combined.to_netcdf(path_out_yearly)
print(f"Saved yearly climate statistics to {path_out_yearly}")

# ---------------------------------------------------------------------------
# Seasonal statistics
# ---------------------------------------------------------------------------
print("Computing seasonal timeseries of climate variables (by region)")

ds_seasonal_regional_list = []
ds_seasonal_ch_list = []

for var, meta in CLIMATE_VARS.items():
    if var not in ds_climate_vars:
        continue
    ds_var = ds_climate_vars[var]
    agg = meta["agg_seasonal"]

    # Resample sub-daily data to daily means before seasonal aggregation
    if meta["temp_res"] == "3hr":
        ds_var = ds_var.resample(time="D").mean(dim="time")

    print(f"  {var} (seasonal {agg}) ...")
    with ProgressBar(dt=1):
        ds_seasonal = compute_seasonal_statistics(ds_var, method=agg)
        ds_regional = compute_regional_values(
            ds_seasonal,
            regions_mask,
            gdf_regions_ch["ProdregN_3"].values,
            method=agg,
        ).compute()
        ds_seasonal_regional_list.append(ds_regional)

        # Switzerland-wide
        ds_ch = (
            compute_seasonal_statistics(ds_var.where(mask_switzerland), method=agg)
        )
        if agg == "sum":
            ds_ch = ds_ch.sum(dim=["lat", "lon"]).compute()
        else:
            ds_ch = ds_ch.mean(dim=["lat", "lon"]).compute()
        ds_ch = ds_ch.expand_dims({"mask": ["Switzerland"]})
        ds_seasonal_ch_list.append(ds_ch)

ds_seasonal_regional = xr.merge(ds_seasonal_regional_list)
ds_seasonal_ch = xr.merge(ds_seasonal_ch_list)
ds_seasonal_combined = xr.concat([ds_seasonal_regional, ds_seasonal_ch], dim="mask")

path_out_seasonal = path_data_processed / "energy" / "ch" / "statistics" / "regional_stats_climate_seasonal.nc"
path_out_seasonal.parent.mkdir(parents=True, exist_ok=True)
if path_out_seasonal.exists():
    path_out_seasonal.unlink()
ds_seasonal_combined.to_netcdf(path_out_seasonal)
print(f"Saved seasonal climate statistics to {path_out_seasonal}")
