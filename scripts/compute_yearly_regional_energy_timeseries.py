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
    CH_TURBINES,
    EXTENT_CH,
    EXTENT_BC
)
from cordex2energy.utils.utils_xarray import sel_extent
from cordex2energy.conversion_energy.aggregate_data import create_regions_mask
from cordex2energy_analysis.analysis import compute_seasonal_statistics, compute_regional_values
from cordex2energy.conversion_energy.demand import (
    process_population_data,
    normalize_population_data
)
from cordex2energy.data_processing.load_data import ClimateModelData

paths = json.load(open("../../paths_so4.json"))

path_data_processed = Path(paths["path_data_processed"])

dask.config.set(num_workers=12, threads_per_worker=4)
dask.config.set({"array.chunk-size": "500MiB"})

climate_scenario = "rcp85"
selected_turbine_ch = CH_TURBINES[0]
selected_pv_tilt_azimuth_ch = "tilt_35_azimuth_S"

path_energy_europe = path_data_processed / "energy" / "europe"
path_energy_ch = path_data_processed / "energy" / "ch"

# Load data
path_pv_ch = path_energy_ch / "pv"
path_pv_ch_zarr = sorted(path_pv_ch.glob(f"*/*{climate_scenario}*.zarr"))

ds_pv_ch = xr.concat(
    [
        xr.open_zarr(path, chunks="auto")[[selected_pv_tilt_azimuth_ch]]
        for path in path_pv_ch_zarr
    ],
    dim="model",
).rename({selected_pv_tilt_azimuth_ch: "pv"})
ds_pv_ch["model"] = [path.stem.split("pv_")[1] for path in path_pv_ch_zarr]

path_wind_ch = path_energy_ch / "wind_power"
path_wind_ch_zarr = sorted(path_wind_ch.glob(f"*{climate_scenario}*.zarr"))

ds_wind_ch = xr.concat(
    [
        xr.open_zarr(path, chunks="auto")[[selected_turbine_ch.replace("/", "_")]]
        for path in path_wind_ch_zarr
    ],
    dim="model",
).rename({selected_turbine_ch.replace("/", "_"): "wind"})
ds_wind_ch["model"] = [path.stem.split("wind_power_")[1] for path in path_wind_ch_zarr]

# Load hydropower run-of-river
path_hydropower_ror = path_energy_europe / "hydropower"
path_hydropower_ror_zarr = sorted(path_hydropower_ror.glob(f"*/hydropower_ror_*{climate_scenario}*.zarr"))
ds_hydropower_ror_ch = sel_extent(
    xr.concat(
        [
            xr.open_zarr(path, chunks="auto")
            .convert_calendar("standard", align_on="year")
            .sortby("lat", ascending=True)
            for path in path_hydropower_ror_zarr
        ],
        dim="model",
    ),
    extent=EXTENT_CH,
)
ds_hydropower_ror_ch["lat"] = ds_pv_ch["lat"]
ds_hydropower_ror_ch["lon"] = ds_pv_ch["lon"]
ds_hydropower_ror_ch["model"] = [
    path.stem.split("hydropower_ror_")[1] for path in path_hydropower_ror_zarr
]

# Load hydropower reservoir
path_hydropower_reservoir = path_energy_europe / "hydropower"
path_hydropower_reservoir_zarr = sorted(path_hydropower_reservoir.glob(f"*/hydropower_reservoir_*{climate_scenario}*.zarr"))
ds_hydropower_reservoir_ch = sel_extent(
    xr.concat(
        [
            xr.open_zarr(path, chunks="auto")
            .convert_calendar("standard", align_on="year")
            .sortby("lat", ascending=True)
            for path in path_hydropower_reservoir_zarr
        ],
        dim="model",
    ),
    extent=EXTENT_CH,
)
ds_hydropower_reservoir_ch["lat"] = ds_pv_ch["lat"]
ds_hydropower_reservoir_ch["lon"] = ds_pv_ch["lon"]
ds_hydropower_reservoir_ch["model"] = [
    path.stem.split("hydropower_reservoir_")[1] for path in path_hydropower_reservoir_zarr
]

path_demand = path_energy_europe / "demand"
paths_demand_zarr = sorted(path_demand.glob(f"*{climate_scenario}.zarr"))

ds_cdd_hdd_ch = sel_extent(
    xr.concat(
        [
            xr.open_zarr(path, chunks="auto").isel(band=0).drop_duplicates(dim="time")
            for path in paths_demand_zarr
        ],
        dim="model",
    ),
    extent=EXTENT_CH,
)
ds_cdd_hdd_ch["model"] = [path.stem for path in paths_demand_zarr]

extent_pop = {
    "lat_min": EXTENT_BC["lat_min"] - 1,
    "lat_max": EXTENT_BC["lat_max"] + 1,
    "lon_min": EXTENT_BC["lon_min"] - 1,
    "lon_max": EXTENT_BC["lon_max"] + 1,
}

ds_tas_bc = ClimateModelData(
    Path(paths["path_data_processed"])
    / "cordex_processed"
    / f"{ds_cdd_hdd_ch.model.values[0]}"
    / "bias_corrected"
    / "3hr"
    / "tas.zarr",
    extent=EXTENT_BC,
    chunks="auto"
)

da_worldpop_coarse = process_population_data(
    sel_extent(
        rxr.open_rasterio(
            Path(paths["path_data_processed"])
            / "pop"
            / "ppp_2020_1km_Aggregated.tif"
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
ds_cdd_hdd_ch = (ds_cdd_hdd_ch * da_worldpop_coarse).isel(band=0)


gdf_regions_ch = gpd.read_file(
    path_data_processed / "maps" / "regions_ch" / "PRODREG.shp"
).to_crs("EPSG:4326")
regions_ch, regions_mask = create_regions_mask(
    gdf_regions_ch, names="ProdregN_3", da=ds_wind_ch.wind
)

# Create Switzerland country mask from the existing worldpop mask
mask_switzerland = sel_extent((mask_countries_worldpop.cf == "Switzerland"), extent=EXTENT_CH)

print("Computing yearly timeseries of PV, Wind, Hydropower and Demand (by region)")
with ProgressBar(dt=1):
    ds_pv_ch_mean_yearly = compute_regional_values(
        ds_pv_ch.resample(time="YE").mean(dim="time"),
        regions_mask,
        gdf_regions_ch["ProdregN_3"].values,
    ).compute()
    ds_wind_ch_mean_yearly = compute_regional_values(
        ds_wind_ch.resample(time="YE").mean(dim="time"),
        regions_mask,
        gdf_regions_ch["ProdregN_3"].values,
    ).compute()
    ds_hydropower_ror_ch_mean_yearly = compute_regional_values(
        ds_hydropower_ror_ch.resample(time="YE").mean(dim="time"),
        regions_mask,
        gdf_regions_ch["ProdregN_3"].values,
    ).compute()
    ds_hydropower_reservoir_ch_mean_yearly = compute_regional_values(
        ds_hydropower_reservoir_ch.resample(time="YE").mean(dim="time"),
        regions_mask,
        gdf_regions_ch["ProdregN_3"].values,
    ).compute()
    ds_cdd_hdd_sum_yearly = compute_regional_values(
        ds_cdd_hdd_ch.resample(time="YE").sum(dim="time"),
        regions_mask,
        gdf_regions_ch["ProdregN_3"].values,
        method="sum",
    ).compute()

ds_mean_yearly = xr.merge(
    [ds_pv_ch_mean_yearly, ds_wind_ch_mean_yearly, ds_hydropower_ror_ch_mean_yearly, ds_hydropower_reservoir_ch_mean_yearly, ds_cdd_hdd_sum_yearly]
)

# Compute Switzerland-wide timeseries and add as a new mask value
print("Computing yearly timeseries for entire extent (Switzerland-wide)")
with ProgressBar(dt=1):
    ds_pv_ch_mean_yearly_ch = ds_pv_ch.where(mask_switzerland).resample(time="YE").mean(dim="time").mean(dim=["lat", "lon"]).compute()
    ds_wind_ch_mean_yearly_ch = ds_wind_ch.where(mask_switzerland).resample(time="YE").mean(dim="time").mean(dim=["lat", "lon"]).compute()
    ds_hydropower_ror_ch_mean_yearly_ch = ds_hydropower_ror_ch.where(mask_switzerland).resample(time="YE").mean(dim="time").mean(dim=["lat", "lon"]).compute()
    ds_hydropower_reservoir_ch_mean_yearly_ch = ds_hydropower_reservoir_ch.where(mask_switzerland).resample(time="YE").mean(dim="time").mean(dim=["lat", "lon"]).compute()
    ds_cdd_hdd_sum_yearly_ch = ds_cdd_hdd_ch.where(mask_switzerland).resample(time="YE").sum(dim="time").sum(dim=["lat", "lon"]).compute()

ds_mean_yearly_ch = xr.merge(
    [ds_pv_ch_mean_yearly_ch, ds_wind_ch_mean_yearly_ch, ds_hydropower_ror_ch_mean_yearly_ch, ds_hydropower_reservoir_ch_mean_yearly_ch, ds_cdd_hdd_sum_yearly_ch]
)
# Add mask dimension with value "Switzerland"
ds_mean_yearly_ch = ds_mean_yearly_ch.expand_dims({"mask": ["Switzerland"]})

# Combine regional and Switzerland-wide datasets
ds_mean_yearly_combined = xr.concat([ds_mean_yearly, ds_mean_yearly_ch], dim="mask")

path_mean_yearly = path_energy_ch / "statistics" / "regional_stats_yearly.nc"
path_mean_yearly.parent.mkdir(parents=True, exist_ok=True)
if path_mean_yearly.exists():
    path_mean_yearly.unlink()
ds_mean_yearly_combined.to_netcdf(path_mean_yearly)

print("Computing seasonal timeseries of PV, Wind, Hydropower and Demand (by region)")
method_energy = "mean"
with ProgressBar(dt=1):
    ds_pv_ch_mean_seasonal = compute_regional_values(
        compute_seasonal_statistics(ds_pv_ch, method=method_energy),
        regions_mask,
        gdf_regions_ch["ProdregN_3"].values,
    ).compute()
    ds_wind_ch_mean_seasonal = compute_regional_values(
        compute_seasonal_statistics(ds_wind_ch, method=method_energy),
        regions_mask,
        gdf_regions_ch["ProdregN_3"].values,
    ).compute()
    ds_hydropower_ror_ch_mean_seasonal = compute_regional_values(
        compute_seasonal_statistics(ds_hydropower_ror_ch, method=method_energy),
        regions_mask,
        gdf_regions_ch["ProdregN_3"].values,
    ).compute()
    ds_hydropower_reservoir_ch_mean_seasonal = compute_regional_values(
        compute_seasonal_statistics(ds_hydropower_reservoir_ch, method=method_energy),
        regions_mask,
        gdf_regions_ch["ProdregN_3"].values,
    ).compute()
    ds_cdd_hdd_sum_seasonal = compute_regional_values(
        compute_seasonal_statistics(ds_cdd_hdd_ch, method="sum"),
        regions_mask,
        gdf_regions_ch["ProdregN_3"].values,
    ).compute()

ds_mean_seasonal = xr.merge(
    [ds_pv_ch_mean_seasonal, ds_wind_ch_mean_seasonal, ds_hydropower_ror_ch_mean_seasonal, ds_hydropower_reservoir_ch_mean_seasonal, ds_cdd_hdd_sum_seasonal]
)

# Compute Switzerland-wide seasonal timeseries and add as a new mask value
print("Computing seasonal timeseries for entire extent (Switzerland-wide)")
with ProgressBar(dt=1):
    ds_pv_ch_mean_seasonal_ch = compute_seasonal_statistics(ds_pv_ch.where(mask_switzerland), method="mean").mean(dim=["lat", "lon"]).compute()
    ds_wind_ch_mean_seasonal_ch = compute_seasonal_statistics(ds_wind_ch.where(mask_switzerland), method="mean").mean(dim=["lat", "lon"]).compute()
    ds_hydropower_ror_ch_mean_seasonal_ch = compute_seasonal_statistics(ds_hydropower_ror_ch.where(mask_switzerland), method="mean").mean(dim=["lat", "lon"]).compute()
    ds_hydropower_reservoir_ch_mean_seasonal_ch = compute_seasonal_statistics(ds_hydropower_reservoir_ch.where(mask_switzerland), method="mean").mean(dim=["lat", "lon"]).compute()
    ds_cdd_hdd_sum_seasonal_ch = compute_seasonal_statistics(ds_cdd_hdd_ch.where(mask_switzerland), method="sum").sum(dim=["lat", "lon"]).compute()

ds_mean_seasonal_ch = xr.merge(
    [ds_pv_ch_mean_seasonal_ch, ds_wind_ch_mean_seasonal_ch, ds_hydropower_ror_ch_mean_seasonal_ch, ds_hydropower_reservoir_ch_mean_seasonal_ch, ds_cdd_hdd_sum_seasonal_ch]
)
# Add mask dimension with value "Switzerland"
ds_mean_seasonal_ch = ds_mean_seasonal_ch.expand_dims({"mask": ["Switzerland"]})

# Combine regional and Switzerland-wide datasets
ds_mean_seasonal_combined = xr.concat([ds_mean_seasonal, ds_mean_seasonal_ch], dim="mask")

path_mean_seasonal = path_energy_ch / "statistics" / "regional_stats_seasonal.nc"
path_mean_seasonal.parent.mkdir(parents=True, exist_ok=True)
if path_mean_seasonal.exists():
    path_mean_seasonal.unlink()
ds_mean_seasonal_combined.to_netcdf(path_mean_seasonal)
