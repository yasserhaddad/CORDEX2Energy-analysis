"""European area-mean temperature, precipitation and runoff per model and warming level.

Three sources, one table (``<processed>/european_t_p_by_gwl.csv``):

    gcm    the CMIP5 ensemble, raw, from the cmip5-ng annual archive (tas, pr)
    rcm    the chains, bias-corrected, from the warming-level means written by
           ``compute_climate_means_gwl.py`` (tas, pr, mrro)
    cerra  the reanalysis the bias correction targets, over 1991-2020 (tas, pr)

Area means are cosine-latitude weighted over :data:`EXTENT`. The chains and CERRA are averaged
over a common valid mask, the cells where CERRA has data (the bias-corrected output is filled
beyond the CERRA domain by extrapolation); runoff is averaged over land only, because the product
stores 0 over the sea. The GCMs use the same box without a mask, once their 0-360 longitudes
are moved to -180-180. Both masks are cached as
``<processed>/cerra_valid_mask.nc`` and ``<processed>/european_land_mask.nc``, and are reused by
:func:`~cordex2energy_analysis.pipeline_inputs.european_seasonal_change`.

Units follow the bias-corrected product: temperature in degC (converted from K), precipitation
and runoff in mm d-1 (stored as ``kg m**-2`` per day). CMIP5 ``pr`` is in kg m-2 s-1 and is
multiplied by 86400.

Reading CERRA is slow: its zarrs hold the whole time axis in one chunk per tile, so the
analysis extent reads ~10 GB for ``tp`` and ~100 GB for ``2t``. A run with ``--sources`` replaces
only those sources' rows of an existing table.

Examples
--------
    python compute_european_t_p.py --sources gcm rcm
    python compute_european_t_p.py --sources cerra
    python compute_european_t_p.py
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

#: The five GCMs driving the chains, with the ensemble member each chain used.
DRIVING_GCMS = {"CNRM-CM5": "r1i1p1", "HadGEM2-ES": "r1i1p1", "NorESM1-M": "r1i1p1",
                "MPI-ESM-LR": "r1i1p1", "EC-EARTH": "r12i1p1"}
#: CORDEX institute-prefixed GCM names to the CMIP5 names of cmip5-ng and the warming-level YAML.
CORDEX_TO_CMIP5 = {"CNRM-CERFACS-CNRM-CM5": "CNRM-CM5", "MOHC-HadGEM2-ES": "HadGEM2-ES",
                   "NCC-NorESM1-M": "NorESM1-M", "MPI-M-MPI-ESM-LR": "MPI-ESM-LR",
                   "ICHEC-EC-EARTH": "EC-EARTH"}
#: The analysis domain.
EXTENT = {"lat": slice(30, 72.6), "lon": slice(-12, 39)}
#: CERRA's bias-correction training window.
REFERENCE_PERIOD = ("1991", "2020")

parser = argparse.ArgumentParser(description="European mean T and P per model and warming level")
parser.add_argument("--paths_file", type=str,
                    default=str(Path(__file__).resolve().parent.parent / "paths.json"))
parser.add_argument("--models_json", type=str,
                    default=str(Path(__file__).resolve().parent.parent / "models.json"))
parser.add_argument("--cmip5ng", type=str, default="/net/atmos/data/cmip5-ng")
parser.add_argument("--scenario", type=str, default="rcp85")
parser.add_argument("--warming_levels", type=float, nargs="+", default=[1.5, 2.0, 3.0])
parser.add_argument("--sources", type=str, nargs="+", default=["gcm", "rcm", "cerra"],
                    choices=["gcm", "rcm", "cerra"])
args = parser.parse_args()

with open(args.paths_file) as handle:
    paths = json.load(handle)
processed = Path(paths["path_data_processed"])
out_path = processed / "european_t_p_by_gwl.csv"
gwl_dir = processed / "energy" / "analysis" / "annual" / "gwl"


def area_mean(field, mask=None):
    """Cosine-latitude weighted spatial mean, optionally over a mask.

    Parameters
    ----------
    field : xarray.DataArray
        Field on ``(lat, lon)``.
    mask : xarray.DataArray, optional
        Boolean mask on the same grid.

    Returns
    -------
    float
    """
    weights = np.cos(np.deg2rad(field["lat"]))
    if mask is not None:
        field = field.where(mask)
        weights = weights.broadcast_like(field).where(mask)
    return float(field.weighted(weights.fillna(0.0)).mean(dim=["lat", "lon"]))


def cerra_mask():
    """Cells where CERRA has data, on the 0.1 deg grid, over the analysis extent.

    Read from ``cerra_valid_mask.nc`` if it exists; otherwise built from the first time step of
    the regridded ``tp`` and written there (reading one step decompresses ~10 GB).

    Returns
    -------
    xarray.DataArray
        Boolean ``cerra_valid`` on ``(lat, lon)``.
    """
    cache = processed / "cerra_valid_mask.nc"
    if cache.exists():
        return xr.open_dataarray(cache)
    tp = xr.open_zarr(processed / "cerra_processed" / "regridded" / "tp.zarr")["tp"]
    mask = tp.isel(time=0).sel(**EXTENT).notnull().drop_vars(["time"], errors="ignore").compute()
    mask.name = "cerra_valid"
    mask.attrs = {"long_name": "CERRA has data here",
                  "source": "cerra_processed/regridded/tp.zarr, first timestep"}
    mask.to_netcdf(cache)
    print(f"  cached CERRA valid mask -> {cache} "
          f"({int(mask.sum())}/{mask.size} cells, {100 * float(mask.mean()):.1f}%)", flush=True)
    return mask


def land_mask(cerra_valid):
    """Land cells of the analysis extent inside the CERRA footprint, on the 0.1 deg grid.

    Read from ``european_land_mask.nc`` if it exists; otherwise built from the Natural Earth
    1:10 m land polygons (regionmask) and written there. Runoff is averaged over this mask
    because the product stores runoff as 0, not NaN, over the sea; a mask taken from the field
    itself would also drop land cells with zero runoff.

    Parameters
    ----------
    cerra_valid : xarray.DataArray
        From :func:`cerra_mask`.

    Returns
    -------
    xarray.DataArray
        Boolean ``european_land`` on ``(lat, lon)``.
    """
    cache = processed / "european_land_mask.nc"
    if cache.exists():
        return xr.open_dataarray(cache)
    import regionmask
    land = regionmask.defined_regions.natural_earth_v5_1_2.land_10
    mask = land.mask(cerra_valid["lon"], cerra_valid["lat"]).notnull() & (cerra_valid > 0)
    mask.name = "european_land"
    mask.attrs = {"long_name": "land, inside the CERRA footprint",
                  "source": "regionmask natural_earth_v5_1_2.land_10 & cerra_valid_mask.nc"}
    mask.to_netcdf(cache)
    print(f"  cached European land mask -> {cache} "
          f"({int(mask.sum())}/{mask.size} cells, {100 * float(mask.mean()):.1f}%)", flush=True)
    return mask


rows = []

# ── The 9 chains, bias-corrected ───────────────────────────────────────────────
if "rcm" in args.sources:
    mask = cerra_mask()
    land = land_mask(mask)
    with open(args.models_json) as handle:
        chain_metadata = {chain["id"]: chain for chain in json.load(handle)["model_chains"]}
    for level in args.warming_levels:
        path = gwl_dir / f"mean_gwl_{int(level * 10)}_climate.nc"
        means = xr.open_dataset(path).sel(**EXTENT)
        assert means["pr"].attrs.get("units") == "mm d-1", f"{path.name}: pr is not mm d-1"
        assert means["tas"].attrs.get("units") == "K", f"{path.name}: tas is not K"
        # "kg m**-2" is the product's label for a daily total, i.e. mm d-1: no factor 86400.
        assert means["mrro"].attrs.get("units") == "kg m**-2", f"{path.name}: mrro is not kg m**-2"
        assert np.array_equal(means["lat"].values, mask["lat"].values) and \
               np.array_equal(means["lon"].values, mask["lon"].values), \
            "the GWL grid and the CERRA mask are not the same grid"
        for chain in [str(model) for model in means["model"].values]:
            base = chain[: -len(f"_{args.scenario}")] if chain.endswith(f"_{args.scenario}") else chain
            info = chain_metadata.get(base, {})
            rows.append(dict(source="rcm", model=base, ensemble=info.get("ensemble"),
                             rcm=info.get("rcm", base.split("_")[0]),
                             gcm=CORDEX_TO_CMIP5.get(info.get("gcm"), info.get("gcm")),
                             driving=False, warming_level=level,
                             tas_degC=area_mean(means["tas"].sel(model=chain), mask) - 273.15,
                             pr_mm_per_day=area_mean(means["pr"].sel(model=chain), mask),
                             # Land only (see land_mask); temperature keeps the domain mask.
                             mrro_mm_per_day=area_mean(means["mrro"].sel(model=chain), land)))
        print(f"  rcm  +{level}C  {len(means['model'])} chains from {path.name}", flush=True)

# ── The CMIP5 ensemble, raw ────────────────────────────────────────────────────
# The whole ensemble, not only the five driving GCMs.
if "gcm" in args.sources:
    import re
    from cordex2energy_analysis.analysis import load_warming_level_periods
    windows = load_warming_level_periods(
        processed / "cmip5_warming_levels_all_ens_1861_1900.yml", args.warming_levels, None)

    def members_with(variable):
        """Runs with an annual g025 file of one variable for the scenario.

        Parameters
        ----------
        variable : str
            cmip5-ng variable name.

        Returns
        -------
        set of tuple
            ``(model, ensemble)`` pairs.
        """
        pattern = re.compile(rf"{variable}_ann_(.+)_{args.scenario}_(r\d+i\d+p\d+)_g025$")
        return {match.groups() for path in Path(args.cmip5ng, variable).glob(
            f"{variable}_ann_*_{args.scenario}_*_g025.nc") if (match := pattern.match(path.stem))}

    # A run is usable only with both variables and a window at every requested level.
    usable = sorted(
        members_with("tas") & members_with("pr")
        & {key for key, scenarios in windows.items()
           if set(args.warming_levels) <= set(scenarios.get(args.scenario, {}))})
    print(f"  gcm  {len(usable)} runs over {len({model for model, _ in usable})} models", flush=True)
    missing_drivers = [gcm for gcm in DRIVING_GCMS if (gcm, DRIVING_GCMS[gcm]) not in set(usable)]
    assert not missing_drivers, f"driving GCM(s) not usable: {missing_drivers}"

    for gcm, member in usable:
        fields = {}
        for variable in ("tas", "pr"):
            hits = sorted(Path(args.cmip5ng, variable).glob(
                f"{variable}_ann_{gcm}_{args.scenario}_{member}_g025.nc"))
            assert len(hits) == 1, f"expected one {variable} file for {gcm} {member}, found {hits}"
            # cmip5-ng longitudes run 0-360; move them to -180-180 so that EXTENT keeps the
            # cells west of Greenwich.
            field = xr.open_dataset(hits[0])[variable]
            field = field.assign_coords(lon=((field["lon"] + 180) % 360) - 180).sortby("lon")
            fields[variable] = field.sel(**EXTENT)
            longitudes = fields[variable]["lon"]
            assert float(longitudes.min()) < -10 and float(longitudes.max()) > 37, \
                f"{hits[0].name}: the selected longitudes do not span the analysis box"
        driving = DRIVING_GCMS.get(gcm) == member
        for level in args.warming_levels:
            window = windows[(gcm, member)][args.scenario][level]
            years = slice(str(window["start_year"]), str(window["end_year"]))
            rows.append(dict(source="gcm", model=f"{gcm}_{member}", ensemble=member,
                             rcm=None, gcm=gcm, driving=driving, warming_level=level,
                             tas_degC=area_mean(fields["tas"].sel(year=years).mean("year")) - 273.15,
                             # cmip5-ng pr is in kg m-2 s-1.
                             pr_mm_per_day=area_mean(fields["pr"].sel(year=years).mean("year")) * 86400.0))
        if driving:
            print(f"       driver {gcm:<12} {member}", flush=True)

# ── CERRA, the reference the correction targets ────────────────────────────────
if "cerra" in args.sources:
    mask = cerra_mask()
    regridded = processed / "cerra_processed" / "regridded"
    tp = xr.open_zarr(regridded / "tp.zarr")["tp"].sel(**EXTENT).sel(time=slice(*REFERENCE_PERIOD))
    print(f"  cerra tp  {tp.sizes['time']} days, reading ~10 GB ...", flush=True)
    pr_mean = area_mean(tp.mean("time").compute(), mask)
    t2m = xr.open_zarr(regridded / "2t.zarr")["2t"].sel(**EXTENT).sel(time=slice(*REFERENCE_PERIOD))
    t2m = t2m.sel(height=2) if "height" in t2m.dims else t2m.squeeze("height", drop=True)
    print(f"  cerra 2t  {t2m.sizes['time']} steps, reading ~100 GB ...", flush=True)
    tas_mean = area_mean(t2m.mean("time").compute(), mask) - 273.15
    rows.append(dict(source="cerra", model="CERRA", ensemble=None, rcm=None, gcm=None,
                     driving=False, warming_level=np.nan,
                     tas_degC=tas_mean, pr_mm_per_day=pr_mean))
    # Plausibility checks against unit errors.
    assert 5.0 < tas_mean < 15.0, f"European mean CERRA temperature {tas_mean:.2f} degC is implausible"
    assert 1.0 < pr_mean < 5.0, f"European mean CERRA precipitation {pr_mean:.2f} mm/d is implausible"
    print(f"  cerra {REFERENCE_PERIOD[0]}-{REFERENCE_PERIOD[1]}: "
          f"{tas_mean:.2f} degC, {pr_mean:.2f} mm/d", flush=True)

frame = pd.DataFrame(rows)
if out_path.exists() and set(args.sources) != {"gcm", "rcm", "cerra"}:
    kept = pd.read_csv(out_path)
    frame = pd.concat([kept[~kept["source"].isin(args.sources)], frame], ignore_index=True)
frame = frame.sort_values(["source", "model", "warming_level"]).reset_index(drop=True)
frame.to_csv(out_path, index=False)
print(f"\nwrote {out_path}  ({len(frame)} rows: "
      + ", ".join(f"{count} {source}" for source, count in frame['source'].value_counts().items())
      + ")")
