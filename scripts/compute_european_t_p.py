"""Area-mean European temperature and precipitation per model, at each warming level.

Feeds Supplementary Fig. S5, which asks how the nine chains differ in mean European climate and how
much of that they inherit from the five driving GCMs.  Three sources, one table:

    gcm    the 5 driving GCMs, RAW, from the cmip5-ng annual archive
    rcm    the 9 chains, BIAS-CORRECTED — the data every result in the paper is computed from
    cerra  the reanalysis the correction targets, over its 1991-2020 training period

Why the RCM side costs nothing
------------------------------
``energy/analysis/annual/gwl/mean_gwl_*_climate.nc`` already holds exactly this: per-chain
bias-corrected 20-year GWL-window means of ``tas`` (K) and ``pr`` (mm d-1) on the analysis grid,
each chain sliced with its own window.  Written by ``compute_climate_means_gwl.py``.  All that is
left is an area-weighted spatial mean.

Why the CERRA side is the expensive one
---------------------------------------
Both CERRA zarrs are chunked with the ENTIRE time axis in one chunk (``tp`` (13260,20,20), ``2t``
(107544,20,20,1)), so touching any cell pulls that tile's whole series.  Over the analysis extent
that is ~10 GB for ``tp`` and ~100 GB for ``2t``, decompression-bound.  Subsampling in time or
space does not help — the chunk is read either way.  Hence the cache.

Two corrections that make the three sources comparable
------------------------------------------------------
1. **A common valid mask.**  CERRA is NaN outside its native Lambert domain, but the bias-corrected
   CORDEX output is NaN-free because the correction script forward/back-filled its borders
   (``bias_correct_climate_model_data.py:192-216``) — those corners are nearest-neighbour
   extrapolation, not data.  Averaging each over "its own" valid cells would compare different
   samples.  The mask is built from CERRA and applied to CERRA and the chains alike.  Over this
   extent it costs only the far south-east corner.
2. **cos-latitude weighting**, since all three are regular lat-lon grids.

The GCM panel cannot take the 0.1 deg mask (it is 2.5 deg), so it uses the same geographic box with
cos-lat weights and no mask.  That is a real difference and the figure's caption says so: part of
the GCM offset from CERRA is resolution, not bias.

A third variable, land-only
---------------------------
``mrro`` rides along from the same GWL files for S5c, which asks the same question in runoff -- the
variable ``compute_hydropower.py`` consumes, and so the one behind the Hydro drivers of Figure 3d.
It is averaged over LAND alone because the product stores 0 over the sea; ``tas`` and ``pr`` keep
the domain mask.  The CMIP5 side has no runoff counterpart: HadGEM2-ES and EC-EARTH archive no
``mrro`` at all, so S5c drops the ensemble cloud rather than draw one missing two of five drivers.

Units are those of the bias-corrected product: temperature in degC, precipitation in mm d-1.
``tp`` is stored as ``kg m**-2``, which IS mm d-1 — the bias-correction script only relabels it, it
does not scale it.  Do not multiply by 86400 (that is for CORDEX ``pr`` in kg m-2 s-1).

Example:
    python compute_european_t_p.py --sources gcm rcm      # seconds
    python compute_european_t_p.py --sources cerra        # the long one
    python compute_european_t_p.py
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

DRIVING_GCMS = {"CNRM-CM5": "r1i1p1", "HadGEM2-ES": "r1i1p1", "NorESM1-M": "r1i1p1",
                "MPI-ESM-LR": "r1i1p1", "EC-EARTH": "r12i1p1"}
# CORDEX institute-prefixed names -> the CMIP5 names cmip5-ng and the warming-level YAML use.
CORDEX_TO_CMIP5 = {"CNRM-CERFACS-CNRM-CM5": "CNRM-CM5", "MOHC-HadGEM2-ES": "HadGEM2-ES",
                   "NCC-NorESM1-M": "NorESM1-M", "MPI-M-MPI-ESM-LR": "MPI-ESM-LR",
                   "ICHEC-EC-EARTH": "EC-EARTH"}
EXTENT = {"lat": slice(30, 72.6), "lon": slice(-12, 39)}      # the paper's analysis domain
REFERENCE_PERIOD = ("1991", "2020")                            # CERRA's bias-correction training window

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

paths = json.load(open(args.paths_file))
processed = Path(paths["path_data_processed"])
out_path = processed / "european_t_p_by_gwl.csv"
gwl_dir = processed / "energy" / "analysis" / "annual" / "gwl"


def area_mean(field, mask=None):
    """cos-latitude weighted spatial mean, optionally over a shared valid mask."""
    weights = np.cos(np.deg2rad(field["lat"]))
    if mask is not None:
        field = field.where(mask)
        weights = weights.broadcast_like(field).where(mask)
    return float(field.weighted(weights.fillna(0.0)).mean(dim=["lat", "lon"]))


def cerra_mask():
    """Cells where CERRA has data, on the 0.1 deg grid, over the analysis extent.

    Cached, because `tp.zarr` holds the whole time axis in one chunk per 20x20 tile: reading a
    SINGLE timestep over this extent still decompresses ~10 GB. The mask is a fixed geometric
    footprint, so paying that once and keeping the 40 kB result is the whole point.
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
    """Land cells of the analysis extent, on the 0.1 deg grid, inside the CERRA footprint.

    Runoff needs this and precipitation does not.  The bias-corrected product stores runoff as
    exactly 0 over the sea rather than NaN, so a domain mean is (land runoff x land fraction): over
    this extent that is 0.415 against 0.661 mm d-1, a factor of 1.59.  Precipitation falls on the
    sea too, so its domain mean is a real quantity and stays as it is.

    Natural Earth rather than a threshold on the field itself: a cell with genuinely zero runoff --
    karst, the drier south -- is land, and a mask that cannot tell it from the Atlantic would drop
    exactly the cells the figure is about.  Same coastline source as the country maps.
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
    meta = {m["id"]: m for m in json.load(open(args.models_json))["model_chains"]}
    for level in args.warming_levels:
        path = gwl_dir / f"mean_gwl_{int(level * 10)}_climate.nc"
        ds = xr.open_dataset(path).sel(**EXTENT)
        assert ds["pr"].attrs.get("units") == "mm d-1", f"{path.name}: pr is not mm d-1"
        assert ds["tas"].attrs.get("units") == "K", f"{path.name}: tas is not K"
        # `kg m**-2` is this pipeline's label for a daily total, i.e. mm d-1 -- the same relabelling
        # documented above for CERRA `tp`. Do NOT multiply by 86400: that is for raw CMIP5 `mrro`,
        # which really is kg m-2 s-1. The land mean lands at ~0.66 mm d-1, which is Europe.
        assert ds["mrro"].attrs.get("units") == "kg m**-2", f"{path.name}: mrro is not kg m**-2"
        assert np.array_equal(ds["lat"].values, mask["lat"].values) and \
               np.array_equal(ds["lon"].values, mask["lon"].values), \
            "the GWL grid and the CERRA mask are not the same grid"
        for chain in [str(m) for m in ds["model"].values]:
            base = chain[: -len(f"_{args.scenario}")] if chain.endswith(f"_{args.scenario}") else chain
            info = meta.get(base, {})
            rows.append(dict(source="rcm", model=base, ensemble=info.get("ensemble"),
                             rcm=info.get("rcm", base.split("_")[0]),
                             gcm=CORDEX_TO_CMIP5.get(info.get("gcm"), info.get("gcm")),
                             driving=False, warming_level=level,
                             tas_degC=area_mean(ds["tas"].sel(model=chain), mask) - 273.15,
                             pr_mm_per_day=area_mean(ds["pr"].sel(model=chain), mask),
                             # LAND only, unlike the two above -- see land_mask(). Temperature
                             # deliberately keeps the domain mask so S5c's x axis is the same
                             # quantity, and the same nine positions, as S5a's and S5b's.
                             mrro_mm_per_day=area_mean(ds["mrro"].sel(model=chain), land)))
        print(f"  rcm  +{level}C  {len(ds['model'])} chains from {path.name}", flush=True)

# ── The CMIP5 ensemble, raw ────────────────────────────────────────────────────
# The WHOLE ensemble, not just the five drivers: S5's grey cloud is what makes "where do our five
# sit" answerable, and at ~9.6 MB per annual g025 file the other ~33 models cost minutes.
if "gcm" in args.sources:
    import re
    from cordex2energy_analysis.analysis import load_warming_level_periods
    windows = load_warming_level_periods(
        processed / "cmip5_warming_levels_all_ens_1861_1900.yml", args.warming_levels, None)

    def members_with(var):
        """(model, ensemble) pairs having an annual g025 file for this variable and scenario."""
        pattern = re.compile(rf"{var}_ann_(.+)_{args.scenario}_(r\d+i\d+p\d+)_g025$")
        return {m.groups() for path in Path(args.cmip5ng, var).glob(
            f"{var}_ann_*_{args.scenario}_*_g025.nc") if (m := pattern.match(path.stem))}

    # A run is usable only if it has BOTH variables and a window at EVERY level being plotted —
    # otherwise a model would appear at some levels and not others and the cloud would be ragged.
    usable = sorted(
        members_with("tas") & members_with("pr")
        & {key for key, scenarios in windows.items()
           if set(args.warming_levels) <= set(scenarios.get(args.scenario, {}))})
    print(f"  gcm  {len(usable)} runs over {len({m for m, _ in usable})} models", flush=True)
    _missing = [g for g in DRIVING_GCMS if (g, DRIVING_GCMS[g]) not in set(usable)]
    assert not _missing, f"driving GCM(s) not usable: {_missing}"

    for gcm, member in usable:
        fields = {}
        for var in ("tas", "pr"):
            hits = sorted(Path(args.cmip5ng, var).glob(
                f"{var}_ann_{gcm}_{args.scenario}_{member}_g025.nc"))
            assert len(hits) == 1, f"expected one {var} file for {gcm} {member}, found {hits}"
            fields[var] = xr.open_dataset(hits[0])[var].sel(**EXTENT)
        driving = DRIVING_GCMS.get(gcm) == member
        for level in args.warming_levels:
            window = windows[(gcm, member)][args.scenario][level]
            years = slice(str(window["start_year"]), str(window["end_year"]))
            rows.append(dict(source="gcm", model=f"{gcm}_{member}", ensemble=member,
                             rcm=None, gcm=gcm, driving=driving, warming_level=level,
                             tas_degC=area_mean(fields["tas"].sel(year=years).mean("year")) - 273.15,
                             # cmip5-ng pr is kg m-2 s-1; CORDEX pr was converted the same way.
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
    # A unit slip here would be invisible on the figure — it would just move the reference point.
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
      + ", ".join(f"{n} {s}" for s, n in frame['source'].value_counts().items()) + ")")
