"""Annual global-mean surface temperature anomaly of the CMIP5 ensemble (1861-1900 baseline).

Shows where the five GCMs driving the EURO-CORDEX chains sit within CMIP5 (Supplementary Fig.
S1a). For every model with both a historical and a scenario run, one member is read from the raw
monthly archive: the member that drives a chain, else ``r1i1p1``, else the first available. The
raw archive is used because the paper's warming levels are defined against 1861-1900 and the
annual cmip5-ng series start in 1870.

The global mean is cosine-latitude weighted, and the monthly means are aggregated to annual means
weighted by ``days_in_month`` on each model's own calendar. Output:
``<processed>/cmip5_gmst_annual_<scenario>.nc``, variable ``gmst_anomaly`` (K) on
``(year, model)`` with ``model = <model>_<member>``.

Examples
--------
    python compute_cmip5_gmst.py --dry_run    # the five driving GCMs only
    python compute_cmip5_gmst.py
"""
import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

#: The five GCMs driving the chains, with the ensemble member each chain used (EC-EARTH: r12i1p1).
DRIVING_GCMS = {"CNRM-CM5": "r1i1p1", "HadGEM2-ES": "r1i1p1", "NorESM1-M": "r1i1p1",
                "MPI-ESM-LR": "r1i1p1", "EC-EARTH": "r12i1p1"}

parser = argparse.ArgumentParser(description="Cos-lat weighted CMIP5 GMST anomaly vs 1861-1900")
parser.add_argument("--paths_file", type=str,
                    default=str(Path(__file__).resolve().parent.parent / "paths.json"))
parser.add_argument("--archive", type=str, default="/net/atmos/data/cmip5")
parser.add_argument("--scenario", type=str, default="rcp85")
parser.add_argument("--baseline", type=int, nargs=2, default=[1861, 1900])
parser.add_argument("--end_year", type=int, default=2100,
                    help="HadGEM2-ES rcp85 runs to 2299; every series is cut here.")
parser.add_argument("--dry_run", action="store_true", help="the 5 driving GCMs only")
args = parser.parse_args()

with open(args.paths_file) as handle:
    paths = json.load(handle)
out_path = Path(paths["path_data_processed"]) / f"cmip5_gmst_annual_{args.scenario}.nc"
archive = Path(args.archive)
baseline = tuple(args.baseline)


def annual_global_mean(files):
    """Cosine-latitude weighted global mean of monthly ``tas``, as calendar-weighted annual means.

    Parameters
    ----------
    files : list of path-like
        Monthly ``tas`` files of one run (historical then scenario).

    Returns
    -------
    xarray.DataArray
        Annual means on ``year``, each month weighted by its length on the file's calendar.
    """
    # cftime: the series span 1850-2100 on 360-day and noleap calendars.
    dataset = xr.open_mfdataset(files, combine="nested", concat_dim="time",
                                decode_times=xr.coders.CFDatetimeCoder(use_cftime=True),
                                coords="minimal", data_vars="minimal", compat="override",
                                parallel=False).sortby("time")
    tas = dataset["tas"]
    weights = np.cos(np.deg2rad(tas["lat"]))
    field_mean = tas.weighted(weights).mean(dim=["lat", "lon"])
    days = field_mean["time"].dt.days_in_month
    annual = ((field_mean * days).groupby("time.year").sum()
              / days.groupby("time.year").sum())
    return annual.compute()


def member_for(model, available):
    """Ensemble member to read for one model.

    Parameters
    ----------
    model : str
        CMIP5 model name.
    available : set of str
        Members with both a historical and a scenario run.

    Returns
    -------
    str or None
        The driving member if the model drives a chain, else ``r1i1p1``, else the first
        available in sorted order; ``None`` if none is available.
    """
    if model in DRIVING_GCMS and DRIVING_GCMS[model] in available:
        return DRIVING_GCMS[model]
    return "r1i1p1" if "r1i1p1" in available else (sorted(available)[0] if available else None)


historical_root = archive / "historical" / "Amon" / "tas"
scenario_root = archive / args.scenario / "Amon" / "tas"
models = sorted({folder.name for folder in historical_root.iterdir() if folder.is_dir()}
                & {folder.name for folder in scenario_root.iterdir() if folder.is_dir()})
if args.dry_run:
    models = [model for model in models if model in DRIVING_GCMS]
missing = set(DRIVING_GCMS) - set(models)
assert not missing, f"driving GCM(s) absent from {archive}: {sorted(missing)}"
print(f"{len(models)} models with both historical and {args.scenario} "
      f"({len(DRIVING_GCMS)} of them driving)")

series, failed = {}, {}
for i, model in enumerate(models, 1):
    shared = ({folder.name for folder in (historical_root / model).iterdir() if folder.is_dir()}
              & {folder.name for folder in (scenario_root / model).iterdir() if folder.is_dir()})
    member = member_for(model, shared)
    if member is None:
        failed[model] = "no member with both experiments"
        continue
    files = (sorted((historical_root / model / member).glob("*.nc"))
             + sorted((scenario_root / model / member).glob("*.nc")))
    try:
        annual = annual_global_mean(files)
        annual = annual.sel(year=slice(baseline[0], args.end_year))
        reference = float(annual.sel(year=slice(*baseline)).mean())
        if not np.isfinite(reference):
            raise ValueError(f"no data in the {baseline[0]}-{baseline[1]} baseline")
        series[f"{model}_{member}"] = (annual - reference).to_series()
        print(f"  [{i:2d}/{len(models)}] {model:<18} {member:<8} "
              f"{int(annual.year.min())}-{int(annual.year.max())}  "
              f"{float(series[f'{model}_{member}'].iloc[-1]):+5.2f} K by {int(annual.year.max())}",
              flush=True)
    except Exception as exc:                                    # noqa: BLE001 — report and continue
        failed[model] = f"{type(exc).__name__}: {exc}"
        print(f"  [{i:2d}/{len(models)}] {model:<18} {member:<8} FAILED  {failed[model]}", flush=True)
        traceback.print_exc(file=sys.stderr)

assert series, "no model produced a series"
frame = pd.DataFrame(series)
frame.index.name = "year"
out = xr.Dataset({"gmst_anomaly": (("year", "model"), frame.values)},
                 coords={"year": frame.index.values, "model": list(frame.columns)})
out["gmst_anomaly"].attrs = {
    "units": "K", "long_name": "Global-mean surface air temperature anomaly",
    "baseline": f"{baseline[0]}-{baseline[1]}",
    "weighting": "cos(latitude)",
    "note": "monthly means aggregated to annual weighted by days_in_month on each model's calendar",
}
out.attrs = {"scenario": args.scenario, "archive": str(archive),
             "driving_gcms": ", ".join(f"{gcm}_{member}"
                                       for gcm, member in sorted(DRIVING_GCMS.items())),
             "n_models": len(frame.columns)}

# The anomaly is zero over its own baseline by construction; anything else means the baseline
# slice missed the data.
baseline_mean = out["gmst_anomaly"].sel(year=slice(*baseline)).mean("year")
assert np.allclose(baseline_mean.values, 0.0, atol=1e-9), \
    f"baseline mean is not zero: max |mean| = {np.abs(baseline_mean.values).max():.3g}"

out_path.parent.mkdir(parents=True, exist_ok=True)
out.to_netcdf(out_path)
print(f"\nwrote {out_path}  ({out_path.stat().st_size / 1e3:.0f} kB, "
      f"{len(frame.columns)} models x {len(frame)} years)")
if failed:
    print(f"\n{len(failed)} model(s) skipped:")
    for model, reason in sorted(failed.items()):
        print(f"  {model:<18} {reason}")
