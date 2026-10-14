"""Cos-latitude weighted global-mean surface temperature anomaly for the CMIP5 ensemble, RCP8.5.

Feeds Supplementary Fig. S4a, which shows where the five GCMs driving this paper's EURO-CORDEX
chains sit within CMIP5.  Every result in the paper is stated at a global warming level, and a
warming level is a *date* that depends entirely on how fast the driving GCM warms — so "are these
five a warm corner of the ensemble?" is a question the supplement has to answer.

Why the raw archive and not cmip5-ng
------------------------------------
The paper's warming levels are defined against **1861-1900** (that is the baseline of
``cmip5_warming_levels_all_ens_1861_1900.yml``, from which every GWL window in the paper comes).
The convenient archive ``/net/atmos/data/cmip5-ng`` is already annual and on one 2.5 deg grid, but
its series **start in 1870** and cannot express that baseline.  So this reads the raw monthly
archive, whose historical files start 185001.

Cost: ~29 GB for one member per model.  The output is a few kB, which the notebook loads instantly.
That asymmetry is the whole reason this is a script.

Weighting
---------
The global mean is **cos-latitude weighted**.  ``compute_gmst_anomalies`` in
``cordex2energy_analysis/analysis.py`` (and its twin in the ``cordex2energy`` package) takes a bare
``.mean(dim=["lat","lon"])`` with no weights, which over-weights the poles; on an anomaly it
partially cancels, but polar amplification means it does not cancel to zero.  Do not reuse it here.

Monthly -> annual is weighted by ``days_in_month`` on each file's own calendar, so the 360-day and
noleap models are not silently biased by treating February like July.

Example:
    python compute_cmip5_gmst.py --dry_run      # the 5 driving GCMs only
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

# The five GCMs driving this paper's chains, with the ENSEMBLE MEMBER each chain actually used.
# EC-EARTH is r12i1p1, not r1i1p1 — taking r1i1p1 for it would put a different realisation of a
# different climate on the figure and label it as the paper's driver.
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

paths = json.load(open(args.paths_file))
out_path = Path(paths["path_data_processed"]) / f"cmip5_gmst_annual_{args.scenario}.nc"
archive = Path(args.archive)
baseline = tuple(args.baseline)


def annual_global_mean(files):
    """Cos-lat weighted global mean of monthly `tas`, aggregated to calendar-weighted annual means."""
    # cftime, not datetime64: these span 1850-2100 on 360-day and noleap calendars, which numpy
    # cannot represent. Passed as a coder rather than the deprecated `use_cftime=` kwarg.
    ds = xr.open_mfdataset(files, combine="nested", concat_dim="time",
                           decode_times=xr.coders.CFDatetimeCoder(use_cftime=True),
                           coords="minimal", data_vars="minimal", compat="override",
                           parallel=False).sortby("time")
    tas = ds["tas"]
    weights = np.cos(np.deg2rad(tas["lat"]))
    field_mean = tas.weighted(weights).mean(dim=["lat", "lon"])
    days = field_mean["time"].dt.days_in_month
    annual = ((field_mean * days).groupby("time.year").sum()
              / days.groupby("time.year").sum())
    return annual.compute()


def member_for(model, available):
    """The driving member where this model drives a chain, else r1i1p1, else the first available."""
    if model in DRIVING_GCMS and DRIVING_GCMS[model] in available:
        return DRIVING_GCMS[model]
    return "r1i1p1" if "r1i1p1" in available else (sorted(available)[0] if available else None)


hist_root, scen_root = archive / "historical" / "Amon" / "tas", archive / args.scenario / "Amon" / "tas"
models = sorted({p.name for p in hist_root.iterdir() if p.is_dir()}
                & {p.name for p in scen_root.iterdir() if p.is_dir()})
if args.dry_run:
    models = [m for m in models if m in DRIVING_GCMS]
missing = set(DRIVING_GCMS) - set(models)
assert not missing, f"driving GCM(s) absent from {archive}: {sorted(missing)}"
print(f"{len(models)} models with both historical and {args.scenario} "
      f"({len(DRIVING_GCMS)} of them driving)")

series, failed = {}, {}
for i, model in enumerate(models, 1):
    shared = ({p.name for p in (hist_root / model).iterdir() if p.is_dir()}
              & {p.name for p in (scen_root / model).iterdir() if p.is_dir()})
    member = member_for(model, shared)
    if member is None:
        failed[model] = "no member with both experiments"
        continue
    files = (sorted((hist_root / model / member).glob("*.nc"))
             + sorted((scen_root / model / member).glob("*.nc")))
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
    except Exception as exc:                                    # noqa: BLE001 — report, do not abort
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
             "driving_gcms": ", ".join(f"{g}_{e}" for g, e in sorted(DRIVING_GCMS.items())),
             "n_models": len(frame.columns)}

# The anomaly must be zero over its own baseline, by construction — a non-zero value would mean the
# baseline slice missed the data (a calendar or units problem) rather than that the model is warm.
_base = out["gmst_anomaly"].sel(year=slice(*baseline)).mean("year")
assert np.allclose(_base.values, 0.0, atol=1e-9), \
    f"baseline mean is not zero: max |mean| = {np.abs(_base.values).max():.3g}"

out_path.parent.mkdir(parents=True, exist_ok=True)
out.to_netcdf(out_path)
print(f"\nwrote {out_path}  ({out_path.stat().st_size / 1e3:.0f} kB, "
      f"{len(frame.columns)} models x {len(frame)} years)")
if failed:
    print(f"\n{len(failed)} model(s) skipped:")
    for model, why in sorted(failed.items()):
        print(f"  {model:<18} {why}")
