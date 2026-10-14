"""Disk cache for the publication notebooks' expensive intermediates.

Two kinds of entry, both pickled under ``<path_data_analysis>/figure_cache/``:

* **Monthly totals** of the per-country run output and of the demand product. Every consumer of
  those series (``compute_country_gwl_field``, ``compute_c2e_gwl_field``, the Europe-wide table)
  slices whole-year GWL windows, filters whole months and sums by year, so calendar-month sums are
  an EXACT substitute for the 3-hourly / daily series -- at ~1/250 of the size, which is what turns
  every country field from a pass over the network disk into an in-memory sum.
* **Analysis tables** (event statistics, attenuation, ...) through :func:`cached`.

Every key carries the size and mtime of the raw input files, and the analysis tables' keys also
carry a hash of every non-plotting module in this package: a library fix (the severity-units fix,
say) invalidates every table automatically, while a styling change never does.

``C2E_FIGCACHE=refresh`` recomputes and overwrites every entry; ``C2E_FIGCACHE=off`` bypasses the
cache entirely. Each lookup prints one ``[cache]`` line, so a notebook's output records what it
reused.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import time
from pathlib import Path

import xarray as xr

PACKAGE = Path(__file__).resolve().parent
#: Bumped whenever the construction of a monthly entry changes, which no file stamp would notice.
MONTHLY_VERSION = 1


def cache_mode() -> str:
    mode = os.environ.get("C2E_FIGCACHE", "on").strip().lower()
    if mode not in ("on", "refresh", "off"):
        raise ValueError(f"C2E_FIGCACHE={mode!r}: expected on, refresh or off")
    return mode


def cache_root() -> Path:
    paths = json.loads((PACKAGE.parent / "paths.json").read_text())
    return Path(paths["path_data_analysis"]) / "figure_cache"


def code_fingerprint() -> str:
    """Hash of every analysis module in the package -- everything except the plotting modules."""
    digest = hashlib.sha256()
    for path in sorted(PACKAGE.glob("*.py")):
        if path.name.startswith("plotting") or path.name == "figure_cache.py":
            continue
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def file_stamps(files) -> list:
    """``(path, size, mtime_ns)`` per input file: a re-produced input is a different entry."""
    stamps = []
    for path in map(Path, files):
        stat = path.stat()
        stamps.append((str(path), stat.st_size, stat.st_mtime_ns))
    return stamps


def _jsonable(value):
    if hasattr(value, "to_dict"):                      # DataArray, Series, DataFrame
        return value.to_dict()
    if hasattr(value, "tolist"):                       # numpy
        return value.tolist()
    return str(value)


def cached(name: str, compute, *, key=None, files=(), code: bool = True):
    """``compute()``, or its pickled result from an earlier run with the same key.

    ``key``: every setting the result depends on (JSON-serialisable, or an xarray/pandas/numpy
    object). ``files``: the raw inputs it reads. ``code``: include the analysis-code hash.
    """
    mode = cache_mode()
    if mode == "off":
        return compute()
    parts = {"key": key, "files": file_stamps(files),
             "code": code_fingerprint() if code else None}
    digest = hashlib.sha256(json.dumps(parts, sort_keys=True, default=_jsonable).encode())
    path = cache_root() / f"{name}-{digest.hexdigest()[:16]}.pkl"
    if mode == "on" and path.exists():
        started = time.perf_counter()
        with open(path, "rb") as handle:
            value = pickle.load(handle)
        print(f"[cache] hit   {name}  ({path.stat().st_size / 1e6:.0f} MB, "
              f"{time.perf_counter() - started:.0f}s)")
        return value
    started = time.perf_counter()
    value = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(f".{os.getpid()}.part")     # two notebooks may race on one entry
    with open(partial, "wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    partial.replace(path)
    print(f"[cache] {'refresh' if mode == 'refresh' else 'miss  '} {name}  "
          f"(computed in {time.perf_counter() - started:.0f}s, saved {path.stat().st_size / 1e6:.0f} MB)")
    return value


def monthly_totals(data):
    """Calendar-month sums along ``time`` of every time-carrying variable, loaded into memory.

    Exact for any reduction built from whole months (see the module docstring). Variables without a
    ``time`` dimension are dropped: a resample would otherwise broadcast them along time.
    """
    if isinstance(data, xr.Dataset):
        data = data[[name for name, variable in data.data_vars.items() if "time" in variable.dims]]
    return data.resample(time="MS").sum("time", skipna=True).load()


def country_monthly(run_dirs: dict, files) -> dict:
    """``{model: Dataset}`` of monthly totals of the per-country run output, ``files`` merged.

    ``run_dirs``: ``{model: <residual_load>/<chain>_<scenario>/<run_tag>}``.
    """
    files = tuple(files)
    inputs = [Path(directory) / name for directory in run_dirs.values() for name in files]

    def compute():
        out = {}
        for model, directory in run_dirs.items():
            datasets = [xr.open_dataset(Path(directory) / name) for name in files]
            out[model] = monthly_totals(xr.merge(datasets))
            for dataset in datasets:
                dataset.close()
        return out

    label = "+".join(Path(name).stem for name in files)
    return cached(f"country_monthly_{label}", compute,
                  key={"version": MONTHLY_VERSION, "models": list(run_dirs), "files": files},
                  files=inputs, code=False)


def demand_monthly(path_demand, models, scenario: str, variables=("heating", "cooling")) -> dict:
    """``{variable: DataArray(model, country, time)}``: monthly sums of the DAILY-MEAN demand, GW.

    The notebooks' former ``_demand_daily`` followed by monthly sums, in float64 (that construction
    reduced in the file's float32), so a period-mean total is still ``sum * 24e-3`` TWh. Countries stay keyed by NAME; the ISO2 mapping
    is the caller's, as before.
    """
    inputs = [Path(path_demand) / f"{model}_{scenario}.nc" for model in models]

    def compute():
        out = {}
        for variable in variables:
            per_model = []
            for model, path in zip(models, inputs):
                with xr.open_dataset(path) as dataset:
                    # float64 BEFORE any reduction: the product is stored as float32, and summing
                    # ~7300 float32 daily means per window drifts by ~5e-6 relative.
                    daily = (dataset[f"{variable}_demand"].astype("float64")
                             .resample(time="1D").mean())
                    per_model.append(monthly_totals(daily).expand_dims(model=[model]))
            out[variable] = xr.concat(per_model, dim="model", join="outer")
        return out

    return cached("demand_monthly", compute,
                  key={"version": MONTHLY_VERSION, "models": list(models), "variables": list(variables)},
                  files=inputs, code=False)
