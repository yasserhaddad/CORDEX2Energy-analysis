"""Disk cache for the publication notebooks' expensive intermediates.

Two kinds of entry, both pickled under ``<path_data_analysis>/figure_cache/``:

* **Monthly totals** of the per-country run output and of the demand product. Every consumer of
  those series (``compute_country_gwl_field``, ``compute_c2e_gwl_field``, the Europe-wide table)
  slices whole-year warming-level windows, filters whole months and sums by year, so
  calendar-month sums are an exact substitute for the 3-hourly or daily series at a fraction of
  the size.
* **Analysis tables** (event statistics, attenuation, ...) through :func:`cached`.

Every key carries the size and modification time of the raw input files; the analysis tables'
keys also carry a hash of every non-plotting module of the package, so a change to the analysis
code invalidates them while a styling change does not.

The environment variable ``C2E_FIGCACHE`` selects the mode: ``on`` (default), ``refresh``
(recompute and overwrite every entry) or ``off`` (bypass the cache). Each lookup prints one
``[cache]`` line.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import pickle
import time
from pathlib import Path

import xarray as xr

#: Directory of this package; ``paths.json`` is read from its parent.
PACKAGE = Path(__file__).resolve().parent
#: Version of the monthly-entry construction, part of their keys; bump it when that changes.
MONTHLY_VERSION = 1


def cache_mode() -> str:
    """Cache mode from ``C2E_FIGCACHE``.

    Returns
    -------
    {"on", "refresh", "off"}

    Raises
    ------
    ValueError
        If the variable holds anything else.
    """
    mode = os.environ.get("C2E_FIGCACHE", "on").strip().lower()
    if mode not in ("on", "refresh", "off"):
        raise ValueError(f"C2E_FIGCACHE={mode!r}: expected on, refresh or off")
    return mode


def cache_root() -> Path:
    """Cache directory, ``<path_data_analysis>/figure_cache`` from ``paths.json``.

    Returns
    -------
    pathlib.Path
    """
    paths = json.loads((PACKAGE.parent / "paths.json").read_text())
    return Path(paths["path_data_analysis"]) / "figure_cache"


def _code_without_docstrings(source: str) -> str:
    """The abstract syntax tree of a module, without docstrings, as a string.

    Comments, formatting and docstrings do not change it, so editing documentation does not
    invalidate the cache; any change to the code does.

    Parameters
    ----------
    source : str
        Module source.

    Returns
    -------
    str
    """
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
                    and isinstance(body[0].value.value, str):
                node.body = body[1:] or [ast.Pass()]
    return ast.dump(tree, include_attributes=False)


def code_fingerprint() -> str:
    """Hash of the code of every analysis module (not the plotting and figure modules).

    Returns
    -------
    str
        SHA-256 hex digest over the module names and their code without docstrings or comments,
        in sorted order.
    """
    digest = hashlib.sha256()
    for path in sorted(PACKAGE.glob("*.py")):
        if path.name.startswith(("plotting", "figures_")) or path.name == "figure_cache.py":
            continue
        digest.update(path.name.encode())
        digest.update(_code_without_docstrings(path.read_text()).encode())
    return digest.hexdigest()


def file_stamps(files) -> list:
    """Identity of each input file, so a re-produced input yields a different key.

    Parameters
    ----------
    files : iterable of path-like
        Input files.

    Returns
    -------
    list of tuple
        ``(path, size, mtime_ns)`` per file.
    """
    stamps = []
    for path in map(Path, files):
        stat = path.stat()
        stamps.append((str(path), stat.st_size, stat.st_mtime_ns))
    return stamps


def _jsonable(value):
    """JSON fallback for key values: ``to_dict()``, ``tolist()``, else ``str()``.

    Parameters
    ----------
    value : object
        Value ``json.dumps`` cannot serialise.

    Returns
    -------
    object
        A JSON-serialisable stand-in.
    """
    if hasattr(value, "to_dict"):                      # DataArray, Series, DataFrame
        return value.to_dict()
    if hasattr(value, "tolist"):                       # numpy
        return value.tolist()
    return str(value)


def cached(name: str, compute, *, key=None, files=(), code: bool = True):
    """``compute()``, or its pickled result from an earlier run with the same key.

    Parameters
    ----------
    name : str
        Entry name, the prefix of the cache file name.
    compute : callable
        Computes the value when it is not cached.
    key : object, optional
        Every setting the result depends on: JSON-serialisable, or xarray, pandas or numpy
        objects.
    files : iterable of path-like
        Raw inputs the result is computed from (see :func:`file_stamps`).
    code : bool
        Include :func:`code_fingerprint` in the key.

    Returns
    -------
    object
        The computed or cached value.
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
    """Calendar-month sums along ``time``, loaded into memory.

    Parameters
    ----------
    data : xarray.Dataset or xarray.DataArray
        Series with a ``time`` dimension. Dataset variables without ``time`` are dropped, since
        a resample would broadcast them along time.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        One step per month, labelled by its first day; NaNs are skipped.
    """
    if isinstance(data, xr.Dataset):
        data = data[[name for name, variable in data.data_vars.items() if "time" in variable.dims]]
    return data.resample(time="MS").sum("time", skipna=True).load()


def country_monthly(run_dirs: dict, files) -> dict:
    """Monthly totals of the per-country run output of every chain. Cached.

    Parameters
    ----------
    run_dirs : dict
        ``{chain: <residual_load>/<chain>_<scenario>/<run tag>}``.
    files : iterable of str
        Output files of each run folder, merged.

    Returns
    -------
    dict
        ``{chain: xarray.Dataset}`` of :func:`monthly_totals`.
    """
    files = tuple(files)
    inputs = [Path(directory) / name for directory in run_dirs.values() for name in files]

    def compute():
        """Monthly totals of every chain's run files."""
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
    """Monthly sums of the daily-mean heating and cooling demand of every chain. Cached.

    Parameters
    ----------
    path_demand : path-like
        Folder of the demand product, one ``<chain>_<scenario>.nc`` per chain.
    models : sequence of str
        Chains.
    scenario : str
        Emission scenario of the file names.
    variables : sequence of str
        Demand variables; ``"<variable>_demand"`` is read from each file.

    Returns
    -------
    dict
        ``{variable: DataArray (model, country, time)}`` in GW-days per month, float64, so a
        period sum times ``24e-3`` is TWh. Countries are keyed by name, as in the product.
    """
    inputs = [Path(path_demand) / f"{model}_{scenario}.nc" for model in models]

    def compute():
        """Monthly demand totals of every chain and variable."""
        out = {}
        for variable in variables:
            per_model = []
            for model, path in zip(models, inputs):
                with xr.open_dataset(path) as dataset:
                    # float64 before any reduction: the product is float32, and summing ~7300
                    # float32 daily means per window drifts by ~5e-6 relative.
                    daily = (dataset[f"{variable}_demand"].astype("float64")
                             .resample(time="1D").mean())
                    per_model.append(monthly_totals(daily).expand_dims(model=[model]))
            out[variable] = xr.concat(per_model, dim="model", join="outer")
        return out

    return cached("demand_monthly", compute,
                  key={"version": MONTHLY_VERSION, "models": list(models), "variables": list(variables)},
                  files=inputs, code=False)
