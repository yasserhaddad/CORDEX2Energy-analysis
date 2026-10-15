"""Compute TYNDP-scenario residual load and its generation/demand components.

Given a CORDEX climate-model chain (or the CERRA reanalysis), a climate
scenario, a TYNDP demand scenario (REF/GA/DE) with its target year, and a
TYNDP supply scenario (LOW/BE/HIGH) with its target year, this script:

1. Loads the TYNDP supply capacities (solar, onshore and offshore wind).
2. Loads the TYNDP demand table and builds the country-code mapping.
3. Loads the climate-driven capacity-factor, hydro and demand datasets.
4. Loads or computes (and caches) the TYNDP baseload for the requested
   demand scenario and year.
5. Computes the estimated generation of every technology.
6. Computes the residual load and its generation/demand components.
7. Optionally dispatches reservoir hydro and batteries.
8. Saves country-level and zonal residual load and components as NetCDF.

Two properties of the volumetric rescaling in step 6 matter when
interpreting a run.

**The rescaling cancels any per-country constant applied upstream.**
``k = target / E_simulated`` is computed from the same series it multiplies,
so a constant factor ``c`` on a country's demand rescales ``k`` by ``1/c``
and leaves the product identical. The cooling model's
``corrective_ratio_cool``, which anchors the cooling level to JRC-IDEES, is
such a constant (flat to 3.6e-07 across 262 968 hours), so anchored and
unanchored cooling give the identical residual-load system in any
target-rescaled run. Under ``--no-cooling-target`` ``k_cool = 1``, the
cancellation stops, and the run uses anchored cooling because it reads the
default anchored demand product. ``--no-ac-growth`` changes the shape, not a constant (its ratio
varies by 1.658 within a country), so it never cancels.

**``k`` scales the climate trend, not just the level.** It multiplies the
whole series, so a country's projected growth is amplified by its own
``k``. Over 1991-2020 on CERRA, ``k_cool`` ranges from about 0.5 (Germany)
to 7 (Ireland), a factor of ~16, and Italy (3.55) against Spain (0.87)
inverts the model's Spain>Italy ordering, which is the dominant control on
the projected southern summer deficit. The realised ``k_heat``/``k_cool``
table is therefore printed on every run.
"""

import argparse
import json
import os
import time
from pathlib import Path

import dask
import numpy as np
import pandas as pd
import xarray as xr

from cordex2energy_analysis.power_system import (
    aggregate_battery_capacities_by_zone,
    aggregate_by_balancing_zones,
    apply_battery_dispatch,
    apply_prosumer_battery_dispatch,
    apply_reservoir_dispatch,
    build_c2e_name_to_iso2,
    build_thermal_targets,
    compute_estimated_generation,
    compute_residual_load,
    compute_or_load_baseload,
    load_c2e_demand_nc,
    load_tyndp_battery_capacities,
    load_tyndp_country_codes,
    load_tyndp_market_profiles,
    load_tyndp_supply_capacities,
    preprocess_hydro_ror,
)
from cordex2energy.utils.constants import REFERENCE_PERIOD

# Limit internal threading of numerical libraries so that Dask can manage
# parallelism at the worker level without contention.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# CLI supply scenario -> column suffix in the TYNDP supply Excel file.
SUPPLY_SCENARIO_COLUMNS = {"LOW": "LOW", "BE": "Best Estimate", "HIGH": "HIGH"}

# Demand model -> token in the run tag and the baseload cache key.
DEMAND_MODEL_TAGS = {"demand.ninja": "dninja", "cordex2energy_demand": "c2edem"}

# Years of the CERRA record used for the reference run.
CERRA_YEARS = ("1991", "2020")

# Capacity [MW] x capacity factor (3-hourly mean) -> GWh per 3-hour step:
# 3 h x 1e-3 (MWh -> GWh).
GENERATION_SCALE = 3 * 1e-3

# Variables returned by the battery dispatch functions next to the residual load.
BATTERY_STATE_VARIABLES = ["residual_load_with_batteries", "soc_gwh", "battery_dispatch_gwh"]


def to_iso2(labels, name_to_iso2: dict, strip: bool = True) -> list:
    """Map country labels to ISO2 codes, leaving unknown labels unchanged.

    Parameters
    ----------
    labels : iterable
        Country labels (full names or codes).
    name_to_iso2 : dict
        Full-name to ISO2 mapping.
    strip : bool, optional
        Convert each label to a stripped string before the lookup.

    Returns
    -------
    list
        One code per label.
    """
    if strip:
        return [name_to_iso2.get(str(label).strip(), str(label).strip()) for label in labels]
    return [name_to_iso2.get(label, label) for label in labels]


def country_time_dataset(
    table: pd.DataFrame,
    name_to_iso2: dict,
    variable: str,
    strip: bool = True,
) -> xr.Dataset:
    """Convert a country-by-time table into a one-variable Dataset.

    Parameters
    ----------
    table : pandas.DataFrame
        Countries as the index, timestamps as the columns.
    name_to_iso2 : dict
        Full-name to ISO2 mapping applied to the index.
    variable : str
        Name of the data variable.
    strip : bool, optional
        Passed to :func:`to_iso2`.

    Returns
    -------
    xarray.Dataset
        ``variable`` on dimensions ``(country, time)``.
    """
    return xr.DataArray(
        table.values,
        dims=["country", "time"],
        coords={
            "country": to_iso2(table.index, name_to_iso2, strip=strip),
            "time": pd.to_datetime(table.columns),
        },
        name=variable,
    ).to_dataset()


def relabel_countries(dataset: xr.Dataset, name_to_iso2: dict) -> xr.Dataset:
    """Replace full country names on the ``country`` coordinate by ISO2 codes.

    Parameters
    ----------
    dataset : xarray.Dataset
        Dataset with a ``country`` coordinate of full names.
    name_to_iso2 : dict
        Full-name to ISO2 mapping.

    Returns
    -------
    xarray.Dataset
        The same data with ISO2 country labels.
    """
    return dataset.assign_coords(country=to_iso2(dataset.country.values, name_to_iso2))


def load_cordex_energy_data(
    path_c2e: Path,
    path_energy_europe: Path,
    climate_model: str,
    model_prefix: str,
    climate_scenario: str,
    rcp_suffix: str,
    turbine_name: str,
    name_to_iso2: dict,
    demand_model: str = "demand.ninja",
) -> dict:
    """Load the CORDEX-derived energy datasets of one model chain.

    Parameters
    ----------
    path_c2e : Path
        Root of the C2E outputs (one subdirectory per model).
    path_energy_europe : Path
        Root of the European energy outputs (``pv/``, ``wind_power/``,
        ``demand/``).
    climate_model : str
        Full model-chain identifier.
    model_prefix : str
        Model-chain identifier without the ensemble member (``_r1i1p1``).
    climate_scenario : str
        RCP scenario, e.g. ``"rcp85"``.
    rcp_suffix : str
        Last two characters of the RCP scenario, e.g. ``"85"``.
    turbine_name : str
        Turbine model with ``/`` replaced by ``_``.
    name_to_iso2 : dict
        Full-name to ISO2 mapping for the country labels.
    demand_model : str, optional
        ``"demand.ninja"`` (3-hourly C2E CSVs) or ``"cordex2energy_demand"``
        (hourly JRC NetCDF).

    Returns
    -------
    dict
        Keys ``pv``, ``wind_onshore``, ``wind_offshore``, ``hydro_ror``,
        ``hydro_inflow``, ``heating_demand`` and ``cooling_demand``.

    Notes
    -----
    The CORDEX PV time labels are moved back by 1 h 30 min to correct an
    offset in the PV outputs.
    """
    print("  Loading PV capacity factors …")
    pv = xr.open_dataset(
        path_energy_europe
        / "pv"
        / f"{climate_model}_{climate_scenario}"
        / f"pv_tyndp_agg_{climate_model}_{climate_scenario}.nc"
    )
    pv["time"] = pv.time - pd.Timedelta("1h30min")

    print("  Loading onshore wind capacity factors …")
    wind_onshore = xr.open_dataset(
        path_energy_europe
        / "wind_power"
        / f"{climate_model}_{climate_scenario}"
        / f"wind_power_tyndp_agg_{climate_model}_{climate_scenario}_{turbine_name}.nc"
    )

    print("  Loading offshore wind capacity factors …")
    # Offshore wind uses full country names in the 'country' dimension.
    wind_offshore = relabel_countries(
        xr.open_dataset(
            path_energy_europe
            / "wind_power"
            / f"{climate_model}_{climate_scenario}"
            / f"wind_power_offshore_{climate_model}_{climate_scenario}_{turbine_name}.nc"
        ),
        name_to_iso2,
    )

    model_dir = path_c2e / model_prefix / "output_variables"

    print("  Loading hydro run-of-river …")
    hydro_ror_table = pd.read_csv(
        model_dir / f"hydro_ror_CORDEX_{model_prefix}_{rcp_suffix}.csv",
        index_col=0,
    )
    hydro_ror = preprocess_hydro_ror(
        hydro_ror_table, name_to_iso2, freq="3h", var_name="ror"
    )

    print("  Loading hydro inflow …")
    hydro_inflow = country_time_dataset(
        pd.read_csv(
            model_dir / f"hydro_inflow_CORDEX_{model_prefix}_{rcp_suffix}.csv",
            index_col=0,
        ),
        name_to_iso2,
        "inflow",
    )

    if demand_model == "cordex2energy_demand":
        print("  Loading heating/cooling demand (JRC cordex2energy_demand NetCDF) …")
        heating_demand, cooling_demand = load_c2e_demand_nc(
            path_energy_europe / "demand" / f"{climate_model}_{climate_scenario}.nc",
            name_to_iso2,
        )
    else:
        print("  Loading heating demand …")
        heating_demand = country_time_dataset(
            pd.read_csv(
                model_dir / f"heating-demand_CORDEX_{model_prefix}_{rcp_suffix}.csv",
                index_col=0,
            ),
            name_to_iso2,
            "heating_demand",
            strip=False,
        )

        print("  Loading cooling demand …")
        cooling_demand = country_time_dataset(
            pd.read_csv(
                model_dir / f"cooling-demand_CORDEX_{model_prefix}_{rcp_suffix}.csv",
                index_col=0,
            ),
            name_to_iso2,
            "cooling_demand",
            strip=False,
        )

    return {
        "pv": pv,
        "wind_onshore": wind_onshore,
        "wind_offshore": wind_offshore,
        "hydro_ror": hydro_ror,
        "hydro_inflow": hydro_inflow,
        "heating_demand": heating_demand,
        "cooling_demand": cooling_demand,
    }


def load_cerra_energy_data(
    path_c2e: Path,
    path_energy_europe: Path,
    turbine_name: str,
    name_to_iso2: dict,
    demand_model: str = "demand.ninja",
) -> dict:
    """Load the CERRA-derived energy datasets, restricted to 1991-2020.

    Parameters
    ----------
    path_c2e : Path
        Root of the C2E outputs (contains ``CERRA/``).
    path_energy_europe : Path
        Root of the European energy outputs (``pv/``, ``wind_power/``,
        ``demand/``).
    turbine_name : str
        Turbine model with ``/`` replaced by ``_``.
    name_to_iso2 : dict
        Full-name to ISO2 mapping for the country labels.
    demand_model : str, optional
        ``"demand.ninja"`` (3-hourly C2E CSVs) or ``"cordex2energy_demand"``
        (hourly JRC NetCDF).

    Returns
    -------
    dict
        Same keys as :func:`load_cordex_energy_data`. Offshore wind is all
        zeros when no CERRA offshore file exists.
    """
    print("  Loading CERRA PV capacity factors …")
    pv = xr.open_dataset(
        path_energy_europe / "pv" / "cerra" / "pv_tyndp_agg_cerra.nc"
    )
    # Like CORDEX, the CERRA PV product is labelled at the window midpoint; relabel to the
    # window start so that PV covers the same 3 h as demand.
    if pd.Timestamp(pv.time.values[0]).strftime("%H:%M") != "01:30":
        raise ValueError(
            "pv_tyndp_agg_cerra.nc is not labelled at window midpoints; recompute it with "
            "CORDEX2Energy/scripts/conversion_energy/compute_pv.py."
        )
    pv["time"] = pv.time - pd.Timedelta("1h30min")

    print("  Loading CERRA onshore wind capacity factors …")
    wind_onshore = xr.open_dataset(
        path_energy_europe / "wind_power" / "cerra"
        / f"wind_power_tyndp_agg_cerra_{turbine_name}.nc"
    )

    print("  Loading CERRA offshore wind capacity factors …")
    path_wind_offshore = (
        path_energy_europe / "wind_power" / "cerra"
        / f"wind_power_offshore_cerra_{turbine_name}.nc"
    )
    if path_wind_offshore.exists():
        wind_offshore = relabel_countries(xr.open_dataset(path_wind_offshore), name_to_iso2)
    else:
        print("    [INFO] Offshore wind not available for CERRA — using zeros.")
        wind_offshore = xr.zeros_like(wind_onshore)

    cerra_dir = path_c2e / "CERRA" / "output_variables"

    print("  Loading CERRA hydro run-of-river …")
    hydro_ror_table = pd.read_csv(cerra_dir / "hydro_ror_CERRA.csv", index_col=0)
    hydro_ror = preprocess_hydro_ror(
        hydro_ror_table, name_to_iso2, freq="3h", var_name="ror"
    ).sel(time=slice(*CERRA_YEARS))

    print("  Loading CERRA hydro inflow …")
    hydro_inflow = country_time_dataset(
        pd.read_csv(cerra_dir / "hydro_inflow_CERRA.csv", index_col=0),
        name_to_iso2,
        "inflow",
    )

    if demand_model == "cordex2energy_demand":
        print("  Loading CERRA heating/cooling demand (JRC cordex2energy_demand NetCDF) …")
        heating_demand, cooling_demand = load_c2e_demand_nc(
            path_energy_europe / "demand" / "cerra.nc",
            name_to_iso2,
            time_slice=REFERENCE_PERIOD,
        )
    else:
        print("  Loading CERRA heating demand …")
        heating_demand = country_time_dataset(
            pd.read_csv(cerra_dir / "heating-demand_CERRA.csv", index_col=0),
            name_to_iso2,
            "heating_demand",
        )

        print("  Loading CERRA cooling demand …")
        cooling_demand = country_time_dataset(
            pd.read_csv(cerra_dir / "cooling-demand_CERRA.csv", index_col=0),
            name_to_iso2,
            "cooling_demand",
        )

    datasets = {
        "pv": pv,
        "wind_onshore": wind_onshore,
        "wind_offshore": wind_offshore,
        "hydro_ror": hydro_ror,
        "hydro_inflow": hydro_inflow,
        "heating_demand": heating_demand,
        "cooling_demand": cooling_demand,
    }
    return {key: data.sel(time=slice(*CERRA_YEARS)) for key, data in datasets.items()}


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser for all run options.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Compute TYNDP-scenario residual load from CORDEX climate projections "
            "or the CERRA reanalysis."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--paths_file",
        type=str,
        default="../../paths_so4.json",
        help="Path to the JSON file containing data paths.",
    )
    parser.add_argument(
        "--climate_model",
        type=str,
        default="CNRM-ALADIN63_CNRM-CERFACS-CNRM-CM5_r1i1p1",
        help=(
            "CORDEX climate model chain identifier "
            "(e.g. CNRM-ALADIN63_CNRM-CERFACS-CNRM-CM5_r1i1p1)."
        ),
    )
    parser.add_argument(
        "--climate_scenario",
        type=str,
        default="rcp85",
        choices=["rcp26", "rcp45", "rcp85", "none"],
        help="CORDEX RCP climate scenario. Use 'none' when --climate_model is 'cerra'.",
    )
    parser.add_argument(
        "--demand_scenario",
        type=str,
        default="DE",
        choices=["REF", "GA", "DE"],
        help="TYNDP demand scenario (REF=Reference, GA=Global Ambition, DE=Distributed Energy).",
    )
    parser.add_argument(
        "--demand_year",
        type=int,
        default=2050,
        choices=[2019, 2040, 2050],
        help="Target year for the demand scenario (2019 for REF; 2040 or 2050 for GA/DE).",
    )
    parser.add_argument(
        "--demand_model",
        type=str,
        default="demand.ninja",
        choices=["demand.ninja", "cordex2energy_demand"],
        help=(
            "Source of the climate-driven heating/cooling demand. "
            "'demand.ninja' (default) uses the 3-hourly c2e CSVs; "
            "'cordex2energy_demand' uses the hourly JRC-coefficient NetCDFs "
            "(energy/europe/demand/*.nc). The choice is encoded in the run tag "
            "and the baseload cache key."
        ),
    )
    parser.add_argument(
        "--legacy_thermal",
        action="store_true",
        default=False,
        help=(
            "Reproduce the historical thermal handling: lump heating+cooling "
            "into one target, compute the rescale factor over the whole series "
            "(not the reference period), and disable baseload modulation. "
            "For validation against pre-change runs."
        ),
    )
    parser.add_argument(
        "--no-cooling-target",
        dest="rescale_cooling",
        action="store_false",
        default=True,
        help=(
            "Do not rescale cooling to its TYNDP annual target (k_cool = 1), keeping the "
            "physically derived cooling level from the demand model. Heating stays pinned to its "
            "own target and the baseload is unchanged, so this isolates the effect of the cooling "
            "target alone. The target rescaling amplifies each country's climate growth by its own "
            "k_cool, which spans ~16x across Europe (about 0.5 in Germany to 7 in Ireland; Italy "
            "3.55, Spain 0.87) and inverts the "
            "model's Spain>Italy ranking — that amplification is what this flag removes. Output "
            "goes to a '_freecool' run tag. Sector-split cooling only (cordex2energy_demand)."
        ),
    )
    parser.add_argument(
        "--supply_scenario",
        type=str,
        default="HIGH",
        choices=["LOW", "BE", "HIGH"],
        help=(
            "TYNDP supply capacity scenario "
            "(LOW, BE=Best Estimate, HIGH). "
            "Maps to the column suffix in the TYNDP supply Excel file."
        ),
    )
    parser.add_argument(
        "--supply_year",
        type=int,
        default=2050,
        choices=[2030, 2040, 2050],
        help="Target year for the supply capacity scenario.",
    )
    parser.add_argument(
        "--rolling_window_residual_load",
        type=int,
        default=8,
        help=(
            "Rolling window size in timesteps for smoothing the residual load. "
            "Set to 1 or None to disable smoothing."
        ),
    )
    parser.add_argument(
        "--turbine",
        type=str,
        default="E-126/7580",
        help=(
            "Wind turbine model used in the CORDEX energy files "
            "(e.g. E-126/7580). Slashes are replaced by underscores in file names."
        ),
    )
    parser.add_argument(
        "--floor_fraction",
        type=float,
        default=0.10,
        help="Floor cap for baseload as a fraction of mean market demand (default 0.10).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=15,
        help="Number of Dask workers.",
    )
    parser.add_argument(
        "--use_reservoir_hydro",
        action="store_true",
        default=False,
        help=(
            "Apply optimal reservoir hydro dispatch (Gurobi QP, annual cyclical) "
            "to the country-level residual load before zonal aggregation. "
            "Requires a Gurobi licence and reservoir capacity CSVs under "
            "<path_data_extremes>/reservoir_capacity/."
        ),
    )
    parser.add_argument(
        "--reservoir_min_storage_fraction",
        type=float,
        default=0.1,
        help=(
            "Minimum reservoir storage level as a fraction of maximum capacity "
            "(e.g. 0.05).  Passed to the Gurobi QP as a lower bound on storage "
            "at every time step."
        ),
    )
    parser.add_argument(
        "--reservoir_efficiency",
        type=float,
        default=0.9,
        help=(
            "Reservoir turbine (water-to-wire) conversion efficiency in (0, 1]. "
            "Generating G electrical GWh draws G / efficiency from storage "
            "(default 0.9)."
        ),
    )
    parser.add_argument(
        "--use_batteries",
        action="store_true",
        default=False,
        help="Apply utility battery dispatch (with lookahead) to the country-level residual load.",
    )
    parser.add_argument(
        "--use_prosumer_batteries",
        action="store_true",
        default=False,
        help=(
            "Also apply PV-aware prosumer battery dispatch before utility dispatch. "
            "Requires --use_batteries."
        ),
    )
    parser.add_argument(
        "--battery_duration_h",
        type=float,
        default=4.0,
        help="Battery energy-to-power ratio in hours (default 4).",
    )
    parser.add_argument(
        "--battery_roundtrip_eff",
        type=float,
        default=0.85,
        help="Round-trip efficiency for batteries (default 0.85).",
    )
    parser.add_argument(
        "--battery_initial_soc",
        type=float,
        default=0.50,
        help="Initial state-of-charge as a fraction of capacity (default 0.5).",
    )
    parser.add_argument(
        "--battery_dispatch_lookahead_h",
        type=float,
        default=24.0,
        help=(
            "Utility battery foresight window in hours (default 24). "
            "For --battery_dispatch_method heuristic this is the greedy-discharge "
            "rationing window; for 'optimal' it is the extra look-ahead solved "
            "beyond --battery_commit_h and then discarded (rolling day-ahead overlap)."
        ),
    )
    parser.add_argument(
        "--battery_dispatch_method",
        type=str,
        default="heuristic",
        choices=["heuristic", "optimal"],
        help=(
            "Utility battery dispatch method (prosumer batteries are always "
            "heuristic). 'heuristic' (default) uses the greedy lookahead loop; "
            "'optimal' runs a rolling day-ahead QP (Gurobi) that minimises squared "
            "net residual load over a "
            "--battery_commit_h + --battery_dispatch_lookahead_h window, committing "
            "only the first --battery_commit_h. Requires a Gurobi licence."
        ),
    )
    parser.add_argument(
        "--battery_commit_h",
        type=float,
        default=24.0,
        help=(
            "Committed (kept) horizon in hours per rolling day-ahead solve, and the "
            "window slide step. Used only with --battery_dispatch_method optimal "
            "(default 24)."
        ),
    )
    parser.add_argument(
        "--battery_min_soc_fraction",
        type=float,
        default=None,
        help=(
            "Minimum utility battery state-of-charge as a fraction of capacity, used "
            "with --battery_dispatch_method optimal and with --flexibility_scheme system. "
            "Default: no floor."
        ),
    )
    parser.add_argument(
        "--flexibility_scheme",
        type=str,
        default="market",
        choices=["market", "system"],
        help=(
            "How storage is operated. 'market' (default) models each asset as it is "
            "operated in a market: prosumer batteries use the behind-the-meter "
            "heuristic and utility batteries use --battery_dispatch_method at "
            "--utility_dispatch_level. 'system' instead dispatches ALL flexibility "
            "under one scheme consistent with the reservoir: prosumer + utility "
            "batteries are lumped into a single storage asset and optimised with the "
            "annual-cyclical perfect-foresight QP at country level (net-load-event "
            "mitigation). Under 'system', --battery_dispatch_method and "
            "--utility_dispatch_level are ignored."
        ),
    )
    parser.add_argument(
        "--utility_dispatch_level",
        type=str,
        default="zone",
        choices=["zone", "country"],
        help=(
            "Spatial level for utility battery dispatch. 'zone' (default) pools "
            "utility batteries across the 6 real-time balancing zones (grid/merchant "
            "assets, ~5x fewer QP solves); 'country' dispatches per country. Prosumer "
            "batteries and reservoir hydro are always country-level."
        ),
    )
    return parser


def build_run_tag(args: argparse.Namespace, demand_model_tag: str, turbine_name: str) -> str:
    """Build the run tag that names the output subfolder.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.
    demand_model_tag : str
        Short demand-model token (``dninja`` or ``c2edem``).
    turbine_name : str
        Turbine model with ``/`` replaced by ``_``.

    Returns
    -------
    str
        E.g. ``DE_2050_c2edem_LOW_2050_freecool_reservoirdispatch_batteries_prosumer_sysflex_SWT120_3600``.

    Notes
    -----
    Every option that changes the residual-load system is encoded, so two
    different systems never share a folder. The demand-model token and the
    turbine are always present; ``_utilcountry`` is added only for the
    non-default utility level, and under ``--flexibility_scheme system`` the
    battery method and level flags do not apply and are not encoded.
    """
    run_tag = (
        f"{args.demand_scenario}_{args.demand_year}_{demand_model_tag}"
        f"_{args.supply_scenario}_{args.supply_year}"
    )
    if args.legacy_thermal:
        run_tag += "_legacythermal"
    if not args.rescale_cooling:
        run_tag += "_freecool"
    if args.rolling_window_residual_load and args.rolling_window_residual_load > 1:
        run_tag += f"_rlsmooth_{args.rolling_window_residual_load}"
    if args.use_reservoir_hydro:
        run_tag += "_reservoirdispatch"
    if args.use_batteries:
        run_tag += "_batteries"
        if args.use_prosumer_batteries:
            run_tag += "_prosumer"
        if args.flexibility_scheme == "system":
            run_tag += "_sysflex"
        else:
            if args.battery_dispatch_method == "optimal":
                run_tag += "_optbatt"
            if args.utility_dispatch_level == "country":
                run_tag += "_utilcountry"
    run_tag += f"_{turbine_name}"
    return run_tag


def as_data_array(data, variable: str) -> xr.DataArray:
    """Return ``data[variable]`` for a Dataset, or ``data`` itself otherwise.

    Parameters
    ----------
    data : xarray.Dataset or xarray.DataArray
        Demand series.
    variable : str
        Variable to select from a Dataset.

    Returns
    -------
    xarray.DataArray
        The demand series.
    """
    return data[variable] if isinstance(data, xr.Dataset) else data


def annual_energy_twh(series: xr.DataArray, reference_period) -> float:
    """Mean annual energy of an hourly GW series, in TWh/yr.

    Parameters
    ----------
    series : xarray.DataArray
        Hourly demand of one country in GW (cordex2energy_demand units).
    reference_period : tuple of str or None
        Period to average over; the whole series when ``None``.

    Returns
    -------
    float
        Sum over the period x 1e-3, divided by the number of distinct years
        (at least 1).
    """
    window = (series.sel(time=slice(*REFERENCE_PERIOD))
              if reference_period else series)
    year_count = max(1, len(np.unique(window["time"].dt.year.values)))
    return float(window.sum()) * 1e-3 / year_count


def print_rescale_factors(
    heating_demand,
    cooling_demand,
    thermal_targets: dict,
    reference_period,
    rescale_cooling: bool,
) -> None:
    """Print the realised heating and cooling rescale factors per country.

    Parameters
    ----------
    heating_demand, cooling_demand : xarray.Dataset or xarray.DataArray
        Hourly demand in GW with an ISO2 ``country`` coordinate.
    thermal_targets : dict
        Output of ``build_thermal_targets`` (``heating``/``cooling`` dicts
        of TWh/yr by ISO2 code).
    reference_period : tuple of str or None
        Period over which the model energy is averaged.
    rescale_cooling : bool
        Whether cooling is rescaled to its target; ``k_cool`` is 1 otherwise.

    Returns
    -------
    None

    Notes
    -----
    ``k = target / model energy`` multiplies the whole series, so it scales
    each country's climate growth by the same factor. The factors are
    computed from the unsmoothed series; the centred rolling mean applied in
    ``compute_residual_load`` preserves the annual mean, so they match the
    applied ``k`` to well under a percent.
    """
    heating = as_data_array(heating_demand, "heating_demand")
    cooling = as_data_array(cooling_demand, "cooling_demand")

    available = set(map(str, heating.country.values))
    rows = []
    for iso2 in sorted(set(thermal_targets["heating"]) | set(thermal_targets["cooling"])):
        if iso2 not in available:
            continue
        model_heating = annual_energy_twh(heating.sel(country=iso2), reference_period)
        model_cooling = annual_energy_twh(cooling.sel(country=iso2), reference_period)
        target_heating = thermal_targets["heating"].get(iso2, np.nan)
        target_cooling = thermal_targets["cooling"].get(iso2, np.nan)
        rows.append(dict(
            iso=iso2, model_heat=model_heating, target_heat=target_heating,
            k_heat=target_heating / model_heating if model_heating > 0 else np.nan,
            model_cool=model_cooling, target_cool=target_cooling,
            k_cool=((target_cooling / model_cooling if model_cooling > 0 else np.nan)
                    if rescale_cooling else 1.0)))
    if rows:
        factors = pd.DataFrame(rows).set_index("iso")
        print(f"\n  Realised rescale factors over {REFERENCE_PERIOD} "
              f"(TWh/yr, k = target/model):")
        print(factors.round(2).to_string().replace("\n", "\n    "))
        for label, column in (("k_heat", factors.k_heat), ("k_cool", factors.k_cool)):
            note = ("   (rescaling OFF -> all 1.0)"
                    if label == "k_cool" and not rescale_cooling else "")
            print(f"    {label}  min {column.min():.2f}  median {column.median():.2f}"
                  f"  max {column.max():.2f}{note}")


def residual_load_after_batteries(dispatch_result: xr.Dataset) -> xr.Dataset:
    """Keep only the post-dispatch residual load of a battery dispatch result.

    Parameters
    ----------
    dispatch_result : xarray.Dataset
        Output of ``apply_battery_dispatch`` or
        ``apply_prosumer_battery_dispatch``.

    Returns
    -------
    xarray.Dataset
        ``residual_load`` replaced by ``residual_load_with_batteries``, with
        the battery state variables dropped.
    """
    return dispatch_result.assign(
        residual_load=dispatch_result["residual_load_with_batteries"]
    ).drop_vars(BATTERY_STATE_VARIABLES)


def main() -> None:
    """Run one residual-load computation from the command line.

    Returns
    -------
    None
        Writes ``residual_load.nc``, ``residual_load_zonal.nc``,
        ``components.nc`` and ``components_zonal.nc`` to
        ``<path_data_processed>/extremes/residual_load/<model>/<run tag>/``.

    Notes
    -----
    Generation is in GWh per 3-hour step. The hourly cordex2energy_demand
    NetCDFs are only resampled to 3-hourly sums; the 3-hourly demand.ninja
    CSVs are cubic-upsampled to 1 h first.
    """
    start_time = time.time()
    parser = build_parser()
    args = parser.parse_args()
    if args.use_prosumer_batteries and not args.use_batteries:
        parser.error("--use_prosumer_batteries needs --use_batteries.")

    climate_model = args.climate_model
    climate_scenario = args.climate_scenario
    demand_scenario = args.demand_scenario
    demand_year = args.demand_year
    supply_scenario = args.supply_scenario
    supply_year = args.supply_year
    rolling_window_residual_load = args.rolling_window_residual_load

    demand_model = args.demand_model
    legacy_thermal = args.legacy_thermal
    demand_model_tag = DEMAND_MODEL_TAGS[demand_model]
    upsample_to_1h = demand_model == "demand.ninja"

    # Validate before anything is loaded or any output directory is created.
    if not args.rescale_cooling and not (
            demand_model == "cordex2energy_demand" and not legacy_thermal):
        raise SystemExit(
            "--no-cooling-target requires the split-target form: it is only meaningful when "
            "heating and cooling carry separate targets, i.e. --demand_model "
            "cordex2energy_demand without --legacy_thermal."
        )

    is_cerra = climate_model.lower() == "cerra"
    supply_scenario_column = SUPPLY_SCENARIO_COLUMNS[supply_scenario]
    model_prefix = "cerra" if is_cerra else "_".join(climate_model.split("_")[:-1])
    turbine_name = args.turbine.replace("/", "_")
    rcp_suffix = "none" if is_cerra else climate_scenario[-2:]

    print(f"Climate model    : {climate_model}")
    print(f"Climate scenario : {climate_scenario}")
    print(f"Demand scenario  : {demand_scenario} {demand_year}")
    print(f"Demand model     : {demand_model}"
          f"{' (legacy thermal)' if legacy_thermal else ''}")
    print(f"Supply scenario  : {supply_scenario} ({supply_scenario_column}) {supply_year}")
    print(f"Wind Turbine: {args.turbine}")
    print(f"Flexibility scheme: {args.flexibility_scheme}")
    print(f"Battery dispatch method: {args.battery_dispatch_method}")
    print(f"Utility dispatch level: {args.utility_dispatch_level}")

    # ── Dask ──────────────────────────────────────────────────────────────────
    dask.config.set(num_workers=args.num_workers, threads_per_worker=16)
    dask.config.set(
        {
            "distributed.worker.memory.target": 0.7,
            "distributed.worker.memory.spill": 0.8,
            "distributed.worker.memory.pause": 0.9,
            "array.chunk-size": "512MiB",
            "logging.distributed": "error",
        }
    )

    # ── Paths ─────────────────────────────────────────────────────────────────
    paths = json.load(open(args.paths_file))
    path_data_processed = Path(paths["path_data_processed"])
    path_data_extremes = path_data_processed / "extremes"
    path_tyndp = path_data_extremes / "tyndp"
    path_energy_europe = path_data_processed / "energy" / "europe"
    path_c2e = path_data_processed / "c2e" / "outputs"
    path_baseload_cache = path_tyndp / "baseload"

    run_tag = build_run_tag(args, demand_model_tag, turbine_name)
    model_subdir = "cerra" if is_cerra else f"{climate_model}_{climate_scenario}"
    path_out = path_data_extremes / "residual_load" / model_subdir / run_tag

    for directory in (path_baseload_cache, path_out):
        directory.mkdir(parents=True, exist_ok=True)

    # ── TYNDP supply capacities ───────────────────────────────────────────────
    print("\n── Loading TYNDP supply capacities ──────────────────────────────────────────")
    path_supply_excel = (
        path_tyndp / "20231103 - Final Supply Inputs for TYNDP 2024 Scenarios.xlsx"
    )
    solar_capacities, wind_onshore_capacities, wind_offshore_capacities = (
        load_tyndp_supply_capacities(path_supply_excel)
    )
    print(f"  Solar:           {len(solar_capacities)} rows")
    print(f"  Onshore wind:    {len(wind_onshore_capacities)} rows")
    print(f"  Offshore wind:   {len(wind_offshore_capacities)} rows")

    # ── Country-code mapping ──────────────────────────────────────────────────
    print("\n── Building country code mapping ────────────────────────────────────────────")
    path_demand_excel = (
        path_tyndp / "Demand_Scenarios_TYNDP_2024_After_Public_Consultation.xlsb"
    )
    tyndp_country_codes = load_tyndp_country_codes(path_demand_excel)
    name_to_iso2 = build_c2e_name_to_iso2(tyndp_country_codes)
    print(f"  {len(tyndp_country_codes)} countries in TYNDP mapping.")

    # ── TYNDP demand scenario table ───────────────────────────────────────────
    print("\n── Loading TYNDP demand scenario table ──────────────────────────────────────")
    demand_table = pd.read_excel(
        path_demand_excel,
        sheet_name="3_DEMAND_OUTPUT",
        header=[0, 1],
    )
    # Flatten the two header rows: year columns become "<year>_<scenario>".
    demand_table.columns = [
        f"{lower}_{upper}" if str(lower) in ["2019", "2040", "2050"] else lower
        for upper, lower in demand_table.columns
    ]

    # ── Climate-driven energy datasets ────────────────────────────────────────
    if is_cerra:
        print("\n── Loading CERRA energy data ────────────────────────────────────────────────")
        energy_data = load_cerra_energy_data(
            path_c2e=path_c2e,
            path_energy_europe=path_energy_europe,
            turbine_name=turbine_name,
            name_to_iso2=name_to_iso2,
            demand_model=demand_model,
        )
    else:
        print("\n── Loading CORDEX energy data ───────────────────────────────────────────────")
        energy_data = load_cordex_energy_data(
            path_c2e=path_c2e,
            path_energy_europe=path_energy_europe,
            climate_model=climate_model,
            model_prefix=model_prefix,
            climate_scenario=climate_scenario,
            rcp_suffix=rcp_suffix,
            turbine_name=turbine_name,
            name_to_iso2=name_to_iso2,
            demand_model=demand_model,
        )
    heating_demand = energy_data["heating_demand"]
    cooling_demand = energy_data["cooling_demand"]

    # ── Baseload ──────────────────────────────────────────────────────────────
    print("\n── Computing / loading TYNDP baseload ───────────────────────────────────────")
    # The baseload uses the same demand model as the residual load and, unless
    # legacy, is modulated to (TYNDP total − heating − cooling).
    baseload_typical_year = compute_or_load_baseload(
        path_c2e_cerra=path_data_processed / "c2e" / "outputs" / "CERRA",
        path_baseload_cache=path_baseload_cache,
        path_tyndp=path_tyndp,
        demand_scenario=demand_scenario,
        demand_year=demand_year,
        demand_table=demand_table,
        name_to_iso2=name_to_iso2,
        floor_fraction=args.floor_fraction,
        demand_model=demand_model,
        path_energy_demand=path_energy_europe / "demand",
        modulate_baseload=not legacy_thermal,
        reference_period=REFERENCE_PERIOD,
        legacy_thermal=legacy_thermal,
    )

    # ── Thermal targets ───────────────────────────────────────────────────────
    print("\n── Building thermal targets for residual load ───────────────────────────────")
    market_demand, _ = load_tyndp_market_profiles(
        path_tyndp / "Demand Profiles", demand_scenario, demand_year
    )
    thermal_targets = build_thermal_targets(
        demand_table,
        scenario=demand_scenario,
        year=demand_year,
        market_demand=market_demand,
    )
    print(f"  heating targets: {len(thermal_targets['heating'])}, "
          f"cooling targets: {len(thermal_targets['cooling'])}, "
          f"total (TYNDP-scenario) countries: {len(thermal_targets['total'])}.")

    # Split heating/cooling targets only for cordex2energy_demand; demand.ninja
    # uses one lumped target (its near-zero reference-period cooling makes a
    # split k_cool blow up in cool countries). Legacy also drops the
    # reference-period anchor.
    split_targets = (demand_model == "cordex2energy_demand") and not legacy_thermal
    if split_targets:
        residual_targets = {
            "heating_demand": thermal_targets["heating"],
            "cooling_demand": thermal_targets["cooling"],
        }
        if not args.rescale_cooling:
            # A missing key (not an empty one) switches cooling rescaling off,
            # so k_cool = 1 while heating keeps its target. The baseload reads
            # thermal_targets directly and is unaffected.
            del residual_targets["cooling_demand"]
            print("  cooling target DISABLED (--no-cooling-target): k_cool = 1, heating unchanged.")
    else:
        residual_targets = {
            country: thermal_targets["heating"].get(country, 0.0)
            + thermal_targets["cooling"].get(country, 0.0)
            for country in set(thermal_targets["heating"]) | set(thermal_targets["cooling"])
        }
    residual_reference_period = None if legacy_thermal else REFERENCE_PERIOD
    # cordex2energy_demand is already electric, so no COP loss factor.
    thermal_loss_factor = 1.0 if demand_model == "cordex2energy_demand" else 1.176

    if split_targets:
        print_rescale_factors(
            heating_demand,
            cooling_demand,
            thermal_targets,
            residual_reference_period,
            args.rescale_cooling,
        )

    # ── Estimated generation ──────────────────────────────────────────────────
    print("\n── Computing estimated generation ───────────────────────────────────────────")
    print(
        f"  Supply scenario: {supply_scenario_column} {supply_year}  "
        f"  (TYNDP column: {supply_year}_{supply_scenario_column})"
    )
    pv_generation, wind_onshore_generation, wind_offshore_generation = (
        compute_estimated_generation(
            energy_data["pv"],
            energy_data["wind_onshore"],
            energy_data["wind_offshore"],
            solar_capacities,
            wind_onshore_capacities,
            wind_offshore_capacities,
            scenario=supply_scenario_column,
            year=supply_year,
        )
    )
    pv_generation = pv_generation * GENERATION_SCALE
    wind_onshore_generation = wind_onshore_generation * GENERATION_SCALE
    wind_offshore_generation = wind_offshore_generation * GENERATION_SCALE

    # ── Residual load ─────────────────────────────────────────────────────────
    print("\n── Computing residual load ───────────────────────────────────────────────────")
    residual_load, components = compute_residual_load(
        pv_generation,
        wind_onshore_generation,
        wind_offshore_generation,
        energy_data["hydro_ror"],
        baseload_typical_year,   # hour_of_year dimension, expanded to the time axis
        heating_demand,
        cooling_demand,
        target_twh_per_country=residual_targets,
        loss_factor=thermal_loss_factor,
        upsample_to_1h=upsample_to_1h,
        reference_period=residual_reference_period,
        return_generation_demand_components=True,
        rolling_window_residual_load=rolling_window_residual_load,
    )

    # ── Reservoir hydro dispatch ──────────────────────────────────────────────
    if args.use_reservoir_hydro:
        print("\n── Applying reservoir hydro dispatch ────────────────────────────────────────")
        hydro_capacities = pd.read_csv(path_data_extremes / 'hydro_capacities.csv', index_col=0)
        reservoir_result = apply_reservoir_dispatch(
            residual_load,
            energy_data["hydro_inflow"],
            hydro_capacities,
            min_storage_level_fraction=args.reservoir_min_storage_fraction,
            efficiency=args.reservoir_efficiency,
        )
        # The reservoir state variables go to the components.
        residual_load = reservoir_result.drop_vars(
            ["reservoir_dispatch_gwh", "reservoir_storage_gwh"]
        )
        components = components.assign(
            reservoir_dispatch_gwh=reservoir_result["reservoir_dispatch_gwh"],
            reservoir_storage_gwh=reservoir_result["reservoir_storage_gwh"],
        )
        print("  Done.")

    # ── Battery dispatch (country level) ──────────────────────────────────────
    if args.use_batteries:
        print("\n── Applying battery dispatch ─────────────────────────────────────────────────")
        prosumer_capacity, utility_capacity = load_tyndp_battery_capacities(
            path_supply_excel,
            scenario=supply_scenario_column,
            year=supply_year,
            return_separate=True,
        )

        prosumer_dispatch_gwh = None
        utility_dispatch_gwh = None

        if args.flexibility_scheme == "system":
            # Prosumer + utility as one storage asset, dispatched with the
            # reservoir's objective and horizon, after the reservoir.
            total_capacity = (
                utility_capacity.add(prosumer_capacity, fill_value=0.0)
                if args.use_prosumer_batteries
                else utility_capacity
            )
            lumped = "prosumer + utility" if args.use_prosumer_batteries else "utility only"
            print(f"  Lumped batteries ({lumped}), annual-cyclical QP, country level …")
            print(
                "  [INFO] --battery_dispatch_method and --utility_dispatch_level are "
                "ignored under --flexibility_scheme system."
            )
            battery_result = apply_battery_dispatch(
                residual_load,
                total_capacity,
                battery_duration_h=args.battery_duration_h,
                roundtrip_efficiency=args.battery_roundtrip_eff,
                zone_dim="country",
                method="annual",
                min_soc_fraction=args.battery_min_soc_fraction,
            )
            battery_dispatch_gwh = battery_result["battery_dispatch_gwh"]
            battery_soc_gwh = battery_result["soc_gwh"]
            residual_load = residual_load_after_batteries(battery_result)
            # Only the combined dispatch exists in this scheme.
            components = components.assign(
                battery_dispatch_gwh=battery_dispatch_gwh,
                battery_soc_gwh=battery_soc_gwh,
            )
            print("  Done.")

        elif args.flexibility_scheme == "market":
            if args.use_prosumer_batteries:
                # Behind-the-meter assets: always the greedy heuristic.
                print("  Prosumer batteries (PV-aware, daily cycling) …")
                prosumer_result = apply_prosumer_battery_dispatch(
                    residual_load,
                    components["gen_pv"],
                    prosumer_capacity,
                    battery_duration_h=args.battery_duration_h,
                    roundtrip_efficiency=args.battery_roundtrip_eff,
                    initial_soc_fraction=args.battery_initial_soc,
                    zone_dim="country",
                )
                prosumer_dispatch_gwh = prosumer_result["battery_dispatch_gwh"]
                residual_load = residual_load_after_batteries(prosumer_result)
                components = components.assign(prosumer_dispatch_gwh=prosumer_dispatch_gwh)

            if args.battery_dispatch_method == "optimal":
                utility_method = (
                    f"optimal, rolling day-ahead: commit={args.battery_commit_h:.0f} h, "
                    f"lookahead={args.battery_dispatch_lookahead_h:.0f} h"
                )
            else:
                utility_method = f"lookahead={args.battery_dispatch_lookahead_h:.0f} h"

            # At the default 'zone' level, utility dispatch runs after zonal
            # aggregation instead (below).
            if args.utility_dispatch_level == "country":
                print(f"  Utility batteries (country, {utility_method}) …")
                utility_result = apply_battery_dispatch(
                    residual_load,
                    utility_capacity,
                    battery_duration_h=args.battery_duration_h,
                    roundtrip_efficiency=args.battery_roundtrip_eff,
                    initial_soc_fraction=args.battery_initial_soc,
                    dispatch_lookahead_h=args.battery_dispatch_lookahead_h,
                    zone_dim="country",
                    method=args.battery_dispatch_method,
                    min_soc_fraction=args.battery_min_soc_fraction,
                    commit_h=args.battery_commit_h,
                )
                utility_dispatch_gwh = utility_result["battery_dispatch_gwh"]
                residual_load = residual_load_after_batteries(utility_result)
                components = components.assign(utility_dispatch_gwh=utility_dispatch_gwh)

            # Country-level combined dispatch: prosumer, plus utility when it
            # was dispatched per country. The zonal total is assembled after
            # aggregation.
            country_dispatch_parts = [
                part for part in (prosumer_dispatch_gwh, utility_dispatch_gwh)
                if part is not None
            ]
            if country_dispatch_parts:
                components = components.assign(
                    battery_dispatch_gwh=sum(country_dispatch_parts)
                )
            print("  Done.")

    # ── Aggregate countries to the real-time balancing zones ──────────────────
    residual_load_zonal = aggregate_by_balancing_zones(residual_load)
    components_zonal = aggregate_by_balancing_zones(components)

    # ── Utility battery dispatch (zone level) ─────────────────────────────────
    # Utility batteries pooled per balancing zone, dispatched on the zonal
    # residual load after reservoir and prosumer dispatch.
    if (
        args.use_batteries
        and args.flexibility_scheme == "market"
        and args.utility_dispatch_level == "zone"
    ):
        print("\n── Applying zone-level utility battery dispatch ─────────────────────────────")
        zonal_utility_capacity = aggregate_battery_capacities_by_zone(utility_capacity)
        print(f"  Utility batteries (zone, {utility_method}) …")
        zonal_utility_result = apply_battery_dispatch(
            residual_load_zonal,
            zonal_utility_capacity,
            battery_duration_h=args.battery_duration_h,
            roundtrip_efficiency=args.battery_roundtrip_eff,
            initial_soc_fraction=args.battery_initial_soc,
            dispatch_lookahead_h=args.battery_dispatch_lookahead_h,
            zone_dim="zone",
            method=args.battery_dispatch_method,
            min_soc_fraction=args.battery_min_soc_fraction,
            commit_h=args.battery_commit_h,
        )
        zonal_utility_dispatch_gwh = zonal_utility_result["battery_dispatch_gwh"]
        residual_load_zonal = residual_load_after_batteries(zonal_utility_result)
        components_zonal = components_zonal.assign(
            utility_dispatch_gwh=zonal_utility_dispatch_gwh
        )
        if "prosumer_dispatch_gwh" in components_zonal:
            components_zonal = components_zonal.assign(
                battery_dispatch_gwh=components_zonal["prosumer_dispatch_gwh"]
                + zonal_utility_dispatch_gwh
            )
        else:
            components_zonal = components_zonal.assign(
                battery_dispatch_gwh=zonal_utility_dispatch_gwh
            )
        print("  Done.")

    # ── Save ──────────────────────────────────────────────────────────────────
    print("\n── Saving outputs ───────────────────────────────────────────────────────────")
    run_attributes = {
        "climate_model": climate_model,
        "climate_scenario": "reanalysis" if is_cerra else climate_scenario,
        "demand_scenario": demand_scenario,
        "demand_year": demand_year,
        "demand_model": demand_model,
        "legacy_thermal": str(legacy_thermal),
        "supply_scenario": supply_scenario_column,
        "supply_year": supply_year,
        "turbine": args.turbine,
        "rolling_window_residual_load": rolling_window_residual_load,
        "use_reservoir_hydro": str(args.use_reservoir_hydro),
        "reservoir_min_storage_fraction": str(args.reservoir_min_storage_fraction),
        "reservoir_efficiency": args.reservoir_efficiency,
        "use_batteries": str(args.use_batteries),
        "use_prosumer_batteries": str(args.use_prosumer_batteries),
        "flexibility_scheme": args.flexibility_scheme,
        "battery_dispatch_method": args.battery_dispatch_method,
        "battery_min_soc_fraction": str(args.battery_min_soc_fraction),
        "battery_commit_h": args.battery_commit_h,
        "utility_dispatch_level": args.utility_dispatch_level,
        "battery_duration_h": args.battery_duration_h,
        "battery_roundtrip_eff": args.battery_roundtrip_eff,
        "battery_initial_soc": args.battery_initial_soc,
        "battery_dispatch_lookahead_h": args.battery_dispatch_lookahead_h,
    }
    if args.use_batteries and args.flexibility_scheme == "system":
        # The system scheme runs one annually cyclic QP per country; the rolling and heuristic
        # settings do not apply.
        run_attributes.update({
            "battery_dispatch_method": "annual",
            "utility_dispatch_level": "country",
            "battery_commit_h": "not used",
            "battery_initial_soc": "not used (cyclic)",
            "battery_dispatch_lookahead_h": "not used",
        })

    outputs = [
        (residual_load, "residual_load.nc"),
        (residual_load_zonal, "residual_load_zonal.nc"),
        (components, "components.nc"),
        (components_zonal, "components_zonal.nc"),
    ]
    for dataset, file_name in outputs:
        path = path_out / file_name
        dataset.attrs.update(run_attributes)
        if path.exists():
            path.unlink()
        dataset.to_netcdf(path)
        print(f"  Saved: {path}")

    print(f"\nDone. Time elapsed: {(time.time() - start_time) / 60:.2f} minutes")


if __name__ == "__main__":
    main()
