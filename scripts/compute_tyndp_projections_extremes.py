"""Compute TYNDP-scenario residual load and its generation/demand components.

Given a CORDEX climate-model chain, a climate scenario, a TYNDP demand
scenario (REF/GA/DE) with an associated target year, and a TYNDP supply
projection scenario (LOW/BE/HIGH) with an associated target year, this script:

1. Loads TYNDP supply capacity projections (solar, onshore/offshore wind).
2. Loads TYNDP demand scenario data and builds country code mappings.
3. Loads CORDEX-derived capacity-factor and demand datasets.
4. Loads or computes (and caches) the TYNDP baseload for the requested
   demand scenario / year combination.
5. Computes estimated generation for all technologies.
6. Computes the residual load and its generation/demand components.
7. Saves the residual load and components as NetCDF files.
"""

import argparse
import time
import json
import os
from pathlib import Path

import dask
import pandas as pd
import xarray as xr

from cordex2energy.conversion_energy.tyndp import (
    aggregate_battery_capacities_by_zone,
    aggregate_by_balancing_zones,
    apply_battery_dispatch,
    apply_prosumer_battery_dispatch,
    apply_reservoir_dispatch,
    build_c2e_name_to_iso2,
    build_thermal_targets,
    build_tyndp_nuts_mapping,
    build_tyndp_zone_geometries,
    compute_estimated_generation,
    compute_residual_load,
    compute_or_load_baseload,
    drop_overlapping_zones,
    load_c2e_demand_nc,
    load_nuts_geodata,
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

def load_cordex_energy_data(
    path_c2e: Path,
    path_energy_europe: Path,
    climate_model: str,
    model_prefix: str,
    climate_scenario: str,
    rcp_suffix: str,
    turbine_fn: str,
    name_to_iso2: dict,
    demand_model: str = "demand.ninja",
) -> dict:
    """Load all CORDEX-derived energy datasets needed for residual load computation.

    Parameters
    ----------
    path_c2e:
        Root C2E outputs directory (contains per-model subdirectories).
    path_energy_europe:
        Root of the European energy outputs (pv/, wind_power/, …).
    climate_model:
        Full climate model chain identifier.
    model_prefix:
        Climate model identifier without ensemble member (e.g. without _r1i1p1).
    climate_scenario:
        RCP scenario string (e.g. "rcp85").
    rcp_suffix:
        Last two characters of the RCP scenario (e.g. "85").
    turbine_fn:
        Turbine model name with slashes replaced by underscores.
    name_to_iso2:
        Full-name → ISO2 mapping for CORDEX country labels.

    Returns
    -------
    dict with keys:
        ds_pv, ds_wind, ds_wind_offshore, ds_hydro_ror, ds_hydro_inflow,
        ds_heating_demand, ds_cooling_demand
    """
    print("  Loading PV capacity factors …")
    ds_pv = xr.open_dataset(
        path_energy_europe
        / "pv"
        / f"{climate_model}_{climate_scenario}"
        / f"pv_tyndp_agg_{climate_model}_{climate_scenario}.nc"
    )
    # Correct 1h30min time offset present in some CORDEX PV outputs.
    ds_pv["time"] = ds_pv.time - pd.Timedelta("1h30min")

    print("  Loading onshore wind capacity factors …")
    ds_wind = xr.open_dataset(
        path_energy_europe
        / "wind_power"
        / f"{climate_model}_{climate_scenario}"
        / f"wind_power_tyndp_agg_{climate_model}_{climate_scenario}_{turbine_fn}.nc"
    )

    print("  Loading offshore wind capacity factors …")
    _ds_wind_offshore_raw = xr.open_dataset(
        path_energy_europe
        / "wind_power"
        / f"{climate_model}_{climate_scenario}"
        / f"wind_power_offshore_{climate_model}_{climate_scenario}_{turbine_fn}.nc"
    )
    # Offshore wind uses full country names in the 'country' dimension.
    ds_wind_offshore = _ds_wind_offshore_raw.assign_coords(
        country=[
            name_to_iso2.get(str(c).strip(), str(c).strip())
            for c in _ds_wind_offshore_raw.country.values
        ]
    )

    _c2e_model_dir = path_c2e / model_prefix / "output_variables"

    print("  Loading hydro run-of-river …")
    df_hydro_ror = pd.read_csv(
        _c2e_model_dir
        / f"hydro_ror_CORDEX_{model_prefix}_{rcp_suffix}.csv",
        index_col=0,
    )
    ds_hydro_ror = preprocess_hydro_ror(
        df_hydro_ror, name_to_iso2, freq="3h", var_name="ror"
    )

    print("  Loading hydro inflow …")
    df_hydro_inflow = pd.read_csv(
        _c2e_model_dir
        / f"hydro_inflow_CORDEX_{model_prefix}_{rcp_suffix}.csv",
        index_col=0,
    )
    ds_hydro_inflow = xr.DataArray(
        df_hydro_inflow.values,
        dims=["country", "time"],
        coords={
            "country": [
                name_to_iso2.get(str(c).strip(), str(c).strip())
                for c in df_hydro_inflow.index
            ],
            "time": pd.to_datetime(df_hydro_inflow.columns),
        },
        name="inflow",
    ).to_dataset()

    if demand_model == "cordex2energy_demand":
        print("  Loading heating/cooling demand (JRC cordex2energy_demand NetCDF) …")
        ds_heating_demand, ds_cooling_demand = load_c2e_demand_nc(
            path_energy_europe / "demand" / f"{climate_model}_{climate_scenario}.nc",
            name_to_iso2,
        )
    else:
        print("  Loading heating demand …")
        df_heating = pd.read_csv(
            _c2e_model_dir
            / f"heating-demand_CORDEX_{model_prefix}_{rcp_suffix}.csv",
            index_col=0,
        )
        ds_heating_demand = xr.DataArray(
            df_heating.values,
            dims=["country", "time"],
            coords={
                "country": [name_to_iso2.get(c, c) for c in df_heating.index],
                "time": pd.to_datetime(df_heating.columns),
            },
            name="heating_demand",
        ).to_dataset()

        print("  Loading cooling demand …")
        df_cooling = pd.read_csv(
            _c2e_model_dir
            / f"cooling-demand_CORDEX_{model_prefix}_{rcp_suffix}.csv",
            index_col=0,
        )
        ds_cooling_demand = xr.DataArray(
            df_cooling.values,
            dims=["country", "time"],
            coords={
                "country": [name_to_iso2.get(c, c) for c in df_cooling.index],
                "time": pd.to_datetime(df_cooling.columns),
            },
            name="cooling_demand",
        ).to_dataset()

    return {
        "ds_pv": ds_pv,
        "ds_wind": ds_wind,
        "ds_wind_offshore": ds_wind_offshore,
        "ds_hydro_ror": ds_hydro_ror,
        "ds_hydro_inflow": ds_hydro_inflow,
        "ds_heating_demand": ds_heating_demand,
        "ds_cooling_demand": ds_cooling_demand,
    }

def load_cerra_energy_data(
    path_c2e: Path,
    path_energy_europe: Path,
    turbine_fn: str,
    name_to_iso2: dict,
    demand_model: str = "demand.ninja",
) -> dict:
    """Load CERRA-derived energy datasets for residual load computation."""
    print("  Loading CERRA PV capacity factors …")
    ds_pv = xr.open_dataset(
        path_energy_europe / "pv" / "cerra" / "pv_tyndp_agg_cerra.nc"
    )

    print("  Loading CERRA onshore wind capacity factors …")
    ds_wind = xr.open_dataset(
        path_energy_europe / "wind_power" / "cerra"
        / f"wind_power_tyndp_agg_cerra_{turbine_fn}.nc"
    )

    print("  Loading CERRA offshore wind capacity factors …")
    _path_wind_off = (
        path_energy_europe / "wind_power" / "cerra"
        / f"wind_power_offshore_cerra_{turbine_fn}.nc"
    )
    if _path_wind_off.exists():
        _ds_wind_offshore_raw = xr.open_dataset(_path_wind_off)
        ds_wind_offshore = _ds_wind_offshore_raw.assign_coords(
            country=[
                name_to_iso2.get(str(c).strip(), str(c).strip())
                for c in _ds_wind_offshore_raw.country.values
            ]
        )
    else:
        print("    [INFO] Offshore wind not available for CERRA — using zeros.")
        ds_wind_offshore = xr.zeros_like(ds_wind)

    _c2e_cerra_dir = path_c2e / "CERRA" / "output_variables"

    print("  Loading CERRA hydro run-of-river …")
    df_hydro_ror = pd.read_csv(
        _c2e_cerra_dir / "hydro_ror_CERRA.csv",
        index_col=0,
    )
    ds_hydro_ror = preprocess_hydro_ror(
        df_hydro_ror, name_to_iso2, freq="3h", var_name="ror"
    ).sel(time=slice("1991", "2020"))

    print("  Loading CERRA hydro inflow …")
    df_hydro_inflow = pd.read_csv(
        _c2e_cerra_dir / "hydro_inflow_CERRA.csv",
        index_col=0,
    )
    ds_hydro_inflow = xr.DataArray(
        df_hydro_inflow.values,
        dims=["country", "time"],
        coords={
            "country": [
                name_to_iso2.get(str(c).strip(), str(c).strip())
                for c in df_hydro_inflow.index
            ],
            "time": pd.to_datetime(df_hydro_inflow.columns),
        },
        name="inflow",
    ).to_dataset()

    if demand_model == "cordex2energy_demand":
        print("  Loading CERRA heating/cooling demand (JRC cordex2energy_demand NetCDF) …")
        ds_heating_demand, ds_cooling_demand = load_c2e_demand_nc(
            path_energy_europe / "demand" / "cerra.nc",
            name_to_iso2,
            time_slice=REFERENCE_PERIOD,
        )
    else:
        print("  Loading CERRA heating demand …")
        df_heating = pd.read_csv(
            _c2e_cerra_dir / "heating-demand_CERRA.csv",
            index_col=0,
        )
        ds_heating_demand = xr.DataArray(
            df_heating.values,
            dims=["country", "time"],
            coords={
                "country": [name_to_iso2.get(str(c).strip(), str(c).strip()) for c in df_heating.index],
                "time": pd.to_datetime(df_heating.columns),
            },
            name="heating_demand",
        ).to_dataset()

        print("  Loading CERRA cooling demand …")
        df_cooling = pd.read_csv(
            _c2e_cerra_dir / "cooling-demand_CERRA.csv",
            index_col=0,
        )
        ds_cooling_demand = xr.DataArray(
            df_cooling.values,
            dims=["country", "time"],
            coords={
                "country": [name_to_iso2.get(str(c).strip(), str(c).strip()) for c in df_cooling.index],
                "time": pd.to_datetime(df_cooling.columns),
            },
            name="cooling_demand",
        ).to_dataset()

    return {
        "ds_pv": ds_pv.sel(time=slice("1991", "2020")),
        "ds_wind": ds_wind.sel(time=slice("1991", "2020")),
        "ds_wind_offshore": ds_wind_offshore.sel(time=slice("1991", "2020")),
        "ds_hydro_ror": ds_hydro_ror.sel(time=slice("1991", "2020")),
        "ds_hydro_inflow": ds_hydro_inflow.sel(time=slice("1991", "2020")),
        "ds_heating_demand": ds_heating_demand.sel(time=slice("1991", "2020")),
        "ds_cooling_demand": ds_cooling_demand.sel(time=slice("1991", "2020")),
    }


if __name__ == "__main__":
    # ── CLI ───────────────────────────────────────────────────────────────────────
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
            "at every time step.  Default None (no floor)."
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
            "only with --battery_dispatch_method optimal. Default None (no floor)."
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

    t0 = time.time()
    args = parser.parse_args()

    # ── Derived parameters ────────────────────────────────────────────────────────

    climate_model = args.climate_model
    climate_scenario = args.climate_scenario
    demand_scenario = args.demand_scenario
    demand_year = args.demand_year
    supply_scenario = args.supply_scenario
    supply_year = args.supply_year
    rolling_window_residual_load = args.rolling_window_residual_load

    # Demand-model selection. The hourly cordex2energy_demand NetCDFs must NOT be
    # cubic-upsampled (only resampled to 3-hourly sums); the 3-hourly demand.ninja
    # CSVs are upsampled to 1 h first. The model is encoded in the run tag and the
    # baseload cache key so runs never collide.
    demand_model = args.demand_model
    legacy_thermal = args.legacy_thermal
    dm_tag = {"demand.ninja": "dninja", "cordex2energy_demand": "c2edem"}[demand_model]
    upsample_to_1h = demand_model == "demand.ninja"

    is_cerra = climate_model.lower() == "cerra"

    # Map CLI supply scenario to the column suffix used in the TYNDP Excel files.
    _SUPPLY_SCENARIO_MAP = {"LOW": "LOW", "BE": "Best Estimate", "HIGH": "HIGH"}
    supply_scenario_col = _SUPPLY_SCENARIO_MAP[supply_scenario]

    # Derive the C2E model prefix (drops the ensemble member, e.g. _r1i1p1).
    model_prefix = "cerra" if is_cerra else "_".join(climate_model.split("_")[:-1])

    # Turbine name as it appears in filenames (/ → _).
    turbine_fn = args.turbine.replace("/", "_")

    # Short RCP suffix used in some C2E filenames (e.g. "85" from "rcp85").
    rcp_suffix = "none" if is_cerra else climate_scenario[-2:]

    print(f"Climate model    : {climate_model}")
    print(f"Climate scenario : {climate_scenario}")
    print(f"Demand scenario  : {demand_scenario} {demand_year}")
    print(f"Demand model     : {demand_model}"
          f"{' (legacy thermal)' if legacy_thermal else ''}")
    print(f"Supply scenario  : {supply_scenario} ({supply_scenario_col}) {supply_year}")
    print(f"Wind Turbine: {args.turbine}")
    print(f"Flexibility scheme: {args.flexibility_scheme}")
    print(f"Battery dispatch method: {args.battery_dispatch_method}")
    print(f"Utility dispatch level: {args.utility_dispatch_level}")

    # ── Dask ──────────────────────────────────────────────────────────────────────

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

    # ── Paths ─────────────────────────────────────────────────────────────────────

    paths = json.load(open(args.paths_file))
    path_data_processed = Path(paths["path_data_processed"])

    path_data_extremes = path_data_processed / "extremes"
    path_tyndp = path_data_extremes / "tyndp"
    path_data_energy_europe = path_data_processed / "energy" / "europe"
    path_c2e = path_data_processed / "c2e" / "outputs"

    path_baseload_cache = path_tyndp / "baseload"

    # Build run tag before path_out so it can be used as the output subfolder.
    # The demand-model token is always present (the demand series — and therefore
    # the whole residual-load system — differs by model); the legacy-thermal
    # marker is appended only when that flag is set.
    _model_tag = model_prefix
    _run_tag = (
        f"{demand_scenario}_{demand_year}_{dm_tag}"
        f"_{supply_scenario}_{supply_year}"
    )
    if legacy_thermal:
        _run_tag += "_legacythermal"
    if rolling_window_residual_load and rolling_window_residual_load > 1:
        _run_tag += f"_rlsmooth_{rolling_window_residual_load}"
    if args.use_reservoir_hydro:
        _run_tag += "_reservoirdispatch"
    if args.use_batteries:
        _run_tag += "_batteries"
        if args.use_prosumer_batteries:
            _run_tag += "_prosumer"
        if args.flexibility_scheme == "system":
            # Lumped batteries + annual QP at country level; the method/level flags
            # do not apply, so they are not encoded in the tag.
            _run_tag += "_sysflex"
        else:
            if args.battery_dispatch_method == "optimal":
                _run_tag += "_optbatt"
            # Default utility level is 'zone'; tag only the non-default so the common
            # case keeps stable folder names and country-level comparison runs don't
            # collide with zone-level outputs.
            if args.utility_dispatch_level == "country":
                _run_tag += "_utilcountry"

    # The turbine is ALWAYS tagged, including the default. It sets the wind capacity factor outright
    # (mean onshore CF is 0.20 for E-126/7580 vs 0.40 for SWT142/3150), so two runs that differ only in
    # turbine are different systems and must never share an output folder.
    _run_tag += f"_{turbine_fn}"

    # Output directory: one subfolder per run configuration.
    _model_subdir = "cerra" if is_cerra else f"{climate_model}_{climate_scenario}"
    path_out = path_data_extremes / "residual_load" / _model_subdir / _run_tag

    for p in (path_baseload_cache, path_out):
        p.mkdir(parents=True, exist_ok=True)

    # ── Load TYNDP supply capacities ──────────────────────────────────────────────

    print("\n── Loading TYNDP supply capacities ──────────────────────────────────────────")
    path_supply_excel = (
        path_tyndp / "20231103 - Final Supply Inputs for TYNDP 2024 Scenarios.xlsx"
    )
    df_tyndp_solar, df_tyndp_wind_onshore, df_tyndp_wind_offshore = (
        load_tyndp_supply_capacities(path_supply_excel)
    )
    print(f"  Solar:           {len(df_tyndp_solar)} rows")
    print(f"  Onshore wind:    {len(df_tyndp_wind_onshore)} rows")
    print(f"  Offshore wind:   {len(df_tyndp_wind_offshore)} rows")

    # ── Load TYNDP demand metadata and build country code mapping ─────────────────

    print("\n── Building country code mapping ────────────────────────────────────────────")
    path_demand_excel = (
        path_tyndp
        / "Demand_Scenarios_TYNDP_2024_After_Public_Consultation.xlsb"
    )
    tyndp_country_codes = load_tyndp_country_codes(path_demand_excel)
    C2E_NAME_TO_ISO2 = build_c2e_name_to_iso2(tyndp_country_codes)
    print(f"  {len(tyndp_country_codes)} countries in TYNDP mapping.")

    # ── Load TYNDP demand scenario table ──────────────────────────────────────────

    print("\n── Loading TYNDP demand scenario table ──────────────────────────────────────")
    df_tyndp_demand = pd.read_excel(
        path_demand_excel,
        sheet_name="3_DEMAND_OUTPUT",
        header=[0, 1],
    )
    df_tyndp_demand.columns = [
        f"{l2}_{l1}" if str(l2) in ["2019", "2040", "2050"] else l2
        for l1, l2 in df_tyndp_demand.columns
    ]

    # ── Build TYNDP zone → NUTS mapping and zone geometries ───────────────────────

    print("\n── Building TYNDP zone geometries ───────────────────────────────────────────")
    tyndp_bidding_zones = (
        df_tyndp_solar[["Country"]]
        .copy()
        .rename(columns={"Country": "Bidding Zone"})
        .iloc[:-3]
    )
    tyndp_bidding_zones["iso2"] = tyndp_bidding_zones["Bidding Zone"].apply(
        lambda x: str(x)[:2]
    )
    all_zones = sorted(tyndp_bidding_zones["Bidding Zone"].unique())

    gdf_nuts1, gdf_nuts2, gdf_uk_nuts1 = load_nuts_geodata(
        path_data_processed / "maps" / "nuts"
    )
    tyndp_nuts2_mapping, df_tyndp_nuts_mapping = build_tyndp_nuts_mapping(
        all_zones, gdf_nuts2, gdf_uk_nuts1
    )
    gdf_tyndp_zones = build_tyndp_zone_geometries(
        tyndp_nuts2_mapping, df_tyndp_nuts_mapping, gdf_nuts2, gdf_uk_nuts1
    )
    gdf_tyndp_zones = drop_overlapping_zones(gdf_tyndp_zones)
    print(f"  {len(gdf_tyndp_zones)} zones after dropping overlapping geometries.")

    # ── Load energy datasets ──────────────────────────────────────────────────────

    if is_cerra:
        print("\n── Loading CERRA energy data ────────────────────────────────────────────────")
        energy_data = load_cerra_energy_data(
            path_c2e=path_c2e,
            path_energy_europe=path_data_energy_europe,
            turbine_fn=turbine_fn,
            name_to_iso2=C2E_NAME_TO_ISO2,
            demand_model=demand_model,
        )
    else:
        print("\n── Loading CORDEX energy data ───────────────────────────────────────────────")
        energy_data = load_cordex_energy_data(
            path_c2e=path_c2e,
            path_energy_europe=path_data_energy_europe,
            climate_model=climate_model,
            model_prefix=model_prefix,
            climate_scenario=climate_scenario,
            rcp_suffix=rcp_suffix,
            turbine_fn=turbine_fn,
            name_to_iso2=C2E_NAME_TO_ISO2,
            demand_model=demand_model,
        )
    ds_pv = energy_data["ds_pv"]
    ds_wind = energy_data["ds_wind"]
    ds_wind_offshore = energy_data["ds_wind_offshore"]
    ds_hydro_ror = energy_data["ds_hydro_ror"]
    ds_hydro_inflow = energy_data["ds_hydro_inflow"]
    ds_heating_demand = energy_data["ds_heating_demand"]
    ds_cooling_demand = energy_data["ds_cooling_demand"]

    # ── Compute or load baseload ──────────────────────────────────────────────────

    print("\n── Computing / loading TYNDP baseload ───────────────────────────────────────")
    # Baseload uses the SAME demand model as the residual load (subtract-then-add
    # symmetry) and, unless legacy, is modulated to (TYNDP total − heat − cool).
    ds_tyndp_baseload_typical = compute_or_load_baseload(
        path_c2e_cerra=path_data_processed / "c2e" / "outputs" / "CERRA",
        path_baseload_cache=path_baseload_cache,
        path_tyndp=path_tyndp,
        demand_scenario=demand_scenario,
        demand_year=demand_year,
        df_tyndp_demand=df_tyndp_demand,
        C2E_NAME_TO_ISO2=C2E_NAME_TO_ISO2,
        floor_fraction=args.floor_fraction,
        demand_model=demand_model,
        path_energy_demand=path_data_energy_europe / "demand",
        modulate_baseload=not legacy_thermal,
        reference_period=REFERENCE_PERIOD,
        legacy_thermal=legacy_thermal,
    )

    # ── Build thermal targets for residual load computation ───────────────────────

    print("\n── Building thermal targets for residual load ───────────────────────────────")
    # We need the market profiles again to build thermal targets for `compute_residual_load`.
    _path_demand_profiles_dir = path_tyndp / "Demand Profiles"
    _ds_market_for_targets, _ = load_tyndp_market_profiles(
        _path_demand_profiles_dir, demand_scenario, demand_year
    )
    thermal_targets = build_thermal_targets(
        df_tyndp_demand,
        scenario=demand_scenario,
        year=demand_year,
        ds_market=_ds_market_for_targets,
    )
    print(f"  heating targets: {len(thermal_targets['heating'])}, "
          f"cooling targets: {len(thermal_targets['cooling'])}, "
          f"total (TYNDP-scenario) countries: {len(thermal_targets['total'])}.")

    # Residual-load target form. Split heating/cooling targets ONLY for
    # cordex2energy_demand; demand.ninja uses a single lumped target (its
    # near-zero reference-period cooling makes a split k_cool blow up in cool
    # countries). Legacy additionally drops the reference-period anchor.
    split_targets = (demand_model == "cordex2energy_demand") and not legacy_thermal
    if split_targets:
        residual_targets = {
            "heating_demand": thermal_targets["heating"],
            "cooling_demand": thermal_targets["cooling"],
        }
    else:
        residual_targets = {
            c: thermal_targets["heating"].get(c, 0.0) + thermal_targets["cooling"].get(c, 0.0)
            for c in set(thermal_targets["heating"]) | set(thermal_targets["cooling"])
        }
    residual_reference_period = None if legacy_thermal else REFERENCE_PERIOD
    # cordex2energy_demand is already electric → no COP loss factor (it cancels
    # under target rescaling anyway, but matters for unmatched Balkan countries).
    thermal_loss_factor = 1.0 if demand_model == "cordex2energy_demand" else 1.176

    # ── Compute estimated generation ──────────────────────────────────────────────

    print("\n── Computing estimated generation ───────────────────────────────────────────")
    print(
        f"  Supply scenario: {supply_scenario_col} {supply_year}  "
        f"  (TYNDP column: {supply_year}_{supply_scenario_col})"
    )
    ds_gen_pv, ds_gen_wind_on, ds_gen_wind_off = compute_estimated_generation(
        ds_pv,
        ds_wind,
        ds_wind_offshore,
        df_tyndp_solar,
        df_tyndp_wind_onshore,
        df_tyndp_wind_offshore,
        scenario=supply_scenario_col,
        year=supply_year,
    )

    # Convert from MW (capacity × CF, 3-hourly average) to GWh per 3-hour interval.
    _GEN_SCALE = 3 * 1e-3   # 3 h × 1e-3 (MWh → GWh)
    ds_gen_pv = ds_gen_pv * _GEN_SCALE
    ds_gen_wind_on = ds_gen_wind_on * _GEN_SCALE
    ds_gen_wind_off = ds_gen_wind_off * _GEN_SCALE

    # ── Compute residual load ─────────────────────────────────────────────────────

    print("\n── Computing residual load ───────────────────────────────────────────────────")
    ds_residual_load, ds_components = compute_residual_load(
        ds_gen_pv,
        ds_gen_wind_on,
        ds_gen_wind_off,
        ds_hydro_ror,
        ds_tyndp_baseload_typical,   # typical-year (hour_of_year dim) — auto-expanded
        ds_heating_demand,
        ds_cooling_demand,
        target_twh_per_country=residual_targets,
        loss_factor=thermal_loss_factor,
        upsample_to_1h=upsample_to_1h,
        reference_period=residual_reference_period,
        return_generation_demand_components=True,
        rolling_window_residual_load=rolling_window_residual_load,
    )

    # ── Optional reservoir hydro dispatch ─────────────────────────────────────
    if args.use_reservoir_hydro:
        print("\n── Applying reservoir hydro dispatch ────────────────────────────────────────")
        df_hydro_capacities = pd.read_csv(path_data_extremes / 'hydro_capacities.csv', index_col=0)
        _ds_reservoir = apply_reservoir_dispatch(
            ds_residual_load,
            ds_hydro_inflow,
            df_hydro_capacities,
            min_storage_level_fraction=args.reservoir_min_storage_fraction,
            efficiency=args.reservoir_efficiency,
        )
        # Updated residual load only — reservoir state variables go to components.
        ds_residual_load = _ds_reservoir.drop_vars(
            ["reservoir_dispatch_gwh", "reservoir_storage_gwh"]
        )
        ds_components = ds_components.assign(
            reservoir_dispatch_gwh=_ds_reservoir["reservoir_dispatch_gwh"],
            reservoir_storage_gwh=_ds_reservoir["reservoir_storage_gwh"],
        )
        print("  Done.")

    # ── Optional battery dispatch ──────────────────────────────────────────────
    if args.use_batteries:
        print("\n── Applying battery dispatch ─────────────────────────────────────────────────")
        s_batt_prosumer, s_batt_utility = load_tyndp_battery_capacities(
            path_supply_excel,
            scenario=supply_scenario_col,
            year=supply_year,
            return_separate=True,
        )

        prosumer_dispatch_gwh = None
        utility_dispatch_gwh = None

    # ── 'system' scheme: lumped batteries, annual-cyclical QP, country level ───
    if args.use_batteries and args.flexibility_scheme == "system":
        # Prosumer + utility are a single storage asset, dispatched with the same
        # objective and horizon as the reservoir (sequentially, after it).
        s_batt_total = (
            s_batt_utility.add(s_batt_prosumer, fill_value=0.0)
            if args.use_prosumer_batteries
            else s_batt_utility
        )
        _lump = "prosumer + utility" if args.use_prosumer_batteries else "utility only"
        print(f"  Lumped batteries ({_lump}), annual-cyclical QP, country level …")
        print(
            "  [INFO] --battery_dispatch_method and --utility_dispatch_level are "
            "ignored under --flexibility_scheme system."
        )
        _ds_batt = apply_battery_dispatch(
            ds_residual_load,
            s_batt_total,
            battery_duration_h=args.battery_duration_h,
            roundtrip_efficiency=args.battery_roundtrip_eff,
            zone_dim="country",
            method="annual",
            min_soc_fraction=args.battery_min_soc_fraction,
        )
        battery_dispatch_gwh = _ds_batt["battery_dispatch_gwh"]
        battery_soc_gwh = _ds_batt["soc_gwh"]
        ds_residual_load = _ds_batt.assign(
            residual_load=_ds_batt["residual_load_with_batteries"]
        ).drop_vars(["residual_load_with_batteries", "soc_gwh", "battery_dispatch_gwh"])
        # Only the combined dispatch exists in this scheme (no prosumer/utility split).
        ds_components = ds_components.assign(
            battery_dispatch_gwh=battery_dispatch_gwh,
            battery_soc_gwh=battery_soc_gwh,
        )
        print("  Done.")

    # ── 'market' scheme: prosumer heuristic + utility (rolling/greedy) ─────────
    if args.use_batteries and args.flexibility_scheme == "market":
        if args.use_prosumer_batteries:
            print("  Prosumer batteries (PV-aware, daily cycling) …")
            # Prosumer batteries are behind-the-meter self-consumption assets and are
            # always dispatched with the greedy heuristic (the optimal QP applies to
            # utility batteries only).
            _ds_pro = apply_prosumer_battery_dispatch(
                ds_residual_load,
                ds_components["gen_pv"],
                s_batt_prosumer,
                battery_duration_h=args.battery_duration_h,
                roundtrip_efficiency=args.battery_roundtrip_eff,
                initial_soc_fraction=args.battery_initial_soc,
                zone_dim="country",
            )
            prosumer_dispatch_gwh = _ds_pro["battery_dispatch_gwh"]
            ds_residual_load = _ds_pro.assign(
                residual_load=_ds_pro["residual_load_with_batteries"]
            ).drop_vars(["residual_load_with_batteries", "soc_gwh", "battery_dispatch_gwh"])
            ds_components = ds_components.assign(prosumer_dispatch_gwh=prosumer_dispatch_gwh)

        # Describe the utility dispatch method once (reused across levels).
        if args.battery_dispatch_method == "optimal":
            _util_desc = (
                f"optimal, rolling day-ahead: commit={args.battery_commit_h:.0f} h, "
                f"lookahead={args.battery_dispatch_lookahead_h:.0f} h"
            )
        else:
            _util_desc = f"lookahead={args.battery_dispatch_lookahead_h:.0f} h"

        # Utility batteries are dispatched at country level here only when
        # --utility_dispatch_level country; the default ('zone') defers utility
        # dispatch to after zonal aggregation (see below).
        if args.utility_dispatch_level == "country":
            print(f"  Utility batteries (country, {_util_desc}) …")
            _ds_util = apply_battery_dispatch(
                ds_residual_load,
                s_batt_utility,
                battery_duration_h=args.battery_duration_h,
                roundtrip_efficiency=args.battery_roundtrip_eff,
                initial_soc_fraction=args.battery_initial_soc,
                dispatch_lookahead_h=args.battery_dispatch_lookahead_h,
                zone_dim="country",
                method=args.battery_dispatch_method,
                min_soc_fraction=args.battery_min_soc_fraction,
                commit_h=args.battery_commit_h,
            )
            utility_dispatch_gwh = _ds_util["battery_dispatch_gwh"]
            ds_residual_load = _ds_util.assign(
                residual_load=_ds_util["residual_load_with_batteries"]
            ).drop_vars(["residual_load_with_batteries", "soc_gwh", "battery_dispatch_gwh"])
            ds_components = ds_components.assign(utility_dispatch_gwh=utility_dispatch_gwh)

        # Country-level combined battery dispatch (prosumer, plus utility if it was
        # dispatched at country level).  When utility is zone-level this holds prosumer
        # only; the zonal total is assembled after aggregation.
        _country_batt_parts = [
            p for p in (prosumer_dispatch_gwh, utility_dispatch_gwh) if p is not None
        ]
        if _country_batt_parts:
            ds_components = ds_components.assign(
                battery_dispatch_gwh=sum(_country_batt_parts)
            )
        print("  Done.")

    # Aggregate from individual countries to real-time balancing zones.
    ds_residual_load_zonal = aggregate_by_balancing_zones(ds_residual_load)
    ds_components_zonal = aggregate_by_balancing_zones(ds_components)

    # ── Zone-level utility battery dispatch ────────────────────────────────────
    # Utility-scale batteries are grid/merchant assets, pooled across the 6 balancing
    # zones and dispatched on the (post reservoir + prosumer) zonal residual load.
    if (
        args.use_batteries
        and args.flexibility_scheme == "market"
        and args.utility_dispatch_level == "zone"
    ):
        print("\n── Applying zone-level utility battery dispatch ─────────────────────────────")
        s_util_zone = aggregate_battery_capacities_by_zone(s_batt_utility)
        print(f"  Utility batteries (zone, {_util_desc}) …")
        _ds_util_zone = apply_battery_dispatch(
            ds_residual_load_zonal,
            s_util_zone,
            battery_duration_h=args.battery_duration_h,
            roundtrip_efficiency=args.battery_roundtrip_eff,
            initial_soc_fraction=args.battery_initial_soc,
            dispatch_lookahead_h=args.battery_dispatch_lookahead_h,
            zone_dim="zone",
            method=args.battery_dispatch_method,
            min_soc_fraction=args.battery_min_soc_fraction,
            commit_h=args.battery_commit_h,
        )
        utility_dispatch_gwh_zonal = _ds_util_zone["battery_dispatch_gwh"]
        ds_residual_load_zonal = _ds_util_zone.assign(
            residual_load=_ds_util_zone["residual_load_with_batteries"]
        ).drop_vars(["residual_load_with_batteries", "soc_gwh", "battery_dispatch_gwh"])
        ds_components_zonal = ds_components_zonal.assign(
            utility_dispatch_gwh=utility_dispatch_gwh_zonal
        )
        # Combined zonal battery dispatch = zonal prosumer (if any) + zonal utility.
        if "prosumer_dispatch_gwh" in ds_components_zonal:
            ds_components_zonal = ds_components_zonal.assign(
                battery_dispatch_gwh=ds_components_zonal["prosumer_dispatch_gwh"]
                + utility_dispatch_gwh_zonal
            )
        else:
            ds_components_zonal = ds_components_zonal.assign(
                battery_dispatch_gwh=utility_dispatch_gwh_zonal
            )
        print("  Done.")

    # ── Save outputs ──────────────────────────────────────────────────────────────

    print("\n── Saving outputs ───────────────────────────────────────────────────────────")

    # Attach run metadata as global attributes.
    _attrs = {
        "climate_model": climate_model,
        "climate_scenario": "reanalysis" if is_cerra else climate_scenario,
        "demand_scenario": demand_scenario,
        "demand_year": demand_year,
        "demand_model": demand_model,
        "legacy_thermal": str(legacy_thermal),
        "supply_scenario": supply_scenario_col,
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

    path_rl = path_out / "residual_load.nc"
    path_rl_zonal = path_out / "residual_load_zonal.nc"
    path_comp = path_out / "components.nc"
    path_comp_zonal = path_out / "components_zonal.nc"

    for ds, path in [
        (ds_residual_load, path_rl),
        (ds_residual_load_zonal, path_rl_zonal),
        (ds_components, path_comp),
        (ds_components_zonal, path_comp_zonal),
    ]:
        ds.attrs.update(_attrs)
        if path.exists():
            path.unlink()
        ds.to_netcdf(path)
        print(f"  Saved: {path}")

    t1 = time.time()
    print(f"\nDone. Time elapsed: {(t1 - t0) / 60:.2f} minutes")