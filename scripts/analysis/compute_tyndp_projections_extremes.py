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
import json
import os
from pathlib import Path

import dask
import numpy as np
import pandas as pd
import xarray as xr

from cordex2energy.conversion_energy.tyndp import (
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
    load_nuts_geodata,
    load_tyndp_battery_capacities,
    load_tyndp_country_codes,
    load_tyndp_market_profiles,
    load_tyndp_supply_capacities,
    preprocess_hydro_ror,
)

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

if __name__ == "__main__":
    # ── CLI ───────────────────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description=(
            "Compute TYNDP-scenario residual load from CORDEX climate projections."
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
        choices=["rcp26", "rcp45", "rcp85"],
        help="CORDEX RCP climate scenario.",
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
        default=None,
        help=(
            "Minimum reservoir storage level as a fraction of maximum capacity "
            "(e.g. 0.05).  Passed to the Gurobi QP as a lower bound on storage "
            "at every time step.  Default None (no floor)."
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
        help="Lookahead window in hours for utility battery discharge rationing (default 24).",
    )

    args = parser.parse_args()

    # ── Derived parameters ────────────────────────────────────────────────────────

    climate_model = args.climate_model
    climate_scenario = args.climate_scenario
    demand_scenario = args.demand_scenario
    demand_year = args.demand_year
    supply_scenario = args.supply_scenario
    supply_year = args.supply_year
    rolling_window_residual_load = args.rolling_window_residual_load

    # Map CLI supply scenario to the column suffix used in the TYNDP Excel files.
    _SUPPLY_SCENARIO_MAP = {"LOW": "LOW", "BE": "Best Estimate", "HIGH": "HIGH"}
    supply_scenario_col = _SUPPLY_SCENARIO_MAP[supply_scenario]

    # Derive the C2E model prefix (drops the ensemble member, e.g. _r1i1p1).
    model_prefix = "_".join(climate_model.split("_")[:-1])

    # Turbine name as it appears in filenames (/ → _).
    turbine_fn = args.turbine.replace("/", "_")

    # Short RCP suffix used in some C2E filenames (e.g. "85" from "rcp85").
    rcp_suffix = climate_scenario[-2:]

    print(f"Climate model    : {climate_model}")
    print(f"Climate scenario : {climate_scenario}")
    print(f"Demand scenario  : {demand_scenario} {demand_year}")
    print(f"Supply scenario  : {supply_scenario} ({supply_scenario_col}) {supply_year}")

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
    _model_tag = model_prefix
    _run_tag = (
        f"{demand_scenario}_{demand_year}"
        f"_{supply_scenario}_{supply_year}"
    )
    if rolling_window_residual_load and rolling_window_residual_load > 1:
        _run_tag += f"_rlsmooth_{rolling_window_residual_load}"
    if args.use_reservoir_hydro:
        _run_tag += "_reservoirdispatch"
    if args.use_batteries:
        _run_tag += "_batteries"
        if args.use_prosumer_batteries:
            _run_tag += "_prosumer"

    # Output directory: one subfolder per run configuration.
    path_out = (
        path_data_extremes / "residual_load"
        / f"{climate_model}_{climate_scenario}"
        / _run_tag
    )

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

    # ── Load CORDEX energy datasets ───────────────────────────────────────────────

    print("\n── Loading CORDEX energy data ───────────────────────────────────────────────")
    cordex_data = load_cordex_energy_data(
        path_c2e=path_c2e,
        path_energy_europe=path_data_energy_europe,
        climate_model=climate_model,
        model_prefix=model_prefix,
        climate_scenario=climate_scenario,
        rcp_suffix=rcp_suffix,
        turbine_fn=turbine_fn,
        name_to_iso2=C2E_NAME_TO_ISO2,
    )
    ds_pv = cordex_data["ds_pv"]
    ds_wind = cordex_data["ds_wind"]
    ds_wind_offshore = cordex_data["ds_wind_offshore"]
    ds_hydro_ror = cordex_data["ds_hydro_ror"]
    ds_hydro_inflow = cordex_data["ds_hydro_inflow"]
    ds_heating_demand = cordex_data["ds_heating_demand"]
    ds_cooling_demand = cordex_data["ds_cooling_demand"]

    # ── Compute or load baseload ──────────────────────────────────────────────────

    print("\n── Computing / loading TYNDP baseload ───────────────────────────────────────")
    ds_tyndp_baseload_typical = compute_or_load_baseload(
        path_c2e_cerra=path_data_processed / "c2e" / "outputs" / "CERRA",
        path_baseload_cache=path_baseload_cache,
        path_tyndp=path_tyndp,
        demand_scenario=demand_scenario,
        demand_year=demand_year,
        df_tyndp_demand=df_tyndp_demand,
        C2E_NAME_TO_ISO2=C2E_NAME_TO_ISO2,
        floor_fraction=args.floor_fraction,
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
    print(f"  {len(thermal_targets)} countries with thermal targets.")

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
        target_twh_per_country=thermal_targets,
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
        if args.use_prosumer_batteries:
            print("  Prosumer batteries (PV-aware, daily cycling) …")
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

        print(f"  Utility batteries (lookahead={args.battery_dispatch_lookahead_h:.0f} h) …")
        _ds_util = apply_battery_dispatch(
            ds_residual_load,
            s_batt_utility,
            battery_duration_h=args.battery_duration_h,
            roundtrip_efficiency=args.battery_roundtrip_eff,
            initial_soc_fraction=args.battery_initial_soc,
            dispatch_lookahead_h=args.battery_dispatch_lookahead_h,
            zone_dim="country",
        )
        utility_dispatch_gwh = _ds_util["battery_dispatch_gwh"]
        ds_residual_load = _ds_util.assign(
            residual_load=_ds_util["residual_load_with_batteries"]
        ).drop_vars(["residual_load_with_batteries", "soc_gwh", "battery_dispatch_gwh"])
        ds_components = ds_components.assign(utility_dispatch_gwh=utility_dispatch_gwh)

        battery_dispatch_gwh = (
            prosumer_dispatch_gwh + utility_dispatch_gwh
            if prosumer_dispatch_gwh is not None
            else utility_dispatch_gwh
        )
        ds_components = ds_components.assign(battery_dispatch_gwh=battery_dispatch_gwh)
        print("  Done.")

    # Aggregate from individual countries to real-time balancing zones.
    ds_residual_load_zonal = aggregate_by_balancing_zones(ds_residual_load)
    ds_components_zonal = aggregate_by_balancing_zones(ds_components)

    # ── Save outputs ──────────────────────────────────────────────────────────────

    print("\n── Saving outputs ───────────────────────────────────────────────────────────")

    # Attach run metadata as global attributes.
    _attrs = {
        "climate_model": climate_model,
        "climate_scenario": climate_scenario,
        "demand_scenario": demand_scenario,
        "demand_year": demand_year,
        "supply_scenario": supply_scenario_col,
        "supply_year": supply_year,
        "turbine": args.turbine,
        "rolling_window_residual_load": rolling_window_residual_load,
        "use_reservoir_hydro": str(args.use_reservoir_hydro),
        "reservoir_min_storage_fraction": args.reservoir_min_storage_fraction,
        "use_batteries": str(args.use_batteries),
        "use_prosumer_batteries": str(args.use_prosumer_batteries),
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

    print("\nDone.")