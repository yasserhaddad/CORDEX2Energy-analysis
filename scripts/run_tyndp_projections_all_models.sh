paths_file="../paths.json"

models=(
    "CNRM-ALADIN63_CNRM-CERFACS-CNRM-CM5_r1i1p1"
    "CNRM-ALADIN63_MOHC-HadGEM2-ES_r1i1p1"
    "CNRM-ALADIN63_NCC-NorESM1-M_r1i1p1"
    "CNRM-ALADIN63_MPI-M-MPI-ESM-LR_r1i1p1"
    "MOHC-HadREM3-GA7-05_MOHC-HadGEM2-ES_r1i1p1"
    "MOHC-HadREM3-GA7-05_ICHEC-EC-EARTH_r12i1p1"
    "MOHC-HadREM3-GA7-05_MPI-M-MPI-ESM-LR_r1i1p1"
    "MOHC-HadREM3-GA7-05_NCC-NorESM1-M_r1i1p1"
    "MOHC-HadREM3-GA7-05_CNRM-CERFACS-CNRM-CM5_r1i1p1"
)

scenario=("rcp85")

demand_scenario="DE"
# demand_scenario="GA"
demand_year=2050

# ── Demand model ──────────────────────────────────────────────────────────────
# Source of the climate-driven heating/cooling demand. Encoded in the run tag and
# baseload cache key. 'cordex2energy_demand' uses the hourly JRC-coefficient
# NetCDFs (energy/europe/demand/*.nc); 'demand.ninja' uses the 3-hourly c2e CSVs.
# demand_model="demand.ninja"
demand_model="cordex2energy_demand"
# Set legacy_thermal=true to reproduce the pre-change lumped/whole-series thermal
# handling (no baseload modulation) for validation.
legacy_thermal=false

# ── Cooling target ────────────────────────────────────────────────────────────
# rescale_cooling=false (the paper's setting) leaves cooling at the level the demand model derives
# and passes --no-cooling-target, writing to a '_freecool' run tag. Heating stays pinned to its own
# TYNDP target and the baseload is unchanged.
#
# The asymmetry is empirical, not a preference. Comparing each TYNDP target with the JRC-IDEES
# series the model is actually fitted to, in the same base year (2019):
#   heating  TYNDP/JRC = 1.04x median (1.08x aggregate), scattered — 6 of 12 countries below 1
#   cooling  TYNDP/JRC = 2.03x median (1.71x aggregate), biased  — 10 of 12 above 1
# TYNDP's 'Cooling' is systematically broader than the space cooling the model fits, and its
# Italy/Spain ordering is inverted relative to every observed year. Because k multiplies the whole
# series it amplifies the climate TREND by the same factor, so k_cool = 3.55 for Italy drove the
# South-East zone into threshold saturation (43.7 % of JJA timesteps at +3.0 C). See
# notebooks/paper_metric_and_target_diagnostics.ipynb.
#
# Set rescale_cooling=true for the upper-bracket sensitivity (cooling pinned to the TYNDP target).
rescale_cooling=false

# supply_scenario="LOW"
supply_scenario="HIGH"
supply_year=2050
# supply_year=2040
# rolling_window_residual_load=8
rolling_window_residual_load=1

# ── Wind turbine ──────────────────────────────────────────────────────────────
# Selects which precomputed wind capacity factors are read, and is ALWAYS part of
# the run tag (so turbines never share an output folder). The choice sets the CF
# level outright — mean onshore CF is 0.20 for E-126/7580 vs 0.40 for SWT142/3150 —
# so each turbine needs its own CERRA pass to re-calibrate the p95 threshold.
# Only these three have precomputed capacity factors (EUROPE_TURBINES).
# turbine="E-126/7580"
turbine="SWT120/3600" # the paper's default
# turbine="SWT142/3150"

# ── Reservoir hydro dispatch ──────────────────────────────────────────────────
# Set use_reservoir_hydro=true to apply Gurobi QP dispatch before aggregation.
# reservoir_min_storage_fraction: optional floor on storage (e.g. 0.10); leave
# empty to disable the floor constraint.
use_reservoir_hydro=true
reservoir_min_storage_fraction="0.10"   # e.g. "0.10" or leave empty for no floor
reservoir_efficiency=0.9                # turbine water-to-wire efficiency (0,1]

# ── Battery dispatch ──────────────────────────────────────────────────────────
# use_batteries=true applies utility battery dispatch.
# use_prosumer_batteries=true additionally applies PV-aware prosumer dispatch
#   (prosumer is ALWAYS the greedy heuristic; the method switch below is for utility).
#
# battery_dispatch_method: "heuristic" (greedy + lookahead rationing) or
#   "optimal" (rolling day-ahead Gurobi QP; requires a Gurobi licence).
#   For "optimal", battery_commit_h is the committed horizon per solve and
#   battery_dispatch_lookahead_h the extra look-ahead solved and discarded.
# battery_min_storage_fraction: optional SOC floor for the optimal method
#   (e.g. 0.05); leave empty to disable.
#
# flexibility_scheme: "market" replicates market operation (prosumer heuristic +
#   utility battery_dispatch_method at utility_dispatch_level).  "system" lumps
#   prosumer+utility into one asset and optimises it with the same annual-cyclical
#   QP as the reservoir, at country level (net-load-event mitigation); under
#   "system" battery_dispatch_method and utility_dispatch_level are ignored.
use_batteries=true
use_prosumer_batteries=true
flexibility_scheme="system"              # "market" or "system"
battery_dispatch_method="optimal"        # "heuristic" or "optimal"  (market only)
battery_duration_h=4.0
battery_roundtrip_eff=0.85
battery_initial_soc=0.50
battery_dispatch_lookahead_h=24.0
battery_commit_h=24.0                     # optimal only: committed day-ahead horizon
battery_min_storage_fraction=""           # optimal only: e.g. "0.05" or empty for none
utility_dispatch_level="country"             # "zone" (pool over 6 balancing zones) or "country"

reservoir_args=()
if [ "$use_reservoir_hydro" = true ]; then
    reservoir_args+=(--use_reservoir_hydro)
    reservoir_args+=(--reservoir_efficiency "$reservoir_efficiency")
    if [ -n "$reservoir_min_storage_fraction" ]; then
        reservoir_args+=(--reservoir_min_storage_fraction "$reservoir_min_storage_fraction")
    fi
fi

battery_args=()
if [ "$use_batteries" = true ]; then
    battery_args+=(--use_batteries)
    battery_args+=(--flexibility_scheme "$flexibility_scheme")
    battery_args+=(--battery_duration_h "$battery_duration_h")
    battery_args+=(--battery_roundtrip_eff "$battery_roundtrip_eff")
    battery_args+=(--battery_initial_soc "$battery_initial_soc")
    if [ "$use_prosumer_batteries" = true ]; then
        battery_args+=(--use_prosumer_batteries)
    fi
    if [ "$flexibility_scheme" = "system" ]; then
        # Lumped batteries, annual-cyclical QP at country level.  Only the SOC
        # floor applies; method / level / lookahead / commit are ignored.
        if [ -n "$battery_min_storage_fraction" ]; then
            battery_args+=(--battery_min_soc_fraction "$battery_min_storage_fraction")
        fi
    else
        battery_args+=(--battery_dispatch_method "$battery_dispatch_method")
        battery_args+=(--utility_dispatch_level "$utility_dispatch_level")
        battery_args+=(--battery_dispatch_lookahead_h "$battery_dispatch_lookahead_h")
        if [ "$battery_dispatch_method" = "optimal" ]; then
            battery_args+=(--battery_commit_h "$battery_commit_h")
            if [ -n "$battery_min_storage_fraction" ]; then
                battery_args+=(--battery_min_soc_fraction "$battery_min_storage_fraction")
            fi
        fi
    fi
fi

# ── Demand-model args (model selector, legacy-thermal and cooling-target flags) ─
demand_args=(--demand_model "$demand_model")
if [ "$legacy_thermal" = "true" ]; then
    demand_args+=(--legacy_thermal)
fi
if [ "$rescale_cooling" = "false" ]; then
    demand_args+=(--no-cooling-target)
fi

for model in "${models[@]}"; do
    echo "Running TYNDP projections for model: $model, scenario: $scenario"



    python compute_tyndp_projections_extremes.py \
        --paths_file "$paths_file" \
        --climate_model "$model" \
        --climate_scenario "$scenario" \
        --demand_scenario "$demand_scenario" \
        --demand_year "$demand_year" \
        --supply_scenario "$supply_scenario" \
        --supply_year "$supply_year" \
        --turbine "$turbine" \
        --rolling_window_residual_load "$rolling_window_residual_load" \
        "${demand_args[@]}" \
        "${reservoir_args[@]}" \
        "${battery_args[@]}"
done

python compute_tyndp_projections_extremes.py \
    --paths_file "$paths_file" \
    --climate_model cerra \
    --climate_scenario none \
    --demand_scenario "$demand_scenario" \
    --demand_year "$demand_year" \
    --supply_scenario "$supply_scenario" \
    --supply_year "$supply_year" \
    --turbine "$turbine" \
    --rolling_window_residual_load "$rolling_window_residual_load" \
    "${demand_args[@]}" \
    "${reservoir_args[@]}" \
    "${battery_args[@]}"

