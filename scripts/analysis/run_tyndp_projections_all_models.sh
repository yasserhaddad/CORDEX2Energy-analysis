paths_file="../../paths_so4.json"

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
# supply_scenario="LOW"
supply_scenario="HIGH"
# supply_year=2050
supply_year=2040
# rolling_window_residual_load=8
rolling_window_residual_load=1

# ── Reservoir hydro dispatch ──────────────────────────────────────────────────
# Set use_reservoir_hydro=true to apply Gurobi QP dispatch before aggregation.
# reservoir_min_storage_fraction: optional floor on storage (e.g. 0.10); leave
# empty to disable the floor constraint.
use_reservoir_hydro=true
reservoir_min_storage_fraction="0.10"   # e.g. "0.10" or leave empty for no floor

# ── Battery dispatch ──────────────────────────────────────────────────────────
# use_batteries=true applies utility battery dispatch (24 h lookahead).
# use_prosumer_batteries=true additionally applies PV-aware prosumer dispatch.
use_batteries=true
use_prosumer_batteries=true
battery_duration_h=4.0
battery_roundtrip_eff=0.85
battery_initial_soc=0.50
battery_dispatch_lookahead_h=24.0

for model in "${models[@]}"; do
    echo "Running TYNDP projections for model: $model, scenario: $scenario"

    reservoir_args=()
    if [ "$use_reservoir_hydro" = true ]; then
        reservoir_args+=(--use_reservoir_hydro)
        if [ -n "$reservoir_min_storage_fraction" ]; then
            reservoir_args+=(--reservoir_min_storage_fraction "$reservoir_min_storage_fraction")
        fi
    fi

    battery_args=()
    if [ "$use_batteries" = true ]; then
        battery_args+=(--use_batteries)
        battery_args+=(--battery_duration_h "$battery_duration_h")
        battery_args+=(--battery_roundtrip_eff "$battery_roundtrip_eff")
        battery_args+=(--battery_initial_soc "$battery_initial_soc")
        battery_args+=(--battery_dispatch_lookahead_h "$battery_dispatch_lookahead_h")
        if [ "$use_prosumer_batteries" = true ]; then
            battery_args+=(--use_prosumer_batteries)
        fi
    fi

    python compute_tyndp_projections_extremes.py \
        --paths_file "$paths_file" \
        --climate_model "$model" \
        --climate_scenario "$scenario" \
        --demand_scenario "$demand_scenario" \
        --demand_year "$demand_year" \
        --supply_scenario "$supply_scenario" \
        --supply_year "$supply_year" \
        --rolling_window_residual_load "$rolling_window_residual_load" \
        "${reservoir_args[@]}" \
        "${battery_args[@]}"
done
