from typing import Union

import numpy as np
import pandas as pd
import scipy.stats as stats
import xarray as xr
from scipy.ndimage import label


def _gpd_right_tail_ad_statistic(z_sorted: np.ndarray) -> float:
    """Right-tail Anderson-Darling statistic for n sorted PIT values in (0, 1).

    Formula (Choulakian & Stephens 2001; Bader, Yan & Zhang 2018 ``eva`` package):

        A²_right = -n/2 - (1/n) Σᵢ₌₁ⁿ (2n − 2i + 1) ln(1 − z_i)

    where z_i are ordered probability-integral-transform values of the excesses
    under the fitted GPD.  Larger values indicate worse fit in the upper tail.
    """
    n = len(z_sorted)
    z = np.clip(z_sorted, 1e-10, 1.0 - 1e-10)
    i = np.arange(1, n + 1)
    return float(-n / 2.0 - np.sum((2.0 * n - 2.0 * i + 1.0) * np.log(1.0 - z)) / n)


def _gpd_right_tail_ad_batch(z_sorted_2d: np.ndarray) -> np.ndarray:
    """Vectorised right-tail AD statistic for a 2-D array of PIT rows.

    Parameters
    ----------
    z_sorted_2d : (m, n) array
        Each row contains n sorted PIT values in (0, 1).

    Returns
    -------
    (m,) array of A²_right statistics.
    """
    m, n = z_sorted_2d.shape
    z = np.clip(z_sorted_2d, 1e-10, 1.0 - 1e-10)
    weights = 2.0 * n - 2.0 * np.arange(1, n + 1) + 1.0  # (n,)
    return -n / 2.0 - (weights * np.log(1.0 - z)).sum(axis=1) / n  # (m,)


def find_evt_pot_threshold(
    zone_data,
    min_p: float = 0.85,
    max_p: float = 0.995,
    num_candidates: int = 50,
    n_bootstrap: int = 199,
    p_value_threshold: float = 0.05
) -> float:
    """Find a statistically robust POT threshold using the right-tail Anderson-Darling test.

    Scans candidate thresholds from low to high percentile (85th → 99.5th), fitting a
    GPD to the excesses and testing the fit with the right-tail weighted Anderson-Darling
    (A²_right) test.  The first (lowest) threshold where the test does not reject at
    α = p_value_threshold is returned, the default being 0.05 to maximise the number of
    exceedances and minimise estimation variance.

    The p-value is estimated via a parametric bootstrap.  To keep the bootstrap fast,
    the bootstrap refits use L-moment estimators (closed-form, no optimiser) rather than
    MLE.  L-moments are consistent for GPD when ξ < 1 (all finite-mean cases encountered
    in practice) and produce a valid null distribution for the purpose of calibrating the
    test.  MLE is still used for the single fit to the observed excesses.

    Parameters
    ----------
    zone_data :
        1-D array of 3-hourly residual loads for the reference period.
    min_p :
        Lowest percentile (of positive RL values) to consider as a candidate.
    max_p :
        Highest percentile to consider.
    num_candidates :
        Number of evenly spaced candidate thresholds.
    n_bootstrap :
        Bootstrap replicates for the p-value estimate.

    Returns
    -------
    float
        Robust POT threshold in GW.  Falls back to the 99th percentile of
        positive RL values if no candidate passes the test.
    """
    positive_rl = zone_data[zone_data > 0]

    candidates = np.percentile(
        positive_rl, np.linspace(min_p * 100, max_p * 100, num_candidates)
    )

    best_threshold = None
    rng = np.random.default_rng(0)  # fixed seed → deterministic across models

    # Scan LOW→HIGH: stop at the first (lowest) threshold where GPD holds.
    # This maximises exceedances and minimises estimation variance.
    for u in candidates:
        excesses = positive_rl[positive_rl > u] - u

        if len(excesses) < 30:
            break  # going higher will only reduce exceedances further

        # MLE for the observed excesses (accurate single fit)
        try:
            c, loc, scale = stats.genpareto.fit(excesses, floc=0)
        except Exception:
            continue

        z_obs = np.sort(stats.genpareto.cdf(excesses, c, loc=loc, scale=scale))
        t_obs = _gpd_right_tail_ad_statistic(z_obs)

        # ── Vectorised parametric bootstrap ───────────────────────────────────
        # Draw all bootstrap samples at once and fit via L-moments (closed-form).
        # L-moments: λ1 = σ/(1−ξ), λ2 = σ/((1−ξ)(2−ξ))  →  ξ = 2 − λ1/λ2
        n_exc = len(excesses)
        sim_sorted = np.sort(
            stats.genpareto.rvs(c, loc=0, scale=scale, size=(n_bootstrap, n_exc), random_state=rng),
            axis=1,
        )

        # PWM-based L-moment estimators (vectorised over bootstrap dimension)
        w_b1 = np.arange(n_exc) / max(n_exc - 1, 1)  # (n_exc,) weights for β1
        l1_b = sim_sorted.mean(axis=1)                # (n_bootstrap,)
        b1_b = (sim_sorted * w_b1).mean(axis=1)
        l2_b = 2.0 * b1_b - l1_b
        xi_b = 2.0 - l1_b / l2_b
        sigma_b = l1_b * (1.0 - xi_b)

        # Discard rows with degenerate L-moment fits
        valid = (sigma_b > 0) & np.isfinite(xi_b) & np.isfinite(sigma_b) & (xi_b < 1.0)
        n_valid = int(valid.sum())
        if n_valid == 0:
            continue

        # GPD CDF for all valid bootstrap samples (vectorised)
        xi_v = xi_b[valid, np.newaxis]     # (n_valid, 1)
        sig_v = sigma_b[valid, np.newaxis]  # (n_valid, 1)
        x_v = sim_sorted[valid]             # (n_valid, n_exc)

        inner = np.maximum(1.0 + xi_v * x_v / sig_v, 1e-10)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            z_pow = 1.0 - inner ** (-1.0 / xi_v)
            z_exp = 1.0 - np.exp(-x_v / sig_v)  # limit as ξ → 0
            z_b = np.where(np.abs(xi_v) > 1e-6, z_pow, z_exp)

        # Right-tail AD statistics for all bootstrap samples in one call
        t_b = _gpd_right_tail_ad_batch(z_b)

        p_value = (float(np.sum(t_b >= t_obs)) + 1.0) / (n_valid + 1.0)

        if p_value > p_value_threshold:
            best_threshold = u
            break  # lowest threshold where GPD holds — stop here

    if best_threshold is None:
        best_threshold = np.percentile(positive_rl, 99)

    return float(best_threshold)


def compute_evt_thresholds_for_all_zones(
    ds_reference, p_value_threshold=0.05, country_dim="zone"
) -> "tuple[xr.DataArray, xr.DataArray]":
    """Vectorized wrapper to run the EVT threshold identification across all zones.

    Returns
    -------
    da_thresholds :
        EVT threshold in GW for each zone.
    da_percentiles :
        Empirical percentile (0–100) of the threshold within all 3-hourly RL
        values of the reference period (including negative values).  This makes
        the threshold directly comparable across zones regardless of the
        fraction of negative-RL hours each zone has.
    """
    threshold_dict: dict = {}
    percentile_dict: dict = {}

    for zone in ds_reference[country_dim].values:
        zone_series = ds_reference.sel({country_dim: zone}).values.flatten()

        u_robust = find_evt_pot_threshold(zone_series, p_value_threshold=p_value_threshold)
        threshold_dict[zone] = u_robust
        # Percentile of the threshold in the full (signed) RL distribution
        percentile_dict[zone] = float(np.mean(zone_series <= u_robust) * 100)

    _zones = list(threshold_dict.keys())
    da_thresholds = xr.DataArray(
        list(threshold_dict.values()),
        coords={country_dim: _zones},
        dims=[country_dim],
        name="evt_threshold",
    )
    da_percentiles = xr.DataArray(
        list(percentile_dict.values()),
        coords={country_dim: _zones},
        dims=[country_dim],
        name="evt_threshold_pct",
    )

    return da_thresholds, da_percentiles


def calculate_climatological_baseline(ds_reference_period, time_dim="time"):
    """Computes expected 3-hourly seasonal baselines for demand and EVERY individual

    generation asset type using a combined MultiIndex coordinate.
    """
    # 1. Extract the time components
    months = ds_reference_period[time_dim].dt.month
    hours = ds_reference_period[time_dim].dt.hour

    # 2. Create a pandas MultiIndex from the two arrays
    multi_idx = pd.MultiIndex.from_arrays(
        [months.values, hours.values], names=["month", "hour"]
    )

    # 3. Assign this index as a new coordinate matching the 'time' dimension
    ds_stacked = ds_reference_period.assign_coords(
        month_hour=(time_dim, multi_idx)
    )

    # Target variables include demand and all granular generation types
    target_vars = ["total_demand", "gen_pv", "gen_wind_on", "gen_wind_off", "gen_hydro_ror"]
    if "reservoir_dispatch_gwh" in ds_reference_period.data_vars:
        target_vars.append("reservoir_dispatch_gwh")
    for _bat_var in ["battery_dispatch_gwh", "prosumer_dispatch_gwh", "utility_dispatch_gwh"]:
        if _bat_var in ds_reference_period.data_vars:
            target_vars.append(_bat_var)

    # 4. Group by the combined coordinate and take the mean, then unstack it back
    # into clean 'month' and 'hour' dimensions
    baseline = (
        ds_stacked[target_vars]
        .groupby("month_hour")
        .mean(dim=time_dim)
        .unstack("month_hour")
    )

    return baseline


def compute_continuous_system_metrics(
    ds_residual_load, ds_components, ds_baseline, zone_dim="zone", time_dim="time"
):
    """Computes continuous time-series metrics (Ramping and PCIT) across the
    entire dataset before slicing into isolated extreme event blocks.
    """
    # Create a shallow copy to store new variables without modifying input data
    ds_metrics = ds_residual_load.copy()

    # --- 1. COMPUTE RAMPING RATES (GW/hour) ---
    # Shift the dataset by 1 time step to get the previous step's residual load
    rl_prev = ds_metrics["residual_load"].shift({time_dim: 1})

    # Delta over 3 hours divided by 3 gives the average hourly rate of change
    ds_metrics["ramping_rate"] = (ds_metrics["residual_load"] - rl_prev) / 3.0

    # Fill the first NaN step with 0 to keep array lengths identical
    ds_metrics["ramping_rate"] = ds_metrics["ramping_rate"].fillna(0.0)

    # --- 2. COMPUTE CONTINUOUS GLOBAL PCIT (GW) ---
    # Sum of absolute imbalances across all zones per time step
    sum_absolute_imbalances = np.abs(ds_metrics["residual_load"]).sum(dim=zone_dim)

    # Absolute value of the net continental imbalance per time step
    net_continental_imbalance = np.abs(ds_metrics["residual_load"].sum(dim=zone_dim))
    ds_metrics["net_continental_imbalance"] = net_continental_imbalance
    ds_metrics["net_system_residual_load"] = ds_metrics["residual_load"].sum(dim=zone_dim)

    # Apply the 1/2 scalar to extract cross-border transport volume
    ds_metrics["pcit"] = 0.5 * (sum_absolute_imbalances - net_continental_imbalance)

    # --- 3. COMPUTE CLIMATE ANOMALIES VIA EXTERNAL BASELINE ---
    # Extract time components from the current target dataset
    current_months = ds_metrics[time_dim].dt.month
    current_hours = ds_metrics[time_dim].dt.hour

    # Vectorized broadcasting: map the external (month, hour) baseline matrix
    # directly onto the full continuous timeline of the target dataset
    # Clip negative values to 0.0 to prevent localized generation or demand
    # surpluses from mathematically masking climate drivers during crisis hours
    expected_d = ds_baseline["total_demand"].sel(month=current_months, hour=current_hours)
    ds_metrics["anomaly_demand"] = (ds_components["total_demand"] - expected_d).clip(min=0.0)

    for tech in ["gen_pv", "gen_wind_on", "gen_wind_off", "gen_hydro_ror"]:
        expected_tech = ds_baseline[tech].sel(month=current_months, hour=current_hours)
        # Positive anomaly means generation fell below the seasonal expectation
        ds_metrics[f"anomaly_{tech}"] = (expected_tech - ds_components[tech]).clip(min=0.0)

    if (
        "reservoir_dispatch_gwh" in ds_components.data_vars
        and "reservoir_dispatch_gwh" in ds_baseline.data_vars
    ):
        expected_res = ds_baseline["reservoir_dispatch_gwh"].sel(
            month=current_months, hour=current_hours
        )
        ds_metrics["anomaly_reservoir_dispatch_gwh"] = (
            expected_res - ds_components["reservoir_dispatch_gwh"]
        ).clip(min=0.0)

    # Battery: positive anomaly = dispatched less than seasonal expectation
    if (
        "battery_dispatch_gwh" in ds_components.data_vars
        and "battery_dispatch_gwh" in ds_baseline.data_vars
    ):
        expected_bat = ds_baseline["battery_dispatch_gwh"].sel(
            month=current_months, hour=current_hours
        )
        ds_metrics["anomaly_battery_dispatch_gwh"] = (
            expected_bat - ds_components["battery_dispatch_gwh"]
        ).clip(min=0.0)

    return ds_metrics


def compute_comprehensive_event_metrics(
    ds_with_continuous_metrics, da_thresholds, zone_dim="zone"
):
    """Computes Frequency, Duration, Peak Capacity, and Energy Integral metrics,
    while contextually extracting Ramping and PCIT signatures during threshold violations.
    """
    metrics_summary = {}

    # Extract years once — shared across all zones in this period
    year_series = ds_with_continuous_metrics.time.dt.year.values
    all_years   = np.unique(year_series)

    for zone in ds_with_continuous_metrics[zone_dim].values:
        # Extract the specific threshold for this zone
        H = float(da_thresholds.sel({zone_dim: zone}).values)

        # Pull the continuous numpy arrays for the current zone
        rl_series = (
            ds_with_continuous_metrics["residual_load"].sel({zone_dim: zone}).values
        )
        ramp_series = (
            ds_with_continuous_metrics["ramping_rate"].sel({zone_dim: zone}).values
        )

        # The demand and generation anomalies, crucial for understanding the underlying climate drivers during extreme events
        d_anom_series = (
            ds_with_continuous_metrics["anomaly_demand"].sel({zone_dim: zone}).values
        )
        pv_anom_series = (
            ds_with_continuous_metrics["anomaly_gen_pv"].sel({zone_dim: zone}).values
        )
        w_anom_series = (
            ds_with_continuous_metrics["anomaly_gen_wind_on"].sel({zone_dim: zone}).values
        )
        woff_anom_series = (
            ds_with_continuous_metrics["anomaly_gen_wind_off"].sel({zone_dim: zone}).values
        )
        hydro_anom_series = (
            ds_with_continuous_metrics["anomaly_gen_hydro_ror"].sel({zone_dim: zone}).values
        )
        if "anomaly_reservoir_dispatch_gwh" in ds_with_continuous_metrics:
            reservoir_anom_series = (
                ds_with_continuous_metrics["anomaly_reservoir_dispatch_gwh"].sel({zone_dim: zone}).values
            )
        else:
            reservoir_anom_series = np.zeros(len(rl_series))

        if "anomaly_battery_dispatch_gwh" in ds_with_continuous_metrics:
            battery_anom_series = (
                ds_with_continuous_metrics["anomaly_battery_dispatch_gwh"].sel({zone_dim: zone}).values
            )
        else:
            battery_anom_series = np.zeros(len(rl_series))

        # PCIT is shared globally across the system, so we don't slice it by zone
        pcit_series = ds_with_continuous_metrics["pcit"].values
        net_system_rl_series = ds_with_continuous_metrics["net_system_residual_load"].values

        # Number of years in this period (used to normalise total → annual)
        n_years = len(all_years)

        # Identify threshold violations
        is_extreme = rl_series > H

        # Label continuous blocks of threshold violations
        labeled_events, num_features = label(is_extreme)

        if num_features == 0:
            metrics_summary[zone] = {
                "Frequency": 0,
                "Avg_Duration_Hours": 0.0,
                "Max_Duration_Hours": 0.0,
                "Max_Peak_Intensity_GW": 0.0,
                "Max_Upward_Ramp_GW_per_hr": 0.0,
                "Total_Annual_Energy_Severity_GWh": 0.0,
                "Avg_Energy_Severity_GWh": 0.0,
                "Max_Single_Event_Severity_GWh": 0.0,
                "Avg_PCIT_During_Events_GW": 0.0,
                "Max_PCIT_During_Events_GW": 0.0,
                "Avg_Net_System_Residual_Load_During_Events_GW": 0.0,
                "Max_Net_System_Residual_Load_During_Events_GW": 0.0,
                "Std_Annual_Energy_Severity_GWh": 0.0,
                "P90_Annual_Energy_Severity_GWh": 0.0,
                "CV_Annual_Energy_Severity": 0.0,
                "Demand_Driven_Share_%": 0.0,
                "Generation_Driven_Share_%": 0.0,
                "PV_Driven_Share_%": 0.0,
                "Wind_Onshore_Driven_Share_%": 0.0,
                "Wind_Offshore_Driven_Share_%": 0.0,
                "Hydro_RoR_Driven_Share_%": 0.0,
                "Reservoir_Hydro_Driven_Share_%": 0.0,
                "Battery_Driven_Share_%": 0.0,
                "Avg_Demand_Anomaly_GWh": 0.0,
                "Avg_PV_Anomaly_GWh": 0.0,
                "Avg_Wind_Onshore_Anomaly_GWh": 0.0,
                "Avg_Wind_Offshore_Anomaly_GWh": 0.0,
                "Avg_Hydro_RoR_Anomaly_GWh": 0.0,
                "Avg_Reservoir_Anomaly_GWh": 0.0,
                "Avg_Battery_Anomaly_GWh": 0.0,
            }
            continue

        (
            event_durations,
            event_peaks,
            event_integrals,
            event_max_ramps,
            event_mean_pcits,
            event_mean_net_system_rls,
        ) = [], [], [], [], [], []
        per_year_severity = {}   # {year: cumulative GWh above threshold}
        tot_d_anom = 0.0
        tot_pv_anom = 0.0
        tot_w_anom = 0.0
        tot_woff_anom = 0.0
        tot_hydro_anom = 0.0
        tot_reservoir_anom = 0.0
        tot_battery_anom = 0.0

        # Process each isolated event window
        for i in range(1, num_features + 1):
            event_indices = np.where(labeled_events == i)[0]

            # 1. Duration (1 step = 3 hours)
            duration_hours = len(event_indices) * 3
            event_durations.append(duration_hours)

            # 2. Peak Intensity (GW)
            event_rl_values = rl_series[event_indices]
            peak_gw = np.max(event_rl_values)
            event_peaks.append(peak_gw)

            # 3. Integral Energy Severity (GWh) with 3-hour Riemann adjustment
            integral_gwh = np.sum(event_rl_values - H) * 3
            event_integrals.append(integral_gwh)
            # Tag this event's energy to the calendar year it started in
            event_year = year_series[event_indices[0]]
            per_year_severity[event_year] = (
                per_year_severity.get(event_year, 0.0) + integral_gwh
            )

            # 4. Contextual Ramping (GW/hour)
            # Find the steepest upward ramp rate that occurred within this event block
            event_ramp_values = ramp_series[event_indices]
            max_ramp = np.max(event_ramp_values)
            event_max_ramps.append(max_ramp)

            # 5. Contextual PCIT (GW)
            # Find the average cross-border congestion volume during this specific event block
            event_pcit_values = pcit_series[event_indices]
            mean_pcit = np.mean(event_pcit_values)
            event_mean_pcits.append(mean_pcit)

            event_net_system_rl_values = net_system_rl_series[event_indices]
            mean_net_system_rl = np.mean(event_net_system_rl_values)
            event_mean_net_system_rls.append(mean_net_system_rl)

            # 6. Climate Driver Attribution
            # Sum the demand and generation anomalies during this event block to understand the underlying climate drivers
            # Accumulate the absolute GW deficits across all crisis hours for this zone
            tot_d_anom += np.sum(d_anom_series[event_indices])
            tot_pv_anom += np.sum(pv_anom_series[event_indices])
            tot_w_anom += np.sum(w_anom_series[event_indices])
            tot_woff_anom += np.sum(woff_anom_series[event_indices])
            tot_hydro_anom += np.sum(hydro_anom_series[event_indices])
            tot_reservoir_anom += np.sum(reservoir_anom_series[event_indices])
            tot_battery_anom += np.sum(battery_anom_series[event_indices])

        # ── Annual severity statistics (year-to-year variability) ─────────────
        # Years with no events contribute 0 GWh to the distribution.
        annual_vals = np.array([per_year_severity.get(yr, 0.0) for yr in all_years])
        std_annual  = float(np.std(annual_vals))
        p90_annual  = float(np.percentile(annual_vals, 90))
        mean_annual = float(np.sum(event_integrals)) / n_years  # consistent with Total_Annual
        cv_annual   = std_annual / mean_annual if mean_annual > 0 else 0.0

        # Compute climate driver percentages over the evaluated period
        # Total anomaly mass
        denominator = (
            tot_d_anom + tot_pv_anom + tot_w_anom + tot_woff_anom
            + tot_hydro_anom + tot_reservoir_anom + tot_battery_anom
        )

        if denominator > 0:
            d_share = (tot_d_anom / denominator) * 100
            pv_share = (tot_pv_anom / denominator) * 100
            w_share = (tot_w_anom / denominator) * 100
            woff_share = (tot_woff_anom / denominator) * 100
            hydro_share = (tot_hydro_anom / denominator) * 100
            reservoir_share = (tot_reservoir_anom / denominator) * 100
            battery_share = (tot_battery_anom / denominator) * 100
        else:
            d_share = pv_share = w_share = woff_share = hydro_share = reservoir_share = battery_share = 100.0 / 7

        # Compile the final dictionary summarizing the operational profile of the zone
        metrics_summary[zone] = {
            # --- Event Frequency & Longevity ---
            "Frequency": int(num_features),
            "Avg_Duration_Hours": float(np.mean(event_durations)),
            "Max_Duration_Hours": float(
                np.max(event_durations)
            ),  # <-- CRITICAL CAPABILITY FIXED
            # --- Capacity & Ramping Intensity (Power Constraints) ---
            "Max_Peak_Intensity_GW": float(np.max(event_peaks)),
            "Max_Upward_Ramp_GW_per_hr": float(np.max(event_max_ramps)),
            # --- Energy Volume Severity (Storage Constraints) ---
            "Total_Annual_Energy_Severity_GWh": float(np.sum(event_integrals)) / n_years,
            "Avg_Energy_Severity_GWh": float(np.mean(event_integrals)),
            "Max_Single_Event_Severity_GWh": float(
                np.max(event_integrals)
            ),  # <-- ISOLATED CRISIS BOUND
            # --- Grid Interconnection Support ---
            "Avg_PCIT_During_Events_GW": float(np.mean(event_mean_pcits)),
            "Max_PCIT_During_Events_GW": float(
                np.max(event_mean_pcits)
            ),  # <-- WORST-CASE INTERCONNECTOR STRAIN
            "Avg_Net_System_Residual_Load_During_Events_GW": float(np.mean(event_mean_net_system_rls)),
            "Max_Net_System_Residual_Load_During_Events_GW": float(
                np.max(event_mean_net_system_rls)
            ),  # <-- WORST-CASE CONTINENTAL IMBALANCE
            # --- Annual Severity Statistics (year-to-year variability) ---
            "Std_Annual_Energy_Severity_GWh": std_annual,
            "P90_Annual_Energy_Severity_GWh": p90_annual,
            "CV_Annual_Energy_Severity": cv_annual,
            # --- Climate Driver Attribution (shares) ---
            "Demand_Driven_Share_%": round(float(d_share), 1),
            "PV_Driven_Share_%": round(float(pv_share), 1),
            "Wind_Onshore_Driven_Share_%": round(float(w_share), 1),
            "Wind_Offshore_Driven_Share_%": round(float(woff_share), 1),
            "Hydro_RoR_Driven_Share_%": round(float(hydro_share), 1),
            "Reservoir_Hydro_Driven_Share_%": round(float(reservoir_share), 1),
            "Battery_Driven_Share_%": round(float(battery_share), 1),
            # --- Climate Driver Attribution (signed GWh anomaly per event) ---
            # Demand: positive = excess demand above baseline (raises RL).
            # Generation / flexibility: negative = shortfall / under-dispatch below baseline
            # (also raises RL, but shown with natural sign of the deviation).
            "Avg_Demand_Anomaly_GWh": round(float(tot_d_anom * 3 / num_features), 2),
            "Avg_PV_Anomaly_GWh": round(-float(tot_pv_anom * 3 / num_features), 2),
            "Avg_Wind_Onshore_Anomaly_GWh": round(-float(tot_w_anom * 3 / num_features), 2),
            "Avg_Wind_Offshore_Anomaly_GWh": round(-float(tot_woff_anom * 3 / num_features), 2),
            "Avg_Hydro_RoR_Anomaly_GWh": round(-float(tot_hydro_anom * 3 / num_features), 2),
            "Avg_Reservoir_Anomaly_GWh": round(-float(tot_reservoir_anom * 3 / num_features), 2),
            "Avg_Battery_Anomaly_GWh": round(-float(tot_battery_anom * 3 / num_features), 2),
        }

    return metrics_summary


def compile_and_compare_periods(*period_dicts, period_names=None):
    """Compile metrics from multiple eras into a structured DataFrame.

    Parameters
    ----------
    *period_dicts :
        One dict per period, each keyed by zone and containing metric dicts.
    period_names : list[str], optional
        Internal labels assigned to each period in the output DataFrame.
        When omitted and exactly three dicts are supplied the function uses
        ``["1_Reference", "2_Mid-Century", "3_End-Century"]`` for backward
        compatibility with callers that rely on ``PERIOD_SHORT`` remapping.
        Required when passing any number of periods other than three.
    """
    _legacy_names = ["1_Reference", "2_Mid-Century", "3_End-Century"]
    if period_names is None:
        if len(period_dicts) != 3:
            raise ValueError(
                "period_names is required when passing a number of periods other than 3"
            )
        period_names = _legacy_names

    records = []
    for period_name, period_data in zip(period_names, period_dicts):
        for zone, metrics in period_data.items():
            row = {"Period": period_name, "Zone": zone}
            row.update(metrics)
            records.append(row)

    df = pd.DataFrame(records)
    df_pivot = df.set_index(["Zone", "Period"]).sort_index()

    summary_metrics = [
        "Frequency",
        "Avg_Duration_Hours",
        "Max_Duration_Hours",
        "Total_Annual_Energy_Severity_GWh",
        "Avg_Energy_Severity_GWh",
        "Max_Single_Event_Severity_GWh",
        "Avg_PCIT_During_Events_GW",
        "Max_PCIT_During_Events_GW",
        "Std_Annual_Energy_Severity_GWh",
        "P90_Annual_Energy_Severity_GWh",
        "CV_Annual_Energy_Severity",
        "Demand_Driven_Share_%",
        "PV_Driven_Share_%",
        "Wind_Onshore_Driven_Share_%",
        "Wind_Offshore_Driven_Share_%",
        "Hydro_RoR_Driven_Share_%",
        "Reservoir_Hydro_Driven_Share_%",
        "Battery_Driven_Share_%",
        "Avg_Demand_Anomaly_GWh",
        "Avg_PV_Anomaly_GWh",
        "Avg_Wind_Onshore_Anomaly_GWh",
        "Avg_Wind_Offshore_Anomaly_GWh",
        "Avg_Hydro_RoR_Anomaly_GWh",
        "Avg_Reservoir_Anomaly_GWh",
        "Avg_Battery_Anomaly_GWh",
    ]

    ref_label = period_names[0]
    _use_legacy_suffixes = list(period_names) == _legacy_names

    deltas = {}
    for zone in df["Zone"].unique():
        deltas[zone] = {}
        for metric in summary_metrics:
            if metric not in df_pivot.columns:
                continue
            ref_val = df_pivot.loc[(zone, ref_label), metric]
            for j, other_label in enumerate(period_names[1:], start=1):
                other_val = df_pivot.loc[(zone, other_label), metric]
                pct = ((other_val - ref_val) / ref_val * 100) if ref_val > 0 else 0
                if _use_legacy_suffixes:
                    suffix = "Mid" if j == 1 else "End"
                else:
                    suffix = other_label.split("_", 1)[-1] if "_" in other_label else other_label
                deltas[zone][f"{metric}_{suffix}_Change_%"] = round(pct, 1)

    df_deltas = pd.DataFrame.from_dict(deltas, orient="index")
    df_deltas.index.name = "Zone"

    return df_pivot, df_deltas


# ---------------------------------------------------------------------------
# GWL period resolution
# ---------------------------------------------------------------------------

# CORDEX GCM field (institution-prefixed) → CMIP5 short name used in the YAML.
# Hard-coded because this mapping is fixed for the EURO-CORDEX 9-model ensemble.
_CORDEX_GCM_TO_CMIP5: dict[str, str] = {
    "CNRM-CERFACS-CNRM-CM5": "CNRM-CM5",
    "MOHC-HadGEM2-ES":        "HadGEM2-ES",
    "NCC-NorESM1-M":          "NorESM1-M",
    "MPI-M-MPI-ESM-LR":       "MPI-ESM-LR",
    "ICHEC-EC-EARTH":         "EC-EARTH",
}

_GWL_COLORS = ["#4A90E2", "#F5A623", "#E24A4A", "#7B2D8B"]


def resolve_gwl_period_definitions(
    warming_levels: list[float],
    models: list[str],
    chain_meta_path,
    gwl_yaml_path,
    climate_scenario: str = "rcp85",
) -> tuple[dict, dict, dict]:
    """Build per-model GWL period definitions from a warming-levels YAML.

    Parameters
    ----------
    warming_levels :
        GWL values to include (e.g. ``[1.5, 2.0, 3.0]``).
    models :
        Model chain IDs (e.g. ``"CNRM-ALADIN63_CNRM-CERFACS-CNRM-CM5_r1i1p1"``).
    chain_meta_path :
        Path to ``models.json`` listing GCM/ensemble info per chain.
    gwl_yaml_path :
        Path to the warming-levels YAML (Mathias Hauser format).
    climate_scenario :
        RCP scenario string used as the YAML lookup key (default ``"rcp85"``).

    Returns
    -------
    period_defs : dict
        ``{"+X.X°C": {chain_id: (start_str, end_str)}}`` — per-model windows.
    period_label : dict
        ``{"+X.X°C": "N_+X.X°C"}`` — sortable internal label used by
        ``compile_and_compare_periods``.
    period_colors : dict
        ``{"+X.X°C": hex_color}`` — colour per GWL for plotting.
    """
    import json as _json
    from pathlib import Path as _Path
    from cordex2energy.results.analysis import load_warming_level_periods

    chain_meta_path = _Path(chain_meta_path)
    gwl_yaml_path = _Path(gwl_yaml_path)

    _models_config = _json.load(open(chain_meta_path))
    _chain_meta = {m["id"]: m for m in _models_config["model_chains"]}

    _gcm_ens_map: dict[str, str] = {}
    for chain_id in models:
        meta = _chain_meta.get(chain_id, {})
        cmip5 = _CORDEX_GCM_TO_CMIP5.get(meta.get("gcm", ""))
        if cmip5:
            _gcm_ens_map[cmip5] = meta.get("ensemble", "r1i1p1")

    gwl_data = load_warming_level_periods(
        yml_path=gwl_yaml_path,
        warming_levels=warming_levels,
        gcms=_gcm_ens_map,
    )

    period_defs: dict = {}
    period_label: dict = {}
    period_colors: dict = {}

    for i, wl in enumerate(warming_levels):
        label = f"+{wl:.1f}°C"
        windows: dict[str, tuple[str, str]] = {}
        for chain_id in models:
            meta = _chain_meta.get(chain_id, {})
            cmip5 = _CORDEX_GCM_TO_CMIP5.get(meta.get("gcm", ""))
            if cmip5 is None:
                continue
            r = (
                gwl_data
                .get((cmip5, meta.get("ensemble", "r1i1p1")), {})
                .get(climate_scenario, {})
                .get(wl)
            )
            if r:
                windows[chain_id] = (str(r["start_year"]), str(r["end_year"]))
        period_defs[label] = windows
        period_label[label] = f"{i + 1}_{label}"
        period_colors[label] = _GWL_COLORS[i % len(_GWL_COLORS)]

    return period_defs, period_label, period_colors


# ---------------------------------------------------------------------------
# Seasonal EVT threshold calibration
# ---------------------------------------------------------------------------


def calibrate_seasonal_evt_thresholds(
    ds_rl_ref: xr.Dataset,
    zones,
    season_order: list[str],
    season_months: dict[str, list[int]],
    zone_dim: str = "zone",
) -> "tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]":
    """Calibrate per-season EVT thresholds for all zones from the reference period.

    Parameters
    ----------
    ds_rl_ref :
        Reference-period residual load dataset (must contain ``residual_load``).
    zones :
        Zone coordinate values.
    season_order :
        Season names in display order (e.g. ``["DJF", "MAM", "JJA", "SON"]``).
    season_months :
        Mapping ``{season: [month_ints]}``.
    zone_dim :
        Name of the zone dimension (default ``"zone"``).

    Returns
    -------
    thresholds :
        ``{zone: {season: float}}`` — EVT threshold in GW for each (zone, season).
    percentiles :
        ``{zone: {season: float}}`` — empirical percentile (0–100) of each
        threshold within all 3-hourly RL values of that season's reference slice.
    """
    times_ref = pd.DatetimeIndex(ds_rl_ref["time"].values)
    thresholds: dict[str, dict[str, float]] = {}
    percentiles: dict[str, dict[str, float]] = {}
    for zone in zones:
        zk = str(zone)
        thresholds[zk] = {}
        percentiles[zk] = {}
        rl_ref_z = ds_rl_ref["residual_load"].sel({zone_dim: zone}).values
        for season in season_order:
            mask = np.isin(times_ref.month, season_months[season])
            season_rl = rl_ref_z[mask]
            thr = find_evt_pot_threshold(season_rl)
            thresholds[zk][season] = thr
            percentiles[zk][season] = float(np.mean(season_rl <= thr) * 100)
    return thresholds, percentiles


# ---------------------------------------------------------------------------
# Seasonal EVT event metrics
# ---------------------------------------------------------------------------


def compute_seasonal_evt_metrics(
    ds_rl_p: xr.Dataset,
    ds_met_p: xr.Dataset,
    season_order: list[str],
    season_months: dict[str, list[int]],
    season_thresholds: dict[str, dict[str, float]],
    model: str,
    period_name: str,
    zones,
    zone_dim: str = "zone",
) -> list[dict]:
    """Compute seasonal-conditional EVT event metrics for one model-period slice.

    For each (zone, season) the function:
    - Zeroes out non-season timesteps so events cannot span season boundaries.
    - Detects connected exceedance blocks above the season-specific threshold.
    - Accumulates durations, energy severities, PCIT, and climate driver shares.

    Parameters
    ----------
    ds_rl_p :
        Post-battery residual load for the target period (``residual_load`` variable).
    ds_met_p :
        Output of ``compute_continuous_system_metrics`` for the same period.
    season_order :
        Season names in display order.
    season_months :
        ``{season: [month_ints]}``.
    season_thresholds :
        ``{zone: {season: float}}`` from ``calibrate_seasonal_evt_thresholds``.
    model :
        Model chain ID string (carried into output rows).
    period_name :
        Period label string (carried into output rows).
    zones :
        Zone coordinate values.
    zone_dim :
        Name of the zone dimension (default ``"zone"``).

    Returns
    -------
    list[dict]
        One row-dict per (zone, season) ready for ``pd.DataFrame``.
    """
    from scipy.ndimage import label as _label

    times_p = pd.DatetimeIndex(ds_rl_p["time"].values)
    pcit_series = ds_met_p["pcit"].values
    has_res = "anomaly_reservoir_dispatch_gwh" in ds_met_p
    has_bat = "anomaly_battery_dispatch_gwh" in ds_met_p
    rows: list[dict] = []

    for season in season_order:
        s_mask = np.isin(times_p.month, season_months[season])
        s_years = np.unique(ds_rl_p["time"].dt.year.values[s_mask])

        for zone in zones:
            zone_key = str(zone)
            rl = ds_rl_p["residual_load"].sel({zone_dim: zone}).values
            thr = season_thresholds[zone_key][season]

            rl_s = rl.copy()
            rl_s[~s_mask] = thr - 1.0  # block cross-season events

            labeled, n_ev = _label(rl_s > thr)

            if n_ev == 0:
                rows.append({
                    "model": model, "zone": zone_key, "season": season, "Period": period_name,
                    "Frequency": 0,
                    "Avg_Duration_Hours": np.nan, "Max_Duration_Hours": np.nan,
                    "Avg_Energy_Severity_GWh": np.nan, "Max_Energy_Severity_GWh": np.nan,
                    "Total_Annual_Energy_Severity_GWh": 0.0,
                    "P90_Annual_Energy_Severity_GWh": 0.0,
                    "Std_Annual_Energy_Severity_GWh": 0.0,
                    "CV_Annual_Energy_Severity": np.nan,
                    "Avg_PCIT_During_Events_GW": np.nan,
                    "Max_PCIT_During_Events_GW": np.nan,
                    "Demand_Driven_Share_%": np.nan, "PV_Driven_Share_%": np.nan,
                    "Wind_Onshore_Driven_Share_%": np.nan, "Wind_Offshore_Driven_Share_%": np.nan,
                    "Hydro_RoR_Driven_Share_%": np.nan, "Reservoir_Hydro_Driven_Share_%": np.nan,
                    "Battery_Driven_Share_%": np.nan,
                    "Avg_Demand_Anomaly_GWh": np.nan, "Avg_PV_Anomaly_GWh": np.nan,
                    "Avg_Wind_Onshore_Anomaly_GWh": np.nan, "Avg_Wind_Offshore_Anomaly_GWh": np.nan,
                    "Avg_Hydro_RoR_Anomaly_GWh": np.nan, "Avg_Reservoir_Anomaly_GWh": np.nan,
                    "Avg_Battery_Anomaly_GWh": np.nan,
                })
                continue

            d_anom    = ds_met_p["anomaly_demand"].sel({zone_dim: zone}).values
            pv_anom   = ds_met_p["anomaly_gen_pv"].sel({zone_dim: zone}).values
            won_anom  = ds_met_p["anomaly_gen_wind_on"].sel({zone_dim: zone}).values
            woff_anom = ds_met_p["anomaly_gen_wind_off"].sel({zone_dim: zone}).values
            hror_anom = ds_met_p["anomaly_gen_hydro_ror"].sel({zone_dim: zone}).values
            res_anom  = (
                ds_met_p["anomaly_reservoir_dispatch_gwh"].sel({zone_dim: zone}).values
                if has_res else np.zeros(len(rl))
            )
            bat_anom  = (
                ds_met_p["anomaly_battery_dispatch_gwh"].sel({zone_dim: zone}).values
                if has_bat else np.zeros(len(rl))
            )

            durations: list[int] = []
            severities: list[float] = []
            event_pcit: list[float] = []
            annual_sev: dict[int, float] = {int(yr): 0.0 for yr in s_years}
            tot_d = tot_pv = tot_won = tot_woff = tot_hror = tot_res = tot_bat = 0.0

            for i in range(1, n_ev + 1):
                ev_idx = np.where(labeled == i)[0]
                sev = float((rl[ev_idx] - thr).sum()) * 3  # GWh (3h steps)
                ev_year = int(times_p[ev_idx[0]].year)
                durations.append(int(ev_idx.size) * 3)
                severities.append(sev)
                annual_sev[ev_year] = annual_sev.get(ev_year, 0.0) + sev
                event_pcit.extend(pcit_series[ev_idx].tolist())
                tot_d    += float(d_anom[ev_idx].sum())
                tot_pv   += float(pv_anom[ev_idx].sum())
                tot_won  += float(won_anom[ev_idx].sum())
                tot_woff += float(woff_anom[ev_idx].sum())
                tot_hror += float(hror_anom[ev_idx].sum())
                tot_res  += float(res_anom[ev_idx].sum())
                tot_bat  += float(bat_anom[ev_idx].sum())

            ann_arr = np.array(list(annual_sev.values()))
            denom = tot_d + tot_pv + tot_won + tot_woff + tot_hror + tot_res + tot_bat
            if denom > 0:
                d_sh    = tot_d    / denom * 100
                pv_sh   = tot_pv   / denom * 100
                won_sh  = tot_won  / denom * 100
                woff_sh = tot_woff / denom * 100
                hror_sh = tot_hror / denom * 100
                res_sh  = tot_res  / denom * 100
                bat_sh  = tot_bat  / denom * 100
            else:
                d_sh = pv_sh = won_sh = woff_sh = hror_sh = res_sh = bat_sh = 100.0 / 7

            rows.append({
                "model": model, "zone": zone_key, "season": season, "Period": period_name,
                "Frequency": n_ev,
                "Avg_Duration_Hours": float(np.mean(durations)),
                "Max_Duration_Hours": float(np.max(durations)),
                "Avg_Energy_Severity_GWh": float(np.mean(severities)),
                "Max_Energy_Severity_GWh": float(np.max(severities)),
                "Total_Annual_Energy_Severity_GWh": float(ann_arr.mean()),
                "P90_Annual_Energy_Severity_GWh": float(np.percentile(ann_arr, 90)),
                "Std_Annual_Energy_Severity_GWh": float(ann_arr.std()),
                "CV_Annual_Energy_Severity": (
                    float(ann_arr.std() / ann_arr.mean()) if ann_arr.mean() > 0 else np.nan
                ),
                "Avg_PCIT_During_Events_GW": float(np.mean(event_pcit)),
                "Max_PCIT_During_Events_GW": float(np.max(event_pcit)),
                "Demand_Driven_Share_%": d_sh, "PV_Driven_Share_%": pv_sh,
                "Wind_Onshore_Driven_Share_%": won_sh, "Wind_Offshore_Driven_Share_%": woff_sh,
                "Hydro_RoR_Driven_Share_%": hror_sh, "Reservoir_Hydro_Driven_Share_%": res_sh,
                "Battery_Driven_Share_%": bat_sh,
                "Avg_Demand_Anomaly_GWh": round( tot_d   * 3 / n_ev, 2),
                "Avg_PV_Anomaly_GWh":     round(-tot_pv  * 3 / n_ev, 2),
                "Avg_Wind_Onshore_Anomaly_GWh":  round(-tot_won  * 3 / n_ev, 2),
                "Avg_Wind_Offshore_Anomaly_GWh": round(-tot_woff * 3 / n_ev, 2),
                "Avg_Hydro_RoR_Anomaly_GWh":     round(-tot_hror * 3 / n_ev, 2),
                "Avg_Reservoir_Anomaly_GWh":     round(-tot_res  * 3 / n_ev, 2),
                "Avg_Battery_Anomaly_GWh":       round(-tot_bat  * 3 / n_ev, 2),
            })

    return rows


# ---------------------------------------------------------------------------
# Seasonal exceedance fractions
# ---------------------------------------------------------------------------


def compute_seasonal_exceedance_fractions(
    cache_rl: dict,
    evt_thresholds: dict,
    period_defs: dict,
    models,
    season_map: dict[int, str],
    season_order: list[str],
    zone_dim: str = "zone",
) -> pd.DataFrame:
    """Compute the seasonal distribution of annual EVT exceedances.

    For each (model, period, zone), counts what fraction of timesteps that
    exceed the annual common threshold fall in each meteorological season.

    Parameters
    ----------
    cache_rl :
        ``{model: ds_rl}`` — post-battery residual load datasets.
    evt_thresholds :
        ``{model: da_thresholds}`` — annual EVT threshold DataArrays.
    period_defs :
        Fixed or per-model period definitions (same format as used by
        ``resolve_model_periods``).
    models :
        Iterable of model chain IDs.
    season_map :
        ``{month_int: season_str}`` mapping.
    season_order :
        Season names in display order.
    zone_dim :
        Name of the zone dimension (default ``"zone"``).

    Returns
    -------
    pd.DataFrame
        Columns: model, zone, period, DJF, MAM, JJA, SON (fractions in %).
    """
    from cordex2energy.results.plotting_extremes import resolve_model_periods

    records: list[dict] = []
    for model in models:
        ds_rl_m = cache_rl[model]
        da_evt_m = evt_thresholds[model]
        resolved = {
            k: v for k, v in resolve_model_periods(period_defs, model).items()
            if v is not None
        }
        for period_name, (t0, t1) in resolved.items():
            ds_rl_p = ds_rl_m.sel(time=slice(t0, t1))
            times_p = pd.DatetimeIndex(ds_rl_p["time"].values)
            zones = ds_rl_p[zone_dim].values
            for zone in zones:
                rl = ds_rl_p["residual_load"].sel({zone_dim: zone}).values
                thr = float(da_evt_m.sel({zone_dim: zone}).values)
                exc_months = times_p[rl > thr].month
                seasons = np.array([season_map[int(m)] for m in exc_months])
                total = len(seasons)
                row: dict = {"model": model, "zone": str(zone), "period": period_name}
                for s in season_order:
                    row[s] = float((seasons == s).sum()) / total * 100 if total > 0 else 0.0
                records.append(row)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Zone-pair co-occurrence
# ---------------------------------------------------------------------------


def _cooccurrence_on_array(
    exc_masks: dict,
    zones,
    dilation_struct,
) -> dict:
    """Compute dilated masks and return {zone: dilated_bool_array}."""
    from scipy.ndimage import binary_dilation

    return {
        str(z): binary_dilation(exc_masks[str(z)], structure=dilation_struct)
        for z in zones
    }


def compute_zone_pair_cooccurrence(
    cache_rl: dict,
    evt_thresholds: dict,
    period_defs: dict,
    models,
    zones,
    window_steps: int = 24,
    season_map: "dict[int, str] | None" = None,
    zone_dim: str = "zone",
) -> pd.DataFrame:
    """Compute zone-pair co-occurrence of EVT exceedances with a dilation window.

    For each (model, period, zone_a, zone_b): the fraction of Zone A exceedance
    timesteps that co-occur with a Zone B exceedance within ±``window_steps``
    time steps (default ±72 h at 3 h resolution).

    Parameters
    ----------
    cache_rl :
        ``{model: ds_rl}`` — post-battery residual load datasets.
    evt_thresholds :
        Two formats are accepted:

        * **Annual** — ``{model: xr.DataArray}`` with a zone dimension, as
          returned by ``compute_evt_thresholds_for_all_zones``.
        * **Seasonal** — ``{model: {zone: {season: float}}}``, as returned by
          ``calibrate_seasonal_evt_thresholds``.  The seasonal format is detected
          automatically when ``season_map`` is provided and the per-model value
          is a plain ``dict``.  Falls back to the annual threshold for any
          (zone, season) key not found.

    period_defs :
        Fixed or per-model period definitions.
    models :
        Iterable of model chain IDs.
    zones :
        Zone coordinate values.
    window_steps :
        Half-width of the dilation kernel in time steps (default 24 = ±72 h).
    season_map :
        Optional ``{month_int: season_name}`` mapping (e.g. ``{12: "DJF", 1: "DJF",
        ...}``).  When provided the function computes co-occurrence separately for
        each season and the returned DataFrame contains a ``season`` column.
        When ``None`` (default) co-occurrence is computed over the full period.
    zone_dim :
        Name of the zone dimension (default ``"zone"``).

    Returns
    -------
    pd.DataFrame
        Columns: model, period, zone_a, zone_b, co_occurrence_pct[, season].
    """
    from cordex2energy.results.plotting_extremes import resolve_model_periods

    dilation_struct = np.ones(2 * window_steps + 1, dtype=bool)
    records: list[dict] = []

    # Build ordered season list preserving insertion order of season_map values
    if season_map is not None:
        seen: dict = {}
        for s in season_map.values():
            seen[s] = None
        season_names = list(seen.keys())
    else:
        season_names = [None]  # sentinel: single "full period" pass

    for model in models:
        ds_rl_m = cache_rl[model]
        thr_m = evt_thresholds[model]
        seasonal_thr = isinstance(thr_m, dict)  # True → {zone: {season: float}}
        resolved = {
            k: v for k, v in resolve_model_periods(period_defs, model).items()
            if v is not None
        }
        for period_name, (t0, t1) in resolved.items():
            ds_rl_p = ds_rl_m.sel(time=slice(t0, t1))
            time_months = ds_rl_p["time"].dt.month.values

            for season in season_names:
                if season is None:
                    s_mask = np.ones(len(time_months), dtype=bool)
                else:
                    target_months = {m for m, s in season_map.items() if s == season}
                    s_mask = np.isin(time_months, list(target_months))

                exc_masks: dict = {}
                for zone in zones:
                    rl_full = ds_rl_p["residual_load"].sel({zone_dim: zone}).values
                    if seasonal_thr and season is not None:
                        thr = float(thr_m[str(zone)][season])
                    else:
                        thr = float(thr_m.sel({zone_dim: zone}).values)
                    # Exceedance only at season timesteps; treated as False elsewhere
                    exc = (rl_full > thr) & s_mask
                    exc_masks[str(zone)] = exc

                dilated_masks = _cooccurrence_on_array(exc_masks, zones, dilation_struct)

                for za in zones:
                    za_key = str(za)
                    # Count only exceedances within the season as the denominator
                    n_a = int((exc_masks[za_key] & s_mask).sum())
                    for zb in zones:
                        zb_key = str(zb)
                        frac = (
                            float((exc_masks[za_key] & dilated_masks[zb_key]).sum()) / n_a * 100
                            if n_a > 0 else np.nan
                        )
                        row = {
                            "model": model, "period": period_name,
                            "zone_a": za_key, "zone_b": zb_key,
                            "co_occurrence_pct": frac,
                        }
                        if season is not None:
                            row["season"] = season
                        records.append(row)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Duration-severity event extraction
# ---------------------------------------------------------------------------


def extract_evt_duration_severity_records(
    cache_rl: dict,
    evt_thresholds: dict,
    period_defs: dict,
    models,
    zones,
    zone_dim: str = "zone",
) -> pd.DataFrame:
    """Extract individual EVT event records (duration, severity) for power-law analysis.

    Parameters
    ----------
    cache_rl :
        ``{model: ds_rl}`` — post-battery residual load datasets.
    evt_thresholds :
        ``{model: da_thresholds}`` — annual EVT threshold DataArrays.
    period_defs :
        Fixed or per-model period definitions.
    models :
        Iterable of model chain IDs.
    zones :
        Zone coordinate values.
    zone_dim :
        Name of the zone dimension (default ``"zone"``).

    Returns
    -------
    pd.DataFrame
        Columns: model, period, zone, duration_h, severity_gwh.
    """
    from scipy.ndimage import label as _label
    from cordex2energy.results.plotting_extremes import resolve_model_periods

    records: list[dict] = []
    for model in models:
        ds_rl_m = cache_rl[model]
        da_evt_m = evt_thresholds[model]
        resolved = {
            k: v for k, v in resolve_model_periods(period_defs, model).items()
            if v is not None
        }
        for period_name, (t0, t1) in resolved.items():
            ds_rl_p = ds_rl_m.sel(time=slice(t0, t1))
            for zone in zones:
                rl = ds_rl_p["residual_load"].sel({zone_dim: zone}).values
                thr = float(da_evt_m.sel({zone_dim: zone}).values)
                labeled, n_ev = _label(rl > thr)
                for i in range(1, n_ev + 1):
                    ev_mask = labeled == i
                    records.append({
                        "model": model, "period": period_name, "zone": str(zone),
                        "duration_h": int(ev_mask.sum()) * 3,
                        "severity_gwh": float((rl[ev_mask] - thr).sum()) * 3,
                    })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Unserved energy fraction
# ---------------------------------------------------------------------------


def compute_unserved_energy_fraction(
    cache_rl: dict,
    cache_comp: dict,
    period_defs: dict,
    models,
    zones,
    season_map: "dict[int, str] | None" = None,
    zone_dim: str = "zone",
    demand_var: str = "total_demand",
) -> pd.DataFrame:
    """Annual share of unserved energy (USE) per model, period, zone, and year.

    USE is defined as the fraction of total electricity demand not covered by
    generation in the post-battery residual-load series::

        USE_year[, season] = Σ_t max(RL_t, 0) / Σ_t demand_t   [dimensionless, 0–1]

    where the sums run over all 3-hourly steps in a calendar year (optionally
    restricted to one season).  Both numerator and denominator are in GWh
    (3-hourly sums), so no unit conversion is needed.

    Parameters
    ----------
    cache_rl :
        ``{model: ds_rl}`` — post-battery residual load datasets with variable
        ``"residual_load"`` (GWh, 3-hourly).
    cache_comp :
        ``{model: ds_comp}`` — generation/demand component datasets with variable
        ``demand_var`` (GWh, 3-hourly).
    period_defs :
        Fixed or per-model GWL/calendar period definitions.
    models :
        Iterable of model chain IDs.
    zones :
        Zone coordinate values.
    season_map :
        Optional ``{month_int: season_name}`` mapping.  When provided, USE is
        computed separately for each season (only timesteps in that season
        contribute to numerator and denominator).  The returned DataFrame gains
        a ``season`` column.  When ``None`` (default) USE is computed over the
        full year.
    zone_dim :
        Name of the zone dimension (default ``"zone"``).
    demand_var :
        Name of the demand variable in *cache_comp* (default ``"total_demand"``).

    Returns
    -------
    pd.DataFrame
        Raw annual USE fractions.
        Columns: ``model``, ``period``, ``zone``, ``year``[, ``season``], ``use_fraction``.

        Aggregate as needed::

            # ensemble-median mean USE per period × zone
            df.groupby(["period","zone"])["use_fraction"].median()

            # worst year per model × period × zone
            df.groupby(["model","period","zone"])["use_fraction"].max()
    """
    from cordex2energy.results.plotting_extremes import resolve_model_periods

    # Build ordered season list, or a single None sentinel for full-year mode
    if season_map is not None:
        seen: dict = {}
        for s in season_map.values():
            seen[s] = None
        season_names: list = list(seen.keys())
    else:
        season_names = [None]

    records: list[dict] = []

    for model in models:
        ds_rl_m   = cache_rl[model]
        ds_comp_m = cache_comp[model]
        resolved  = {
            k: v for k, v in resolve_model_periods(period_defs, model).items()
            if v is not None
        }
        for period_name, (t0, t1) in resolved.items():
            ds_rl_p    = ds_rl_m.sel(time=slice(t0, t1))
            ds_comp_p  = ds_comp_m.sel(time=slice(t0, t1))
            time_months = ds_rl_p["time"].dt.month.values
            time_years  = ds_rl_p["time"].dt.year.values
            years       = np.unique(time_years)

            for zone in zones:
                rl     = ds_rl_p["residual_load"].sel({zone_dim: zone}).values
                demand = ds_comp_p[demand_var].sel({zone_dim: zone}).values

                for season in season_names:
                    if season is None:
                        s_mask = np.ones(len(time_months), dtype=bool)
                    else:
                        target_months = {m for m, s in season_map.items() if s == season}
                        s_mask = np.isin(time_months, list(target_months))

                    for year in years:
                        y_s_mask   = s_mask & (time_years == year)
                        if not y_s_mask.any():
                            continue
                        pos_rl_sum = float(np.maximum(rl[y_s_mask], 0.0).sum())
                        demand_sum = float(demand[y_s_mask].sum())
                        use = pos_rl_sum / demand_sum if demand_sum > 0 else np.nan
                        row = {
                            "model":        model,
                            "period":       period_name,
                            "zone":         str(zone),
                            "year":         int(year),
                            "use_fraction": use,
                        }
                        if season is not None:
                            row["season"] = season
                        records.append(row)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Flexibility attenuation of residual-load event severity
# ---------------------------------------------------------------------------


def compute_flexibility_attenuation_metrics(
    cache_rl: dict,
    cache_comp: dict,
    evt_thresholds: dict,
    period_defs: dict,
    models,
    zones,
    season_map: "dict[int, str] | None" = None,
    zone_dim: str = "zone",
) -> pd.DataFrame:
    """Quantify battery and reservoir flexibility deployed during EVT events.

    For each detected extreme event (connected block above the EVT threshold in
    the post-flexibility residual load) the function computes:

    * **deployed GWh** — total flexibility energy dispatched during the event.
    * **counterfactual severity** — severity without that flexibility source.
    * **attenuation %** — ``(counterfactual − actual) / counterfactual × 100``.

    Battery dispatch is read from ``cache_comp`` and can be split into prosumer
    and utility components when both ``prosumer_dispatch_gwh`` and
    ``utility_dispatch_gwh`` are present.  The aggregate ``battery_dispatch_gwh``
    (prosumer + utility) is used for the combined battery metrics and as the
    sole battery signal when the split variables are absent (old runs).

    Sign convention: all dispatch variables are positive when discharging /
    generating (i.e. reducing residual load).

    Parameters
    ----------
    cache_rl :
        ``{model: ds_rl}`` — post-dispatch residual load datasets.
    cache_comp :
        ``{model: ds_comp}`` — component datasets.  Battery variables
        (``battery_dispatch_gwh``, ``prosumer_dispatch_gwh``,
        ``utility_dispatch_gwh``) and ``reservoir_dispatch_gwh`` are read
        from here.  Missing variables produce NaN metrics.
    evt_thresholds :
        Two formats accepted (auto-detected):

        * **Annual** — ``{model: xr.DataArray}`` with a zone dimension.
        * **Seasonal** — ``{model: {zone: {season: float}}}`` as returned by
          ``calibrate_seasonal_evt_thresholds``.

    period_defs :
        Period definitions.
    models :
        Iterable of model chain IDs.
    zones :
        Zone coordinate values.
    season_map :
        Optional ``{month_int: season_name}`` mapping.  When provided the
        function computes metrics separately per season; non-season timesteps
        are masked out.  The returned DataFrame gains a ``season`` column.
    zone_dim :
        Name of the zone dimension (default ``"zone"``).

    Returns
    -------
    pd.DataFrame
        One row per (model, period, zone[, season]).  Columns:

        ``model``, ``period``, ``zone`` [, ``season``],
        ``n_events``,
        ``battery_deployed_gwh_per_event``,
        ``prosumer_deployed_gwh_per_event``,
        ``utility_deployed_gwh_per_event``,
        ``reservoir_deployed_gwh_per_event``,
        ``battery_attenuation_pct``,
        ``prosumer_attenuation_pct``,
        ``utility_attenuation_pct``,
        ``reservoir_attenuation_pct``,
        ``combined_attenuation_pct``.

        Prosumer / utility columns are NaN when the corresponding dispatch
        variables are absent from ``cache_comp``.
    """
    from scipy.ndimage import label as ndlabel
    from cordex2energy.results.plotting_extremes import resolve_model_periods

    # Build ordered season list, or single None sentinel for full-year mode
    if season_map is not None:
        seen: dict = {}
        for s in season_map.values():
            seen[s] = None
        season_names: list = list(seen.keys())
    else:
        season_names = [None]

    records: list[dict] = []

    for model in models:
        ds_rl_m   = cache_rl[model]
        ds_comp_m = cache_comp[model]
        thr_m     = evt_thresholds[model]
        seasonal_thr = isinstance(thr_m, dict)

        has_battery   = "battery_dispatch_gwh"   in ds_comp_m.data_vars
        has_prosumer  = "prosumer_dispatch_gwh"  in ds_comp_m.data_vars
        has_utility   = "utility_dispatch_gwh"   in ds_comp_m.data_vars
        has_reservoir = "reservoir_dispatch_gwh" in ds_comp_m.data_vars

        resolved = {
            k: v for k, v in resolve_model_periods(period_defs, model).items()
            if v is not None
        }
        for period_name, (t0, t1) in resolved.items():
            ds_rl_p     = ds_rl_m.sel(time=slice(t0, t1))
            ds_comp_p   = ds_comp_m.sel(time=slice(t0, t1))
            time_months = ds_rl_p["time"].dt.month.values

            for zone in zones:
                rl_full = ds_rl_p["residual_load"].sel({zone_dim: zone}).values

                def _get(var, has_flag):
                    return (
                        ds_comp_p[var].sel({zone_dim: zone}).values
                        if has_flag else np.zeros_like(rl_full)
                    )

                bat_disp_full  = _get("battery_dispatch_gwh",  has_battery)
                pro_disp_full  = _get("prosumer_dispatch_gwh", has_prosumer)
                util_disp_full = _get("utility_dispatch_gwh",  has_utility)
                res_disp_full  = _get("reservoir_dispatch_gwh", has_reservoir)

                for season in season_names:
                    if seasonal_thr and season is not None:
                        thr = float(thr_m[str(zone)][season])
                    else:
                        thr = float(thr_m.sel({zone_dim: zone}).values)

                    if season is None:
                        rl        = rl_full
                        bat_disp  = bat_disp_full
                        pro_disp  = pro_disp_full
                        util_disp = util_disp_full
                        res_disp  = res_disp_full
                    else:
                        target_months = {m for m, s in season_map.items() if s == season}
                        s_mask = np.isin(time_months, list(target_months))
                        # Block cross-season events: non-season steps cannot exceed thr
                        rl = rl_full.copy()
                        rl[~s_mask] = thr - 1.0
                        bat_disp  = bat_disp_full.copy()
                        bat_disp[~s_mask]  = 0.0
                        pro_disp  = pro_disp_full.copy()
                        pro_disp[~s_mask]  = 0.0
                        util_disp = util_disp_full.copy()
                        util_disp[~s_mask] = 0.0
                        res_disp  = res_disp_full.copy()
                        res_disp[~s_mask]  = 0.0

                    exc = rl > thr
                    labeled, n_ev = ndlabel(exc)

                    base_row = {"model": model, "period": period_name, "zone": str(zone)}
                    if season is not None:
                        base_row["season"] = season

                    if n_ev == 0:
                        records.append({
                            **base_row,
                            "n_events": 0,
                            "battery_deployed_gwh_per_event":  np.nan,
                            "prosumer_deployed_gwh_per_event": np.nan,
                            "utility_deployed_gwh_per_event":  np.nan,
                            "reservoir_deployed_gwh_per_event": np.nan,
                            "battery_attenuation_pct":  np.nan,
                            "prosumer_attenuation_pct": np.nan,
                            "utility_attenuation_pct":  np.nan,
                            "reservoir_attenuation_pct": np.nan,
                            "combined_attenuation_pct": np.nan,
                        })
                        continue

                    bat_dep, pro_dep, util_dep, res_dep = [], [], [], []
                    bat_att, pro_att, util_att, res_att, comb_att = [], [], [], [], []

                    for ev_id in range(1, n_ev + 1):
                        ev_mask = labeled == ev_id
                        sev_actual = float(np.maximum(rl[ev_mask] - thr, 0.0).sum()) * 3

                        def _sev(discharge):
                            # discharge must be the non-negative (discharge-only) portion of the
                            # dispatch array.  Using raw dispatch (which includes negative/charging
                            # values from surplus countries in the same zone) would give a
                            # counterfactual severity *below* actual and yield negative attenuation.
                            return float(np.maximum((rl[ev_mask] + discharge) - thr, 0.0).sum()) * 3

                        bat_dis  = np.maximum(bat_disp[ev_mask],  0.0)
                        pro_dis  = np.maximum(pro_disp[ev_mask],  0.0)
                        util_dis = np.maximum(util_disp[ev_mask], 0.0)
                        res_dis  = np.maximum(res_disp[ev_mask],  0.0)  # reservoir is always >= 0

                        sev_no_bat  = _sev(bat_dis)
                        sev_no_pro  = _sev(pro_dis)  if has_prosumer else np.nan
                        sev_no_util = _sev(util_dis) if has_utility  else np.nan
                        sev_no_res  = _sev(res_dis)
                        sev_no_both = _sev(bat_dis + res_dis)

                        def _atten(sev_no, flag):
                            if not flag:
                                return np.nan
                            return (sev_no - sev_actual) / sev_no * 100 if sev_no > 0 else 0.0

                        bat_dep.append(float(bat_dis.sum())  * 3)
                        pro_dep.append(float(pro_dis.sum())  * 3 if has_prosumer else np.nan)
                        util_dep.append(float(util_dis.sum()) * 3 if has_utility  else np.nan)
                        res_dep.append(float(res_dis.sum())  * 3)

                        bat_att.append(_atten(sev_no_bat,  has_battery))
                        pro_att.append(_atten(sev_no_pro,  has_prosumer))
                        util_att.append(_atten(sev_no_util, has_utility))
                        res_att.append(_atten(sev_no_res,  has_reservoir))
                        comb_att.append(
                            (sev_no_both - sev_actual) / sev_no_both * 100
                            if sev_no_both > 0 else 0.0
                        )

                    def _mean(lst):
                        arr = np.array(lst, dtype=float)
                        return float(np.nanmean(arr)) if arr.size else np.nan

                    records.append({
                        **base_row,
                        "n_events": n_ev,
                        "battery_deployed_gwh_per_event":  _mean(bat_dep),
                        "prosumer_deployed_gwh_per_event": _mean(pro_dep),
                        "utility_deployed_gwh_per_event":  _mean(util_dep),
                        "reservoir_deployed_gwh_per_event": _mean(res_dep),
                        "battery_attenuation_pct":  _mean(bat_att),
                        "prosumer_attenuation_pct": _mean(pro_att),
                        "utility_attenuation_pct":  _mean(util_att),
                        "reservoir_attenuation_pct": _mean(res_att),
                        "combined_attenuation_pct": _mean(comb_att),
                    })

    return pd.DataFrame(records)
