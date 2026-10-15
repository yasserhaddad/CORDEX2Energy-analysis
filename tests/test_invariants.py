"""Identities the figures rest on. Each holds exactly (to rounding) by construction."""

import numpy as np
import pytest

from conftest import requires_data

pytestmark = requires_data


def test_seasonal_counts_sum_to_annual(tables):
    """Each event is assigned whole to one season, so the four seasons add up to the year."""
    annual = tables["events_annual"].set_index(
        ["model", "period", "zone", "duration_class"])["freq_per_year"].sort_index()
    seasonal = tables["events_by_season"].groupby(
        ["model", "period", "zone", "duration_class"])["freq_per_year"].sum().sort_index()
    assert np.allclose(seasonal.values, annual.reindex(seasonal.index).values)


def test_event_hours_are_counts_times_duration(tables):
    events = tables["events_by_season"].dropna(subset=["avg_duration_h"])
    events = events[events["freq_per_year"] > 0]
    gap = (events["hours_yr"] - events["freq_per_year"] * events["avg_duration_h"]).abs().max()
    assert gap < 1e-9


def test_multi_day_attribution_covers_all_seasons(tables):
    assert {"MAM_Demand", "SON_RL"} <= set(tables["multi_day_events"].columns)


def test_storage_shares_sum_to_combined(tables):
    """The reservoir/battery Shapley split closes on the combined attenuation."""
    shares = tables["storage_attenuation"].dropna(subset=["combined_attenuation_pct"])
    residual = (shares["reservoir_attenuation_share_pct"] + shares["battery_attenuation_share_pct"]
                - shares["combined_attenuation_pct"]).abs()
    assert residual.max() < 1e-6


def test_balanced_plus_unbalanced_is_severity_without_storage(tables):
    events = tables["storage_events"]
    balanced = events.sev_no_storage - events.sev_actual
    assert np.allclose(balanced + events.sev_actual, events.sev_no_storage)


def test_residual_load_is_demand_minus_supply(europe_change):
    from cordex2energy_analysis.tables import EUROPE_COLUMNS
    for column in EUROPE_COLUMNS:
        gap = (europe_change[column, "Residual load"].twh
               - (europe_change[column, "Total demand"].twh
                  - europe_change[column, "Total supply"].twh)).abs()
        scale = max(1.0, float(europe_change[column, "Total demand"].twh.abs().max()))
        assert float(gap.max()) < 1e-6 * scale, column


def test_zones_close_on_europe(europe_rows, europe_change, context):
    """Per chain, the six zonal changes sum to the Europe-wide change in every row."""
    from cordex2energy_analysis.config import ZONE_ORDER
    from cordex2energy_analysis.plotting_publication import europe_change_table
    from cordex2energy_analysis.power_system import REALTIME_BALANCING_ZONES
    from cordex2energy_analysis.tables import EUROPE_COLUMNS
    zone_change = {zone: europe_change_table(europe_rows, EUROPE_COLUMNS, context.reference,
                                             context.warmest, members=REALTIME_BALANCING_ZONES[zone])
                   for zone in ZONE_ORDER}
    for (column, label), table in europe_change.items():
        total = sum(zone_change[zone][column, label].twh for zone in ZONE_ORDER)
        scale = max(1.0, float(table.twh.abs().max()))
        assert float((total - table.twh).abs().max()) < 1e-6 * scale, (column, label)


def test_seasonal_contributions_close_per_chain(tables, context):
    """A chain's four seasonal contributions sum to its own annual per-cent change (Figure 6b, S31b)."""
    from cordex2energy_analysis.config import DURATION_BINS, SEASON_ORDER
    from cordex2energy_analysis.plotting_publication import season_contribution_by_model
    events_annual, events_by_season = tables["events_annual"], tables["events_by_season"]
    worst = 0.0
    for duration_class in DURATION_BINS:
        for zone in context.style.zones:
            per_chain = [season_contribution_by_model(events_by_season, events_annual, zone, season,
                                                      duration_class, context.style)[0]
                         for season in SEASON_ORDER]
            annual = {period: events_annual[(events_annual.zone == zone)
                                            & (events_annual.duration_class == duration_class)
                                            & (events_annual.period == period)
                                            ].set_index("model")["freq_per_year"]
                      for period in (context.reference, context.warmest)}
            chains = (set.intersection(*(set(series.index) for series in per_chain))
                      if all(len(series) for series in per_chain) else set())
            for chain in chains:
                reference = annual[context.reference].get(chain, 0.0)
                if reference <= 0:
                    continue
                expected = 100 * (annual[context.warmest][chain] - reference) / reference
                worst = max(worst, abs(sum(series[chain] for series in per_chain) - expected))
    assert worst < 1e-6


def test_event_anomaly_split_is_additive(high_run, high_on_paper_threshold, context):
    """Weather plus diurnal-position anomaly is the total anomaly, per chain (S33c/d)."""
    from cordex2energy_analysis.config import EVENT_DETECTION, SEASON_MAP
    from cordex2energy_analysis.extremes_analysis import event_anomaly_vs_hourly_climatology
    table = event_anomaly_vs_hourly_climatology(
        high_run.residual_load, high_run.components, high_on_paper_threshold.threshold_by_model,
        context.period_windows, high_run.zones, SEASON_MAP, method=EVENT_DETECTION,
        max_duration_h=24.0)
    winter = table[table.season == "DJF"].dropna(subset=["net"])
    assert np.allclose(winter["net"] + winter["DIU_net"], winter["TOT_net"], atol=1e-6)


def test_map_change_bins_are_symmetric():
    from cordex2energy_analysis.figures_main import MAP_CHANGE_BINS
    for edges in MAP_CHANGE_BINS.values():
        assert np.allclose(sorted(-np.array(edges)), edges)


def test_print_geometry_keeps_every_country(context):
    assert set(context.countries_print["ISO2"]) == set(context.countries["ISO2"])
    assert context.countries_print.geometry.is_valid.all()


@pytest.mark.parametrize("seed", [0, 1])
def test_fast_sign_agreement_matches_reference(seed):
    """The vectorised sign agreement the maps use equals the reference implementation."""
    import xarray as xr
    from cordex2energy_analysis.analysis import compute_sign_agreement
    from cordex2energy_analysis.plotting_publication import sign_agreement
    random = np.random.RandomState(seed)
    probe = xr.DataArray(random.randn(9, 40, 50), dims=("model", "lat", "lon")).where(
        random.rand(9, 40, 50) > 0.15)
    assert (compute_sign_agreement(probe, model_dim="model").values
            == sign_agreement(probe).values).all()
