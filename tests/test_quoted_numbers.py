"""Statements the paper makes about the numbers. A failure here means the text needs revisiting."""

import numpy as np
import pytest

from conftest import requires_data

pytestmark = requires_data

#: CERRA p95 per zone (GWh per 3h step): LOW/2050 (the paper), HIGH/2050's own, and the
#: percentile at which the paper's threshold sits inside HIGH/2050's distribution (S27).
THRESHOLDS = {
    "Iberian_Peninsula": (71.8, 43.7, 99.8), "South_East": (142.8, 89.3, 100.0),
    "Central_Western": (445.0, 370.7, 98.4), "Central_Eastern": (194.0, 155.8, 99.3),
    "Nordic": (56.9, 25.5, 100.0), "British_Isles": (145.9, 116.4, 98.8),
}


def sign_agreement_count(series) -> int:
    return int((np.sign(series) == np.sign(series.median())).sum())


@pytest.mark.parametrize("zone", list(THRESHOLDS))
def test_thresholds(zone, run, high_run):
    paper, own, percentile = THRESHOLDS[zone]
    assert abs(float(run.threshold.sel(zone=zone)) - paper) <= 0.5
    assert abs(float(high_run.threshold.sel(zone=zone)) - own) <= 0.5
    values = high_run.cerra_deficit.sel(zone=zone).values
    positive = values[np.isfinite(values)]
    assert abs(100 * float((positive <= paper).mean()) - percentile) <= 0.3


def test_paper_threshold_is_stricter_than_high_everywhere(run, high_run):
    """The LOW/2050 threshold is above HIGH/2050's own p95 in every zone, at its p98 or beyond."""
    for zone in run.zones:
        paper = float(run.threshold.sel(zone=zone))
        assert paper > float(high_run.threshold.sel(zone=zone))
        values = high_run.cerra_deficit.sel(zone=zone).values
        positive = values[np.isfinite(values)]
        assert 100 * float((positive <= paper).mean()) >= 98.0
    nordic = high_run.cerra_deficit.sel(zone="Nordic").values
    exceed = 100 * float((nordic[np.isfinite(nordic)] > float(run.threshold.sel(zone="Nordic"))).sum())
    assert exceed / len(nordic) <= 0.01


def test_figure_3_no_country_changes_sign(country_output, context):
    """No country's residual load crosses between surplus and deficit at any warming level."""
    from cordex2energy_analysis.fields import country_field
    def residual_load(period):
        return country_field(country_output, context.period_windows,
                             lambda data: data["residual_load"], period)
    reference_sign = np.sign(residual_load(context.reference).median("model").values)
    for period in context.period_names[1:]:
        sign = np.sign(residual_load(period).median("model").values)
        both = np.isfinite(reference_sign) & np.isfinite(sign)
        assert (reference_sign[both] == sign[both]).all(), period


def test_figure_4_claims(europe_change):
    annual = {k: europe_change["Annual", k].twh.median() for k in ("Total demand", "Total supply")}
    assert abs(annual["Total supply"]) > abs(annual["Total demand"])      # annually supply-led
    summer_residual = europe_change["JJA", "Residual load"].twh
    assert summer_residual.median() > 0 and sign_agreement_count(summer_residual) >= 8
    assert (europe_change["JJA", "Total demand"].twh.median()
            > abs(europe_change["JJA", "Total supply"].twh.median()))     # summer demand-led


def test_figure_5_rise_is_sub_daily_and_energy_multi_day(tables, context):
    from cordex2energy_analysis.tables import DURATION_BIN_LABELS, duration_spectrum
    counts = duration_spectrum(tables["storage_events"], "counts").groupby(["period", "bin"])["value"].median()
    severity = duration_spectrum(tables["storage_events"], "severity").groupby(["period", "bin"])["value"].median()
    added = {name: sum(counts[(context.warmest, b)] - counts[(context.reference, b)] for b in bins)
             for name, bins in (("sub_daily", DURATION_BIN_LABELS[:3]),
                                ("multi_day", DURATION_BIN_LABELS[3:]))}
    assert added["sub_daily"] > 10 * abs(added["multi_day"])
    peak = max(DURATION_BIN_LABELS, key=lambda b: severity[(context.warmest, b)])
    assert peak in DURATION_BIN_LABELS[3:]


def test_figure_6_deficit_window_and_drivers(run, tables, context):
    from cordex2energy_analysis.config import MAIN_SEASONS, SOUTHERN_ZONES
    from cordex2energy_analysis.extremes_analysis import (DEFICIT_WINDOW_HOURS,
                                                          attribution_change_by_model)
    from cordex2energy_analysis.runs import exceedance_by_hour, pooled_exceedance_by_hour
    southern = pooled_exceedance_by_hour(exceedance_by_hour(run, context), SOUTHERN_ZONES)
    warm = southern[(southern.period == context.warmest) & southern.season.isin(MAIN_SEASONS)]
    share = warm.groupby("hour")["hours_yr"].sum() / warm["hours_yr"].sum()
    assert float(share[share.index.isin(DEFICIT_WINDOW_HOURS)].sum()) >= 0.90
    expected_sign = {("Iberian_Peninsula", "DJF"): +1, ("South_East", "DJF"): -1,
                     ("Iberian_Peninsula", "JJA"): +1, ("South_East", "JJA"): +1}
    for (zone, season), sign in expected_sign.items():
        median = attribution_change_by_model(tables["evening_peak_components"], zone, season,
                                             context.reference, context.warmest,
                                             context.style.drivers).median()
        assert np.sign(float(median["net"])) == sign, (zone, season)
        if season == "JJA":
            assert max(context.style.drivers, key=lambda d: abs(median[d])) == "Demand", zone


def test_figure_7_storage_balances_less_on_long_events(tables, context):
    attenuation = tables["storage_attenuation"]
    warm = attenuation[attenuation.period == context.warmest]
    for zone in context.style.zones:
        combined = []
        for duration_class in ("sub_daily", "multi_day"):
            rows = warm[(warm.duration_class == duration_class) & (warm.zone == zone)]
            combined.append(rows["reservoir_attenuation_share_pct"].median()
                            + rows["battery_attenuation_share_pct"].median())
        assert combined[1] < combined[0], zone
    british_isles = warm[warm.zone == "British_Isles"]
    assert np.isclose(british_isles["reservoir_attenuation_share_pct"].median(), 0.0)


def test_figure_8_high_damps_every_zone(run, high_on_paper_threshold, context):
    from cordex2energy_analysis.runs import positive_residual_load_share
    from cordex2energy_analysis.tables import paired_increment
    low = positive_residual_load_share(run, context, "LOW")
    high = positive_residual_load_share(high_on_paper_threshold, context, "HIGH")
    for zone in run.zones:
        low_increment = paired_increment(low, zone, "share_of_demand_pct", context.reference,
                                         context.warmest)[0]
        high_increment = paired_increment(high, zone, "share_of_demand_pct", context.reference,
                                          context.warmest)[0]
        assert 0.0 <= high_increment / low_increment < 1.0, zone


def test_s29_southern_zones_carry_the_sub_daily_rise(tables, context):
    from cordex2energy_analysis.tables import DURATION_BIN_LABELS, duration_spectrum
    spectrum = duration_spectrum(tables["storage_events"], "counts", by=("model", "period", "zone"))
    sub_daily = spectrum[spectrum.bin.isin(DURATION_BIN_LABELS[:3])]
    per_zone = sub_daily.groupby(["model", "period", "zone"])["value"].sum().unstack("period")
    change = (per_zone[context.warmest] - per_zone[context.reference]).groupby("zone").median()
    assert float(change[["Iberian_Peninsula", "South_East"]].sum()) > 0.8 * float(change.sum())


def test_s31_multi_day_total_is_a_cancellation(tables, context):
    events = tables["events_annual"]
    by_class = (events.groupby(["model", "period", "duration_class"])["freq_per_year"].sum()
                .groupby(["period", "duration_class"]).median())
    change = {c: by_class[(context.warmest, c)] - by_class[(context.reference, c)]
              for c in ("sub_daily", "multi_day")}
    assert change["sub_daily"] > 10 * abs(change["multi_day"])
    zonal = (events[events.duration_class == "multi_day"]
             .pivot_table(index=["zone", "model"], columns="period", values="freq_per_year"))
    paired = ((zonal[context.warmest] / zonal[context.reference] - 1) * 100).groupby("zone").median()
    assert (paired < 0).any() and (paired > 0).any()


def test_s33_high_survivors(run, tables, high_on_paper_threshold, context):
    from cordex2energy_analysis.runs import annual_event_counts, seasonal_event_counts

    def europe_sub_daily(table):
        return float(table[table.duration_class == "sub_daily"].groupby(["model", "period"])
                     ["freq_per_year"].sum().groupby("period").median().get(context.warmest, 0.0))
    high_annual = annual_event_counts(high_on_paper_threshold, context)
    assert europe_sub_daily(high_annual) < 0.2 * europe_sub_daily(tables["events_annual"])
    high_seasonal = seasonal_event_counts(high_on_paper_threshold, context)
    summer = high_seasonal[(high_seasonal.duration_class == "sub_daily")
                           & (high_seasonal.season == "JJA")]
    assert summer.groupby(["zone", "period"])["freq_per_year"].median().max() <= 1e-9


#: Panels whose printed paired change disagrees in sign with the medians drawn beside it, examined
#: and accepted. None since the CERRA PV time-alignment fix (code review R1).
KNOWN_ANNOTATION_CONFLICTS = set()


def test_annotation_signs_match_the_drawn_medians(tables, context):
    """Figure 6a and S31a print paired changes; flag any that contradict the drawn medians."""
    events = tables["events_annual"]
    conflicts = set()
    for duration_class in ("sub_daily", "multi_day"):
        data = events[events.duration_class == duration_class]
        for zone in context.style.zones:
            rows = data[data.zone == zone]
            reference = rows[rows.period == context.reference].set_index("model")["freq_per_year"]
            warm = rows[rows.period == context.warmest].set_index("model")["freq_per_year"]
            shared = reference.index.intersection(warm.index)
            if not len(shared) or not np.isfinite(reference.loc[shared]).any():
                continue
            paired = float(((warm.loc[shared] - reference.loc[shared]) / reference.loc[shared] * 100).median())
            low, high = float(reference.loc[shared].median()), float(warm.loc[shared].median())
            unpaired = (high - low) / low * 100 if abs(low) > 1e-9 else np.nan
            # A zero change on either side does not contradict the other.
            if (np.isfinite(unpaired) and np.sign(paired) * np.sign(unpaired) < 0
                    and max(abs(paired), abs(unpaired)) > 2.0):
                conflicts.add((duration_class, zone))
    assert conflicts == KNOWN_ANNOTATION_CONFLICTS


def test_s33_survivors_are_wind_droughts(high_run, high_on_paper_threshold, context):
    from cordex2energy_analysis.config import EVENT_DETECTION, SEASON_MAP
    from cordex2energy_analysis.extremes_analysis import event_anomaly_vs_hourly_climatology
    from cordex2energy_analysis.tables import event_anomaly
    table = event_anomaly_vs_hourly_climatology(
        high_run.residual_load, high_run.components, high_on_paper_threshold.threshold_by_model,
        context.period_windows, high_run.zones, SEASON_MAP, method=EVENT_DETECTION,
        max_duration_h=24.0)
    for zone in ("Central_Western", "Central_Eastern"):
        median = event_anomaly(table, "weather", zone, "DJF", context.warmest,
                               context.style.drivers).median()
        wind = median["Wind Onshore"] + median["Wind Offshore"]
        others = [median[d] for d in context.style.drivers if d not in ("Wind Onshore", "Wind Offshore")]
        assert wind > max(others), zone
    winter = table[(table.season == "DJF") & (table.period == context.warmest)]
    assert float(winter["evening_share"].mean()) < 0.8
