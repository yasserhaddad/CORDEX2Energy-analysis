"""Supplementary Note 1 (the conversion pipeline, Figs. S1-S16 and the Figure 1 insets).

The checks that used to sit in ``notebooks/si_note1_pipeline.ipynb``, and the numbers its captions
and prose quote. A failure means the data moved under the text or the figures.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from conftest import requires_data  # noqa: E402

pytestmark = requires_data

#: Years in one warming-level window.
WINDOW_YEARS = 20
#: A cell counts as carrying runoff when it exceeds this fraction of the field maximum (S8).
SIGNIFICANT_FRACTION = 0.01


# ---------------------------------------------------------------------------
# Fixtures: the Note 1 data, read once per module
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def climate_annual(context):
    from cordex2energy_analysis.pipeline_inputs import load_gwl_grids
    return load_gwl_grids(context.paths.gwl_grids, "climate", context.period_names)


@pytest.fixture(scope="module")
def climate_seasonal(context):
    from cordex2energy_analysis.pipeline_inputs import load_gwl_grids
    return load_gwl_grids(context.paths.gwl_grids, "climate", context.period_names, seasonal=True)


@pytest.fixture(scope="module")
def energy_grids(context):
    """Annual and seasonal gridded energy variables (capacity factors)."""
    from cordex2energy_analysis.pipeline_inputs import load_gwl_grids
    return (load_gwl_grids(context.paths.gwl_grids, "", context.period_names),
            load_gwl_grids(context.paths.gwl_grids, "", context.period_names, seasonal=True))


@pytest.fixture(scope="module")
def spain_cooling(context):
    from cordex2energy_analysis.pipeline_inputs import annual_country_cooling
    return annual_country_cooling(context, "Spain")


@pytest.fixture(scope="module")
def gmst(context):
    from cordex2energy_analysis.pipeline_inputs import load_cmip5_gmst
    return load_cmip5_gmst(context.paths.processed)


@pytest.fixture(scope="module")
def crossing_years(context):
    from cordex2energy_analysis.pipeline_inputs import cmip5_crossing_years
    return cmip5_crossing_years(context.paths.warming_level_years)


@pytest.fixture(scope="module")
def seasonal_change(context):
    from cordex2energy_analysis.pipeline_inputs import european_seasonal_change
    return european_seasonal_change(context)


@pytest.fixture(scope="module")
def calibration(context):
    from cordex2energy_analysis.demand_validation import load_calibration_data
    return load_calibration_data(context.paths.processed / "energy" / "inputs")


@pytest.fixture(scope="module")
def curvatures(calibration):
    from cordex2energy_analysis.demand_validation import production_curvatures
    return production_curvatures(calibration.coefficients)


@pytest.fixture(scope="module")
def heating_evaluation(calibration):
    """``(per-country table, pooled modelled/observed ratio of every country-year)``."""
    from cordex2energy_analysis.demand_validation import heating_performance
    return heating_performance(calibration)


@pytest.fixture(scope="module")
def heating_scores(heating_evaluation):
    return heating_evaluation[0]


@pytest.fixture(scope="module")
def heating_country_year_ratio(heating_evaluation):
    return heating_evaluation[1]


@pytest.fixture(scope="module")
def cooling_scores(calibration):
    from cordex2energy_analysis.demand_validation import cooling_performance
    return cooling_performance(calibration)


@pytest.fixture(scope="module")
def anchor_ratios(calibration):
    from cordex2energy_analysis.demand_validation import cooling_anchor_ratios
    return cooling_anchor_ratios(calibration.coefficients)


@pytest.fixture(scope="module")
def heating_held_out(calibration):
    from cordex2energy_analysis.demand_validation import heating_leave_one_out
    return heating_leave_one_out(calibration)


@pytest.fixture(scope="module")
def cooling_held_out(calibration, curvatures):
    from cordex2energy_analysis.demand_validation import cooling_leave_one_out
    return cooling_leave_one_out(calibration, curvatures)


@pytest.fixture(scope="module")
def candidates(calibration):
    from cordex2energy_analysis.demand_validation import curvature_candidates
    return curvature_candidates(calibration)


def held_out_error_share(table) -> float:
    """Summed absolute hold-out error over summed observed energy."""
    return float(table.loco_err_TWh.abs().sum() / table.obs_TWh.sum())


# ---------------------------------------------------------------------------
# S3-S8 inputs: the climate grids
# ---------------------------------------------------------------------------

def test_climate_variables_present_with_change_type(climate_annual, climate_seasonal, context):
    """Every climate variable and Qrouted is in the annual and seasonal files at every level,
    tagged as an absolute or a percent change."""
    from cordex2energy_analysis.pipeline_inputs import CLIMATE_VARIABLES
    for name, grids in (("annual", climate_annual), ("seasonal", climate_seasonal)):
        for variable in CLIMATE_VARIABLES + ["Qrouted"]:
            missing = [p for p in context.period_names if variable not in grids[p].data_vars]
            assert not missing, f"{variable} missing from the {name} files at {missing}"
            change_type = grids[context.period_names[0]][variable].attrs.get("change_type")
            assert change_type in ("absolute", "percent"), f"{variable}: {change_type!r}"


def test_temperature_change_is_absolute(climate_annual, context):
    """Temperature changes are drawn in kelvin: a percentage change in kelvin is meaningless."""
    assert climate_annual[context.reference]["tas"].attrs["change_type"] == "absolute"


# ---------------------------------------------------------------------------
# Figure 1 insets: Spanish cooling demand
# ---------------------------------------------------------------------------

def test_pipeline_inset_data_shape(spain_cooling, context):
    """Every chain contributes one full 20-year window at every warming level."""
    for period in context.period_names:
        counts = spain_cooling[spain_cooling.period == period].groupby("model")["year_in_window"].count()
        assert len(counts) == len(context.models), f"{period}: {len(counts)} chains"
        assert set(counts) == {WINDOW_YEARS}, f"{period}: window lengths {sorted(set(counts))}"


def test_pipeline_inset_unit_is_plausible(spain_cooling):
    """Spanish annual cooling lies within 1-100 TWh/yr, guarding the GW-hours to TWh conversion."""
    assert 1.0 < spain_cooling["twh"].median() < 100.0


def test_pipeline_inset_median_cooling_rises_with_warming(spain_cooling, context):
    """The ensemble-median Spanish cooling demand is higher at +3.0 than at +1.5, which is what
    the three insets exist to show."""
    medians = {period: float(spain_cooling[spain_cooling.period == period]
                             .groupby("year_in_window")["twh"].median().mean())
               for period in context.period_names}
    assert medians[context.warmest] > medians[context.reference], medians


def test_pipeline_insets_share_one_y_limit_and_draw_every_chain(spain_cooling, context):
    """The three inset figures share one y limit and each draws every chain plus the median,
    20 points per line."""
    from cordex2energy_analysis.figures_pipeline import pipeline_panels
    figures = pipeline_panels(spain_cooling, context.period_names)
    try:
        limits = {tuple(round(v, 6) for v in fig.axes[0].get_ylim()) for fig in figures.values()}
        assert len(limits) == 1, f"the inset figures do not share a y limit: {limits}"
        for suffix, fig in figures.items():
            lines = [line.get_xydata() for line in fig.axes[0].lines]
            assert len(lines) == len(context.models) + 1, f"{suffix}: {len(lines)} lines"
            assert {len(points) for points in lines} == {WINDOW_YEARS}, suffix
    finally:
        for fig in figures.values():
            plt.close(fig)


# ---------------------------------------------------------------------------
# S1: the chains and their warming-level windows
# ---------------------------------------------------------------------------

def test_every_chain_model_has_a_display_name(context):
    """Every chain's driving GCM and RCM has a display name for the S1 axis and legend."""
    from cordex2energy_analysis.pipeline_inputs import GCM_DISPLAY, RCM_DISPLAY, split_chain
    chains = [split_chain(model) for model in context.models]
    assert {gcm for _, gcm in chains} <= set(GCM_DISPLAY)
    assert {rcm for rcm, _ in chains} <= set(RCM_DISPLAY)


def test_chains_sharing_a_gcm_share_its_windows(context):
    """Chains driven by the same GCM share its warming-level windows, so S1b's one row per GCM
    is exact."""
    from cordex2energy_analysis.pipeline_inputs import driving_gcm_rows, split_chain
    for gcm, representative in driving_gcm_rows(context.models):
        for model in context.models:
            if split_chain(model)[1] != gcm:
                continue
            for period in context.period_names:
                windows = context.period_windows[period]
                assert windows.get(model) == windows.get(representative), \
                    f"{model} and {representative} share a GCM but not the {period} window"


def test_cmip5_gmst_table_has_the_ensemble_and_the_drivers(gmst):
    """The CMIP5 GMST table holds at least 35 models, including the five driving GCMs."""
    from cordex2energy_analysis.pipeline_inputs import DRIVING_KEYS
    assert gmst.sizes["model"] >= 35
    absent = [key for key in DRIVING_KEYS.values() if key not in set(gmst.model.values)]
    assert not absent, f"driving GCM(s) missing from the GMST table: {absent}"


def test_cmip5_gmst_anomaly_is_zero_over_its_baseline(gmst):
    """The anomaly averages to zero over 1861-1900; anything else is a calendar or units fault."""
    baseline = gmst.sel(year=slice(1861, 1900)).mean("year")
    assert np.allclose(baseline.values, 0.0, atol=1e-9), float(np.abs(baseline.values).max())


def test_crossing_years_are_the_paper_window_starts(crossing_years, context):
    """S1b's crossing years are the very windows the rest of the paper is computed over."""
    from cordex2energy_analysis.config import WARMING_LEVELS
    from cordex2energy_analysis.pipeline_inputs import DRIVING_MEMBERS, GCM_DISPLAY, split_chain
    for gcm, member in DRIVING_MEMBERS.items():
        chain = next(model for model in context.models
                     if GCM_DISPLAY[split_chain(model)[1]] == gcm)
        for level, period in zip(WARMING_LEVELS, context.period_names):
            match = crossing_years[(crossing_years.model == gcm)
                                   & (crossing_years.ensemble == member)
                                   & (crossing_years.warming_level == level)]
            assert len(match), f"{gcm} {member} +{level}: no crossing year"
            assert int(match["start_year"].iloc[0]) == int(context.period_windows[period][chain][0]), \
                f"{gcm} +{level}: S1b and the paper's windows disagree"


# ---------------------------------------------------------------------------
# S2: European seasonal precipitation and runoff
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("season", ["DJF", "JJA"])
def test_precipitation_and_runoff_medians_share_a_sign(seasonal_change, season):
    """In winter and summer the median precipitation and runoff changes move together."""
    block = seasonal_change[seasonal_change.season == season]
    assert np.sign(block.dP.median()) == np.sign(block.dR.median()), \
        f"{season}: {block.dP.median():+.3f} vs {block.dR.median():+.3f}"


def test_spring_snowmelt_signature(seasonal_change):
    """In spring most chains dry in precipitation while runoff rises: the snowmelt signature."""
    spring = seasonal_change[seasonal_change.season == "MAM"]
    assert (spring.dP < 0).sum() > len(spring) / 2
    assert (spring.dR > 0).sum() > len(spring) / 2


def test_spring_chain_counts_quoted(seasonal_change):
    """S2 prose: spring precipitation falls in seven of nine chains, runoff rises in eight."""
    spring = seasonal_change[seasonal_change.season == "MAM"]
    assert len(spring) == 9
    assert int((spring.dP < 0).sum()) == 7
    assert int((spring.dR > 0).sum()) == 8


# ---------------------------------------------------------------------------
# S8: runoff before and after routing
# ---------------------------------------------------------------------------

def routing_shares(climate_annual, context) -> dict:
    """Share of cells above 1 % of the field maximum, ensemble median at the reference level."""
    shares = {}
    for variable in ("mrro", "Qrouted"):
        field = climate_annual[context.reference][variable].median("model")
        shares[variable] = float((field > SIGNIFICANT_FRACTION * float(field.max())).mean())
    return shares


def test_routing_concentrates_runoff(climate_annual, context):
    """Qrouted covers less than a fifth of mrro's significant area: routing gathers a diffuse
    field into a sparse network."""
    shares = routing_shares(climate_annual, context)
    assert shares["Qrouted"] < 0.2 * shares["mrro"], shares


def test_routing_shares_quoted(climate_annual, context):
    """S8 caption: the diffuse field covers 49 % of the domain and the network 3 %."""
    shares = routing_shares(climate_annual, context)
    assert round(100 * shares["mrro"]) == 49, shares
    assert round(100 * shares["Qrouted"]) == 3, shares


def test_routed_discharge_range_quoted(climate_annual, context):
    """S8 prose: Qrouted runs from 1 m3/s at the median wet cell to 6838 at the Danube mouth."""
    field = climate_annual[context.reference]["Qrouted"].median("model").values
    finite = field[np.isfinite(field)]
    assert round(float(np.percentile(finite[finite > 0], 50))) == 1
    assert round(float(finite.max())) == 6838


# ---------------------------------------------------------------------------
# S9-S10: the demand model against its calibration target
# ---------------------------------------------------------------------------

def test_production_curvatures(curvatures):
    """The fitted cooling curvatures are those the paper reports."""
    assert abs(curvatures["res"] - 0.001564) < 1e-5, curvatures
    assert abs(curvatures["srv"] - 0.011493) < 1e-5, curvatures


def test_fitted_countries_have_iso2(heating_scores):
    """Every country the heating model is evaluated on has an ISO2 code for the S9 maps."""
    from cordex2energy_analysis.demand_validation import jrc_name_to_iso2
    assert not set(heating_scores.index) - set(jrc_name_to_iso2())


def test_no_jrc_countries_match_the_gap_fill_flag(calibration):
    """The countries drawn as gap-filled are those the coefficients flag as gap-filled."""
    from cordex2energy_analysis.demand_validation import NO_JRC_ISO2
    coefficients = calibration.coefficients
    assert set(NO_JRC_ISO2) == set(coefficients.index[coefficients["cool_gap_filled"] == 1])


def test_no_jrc_codes_are_in_the_geometry(context):
    """Every gap-filled country can be hatched on the map."""
    from cordex2energy_analysis.demand_validation import NO_JRC_ISO2
    assert not set(NO_JRC_ISO2.values()) - set(context.countries["ISO2"])


def test_heating_nrmse_median_below_twelve_percent(heating_scores):
    """The heating NRMSE stays where the caption describes it."""
    assert heating_scores.nrmse_pct.median() < 12


def test_heating_level_reproduced_by_construction(heating_scores):
    """Every country's modelled mean heating is within 10 % of observed: the slope is calibrated
    to reproduce the level."""
    assert (heating_scores.modelled / heating_scores.observed - 1).abs().max() < 0.10


def test_heating_underproduces_interannual_spread(heating_scores):
    """The model's 10-90 % interannual spread is narrower than observed in the median: the
    variance term of the decomposition."""
    spread = ((heating_scores.mod_hi - heating_scores.mod_lo)
              / (heating_scores.obs_hi - heating_scores.obs_lo))
    assert spread.median() < 1.0


def test_unanchored_cooling_underpredicts(anchor_ratios):
    """More than 70 % of anchor ratios exceed one: the un-anchored model under-predicts."""
    assert (anchor_ratios > 1).sum() > 0.7 * len(anchor_ratios)


@pytest.mark.parametrize("fuel", ["heating", "cooling"])
def test_mse_decomposition_closes(fuel, heating_scores, cooling_scores):
    """Bias, variance and correlation shares sum to one for every country."""
    table = heating_scores if fuel == "heating" else cooling_scores
    total = table.bias_share + table.var_share + table.corr_share
    assert (total - 1).abs().max() < 1e-9


def test_heating_country_years_within_ten_percent(heating_country_year_ratio):
    """More than 70 % of country-years land within +-10 % of observed heating."""
    assert np.mean(np.abs(heating_country_year_ratio - 1) <= 0.10) > 0.70


def test_heating_error_dominated_by_correlation(heating_scores):
    """Lack of correlation is the dominant heating error term (median share above 0.6)."""
    assert heating_scores.corr_share.median() > 0.6


def test_held_out_tables_ran_the_same_test(heating_held_out, cooling_held_out):
    """Both hold-out tables carry the same columns, so their numbers are comparable."""
    for table in (heating_held_out, cooling_held_out):
        assert set(table.columns) >= {"loco_err_TWh", "obs_TWh"}


def test_cooling_gap_fill_worse_than_heating(heating_held_out, cooling_held_out):
    """Cooling's gap-fill rule misses by more, relative to observed, than heating's."""
    assert held_out_error_share(cooling_held_out) > held_out_error_share(heating_held_out)


@pytest.mark.parametrize("sector", ["res", "srv"])
def test_curvature_candidates_span_an_order_of_magnitude(candidates, sector):
    """The candidate curvatures of each sector span more than a factor of ten."""
    block = candidates[candidates.sector == sector]
    assert block.theta.max() / block.theta.min() > 10


def test_heating_nrmse_quoted(heating_scores):
    """S9/S10 table: heating NRMSE median 7.9 % (range 3.5-43 %)."""
    assert round(heating_scores.nrmse_pct.median(), 1) == 7.9
    assert round(heating_scores.nrmse_pct.min(), 1) == 3.5
    assert round(heating_scores.nrmse_pct.max()) == 43


def test_cooling_nrmse_quoted(cooling_scores):
    """S9/S10 table: cooling NRMSE median 23.8 % (range 2.0-77.5 %)."""
    assert round(cooling_scores.nrmse_pct.median(), 1) == 23.8
    assert round(cooling_scores.nrmse_pct.min(), 1) == 2.0
    assert round(cooling_scores.nrmse_pct.max(), 1) == 77.5


def test_heating_country_year_share_quoted(heating_country_year_ratio):
    """S9/S10 prose: 75 % of 583 country-years within +-10 % of observed."""
    assert len(heating_country_year_ratio) == 583
    assert round(100 * np.mean(np.abs(heating_country_year_ratio - 1) <= 0.10)) == 75


@pytest.mark.parametrize("fuel, expected", [("heating", (0, 10, 90)), ("cooling", (42, 34, 24))])
def test_mse_split_quoted(fuel, expected, heating_scores, cooling_scores):
    """S9/S10 table: median bias / variance / correlation shares, in whole per cent."""
    table = heating_scores if fuel == "heating" else cooling_scores
    split = tuple(round(100 * table[column].median())
                  for column in ("bias_share", "var_share", "corr_share"))
    assert split == expected


def test_median_levels_and_heating_correlation_quoted(heating_scores, cooling_scores):
    """S9/S10 caption: median modelled/observed 0.999 heating and 1.145 cooling, median heating
    interannual correlation 0.55."""
    assert round((heating_scores.modelled / heating_scores.observed).median(), 3) == 0.999
    assert round((cooling_scores.modelled / cooling_scores.observed).median(), 3) == 1.145
    assert round(heating_scores.r.median(), 2) == 0.55


def test_cooling_correlation_by_country_quoted(cooling_scores):
    """S9/S10 caption: cooling correlation 0.90 in Italy and France, 0.86 in Greece, negative in
    Sweden and the United Kingdom."""
    assert round(cooling_scores.r["Italy"], 2) == 0.90
    assert round(cooling_scores.r["France"], 2) == 0.90
    assert round(cooling_scores.r["Greece"], 2) == 0.86
    assert cooling_scores.r["Sweden"] < 0 and cooling_scores.r["United Kingdom"] < 0


@pytest.mark.parametrize("fuel, error_twh, observed_twh, share_pct, decimals", [
    ("heating", 685, 3327, 21, 0),
    ("cooling", 19.6, 39.9, 49, 1),
])
def test_leave_one_country_out_quoted(fuel, error_twh, observed_twh, share_pct, decimals,
                                      heating_held_out, cooling_held_out):
    """S9/S10 table: summed hold-out error of the gap-fill rule, heating 685 of 3327 TWh/yr
    (21 %), cooling 19.6 of 39.9 TWh/yr (49 %)."""
    table = heating_held_out if fuel == "heating" else cooling_held_out
    assert round(float(table.loco_err_TWh.abs().sum()), decimals) == error_twh
    assert round(float(table.obs_TWh.sum()), decimals) == observed_twh
    assert round(100 * held_out_error_share(table)) == share_pct


def test_cooling_anchor_quoted(anchor_ratios):
    """S9/S10 table: cooling level anchor median 1.12, IQR 1.03-1.28, minimum 0.32."""
    assert round(anchor_ratios.median(), 2) == 1.12
    assert round(anchor_ratios.quantile(0.25), 2) == 1.03
    assert round(anchor_ratios.quantile(0.75), 2) == 1.28
    assert round(anchor_ratios.min(), 2) == 0.32


def test_cooling_anchor_maximum_quoted(anchor_ratios):
    """S9/S10 table: cooling level anchor range up to 1.43 (the data give 1.4348)."""
    assert round(anchor_ratios.max(), 2) == 1.43


def test_electrified_share_quoted(calibration):
    """S9/S10 prose: electrified_share spans 53x, Romania 0.014 to Malta 0.768, median 0.116."""
    share = calibration.coefficients["electrified_share"].dropna()
    assert share.idxmin() == "Romania" and round(share.min(), 3) == 0.014
    assert share.idxmax() == "Malta" and round(share.max(), 3) == 0.768
    assert round(share.median(), 3) == 0.116
    assert round(share.max() / share.min()) == 53


def test_r2_selection_would_shrink_residential_curvature_quoted(candidates, curvatures):
    """S10 prose: selecting on R2 instead would take a residential curvature 13 times smaller."""
    residential = candidates[candidates.sector == "res"]
    best_r2 = residential.loc[residential.r2.idxmax()]
    assert round(curvatures["res"] / best_r2.theta) == 13


# ---------------------------------------------------------------------------
# S11-S16: the conversion steps
# ---------------------------------------------------------------------------

def test_demand_change_bins_are_symmetric():
    """Every demand change-class edge set is symmetric about zero."""
    from cordex2energy_analysis.pipeline_inputs import DEMAND_BINS, HYDRO_BINS
    for edges in list(DEMAND_BINS.values()) + [HYDRO_BINS]:
        assert np.allclose(sorted(-np.array(edges)), edges), edges


@pytest.mark.parametrize("label", ["Solar power", "Wind power"])
def test_gridded_conversion_rows_share_scales(label, energy_grids, context):
    """S13/S14 put every row (annual and the four seasons) on one change scale and one
    absolute scale."""
    from cordex2energy_analysis.figures_pipeline import conversion_figure
    from cordex2energy_analysis.pipeline_inputs import conversion_rows
    annual, seasonal = energy_grids
    rows = conversion_rows(label, {}, {}, annual, seasonal)
    fig = conversion_figure(rows, context.countries_print, context.style, gridded=True)
    try:
        change_limits = {round(bar.norm.vmax, 6) for bar in fig.change_colorbars}
        absolute_limits = {round(bar.norm.vmax, 6) for bar in fig.reference_colorbars}
        assert len(fig.change_colorbars) == len(rows)
        assert len(change_limits) == 1, f"rows carry different change limits {change_limits}"
        assert len(absolute_limits) <= 1, f"rows carry different absolute limits {absolute_limits}"
    finally:
        plt.close(fig)
