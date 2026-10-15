"""The TYNDP 2024 build-outs the paper runs on (Supplementary Figures S17, S18)."""

import numpy as np
import pytest

from conftest import requires_data

pytestmark = requires_data


@pytest.fixture(scope="module")
def capacities(context):
    from cordex2energy_analysis.config import COMPARISON_SCENARIO, SUPPLY_SCENARIO, SUPPLY_YEAR
    from cordex2energy_analysis.power_system import installed_capacities_by_country
    workbook = context.paths.tyndp / "20231103 - Final Supply Inputs for TYNDP 2024 Scenarios.xlsx"
    return {scenario: installed_capacities_by_country(workbook, scenario, SUPPLY_YEAR)
            for scenario in (SUPPLY_SCENARIO, COMPARISON_SCENARIO)}


def test_every_modelled_country_has_a_zone():
    from cordex2energy_analysis.power_system import COUNTRY_ZONE, MODELLED_COUNTRIES
    assert set(MODELLED_COUNTRIES) <= set(COUNTRY_ZONE)


def test_zonal_and_european_totals_match_the_countries(capacities):
    from cordex2energy_analysis.config import (COMPARISON_SCENARIO, SUPPLY_SCENARIO, SUPPLY_YEAR,
                                               ZONE_ORDER)
    from cordex2energy_analysis.power_system import compare_build_outs, sum_by_zone
    totals, added, reference_by_zone = compare_build_outs(
        capacities, SUPPLY_SCENARIO, COMPARISON_SCENARIO, SUPPLY_YEAR, ZONE_ORDER)
    low = capacities[SUPPLY_SCENARIO]
    for technology in low.columns:
        country_total = float(low[technology].sum())
        assert np.isclose(country_total, float(sum_by_zone(low, ZONE_ORDER)[technology].sum()))
        assert np.isclose(country_total, totals.loc[technology, f"{SUPPLY_SCENARIO} / {SUPPLY_YEAR}"])
        europe_added = (totals.loc[technology].iloc[1] - totals.loc[technology].iloc[0])
        assert np.isclose(europe_added, added[technology].sum())


def test_high_adds_capacity_everywhere(capacities):
    from cordex2energy_analysis.config import (COMPARISON_SCENARIO, SUPPLY_SCENARIO, SUPPLY_YEAR,
                                               ZONE_ORDER)
    from cordex2energy_analysis.power_system import compare_build_outs
    totals, added, _ = compare_build_outs(capacities, SUPPLY_SCENARIO, COMPARISON_SCENARIO,
                                          SUPPLY_YEAR, ZONE_ORDER)
    assert (totals["change %"] > 0).all()
    assert (added.sum(axis=1) > 0).all()


def test_reservoir_countries_are_all_named(context):
    from cordex2energy_analysis.figures_system import COUNTRY_NAMES
    from cordex2energy_analysis.runs import reservoir_operation_curves
    countries, _, _ = reservoir_operation_curves(context)
    assert set(countries) <= set(COUNTRY_NAMES)
