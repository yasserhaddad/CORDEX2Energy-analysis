"""Battery dispatch on synthetic residual load: energy is conserved by every method."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from cordex2energy_analysis.power_system import apply_battery_dispatch

CAPACITY_GWH = 12.0
INITIAL_SOC_FRACTION = 0.5
ROUNDTRIP_EFFICIENCY = 0.81  # one-way efficiency 0.9


def constant_deficit(days: int, deficit_gwh: float = 5.0) -> xr.Dataset:
    """A single zone with the same positive residual load at every 3-h step."""
    time = pd.date_range("2001-01-01", periods=days * 8, freq="3h")
    values = np.full((1, time.size), deficit_gwh)
    return xr.Dataset({"residual_load": (("zone", "time"), values)},
                      coords={"zone": ["Z"], "time": time})


def discharged_energy(method: str, **options) -> float:
    """Energy the battery delivers over a run that never offers a surplus to charge from."""
    result = apply_battery_dispatch(
        constant_deficit(days=3), pd.Series({"Z": CAPACITY_GWH}), battery_duration_h=4.0,
        roundtrip_efficiency=ROUNDTRIP_EFFICIENCY, initial_soc_fraction=INITIAL_SOC_FRACTION,
        method=method, verbose=False, **options)
    return float(result["battery_dispatch_gwh"].sum())


def deliverable_energy() -> float:
    """The initial charge, after discharge losses."""
    return CAPACITY_GWH * INITIAL_SOC_FRACTION * np.sqrt(ROUNDTRIP_EFFICIENCY)


@pytest.mark.parametrize("lookahead_h", [0.0, 24.0])
def test_heuristic_never_delivers_more_than_the_initial_charge(lookahead_h):
    assert discharged_energy("heuristic", dispatch_lookahead_h=lookahead_h) <= \
        deliverable_energy() + 1e-9


def test_rationing_counts_each_deficit_step_once():
    # Over a 24-h window of 8 deficit steps the first discharge is 1/8 of the deliverable energy.
    result = apply_battery_dispatch(
        constant_deficit(days=1, deficit_gwh=100.0), pd.Series({"Z": 1000.0}),
        battery_duration_h=4.0, roundtrip_efficiency=ROUNDTRIP_EFFICIENCY,
        initial_soc_fraction=INITIAL_SOC_FRACTION, dispatch_lookahead_h=24.0, method="heuristic",
        verbose=False)
    first = float(result["battery_dispatch_gwh"].isel(time=0).squeeze())
    assert np.isclose(first, 1000.0 * INITIAL_SOC_FRACTION * np.sqrt(ROUNDTRIP_EFFICIENCY) / 8)


def test_rolling_qp_carries_the_state_of_charge_forward():
    pytest.importorskip("gurobipy")
    # Without look-ahead each daily window must start from the previous day's end state, so the
    # three days together cannot deliver more than the initial charge.
    assert discharged_energy("optimal", dispatch_lookahead_h=0.0, commit_h=24.0) <= \
        deliverable_energy() + 1e-6
