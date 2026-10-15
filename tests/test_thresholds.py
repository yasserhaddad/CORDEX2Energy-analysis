"""The three threshold methods on a synthetic residual load with a known tail.

The paper uses ``PERCENTILE``; the extreme-value (``EVT``) and ``TAILS`` methods are kept and
tested here so they stay usable.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from cordex2energy_analysis.extremes_thresholds import compute_active_threshold


@pytest.fixture(scope="module")
def synthetic_residual_load():
    """Two zones, 30 years of 3-hourly values: a normal body with a generalised-Pareto tail."""
    random = np.random.default_rng(42)
    time = pd.date_range("1991-01-01", "2020-12-31 21:00", freq="3h")
    values = []
    for scale in (100.0, 50.0):
        body = random.normal(0.0, scale, len(time))
        tail = random.random(len(time)) < 0.05
        body[tail] = scale * (1.5 + random.pareto(4.0, tail.sum()))
        values.append(body)
    return xr.DataArray(np.array(values).T, dims=("time", "zone"),
                        coords={"time": time, "zone": ["A", "B"]})


def test_percentile_is_the_positive_p95(synthetic_residual_load):
    threshold = compute_active_threshold(synthetic_residual_load, "PERCENTILE", percentile=95.0)
    for zone in ("A", "B"):
        values = synthetic_residual_load.sel(zone=zone).values
        expected = np.percentile(values[values > 0], 95.0)
        assert np.isclose(float(threshold.sel(zone=zone)), expected, rtol=1e-6)


@pytest.mark.parametrize("method", ["EVT", "TAILS"])
def test_extreme_value_methods_give_a_positive_tail_threshold(synthetic_residual_load, method):
    threshold = compute_active_threshold(synthetic_residual_load, method)
    for zone in ("A", "B"):
        value = float(threshold.sel(zone=zone))
        values = synthetic_residual_load.sel(zone=zone).values
        assert np.isfinite(value) and value > 0
        # A tail threshold: above the median of the deficits, below the largest value.
        assert np.median(values[values > 0]) < value < values.max()
