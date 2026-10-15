"""Analysis of residual-load extremes from CORDEX2Energy climate-energy outputs.

Import the submodule you need; importing the package itself loads nothing else:

    from cordex2energy_analysis.extremes_analysis import compute_duration_stratified_metrics
    from cordex2energy_analysis.power_system import compute_residual_load

Submodules
----------
analysis              Warming-level windows, sign agreement and ensemble-spread helpers.
power_system          TYNDP inputs, residual load and storage dispatch.
extremes_events       Event detection on residual-load series.
extremes_thresholds   Deficit thresholds (percentile, extreme-value and TAILS methods).
extremes_statistics   Generalised Pareto goodness-of-fit statistics for the EVT thresholds.
extremes_analysis     Event, attribution and storage-attenuation tables per chain and level.
figure_cache          Disk cache for the expensive intermediate tables.
plotting_paper        Map helpers and the shared figure style.
plotting_publication  The publication figure builders.
"""

__version__ = "0.1.0"
