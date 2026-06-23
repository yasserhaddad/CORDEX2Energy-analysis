"""Analysis and visualization tools for CORDEX2Energy climate-energy outputs."""

from .analysis import *
from .plotting import *
from .plotting_extremes import (
    resolve_model_periods,
    plot_residual_load_kde,
    plot_evt_threshold_heatmap,
    plot_spaghetti_iqr,
    plot_violin_grid,
    plot_attribution_stacked_bars,
    plot_attribution_change_bars,
    plot_seasonal_exceedance_distribution,
    plot_seasonal_conditional_metrics,
    plot_seasonal_attribution,
    plot_zone_cooccurrence_heatmaps,
    plot_zone_cooccurrence_boxplots,
    plot_duration_severity_scaling,
    plot_seasonal_attribution_change,
    plot_component_anomaly_distributions,
    plot_seasonal_component_anomaly_distributions,
)
from .extreme_value import (
    resolve_gwl_period_definitions,
    calibrate_seasonal_evt_thresholds,
    compute_seasonal_evt_metrics,
    compute_seasonal_exceedance_fractions,
    compute_zone_pair_cooccurrence,
    extract_evt_duration_severity_records,
    compute_unserved_energy_fraction,
    compute_flexibility_attenuation_metrics,
)

__all__ = [
    "analysis",
    "plotting",
    "plotting_extremes",
    # plotting_extremes — original
    "resolve_model_periods",
    "plot_residual_load_kde",
    "plot_evt_threshold_heatmap",
    "plot_spaghetti_iqr",
    "plot_violin_grid",
    "plot_attribution_stacked_bars",
    "plot_attribution_change_bars",
    # plotting_extremes — seasonal
    "plot_seasonal_exceedance_distribution",
    "plot_seasonal_conditional_metrics",
    "plot_seasonal_attribution",
    "plot_zone_cooccurrence_heatmaps",
    "plot_zone_cooccurrence_boxplots",
    "plot_duration_severity_scaling",
    "plot_seasonal_attribution_change",
    "plot_component_anomaly_distributions",
    "plot_seasonal_component_anomaly_distributions",
    # extreme_value — GWL + seasonal
    "resolve_gwl_period_definitions",
    "calibrate_seasonal_evt_thresholds",
    "compute_seasonal_evt_metrics",
    "compute_seasonal_exceedance_fractions",
    "compute_zone_pair_cooccurrence",
    "extract_evt_duration_severity_records",
    "compute_unserved_energy_fraction",
    "compute_flexibility_attenuation_metrics",
]
