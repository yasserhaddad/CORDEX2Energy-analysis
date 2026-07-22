"""Analysis and visualization tools for CORDEX2Energy climate-energy outputs."""

from .analysis import *
from .plotting import *
from .plotting_paper import (
    PaperStyle,
    load_country_geometry,
    plot_country_choropleth,
    plot_mean_state_maps,
    plot_tail_survival,
    plot_sub_daily_regime,
    plot_multi_day_regime,
    plot_threshold_calibration,
    plot_zone_season_change,
    plot_significance,
    plot_level_vs_persistence,
    plot_threshold_sensitivity,
)
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
    plot_event_diurnal_profile,
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
from .extremes_statistics import (
    _gpd_right_tail_ad_statistic,
    _gpd_right_tail_ad_batch,
    _lmoment_gpd_bootstrap_refit,
    _gpd_cdf_vectorised,
)
from .extremes_thresholds import (
    find_evt_pot_threshold,
    compute_evt_thresholds_for_all_zones,
    calibrate_seasonal_evt_thresholds as _calibrate_seasonal_evt_thresholds_new,
    find_tails_threshold,
    TailsResult,
    compute_tails_thresholds_for_all_zones,
)
from .extremes_events import (
    detect_events,
    detect_seasonal_events,
    calculate_climatological_baseline,
    compute_continuous_system_metrics,
    compute_comprehensive_event_metrics,
    compute_seasonal_evt_metrics as _compute_seasonal_evt_metrics_new,
)
from .extremes_analysis import (
    compile_and_compare_periods,
    resolve_gwl_period_definitions as _resolve_gwl_period_definitions_new,
    compute_seasonal_exceedance_fractions as _compute_seasonal_exceedance_fractions_new,
    compute_zone_pair_cooccurrence as _compute_zone_pair_cooccurrence_new,
    extract_evt_duration_severity_records as _extract_evt_duration_severity_records_new,
    compute_unserved_energy_fraction as _compute_unserved_energy_fraction_new,
    compute_flexibility_attenuation_metrics as _compute_flexibility_attenuation_metrics_new,
    extract_event_diurnal_profiles,
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
    "plot_event_diurnal_profile",
    # extreme_value shim — GWL + seasonal (backwards compatible)
    "resolve_gwl_period_definitions",
    "calibrate_seasonal_evt_thresholds",
    "compute_seasonal_evt_metrics",
    "compute_seasonal_exceedance_fractions",
    "compute_zone_pair_cooccurrence",
    "extract_evt_duration_severity_records",
    "compute_unserved_energy_fraction",
    "compute_flexibility_attenuation_metrics",
    "extract_event_diurnal_profiles",
    # extremes_thresholds — threshold selection
    "find_evt_pot_threshold",
    "compute_evt_thresholds_for_all_zones",
    "find_tails_threshold",
    "TailsResult",
    "compute_tails_thresholds_for_all_zones",
    # extremes_events — detection + metrics
    "detect_events",
    "detect_seasonal_events",
    "calculate_climatological_baseline",
    "compute_continuous_system_metrics",
    "compute_comprehensive_event_metrics",
    # extremes_analysis — derived analyses
    "compile_and_compare_periods",
    # paper figures — derived tables + figure builders
    "PaperStyle",
    "load_country_geometry",
    "plot_country_choropleth",
    "plot_mean_state_maps",
    "plot_tail_survival",
    "plot_sub_daily_regime",
    "plot_multi_day_regime",
    "plot_threshold_calibration",
    "plot_zone_season_change",
    "plot_significance",
    "plot_level_vs_persistence",
    "plot_cooccurrence",
    "plot_threshold_sensitivity",
]
