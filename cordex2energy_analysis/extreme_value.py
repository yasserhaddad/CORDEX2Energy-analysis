"""Re-export shim — all implementations have moved to the ``extremes_*`` modules.

Kept for backwards compatibility with existing notebooks and scripts that
import directly from this module.  New code should import from the canonical
locations:

    from cordex2energy_analysis.extremes_thresholds import find_evt_pot_threshold, ...
    from cordex2energy_analysis.extremes_events     import detect_events, ...
    from cordex2energy_analysis.extremes_analysis   import compile_and_compare_periods, ...
"""
from .extremes_thresholds import (
    find_evt_pot_threshold,
    compute_evt_thresholds_for_all_zones,
    calibrate_seasonal_evt_thresholds,
)
from .extremes_events import (
    calculate_climatological_baseline,
    compute_continuous_system_metrics,
    compute_comprehensive_event_metrics,
    compute_seasonal_evt_metrics,
)
from .extremes_analysis import (
    compile_and_compare_periods,
    resolve_gwl_period_definitions,
    compute_seasonal_exceedance_fractions,
    compute_zone_pair_cooccurrence,
    extract_evt_duration_severity_records,
    compute_unserved_energy_fraction,
    compute_flexibility_attenuation_metrics,
    battery_split_mode,
    flex_metric_columns,
)

__all__ = [
    "find_evt_pot_threshold",
    "compute_evt_thresholds_for_all_zones",
    "calibrate_seasonal_evt_thresholds",
    "calculate_climatological_baseline",
    "compute_continuous_system_metrics",
    "compute_comprehensive_event_metrics",
    "compute_seasonal_evt_metrics",
    "compile_and_compare_periods",
    "resolve_gwl_period_definitions",
    "compute_seasonal_exceedance_fractions",
    "compute_zone_pair_cooccurrence",
    "extract_evt_duration_severity_records",
    "compute_unserved_energy_fraction",
    "compute_flexibility_attenuation_metrics",
    "battery_split_mode",
    "flex_metric_columns",
]
