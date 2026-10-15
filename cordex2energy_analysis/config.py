"""Settings shared by every publication notebook, and the context they all start from.

The analysis settings (run tags, warming levels, threshold, event detector, zones, seasons) are
defined once here. :func:`setup` resolves the data paths, the climate-model chains and their
warming-level windows, and returns them with the figure style and the country geometry.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPOSITORY = Path(__file__).resolve().parent.parent

CLIMATE_SCENARIO = "rcp85"
#: Wind turbine of the paper's runs and of the gridded wind capacity factors.
TURBINE = "SWT120/3600"
#: The paper's residual-load run: DE/2050 demand, TYNDP LOW/2050 supply, the demand model with
#: free cooling, reservoir dispatch, utility and prosumer batteries, the sequential flexibility
#: scheme and SWT120 wind turbines at 3600 W/m2.
RUN_TAG = "DE_2050_c2edem_LOW_2050_freecool_reservoirdispatch_batteries_prosumer_sysflex_SWT120_3600"
#: The same system with the HIGH/2050 build-out, for the scenario comparison.
RUN_TAG_HIGH_2050 = "DE_2050_c2edem_HIGH_2050_freecool_reservoirdispatch_batteries_prosumer_sysflex_SWT120_3600"
SUPPLY_SCENARIO, SUPPLY_YEAR = "LOW", 2050
COMPARISON_SCENARIO = "HIGH"
DEMAND_SCENARIO, DEMAND_YEAR = "DE", 2050
#: Demand model of the paper's runs, as it appears in run tags and baseload file names.
DEMAND_MODEL_TAG = "c2edem"

WARMING_LEVELS = [1.5, 2.0, 3.0]
THRESHOLD_PERCENTILE = 95.0
EVENT_DETECTION = "spa"
DURATION_BINS = {"sub_daily": (0, 24), "multi_day": (24, float("inf"))}
DURATION_LABELS = {"sub_daily": "sub-daily\n(≤ 24 h)", "multi_day": "multi-day\n(> 24 h)"}
DURATION_TITLES = {"sub_daily": "Sub-daily", "multi_day": "Multi-day"}
#: Share of chains that must agree on the sign of a change for it to count as robust.
ROBUSTNESS_FRACTION = 2 / 3
#: Ensemble spread drawn as bands and error bars: the interquartile range across chains.
SPREAD = "iqr"

SEASON_MAP = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
              6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}
SEASON_ORDER = ["DJF", "MAM", "JJA", "SON"]
SEASON_MONTHS = {season: [month for month, name in SEASON_MAP.items() if name == season]
                 for season in SEASON_ORDER}
MAIN_SEASONS = ("DJF", "JJA")
SHOULDER_SEASONS = ("MAM", "SON")

ZONE_ORDER = ["Iberian_Peninsula", "South_East", "Central_Western",
              "Central_Eastern", "Nordic", "British_Isles"]
ZONE_LABELS = {"Iberian_Peninsula": "Iberia", "South_East": "South-East",
               "Central_Western": "Central-West", "Central_Eastern": "Central-East",
               "Nordic": "Nordic", "British_Isles": "British Isles"}
SOUTHERN_ZONES = ["Iberian_Peninsula", "South_East"]

GWL_COLORS = ["#4A90E2", "#F5A623", "#D0021B"]
#: Drivers of the residual load, in the stacking order of the attribution bars.
DRIVER_PALETTE = {"Demand": "#D95F02", "PV": "#E6AB02", "Wind Onshore": "#7570B3",
                  "Wind Offshore": "#1B7C3D", "Hydro RoR": "#3182BD",
                  "Reservoir Hydro": "#08519C", "Battery": "#8B5A2B"}
#: Legend text for drivers whose key reads badly; the key itself stays the column name.
DRIVER_DISPLAY = {"Reservoir Hydro": "Hydro Reservoir"}
DURATION_PALETTE = {"sub_daily": "#3182bd", "multi_day": "#e6550d"}
SCENARIO_LABELS = {"LOW": "LOW / 2050 (the paper)", "HIGH": "HIGH / 2050"}


@dataclass(frozen=True)
class DataPaths:
    """Where the inputs and the figures live, resolved from ``paths.json``.

    Parameters
    ----------
    processed : pathlib.Path
        Root of the processed CORDEX2Energy data.
    figures : pathlib.Path
        Root of the figure output directory.
    """

    processed: Path
    figures: Path

    @property
    def extremes(self) -> Path:
        """Residual-load runs, TYNDP inputs and the CERRA reference."""
        return self.processed / "extremes"

    @property
    def residual_load(self) -> Path:
        """One folder per chain (``<chain>_<scenario>``) holding one folder per run tag."""
        return self.extremes / "residual_load"

    @property
    def demand(self) -> Path:
        """Hourly heating and cooling demand per chain."""
        return self.processed / "energy" / "europe" / "demand"

    @property
    def gwl_grids(self) -> Path:
        """Gridded warming-level means and the country-level energy caches."""
        return self.processed / "energy" / "analysis"

    @property
    def tyndp(self) -> Path:
        """TYNDP 2024 scenario workbooks and the baseload cache."""
        return self.extremes / "tyndp"

    @property
    def baseload(self) -> Path:
        """The baseload typical year the paper's runs used (built by ``compute_or_load_baseload``)."""
        return (self.tyndp / "baseload"
                / f"baseload_{DEMAND_SCENARIO}_{DEMAND_YEAR}_{DEMAND_MODEL_TAG}.nc")

    @property
    def tyndp_demand_workbook(self) -> Path:
        """The TYNDP 2024 demand-scenario workbook, source of the country codes."""
        return self.tyndp / "Demand_Scenarios_TYNDP_2024_After_Public_Consultation.xlsb"

    @property
    def country_geometry(self) -> Path:
        """Natural Earth admin-0 countries at 1:50 m."""
        return self.processed / "maps" / "natural_earth" / "ne_50m_admin_0_countries.geojson"

    @property
    def warming_level_years(self) -> Path:
        """Years at which each CMIP5 run crosses each warming level (1861-1900 baseline)."""
        return self.processed / "cmip5_warming_levels_all_ens_1861_1900.yml"


def load_paths(paths_file: Path | str | None = None) -> DataPaths:
    """Read ``paths.json``.

    Parameters
    ----------
    paths_file : path-like, optional
        JSON file with ``path_data_processed`` and, optionally, ``path_figs``. Defaults to
        ``paths.json`` at the repository root.

    Returns
    -------
    DataPaths
        The resolved data and figure directories.
    """
    paths_file = Path(paths_file) if paths_file is not None else REPOSITORY / "paths.json"
    entries = json.loads(paths_file.read_text())
    return DataPaths(processed=Path(entries["path_data_processed"]),
                     figures=Path(entries.get("path_figs", REPOSITORY / "figs")))


def find_models(paths: DataPaths, run_tag: str = RUN_TAG,
                scenario: str = CLIMATE_SCENARIO) -> list[str]:
    """Climate-model chains that have output for a residual-load run.

    Parameters
    ----------
    paths : DataPaths
        Data locations.
    run_tag : str
        Residual-load run to look for.
    scenario : str
        Emission scenario suffix of the chain folders.

    Returns
    -------
    list of str
        Chain identifiers (``<RCM>_<GCM>_<member>``), sorted.
    """
    return sorted("_".join(folder.name.split("_")[:-1])
                  for folder in paths.residual_load.glob(f"*_{scenario}")
                  if (folder / run_tag).exists())


@dataclass
class PaperContext:
    """Everything a publication notebook needs before its first figure.

    Attributes
    ----------
    paths : DataPaths
        Data locations.
    figure_dir : pathlib.Path
        Output directory of this notebook's figures.
    models : list of str
        Climate-model chains of the paper's run.
    period_windows : dict
        ``{warming level label: {chain: (first year, last year)}}``.
    style : PaperStyle
        Colours, zone order and robustness rule shared by every figure.
    countries : geopandas.GeoDataFrame
        Country polygons in the map projection.
    countries_print : geopandas.GeoDataFrame
        The same polygons simplified to print resolution, for the maps.
    """

    paths: DataPaths
    figure_dir: Path
    models: list
    period_windows: dict
    style: object
    countries: object = field(repr=False)
    countries_print: object = field(repr=False)

    @property
    def period_names(self) -> list[str]:
        """Warming-level labels, coolest first."""
        return list(self.period_windows)

    @property
    def reference(self) -> str:
        """The reference warming level (+1.5 °C)."""
        return self.period_names[0]

    @property
    def warmest(self) -> str:
        """The warmest warming level (+3.0 °C)."""
        return self.period_names[-1]

    @property
    def robustness_threshold(self) -> int:
        """Number of chains that must agree on a sign."""
        return int(np.ceil(len(self.models) * ROBUSTNESS_FRACTION))


def setup(figure_folder: str, paths_file: Path | str | None = None) -> PaperContext:
    """Resolve paths, chains, warming-level windows, style and geometry for one notebook.

    Parameters
    ----------
    figure_folder : str
        Sub-folder of the figure directory this notebook writes to.
    paths_file : path-like, optional
        Passed to :func:`load_paths`.

    Returns
    -------
    PaperContext
        The shared starting point of every publication notebook.
    """
    from .extremes_analysis import resolve_gwl_period_definitions
    from .plotting_paper import MAP_CRS, PaperStyle, load_country_geometry

    paths = load_paths(paths_file)
    figure_dir = paths.figures / figure_folder
    figure_dir.mkdir(parents=True, exist_ok=True)
    models = find_models(paths)
    period_windows, _, _ = resolve_gwl_period_definitions(
        warming_levels=WARMING_LEVELS, models=models,
        chain_meta_path=REPOSITORY / "models.json", gwl_yaml_path=paths.warming_level_years,
        climate_scenario=CLIMATE_SCENARIO)
    period_names = list(period_windows)
    style = PaperStyle(
        zones=ZONE_ORDER, zone_labels=ZONE_LABELS, period_names=period_names,
        gwl_colors=dict(zip(period_names, GWL_COLORS)), driver_palette=DRIVER_PALETTE,
        duration_palette=DURATION_PALETTE, season_order=SEASON_ORDER, n_models=len(models),
        robustness_threshold=int(np.ceil(len(models) * ROBUSTNESS_FRACTION)), spread=SPREAD)
    countries = load_country_geometry(paths.country_geometry, crs=MAP_CRS)
    countries_print = countries.copy()
    # 2 km in the equal-area projection (~0.02 degrees): well below print resolution on a
    # ~45 mm map, and it halves the size of the vector figures.
    countries_print["geometry"] = countries.geometry.simplify(2000.0)
    return PaperContext(paths=paths, figure_dir=figure_dir, models=models,
                        period_windows=period_windows, style=style, countries=countries,
                        countries_print=countries_print)
