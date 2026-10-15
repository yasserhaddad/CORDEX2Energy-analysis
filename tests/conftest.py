"""Shared fixtures: the paper's context, run and tables, built once per test session.

The fixtures read the processed data named in ``paths.json`` and the disk cache
(:mod:`cordex2energy_analysis.figure_cache`); tests that need them are skipped when the data are
not available.
"""

import json
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parent.parent


def data_available() -> bool:
    """Whether ``paths.json`` exists and points at an existing processed-data directory."""
    paths_file = REPOSITORY / "paths.json"
    if not paths_file.exists():
        return False
    return Path(json.loads(paths_file.read_text())["path_data_processed"]).exists()


requires_data = pytest.mark.skipif(not data_available(), reason="processed data not available")


@pytest.fixture(scope="session")
def context():
    from cordex2energy_analysis.config import setup
    return setup("paper_publication")       # an existing figure folder: nothing is written


@pytest.fixture(scope="session")
def run(context):
    from cordex2energy_analysis.runs import load_run
    return load_run(context)


@pytest.fixture(scope="session")
def tables(run, context):
    from cordex2energy_analysis.runs import event_tables
    return event_tables(run, context)


@pytest.fixture(scope="session")
def high_run(context):
    """HIGH/2050 with its own CERRA threshold."""
    from cordex2energy_analysis.config import RUN_TAG_HIGH_2050
    from cordex2energy_analysis.runs import load_run
    return load_run(context, RUN_TAG_HIGH_2050)


@pytest.fixture(scope="session")
def high_on_paper_threshold(context, run):
    """HIGH/2050 scored on the paper's (LOW/2050) threshold."""
    from cordex2energy_analysis.config import RUN_TAG_HIGH_2050
    from cordex2energy_analysis.runs import load_run
    return load_run(context, RUN_TAG_HIGH_2050, threshold=run.threshold)


@pytest.fixture(scope="session")
def country_output(context):
    from cordex2energy_analysis.fields import load_country_output
    return load_country_output(context)


@pytest.fixture(scope="session")
def demand_by_month(context):
    from cordex2energy_analysis.fields import load_demand_by_month
    return load_demand_by_month(context)


@pytest.fixture(scope="session")
def europe_rows(country_output, demand_by_month, context):
    from cordex2energy_analysis.tables import europe_change_rows
    return europe_change_rows(country_output, demand_by_month, context.period_windows)


@pytest.fixture(scope="session")
def europe_change(europe_rows, context):
    from cordex2energy_analysis.plotting_publication import europe_change_table
    from cordex2energy_analysis.tables import EUROPE_COLUMNS
    return europe_change_table(europe_rows, EUROPE_COLUMNS, context.reference, context.warmest)
