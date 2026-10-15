"""Country-level energy quantities per chain and warming level: the inputs of the maps and tables.

Both the residual-load run's per-country output and the heating/cooling demand product are read
as monthly totals (see :mod:`cordex2energy_analysis.figure_cache`): every quantity here is a sum
over whole months of whole-year warming-level windows, which monthly totals reproduce exactly.
"""

from __future__ import annotations

import xarray as xr

from .config import CLIMATE_SCENARIO, RUN_TAG
from .extremes_analysis import compute_c2e_gwl_field, compute_country_gwl_field
from .figure_cache import country_monthly, demand_monthly
from .power_system import build_c2e_name_to_iso2, load_tyndp_country_codes

#: Energy-per-3h-step supply terms whose sum is the run's total supply.
SUPPLY_VARIABLES = ("gen_pv", "gen_wind_on", "gen_wind_off", "gen_hydro_ror",
                    "reservoir_dispatch_gwh", "battery_dispatch_gwh")
#: Daily-mean demand (GW) summed over a period, to TWh: GW x 24 h = GWh, /1000 = TWh.
DAILY_MEAN_GW_TO_TWH = 24e-3


def load_country_output(context, files=("residual_load.nc", "components.nc"),
                        run_tag: str = RUN_TAG) -> dict:
    """Monthly totals of the run's per-country output, per chain. Cached on disk.

    Parameters
    ----------
    context : PaperContext
        Data paths and chains.
    files : tuple of str
        Output files of the run to merge.
    run_tag : str
        The run.

    Returns
    -------
    dict
        ``{chain: Dataset}`` with dimensions ``(time, country)``; country codes are ISO2.
    """
    return country_monthly(
        {model: context.paths.residual_load / f"{model}_{CLIMATE_SCENARIO}" / run_tag
         for model in context.models},
        files)


def load_demand_by_month(context) -> dict:
    """Monthly totals of daily-mean heating and cooling demand, countries keyed by ISO2.

    Parameters
    ----------
    context : PaperContext
        Data paths and chains.

    Returns
    -------
    dict
        ``{"heating": DataArray, "cooling": DataArray}``, dimensions ``(model, country, time)``,
        in GW-days per month (multiply a period sum by :data:`DAILY_MEAN_GW_TO_TWH` for TWh).
    """
    demand = demand_monthly(context.paths.demand, context.models, CLIMATE_SCENARIO)
    name_to_iso2 = build_c2e_name_to_iso2(load_tyndp_country_codes(context.paths.tyndp_demand_workbook))
    return {variable: array.assign_coords(
                country=[name_to_iso2.get(str(name), str(name)) for name in array["country"].values])
            for variable, array in demand.items()}


def country_field(country_output: dict, period_windows: dict, expression, period: str,
                  months=None) -> xr.DataArray:
    """Mean annual (or seasonal) total of a run quantity per chain and country, in TWh.

    Parameters
    ----------
    country_output : dict
        From :func:`load_country_output`.
    period_windows : dict
        ``{warming level: {chain: (first year, last year)}}``.
    expression : callable
        ``expression(dataset) -> DataArray`` selecting or combining run variables (GWh per step).
    period : str
        Warming level.
    months : list of int, optional
        Restrict to these months (a season); ``None`` gives the annual total.

    Returns
    -------
    xarray.DataArray
        Dimensions ``(model, country)``.
    """
    return compute_country_gwl_field(country_output, period_windows, period, expression,
                                     months=months)


def demand_field(demand_by_month: dict, period_windows: dict, variable: str, period: str,
                 months=None) -> xr.DataArray:
    """Mean annual (or seasonal) heating or cooling demand per chain and country, in TWh.

    Parameters
    ----------
    demand_by_month : dict
        From :func:`load_demand_by_month`.
    period_windows : dict
        ``{warming level: {chain: (first year, last year)}}``.
    variable : {"heating", "cooling"}
        Which demand.
    period : str
        Warming level.
    months : list of int, optional
        Restrict to these months (a season); ``None`` gives the annual total.

    Returns
    -------
    xarray.DataArray
        Dimensions ``(model, country)``.
    """
    return compute_c2e_gwl_field(demand_by_month, period_windows, period, variable,
                                 scale=DAILY_MEAN_GW_TO_TWH, months=months)
