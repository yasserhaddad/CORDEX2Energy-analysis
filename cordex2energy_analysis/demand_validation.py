"""How the demand model performs against its calibration target (Supplementary Figs. S9-S10).

The heating and cooling coefficients are derived by
:mod:`cordex2energy.conversion_energy.demand_coefficients` from JRC-IDEES national energy and
CERRA degree days. Nothing here refits them: the functions evaluate the coefficients as the
pipeline applies them, decompose their error, test the gap-fill rule by leaving one country out,
and profile how sharply the data determine the cooling curvature.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from cordex2energy.conversion_energy import demand_coefficients

#: Cooling sectors and their JRC-IDEES columns (GWh per year).
COOL_SECTORS = {"res": "cool_res_gwh", "srv": "cool_srv_gwh"}
SECTOR_LABEL = {"res": "residential", "srv": "service"}
BASELOAD_LABEL = {"FE": "per-country baseload", "noFE": "shared baseload"}
#: Countries in the study that JRC-IDEES does not cover, so their coefficients are gap-filled.
NO_JRC_ISO2 = {"Albania": "AL", "Bosnia and Herzegovina": "BA", "Kosovo": "XK",
               "Montenegro": "ME", "North Macedonia": "MK", "Norway": "NO",
               "Serbia": "RS", "Switzerland": "CH", "Ukraine": "UA"}
#: Curvatures at which the residual sum of squares is profiled.
THETA_GRID = np.geomspace(1e-5, 0.5, 90)


@dataclass
class CalibrationData:
    """The inputs of the demand-coefficient derivation.

    Attributes
    ----------
    coefficients : pandas.DataFrame
        The production demand coefficients, one row per country (by name).
    hdd_annual, cdd_annual : pandas.DataFrame
        National annual CERRA degree days, years by countries.
    jrc : dict
        ``{country: DataFrame}`` of JRC-IDEES annual energy (GWh), 2023 edition or 2015 fallback.
    population_cooling, population_heating : pandas.Series
        World Bank population in 2015 (cooling gap-fill year) and 2020 (heating gap-fill year).
    cool_years : list of int
        Years of the cooling fit window with degree days.
    fitted : list of str
        Countries with JRC data, degree days and population.
    """

    coefficients: pd.DataFrame
    hdd_annual: pd.DataFrame
    cdd_annual: pd.DataFrame
    jrc: dict
    population_cooling: pd.Series
    population_heating: pd.Series
    cool_years: list
    fitted: list

    @property
    def population_millions(self) -> dict:
        """2015 population of the fitted countries, in millions."""
        return {country: float(self.population_cooling[country]) / 1e6 for country in self.fitted}

    @property
    def cdd_fit(self) -> pd.DataFrame:
        """Annual CDD of the fitted countries over the cooling fit window."""
        return self.cdd_annual.loc[self.cool_years, self.fitted]


def annual_degree_days(inputs_dir: Path, name: str) -> pd.DataFrame:
    """National annual degree days from the cached CERRA aggregation.

    Parameters
    ----------
    inputs_dir : pathlib.Path
        The demand-model input folder (``<processed>/energy/inputs``).
    name : {"hdd", "cdd"}
        Which degree days.

    Returns
    -------
    pandas.DataFrame
        Years (int) by countries (name).
    """
    frame = pd.read_csv(inputs_dir / f"{name}_eurostat_cerra_allcountries.csv", index_col=0)
    frame.index = frame.index.astype(str).str.slice(0, 4).astype(int)
    return frame.astype(float)


def load_calibration_data(inputs_dir: Path) -> CalibrationData:
    """Read the coefficients and every input of their derivation.

    Parameters
    ----------
    inputs_dir : pathlib.Path
        The demand-model input folder (``<processed>/energy/inputs``).

    Returns
    -------
    CalibrationData
        Coefficients, degree days, JRC-IDEES series and population.
    """
    coefficients = pd.read_csv(inputs_dir / "demand_coefficients.csv", index_col=0)
    hdd_annual, cdd_annual = annual_degree_days(inputs_dir, "hdd"), annual_degree_days(inputs_dir, "cdd")
    jrc = {}
    for name in coefficients.index:
        series = demand_coefficients.load_jrc_fec_series(inputs_dir / "JRC_2023", name, "2023")
        if series is None:
            series = demand_coefficients.load_jrc_fec_series(inputs_dir / "JRC_2015", name, "2015")
        if series is not None:
            jrc[name] = series
    population_file = inputs_dir / "World_bank_population.csv"
    population_cooling = demand_coefficients.world_bank_population(population_file, "2015")
    population_heating = demand_coefficients.world_bank_population(population_file, "2020")
    cool_years = [year for year in demand_coefficients.COOL_FIT_YEARS if year in cdd_annual.index]
    fitted = sorted(country for country in jrc
                    if country in cdd_annual.columns and country in population_cooling.index)
    return CalibrationData(coefficients=coefficients, hdd_annual=hdd_annual, cdd_annual=cdd_annual,
                           jrc=jrc, population_cooling=population_cooling,
                           population_heating=population_heating, cool_years=cool_years,
                           fitted=fitted)


def production_curvatures(coefficients: pd.DataFrame) -> dict:
    """The fitted cooling curvature of each sector.

    Parameters
    ----------
    coefficients : pandas.DataFrame
        The production demand coefficients.

    Returns
    -------
    dict
        ``{sector: theta}`` (per degree-day).
    """
    return {sector: float(coefficients[f"theta_cool_{sector}"].dropna().iloc[0])
            for sector in COOL_SECTORS}


def error_decomposition(name: str, observed: np.ndarray, modelled: np.ndarray) -> dict:
    """Level, spread and the three-term split of the mean squared error.

    Parameters
    ----------
    name : str
        Country.
    observed, modelled : numpy.ndarray
        Annual energy, same years.

    Returns
    -------
    dict
        Means, 10th/90th percentiles, NRMSE and bias (% of the observed mean), correlation ``r``,
        MSE and the shares of squared bias, variance and lack of correlation in it.

    Notes
    -----
    ``MSE = (mean_m - mean_o)^2 + (sd_m - sd_o)^2 + 2 sd_m sd_o (1 - r)`` (Kobayashi & Salam 2000).
    """
    observed_std, modelled_std = observed.std(), modelled.std()
    correlation = float(np.corrcoef(observed, modelled)[0, 1])
    mean_squared_error = float(np.mean((modelled - observed) ** 2))
    squared_bias = float((modelled.mean() - observed.mean()) ** 2)
    variance = float((modelled_std - observed_std) ** 2)
    lack_of_correlation = float(2 * modelled_std * observed_std * (1 - correlation))
    return dict(country=name,
                observed=observed.mean(), modelled=modelled.mean(),
                obs_lo=np.percentile(observed, 10), obs_hi=np.percentile(observed, 90),
                mod_lo=np.percentile(modelled, 10), mod_hi=np.percentile(modelled, 90),
                nrmse_pct=100 * np.sqrt(mean_squared_error) / observed.mean(),
                bias_pct=100 * (modelled.mean() - observed.mean()) / observed.mean(),
                r=correlation, mse=mean_squared_error,
                bias_share=squared_bias / mean_squared_error,
                var_share=variance / mean_squared_error,
                corr_share=lack_of_correlation / mean_squared_error,
                n=int(observed.size))


def heating_performance(data: CalibrationData) -> tuple[pd.DataFrame, np.ndarray]:
    """The through-origin heating model ``P * HDD`` against JRC-IDEES, per country.

    Parameters
    ----------
    data : CalibrationData
        Coefficients and calibration inputs.

    Returns
    -------
    table : pandas.DataFrame
        :func:`error_decomposition` per country (TWh per year), largest first.
    ratio : numpy.ndarray
        Modelled over observed, every country-year pooled.
    """
    rows, pooled = [], []
    for name, series in data.jrc.items():
        if name not in data.hdd_annual.columns or name not in data.coefficients.index:
            continue
        years = data.hdd_annual.index.intersection(series.index)
        hdd = data.hdd_annual.loc[years, name].to_numpy(float)
        observed = series.loc[years, "heat_thermal_gwh"].to_numpy(float)
        valid = np.isfinite(hdd) & np.isfinite(observed) & (hdd > 0) & (observed > 0)
        hdd, observed = hdd[valid], observed[valid]
        if hdd.size < 8:
            continue
        modelled = float(data.coefficients.loc[name, "P_heat_thermal"]) * 24.0 * hdd
        rows.append(error_decomposition(name, observed / 1e3, modelled / 1e3))
        pooled.append(modelled / observed)
    table = pd.DataFrame(rows).set_index("country").sort_values("observed", ascending=False)
    return table, np.concatenate(pooled)


def cooling_performance(data: CalibrationData) -> pd.DataFrame:
    """The anchored, sector-split saturating cooling model against JRC-IDEES, per country.

    Parameters
    ----------
    data : CalibrationData
        Coefficients and calibration inputs.

    Returns
    -------
    pandas.DataFrame
        :func:`error_decomposition` per country (TWh per year) over the cooling fit window,
        largest first.
    """
    coefficients, cdd_annual, cool_years = data.coefficients, data.cdd_annual, data.cool_years
    smoothed_cdd = cdd_annual.rolling(5, min_periods=1).mean()
    rows = []
    for name in coefficients.index:
        if name not in data.jrc or name not in cdd_annual.columns:
            continue
        ratio = float(coefficients.loc[name, "corrective_ratio_cool"])
        if not np.isfinite(ratio):
            continue
        observed = sum(data.jrc[name][column].reindex(cool_years)
                       for column in COOL_SECTORS.values()) / 1e3
        modelled = 0.0
        for sector in COOL_SECTORS:
            theta = float(coefficients.loc[name, f"theta_cool_{sector}"])
            baseload = float(coefficients.loc[name, f"B_cool_{sector}"])
            intensity = float(coefficients.loc[name, f"P_cool_{sector}"])
            share = 1.0 - np.exp(-theta * smoothed_cdd.loc[cool_years, name])
            # B is a power [GW] -> x8760 h; P is [GW per degC.day] on ANNUAL degree days -> x24 h.
            modelled = modelled + (baseload * 8760
                                   + intensity * (cdd_annual.loc[cool_years, name] * share) * 24) / 1e3
        modelled = modelled * ratio
        valid = np.isfinite(observed) & np.isfinite(modelled) & (observed > 0)
        if valid.sum() < 8:
            continue
        rows.append(error_decomposition(name, observed[valid].to_numpy(float),
                                        modelled[valid].to_numpy(float)))
    return pd.DataFrame(rows).set_index("country").sort_values("observed", ascending=False)


def jrc_name_to_iso2() -> dict:
    """Country name to ISO2 code as the derivation keys them, with Greece as ``GR``.

    Returns
    -------
    dict
        ``{name: ISO2}`` matching the map geometry.
    """
    return {name: ("GR" if iso2 == "EL" else iso2)
            for iso2, name in demand_coefficients._ISO_TO_NAME.items()}


def by_iso2(heating: pd.DataFrame, cooling: pd.DataFrame, column: str) -> dict:
    """One performance column of both fuels, re-keyed by ISO2 for the maps.

    Parameters
    ----------
    heating, cooling : pandas.DataFrame
        From :func:`heating_performance` and :func:`cooling_performance`.
    column : str
        Column to map, e.g. ``"nrmse_pct"`` or ``"r"``.

    Returns
    -------
    dict
        ``{"heating": Series, "cooling": Series}`` indexed by ISO2.
    """
    name_to_iso2 = jrc_name_to_iso2()
    return {"heating": heating[column].rename(index=name_to_iso2),
            "cooling": cooling[column].rename(index=name_to_iso2)}


def cooling_anchor_ratios(coefficients: pd.DataFrame) -> pd.Series:
    """The cooling level anchor (observed over un-anchored modelled) of the fitted countries.

    Parameters
    ----------
    coefficients : pandas.DataFrame
        The production demand coefficients.

    Returns
    -------
    pandas.Series
        ``corrective_ratio_cool`` of every country that is not gap-filled.
    """
    return coefficients.loc[coefficients["cool_gap_filled"] == 0, "corrective_ratio_cool"].dropna()


def heating_leave_one_out(data: CalibrationData) -> pd.DataFrame:
    """Heating error of the production gap-fill rule, each JRC country held out in turn.

    Parameters
    ----------
    data : CalibrationData
        Coefficients and calibration inputs.

    Returns
    -------
    pandas.DataFrame
        Per country: observed mean (``obs_TWh``) and the error of the population-scaled donor-mean
        coefficient (``loco_err_TWh``, ``loco_err_pct``), largest first.
    """
    coefficients, hdd_annual, population = data.coefficients, data.hdd_annual, data.population_heating
    pool = [country for country in data.jrc
            if country in hdd_annual.columns and country in population.index
            and np.isfinite(coefficients.loc[country, "P_heat_thermal"])]
    rows = []
    for held in pool:
        rest = [country for country in pool if country != held]
        predicted = (coefficients.loc[rest, "P_heat_thermal"].mean()
                     * float(population[held]) / float(population[rest].mean()))
        actual = float(coefficients.loc[held, "P_heat_thermal"])
        years = hdd_annual.index.intersection(data.jrc[held].index)
        mean_hdd = hdd_annual.loc[years, held].mean()
        rows.append(dict(country=held,
                         obs_TWh=data.jrc[held].loc[years, "heat_thermal_gwh"].mean() / 1e3,
                         loco_err_TWh=(predicted - actual) * 24.0 * mean_hdd / 1e3,
                         loco_err_pct=100 * (predicted - actual) / actual))
    return pd.DataFrame(rows).set_index("country").sort_values("obs_TWh", ascending=False)


def cooling_per_capita(data: CalibrationData) -> dict:
    """Per-capita cooling energy of the fitted countries over the fit window.

    Parameters
    ----------
    data : CalibrationData
        Calibration inputs.

    Returns
    -------
    dict
        ``{sector: DataFrame}``, years by countries, GWh per million inhabitants.
    """
    population = data.population_millions
    return {sector: pd.DataFrame({country: data.jrc[country][column].reindex(data.cool_years)
                                  / population[country]
                                  for country in data.fitted}, index=data.cool_years)
            for sector, column in COOL_SECTORS.items()}


def cooling_leave_one_out(data: CalibrationData, theta: dict) -> pd.DataFrame:
    """Cooling error of the production gap-fill rule, each fitted country held out in turn.

    Parameters
    ----------
    data : CalibrationData
        Calibration inputs.
    theta : dict
        ``{sector: curvature}`` from :func:`production_curvatures`.

    Returns
    -------
    pandas.DataFrame
        Per country: observed mean (``obs_TWh``), the error of the pooled fit without it
        (``loco_err_TWh``) and that error in % where more than 0.5 TWh is observed.
    """
    population = data.population_millions
    cdd_fit = data.cdd_fit
    per_capita = cooling_per_capita(data)
    errors, observed_means = {}, {}
    for sector in COOL_SECTORS:
        energy = per_capita[sector]
        for held in data.fitted:
            rest = [country for country in data.fitted if country != held]
            fit = demand_coefficients.fit_pooled_cooling(energy[rest], cdd_fit[rest],
                                                         theta=theta[sector])
            share = 1.0 - np.exp(-theta[sector] * cdd_fit[held].rolling(5, min_periods=1).mean())
            predicted = (float(fit["b"].mean()) + fit["p0"] * (cdd_fit[held] * share)) \
                * population[held] / 1e3
            observed = energy[held] * population[held] / 1e3
            valid = np.isfinite(observed) & np.isfinite(predicted)
            errors[(held, sector)] = float(predicted[valid].mean() - observed[valid].mean())
            observed_means[(held, sector)] = float(observed[valid].mean())
    errors, observed_means = pd.Series(errors).unstack(), pd.Series(observed_means).unstack()
    table = pd.DataFrame({"obs_TWh": observed_means.sum(axis=1), "loco_err_TWh": errors.sum(axis=1)})
    table["loco_err_pct"] = np.where(table["obs_TWh"] > 0.5,
                                     100 * table["loco_err_TWh"] / table["obs_TWh"], np.nan)
    return table.sort_values("obs_TWh", ascending=False)


def curvature_candidates(data: CalibrationData) -> pd.DataFrame:
    """Every curvature candidate the selection rule considers, and how sharply each is pinned.

    Parameters
    ----------
    data : CalibrationData
        Calibration inputs.

    Returns
    -------
    pandas.DataFrame
        One row per sector, country group and baseload specification: the fitted ``theta``, its
        ``r2`` and ``width``, the curvatures within 1 % of the best residual sum of squares
        (``band_lo``, ``band_hi``), the CDD span of the group and whether it was ``chosen``.

    Notes
    -----
    The residual sum of squares at a fixed curvature is ``(1 - R2) * SS_tot``, so the profile
    normalised by its minimum is ``(1 - R2) / min(1 - R2)``.
    """
    cdd_fit = data.cdd_fit
    cool_twh = pd.Series({country: sum(data.jrc[country][column].reindex(data.cool_years).mean()
                                       for column in COOL_SECTORS.values()) / 1e3
                          for country in data.fitted})
    groups = demand_coefficients.cooling_country_groups(cdd_fit, cool_twh)
    per_capita = cooling_per_capita(data)
    candidates = []
    for sector in COOL_SECTORS:
        pick = demand_coefficients.select_cooling_curvature(per_capita[sector], cdd_fit, groups)
        for (group, spec), fit in pick["candidates"].items():
            members = [country for country in groups[group]
                       if country in per_capita[sector].columns]
            profile = np.array([
                1.0 - demand_coefficients.fit_pooled_cooling(
                    per_capita[sector][members], cdd_fit[members], theta=float(curvature),
                    theta_grid=np.array([float(curvature)]), fixed_effects=(spec == "FE"))["r2"]
                for curvature in THETA_GRID])
            band = THETA_GRID[profile <= profile.min() * 1.01]
            members_cdd = cdd_fit[members].mean()
            candidates.append(dict(sector=sector, group=group, spec=spec, theta=fit["theta"],
                                   r2=fit["r2"], width=fit["width"],
                                   band_lo=float(band.min()), band_hi=float(band.max()),
                                   cdd_span=float(members_cdd.max() - members_cdd.min()),
                                   chosen=(group == pick["group"] and spec == pick["spec"])))
    return pd.DataFrame(candidates)
