# Code review, part 1: `power_system.py`, data-producing scripts, repo readiness

Reviewed 2026-10-14 against the working tree, which is being refactored at the same time. Line
numbers will drift, so every finding also names its function. Each finding below was checked by
reading the code, and most were also checked with small read-only Python runs on the production
inputs and outputs under `/net/exo/landclim/yhaddad/clim2energy-ch`. No pipeline was re-run.

The paper's configuration, taken from `cordex2energy_analysis/config.py` and `scripts/run_tyndp_projections_all_models.sh`:
`--demand_model cordex2energy_demand --no-cooling-target --rolling_window_residual_load 1
--turbine SWT120/3600 --use_reservoir_hydro --use_batteries --use_prosumer_batteries
--flexibility_scheme system`, with supply LOW/2050 (main) and HIGH/2050 (comparison).

## Status (2026-10-14)

| Finding | Status |
|---|---|
| R1 CERRA PV timing | Fixed and re-run (2026-10-15): `compute_pv.py` (CORDEX2Energy) computes CERRA PV on window midpoints; `load_cerra_energy_data` relabels to window start and refuses end-labelled files. CERRA PV recomputed on the full grid, CERRA LOW and HIGH passes re-run. LOW p95 thresholds: Iberia 76.87 → 71.84 (−6.5 %), South-East 137.21 → 142.79 (+4.1 %), Central-West 434.29 → 444.98 (+2.5 %), others within ±1 %. Headline effect at +3.0 °C vs +1.5 °C (median of per-chain changes): Iberia sub-daily +12.5 → +20.6 events/yr, South-East +8.3 → +3.5, Europe (sum of zone medians) +21.7 → +24.4. Old outputs backed up under `energy/europe/pv_backup_cerra_endlabel_20261014` and `extremes/residual_load_backup_cerra_pv_endlabel_20261014`. |
| R2 Gurobi | Fixed: `gurobi` channel and `gurobi::gurobi=13.0.2` in `env.yml`, `gurobipy>=13.0` required in `pyproject.toml`, licence documented in the README. |
| R3 GCM longitude box | Fixed and regenerated: the driving GCMs move by +0.1 to +0.2 °C and +0.07 to +0.11 mm/d at +1.5 °C; chain and CERRA rows unchanged. |
| R4 gridded wind turbine | Fixed and regenerated: `--turbine` (default SWT120/3600), turbine in the file name (`mean_gwl_<level>_SWT120_3600[_meteorological].nc`), `load_gwl_grids(turbine=...)`. The old E-126 files are left in place. |
| R9, R10 battery heuristics | Fixed, with tests in `tests/test_battery_dispatch.py`. Not on the paper's path. |
| H1 README | Fixed: installation, Gurobi, data sources, citation. |
| H4 help text and metadata | Fixed: k_cool range, min-SOC help, M7 jargon, `--use_prosumer_batteries` without `--use_batteries` is an error, system-scheme attributes. |
| H5 stale docstrings | Fixed. |
| P1 S19 baseload | Fixed: S19 reads `DataPaths.baseload` (`baseload_DE_2050_c2edem.nc`). |
| P2 storage counterfactual | Charging part fixed; detection part open (see P2). |
| P7 baseload cache key | Fixed: settings stored as attributes and checked on load; older files without them are used with a warning. |
| P8 1.44 vs 1.43 | Test now expects 1.43; the manuscript text needs the same change. |

## Summary

- **One issue changes published numbers in the main analysis (R1).** In the CERRA run, PV generation is 3 h behind demand and wind. The CORDEX runs are aligned correctly. The paper's thresholds are the CERRA p95 values, so every exceedance metric inherits the offset.
- **Two issues change SI numbers.** The GCM points in S5 are averaged over the wrong longitude box (R3). The gridded wind capacity-factor GWL means use a different turbine from the paper (R4).
- **The repo cannot be installed or rerun as shipped.**
  - The `gurobi` package cannot be resolved from the declared conda channels. A full Gurobi licence is required, not optional (R2).
  - The run script and the CLI defaults do not reproduce the paper run (R6).
  - `paths.json` is git-ignored and no template is provided (R6).
  - The README has no installation, data or reproduction instructions (H1).
- **The dispatch QPs (reservoir, annual battery) are correct.** I found no unit, sign or efficiency errors on the paper's code path:
  - generation MW × 3 h × 1e-3 → GWh/3h;
  - hourly GW demand summed to GWh/3h;
  - η = √RTE applied once on charge and once on discharge;
  - `E[0]==E[T]` cyclicity;
  - turbine efficiency on discharge only;
  - positive residual load = deficit throughout.

  The defects I found in the battery code sit on code paths the paper does not use (R9–R11).

---

## Critical

### R1. CERRA PV is labelled 3 h later than CORDEX PV, demand and wind

`scripts/compute_tyndp_projections_extremes.py`: `load_cordex_energy_data` (≈l.127) vs `load_cerra_energy_data` (≈l.241).

**What is wrong.** CORDEX PV is shifted by −1h30 (`ds_pv["time"] = ds_pv.time - 1h30`), which takes its labels from 01:30, 04:30, … to 00:00, 03:00, …. CERRA PV gets no shift. I compared the mean JJA 2000 diurnal cycle of `pv_tyndp_agg_*.nc` (rounded) for France, FR00:

| label (UTC) | 06 | 09 | 12 | 15 | 18 |
|---|---|---|---|---|---|
| CORDEX CNRM-ALADIN63, after the shift | 0.258 | **0.526** | 0.487 | 0.207 | 0.004 |
| CERRA, as used | 0.017 | 0.266 | **0.536** | 0.497 | 0.078 |

- The CERRA curve is the CORDEX curve delayed by exactly one 3-h step. PL00 and ES00 show the same pattern.
- Solar geometry fixes the conventions:
  - CERRA 12:00 > 15:00 and ≫ 09:00, so CERRA labels the *end* of the interval (12:00 = [09,12)).
  - The shifted CORDEX value at 09:00 is [09,12), so CORDEX labels the *start*.
- Demand is labelled at the interval start: `compute_residual_load` uses `resample("3h").sum()`, which puts [09,12) at 09:00. CORDEX is therefore consistent with demand. CERRA PV is matched to the demand of the *following* 3 h.

**Why it matters.**
- The paper's deficit threshold is the CERRA p95 of positive residual load per zone (doc/paper_summary.md). Every count, severity and duration metric is measured against it.
- S25 compares CORDEX and CERRA directly.
- On the published CERRA run (`DE_2050_c2edem_HIGH_2050_freecool_…_sysflex_SWT120_3600`), correcting the PV alignment *before flexibility* raises the zonal p95 of positive residual load by +0.1 % (Iberia) to +9 % (Nordic):
  - British Isles +5.5 %
  - Central-East +3.6 %
  - Central-West +3.7 %
  - South-East +7.4 %
- The p95 *after flexibility* is what the paper uses. It needs a re-run because the storage QPs re-optimise, and the effect could be larger.
- Changes between GWLs are measured against the same threshold, so they move less than absolute exceedance levels. They are still not invariant.

**Fix.**
1. Make the label convention explicit and identical for all sources. For example, assert a common start-of-interval convention right after loading.
2. Shift CERRA PV by −3 h, or fix the labelling upstream in `cordex2energy`.
3. Re-run the CERRA pass for every configuration and recompute the thresholds.
4. Add a check that compares the PV diurnal peak hour between CERRA and one CORDEX chain over 1991–2020.

**Confidence.** High that the offset exists and that CERRA is the wrong one. Medium on the post-flexibility magnitude.

#### R1 verification (paper run, LOW/2050, after flexibility)

Run tag: `DE_2050_c2edem_LOW_2050_freecool_reservoirdispatch_batteries_prosumer_sysflex_SWT120_3600`. The CERRA pass was re-run in scratch. Reproducing it first gave a bit-for-bit match (max |ΔRL| = 0).

**Two separate errors.**
1. **Labels.** CERRA ssrd is accumulated over (t−3h, t], so its label is the window end. `compute_pv.py` keeps that label, and `load_cerra_energy_data` never shifts it. Demand and the shifted CORDEX PV cover [t, t+3h), so CERRA PV is one 3-h step late.
2. **Solar geometry.** `compute_pv_from_config` evaluates the sun position at the label time. That is the window midpoint for CORDEX (correct) but the window end for CERRA, 1.5 h late. This cuts CERRA PV in the 15–18 UTC window by 40–70 %. It also lowers the long-term mean capacity factor by about 6–9 % in ES, PT and FR. Relabelling alone does not fix it.

The bias correction itself is aligned correctly: CERRA is relabelled onto the CORDEX midpoints there. The CORDEX chains are also correct.

**p95 thresholds after storage (GWh per 3-h step).** Fix (ii) uses a sample-based recompute of the geometry on 20–40 pixels per zone, applied as a delta. It is an approximation; the exact version needs a full-grid recompute.

| Zone | Paper | (i) relabel only | (ii) relabel + midpoint geometry |
|---|---|---|---|
| Iberian Peninsula | 76.87 | 75.75 (−1.5 %) | **71.81 (−6.6 %)** |
| South-East | 137.21 | 139.55 (+1.7 %) | **142.66 (+4.0 %)** |
| Central-Western | 434.29 | 434.72 (+0.1 %) | **444.98 (+2.5 %)** |
| Central-Eastern | 192.13 | 194.03 (+1.0 %) | 194.18 (+1.1 %) |
| Nordic | 56.96 | 56.99 (+0.1 %) | 56.77 (−0.3 %) |
| British Isles | 145.23 | 145.76 (+0.4 %) | 146.04 (+0.6 %) |

**Effect on the chains' exceedance frequency** (ensemble-median % of 3-h steps above threshold, 9 chains):

| | IP | SE | CW | CE | Nordic | BI |
|---|---|---|---|---|---|---|
| 1991–2020, paper | 2.06 | 6.48 | 2.11 | 4.83 | 1.12 | 1.39 |
| 1991–2020, corrected | 4.03 | 4.91 | 1.88 | 4.47 | 1.14 | 1.34 |
| 2071–2100, paper | 6.70 | 8.67 | 1.84 | 4.19 | 0.80 | 1.88 |
| 2071–2100, corrected | 11.01 | 6.58 | 1.62 | 3.81 | 0.81 | 1.82 |

- **Iberia roughly doubles** (×1.96 now, ×1.65 at end of century), because its residual-load tail is steep around 70–77 GWh.
- **South-East falls** to about ×0.76.
- The corrected/paper ratio is similar in both periods, so relative changes move less than absolute levels.
- The CERRA exceedances shift from the 09 UTC label to 15–21 UTC.

**Fix.**
- **Upstream (CORDEX2Energy).** Compute CERRA PV with timestamps moved to the window midpoint (t − 1h30) before pvlib, over the full grid. Then relabel to window start in `load_cerra_energy_data`, the same as CORDEX.
- **Here.** Re-run the CERRA pass for every run tag, LOW and HIGH. The chains only need re-thresholding, not re-running.

---

## Major

### R2. Gurobi cannot be installed from the declared channels, and the paper needs a full licence

`env.yml` (`- gurobi`), `pyproject.toml` (`optional-dependencies.optimal-dispatch`), `power_system._optimize_reservoir_gurobi` and `_optimize_battery_gurobi`.

**What is wrong.**
- `env.yml` lists only `conda-forge` and `defaults`. `conda search -c conda-forge --override-channels gurobi` finds only `gurobi-logtools`; the `gurobi` package exists only on the `gurobi` channel. As written, `conda env create -f env.yml` fails.
- The pyproject comment says *"the default heuristic needs neither"*. That is misleading:
  - The reservoir has no heuristic; it always calls Gurobi.
  - The paper's `--flexibility_scheme system` always calls the annual battery QP.
  - Every published number therefore depends on Gurobi.
- Each yearly QP has about 4 × 2920 ≈ 11.7 k variables. That exceeds the size-limited licence bundled with pip/conda `gurobipy` (2000 variables/constraints), so a full or academic licence is required. None of this is documented.

**Why it matters.** A reader cannot rebuild the environment or reproduce the dispatch.

**Fix.**
- Add `- gurobi` as a channel, or install `gurobipy` via pip.
- Make Gurobi a required dependency for the published configuration.
- Document the licence requirement, and the Gurobi version used (13.0.2 in `c2e-analysis`), in the README.
- Optionally record `gurobipy.gurobi.version()` in the output attributes.

**Confidence.** High.

### R3. The GCM European means in S5 use the wrong longitude box

`scripts/compute_european_t_p.py`, GCM block (≈l.212: `.sel(**EXTENT)` with `EXTENT["lon"] = slice(-12, 39)`).

**What is wrong.** cmip5-ng `g025` files use 0…360 longitudes (verified: 1.25 … 358.75). `slice(-12, 39)` therefore keeps only 0–39 °E and silently drops −12–0 °E (Iberia west, Ireland, the UK, the Atlantic margin). The docstring claims *"the same geographic box"* as the RCMs.

**Why it matters.** The published `european_t_p_by_gwl.csv` values match the buggy box exactly. I recomputed the correct box at +1.5 °C:

| driving GCM | tas | pr |
|---|---|---|
| CNRM-CM5 | 11.44 → 11.54 °C | 1.741 → 1.850 mm/d (+6 %) |
| HadGEM2-ES | 12.76 → 12.97 °C | 1.887 → 1.956 mm/d |
| EC-EARTH | 11.13 → 11.32 °C | 1.809 → 1.889 mm/d |

This shifts every GCM point in SI Fig. S5, and the whole CMIP5 cloud, relative to the RCM and CERRA points.

**Fix.** `fields[var] = fields[var].assign_coords(lon=((fields[var].lon + 180) % 360) - 180).sortby("lon")` before `.sel`. Then assert that the selected longitude range spans about −11 to 39.

**Confidence.** High.

### R4. The gridded wind CF GWL means use turbine E-126/7580, not the paper's SWT120/3600

`scripts/compute_energy_means_gwl.py`, `selected_turbine = EUROPE_TURBINES[0]` (≈l.298).

**What is wrong.**
- `EUROPE_TURBINES = ["E-126/7580", "SWT120/3600", "SWT142/3150"]`.
- The output `energy/analysis/annual/gwl/mean_gwl_15.nc` carries `wind.attrs['turbine_type'] = 'E-126/7580'`.
- The file name does not record the turbine.
- `si_note1_pipeline.ipynb` (`load_gwl_grids`) loads these grids.

**Why it matters.** The wind-resource maps or changes shown in the SI come from a different power curve from the one behind every residual-load result. The script's own run script notes mean onshore CF 0.20 vs 0.40 across turbines. Relative changes are also curve-dependent.

**Fix.** Add a `--turbine` argument defaulting to `SWT120/3600`, put the turbine in the output file name, and regenerate.

**Confidence.** High that the file uses E-126. Medium on exactly which SI panel uses `wind`.

### R5. Missing hydro coverage is silently filled with zero

`power_system.compute_residual_load` (`_gen`: `.reindex(country=all_countries, fill_value=0.0)`) and `apply_reservoir_dispatch` (countries without inflow skipped).

**What is wrong.**
- Run-of-river comes from a source with only 6 countries (AT, CH, DE, ES, FR, IT). This was verified on both the CERRA CSV and the published CORDEX `components.nc`. NO, SE, PT, RO, FI, the Balkans and the rest get ROR = 0 without any message.
- Reservoir inflow exists for 10 countries. Countries that have `E_store > 0` in `hydro_capacities.csv` but no inflow are skipped silently: FI, BA, HR, GR, LV, PL, SK, SI, CZ, DE.
- `hydro_capacities.csv` uses `GB` while c2e uses `UK`. This has no effect today only because GB `E_store = 0`.

**Why it matters.** The residual load of the Nordic, Iberian and South-East zones lacks part of its hydro. The gap is the same in CERRA and CORDEX, so it affects levels more than changes. It is still a data-coverage limitation that the paper and SI should state.

**Fix.**
- Print, or write to the output attributes, the list of countries with zero-filled ROR or skipped reservoir.
- Treat missing ROR as missing, not zero, unless explicitly whitelisted.
- Map `GB` to `UK`.
- State the coverage in the Methods.

**Confidence.** High.

### R6. The repo, its scripts and their defaults do not reproduce the paper run

- **The run script runs the comparison case, not the main one.** `scripts/run_tyndp_projections_all_models.sh` sets `supply_scenario="HIGH"`, but the paper's main `RUN_TAG` is the LOW/2050 one (`config.py`). Producing both needs a manual edit.
- **The CLI defaults are not the paper settings.** `compute_tyndp_projections_extremes.py`:
  - `--paths_file ../../paths_so4.json` points to a file that does not exist.
  - `--turbine E-126/7580`.
  - `--rolling_window_residual_load 8`. This is a trailing 24-h mean (`center=False`) with a ≈10.5 h phase lag, so a default run would compute "sub-daily" extremes on a day-smoothed series.
  - `--demand_model demand.ninja`, with no free cooling and no flexibility.
- **Path configuration is missing or user-specific.**
  - `paths.json` is git-ignored, holds absolute `/net/exo/…` and `/home/yhaddad/…` paths, and has no `paths.example.json` template.
  - `compute_climate_means_gwl.py` hard-codes `PATH_ROUTED = /net/land/landclim/pseubert/out_rcm/hist`, another user's directory, and uses the `hist` tree for rcp85.
  - `compute_cmip5_gmst.py` and `compute_european_t_p.py` default to `/net/atmos/data/cmip5[-ng]`.
- **The shell script must be run from `scripts/`.** It uses relative `../paths.json` and `python compute_…py`, has no shebang and no `set -euo pipefail`, and does not activate an environment.

**Fix.**
- Make the CLI defaults equal the paper configuration, or add a `--preset paper` option.
- Have the shell script loop over both supply scenarios.
- Commit a `paths.example.json`.
- Turn the external archive paths into documented CLI arguments.
- Add a shebang and `set -euo pipefail`, and `cd "$(dirname "$0")"`.

**Confidence.** High.

### R7. The baseload cache key is incomplete, and the documented floor (15 %) is not the one used (10 %)

`power_system.compute_or_load_baseload` (cache path ≈l.1490) and the script's `--floor_fraction` default (0.10).

**What is wrong.**
- **The cache key is incomplete.** It is `(scenario, year, demand_model, legacy)` only. `floor_fraction`, `reference_period`, `modulate_baseload`, the CERRA demand file (`cerra.nc` vs `cerra_noanchor.nc`/`cerra_nogrowth.nc` both exist) and the TYNDP workbook versions are not part of it. A stale file is silently reused when any of them changes.
- **The cache has no provenance.** The cached `.nc` carries no attributes.
- **The floor used is 10 %, not 15 %.**
  - The function default is 0.15, while the script passes 0.10.
  - The minimum of the cached `baseload_DE_2050_c2edem.nc` over mean (market + prosumer) is 0.1016 for MT. That is consistent with a 10 % floor and impossible under 15 %.
  - `doc/Pipeline_documentation.md` and `doc/paper_summary.md` state *"floored at 15 % of mean market"*.

**Why it matters.**
- The Methods text, if it follows those docs, misstates the floor. The numerical effect is small: only MT and BA come near the floor.
- The cache can silently decouple outputs from the arguments that are supposed to define them.

**Fix.**
- Include every input that affects the result in the key, or a hash of them.
- Write the arguments as NetCDF attributes and check them on load.
- Align the function default, the script default and the docs.

**Confidence.** High.

---

## Minor

### R8. Leap-year handling of the TYNDP baseload works only because two errors cancel

`power_system.load_tyndp_demand_profiles` (≈l.129) and `compute_typical_year_profile` (≈l.1383).

- **The TYNDP sheets are on a 2018 calendar.** Every climate year has 8760 rows, and the sheet's `Date` column runs from 2018-01-01 to 2018-12-31. The loader labels them `pd.date_range(f"{year}-01-01", periods=8760)`, so for 2008 the rows after 28 Feb are labelled one day early, and the data ends on 30 Dec. The comment *"date_range handles leap years (8784 h) automatically"* is false: the length comes from the data.
- **`compute_typical_year_profile` is wrong for a genuine leap-year series.** It drops 29 Feb but does not shift `dayofyear`.
  - A synthetic test on 2008 puts 1 Mar into slot 1440, which is 2 Mar of a non-leap year.
  - Slots 1416–1439 come out missing and slots 8760–8783 appear, contrary to the docstring's "always exactly 8760, 0–8759".
- The two errors cancel for TYNDP 2008, except that 1 Mar averages only 2 years.
- **Thermal subtraction is also misaligned by a day.** `compute_tyndp_baseload` subtracts real-calendar CERRA 2008 thermal demand from the shifted TYNDP labels, a one-day misalignment over about 10 months of 2008. The effect is negligible.

**Fix.** Build the timestamps from the sheet's `Date` column or an explicit noleap mapping, and reuse the `doy` shift from `expand_typical_year` in `compute_typical_year_profile`.

**Confidence.** High on the mechanism, low on any visible effect.

### R9. `apply_battery_dispatch(method="optimal")` with `dispatch_lookahead_h=0` never carries the state of charge forward

- `_optimize_battery_gurobi` returns `E.X[:-1]`.
- With no look-ahead, `commit == len(soc)`, so `if commit < len(soc): soc0 = soc[commit]` never fires.
- Every day therefore restarts at `initial_soc_fraction × cap` with a free terminal SOC, which is free energy every day.

This path is not used by the paper (`system` → `annual`).

**Fix.** Return `E.X` in full and seed the next window with `E[commit]`.

**Confidence.** High.

### R10. The heuristic discharge rationing counts the current step twice

`apply_battery_dispatch`, heuristic branch.

`future = rl_arr[t:t+lookahead_steps]` already includes `t`, and the code then adds `remaining = 1 + count(future > 0)` on top. An isolated peak can therefore draw only half the available energy. This is not used by the paper.

**Fix.** `remaining = max(1, int(np.sum(future > 0)))`.

**Confidence.** High.

### R11. The prosumer heuristic's daily SOC reset does not conserve energy

`apply_prosumer_battery_dispatch` resets the SOC to 50 % at 00 UTC, which creates or destroys stored energy every day. This only affects the `market` scheme, not the paper.

**Fix.** Document it as a modelling simplification, or carry the SOC across days.

**Confidence.** High.

### R12. Silent zero or NaN masking in aggregation

- **`compute_residual_load`.**
  - `total_demand_1h.resample("3h").sum()` has no `min_count`, so empty bins (29 Feb, after the baseload drops it) become 0 GWh demand.
  - With demand.ninja upsampling, the last bin holds 1 of 3 hours.
  - Today both are removed by the time intersection with generation, but any generation input that contains 29 Feb would turn them into spurious large surpluses.
- **`aggregate_by_balancing_zones`.** `groupby().sum()` treats NaN as 0.
- **`compute_tyndp_baseload` and `compute_or_load_baseload`.** `where(baseload >= floor, other=floor)` replaces NaN with the floor.

There are no NaNs in the current outputs (verified on one chain).

**Fix.** Use `.sum(min_count=3)` and `sum(min_count=1)`, and assert there are no NaNs after each join.

**Confidence.** High on the mechanism, with no current effect.

### R13. The CORDEX run-of-river CSV is stamped at 12:00, CERRA's at 00:00

`preprocess_hydro_ror` forward-fills from 12:00. Each CORDEX day's ROR therefore covers 12:00 to 09:00 of the next day, a 12-h phase lag relative to CERRA. The first four steps of 1991 are also lost: the published residual load starts at 1991-01-01 12:00. The effect is negligible because ROR is smooth.

**Fix.** Floor the timestamps to the day before upsampling.

**Confidence.** High.

### R14. Small issues in `compute_estimated_generation`

- Offshore capacity for MT (0.4 GW in 2050 HIGH) is dropped silently because the offshore CF file has no Malta.
- The "reintegrate dropped zone into `children[0]`" step is described as *"exact"*. That holds only with a single child; today the one case is LUG1 → LUB1.
- The output attribute says `"MWh"`, but the value is MW (mean power). The script multiplies by 3 × 1e-3 correctly.

**Fix.** Warn on capacity that is not mapped, and correct the attribute and docstring.

**Confidence.** High.

### R15. The residual-load series has 27-h gaps at every 29 Feb

- Generation is on a noleap calendar, so the residual load skips 29 Feb (27-h gaps; verified on HadREM3/HadGEM2 output).
- The reservoir and battery QPs treat consecutive array elements as contiguous 3-h steps. The effect on dispatch is negligible.

Part 2 must check that event detection and duration code handle these gaps, for example events spanning 28 Feb to 1 Mar.

**Confidence.** High.

### R16. Reservoir inflow handling

`apply_reservoir_dispatch`.

- **Timing.** The weekly totals are labelled at the start of the week and are linearly interpolated *between* those labels. This shifts inflow timing by about half a week and does not conserve each week's total.
- **Missing coverage is hidden.** `.reindex(nearest).bfill().ffill().fillna(0.0)` hides missing coverage.
- **No NaN guard.** Unlike the battery path, NaN in the residual load raises a GurobiError.
- **Wrong annotation.** The return annotation of `_optimize_reservoir_gurobi` declares 3 values; the function returns 4.

**Fix.** Use a step function (`ffill`, then `/56`) with labels at the centre or start of each week, warn when the inflow does not cover the residual-load period, and fix the annotation.

**Confidence.** High on the mechanics, low on the effect.

### R17. Thermal-target definitions

`build_thermal_targets`.

- **Inconsistent hot-water treatment.** Heating combines Households "Space heating", which excludes hot water, with Buildings (tertiary) "Space heating & hot water", which includes it. The tertiary hot-water part, which is not weather-sensitive, is rescaled as if it were heating.
- **Wrong unit in the docstring.** `ds_market` is documented as GW, but it is MW: the code correctly uses ×1e-6 for MWh → TWh.

I checked FR DE-2050: spreadsheet total 513 TWh vs market + prosumer profiles ≈ 502 TWh, so the totals are consistent.

**Fix.** Document the definition, or use the Buildings subsector consistently.

**Confidence.** Medium.

### R18. `compute_cmip5_gmst.py` robustness

- `open_mfdataset` over historical + rcp85 does not drop duplicate or overlapping months; a `drop_duplicates("time")` is needed.
- Partial first and last years are averaged as if they were full years.
- Validation relies on `assert`, which is disabled under `-O`.

The output does not feed the GWL windows: those come from the YAML.

**Confidence.** Medium.

---

## Hygiene

- **H1. README is two lines.** It needs:
  - installation (including the sibling `cordex2energy` repo, which is not on PyPI, and the `-e ../CORDEX2Energy` layout);
  - the Gurobi licence;
  - data availability: TYNDP 2024 workbooks from ENTSO-E, EURO-CORDEX, CERRA, c2e outputs, cmip5/cmip5-ng, mRM routed runoff;
  - `paths.json`;
  - the order of scripts and notebooks;
  - a mapping from figures to notebooks;
  - citation and licence.
- **H2. Dependency declarations do not match the code.**
  - `pyproject` says `requires-python >=3.8` with 3.8–3.11 classifiers. However, `compute_climate_means_gwl.py` evaluates `dict[tuple[str, str], str]` at module level (needs ≥3.9), `env.yml` pins `xarray=2025.12.0` (needs ≥3.11), and `xr.coders.CFDatetimeCoder` needs xarray ≥2025.01.
  - `xarray>=0.19` is far too loose.
  - Declared but never imported: `seaborn`, `cartopy`, `statsmodels`.
  - Used but not declared: `zarr`, `netCDF4`, `pyproj`, `shapely`, `cftime`, `gurobipy`.
  - `env.yml` installs `dask` both via conda and pip, and `pandas` only via pip.
  - The env is named `c2e-analysis`, but the notes and `run_notebook.py` refer to `c2e-recipe`.
- **H3. Dead code in `compute_tyndp_projections_extremes.py`.**
  - The whole NUTS/zone-geometry block (`load_nuts_geodata` … `drop_overlapping_zones`, ≈l.785–809) computes `gdf_tyndp_zones` and never uses it. It adds a NUTS-shapefile data dependency.
  - `_model_tag` is unused.
  - `dask.config.set(num_workers=…, threads_per_worker=16)` and the `distributed.*` keys have no effect: no distributed client is created, and `threads_per_worker` is not a config key.
  - The market profiles are re-read on every run (`_ds_market_for_targets`), even when the baseload comes from cache.
- **H4. Misleading help text and metadata.**
  - The help for `--reservoir_min_storage_fraction` says "Default None", but the default is 0.1.
  - `--use_prosumer_batteries` without `--use_batteries` is silently ignored.
  - Under `system`, the output attributes still record `battery_dispatch_method='heuristic'`, `utility_dispatch_level='zone'` and a 24 h look-ahead (seen in the published CERRA file), none of which apply.
  - The run script says k_cool spans "~8x" while the script help says "~16x".
  - The "M7/M7r" jargon is undefined.
- **H5. Stale docstrings.**
  - Notebooks that do not exist are referenced: `paper_metric_and_target_diagnostics.ipynb`, `analyse_energy_gwl.ipynb`.
  - `compute_energy_means_gwl.py` says it was "adapted from `compute_energy_means.py`", which is now in `archive/`.
  - The `compute_residual_load` docstring says thermal input is 3-hourly; the paper's input is hourly.
  - The `compute_or_load_baseload` docstring names the cache `baseload_{scenario}_{year}.nc`.
- **H6. Logging and validation.** `print` is used for warnings throughout `power_system`, for example the unexpected TYNDP climate years and missing proxies. Use `warnings` or `logging` so they are not lost in batch runs. Scripts use `assert` for data validation.
- **H7. Other loose ends.**
  - The `xr.zeros_like(ds_wind)` fallback for missing CERRA offshore wind has a `tyndp_zone` dimension, not `country`. It would broadcast incorrectly if ever used; the file exists today.
  - `paper_figure_options.ipynb` sits inside the package directory.
  - The memory index and notes reference `scripts/figure_checks`, which `run_notebook.py` uses in place of nbconvert. Document it.
- **H8. Concurrent refactor risk.** `compute_climate_means_gwl.py` and `compute_energy_means_gwl.py` import `extremes_analysis.resolve_gwl_period_definitions` and `analysis.compute_seasonal_statistics`. `compute_european_t_p.py` imports `analysis.load_warming_level_periods`. Re-check these imports after the refactor.

---

# Code review, part 2: findings from the documentation pass over the package

These came out of the module-by-module documentation and naming pass. Nothing was changed to fix
them: every module was verified to produce identical outputs before and after that pass. Ranked by
whether they can touch a published number.

## On the paper's code path

- **P1. S19 draws a stale baseload.** `notebooks/si_note2_system.ipynb` opens
  `tyndp/baseload/baseload_DE_2050.nc` (written 2026-06-07). The paper's run uses the `c2edem`
  demand model, whose baseload is `baseload_DE_2050_c2edem.nc`. They differ by 8 % in total
  energy (verified), so S19 shows the wrong curves. The residual-load runs themselves are not
  affected. Fix: read the `_c2edem` file (a config constant).
- **P2. The storage counterfactual ignores charging.** `extremes_analysis._flex_event_metrics`
  rebuilt the no-storage severity by adding back positive dispatch only. Charging during an event
  (which raised the residual load) was not removed, so the no-storage severity was too high and the
  balanced share was **overstated** (an earlier version of this note said understated). Only
  batteries charge; reservoir dispatch is never negative. Measured on the paper run at +3.0 °C, the
  correction lowers the median balanced share by at most 1.7 percentage points (British Isles and
  Central-East, multi-day events). **Fixed:** the counterfactual now adds back the signed dispatch.
  Still open: events are detected on the residual load *after* storage, so steps and events that
  storage removed completely are not counted, which understates what storage avoided.
- **P3. The last day of run-of-river counts 1/8.** `power_system.preprocess_hydro_ror` upsamples
  daily values with `resample(...).ffill()`, so the series ends at 00:00 of the last day.
- **P4. Heuristic battery rationing counts the current step twice** (`remaining = 1 + sum(...)`).
  Only the heuristic path; the paper uses the annual QP.
- **P5. Missing TYNDP demand hours become 0, not NaN** (`load_tyndp_demand_profiles`).
- **P6. Prosumer PV is aligned to the residual load by position along time**, not by timestamp.
- **P7. Two cache keys are incomplete.** The baseload cache key omits `floor_fraction`,
  `modulate_baseload` and `reference_period` (see also R7).
- **P8. The SI text quotes 1.44 for the cooling-anchor maximum; the data give 1.4348.**
  The test `tests/test_pipeline.py::test_cooling_anchor_maximum_quoted` is marked xfail.

## Off the paper's path (kept methods, robustness)

- TAILS threshold: the knee point uses argmax where a convex tail needs argmin, so the knee is
  always the 0.5 quantile. Its distance (eq. 6) compares bootstrap refits with themselves, never
  with the data. The mapping back to raw units is approximate, and the refit at the optimum is
  unguarded.
- EVT: the significance level is 0.1 in `compute_active_threshold` but 0.05 in the seasonal
  path. `calibrate_seasonal_evt_thresholds` counts NaN in the percentile denominator.
- `compute_flexibility_attenuation_metrics` with seasonal thresholds but no `season_map` raises
  `AttributeError`.
- Season assignment: `compute_multi_day_table` uses the first step's season, while
  `compute_event_season_metrics` uses the majority season.
- `_attribution_shares` falls back to equal shares (100/7 %) when the anomaly is not positive.
- `resolve_gwl_period_definitions` keeps one ensemble member per GCM; the last one wins.
- `peak_gw` is energy per step (GWh), not power. The ramping rate assumes 3 h steps.
- The reservoir QP receives NaN residual load unfilled, and its Gurobi environment is not
  disposed.
- `compute_estimated_generation` drops the capacity of zones without capacity factors, and its
  `unit` attribute says MWh for MW × CF.
- `tables.scenario_table` hard-codes the `n/9` header, and `tables.WINDOW_YEARS` is fixed at 20.
