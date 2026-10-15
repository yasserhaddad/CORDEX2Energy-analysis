# The CORDEX2Energy pipeline

## Slide 1 - Pipeline overview

**From climate model output to residual-load extremes**

* **9 EURO-CORDEX GCM–RCM chains**, RCP8.5, RCMs with transient aerosols.
* Bias-corrected against **CERRA** by empirical quantile mapping (91-day moving window, 99
  quantiles, additive) — `data_processing/bias_correction_xclim.py`.
* Climate → energy conversion, all at **3-hourly, country level**:
  * **PV** — `pvlib` ModelChain: Erbs decomposition of global → direct/diffuse, SAPM cell
    temperature (open-rack glass–glass), 300 W module, γ = −0.4 %/°C, 96 % inverter.
  * **Wind** — `windpowerlib` power curves; **SWT120/3600 at 90 m hub height** is the paper's
    default (E-126/7580 and SWT142/3150 are the sensitivity turbines; mean onshore CF ranges
    0.20–0.40 across them, so each needs its own threshold calibration).
  * **Hydro** — run-of-river from streamflow; reservoir inflow carried separately.
  * **Demand** — the degree-day model (slides 2–5).
* Aggregated to **6 TYNDP balancing zones**: Nordic, British Isles, Central-West, Central-East,
  Iberia, South-East.
* Analysed at **global warming levels** — +1.5 °C (reference), +2.0, +3.0 — as **20-year windows
  placed per chain**, so chains are compared at equal warming rather than equal date.

> Speaker note: the GWL framing is what lets 9 chains with different transient sensitivities be
> pooled; the reference is +1.5 °C, not present day.

---

## Slide 2 - The new demand model — degree days

**Population-weighted national degree days, Eurostat definition**

* `HDD = 18 − T` where `T ≤ 15 °C`;  `CDD = T − 21` where `T ≥ 24 °C` (the Eurostat model).
* Applied to the **daily mean temperature** (Eurostat/EC definition), *not* sub-daily.
* Population weighting: fine population raster coarsened onto the climate grid, Natural Earth
  country masks, weighted mean per country.

**Why the daily mean matters — this is a real choice, not a detail**

Applying the threshold sub-daily counts warm afternoons on days whose *mean* never reaches the base
temperature. On CERRA 1991–2020 the annual CDD ratio (sub-daily ÷ Eurostat) is:

| Spain | UK | Ireland |
|---|---|---|
| **1.4×** | **8.6×** | **75×** |

Heating is barely affected (≤ 7 %) — most of the year the daily mean is well below the heating base temperature, so the threshold rarely binds. **Cooling in cool climates is entirely an artefact of this choice.**

*Source: `conversion_energy/demand.py` (`compute_cdd_hdd_xr`), `demand_model.py`.*

---

## Slide 3 - The new demand model — our own coefficients

**We fit the coefficients rather than adopting demand.ninja's literature parameters**

Regress **annual measured energy** on **annual population-weighted degree days**, per country:

```
E_heat[c, y] = P0 + s_heat[c] · HDD_annual[c, y] + β[c] · (y − y0) + ε
```

* **Energy** — JRC-IDEES final energy consumption, residential (`RES_hh_fec`) + tertiary
  (`SER_hh_fec`), ktoe → GWh (× 11.63). Heating is total across carriers; the electric part is
  recovered via the JRC electrified share. Cooling is essentially all-electric.
* **Degree days** — CERRA reanalysis, same grid as the CORDEX products, so the country weights are
  reused unchanged.
* **Units** — slope → `P = slope / 24` [GW per °C], so `P · HDD_daily` is GW.
* **Time trend `β`** absorbs efficiency/renovation drift so it does not contaminate the climate slope.
* Coverage EU-28; Switzerland, Norway and non-EU are gap-filled from proxy neighbours **in
  per-capita space**, so no population rescaling is needed.

> Speaker note: this is the main methodological change — the demand response is now calibrated to
> European measured consumption rather than transferred from a global parameterisation.

*Source: `conversion_energy/demand_coefficients.py`.*

---

## Slide 4 - Cooling needs a different model — saturating and sector-split

**Cooling is not a linear degree-day response**

Per sector *j* ∈ {residential, service}:

```
E_j(t) = k_c · [ B_j  +  P_j · CDD(t) · s_j(CDD_annual(t)) ]
```

* `s(CDD) = 1 − exp(−θ·CDD)` — the **AC adoption share**, evaluated on a **5-year trailing mean**:
  an air-conditioner installed after a hot summer stays installed, so penetration tracks the
  sustained climate, not a single year.
* `B_j` — a **weather-insensitive baseload** (commercial refrigeration, ventilation).
* `P_j` — the weather coefficient (population already folded in).
* `k_c` — optional level anchor to the observed JRC benchmark.

**Three reasons the structure is what it is**

1. **Splitting the sectors** — services dominate cooling *energy* but are weather-insensitive and
   near saturation; the residential sector carries the emerging climate signal. A lumped fit mixes
   the two and can lose the signal entirely.
2. **Carrying `B` separately** keeps `P` a genuine climate response. Forced through the origin, that
   constant lands on the degree-day slope, which is unbounded where degree days approach zero —
   Ireland has ~0.04 °C·d/yr under the Eurostat definition.
3. **AC growth multiplies the weather term only** — adoption scales the cooling response, not the
   refrigeration load.

**Counterfactual available:** freeze `s` at each country's anchor climate → cooling still grows with
warming through the degree days, but no *new* air-conditioning is installed. (A refitted linear form
is *not* a valid counterfactual: pooled across countries its slope is anchored by the already-
saturated warm ones.)

*Source: `demand_model.py` (`cooling_demand`, `cooling_adoption_factor`).*

---

## Slide 5 - Daily → hourly

* Degree days are **daily by construction** and are never interpolated to hourly — that would
  fabricate a diurnal structure the quantity does not have.
* Demand is upsampled with a **diurnal profile normalised to mean 1**, separately for heating and
  cooling, which guarantees the **daily total is preserved exactly**.
* Output: hourly national heating and cooling demand, `energy/europe/demand/<chain>_<scenario>.nc`.

---

## Slide 6 - TYNDP projections

**TYNDP 2024 supplies the 2050 system; the climate supplies the weather**

* **Demand scenario DE (Distributed Energy) 2050**; **supply scenario LOW 2050**.
* **Geography** — TYNDP bidding zones mapped to NUTS2 (UK via NUTS1 2016, since the UK was dropped
  after Brexit), dissolved into zone polygons, then grouped into the 6 balancing zones.
* **Generation** — TYNDP installed capacities × our CORDEX capacity factors, per technology.
* **Storage** — prosumer and utility battery capacities read from the same supply sheet.
* **Demand** — hourly market and prosumer profiles, per country, per TYNDP climate year.
* **Thermal targets** — per-country annual electric heating/cooling TWh read from the
  `3_DEMAND_OUTPUT` sheet. Non-EU countries (UK, CH, NO) are absent from that EU-27 export, so their
  target is synthesised as *total demand × thermal share of a proxy* — UK→FR (maritime, comparable
  stock), CH→AT (alpine), NO→SE (Nordic). The Western Balkans are deliberately left unproxied.

*Source: `cordex2energy/conversion_energy/tyndp.py` (`build_tyndp_nuts_mapping`) and
`cordex2energy_analysis/power_system.py` (`build_thermal_targets`, `compute_estimated_generation`).*

---

## Slide 7 - Baseload — avoiding double-counting the weather

**The problem:** TYNDP's hourly profiles already contain heating and cooling, for *their* climate
year. We want to substitute *our* climate-driven thermal demand — so theirs must come out first.

```
baseload = (market + prosumer) − preprocessed CERRA thermal      [floored at 15 % of mean market]
```

**Thermal preprocessing — three steps**

1. **Upsample + smooth** — 3 h → 1 h cubic spline, then a 4-hour centred rolling mean (thermal
   inertia). *Skipped for the already-hourly `cordex2energy_demand` product; the smoothing is kept.*
2. **COP / loss correction** — flat 1.176 (= 1/0.85, ENTSO-E DFT). *Set to 1.0 for our
   already-electric product.*
3. **Volumetric rescaling** — `k = E_target / E_simulated`, with `E_simulated` the mean annual sum
   over **1991–2020 only**. Anchoring to the reference period keeps the future thermal trend intact
   instead of diluting it with the target.

**Baseload modulation** — each TYNDP-scenario country's baseload is then scaled so its annual total
equals `TYNDP total − heating − cooling`, and re-clamped to the floor. Finally averaged into a
**typical-year profile** (8760 hour-of-year) across TYNDP climate years.

**Symmetry that must hold:** the thermal subtracted here is from the *same demand model* added back
in the residual load — otherwise the baseload carries a residue of the other model.

---

## Slide 8 - One asymmetry worth defending up front

**Heating is pinned to the TYNDP target. Cooling is not.** (`--no-cooling-target`, the `freecool` tag.)

Comparing each TYNDP target with the JRC-IDEES series the model is fitted to, in the same base year
(2019):

| | TYNDP / JRC, median | character |
|---|---|---|
| **Heating** | **1.04×** (1.08× aggregate) | scattered — 6 of 12 countries below 1 → **noise** |
| **Cooling** | **2.03×** (1.71× aggregate) | biased — 10 of 12 above 1 → **systematic** |

TYNDP's "Cooling" is systematically broader than the space cooling the model fits, and its
Italy/Spain ordering is **inverted** relative to every observed year.

**Why this is not cosmetic:** `k` multiplies the whole series, so it amplifies the **climate trend**
by the same factor. `k_cool` spans ~8× across Europe (Italy 3.55, Spain 0.87). Italy's 3.55 pushed
the South-East zone into **threshold saturation** — 43.7 % of summer timesteps above the p95
deficit threshold at +3.0 °C, which fused the whole season into one event and *reversed* the sign of
the sub-daily result. With `k_cool = 1` that falls to 4.7 %.

> Speaker note: the rescaled-cooling run is retained as the upper-bracket sensitivity, not discarded.

---

## Slide 9 - Residual load

```
residual_load = (baseload + heating + cooling) − (PV + wind_onshore + wind_offshore + hydro_RoR)
```

* Demand assembled at **1-hourly**, then resampled to **3-hourly sums** to match the generation.
* **Sign convention: positive = unmet demand (deficit); negative = surplus.**
* Country alignment by intersection; a missing technology (offshore wind in a landlocked country) is
  filled with zero, not dropped.
* All quantities in GWh. Country level → aggregated to the 6 balancing zones.

*Source: `cordex2energy_analysis/power_system.py` (`compute_residual_load`), driver
`scripts/compute_tyndp_projections_extremes.py`.*

---

## Slide 10 - Reservoir hydro dispatch

**An annual optimisation, per country and calendar year (Gurobi QP)**

```
minimise   Σ_t (residual_load[t] − G[t])²
subject to E[t+1] = E[t] + inflow[t] − G[t]/η − spill[t]      (water balance)
           E[0] = E[T]                                        (annual cyclicity)
           E_min ≤ E ≤ E_max,   0 ≤ G ≤ P_max,   spill ≥ 0
```

* **η = 0.9** turbine (water-to-wire): generating `G` electrical GWh draws `G/η` from storage.
* **Storage floor = 10 %** of capacity.
* Quadratic objective ⇒ **peak-shaving**, not arbitrage: it penalises large deficits
  disproportionately, which is the behaviour we want for extremes.
* Perfect foresight **within the year**, none across years — the cyclicity constraint prevents the
  optimiser from draining the reservoir into December.

---

## Slide 11 - Battery dispatch

**The paper's scheme — `system`:** prosumer + utility batteries are treated as **one asset**,
dispatched at **country level** with the **same annual-cyclical QP as the reservoir**, sequentially
after it. 4-hour duration, 85 % round-trip (one-way = √0.85, applied on charge and on discharge).

**A subtlety that had to be solved explicitly**

The objective depends only on `charge − discharge`, so the two are individually degenerate: inflating
both lowers the state of charge **without changing net load**. That is a free energy-disposal valve —
the battery absorbs surplus, discards it, and absorbs again, never filling up. Round-trip losses do
*not* prevent it, because `Σ net_load²` penalises surplus and therefore *rewards* disposal.
Unconstrained, the relaxation flattens a pure-surplus year to zero with an arbitrarily small battery.

**Fix:** enforce `charge[t] · discharge[t] = 0` by **sign-fixing** — solve, clamp each step to
charge-only or discharge-only by the sign of `charge − discharge`, re-solve until the pattern is
stable (typically 2–3 passes). Each pass is a plain convex QP, no binaries. The clamped problem is
strictly more constrained than the relaxation, so the answer is always **conservative — it can never
overstate the battery's benefit** — and always feasible.

**Alternative scheme (`market`, not used in the paper):** prosumer batteries greedy and PV-aware;
utility batteries on a **rolling day-ahead QP** — solve over `commit + lookahead` hours, commit the
first `commit_h`, carry the SOC forward. That models merchant scheduling with limited foresight;
`system` models a planner minimising net-load events.

---

## Slide 12 - What the dispatch is measured against

* **Events** are detected on the post-dispatch residual load with the **Sequent Peak Algorithm**
  against a **single, season-independent CERRA p95 threshold per zone** (76.9 / 137.2 / 434.3 /
  192.1 / 57.0 / 145.2 GWh per 3 h for Iberia / South-East / Central-West / Central-East / Nordic /
  British Isles).
* Split by duration into **sub-daily (≤ 24 h)** and **multi-day (> 24 h)**.
* **Attenuation** = the severity storage removes, relative to a counterfactual with the dispatch
  added back, split between battery and reservoir by a **two-player Shapley decomposition** (the two
  shares sum to the combined value exactly, per chain).

> Speaker note: the threshold is deliberately season-independent — a season-specific p95 forces 5 %
> exceedance in every season by construction and would erase the winter→summer migration that is the
> paper's thesis.
