# CORDEX2Energy paper — briefing document

Working title: **"Warming shrinks Europe's annual surplus and moves its deficit burden from long
winter events to short summer evenings."**

This is a self-contained brief for a writing assistant. Everything below is drawn from the adopted
analysis run and the current main-figures notebook (`paper_main_figures_v2.ipynb`) and pipeline
documentation. Numbers marked **[asserted]** are checked programmatically in the notebook's
verification cell, so they are the safe ones to quote.

---

## 1. Premise

The question is **not** "will climate change reduce Europe's renewable resource?" — it is:

> In a decarbonised European power system (TYNDP 2050), how does progressive warming reshape the
> *residual load* — demand minus variable renewable generation — and in particular its **extremes**,
> once climate-driven heating *and* cooling demand are modelled consistently with the supply side?

Three framing commitments distinguish the study:

1. **Both sides of the balance respond to the same climate.** Most climate-and-energy studies perturb
   supply (wind, solar, hydro) and hold demand fixed or use a coarse temperature response. Here
   demand is generated from the *same* bias-corrected climate fields as generation, with an explicit,
   calibrated, saturating cooling model. Cooling growth is the novel term.
2. **Extremes, not annual means.** The policy-relevant quantity is the frequency, duration, depth and
   timing of deficit events, and how much of them storage can actually absorb.
3. **Warming levels, not dates.** All results are stated at global warming levels (+1.5, +2.0,
   +3.0 °C), with **+1.5 °C as the reference** — i.e. the paper reports what *further* warming does to
   an already-warmed system, not a change from pre-industrial.

The headline reframing that emerged during the analysis and should anchor the narrative: the earlier
version of this work concluded "the mean state improves but the tail worsens." Under the adopted
demand treatment **that tension disappears — the mean state and the extremes agree, zone by zone**.
Warming does not relieve the European system; it *re-composes* demand (winter heating saved, summer
cooling added), shrinks the annual surplus, and redistributes stress geographically and seasonally.

---

## 2. Methods

### 2.1 Climate input

- **9 EURO-CORDEX GCM–RCM chains**, RCP8.5, RCMs with transient aerosols (2 RCMs ×5 driving GCMs:
  CNRM-ALADIN63 and MOHC-HadREM3-GA7-05, driven by CNRM-CM5, HadGEM2-ES, NorESM1-M, MPI-ESM-LR,
  EC-EARTH).
- **Bias-corrected against CERRA reanalysis** by empirical quantile mapping (91-day moving window,
  99 quantiles, additive).
- CERRA itself is carried through the identical pipeline and serves as the **reanalysis reference**
  used to calibrate event thresholds.

### 2.2 Climate → energy conversion (3-hourly, country level)

| Vector | Method |
|---|---|
| **PV** | `pvlib` ModelChain; Erbs decomposition of global → direct/diffuse; SAPM cell temperature (open-rack glass–glass); 300 W module, γ = −0.4 %/°C, 96 % inverter |
| **Wind** | `windpowerlib` power curves; **SWT120/3600 at 90 m hub height** is the paper's turbine (E-126/7580 and SWT142/3150 are sensitivity turbines; mean onshore CF 0.20–0.40 across them, so each requires its own threshold calibration) |
| **Hydro** | Run-of-river from streamflow; reservoir inflow carried separately for dispatch |
| **Demand** | Degree-day model, below |

### 2.3 The demand model (the main methodological contribution)

**Degree days.** Population-weighted national degree days on the **Eurostat/EC definition**:
`HDD = 18 − T` for `T ≤ 15 °C`, `CDD = T − 21` for `T ≥ 24 °C`, applied to the **daily mean**
temperature. This is a deliberate, defensible choice: applying the threshold sub-daily counts warm
afternoons on days whose mean never reaches the base temperature, inflating annual CDD by 1.4× in
Spain, **8.6× in the UK and 75× in Ireland**. Heating is barely affected (≤7 %). *Cooling in cool
climates is almost entirely an artefact of this choice* — which is why the established EC definition
was adopted.

**Own coefficients, fitted to measured European consumption** (rather than transferring
demand.ninja's literature parameters). Per country, regress annual measured energy on annual
population-weighted degree days:

```
E[c, y] = P0 + s[c]·DD_annual[c, y] + β[c]·(y − y0) + ε
```

- Energy: **JRC-IDEES (2023 edition)** final energy consumption, residential (`RES_hh_fec`) +
  tertiary (`SER_hh_fec`); heating is total across carriers with the electric part recovered via the
  JRC electrified share; cooling is essentially all-electric.
- Degree days from **CERRA**, on the same grid as the CORDEX products, so country weights are reused
  unchanged.
- The **time trend β** absorbs efficiency/renovation and AC-adoption drift so it does not contaminate
  the climate slope. β is fitted for *estimation* only and is **never extrapolated**.
- Coverage EU-28; Switzerland, Norway and non-EU gap-filled from proxy neighbours **in per-capita
  space**.

**Cooling gets a different functional form — saturating and sector-split.** Per sector
*j* ∈ {residential, service}:

```
E_j(t) = k_c · [ B_j + P_j · CDD(t) · s_j(CDD_annual(t)) ]
```

- `s(CDD) = 1 − exp(−θ·CDD)` is the **AC adoption share**, evaluated on a **5-year trailing mean** of
  CDD (an air conditioner installed after a hot summer stays installed, so penetration tracks
  sustained climate, not a single year). Adoption is thus driven by *climate*, is **bounded**, and
  differs between a mild and a hot chain — three properties a linear year-ramp lacks.
- `B_j` is a **weather-insensitive baseload** (commercial refrigeration, ventilation). Carrying it
  separately keeps `P` a genuine climate response; forced through the origin that constant lands on
  the degree-day slope, which is unbounded where degree days approach zero (Ireland ~0.04 °C·d/yr
  under Eurostat).
- **Splitting the sectors matters**: services dominate cooling *energy* but are weather-insensitive
  and near saturation; the **residential sector carries the emerging climate signal**. A lumped fit
  mixes the two and can lose the signal entirely.
- AC growth multiplies the **weather term only**, not the refrigeration baseload.
- A counterfactual is available: freeze `s` at each country's anchor climate → cooling still grows
  with degree days but no *new* air conditioning is installed.

**Daily → hourly.** Degree days are daily by construction and are never interpolated to hourly (that
would fabricate diurnal structure the quantity does not have). Demand is upsampled with a **diurnal
profile normalised to mean 1**, separately for heating and cooling, so the **daily total is preserved
exactly**. Output: hourly national heating and cooling demand.

### 2.4 The 2050 system (TYNDP 2024)

- **Demand scenario DE (Distributed Energy) 2050; supply scenario LOW 2050.**
- TYNDP bidding zones mapped to NUTS2 (UK via NUTS1 2016, since the UK was dropped post-Brexit),
  dissolved into zone polygons, then grouped into **6 balancing zones**: Iberia, South-East,
  Central-West, Central-East, Nordic, British Isles.
- Generation = TYNDP installed capacity × our CORDEX capacity factors, per technology. Prosumer and
  utility battery capacities from the same supply sheet.
- Per-country annual electric heating/cooling TWh targets from the `3_DEMAND_OUTPUT` sheet; non-EU
  countries (UK, CH, NO) absent from that EU-27 export get a synthesised target = total demand ×
  thermal share of a proxy (UK→FR maritime, CH→AT alpine, NO→SE Nordic). Western Balkans deliberately
  unproxied.

**Avoiding double-counting the weather.** TYNDP's hourly profiles already contain heating and cooling
for *their* climate year, so theirs must be removed before ours is substituted:

```
baseload = (market + prosumer) − preprocessed CERRA thermal    [floored at 15 % of mean market]
```

Thermal preprocessing: (1) 3 h→1 h upsample + 4-hour centred rolling mean for thermal inertia
(upsampling skipped for the already-hourly product, smoothing kept); (2) COP/loss correction
(1.0 for our already-electric product); (3) **volumetric rescaling** `k = E_target / E_simulated`,
with `E_simulated` the mean annual sum over **1991–2020 only** — anchoring to the reference period
keeps the future thermal trend intact instead of diluting it with the target. Each TYNDP-scenario
country's baseload is then modulated so its annual total equals `TYNDP total − heating − cooling`, and
averaged into a typical-year 8760-hour profile. **Symmetry constraint:** the thermal subtracted here
must come from the same demand model added back in the residual load.

**One asymmetry to defend up front — heating is pinned to the TYNDP target, cooling is not**
(the `freecool` run). Comparing each TYNDP target with the JRC-IDEES series the model is fitted to,
in the same base year (2019): heating median ratio **1.04×**, scattered, 6 of 12 countries below 1 →
*noise*; cooling median **2.03×**, 10 of 12 above 1 → *systematic bias*. TYNDP's "Cooling" is broader
than the space cooling the model fits, and its Italy/Spain ordering is **inverted** relative to every
observed year. This is not cosmetic: `k` multiplies the whole series, so it amplifies the **climate
trend** by the same factor, and `k_cool` spans ~8× across Europe (Italy 3.55, Spain 0.87). Italy's
3.55 drove the South-East zone into **threshold saturation** — 43.7 % of summer timesteps above the
p95 deficit threshold at +3.0 °C, which fused the whole season into a single event and *reversed the
sign* of the sub-daily result. With `k_cool = 1` that falls to 4.7 %. The rescaled-cooling run is
retained as an upper-bracket sensitivity, not discarded.

### 2.5 Residual load and flexibility dispatch

```
residual_load = (baseload + heating + cooling) − (PV + wind_onshore + wind_offshore + hydro_RoR)
```
Demand assembled hourly then resampled to **3-hourly sums** to match generation. **Sign convention:
positive = unmet demand (deficit), negative = surplus.** All in GWh, country level then aggregated to
the 6 zones. A missing technology (offshore wind in a landlocked country) is filled with zero, not
dropped.

**Reservoir hydro** — annual optimisation per country and calendar year (Gurobi QP):
```
minimise Σ_t (residual_load[t] − G[t])²
s.t.  E[t+1] = E[t] + inflow[t] − G[t]/η − spill[t];  E[0] = E[T];
      E_min ≤ E ≤ E_max;  0 ≤ G ≤ P_max;  spill ≥ 0
```
η = 0.9 water-to-wire; storage floor 10 % of capacity. The **quadratic objective makes this
peak-shaving, not arbitrage** — it penalises large deficits disproportionately, which is the desired
behaviour for extremes. Perfect foresight *within* the year, none across years; the cyclicity
constraint prevents draining the reservoir into December.

**Batteries** — the paper's `system` scheme treats prosumer + utility batteries as **one asset**,
dispatched at country level with the same annual-cyclical QP, sequentially after the reservoir.
4-hour duration, 85 % round-trip (one-way √0.85 on charge and discharge).

*A degeneracy that had to be solved explicitly (worth a methods paragraph):* the objective depends
only on `charge − discharge`, so inflating both lowers the state of charge without changing net load —
a free energy-disposal valve that lets the battery absorb surplus, discard it, and absorb again,
never filling. Round-trip losses do **not** prevent this, because `Σ net_load²` penalises surplus and
therefore *rewards* disposal; unconstrained, the relaxation flattens a pure-surplus year to zero with
an arbitrarily small battery. **Fix:** enforce `charge[t]·discharge[t] = 0` by sign-fixing — solve,
clamp each step to charge-only or discharge-only by the sign of `charge − discharge`, re-solve until
the pattern is stable (2–3 passes). Each pass is a plain convex QP, no binaries. The clamped problem
is strictly more constrained than the relaxation, so the result is always **conservative — it can
never overstate the battery's benefit** — and always feasible.

### 2.6 Event definition and analysis framework

- **Warming levels** +1.5 (reference), +2.0, +3.0 °C, implemented as **20-year windows placed per
  chain** from a CMIP5 warming-level table, so chains are compared at equal warming rather than equal
  date. This is what allows 9 chains with different transient sensitivities to be pooled.
- **Events** detected on the post-dispatch residual load with the **Sequent Peak Algorithm (SPA)**
  against a **single, season-independent CERRA p95 threshold per zone**: **[asserted]** 76.9 / 137.2 /
  434.3 / 192.1 / 57.0 / 145.2 GWh per 3 h for Iberia / South-East / Central-West / Central-East /
  Nordic / British Isles.
- The season-independence is deliberate and should be defended in the text: a season-specific p95
  forces ~5 % exceedance in *every* season of the reference climate by construction, which
  re-baselines each season to its own tail and **would erase the winter→summer migration that is the
  paper's thesis** (it also manufactures events in benign seasons). The seasonal-threshold variant is
  retained only as a sensitivity. Where a zone-season has no exceedance of the absolute level, "no
  events" is reported — that is itself a finding.
- Events split by duration: **sub-daily (≤ 24 h)** and **multi-day (> 24 h)**.
- Detector-free companion metrics (**exceedance hours**, **deficit energy**, **exceedance
  probability**) are used wherever the detector could be suspected of saturating — an important
  robustness device, since SPA absorbs sub-threshold dips while a cumulative surplus is outstanding
  and can fuse a season into one event if the exceedance rate rises far above nominal.
- **Storage attenuation** = severity removed by flexibility relative to a counterfactual with the
  dispatch added back, split between battery and reservoir by a **two-player Shapley decomposition**
  (the two shares sum to the combined value exactly, per chain).
- **Ensemble statistics:** all headline numbers are the **median of the 9 chains**; spread shown as
  Tukey fences; **robustness = sign agreement in ≥ 2/3 of chains (≥ 6/9)**.

---

## 3. Key results

All changes are **+1.5 °C → +3.0 °C**, ensemble median of 9 chains, unless stated.

### 3.1 The mean state re-composes and tightens (Figures 2 and 3)

> Message: warming does not relieve the European system. It re-composes demand and shrinks the
> surplus, and it does so unevenly across regions.

1. **Supply weakens slightly and everywhere.** Europe-wide: solar **+1.2 %** (effectively flat), wind
   **−1.7 %** (onshore −1.7, offshore −1.6), hydro **−0.8 %** (run-of-river −3.4 %). No resource
   collapse — but nothing grows either, so the system gets no help from the supply side.
2. **Demand re-composes rather than falls.** Heating **−15.1 %** (428 → 363 TWh yr⁻¹) against cooling
   **+65.5 %** (57 → 94 TWh yr⁻¹). *The new cooling load cancels more than half the heating saving,
   and it arrives in a different season and a different part of the day.*
3. **The annual surplus shrinks by about a quarter**, −183 → **−139 TWh yr⁻¹** (negative = surplus).
4. **And the burden redistributes.** Four of six zones tighten — British Isles **+17.2**
   **[asserted]**, Iberia **+11.9**, Central-West +6.7, South-East +2.5 TWh yr⁻¹ — while Central-East
   (−3.4) and the **Nordic (−7.6) [asserted]** loosen.

**The zone ordering in (4) is exactly the ordering the extremes recover** — this is the paper's
structural argument for reading mean state and tail as one story:

| Zone | Annual surplus change | Extremes |
|---|---|---|
| British Isles | **+17.2** TWh yr⁻¹ (tighter) | multi-day **+52 %** |
| Iberia | **+11.9** (tighter) | sub-daily **+87 %** |
| Central-West | +6.7 | flat, not robust |
| South-East | +2.5 | sub-daily +34 %; JJA multi-day 0 → 0.75 yr⁻¹ |
| Central-East | −3.4 (looser) | multi-day −12 % |
| **Nordic** | **−7.6** (looser — the only one) | multi-day **−29 %** |

> Every zone whose surplus shrinks is a zone where events multiply; the one zone whose surplus grows
> is the one where they decline.

### 3.2 The burden moves from winter to summer without leaving winter (Figure 5)

> Message: winter long events recede while summer events emerge on both timescales.

1. **The flat multi-day total is a cancellation, not a null result.** Europe-wide the multi-day
   deficit-hour burden barely moves (**+1 %** **[asserted]**), but by season it is DJF 1233 → 1172
   h yr⁻¹ (**−5 %**) against JJA 24 → 96 h yr⁻¹ (**+300 %**) **[both asserted]**. *The aggregate hid
   the single largest change in the figure.*
2. **Short events grow in every season, and most in summer**: DJF **+15 %**, MAM **+6 %**, JJA
   **+267 %**, SON **+36 %** **[all asserted]**; sub-daily total **+32 %** **[asserted]**. There is no
   season in which the sub-daily burden falls.
3. **In the tail the change is a summer change.** At the deficit threshold, JJA exceedance becomes
   **×6.4** more likely in Iberia, **×11.2** in the South-East and **×26.5** in Central-East, while
   DJF stays between ×0.9 and ×1.7 in every zone.
4. **Winter still carries most of the burden** — DJF falls from **73 % to 66 %** of all deficit-hours
   (1698 of 2313 → 1672 of 2522 h yr⁻¹). This is a *redistribution, not a reversal*: the summer
   regime is emerging from a very low base.

### 3.3 Short deficits become a southern, summer, evening phenomenon (Figure 6)

> Message: they multiply in exactly the hours photovoltaics cannot reach.

1. **The surge is southern and unanimous.** Iberia **+87 %** and South-East **+34 %** **[both
   asserted]**, 9/9 chains each; the north is flat or easing (Central-West −3 %, Nordic −9 %; British
   Isles +15 % **[asserted]**). No zone outside the south shows a robust increase.
2. **It is a summer phenomenon.** Pooled over the two southern zones, JJA exceedance rises **~×7.8**
   (17.2 → 133.5 h yr⁻¹ **[asserted]**) while DJF is flat (543 → 559 h yr⁻¹ **[asserted]**).
3. **And an evening one.** In summer there is **no exceedance at all between 06 and 12 UTC** — PV
   covers midday outright **[asserted: 0.0 h yr⁻¹ at 09 and 12 UTC]** — and every added hour lands at
   15–21 UTC (15 UTC 3.4 → 32.4; 18 UTC 5.6 → **39.2** h yr⁻¹ **[asserted]**). *The deficit is not a
   solar-resource failure; it is a timing mismatch.*
4. **Storage absorbs most of it, but less as it grows.** Combined attenuation is 86–95 % in four of
   six zones, yet it *falls* precisely where events grow fastest: Iberia **92.7 → 86.7 → 85.9 %**
   across the three warming levels **[asserted, monotonic]**.

### 3.4 Long deficits do not multiply — they move, and flexibility cannot span them (Supplementary Fig. S31)

1. **The count is flat Europe-wide but the map moves.** British Isles **+52 %** **[asserted]** and
   Iberia **+50 %** against Central-East **−12 %** and Nordic **−29 %** **[both asserted]**
   (7/9–8/9 chains). *The Nordic — the one zone whose annual balance loosens in Figure 3 — is also the
   one whose long deficits become rarest.*
2. **The season moves with it.** South-East winter events recede (DJF 4.95 → 4.45 yr⁻¹) while a summer
   regime appears where there was none (**JJA 0.00 → 0.75 yr⁻¹**). *The South-East is therefore not
   wind-limited: it is recomposing.* Driver attribution must be done **within season** — pooling
   confounds this recomposition.
3. **They are long.** Median duration 43–106 h, longest **489 h** (South-East) — one to twenty days,
   against a 4-hour battery. Duration separates *zones* but is indistinguishable across warming
   levels.
4. **Flexibility absorbs about half of what it does sub-daily** — Central-East **36 %** (against 57 %
   sub-daily), British Isles **40 %** (against 70 %) — and the British Isles has **no reservoir at
   all** (0 % share), which is why it is the one zone where *both* regimes grow. Do **not** write
   "least where events are longest": that ordering does not hold. The South-East's multi-day events
   are the longest at 106 h and it still absorbs 72 %, while Central-East's 69 h events absorb 36 %.
   What holds in *every* zone is the two statements Figure 7a is built to show — combined absorption
   falls from short events to long ones, and the battery's share of what is absorbed falls wherever
   there is a reservoir to take over.

### 3.5 Three results that are best stated as null results

**Estimator note, and it matters here.** Every percentage in this section is now the **paired**
median — each chain's own change, then the median across chains — which is the estimator Figure 6a
and Supplementary Fig. S31a annotate. The earlier version of this section mixed paired and unpaired figures inside a
single sentence, which is how the same Iberian result came to be quoted as both −6.8 and −8.4 pp.
Where an unpaired number is genuinely wanted it is now labelled as one.

- **Mitigation does not degrade as the climate warms — except in one zone, for one reason.** The
  attenuation *share* moves by less than 3 pp between +1.5 and +3.0 °C in five of six zones, in both
  duration classes (largest of them Central-East sub-daily at **+2.8 pp**, which *rises*, with 8 of 9
  chains). Iberia is the sole zone that falls materially: **−6.8 pp sub-daily and −3.7 pp multi-day,
  7 of 9 chains each**, and monotonic across the three levels (92.7 → 85.6 → 85.5 % and
  87.4 → 81.5 → 79.4 %). **The burden grows because the events do, not because storage stops
  working** — everywhere but Iberia, the zone whose events multiply fastest.
  *Iberia is not a stray point and needs its own paragraph* (**Supplementary Fig. S32**): the
  reservoir's Shapley share falls **17.0 pp** on short events while the battery's rises **8.2** (both
  paired; −14.5 and +7.2 as a difference of medians), so the battery covers about half of what the
  reservoir stops doing and the total still falls. The battery is not the constraint — its median
  state of charge at event onset is **100 % at all three warming levels**, while the reservoir's runs
  61.5 → 57.8 % and never reaches its own maximum. And it is not simply
  that the events got bigger: conditioning on each event's counterfactual severity, Iberia's
  absorbed share falls in **every** severity quintile while Central-East's, whose events also grow,
  is flat or higher in every one. The transferable claim is that **attenuation degrades where the
  asset carrying it is itself a climate variable.**
- **The European energy storage leaves unbalanced is steady in total and moving in composition.** Long
  events carry about four times the unbalanced severity of short ones (9.4 against 2.2 TWh yr⁻¹ at
  +1.5 °C; earlier drafts quoted 3× these, see the severity-units note) while being roughly a fifth as frequent. Neither class's own absolute change is robust:
  sub-daily grows under every estimator (+14 % paired at 8/9, +18 % unpaired, +28 % as a mean) but
  multi-day is a null under all four (−8 % paired at 6/9, +0.7 % unpaired, +0.5 % paired mean,
  −0.9 % unpaired mean, per-chain changes −33 to +45 %), and so is the two-class total. What *is*
  unanimous is the composition: the multi-day share falls **81 → 77 %, in 9 of 9 chains**
  (−3.6 pp paired, −4.1 both unpaired and as a mean). This is Figure 7b, and it is why that panel
  annotates the composition rather than either class's change.
- **Absolute flexibility deployment grows but restates the event count** — annual energy deployed
  tracks event frequency nearly one-for-one (Iberia +64 % against **+118 %** more sub-daily events;
  British Isles +18 % against +15 %; Central-East +1 % against +4 %).

**Two conventions the text must not slip on.** (i) The events counted in Figure 6 and Supplementary Fig. S31 are
**post-dispatch residuals** — what is left after the reservoir and battery have already run — not
raw shortfalls, so "storage absorbs 86 % of the severity" and "there are 24.6 events per year" are
statements about the same system, not competing ones. (ii) Iberia's sub-daily count is
**13.1 → 24.6 events per year, +118 %, 9 of 9 chains** on the paired estimator. The +87 % that
appeared in earlier drafts is the unpaired reading of the same table and the +139 % is the ensemble
*mean*; likewise the reservoir energy per event is **165 → 114 GWh** per chain, against 149 → 109
pooled over all events. Quote one estimator per sentence.

### 3.6 Results deliberately demoted, with the evidence for demoting them

Useful for the writing assistant, both to avoid over-claiming and because each is a defensible
"we checked this" paragraph:

- **Sub-daily severity is not an independent result.** Severity ≈ count × depth per event; only
  Iberia's per-event depth moves (**+24 %** against a paired count rise of **+118 %**), every other
  zone −11 % to +2 %. So "they deepen faster than they multiply" is out of the title as well as the
  figure.
- **Multi-day duration does not vary with warming** (separates zones, not warming levels).
- **A diurnal panel normalised by the threshold was cut** because the ratio normalises away the
  growth and put +1.5 °C above +3.0 °C. Replaced by plain **exceedance hours by hour of day** — a
  detector-free count, which orders correctly.
- **Per-capita demand change maps are unnecessary**: population enters the demand model as a
  time-invariant per-country constant folded into the coefficients, and the TYNDP rescaling factor `k`
  is likewise a per-country constant, so both cancel exactly in any ratio — the change columns are
  already per-capita-invariant. Only the absolute column is population-inflated.

---

## 4. Figure inventory

**Conventions after the September review:** type 7 pt (ticks, legends), 7.5 pt (axis labels), 8 pt
(titles, letters); seasons named in the artwork (winter, spring, summer, autumn) with the acronym once
in the caption; every warming-level key titled "Global warming level"; every map in the equal-area
ETRS89-LAEA projection (EPSG:3035) at true aspect.

Eight main figures and thirty-three supplementary ones, numbered sequentially with no suffixes and no
gaps. The authority for any wording is the caption markdown in the notebooks; each PDF also stamps
the panel letters it actually drew into its own `Keywords`, so a caption can be checked against the
artwork rather than against the code that made it.

### Main text — `notebooks/paper_figures_publication.ipynb` → `figs/paper_publication/`

| | Title | Panels |
|---|---|---|
| **1** | The CORDEX2Energy conversion pipeline | External schematic (bias correction → conversion → zonal aggregation → residual load), with three inset panels of Spanish cooling demand rendered by `si_note1_pipeline.ipynb`. |
| **2** | Warming weakens renewable supply slightly and unevenly across Europe | Solar, wind and hydro, three rows, +1.5 °C level beside Δ +2.0 / Δ +3.0 °C; change columns `RdBu_r`, red = more generation. Split from the old six-row map figure (219 mm, over the page) on 2026-10-10. |
| **3** | Warming moves demand from heating to cooling and reshapes the residual load | Heating, cooling and residual load, same layout as Figure 2; the residual-load *level* column is `BrBG_r`, teal for surplus and brown for deficit, its change absolute in TWh yr⁻¹. |
| **4** | Change in Europe-wide energy quantities between +1.5 °C and +3.0 °C | **a** annual, **b** winter, **c** summer; per quantity, summed over the 35 modelled countries, the IQR across chains as a darker box, min–max as paler extensions, the median as a dark line. Added after internal review. Annually the residual-load rise is **supply**-driven (wind −63 TWh/yr; heating −68 and cooling +34 largely cancel); in summer it is **demand**-driven (cooling; ΔRL +43, 9/9 chains). Zonal breakdown: S24. |
| **5** | The deficit burden moves from winter to summer | **a** the six balancing zones; **b** southern survival curves, JJA over DJF; **c** events by duration bin; **d** severity by duration bin. |
| **6** | Short evening and overnight deficits multiply in Southern Europe | **a** sub-daily events per year by zone and level; **b** change in events by season and zone; **c** exceedance hours by hour of day, southern zones, JJA over DJF; **d** evening and overnight drivers, southern zones. All four panels equal width. |
| **7** | Storage absorbs less as events lengthen, and most of what it leaves is long events | **a** absorbed share by zone and duration class; **b** absorbed and unabsorbed severity, stacked, by duration bin. |
| **8** | A full 2050 build-out removes almost all of the deficit, and halves what is left of the warming response | **a** sub-daily and **b** multi-day event counts, LOW/2050 against HIGH/2050, as grouped bars: per zone one pair per build-out, light = +1.5 °C, dark = +3.0 °C. |

### Supplementary — three Supplementary Notes, S1–S33

Restructured 2026-09-03. The three `paper_si_{methods,inputs,results}_publication.ipynb` notebooks
are archived under `notebooks/archive/` with their figure directories renamed `figs/archive_*`; the
numbers below are the CURRENT ones, taken from the notes' own captions.

**Note 1 — the conversion pipeline**, `notebooks/si_note1_pipeline.ipynb` → `figs/si_note1_pipeline/`

**S1**
the driving GCMs and the warming-level windows they define · **S2** change in European precipitation
and runoff against warming, by season · **S3** the bias-corrected EURO-CORDEX variables, annual mean
· **S4**–**S7** the same five variables in winter, spring, summer and autumn, one figure per season (DJF/MAM/JJA/SON) · **S8** total runoff
against the channel discharge it is routed into, at +1.5 °C · **S9** where the demand model's error
is and whether it is in the right years · **S10** the fitted air-conditioning adoption share and how
sharply its curvature is determined · then one figure per conversion step: **S11** heating demand ·
**S12** cooling demand · **S13** solar power · **S14** wind power · **S15** run-of-river generation ·
**S16** reservoir inflow.

**Note 2 — the power system**, `notebooks/si_note2_system.ipynb` → `figs/si_note2_system/`

**S17** the TYNDP-2024 LOW/2050 build-out, per country and per zone · **S18** HIGH/2050 against
LOW/2050 · **S19** baseload demand under the DE/2050 scenario · **S20** reservoir dispatch over the
year · **S21** reservoir storage over the year · **S22** battery throughput by zone.

**Note 3 — the results**, `notebooks/si_note3_results.ipynb` → `figs/si_note3_results/`

**S23** residual load by season · **S24** Figure 4 resolved by balancing zone, closing
exactly on the Europe-wide change · **S25** CORDEX against CERRA in energy space, reference period ·
**S26** the season-specific CERRA p95 deficit levels the paper does *not* use · **S27** the paper's
deficit threshold inside the HIGH/2050 build-out · **S28** exceedance curves of the zonal residual
load, for every balancing zone and every season · **S29** Figure 5c/d resolved by balancing zone
(Iberia +12.5 and South-East +8.3 of the Europe-wide +21.8 sub-daily events/yr) · **S30** exceedance
by hour of day and its driver decomposition, for each balancing zone · **S31** multi-day extreme
events: frequency, seasonal decomposition and drivers · **S32** why Iberian attenuation degrades
under warming · **S33** extreme events that survive the HIGH/2050 build-out, scored on the LOW/2050
threshold.

Two of these carry the load the main text used to: **S30** is Figure 6's second row repeated for
every zone (and absorbs the earlier northern-drivers figure, which existed only because Figure 6d
was cut to the south), and **S31** is the multi-day counterpart of Figure 6, drawn by the same
builder with the same duration-class argument. The main text's deep dive is sub-daily only, because
that is where the change is: on Figure 5c's duration spectrum the sub-daily bins add +23.85 events
per year Europe-wide against +0.45 for the multi-day ones, and on the detector's own duration
classes (S31) it is +19.7 against +0.85. Two counts of the same asymmetry — quote one per sentence.

**The figures the main-text captions cite**, checked by meaning rather than by number: **S18** for
the LOW-against-HIGH capacity difference (Figure 8), **S24** for Figure 4 by balancing zone, **S28**
for the balancing zones and the shoulder seasons Figure 5b omits, **S30** for the four zones Figure
6d omits, **S33** for what survives the HIGH build-out, and **S29** for Figure 5c/d by zone.

---

## 5. Caveats, sensitivities and open items

**Robust across turbine choice.** All three turbines were run. Every sign is preserved; the climate
signal is nearly turbine-independent, because each turbine calibrates its own CERRA p95 threshold,
which absorbs the level shift. The rank ordering and the north–south dipole are unaffected. **But
magnitudes are turbine-sensitive for the demand-driven result** (Iberia sub-daily ranges roughly
+60 % to +169 % across turbines) — do not quote a bare percentage for that one without the range. The
British-Isles multi-day increase strengthens monotonically with turbine specific power, making it the
most robust of the claims. SWT120/3600 was chosen because its specific power (318 W/m²) and capacity
factors (0.26 onshore / 0.51 offshore) sit closest to the real EU fleet and a plausible 2050 machine.

**Detector sensitivity is real and must be acknowledged.** The SPA detector absorbs sub-threshold dips
while a cumulative surplus is outstanding; if the exceedance rate rises far above nominal (above ~2×,
i.e. >10 % for p95) the sub-daily/multi-day split stops being interpretable, because a whole season
can fuse into one event and the sub-daily count can *fall* while the underlying stress rises. This is
exactly what the discarded cooling-rescaled run did to the South-East. The adopted `freecool` run
keeps the South-East summer exceedance rate at 4.7 %, inside the interpretable range. Recommended
framing: report the detector-based counts alongside the detector-free exceedance hours and deficit
energy, and state the exceedance-rate check as a validity condition.

**Known weak points to flag honestly:**
- UK cooling remains the most fragile country-level input — the through-origin, non-saturating form
  combined with a hot-window coefficient extrapolates hard on rare hot days where real UK load is
  capped by low AC penetration. Treat British Isles magnitudes with more caution than signs.
- The AC-adoption law is a modelling choice: matching a historically time/income-driven adoption rate
  with a CDD-driven curve implies adoption saturates comparatively early. The historical European
  cooling surge is largely technology/behaviour diffusion, only loosely income-linked and not well
  explained by CDD alone — no single saturation channel reproduces it. A base-18 °C CDD for the
  saturation term would relax the climate gating and is the natural refinement.
- The TYNDP cooling target is systematically ~2× the JRC space-cooling the model fits, which is why
  `k_cool = 1` was adopted; the rescaled run is the upper bracket.

**Open items:** (i) the contested panel choice; (ii) the SI notebooks still point at the previous run
tag — they derive three sibling configurations (a HIGH/2040 supply scenario and two alternative
turbines) that have no `freecool` equivalent on disk, so repointing them requires roughly three more
ensembles; until then those SI panels live in the main-figures notebook.

---

## 6. Terminology and conventions to keep consistent in the text

- **"extreme event", not "deficit event".** The paper's title is "Global warming shifts residual
  load **extremes** across timescales in Europe", and the figure titles and captions were aligned to
  it. Define it once on first use — *extreme events are spells in which the residual load exceeds
  the CERRA p95 threshold* — so that "extreme" is not left doing definitional work the older word
  did implicitly. **"deficit" survives where it names the STATE rather than the object**: the
  *deficit threshold*, the *deficit tail*, a country in *surplus or deficit*. Those are correct and
  should not be swept.
- **Panel titles name the quantity; the y-label carries the unit; a panel showing a change names
  the two warming levels it compares.** Titles do not assert findings — a title cannot carry "nine
  of nine chains" and should not imply it, so the findings live in the captions with their
  agreement counts.
- **Two metrics, one job each.** *Counts* (events per year) answer how often; *severity* answers how
  much energy. They are not interchangeable and the paper says so: sub-daily events are ~83 % of all
  events but only ~23 % of all severity. Event *duration* is the classifier and the x axis, not a
  result; the residual-load *level* belongs to Figure 3 alone.
- **Every percentage is the paired median** — each chain's own change, then the median across
  chains — with the number of chains agreeing on the sign beside it. Where an unpaired figure is
  wanted, label it as one. See §3.5.

| Term | Meaning |
|---|---|
| **Residual load** | (baseload + heating + cooling) − (PV + onshore + offshore + hydro RoR); **positive = deficit, negative = surplus** |
| **Reference** | **+1.5 °C**, not present day and not pre-industrial |
| **Zones** | Iberia, South-East, Central-West, Central-East, Nordic, British Isles (6 TYNDP balancing zones) |
| **Sub-daily / multi-day** | events ≤ 24 h / > 24 h |
| **Deficit threshold** | CERRA p95 of positive residual load, per zone, **season-independent** |
| **Robust** | sign agreement in ≥ 6 of 9 chains |
| **Ensemble value** | median of the 9 chains |
| **"the south"** | Iberia + South-East pooled |
| `freecool` | the adopted run: heating pinned to the TYNDP annual target, cooling not (`k_cool = 1`) |


## Severity units (corrected 2026-10-01)

The residual-load and storage series are **energy per 3-hourly step** (GWh per 3 h). Event severity
is the plain sum of the per-step excess above the threshold. Until October 2026 the library also
multiplied that sum by the 3 h step length, which overstated every **absolute** severity and every
deployed-storage energy by exactly 3×. Percentages (balanced/absorbed shares, attenuation, Shapley
split, % changes) were unaffected, and so were event detection, counts and durations. Fixed at the
source (`extremes_events.py`, `extremes_analysis.py`); Figure 5d, Figure 7b and S29b now show the
corrected TWh values.
