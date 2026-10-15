# CORDEX2Energy-analysis

Analysis code and figures for *Global warming shifts residual load extremes across timescales in
Europe*. It takes the energy quantities produced by the
[CORDEX2Energy](https://github.com/yasserhaddad/ClimateScenarios4Energy-CH) pipeline (bias-corrected
EURO-CORDEX → wind, solar, hydro and heating/cooling demand), builds the European residual load
under the TYNDP 2024 build-outs, detects its extreme events and draws every figure of the paper.

## Installation

```bash
conda env create -f env.yml          # creates the c2e-analysis environment
conda activate c2e-analysis
pip install -e ../CORDEX2Energy      # the upstream cordex2energy package
pip install -e .
```

`cordex2energy` is not on PyPI: clone
[CORDEX2Energy](https://github.com/yasserhaddad/ClimateScenarios4Energy-CH) next to this repository,
as the `-e ../CORDEX2Energy` line in `env.yml` expects.

### Gurobi

The reservoir and battery dispatch are quadratic programmes solved with
[Gurobi](https://www.gurobi.com); every published number depends on them. The paper used
**Gurobi 13.0.2**, which `env.yml` installs from the `gurobi` conda channel (`gurobipy` from PyPI
works too). A **full licence** is needed: each yearly problem has about 12 000 variables, more than
the size-limited licence bundled with the package allows. Academic licences are free; see the
Gurobi website for how to obtain and install one.

## Data

No data are part of this repository. Point the code at them by copying `paths.example.json` to
`paths.json` (not tracked) and filling in the three directories: the processed data, the disk cache
of intermediate tables (`path_data_analysis`), and the figure output.

The processed data are produced by CORDEX2Energy from these sources:

| Source | Used for |
|---|---|
| EURO-CORDEX EUR-11, RCP8.5 (ESGF) | The 9 regional climate model chains |
| CERRA reanalysis (Copernicus Climate Data Store) | Bias-correction target and reference run |
| CMIP5, annual means regridded to 2.5° (cmip5-ng) | Warming-level windows and Supplementary Figure S5 |
| Routed runoff from the mesoscale routing model | Run-of-river and reservoir inflow |
| ENTSO-E TYNDP 2024 supply and demand scenario workbooks | Installed capacities, demand targets, baseload |
| Natural Earth admin-0 boundaries | Country maps |

## Reproducing the paper

1. **Residual-load runs** (one per chain and for CERRA):
   `scripts/compute_tyndp_projections_extremes.py`, driven by `scripts/run_tyndp_projections_all_models.sh`.
2. **Warming-level means and reference data**: `scripts/compute_climate_means_gwl.py`,
   `scripts/compute_energy_means_gwl.py`, `scripts/compute_cmip5_gmst.py`,
   `scripts/compute_european_t_p.py`.
3. **Figures**: run the four notebooks in `notebooks/`. Expensive intermediate tables are cached on
   disk (`cordex2energy_analysis/figure_cache.py`; set `C2E_FIGCACHE=refresh` to rebuild them).

| Notebook | Figures | Builders |
|---|---|---|
| `paper_figures_publication.ipynb` | Figures 2–8, Table 1 | `figures_main.py` |
| `si_note1_pipeline.ipynb` | Supplementary Figures S1–S16, Figure 1 insets | `figures_pipeline.py` |
| `si_note2_system.ipynb` | Supplementary Figures S17–S22 | `figures_system.py` |
| `si_note3_results.ipynb` | Supplementary Figures S23–S33, Supplementary Table S1 | `figures_results.py` |

Figure 1 (the pipeline schematic) is assembled outside Python from the insets Note 1 draws.

## Checks

```bash
pytest                                        # invariants and the numbers quoted in the paper
python scripts/figure_checks/check_artwork.py # artwork rules: width, fonts, overlaps, panel letters
python scripts/figure_checks/check_crossrefs.py
```

Most tests read the processed data and are skipped without it.

## Package layout

| Module | Contents |
|---|---|
| `config.py` | Paper settings (run tags, warming levels, threshold, zones) and `setup()` |
| `power_system.py` | TYNDP 2024 inputs, residual load, battery and reservoir dispatch |
| `runs.py` | Loading a residual-load run and the event tables built from it |
| `fields.py` | Country-level energy quantities per chain and warming level |
| `tables.py` | Derived tables: zone changes, duration spectrum, storage, scenarios |
| `extremes_events.py`, `extremes_thresholds.py`, `extremes_statistics.py`, `extremes_analysis.py` | Event detection, thresholds and event statistics |
| `analysis.py` | Warming-level windows, sign agreement, ensemble spread |
| `plotting_publication.py`, `plotting_paper.py` | Figure style, panels and map helpers |
| `figures_main.py`, `figures_pipeline.py`, `figures_system.py`, `figures_results.py` | Figure builders |
| `figure_cache.py` | Disk cache of intermediate tables |

## Citation

If you use this code, please cite the paper; the reference will be added here on publication.

## Licence

MIT, see `LICENSE`.
