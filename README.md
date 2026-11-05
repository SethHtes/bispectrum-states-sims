# "Bicoherence States from QPO--Accretion Flow Coupling in Black Hole X-ray Binaries": numerical simulation and figure code

This repository holds the simulation code and the notebook that make the simulated-light-curve figures in the paper (the web / cross / hypotenuse / point pattern maps and the (η, 𝒬) pattern planes).

## Paper information

If using any code in this repository, please cite:

S. R. Larner 2026, submitted to ApJ 

(TODO: update with arXiv link and published copy link when available)

## Contents

| File | What it is |
| --- | --- |
| `sim_lightcurve.py` | The light-curve model: QPO × red-noise flow, the quadratic locked term, slow dilution, Poisson counts. Also computes the bispectrum, bicoherence, biphase and power spectra, and draws the maps. |
| `sim_dQ_grid.py` | Runs a grid of simulations over QPO amplitude `a1` and dilution rms, classifies each run's pattern, and draws the (η, 𝒬) plane. Imports `sim_lightcurve`. |
| `paper_figs.ipynb` | Builds every figure below. |
| `sim_dQ_grid_extra_dense.json`, `sim_dQ_grid_extra_dense_web.json` | Saved results of the two 31 × 36 grids (1116 runs each) used by the (η, 𝒬) figures. |
| `figures/` | The figure PDFs which are used in the paper. |

## Figures

| Notebook cell | Output |
| --- | --- |
| 2 | `figures/patterns_9panel.pdf`: web, cross and hypotenuse rows (b², biphase, power spectrum) |
| 3 | `figures/patterns_12panel.pdf`: the same plus the point state |
| 5 | `figures/eta_Q_grid.pdf`: pattern in the (η, 𝒬) plane |
| 6 | `figures/eta_Q_grid_web.pdf`: the same plane on a WEB-matched base |

## Running

```bash
python -m pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace paper_figs.ipynb
```

Or open `paper_figs.ipynb` in Jupyter and run all cells. The notebook must be run
from this directory.

If the json files are available, the notebook simply loads them to make the (η, 𝒬) figures. Otherwise, regenerating the extra-dense grids is expensive. To force regenerate, run

```bash
python sim_dQ_grid.py --extra-dense            # writes sim_dQ_grid_extra_dense.json
```

(`extra_dense_web` uses a different base parameter set and is only run from the notebook.)

## Tested with

Python 3.12.1, NumPy 2.4.2, Matplotlib 3.10.9, nbconvert 7.16.2, ipykernel 6.29.3
