# Code for: [Paper Title]

This repository contains the code used to reproduce the main results and figures in:
Global Estimation of Bankfull River Discharge Reveals Distinct Flood Recurrences Across Climate Zones

---

## 1. Requirements

- Python [Random forest]
- R [Flood frequency analysis]
- Required packages (see `environment/randomforest.yml`, environment/FFA_POT.yml)

To set up the environment:
conda env create -f yml file
conda activate [env_name]

---

## 2. Data

- Input data sources are described in Supplementary Text S1 of the paper.
- Where available, links to original data sources are provided in the Supplementary.

---

## 3. How to run

Run the scripts in the following order:

1. Train and evaluate model
python scripts/RandomForest.py

2. Analysis the return period of bankfull discharge and 2-year return period of floods

R scripts/FFA_POT.R

---

## 4. Outputs

examples/

---

## 5. Reproducibility notes

- Random seed is controlled via `random_state`
- Results are averaged over multiple runs (`n_run = X`)
- Full workflow is described in the Methods section of the paper

---

## 6. Contact

For questions, please contact: [yinxue.liu@ouce.ox.ac.uk]



