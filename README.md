# Code for: Global Estimation of Bankfull River Discharge Reveals Distinct Flood Recurrences Across Climate Zones

This repository contains the code used to reproduce the main results and figures in:
Global Estimation of Bankfull River Discharge Reveals Distinct Flood Recurrences Across Climate Zones

---

## 1. Requirements

- Python [Random forest]
- R [Flood frequency analysis]
- Required packages (see `environment/randomforest.yml`, environment/FFA_POT.yml)

To set up the environment:
conda env create -f environment/filename.yml
conda activate [env_name]

---

## 2. Data

To train and evaluate the Random Forest model:
- Observed bankfull discharge
- Predictors at each of the above observed bankfull discharge sites

To make predictions of global rivers:
- Trained model from the above step, trained.joblib
- Predictors for global rivers
  
To analyze the return period of the bankfull discharge data and generate the 2-year return period of floods:
- Daily streamflow records matched to the observed or estimated bankfull discharge

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



