# Code for: Global Estimation of Bankfull River Discharge Reveals Distinct Flood Recurrences Across Climate Zones

This repository contains the code used to reproduce the main results and figures in:
Global Estimation of Bankfull River Discharge Reveals Distinct Flood Recurrences Across Climate Zones

---

## 1. Requirements

- Python [Random forest]
- R [Flood frequency analysis]
- Required packages (see `environment/randomforest.yml`, `environment/FFA_POT.yml`)

To set up the environment:  

conda env create -f environment/filename.yml  

conda activate [env_name]

---

## 2. Data Requirement

To train and evaluate the Random Forest model:
- Observed bankfull discharge
- Predictors at each of the above observed bankfull discharge sites
  
To analyze the return period of the bankfull discharge data and generate the 2-year return period of floods:
- Daily streamflow records matched to the observed or estimated bankfull discharge

---

## 3. How to run

Run the scripts in the following order:

1. Train and evaluate model
   
`python scripts/RandomForest.py --station_qbf=RF/Input/observed_qbf.csv --station_predictors=RF/Input/station_predictors.csv --koppen_color=RF/Input/Koppenlabels_colors.csv --output=RF/Output/trained_model.joblib --output_performance=RF/Output/model_performance.csv --ouput_csv=RF/Output/qbf_train_test.csv --n_run=1`

2. Analysis the return period of bankfull discharge and 2-year return period of floods

`R scripts/FFA_POT.R --station_info examples/FFA-POT/Input/GRITv06_stations_QBFobs_USGS_05453100.csv --dir_path examples/FFA-POT/Input/daily_streamflow --loess-span 0.5 --output Output/USGS_05453100.csv `

---

## 4. Outputs

examples/ include reproducible working examples for train and evaluate model and analysis the return period of bankfull discharge and 2-year return period of floods from daily streamflow records

---

## 5. Reproducibility notes
Using the data provided in the examples/RF/Input with the code above (scripts/RandomForest.py) should reproduce the results in examples/RF/Output  

Using the data provided in the examples/FFA-POT/Input with the code above (scripts/FFA_POT.R) should reproduce the results in the examples/FFA-POT/Output  

- Random seed is controlled via `random_state`
- Results are averaged over multiple runs (`n_run = X`)
- Full workflow is described in the Methods section of the paper

---

## 6. Contact

For questions, please contact: [yinxue.liu@ouce.ox.ac.uk] or [yinxue.liu@outlook.com]



