# DiaCare Analytics - HbA1c edition (readmission risk platform)

Flask app on top of the model of the notebook *effet du test HbA1c et des changements de traitement*
(early readmission < 30 days, UCI Diabetes 130-US Hospitals, 69,970 stays). Same look and same pages as the
original DiaCare platform; the model and the numbers are the ones of this notebook.

## Run
    pip install -r requirements.txt
    python app.py            # http://127.0.0.1:5000

The model is stored as plain numbers in `model/model.json` (coefficients of the logistic regression), so there is
no pickle file and no scikit-learn version problem. The app itself only needs Flask, pandas and numpy.

## Pages
- `/`        form for one patient -> risk gauge, level, reasons
- `/batch`   CSV of several patients -> ranked list, downloadable (`sample_patients.csv` = 300 real test patients)
- `/model`   metrics, risk levels, model comparison, odds ratios of the HbA1c study, limits
- `POST /api/predict`  JSON in, JSON out

## Model used by the platform
Logistic regression "model 3" of the notebook: 20 fields (55 columns after one-hot encoding), same cleaning, same
80/20 split (random_state=42), numeric columns standardised on the training set only.
Test set (13,994 stays): AUC-ROC 0.642, AUC-PR 0.166, recall 0.537, precision 0.141, accuracy 0.665 (always answering
"no" would give 0.910).
Follow-up threshold 8.8 %: chosen on the training set (5-fold probabilities) as the cut-off with the best recall
while accuracy stays >= 62 %. Risk levels: lower < 8.8 %, elevated 8.8-11.4 %, high >= 11.4 % (top 20 % of the
training scores).
XGBoost (AUC-ROC 0.652) and models 1 and 2 (0.551 and 0.596) appear only in the comparison table.

## Rebuild the model
    pip install scikit-learn xgboost statsmodels
    python train_model.py path/to/df_clean.csv
Rewrites `model/model.json`, `model/metrics.json` and `sample_patients.csv` (a few seconds).

## Notes
- Labels of the admission / discharge / admission-source codes (`labels.py`) follow the IDS_mapping file of the UCI
  dataset; the model keeps only the codes with at least 1 % of the patients, the others are grouped as "Other".
- The "why this score" bars are the coefficients of the model times the distance to an average patient: they
  are associations in past data, not causes.
