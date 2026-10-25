# DiaCare Analytics - readmission risk platform

Flask app on top of the model of the notebook (early readmission < 30 days, UCI Diabetes 130-US Hospitals).

## Run
    pip install -r requirements.txt
    python app.py            # http://127.0.0.1:5000

The trained model is in `model/bundle.joblib` (scikit-learn version used: see requirements; if you get a
version warning, run `python train_model.py` once, it rebuilds the model in about 8 minutes; it needs
`../SEIF_own/df_model_seif.csv`, produced by the notebook).

## Pages
- `/`        form for one patient -> risk gauge, level, reasons
- `/batch`   CSV of several patients -> ranked list, downloadable
- `/model`   metrics, model comparison, limits
- `POST /api/predict`  JSON in, JSON out

## Model used by the platform
Stacking (logistic regression + gradient boosting + Extra trees) on 18 form fields.
Test set: AUC-ROC 0.652, AUC-PR 0.180, recall 0.64 and precision 0.13 at the follow-up threshold (8.2 %).
Risk levels: lower < 8.2 %, elevated 8.2-12.1 %, high >= 12.1 %.
