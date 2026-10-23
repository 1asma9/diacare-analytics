# DiaCare StayCast: length-of-stay decision support

StayCast tells a clinician how long a patient is likely to stay in hospital, with a usual range, an expected discharge date and a ward-level bed forecast. It is the length-of-stay module of **DiaCare Analytics** (Team 3, 4CCE19, ESPRIT), built by Asma Abdellatif.

**Live demo:** add your GitHub Pages link here.

## What it does
- Forecasts at admission (diagnoses, procedures, medication change) or later in the stay (adds lab tests and medications).
- Shows the likely stay, the usual range, and how 100 similar past patients actually stayed.
- Explains which inputs add or remove time, in plain language.
- Ward planner: beds still occupied by day of stay for a list of patients.
- Runs fully in the browser. No data is sent anywhere or stored.

## Model
Target: `time_in_hospital > 3 days` (cleaned diabetes cohort, 69,970 stays). Pipeline: 80/20 split, 99th-percentile capping, log1p on skewed counts, correlation filter (> 0.1), AdaBoost with 200 stumps. Test accuracy 61.0% at admission and 68.5% later in the stay (AUC 0.65 and 0.76; majority guess 50.2%).
The stumps are summed per variable into lookup tables and recalibrated, so the page needs no ML runtime and the explanations are exact. Stay ranges come from observed stays in ten risk groups.

## Run or rebuild
```bash
# open index.html directly, or serve it
python -m http.server 8000

# rebuild model.json from the data (then paste it into index.html as the M constant)
pip install pandas numpy scikit-learn
python export_model.py data/df_clean.csv
```

## Deploy on GitHub Pages
Push `index.html` to a repo, then Settings > Pages > Deploy from branch > `main` / root.

## Limits
Planning aid, not a diagnosis. The admission-time estimate is only modestly better than chance for a single patient and is most reliable across a ward.

## Project structure
```
index.html            the app (GitHub Pages entry point)
model.json            exported model tables used by the app
export_model.py       rebuilds model.json from the data
data/df_clean.csv     cleaned cohort (69,970 stays)
notebooks/            modelling notebook (XGBoost and AdaBoost)
docs/                 validation sheet
requirements.txt      Python dependencies for export_model.py
```
