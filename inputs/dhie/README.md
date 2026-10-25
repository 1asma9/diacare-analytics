# DiaCare Discharge — Deployment Package

Predicts where a diabetic patient will be discharged:
**Home / Home with care / Transfer / Other**

Also fires a **transfer alert** when the binary model detects a high transfer probability.

---

## Folder structure

```
DiaCareDischarge_deploy/
├── app.py                  ← Flask backend (all routes)
├── preprocessing.py        ← Feature engineering (do NOT modify)
├── requirements.txt        ← Python dependencies
├── artifacts/
│   └── model.joblib        ← Trained model (49 MB)
├── static/
│   ├── style.css
│   └── app.js
└── templates/
    ├── base.html
    ├── index.html          ← Patient form
    ├── result.html         ← Prediction result
    ├── batch.html          ← CSV batch upload
    └── model.html          ← Model info page
```

---

## How to run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the app
```bash
python app.py
```

### 3. Open in browser
```
http://127.0.0.1:5001
```

---

## Pages

| URL | What it does |
|---|---|
| `/` | Single patient form |
| `/predict` | POST — form submission |
| `/api/predict` | POST — JSON API |
| `/batch` | Upload CSV, score up to 5000 patients |
| `/model` | Model metrics and description |

---

## JSON API example

```bash
POST http://127.0.0.1:5001/api/predict
Content-Type: application/json

{
  "age": 75,
  "gender": "Female",
  "race": "Caucasian",
  "admission_type_id": 1,
  "admission_source_id": 7,
  "time_in_hospital": 6,
  "payer_code": "MC",
  "medical_specialty": "Cardiology",
  "num_lab_procedures": 55,
  "num_procedures": 2,
  "num_medications": 18,
  "number_outpatient": 0,
  "number_emergency": 1,
  "number_inpatient": 1,
  "diag_1": "428",
  "diag_2": "250",
  "diag_3": "NA",
  "number_diagnoses": 9,
  "max_glu_serum": "None",
  "A1Cresult": ">7",
  "change": "Ch",
  "diabetesMed": "Yes",
  "sejours_anterieurs": 1,
  "insulin": "Up",
  "metformin": "No"
}
```

Response:
```json
{
  "prediction": "Transfert",
  "probabilities": {
    "Autre": 0.08,
    "Domicile": 0.21,
    "Domicile avec soins": 0.28,
    "Transfert": 0.43
  },
  "transfer_probability": 0.612,
  "transfer_alert": true,
  "threshold": 0.547
}
```

---

## Notes
- Runs on port **5001** (avoids conflict with other team members on 5000)
- No dataset needed — the trained model is already included
- `preprocessing.py` must stay in the same folder as `app.py`
- Academic project — not a clinical tool
