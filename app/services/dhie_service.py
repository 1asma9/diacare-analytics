"""Service for Dhie's Discharge Destination Prediction Model (Ensemble + Class Weights)."""
import io
import json
import os
import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "models", "dhie", "model.joblib")

# Import Dhie's preprocessing functions from models/dhie/preprocessing.py
import sys
DHIE_MODEL_DIR = os.path.join(BASE_DIR, "models", "dhie")
if DHIE_MODEL_DIR not in sys.path:
    sys.path.insert(0, DHIE_MODEL_DIR)
from preprocessing import MED_COLS, age_to_bracket, align, build_features

_BUNDLE = None

def _get_bundle():
    global _BUNDLE
    if _BUNDLE is None:
        _BUNDLE = joblib.load(MODEL_PATH)
    return _BUNDLE

GENDERS = ["Female", "Male"]
RACES = ["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other", "Inconnu"]
ADM_TYPE = [(1, "Emergency"), (2, "Urgent"), (3, "Elective"), (0, "Other / Unknown")]
ADM_SRC = [
    (7, "Emergency room"),
    (1, "Physician referral"),
    (2, "Self-referral"),
    (4, "Transfer from hospital"),
    (5, "Transfer from SNF"),
    (6, "Transfer from another facility"),
    (11, "Normal delivery (not emergency)"),
    (20, "Other"),
]
PAYER = ["MC", "HM", "SP", "BC", "MD", "CP", "UN", "CM", "Inconnu"]
SPECIALTY = [
    "Inconnu", "InternalMedicine", "Cardiology", "Emergency/Trauma",
    "Family/GeneralPractice", "Surgery-General", "Orthopedics",
    "Gastroenterology", "Pulmonology", "Nephrology", "Psychiatry", "Urology",
]
INSULIN = ["No", "Steady", "Up", "Down"]
A1C = ["None", "Norm", ">7", ">8"]
GLU = ["None", "Norm", ">200", ">300"]
CHANGE = [("No", "No"), ("Ch", "Yes")]
YESNO = ["Yes", "No"]
METFORMIN = ["No", "Steady", "Up", "Down"]

DEFAULTS = dict(
    age=70, gender="Female", race="Caucasian",
    admission_type_id=1, admission_source_id=7,
    time_in_hospital=4, payer_code="MC", medical_specialty="Inconnu",
    num_lab_procedures=40, num_procedures=0, num_medications=15,
    number_outpatient=0, number_emergency=0, number_inpatient=0,
    diag_1="428", diag_2="250", diag_3="NA",
    number_diagnoses=7, max_glu_serum="None", A1Cresult="None",
    change="No", diabetesMed="Yes", sejours_anterieurs=0,
    insulin="No", metformin="No",
)

EXAMPLES = {
    "home": dict(
        DEFAULTS,
        age=65, admission_type_id=3, admission_source_id=1,
        time_in_hospital=2, num_lab_procedures=25, num_medications=8,
        diag_1="250", diag_2="401", number_diagnoses=5,
        sejours_anterieurs=0, insulin="Steady", diabetesMed="Yes",
    ),
    "transfer": dict(
        DEFAULTS,
        age=82, admission_type_id=1, admission_source_id=7,
        time_in_hospital=9, num_lab_procedures=68, num_procedures=3,
        num_medications=22, number_inpatient=2, number_emergency=1,
        diag_1="428", diag_2="428", diag_3="250",
        number_diagnoses=11, A1Cresult=">8",
        change="Ch", insulin="Up", metformin="No",
        sejours_anterieurs=3, medical_specialty="Cardiology",
    ),
}

LIMITS = {
    "age": (0, 100),
    "time_in_hospital": (1, 14),
    "num_lab_procedures": (1, 130),
    "num_procedures": (0, 6),
    "num_medications": (1, 80),
    "number_outpatient": (0, 40),
    "number_emergency": (0, 40),
    "number_inpatient": (0, 20),
    "number_diagnoses": (1, 16),
    "sejours_anterieurs": (0, 50),
}

LABELS = {
    "age": "Age (years)", "gender": "Gender", "race": "Race / ethnicity",
    "admission_type_id": "Admission type", "admission_source_id": "Admission source",
    "time_in_hospital": "Length of stay", "payer_code": "Insurance",
    "medical_specialty": "Medical specialty",
    "num_lab_procedures": "Lab procedures", "num_procedures": "Procedures",
    "num_medications": "Medications",
    "number_outpatient": "Outpatient visits (past year)",
    "number_emergency": "Emergency visits (past year)",
    "number_inpatient": "Inpatient stays (past year)",
    "diag_1": "Main diagnosis (ICD-9)", "diag_2": "Secondary diagnosis 2",
    "diag_3": "Secondary diagnosis 3",
    "number_diagnoses": "Total diagnoses", "max_glu_serum": "Max glucose serum",
    "A1Cresult": "HbA1c result", "change": "Medication changed during stay",
    "diabetesMed": "Diabetes medication prescribed",
    "sejours_anterieurs": "Prior stays (this patient)",
    "insulin": "Insulin", "metformin": "Metformin",
}

CLASS_LABEL = {
    "Domicile": "Home",
    "Domicile avec soins": "Home with care",
    "Transfert": "Transfer",
    "Autre": "Other / Unknown",
}
CLASS_CSS = {
    "Domicile": "b-domicile",
    "Domicile avec soins": "b-soins",
    "Transfert": "b-transfert",
    "Autre": "b-autre",
}

METRICS = {
    "f1_macro": 0.520,
    "transfert_recall": 0.705,
    "transfert_precision": 0.462,
    "transfert_auc": 0.834,
    "ci_low": 0.510,
    "ci_high": 0.528,
}

CSV_COLUMNS = [
    "id", "age", "gender", "race", "admission_type_id", "admission_source_id",
    "time_in_hospital", "payer_code", "medical_specialty",
    "num_lab_procedures", "num_procedures", "num_medications",
    "number_outpatient", "number_emergency", "number_inpatient",
    "diag_1", "diag_2", "diag_3", "number_diagnoses",
    "max_glu_serum", "A1Cresult", "change", "diabetesMed",
    "sejours_anterieurs", "insulin", "metformin",
]


def clean(raw):
    """Validate form dict -> (row dict, errors)."""
    row, errors = {}, []
    for k, (lo, hi) in LIMITS.items():
        val = raw.get(k, DEFAULTS[k])
        if k == "age" and isinstance(val, str) and "[" in val:
            # Parse bracket like "[50-60)" -> 55
            try:
                import re
                m = re.search(r"\[(\d+)-", val)
                val = int(m.group(1)) + 5 if m else DEFAULTS["age"]
            except Exception:
                val = DEFAULTS["age"]
        try:
            v = int(val)
            if not lo <= v <= hi:
                errors.append(f"{LABELS[k]} must be between {lo} and {hi}.")
            row[k] = v
        except (TypeError, ValueError):
            errors.append(f"{LABELS[k]} must be a whole number.")

    checks = {
        "gender": GENDERS, "race": RACES, "payer_code": PAYER,
        "medical_specialty": SPECIALTY, "insulin": INSULIN,
        "metformin": METFORMIN, "A1Cresult": A1C,
        "max_glu_serum": GLU, "diabetesMed": YESNO,
    }
    for k, ok in checks.items():
        v = str(raw.get(k, DEFAULTS[k]))
        if v not in ok:
            v = DEFAULTS[k]
        row[k] = v

    ch = str(raw.get("change", DEFAULTS["change"]))
    row["change"] = "Ch" if ch in ("Ch", "Yes") else "No"

    try:
        row["admission_type_id"] = int(raw.get("admission_type_id", DEFAULTS["admission_type_id"]))
    except (TypeError, ValueError):
        row["admission_type_id"] = DEFAULTS["admission_type_id"]
    try:
        row["admission_source_id"] = int(raw.get("admission_source_id", DEFAULTS["admission_source_id"]))
    except (TypeError, ValueError):
        row["admission_source_id"] = DEFAULTS["admission_source_id"]

    for c in ("diag_1", "diag_2", "diag_3"):
        v = str(raw.get(c, DEFAULTS[c])).strip()
        row[c] = v if v else "NA"

    for m in MED_COLS:
        row[m] = str(raw.get(m, "No"))

    return row, errors


def predict_single(row):
    """Predict destination for a single cleaned patient dict."""
    art = _get_bundle()
    w = np.array(art["class_weights"])
    classes = art["classes"]
    thr = float(art["threshold"])

    r = dict(row)
    # Convert age to bracket format expected by preprocessing
    if isinstance(r["age"], (int, float)):
        r["age"] = age_to_bracket(r["age"])
    elif isinstance(r["age"], str) and not r["age"].startswith("["):
        try:
            r["age"] = age_to_bracket(float(r["age"]))
        except ValueError:
            r["age"] = "[70-80)"

    X, _ = build_features(pd.DataFrame([r]), art["params"])
    X = align(X, art["all_columns"])

    proba = (art["hgb"].predict_proba(X) + art["rf"].predict_proba(X))[0] / 2
    pred = classes[int(np.argmax(proba * w))]
    p_tr = float(art["hgb_binary"].predict_proba(X)[0, 1])
    probs = {c: round(float(p), 4) for c, p in zip(classes, proba)}

    return {
        "pred": pred,
        "pred_label": CLASS_LABEL.get(pred, pred),
        "css_class": CLASS_CSS.get(pred, "b-autre"),
        "probs": probs,
        "transfer_prob": round(p_tr, 4),
        "transfer_alert": bool(p_tr >= thr),
        "threshold": round(thr, 3)
    }


def get_form_context(form_data=None, errors=None):
    d = form_data or DEFAULTS
    errors = errors or []
    return {
        "d": d,
        "errors": errors,
        "limits": LIMITS,
        "examples": EXAMPLES,
        "genders": GENDERS,
        "races": RACES,
        "adm_types": ADM_TYPE,
        "adm_srcs": ADM_SRC,
        "payers": PAYER,
        "specialties": SPECIALTY,
        "insulins": INSULIN,
        "metformins": METFORMIN,
        "a1c_opts": A1C,
        "glu_opts": GLU,
        "change_opts": CHANGE,
        "yesno": YESNO,
        "labels": LABELS,
        "class_label": CLASS_LABEL,
        "class_css": CLASS_CSS
    }


def process_batch_file(file_obj):
    """Process uploaded CSV file for batch discharge destination prediction."""
    try:
        df = pd.read_csv(file_obj)
    except Exception:
        return None, "The file could not be read as CSV."

    missing = [c for c in CSV_COLUMNS if c not in df.columns]
    if missing:
        return None, f"Missing columns: {', '.join(missing)}"

    if len(df) > 5000:
        return None, "Maximum 5,000 patients per file."

    rows, bad = [], []
    for i, r in df.iterrows():
        raw = {k: ("" if pd.isna(r.get(k)) else r[k]) for k in CSV_COLUMNS}
        raw["diag_3"] = raw["diag_3"] or "NA"
        raw["diag_2"] = raw["diag_2"] or "NA"
        row, errs = clean(raw)
        if errs:
            bad.append(f"Row {i + 2}: {errs[0]}")
        rows.append(row)

    if bad:
        msg = "; ".join(bad[:5]) + (" ..." if len(bad) > 5 else "")
        return None, msg

    preds_out = []
    for row in rows:
        res = predict_single(row)
        preds_out.append(res)

    out = df[["id"]].copy()
    out["prediction"] = [CLASS_LABEL.get(p["pred"], p["pred"]) for p in preds_out]
    out["transfer_prob"] = [p["transfer_prob"] for p in preds_out]
    out["transfer_alert"] = ["yes" if p["transfer_alert"] else "no" for p in preds_out]
    out = out.sort_values("transfer_prob", ascending=False).reset_index(drop=True)
    out.insert(0, "rank", out.index + 1)

    summary = {
        "n": len(out),
        "transfer_alerts": int((out["transfer_alert"] == "yes").sum()),
        "pct_transfer": round((out["transfer_alert"] == "yes").mean() * 100, 1),
    }

    return {
        "summary": summary,
        "dataframe": out,
        "preview": out.head(25).to_dict("records"),
        "class_label": CLASS_LABEL,
        "class_css": CLASS_CSS
    }, None


def generate_template_csv():
    """Generate example CSV template for Dhie's tool."""
    rows = []
    for i, e in enumerate([EXAMPLES["transfer"], EXAMPLES["home"]], 1):
        rows.append([i] + [e.get(c, "") for c in CSV_COLUMNS[1:]])
    df = pd.DataFrame(rows, columns=CSV_COLUMNS)
    buf = io.BytesIO(df.to_csv(index=False).encode("utf-8"))
    return buf
