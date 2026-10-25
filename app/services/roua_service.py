"""Service for Roua's Medication Load (Polypharmacy) Model (XGBoost Version B)."""
import io
import json
import os
import re
import warnings
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "models", "roua", "model_bundle.joblib")

import sys
LOADERS_DIR = os.path.join(BASE_DIR, "app", "loaders")
if LOADERS_DIR not in sys.path:
    sys.path.insert(0, LOADERS_DIR)
from xgb_compat import load as compat_load

_BUNDLE = None

def _get_bundle():
    global _BUNDLE
    if _BUNDLE is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _BUNDLE = compat_load(MODEL_PATH)
    return _BUNDLE


def _famille(code):
    c = str(code).strip()
    if not c or c in ("nan", "None", "NA", "?", "Inconnu"):
        return "Autre"
    if c.startswith("250"):
        return "Diabete"
    if len(c) > 0 and c[0] in ("V", "E"):
        return "Autre"
    try:
        n = float(c)
    except Exception:
        return "Autre"
    if (390 <= n <= 459) or n == 785:
        return "Circulatoire"
    if (460 <= n <= 519) or n == 786:
        return "Respiratoire"
    if (520 <= n <= 579) or n == 787:
        return "Digestif"
    if 800 <= n <= 999:
        return "Lesions"
    if 710 <= n <= 739:
        return "Musculo"
    if (580 <= n <= 629) or n == 788:
        return "Genito_urinaire"
    if 140 <= n <= 239:
        return "Tumeurs"
    return "Autre"


GENDERS = ["Female", "Male"]
ADMISSION_TYPES = [(1, "Emergency"), (2, "Urgent"), (3, "Elective"), (4, "Other / Unknown")]
ADMISSION_SOURCES = [
    (7, "Emergency Room"), (1, "Physician Referral"), (2, "Clinic Referral"),
    (4, "Transfer from hospital"), (5, "Transfer from SNF"), (6, "Transfer from another facility"),
    (17, "Other / Unknown")
]
SPECIALTIES = [
    "Unknown", "InternalMedicine", "Family/GeneralPractice", "Emergency/Trauma",
    "Cardiology", "Surgery-General", "Orthopedics", "Orthopedics-Reconstructive",
    "Radiologist", "Nephrology", "Other"
]
YESNO = ["Yes", "No"]

LIMITS = {
    "time_in_hospital": (1, 14),
    "num_lab_procedures": (1, 132),
    "num_procedures": (0, 6),
    "number_diagnoses": (1, 16),
    "number_inpatient": (0, 20),
    "number_emergency": (0, 40),
    "number_outpatient": (0, 40)
}

LABELS = {
    "time_in_hospital": "Length of stay (days)",
    "num_lab_procedures": "Number of lab tests",
    "num_procedures": "Number of surgical procedures",
    "number_diagnoses": "Total diagnoses count",
    "number_inpatient": "Inpatient stays (last year)",
    "number_emergency": "Emergency visits (last year)",
    "number_outpatient": "Outpatient visits (last year)",
    "gender": "Gender",
    "admission_type_id": "Admission type",
    "admission_source_id": "Admission source",
    "medical_specialty": "Medical specialty",
    "diag_1": "Primary diagnosis (ICD-9)",
    "diag_2": "Secondary diagnosis 2 (ICD-9)",
    "diag_3": "Secondary diagnosis 3 (ICD-9)",
    "change": "Medication changed during stay",
    "diabetesMed": "Diabetes medication prescribed"
}

DEFAULTS = {
    "time_in_hospital": 4,
    "num_lab_procedures": 43,
    "num_procedures": 1,
    "number_diagnoses": 8,
    "number_inpatient": 0,
    "number_emergency": 0,
    "number_outpatient": 0,
    "gender": "Female",
    "admission_type_id": 1,
    "admission_source_id": 7,
    "medical_specialty": "InternalMedicine",
    "diag_1": "428",
    "diag_2": "250",
    "diag_3": "NA",
    "change": "No",
    "diabetesMed": "Yes"
}

EXAMPLES = {
    "low": dict(
        DEFAULTS,
        time_in_hospital=2, num_lab_procedures=25, num_procedures=0,
        number_diagnoses=4, number_inpatient=0, number_emergency=0,
        medical_specialty="Family/GeneralPractice", diag_1="250", diag_2="NA",
        change="No", diabetesMed="No"
    ),
    "high": dict(
        DEFAULTS,
        time_in_hospital=8, num_lab_procedures=65, num_procedures=3,
        number_diagnoses=10, number_inpatient=2, number_emergency=1,
        medical_specialty="Cardiology", diag_1="428", diag_2="428", diag_3="250",
        change="Yes", diabetesMed="Yes"
    )
}

CLASS_INFO = {
    "low": {
        "title": "Low Medication Load",
        "badge": "b-low",
        "color": "#0ea5e9",
        "description": "≤ 11 medications. Standard medication complexity.",
        "advice": "Standard pharmacy monitoring. Routine medication reconciliation is sufficient."
    },
    "medium": {
        "title": "Moderate Polypharmacy",
        "badge": "b-elevated",
        "color": "#175cdd",
        "description": "12 to 17 medications. Intermediate regimen complexity.",
        "advice": "Consider clinical pharmacist review before discharge to detect potential drug-drug interactions."
    },
    "high": {
        "title": "High Polypharmacy Risk",
        "badge": "b-high",
        "color": "#dc3545",
        "description": "≥ 18 medications. Heavy pharmacological burden.",
        "advice": "Prioritize multidisciplinary medication reconciliation. High alert for contraindications and adverse events."
    }
}

CSV_COLUMNS = [
    "id", "time_in_hospital", "num_lab_procedures", "num_procedures",
    "number_diagnoses", "number_inpatient", "number_emergency", "number_outpatient",
    "gender", "admission_type_id", "admission_source_id", "medical_specialty",
    "diag_1", "diag_2", "diag_3", "change", "diabetesMed"
]


def clean(raw):
    """Validate raw form dictionary."""
    row, errors = {}, []
    for k, (lo, hi) in LIMITS.items():
        try:
            v = float(raw.get(k, DEFAULTS[k]))
            if not lo <= v <= hi:
                errors.append(f"{LABELS[k]} must be between {lo} and {hi}.")
            row[k] = v
        except (TypeError, ValueError):
            errors.append(f"{LABELS[k]} must be a valid number.")

    row["gender"] = str(raw.get("gender", DEFAULTS["gender"]))
    if row["gender"] not in GENDERS:
        row["gender"] = DEFAULTS["gender"]

    try:
        row["admission_type_id"] = int(raw.get("admission_type_id", DEFAULTS["admission_type_id"]))
    except (TypeError, ValueError):
        row["admission_type_id"] = DEFAULTS["admission_type_id"]

    try:
        row["admission_source_id"] = int(raw.get("admission_source_id", DEFAULTS["admission_source_id"]))
    except (TypeError, ValueError):
        row["admission_source_id"] = DEFAULTS["admission_source_id"]

    row["medical_specialty"] = str(raw.get("medical_specialty", DEFAULTS["medical_specialty"]))
    row["diag_1"] = str(raw.get("diag_1", DEFAULTS["diag_1"])).strip() or "NA"
    row["diag_2"] = str(raw.get("diag_2", DEFAULTS["diag_2"])).strip() or "NA"
    row["diag_3"] = str(raw.get("diag_3", DEFAULTS["diag_3"])).strip() or "NA"

    ch = str(raw.get("change", DEFAULTS["change"]))
    row["change"] = "Yes" if ch in ("Yes", "Ch", "1", 1) else "No"

    dm = str(raw.get("diabetesMed", DEFAULTS["diabetesMed"]))
    row["diabetesMed"] = "Yes" if dm in ("Yes", "1", 1) else "No"

    return row, errors


def prepare_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Transform raw dataframe to model input matrix matching Roua's XGBoost pipeline."""
    b = _get_bundle()
    cols = b["columns"]
    scaler = b["scaler"]
    top_spec = b["top_specialties"]
    means = b["train_means"]

    d = raw_df.copy()
    for c in ["diag_1", "diag_2", "diag_3"]:
        if c in d.columns:
            d[c] = d[c].apply(_famille)
        else:
            d[c] = "Autre"

    if "medical_specialty" in d.columns:
        d["medical_specialty"] = d["medical_specialty"].where(d["medical_specialty"].isin(top_spec), "Autre")
    else:
        d["medical_specialty"] = "Autre"

    if "gender" in d.columns:
        d["gender"] = (d["gender"].astype(str) == "Male").astype(int)
    else:
        d["gender"] = 0

    if "change" in d.columns:
        d["change"] = d["change"].isin(["Ch", "Yes", "1", 1]).astype(int)
    else:
        d["change"] = 0

    if "diabetesMed" in d.columns:
        d["diabetesMed"] = (d["diabetesMed"].astype(str) == "Yes").astype(int)
    else:
        d["diabetesMed"] = 1

    for c in ["admission_type_id", "admission_source_id"]:
        if c in d.columns:
            d[c] = d[c].astype(str)
        else:
            d[c] = "1"

    cont = list(means.keys())
    for c in cont:
        if c not in d.columns:
            d[c] = means[c]
        else:
            d[c] = pd.to_numeric(d[c], errors="coerce").fillna(means[c])

    d_enc = pd.get_dummies(d, drop_first=True)
    d_enc.columns = [re.sub(r"[\[\]<>,:\"{}\s]", "_", col) for col in d_enc.columns]
    X = d_enc.reindex(columns=cols, fill_value=0).copy()
    X[cont] = scaler.transform(d[cont])
    return X


def predict_single(row):
    """Predict medication load for a single patient."""
    b = _get_bundle()
    model = b["model"]
    classes = b["class_names"]

    df_row = pd.DataFrame([row])
    X = prepare_features(df_row)
    pred_idx = int(model.predict(X)[0])
    probs = model.predict_proba(X)[0]

    pred_class = classes[pred_idx]
    probs_dict = {classes[i]: round(float(probs[i]), 4) for i in range(len(classes))}

    info = CLASS_INFO.get(pred_class, CLASS_INFO["medium"])
    return {
        "pred": pred_class,
        "tier": pred_class,
        "title": info["title"],
        "badge": info["badge"],
        "color": info["color"],
        "description": info["description"],
        "advice": info["advice"],
        "probabilities": probs_dict,
        "probs_percent": {k: round(v * 100, 1) for k, v in probs_dict.items()},
        "top_prob": round(float(probs[pred_idx]) * 100, 1)
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
        "admission_types": ADMISSION_TYPES,
        "admission_sources": ADMISSION_SOURCES,
        "specialties": SPECIALTIES,
        "yesno": YESNO,
        "labels": LABELS,
        "class_info": CLASS_INFO
    }


def process_batch_file(file_obj):
    """Process uploaded CSV file for batch medication load assessment."""
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
        row, errs = clean(raw)
        if errs:
            bad.append(f"Row {i + 2}: {errs[0]}")
        rows.append(row)

    if bad:
        msg = "; ".join(bad[:5]) + (" ..." if len(bad) > 5 else "")
        return None, msg

    b = _get_bundle()
    model = b["model"]
    classes = b["class_names"]

    X = prepare_features(pd.DataFrame(rows))
    preds = model.predict(X)
    probs = model.predict_proba(X)

    out = df[["id"]].copy()
    out["predicted_class"] = [classes[p] for p in preds]
    out["prob_high"] = np.round(probs[:, classes.index("high")], 4)
    out["level_label"] = [CLASS_INFO[classes[p]]["title"] for p in preds]

    out = out.sort_values("prob_high", ascending=False).reset_index(drop=True)
    out.insert(0, "rank", out.index + 1)

    summary = {
        "n": len(out),
        "high_load": int((out["predicted_class"] == "high").sum()),
        "medium_load": int((out["predicted_class"] == "medium").sum()),
        "low_load": int((out["predicted_class"] == "low").sum()),
    }

    return {
        "summary": summary,
        "dataframe": out,
        "preview": out.head(25).to_dict("records"),
        "class_info": CLASS_INFO
    }, None


def generate_template_csv():
    """Generate example CSV template for Roua's tool."""
    rows = []
    for i, ex in enumerate([EXAMPLES["high"], EXAMPLES["low"]], 1):
        rows.append([i] + [ex.get(c, "") for c in CSV_COLUMNS[1:]])
    df = pd.DataFrame(rows, columns=CSV_COLUMNS)
    buf = io.BytesIO(df.to_csv(index=False).encode("utf-8"))
    return buf
