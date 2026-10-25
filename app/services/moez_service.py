"""Service for Moez's Unsupervised Patient Clustering & Profiling Model (K-Means k=2, PCA)."""
import json
import os
import warnings
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "models", "moez", "cluster_bundle.joblib")
PROFILES_PATH = os.path.join(BASE_DIR, "models", "moez", "cluster_profiles.json")

import sys
LOADERS_DIR = os.path.join(BASE_DIR, "app", "loaders")
if LOADERS_DIR not in sys.path:
    sys.path.insert(0, LOADERS_DIR)
from xgb_compat import load as compat_load

_BUNDLE = None
_PROFILES = None

def _get_bundle():
    global _BUNDLE, _PROFILES
    if _BUNDLE is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _BUNDLE = compat_load(MODEL_PATH)
        with open(PROFILES_PATH, "r", encoding="utf-8") as f:
            _PROFILES = json.load(f)
    return _BUNDLE, _PROFILES


def icd9_group(code):
    c = str(code).strip()
    if not c or c.startswith(("V", "E")) or c in ("nan", "None", "NA", "?"):
        return "Other"
    try:
        x = float(c)
    except Exception:
        return "Other"
    if 250 <= x < 251:
        return "Diabetes"
    if 390 <= x <= 459 or x == 785:
        return "Circulatory"
    if 460 <= x <= 519 or x == 786:
        return "Respiratory"
    if 520 <= x <= 579 or x == 787:
        return "Digestive"
    if 580 <= x <= 629 or x == 788:
        return "Genitourinary"
    if 800 <= x <= 999:
        return "Injury"
    if 710 <= x <= 739:
        return "Musculoskeletal"
    if 140 <= x <= 239:
        return "Neoplasms"
    return "Other"


def adm_type(x):
    try:
        xi = int(x)
    except Exception:
        xi = 0
    return {1: "Emergency", 2: "Urgent", 3: "Elective"}.get(xi, "OtherType")


def adm_src(x):
    try:
        xi = int(x)
    except Exception:
        xi = 0
    if xi == 7:
        return "ER"
    if xi in (1, 2, 3):
        return "Referral"
    if xi in (4, 5, 6, 10, 18, 19, 22, 25, 26):
        return "Transfer"
    return "OtherSource"


def dis(x):
    try:
        xi = int(x)
    except Exception:
        xi = 0
    if xi in (1, 6, 8):
        return "Home"
    if xi in (3, 4, 5, 22, 23, 24, 27, 28, 29, 30, 2, 15, 16, 17):
        return "Facility"
    return "OtherDischarge"


MED_COLS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride", "acetohexamide",
    "glipizide", "glyburide", "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose", "miglitol",
    "troglitazone", "tolazamide", "examide", "citoglipton", "insulin", "glyburide-metformin",
    "glipizide-metformin", "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone"
]

LIMITS = {
    "age_num": (5, 95),
    "time_in_hospital": (1, 14),
    "num_lab_procedures": (1, 130),
    "num_procedures": (0, 6),
    "num_medications": (1, 80),
    "number_outpatient": (0, 40),
    "number_emergency": (0, 40),
    "number_inpatient": (0, 20),
    "number_diagnoses": (1, 16),
    "n_meds_used": (0, 10),
    "n_meds_changed": (0, 6)
}

LABELS = {
    "age_num": "Patient approximate age (years)",
    "gender": "Gender",
    "time_in_hospital": "Length of stay (days)",
    "num_lab_procedures": "Lab procedures performed",
    "num_procedures": "Surgical procedures performed",
    "num_medications": "Total medications administered",
    "number_outpatient": "Outpatient visits (past year)",
    "number_emergency": "Emergency visits (past year)",
    "number_inpatient": "Inpatient admissions (past year)",
    "number_diagnoses": "Total diagnoses recorded",
    "n_meds_used": "Number of diabetes medications used",
    "n_meds_changed": "Number of diabetes drugs adjusted (dose change)",
    "insulin": "Insulin prescribed",
    "metformin": "Metformin prescribed",
    "change": "Overall medication change during stay",
    "diabetesMed": "Diabetes medication prescribed",
    "diag_1": "Primary diagnosis (ICD-9)",
    "admission_type_id": "Admission type",
    "admission_source_id": "Admission source",
    "discharge_disposition_id": "Discharge destination",
    "A1Cresult": "HbA1c test result",
    "max_glu_serum": "Max glucose serum test result"
}

DEFAULTS = {
    "age_num": 65,
    "gender": "Female",
    "time_in_hospital": 4,
    "num_lab_procedures": 43,
    "num_procedures": 1,
    "num_medications": 15,
    "number_outpatient": 0,
    "number_emergency": 0,
    "number_inpatient": 0,
    "number_diagnoses": 8,
    "n_meds_used": 1,
    "n_meds_changed": 0,
    "insulin": "No",
    "metformin": "No",
    "change": "No",
    "diabetesMed": "Yes",
    "diag_1": "428",
    "admission_type_id": 1,
    "admission_source_id": 7,
    "discharge_disposition_id": 1,
    "A1Cresult": "NotTested",
    "max_glu_serum": "NotTested"
}

EXAMPLES = {
    "profile0": dict(
        DEFAULTS,
        age_num=65, time_in_hospital=2, num_lab_procedures=32, num_procedures=0,
        num_medications=10, number_diagnoses=5, n_meds_used=1, n_meds_changed=0,
        change="No", insulin="No", discharge_disposition_id=1, A1Cresult="Norm"
    ),
    "profile1": dict(
        DEFAULTS,
        age_num=75, time_in_hospital=8, num_lab_procedures=65, num_procedures=2,
        num_medications=24, number_diagnoses=10, n_meds_used=3, n_meds_changed=2,
        change="Ch", insulin="Up", discharge_disposition_id=3, A1Cresult=">8"
    )
}

PROFILE_DESCRIPTIONS = {
    0: {
        "title": "Profile 0 — Routine / Moderate-Acuity Stay",
        "badge": "b-low",
        "color": "#175cdd",
        "cohort_share": "62.0%",
        "overall_readmission": "37.3%",
        "early_readmission": "7.7%",
        "summary": "Short to standard hospital stays with lower medication burden, fewer drug adjustments, and typical home discharge.",
        "characteristics": [
            "Mean stay duration: 3.1 days (vs. 6.3 days in Profile 1)",
            "Medication count: average 12.0 drugs (vs. 21.6 in Profile 1)",
            "Medication changes: only 25% of stays involved a dosage change (vs. 77% in Profile 1)",
            "Early readmission rate (<30 days): 7.7% (low clinical urgency)"
        ],
        "recommendation": "Standard discharge pathway is generally suitable. Confirm medication understanding before discharge."
    },
    1: {
        "title": "Profile 1 — Complex / High-Acuity Polypharmacy",
        "badge": "b-high",
        "color": "#dc3545",
        "cohort_share": "38.0%",
        "overall_readmission": "46.4%",
        "early_readmission": "11.1%",
        "summary": "Prolonged stays characterized by heavy polypharmacy, frequent therapeutic modifications, acute comorbidity, and frequent transfer to care facilities.",
        "characteristics": [
            "Mean stay duration: 6.3 days (double that of Profile 0)",
            "Medication count: average 21.6 drugs (heavy polypharmacy burden)",
            "Medication changes: 77% of stays required active dosage titration",
            "Discharge to skilled facility: 27.3% (lift of 1.4x over hospital average)",
            "Uncontrolled HbA1c (>8%): 12.7% of stays (lift of 1.42x)",
            "Early readmission rate (<30 days): 11.1% (substantially elevated)"
        ],
        "recommendation": "Prioritize multidisciplinary discharge planning, pharmacist-led medication reconciliation, and post-discharge telephone check-in within 72 hours."
    }
}


def clean(raw):
    """Clean and validate form data for Moez's model."""
    row, errors = {}, []
    for k, (lo, hi) in LIMITS.items():
        try:
            v = float(raw.get(k, DEFAULTS[k]))
            if not lo <= v <= hi:
                errors.append(f"{LABELS[k]} must be between {lo} and {hi}.")
            row[k] = v
        except (TypeError, ValueError):
            errors.append(f"{LABELS[k]} must be a number.")

    row["gender"] = str(raw.get("gender", DEFAULTS["gender"]))
    row["diag_1"] = str(raw.get("diag_1", DEFAULTS["diag_1"])).strip() or "428"

    for id_col in ("admission_type_id", "admission_source_id", "discharge_disposition_id"):
        try:
            row[id_col] = int(raw.get(id_col, DEFAULTS[id_col]))
        except (TypeError, ValueError):
            row[id_col] = DEFAULTS[id_col]

    for med in ("insulin", "metformin"):
        row[med] = str(raw.get(med, DEFAULTS[med]))

    ch = str(raw.get("change", DEFAULTS["change"]))
    row["change"] = "Ch" if ch in ("Ch", "Yes", "1", 1) else "No"

    dm = str(raw.get("diabetesMed", DEFAULTS["diabetesMed"]))
    row["diabetesMed"] = "Yes" if dm in ("Yes", "1", 1) else "No"

    a1c = str(raw.get("A1Cresult", DEFAULTS["A1Cresult"]))
    row["A1Cresult"] = a1c if a1c in ("Norm", ">7", ">8", "None", "NotTested") else "NotTested"

    glu = str(raw.get("max_glu_serum", DEFAULTS["max_glu_serum"]))
    row["max_glu_serum"] = glu if glu in ("Norm", ">200", ">300", "None", "NotTested") else "NotTested"

    return row, errors


def prepare_features(d_raw: pd.DataFrame):
    """Transform input dataframe into the aligned feature matrix for Moez's bundle."""
    b, _ = _get_bundle()
    num_feats = b["num_feats"]
    bin_feats = b["bin_feats"]
    cat_feats = b["cat_feats"]
    skewed = b["skewed"]
    cols_dum = b["columns_after_dummies"]
    scaler = b["scaler"]

    d = d_raw.copy()
    d["diag_group"] = d["diag_1"].apply(icd9_group)
    d["admission_type"] = d["admission_type_id"].apply(adm_type)
    d["admission_source"] = d["admission_source_id"].apply(adm_src)
    d["discharge"] = d["discharge_disposition_id"].apply(dis)

    d["A1Cresult"] = d["A1Cresult"].astype(str).replace({"None": "NotTested", "nan": "NotTested"})
    d["max_glu_serum"] = d["max_glu_serum"].astype(str).replace({"None": "NotTested", "nan": "NotTested"})

    # Ensure num_feats present
    for nf in num_feats:
        if nf not in d.columns:
            d[nf] = DEFAULTS.get(nf, 0)
        else:
            d[nf] = pd.to_numeric(d[nf], errors="coerce").fillna(DEFAULTS.get(nf, 0))

    # Binary features
    d["insulin_used"] = (d.get("insulin", "No").astype(str) != "No").astype(int)
    d["metformin_used"] = (d.get("metformin", "No").astype(str) != "No").astype(int)
    d["change_bin"] = d.get("change", "No").isin(["Ch", "Yes", "1", 1]).astype(int)
    d["diabetesMed_bin"] = (d.get("diabetesMed", "Yes").astype(str) == "Yes").astype(int)
    d["gender_male"] = (d.get("gender", "Female").astype(str) == "Male").astype(int)

    X_num = d[num_feats].copy()
    X_num[skewed] = np.log1p(X_num[skewed])
    X_num_s = pd.DataFrame(scaler.transform(X_num), columns=num_feats, index=d.index)
    X_cat = pd.get_dummies(d[cat_feats], dtype=int)
    X_full = pd.concat([X_num_s, d[bin_feats], X_cat], axis=1)
    X_aligned = X_full.reindex(columns=cols_dum, fill_value=0)
    return X_aligned


def assign_patient(row):
    """Assign a single patient to Cluster 0 or 1, and return 2D coordinates and profile info."""
    b, profiles = _get_bundle()
    pca = b["pca"]
    pca_2d = b["pca_2d"]
    km = b["km"]
    remap = b["remap"]

    df_row = pd.DataFrame([row])
    X = prepare_features(df_row)
    X_pca = pca.transform(X)
    raw_cluster = int(km.predict(X_pca)[0])
    cluster = remap[raw_cluster]

    coords_2d = pca_2d.transform(X)[0]
    pc1, pc2 = round(float(coords_2d[0]), 2), round(float(coords_2d[1]), 2)

    profile_data = PROFILE_DESCRIPTIONS[cluster]
    stats = profiles[str(cluster)]

    # Distances to cluster centers in PCA space
    centers = km.cluster_centers_
    dist_0 = float(np.linalg.norm(X_pca[0] - centers[0]))
    dist_1 = float(np.linalg.norm(X_pca[0] - centers[1]))

    return {
        "cluster": cluster,
        "title": profile_data["title"],
        "badge": profile_data["badge"],
        "color": profile_data["color"],
        "summary": profile_data["summary"],
        "characteristics": profile_data["characteristics"],
        "recommendation": profile_data["recommendation"],
        "pc1": pc1,
        "pc2": pc2,
        "dist_0": round(dist_0, 2),
        "dist_1": round(dist_1, 2),
        "profile_stats": stats,
        "patient_metrics": {
            "time_in_hospital": row["time_in_hospital"],
            "num_medications": row["num_medications"],
            "number_diagnoses": row["number_diagnoses"],
            "n_meds_changed": row["n_meds_changed"],
            "n_meds_used": row["n_meds_used"]
        }
    }


def get_all_profiles_info():
    """Retrieve full statistical metadata and descriptions for both profiles."""
    _, profiles = _get_bundle()
    return {
        "descriptions": PROFILE_DESCRIPTIONS,
        "statistics": profiles
    }


def get_form_context(form_data=None, errors=None):
    d = form_data or DEFAULTS
    errors = errors or []
    return {
        "d": d,
        "errors": errors,
        "limits": LIMITS,
        "examples": EXAMPLES,
        "labels": LABELS,
        "profiles": PROFILE_DESCRIPTIONS
    }
