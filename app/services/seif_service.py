"""Service for Seif's Readmission Risk Screening Model (Stacking Ensemble)."""
import io
import json
import os
import joblib
import numpy as np
import pandas as pd

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "models", "seif", "bundle.joblib")
METRICS_PATH = os.path.join(BASE_DIR, "models", "seif", "metrics.json")

# Load artifacts
_BUNDLE = None
_METRICS = None

def _get_bundle():
    global _BUNDLE, _METRICS
    if _BUNDLE is None:
        _BUNDLE = joblib.load(MODEL_PATH)
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            _METRICS = json.load(f)
    return _BUNDLE, _METRICS

AGES = [f"{i}-{i + 10}" for i in range(0, 100, 10)]
DISCHARGE = [
    "Home", "Home with home-health care", "Skilled nursing facility",
    "Other short-term hospital", "Rehabilitation facility",
    "Long-term care hospital", "Other inpatient care", "Other"
]
ADMISSION = ["Emergency", "Urgent", "Elective", "Other"]
DIAG = {
    "Circulatory": "Circulatory (heart, vessels)",
    "Diabetes": "Diabetes",
    "Respiratory": "Respiratory",
    "Digestive": "Digestive",
    "Injury": "Injury / poisoning",
    "Musculoskeletal": "Musculoskeletal",
    "Genitourinary": "Genitourinary",
    "Neoplasms": "Neoplasms (cancer)",
    "Other": "Other",
    "Unknown": "Unknown"
}
INSULIN = ["No", "Steady", "Up", "Down"]
A1C = ["None", "Norm", ">7", ">8"]
YESNO = ["Yes", "No"]

LABELS = {
    "age": "Age",
    "time_in_hospital": "Length of stay",
    "num_lab_procedures": "Number of lab tests",
    "num_procedures": "Number of procedures",
    "num_medications": "Number of medications",
    "number_outpatient": "Outpatient visits (last year)",
    "number_emergency": "Emergency visits (last year)",
    "number_inpatient": "Inpatient stays (last year)",
    "number_diagnoses": "Number of diagnoses",
    "n_meds_used": "Diabetes drugs prescribed",
    "n_med_changes": "Drug dose changes",
    "discharge": "Discharge destination",
    "admission": "Type of admission",
    "diag_1": "Main diagnosis",
    "insulin": "Insulin",
    "A1Cresult": "HbA1c result",
    "change": "Change of medication",
    "diabetesMed": "Diabetes medication"
}

LIMITS = {
    "time_in_hospital": (1, 14),
    "num_lab_procedures": (1, 130),
    "num_procedures": (0, 6),
    "num_medications": (1, 80),
    "number_outpatient": (0, 40),
    "number_emergency": (0, 40),
    "number_inpatient": (0, 20),
    "number_diagnoses": (1, 16),
    "n_meds_used": (0, 10),
    "n_med_changes": (0, 6)
}

DEFAULTS = {
    "age": 7, "time_in_hospital": 4, "num_lab_procedures": 43, "num_procedures": 1,
    "num_medications": 15, "number_outpatient": 0, "number_emergency": 0,
    "number_inpatient": 0, "number_diagnoses": 8, "n_meds_used": 1,
    "n_med_changes": 0, "discharge": "Home", "admission": "Emergency",
    "diag_1": "Circulatory", "insulin": "No", "A1Cresult": "None",
    "change": "No", "diabetesMed": "Yes"
}

EXAMPLES = {
    "lower": dict(
        DEFAULTS, age=5, time_in_hospital=2, num_lab_procedures=30, num_medications=9,
        number_diagnoses=5, diabetesMed="No", insulin="No"
    ),
    "higher": dict(
        DEFAULTS, age=8, time_in_hospital=7, num_lab_procedures=60, num_procedures=2,
        num_medications=26, number_outpatient=2, number_emergency=2, number_inpatient=3,
        number_diagnoses=9, n_meds_used=3, n_med_changes=2,
        discharge="Skilled nursing facility", insulin="Up", change="Yes", A1Cresult=">8"
    )
}

CSV_COLUMNS = [
    "id", "age_group", "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency", "number_inpatient",
    "number_diagnoses", "n_meds_used", "n_med_changes", "discharge", "admission",
    "diag_1", "insulin", "A1Cresult", "change", "diabetesMed"
]


def clean(raw):
    """Validate a dict of form values -> (row dict, list of errors)."""
    row, errors = {}, []
    try:
        age = int(raw.get("age", DEFAULTS["age"]))
        if not 1 <= age <= 10:
            raise ValueError
        row["age"] = age
    except (TypeError, ValueError):
        errors.append("Age group is not valid.")

    for k, (lo, hi) in LIMITS.items():
        try:
            v = float(raw.get(k, DEFAULTS[k]))
            if not lo <= v <= hi:
                errors.append(f"{LABELS[k]} must be between {lo} and {hi}.")
            row[k] = v
        except (TypeError, ValueError):
            errors.append(f"{LABELS[k]} must be a number.")

    choices = {
        "discharge": DISCHARGE, "admission": ADMISSION, "diag_1": list(DIAG),
        "insulin": INSULIN, "A1Cresult": A1C, "change": YESNO, "diabetesMed": YESNO
    }
    for k, ok in choices.items():
        v = raw.get(k, DEFAULTS[k])
        if v not in ok:
            errors.append(f"{LABELS[k]} is not valid.")
        row[k] = v

    return row, errors


def score(rows):
    """Score a list of cleaned row dicts -> (probabilities, DataFrame)."""
    b, _ = _get_bundle()
    model, num, cat = b["model"], b["NUM"], b["CAT"]
    X = pd.DataFrame(rows)[num + cat].copy()
    X["change"] = X["change"].map({"Yes": "Ch", "No": "No"})
    X["A1Cresult"] = X["A1Cresult"].replace("None", np.nan)
    return model.predict_proba(X)[:, 1], X


def band(p):
    _, m = _get_bundle()
    high = m["hi"]
    b, _ = _get_bundle()
    thr = b["thr"]
    if p >= high:
        return "high"
    if p >= thr:
        return "elevated"
    return "low"


def display(v, k):
    if k == "age":
        return AGES[int(v) - 1]
    if k == "change":
        return "Yes" if v == "Ch" else "No"
    if k == "A1Cresult" and (v is None or (isinstance(v, float) and np.isnan(v))):
        return "Not measured"
    if isinstance(v, (int, float, np.integer, np.floating)):
        return str(int(v)) if float(v).is_integer() else str(round(float(v), 1))
    return str(v)


def explain(X):
    """Contribution of each field to the score compared to an average patient."""
    b, _ = _get_bundle()
    lr = b["lr"]
    cat = b["CAT"]
    pre, lr_step = lr.named_steps["p"], lr.named_steps["m"]
    z = pre.transform(X.iloc[[0]])
    z = z.toarray().ravel() if hasattr(z, "toarray") else np.ravel(z)
    names, coef = pre.get_feature_names_out(), lr_step.coef_[0]
    out = {}
    for n, c, v in zip(names, coef, z):
        if n.startswith("n__"):
            out[n[3:]] = out.get(n[3:], 0) + c * v
        else:
            f = next(f for f in cat if n.startswith("c__" + f + "_"))
            out[f] = out.get(f, 0) + c * v
    for f in cat:
        out[f] = out.get(f, 0) - b["cat_ref"][f]
    items = sorted(out.items(), key=lambda kv: -abs(kv[1]))
    return [
        dict(field=LABELS[k], value=display(X.iloc[0][k], k), effect=round(float(v), 3))
        for k, v in items[:6] if abs(v) > 0.02
    ]


def get_metadata():
    b, m = _get_bundle()
    return {
        "thr": float(b["thr"]),
        "high": float(m["hi"]),
        "base_rate": float(m["base_rate"]),
        "bands": m["bands"],
        "metrics": m
    }


def get_form_context(form_data=None, errors=None):
    d = form_data or DEFAULTS
    errors = errors or []
    t = lambda xs: [(x, x) for x in xs]
    return {
        "d": d,
        "errors": errors,
        "limits": LIMITS,
        "examples": EXAMPLES,
        "ages": [(i + 1, a) for i, a in enumerate(AGES)],
        "admission": t(ADMISSION),
        "discharge": t(DISCHARGE),
        "diag": list(DIAG.items()),
        "insulin": t(INSULIN),
        "a1c": t(A1C),
        "yesno": t(YESNO),
        "labels": LABELS
    }


def process_batch_file(file_obj):
    """Process uploaded CSV file for batch readmission screening."""
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
        raw = {k: r[k] for k in CSV_COLUMNS}
        raw["age"] = raw.pop("age_group")
        raw = {k: ("" if pd.isna(v) else v) for k, v in raw.items()}
        raw["A1Cresult"] = raw["A1Cresult"] or "None"
        row, errs = clean(raw)
        if errs:
            bad.append(f"Row {i + 2}: {errs[0]}")
        rows.append(row)

    if bad:
        msg = "; ".join(bad[:5]) + (" ..." if len(bad) > 5 else "")
        return None, msg

    b, m = _get_bundle()
    thr = b["thr"]
    p, _ = score(rows)
    out = df[["id"]].copy()
    out["risk"] = np.round(p, 4)
    out["level"] = [band(x) for x in p]
    out["follow_up"] = np.where(p >= thr, "yes", "no")
    out = out.sort_values("risk", ascending=False).reset_index(drop=True)
    out.insert(0, "rank", out.index + 1)

    summary = {
        "n": len(out),
        "flagged": int((out["follow_up"] == "yes").sum()),
        "high": int((out["level"] == "high").sum()),
        "threshold": round(thr, 4)
    }

    return {
        "summary": summary,
        "dataframe": out,
        "preview": out.head(25).to_dict("records")
    }, None


def generate_template_csv():
    """Generate example CSV template."""
    high = EXAMPLES["higher"]
    row1 = [
        1, high["age"], high["time_in_hospital"], high["num_lab_procedures"],
        high["num_procedures"], high["num_medications"], high["number_outpatient"],
        high["number_emergency"], high["number_inpatient"], high["number_diagnoses"],
        high["n_meds_used"], high["n_med_changes"], high["discharge"],
        high["admission"], high["diag_1"], high["insulin"], high["A1Cresult"],
        high["change"], high["diabetesMed"]
    ]
    low = EXAMPLES["lower"]
    row2 = [
        2, low["age"], low["time_in_hospital"], low["num_lab_procedures"],
        low["num_procedures"], low["num_medications"], low["number_outpatient"],
        low["number_emergency"], low["number_inpatient"], low["number_diagnoses"],
        low["n_meds_used"], low["n_med_changes"], low["discharge"],
        low["admission"], low["diag_1"], low["insulin"], low["A1Cresult"],
        low["change"], low["diabetesMed"]
    ]
    df = pd.DataFrame([row1, row2], columns=CSV_COLUMNS)
    buf = io.BytesIO(df.to_csv(index=False).encode("utf-8"))
    return buf
