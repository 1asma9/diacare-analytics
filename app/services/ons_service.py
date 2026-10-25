"""Service for Ons's HbA1c & Treatment Decision Model (Logistic Regression)."""
import io
import json
import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "models", "ons", "model.json")
METRICS_PATH = os.path.join(BASE_DIR, "models", "ons", "metrics.json")

_MODEL_DATA = None
_METRICS = None

def _get_artifacts():
    global _MODEL_DATA, _METRICS
    if _MODEL_DATA is None:
        with open(MODEL_PATH, "r", encoding="utf-8") as f:
            _MODEL_DATA = json.load(f)
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            _METRICS = json.load(f)
    return _MODEL_DATA, _METRICS

AGES = [f"{i}-{i + 10}" for i in range(0, 100, 10)]

GENDER = [("Female", "Female"), ("Male", "Male")]
RACE = [
    ("Caucasian", "Caucasian"), ("AfricanAmerican", "African American"),
    ("Hispanic", "Hispanic"), ("Rare", "Asian"), ("Other", "Other"),
    ("Unknown", "Not recorded")
]
ADMISSION = [
    ("1", "Emergency"), ("2", "Urgent"), ("3", "Elective"),
    ("5", "Not available"), ("6", "Not recorded"), ("Rare", "Other")
]
SOURCE = [
    ("7", "Emergency room"), ("1", "Physician referral"),
    ("2", "Clinic referral"), ("4", "Transfer from a hospital"),
    ("6", "Transfer from another health care facility"),
    ("17", "Not recorded"), ("Rare", "Other")
]
DISCHARGE = [
    ("1", "Home"), ("6", "Home with home-health care"),
    ("3", "Skilled nursing facility"), ("2", "Other short-term hospital"),
    ("22", "Rehabilitation facility"), ("5", "Other inpatient care"),
    ("18", "Not recorded"), ("25", "Not mapped"), ("Rare", "Other")
]
DIAG = {
    "Circulatory": "Circulatory (heart, vessels)", "Diabetes": "Diabetes",
    "Respiratory": "Respiratory", "Digestive": "Digestive",
    "Injury": "Injury / poisoning", "Musculoskeletal": "Musculoskeletal",
    "Genitourinary": "Genitourinary", "Neoplasms": "Neoplasms (cancer)",
    "Other": "Other"
}

INSULIN = ["No", "Steady", "Up", "Down"]
INSULIN_CODE = {"No": 0, "Steady": 1, "Down": 2, "Up": 3}
YESNO = ["Yes", "No"]

LABELS = {
    "age": "Age", "gender": "Gender", "race": "Race", "admission": "Type of admission",
    "source": "Admission source", "discharge": "Discharge destination", "diag_1": "Main diagnosis",
    "time_in_hospital": "Length of stay", "number_diagnoses": "Number of diagnoses",
    "num_lab_procedures": "Number of lab tests", "num_procedures": "Number of procedures",
    "number_inpatient": "Inpatient stays (last year)", "number_emergency": "Emergency visits (last year)",
    "number_outpatient": "Outpatient visits (last year)", "num_medications": "Number of medications",
    "insulin": "Insulin", "a1c_tested": "HbA1c measured", "glu_tested": "Glucose serum test",
    "change": "Change of medication", "diabetesMed": "Diabetes medication"
}

TO_MODEL = {
    "age": "age", "gender": "gender", "race": "race", "admission": "admission_type_id",
    "source": "admission_source_id", "discharge": "discharge_disposition_id", "diag_1": "diag_1_group",
    "time_in_hospital": "time_in_hospital", "number_diagnoses": "number_diagnoses",
    "num_lab_procedures": "num_lab_procedures", "num_procedures": "num_procedures",
    "number_inpatient": "number_inpatient", "number_emergency": "number_emergency",
    "number_outpatient": "number_outpatient", "num_medications": "num_medications",
    "insulin": "insulin_enc", "a1c_tested": "a1c_tested", "glu_tested": "glu_tested",
    "change": "change", "diabetesMed": "diabetesMed"
}

LIMITS = {
    "time_in_hospital": (1, 14), "num_lab_procedures": (1, 132), "num_procedures": (0, 6),
    "num_medications": (1, 81), "number_outpatient": (0, 42), "number_emergency": (0, 42),
    "number_inpatient": (0, 12), "number_diagnoses": (1, 16)
}

DEFAULTS = {
    "age": 7, "gender": "Female", "race": "Caucasian", "admission": "1", "source": "7",
    "discharge": "1", "diag_1": "Circulatory", "time_in_hospital": 3, "number_diagnoses": 8,
    "num_lab_procedures": 44, "num_procedures": 1, "number_inpatient": 0, "number_emergency": 0,
    "number_outpatient": 0, "num_medications": 14, "insulin": "No", "a1c_tested": "No",
    "glu_tested": "No", "change": "No", "diabetesMed": "Yes"
}

EXAMPLES = {
    "lower": dict(
        DEFAULTS, age=5, time_in_hospital=2, num_lab_procedures=30, num_medications=9,
        number_diagnoses=5, diabetesMed="No"
    ),
    "higher": dict(
        DEFAULTS, age=8, time_in_hospital=7, num_lab_procedures=60, num_procedures=2,
        num_medications=26, number_outpatient=2, number_emergency=2, number_inpatient=3,
        number_diagnoses=9, discharge="3", insulin="Up", change="Yes", a1c_tested="Yes"
    )
}

CHOICES = {
    "gender": GENDER, "race": RACE, "admission": ADMISSION, "source": SOURCE,
    "discharge": DISCHARGE, "diag_1": [(k, v) for k, v in DIAG.items()],
    "insulin": [(x, x) for x in INSULIN], "a1c_tested": [(x, x) for x in YESNO],
    "glu_tested": [(x, x) for x in YESNO], "change": [(x, x) for x in YESNO],
    "diabetesMed": [(x, x) for x in YESNO]
}

LOOKUP = {
    k: {**{str(v).lower(): v for v, _ in opts}, **{str(t).lower(): v for v, t in opts}}
    for k, opts in CHOICES.items()
}

CSV_COLUMNS = [
    "id", "age_group", "gender", "race", "admission", "source", "discharge", "diag_1",
    "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
    "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses",
    "insulin", "a1c_tested", "glu_tested", "change", "diabetesMed"
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

    for k in CHOICES:
        v = str(raw.get(k, DEFAULTS[k])).strip()
        ok = LOOKUP[k].get(v.lower())
        if ok is None:
            errors.append(f"{LABELS[k]} is not valid.")
        row[k] = ok if ok is not None else v

    return row, errors


def vectorize(row):
    """Vectorize a row to the 55-column features expected by Ons's logistic regression."""
    md, _ = _get_artifacts()
    cols = md["columns"]
    idx = {c: i for i, c in enumerate(cols)}
    x = np.zeros(len(cols))
    for f, name in TO_MODEL.items():
        if name in md["flag_cols"]:
            x[idx[name]] = 1.0 if row[f] == "Yes" else 0.0
        elif name in md["num_model"]:
            v = INSULIN_CODE[row[f]] if f == "insulin" else float(row[f])
            x[idx[name]] = (v - md["scale_mean"][name]) / md["scale_sd"][name]
        else:
            level = f"[{(row[f] - 1) * 10}-{row[f] * 10})" if f == "age" else str(row[f])
            col = f"{name}_{level}"
            if col in idx:
                x[idx[col]] = 1.0
    return x


def score(rows):
    """Score rows -> (probabilities, X matrix)."""
    md, _ = _get_artifacts()
    coef = np.array(md["coef"])
    intercept = md["intercept"]
    X = np.array([vectorize(r) for r in rows])
    z = X @ coef + intercept
    return 1 / (1 + np.exp(-z)), X


def band(p):
    _, m = _get_artifacts()
    if p >= m["hi"]:
        return "high"
    if p >= m["threshold"]:
        return "elevated"
    return "low"


def display(v, k):
    if k == "age":
        return AGES[int(v) - 1]
    if k in CHOICES:
        return dict(CHOICES[k]).get(str(v), str(v))
    if isinstance(v, (int, float, np.integer, np.floating)):
        return str(int(v)) if float(v).is_integer() else str(round(float(v), 1))
    return str(v)


def explain(x, row):
    """Contribution of each field to the score."""
    md, _ = _get_artifacts()
    cols = md["columns"]
    idx = {c: i for i, c in enumerate(cols)}
    col_mean = np.array([md["col_mean"][c] for c in cols])
    coef = np.array(md["coef"])

    contrib = coef * (x - col_mean)
    out = {}
    for f, name in TO_MODEL.items():
        out[f] = float(sum(contrib[idx[c]] for c in md["fields"][name]))
    items = sorted(out.items(), key=lambda kv: -abs(kv[1]))
    return [
        dict(field=LABELS[k], value=display(row[k], k), effect=round(v, 3))
        for k, v in items[:6] if abs(v) > 0.02
    ]


def get_metadata():
    _, m = _get_artifacts()
    return {
        "thr": float(m["threshold"]),
        "high": float(m["hi"]),
        "base_rate": float(m["base_rate"]),
        "bands": m["bands"],
        "comparison": m.get("comparison", []),
        "odds": m.get("odds", []),
        "metrics": m
    }


def get_form_context(form_data=None, errors=None):
    d = form_data or DEFAULTS
    errors = errors or []
    return {
        "d": d,
        "errors": errors,
        "limits": LIMITS,
        "examples": EXAMPLES,
        "ages": [(i + 1, a) for i, a in enumerate(AGES)],
        "gender": GENDER,
        "race": RACE,
        "admission": ADMISSION,
        "source": SOURCE,
        "discharge": DISCHARGE,
        "diag": list(DIAG.items()),
        "insulin": [(x, x) for x in INSULIN],
        "yesno": [(x, x) for x in YESNO],
        "labels": LABELS
    }


def process_batch_file(file_obj):
    """Process uploaded CSV file for batch HbA1c & readmission evaluation."""
    try:
        df = pd.read_csv(file_obj, dtype=str, keep_default_na=False)
    except Exception:
        return None, "The file could not be read as CSV."

    missing = [c for c in CSV_COLUMNS if c not in df.columns]
    if missing:
        return None, f"Missing columns: {', '.join(missing)}"

    if len(df) > 5000:
        return None, "Maximum 5,000 patients per file."

    rows, bad = [], []
    for i, r in df.iterrows():
        raw = {k: r[k].strip() for k in CSV_COLUMNS}
        raw["age"] = raw.pop("age_group")
        row, errs = clean(raw)
        if errs:
            bad.append(f"Row {i + 2}: {errs[0]}")
        rows.append(row)

    if bad:
        msg = "; ".join(bad[:5]) + (" ..." if len(bad) > 5 else "")
        return None, msg

    _, m = _get_artifacts()
    thr = m["threshold"]
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
    """Generate example CSV template for Ons's tool."""
    rows = []
    for i, ex in enumerate([EXAMPLES["higher"], EXAMPLES["lower"]], start=1):
        ex_disp = {k: (display(v, k) if k in CHOICES else v) for k, v in ex.items()}
        rows.append([
            i, ex_disp["age"], ex_disp["gender"], ex_disp["race"], ex_disp["admission"],
            ex_disp["source"], ex_disp["discharge"], ex_disp["diag_1"],
            ex_disp["time_in_hospital"], ex_disp["num_lab_procedures"],
            ex_disp["num_procedures"], ex_disp["num_medications"],
            ex_disp["number_outpatient"], ex_disp["number_emergency"],
            ex_disp["number_inpatient"], ex_disp["number_diagnoses"],
            ex_disp["insulin"], ex_disp["a1c_tested"], ex_disp["glu_tested"],
            ex_disp["change"], ex_disp["diabetesMed"]
        ])
    df = pd.DataFrame(rows, columns=CSV_COLUMNS)
    buf = io.BytesIO(df.to_csv(index=False).encode("utf-8"))
    return buf
