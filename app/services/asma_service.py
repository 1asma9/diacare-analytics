"""Service for Asma's Length of Stay (>3 Days) Prediction Model."""
import io
import os
import warnings
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "models", "asma", "model_asma_longstay.joblib")

import sys
LOADERS_DIR = os.path.join(BASE_DIR, "app", "loaders")
ASMA_DIR = os.path.join(BASE_DIR, "models", "asma")
for p in (LOADERS_DIR, ASMA_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from xgb_compat import load as compat_load
import asma_longstay_predict as asma_pred

_BUNDLE = None

def _get_bundle():
    global _BUNDLE
    if _BUNDLE is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _BUNDLE = compat_load(MODEL_PATH)
    return _BUNDLE


LIMITS = {
    "number_diagnoses": (1, 16),
    "num_procedures": (0, 6),
    "num_medications": (1, 81),
    "num_lab_procedures": (1, 132)
}

LABELS = {
    "number_diagnoses": "Total diagnoses count",
    "num_procedures": "Surgical procedures count",
    "change": "Medication change during stay",
    "num_medications": "Total medications administered",
    "num_lab_procedures": "Number of lab tests performed"
}

DEFAULTS = {
    "number_diagnoses": 8,
    "num_procedures": 1,
    "change": "No",
    "num_medications": 15,
    "num_lab_procedures": 43,
    "version": "admission"
}

EXAMPLES = {
    "short": {
        "admission": dict(number_diagnoses=4, num_procedures=0, change="No"),
        "after_stay": dict(number_diagnoses=4, num_procedures=0, change="No", num_medications=8, num_lab_procedures=25)
    },
    "long": {
        "admission": dict(number_diagnoses=10, num_procedures=3, change="Yes"),
        "after_stay": dict(number_diagnoses=10, num_procedures=3, change="Yes", num_medications=25, num_lab_procedures=65)
    }
}

CSV_COLUMNS_ADMISSION = ["id", "number_diagnoses", "num_procedures", "change"]
CSV_COLUMNS_AFTER_STAY = ["id", "number_diagnoses", "num_procedures", "change", "num_medications", "num_lab_procedures"]


def clean(raw, version="admission"):
    """Clean and validate form data for Asma's model."""
    row, errors = {}, []
    required_num = ["number_diagnoses", "num_procedures"]
    if version == "after_stay":
        required_num.extend(["num_medications", "num_lab_procedures"])

    for k in required_num:
        lo, hi = LIMITS[k]
        try:
            v = float(raw.get(k, DEFAULTS[k]))
            if not lo <= v <= hi:
                errors.append(f"{LABELS[k]} must be between {lo} and {hi}.")
            row[k] = v
        except (TypeError, ValueError):
            errors.append(f"{LABELS[k]} must be a valid number.")

    ch = str(raw.get("change", DEFAULTS["change"]))
    row["change"] = "Yes" if ch in ("Yes", "Ch", "1", 1) else "No"
    row["version"] = version
    return row, errors


def predict_single(row, version="admission"):
    """Predict long stay probability for a single patient."""
    bundle = _get_bundle()
    thr = float(bundle["decision_threshold"])

    p = float(asma_pred.predict_proba(bundle, row, version=version)[0])
    is_long = bool(p >= thr)
    predicted_label = "Long stay (> 3 days)" if is_long else "Short stay (≤ 3 days)"

    return {
        "version": version,
        "probability": round(p, 4),
        "percentage": round(p * 100, 1),
        "is_long": is_long,
        "label": predicted_label,
        "badge": "b-high" if is_long else "b-low",
        "threshold": thr,
        "advice": (
            "Anticipate bed allocation for >3 days. Begin discharge barrier assessment early."
            if is_long else
            "Routine short stay expected (≤ 3 days). Standard admission bed scheduling."
        )
    }


def get_form_context(form_data=None, errors=None, version="admission"):
    d = form_data or DEFAULTS
    errors = errors or []
    return {
        "d": d,
        "errors": errors,
        "limits": LIMITS,
        "examples": EXAMPLES,
        "labels": LABELS,
        "version": version
    }


def process_batch_file(file_obj, version="admission"):
    """Process uploaded CSV file for batch length of stay prediction."""
    try:
        df = pd.read_csv(file_obj)
    except Exception:
        return None, "The file could not be read as CSV."

    cols = CSV_COLUMNS_AFTER_STAY if version == "after_stay" else CSV_COLUMNS_ADMISSION
    missing = [c for c in cols if c not in df.columns]
    if missing:
        return None, f"Missing columns: {', '.join(missing)}"

    if len(df) > 5000:
        return None, "Maximum 5,000 patients per file."

    rows, bad = [], []
    for i, r in df.iterrows():
        raw = {k: ("" if pd.isna(r.get(k)) else r[k]) for k in cols}
        row, errs = clean(raw, version=version)
        if errs:
            bad.append(f"Row {i + 2}: {errs[0]}")
        rows.append(row)

    if bad:
        msg = "; ".join(bad[:5]) + (" ..." if len(bad) > 5 else "")
        return None, msg

    bundle = _get_bundle()
    probs = asma_pred.predict_proba(bundle, pd.DataFrame(rows), version=version)
    thr = float(bundle["decision_threshold"])

    out = df[["id"]].copy()
    out["prob_long_stay"] = np.round(probs, 4)
    out["predicted_stay"] = ["Long stay (>3d)" if p >= thr else "Short stay (≤3d)" for p in probs]
    out["alert"] = ["yes" if p >= thr else "no" for p in probs]

    out = out.sort_values("prob_long_stay", ascending=False).reset_index(drop=True)
    out.insert(0, "rank", out.index + 1)

    summary = {
        "n": len(out),
        "long_stays": int((out["alert"] == "yes").sum()),
        "pct_long": round((out["alert"] == "yes").mean() * 100, 1),
        "version": version
    }

    return {
        "summary": summary,
        "dataframe": out,
        "preview": out.head(25).to_dict("records"),
        "version": version
    }, None


def generate_template_csv(version="admission"):
    """Generate example CSV template for Asma's tool."""
    cols = CSV_COLUMNS_AFTER_STAY if version == "after_stay" else CSV_COLUMNS_ADMISSION
    ex_short = EXAMPLES["short"][version]
    ex_long = EXAMPLES["long"][version]

    r1 = [1] + [ex_long.get(c, "") for c in cols[1:]]
    r2 = [2] + [ex_short.get(c, "") for c in cols[1:]]

    df = pd.DataFrame([r1, r2], columns=cols)
    buf = io.BytesIO(df.to_csv(index=False).encode("utf-8"))
    return buf
