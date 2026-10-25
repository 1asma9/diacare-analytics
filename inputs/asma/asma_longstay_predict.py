"""Helper for Asma's long-stay model (short vs long stay).
Takes RAW form values and returns the probability of a long stay, exactly as in her notebook:
cap at the 99th percentile -> log(1+x) for medications -> standardise -> model.
Raw fields: admission = number_diagnoses, num_procedures, change ; after_stay adds num_medications, num_lab_procedures.
`change` accepts "Ch"/"Yes"/1 for a medication change and "No"/0 otherwise.
"""
import joblib
import numpy as np
import pandas as pd


def load(path="model_asma_longstay.joblib"):
    return joblib.load(path)


def _change(v):
    return 1 if str(v).strip().lower() in ("ch", "yes", "1", "true") else 0


def prepare(bundle, raw, version="admission"):
    """raw: dict (one patient) or DataFrame (several). Returns the feature table the model expects."""
    X = pd.DataFrame([raw]) if isinstance(raw, dict) else raw.copy()
    cap, sc = bundle["cap_p99"], bundle["scaler"]
    out = pd.DataFrame(index=X.index)
    for c in ("number_diagnoses", "num_procedures", "num_lab_procedures"):
        if c in X:
            v = pd.to_numeric(X[c]).clip(upper=cap[c])
            out[c + "_scaled"] = (v - sc[c][0]) / sc[c][1]
    if "num_medications" in X:
        v = np.log1p(pd.to_numeric(X["num_medications"]).clip(upper=cap["num_medications"]))
        out["log_num_medications_scaled"] = (v - sc["log_num_medications"][0]) / sc["log_num_medications"][1]
    out["change_bin"] = X["change"].map(_change)
    return out[bundle["versions"][version]["features"]]


def predict_proba(bundle, raw, version="admission", model=None):
    model = model or bundle["selected_model"]
    m = bundle["versions"][version]["models"][model]["model"]
    return m.predict_proba(prepare(bundle, raw, version))[:, 1]


def label(bundle, p):
    return bundle["labels"][int(p >= bundle["decision_threshold"])]
