"""DiaCare Discharge — discharge destination predictor (Flask).
Run:  python app.py   then open http://127.0.0.1:5001

Requires:
  - artifacts/model.joblib  (produced by diacare_handoff_v2/train_model.py, then copy here)
  - preprocessing.py        (copy from diacare_handoff_v2/)
"""
import io
import json
import os

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request, send_file

BASE = os.path.dirname(os.path.abspath(__file__))

# ── model artifacts ──────────────────────────────────────────────────────────
art = joblib.load(os.path.join(BASE, "artifacts", "model.joblib"))
W       = np.array(art["class_weights"])
CLASSES = art["classes"]          # ["Autre","Domicile","Domicile avec soins","Transfert"]
THR     = art["threshold"]        # binary Transfert alert threshold

# ── import preprocessing helpers (must be in the same directory) ─────────────
import sys
sys.path.insert(0, BASE)
from preprocessing import MED_COLS, age_to_bracket, align, build_features

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

# ── option lists ─────────────────────────────────────────────────────────────
GENDERS  = ["Female", "Male"]
RACES    = ["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other", "Inconnu"]
ADM_TYPE = [(1, "Emergency"), (2, "Urgent"), (3, "Elective"), (0, "Other / Unknown")]
ADM_SRC  = [
    (7,  "Emergency room"),
    (1,  "Physician referral"),
    (2,  "Self-referral"),
    (4,  "Transfer from hospital"),
    (5,  "Transfer from SNF"),
    (6,  "Transfer from another facility"),
    (11, "Normal delivery (not emergency)"),
    (20, "Other"),
]
PAYER    = ["MC", "HM", "SP", "BC", "MD", "CP", "UN", "CM", "Inconnu"]
SPECIALTY = [
    "Inconnu", "InternalMedicine", "Cardiology", "Emergency/Trauma",
    "Family/GeneralPractice", "Surgery-General", "Orthopedics",
    "Gastroenterology", "Pulmonology", "Nephrology", "Psychiatry", "Urology",
]
INSULIN  = ["No", "Steady", "Up", "Down"]
A1C      = ["None", "Norm", ">7", ">8"]
GLU      = ["None", "Norm", ">200", ">300"]
CHANGE   = [("No", "No"), ("Ch", "Yes")]
YESNO    = ["Yes", "No"]
METFORMIN= ["No", "Steady", "Up", "Down"]

# ── default form values ───────────────────────────────────────────────────────
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
    "home": dict(DEFAULTS,
        age=65, admission_type_id=3, admission_source_id=1,
        time_in_hospital=2, num_lab_procedures=25, num_medications=8,
        diag_1="250", diag_2="401", number_diagnoses=5,
        sejours_anterieurs=0, insulin="Steady", diabetesMed="Yes",
    ),
    "transfer": dict(DEFAULTS,
        age=82, admission_type_id=1, admission_source_id=7,
        time_in_hospital=9, num_lab_procedures=68, num_procedures=3,
        num_medications=22, number_inpatient=2, number_emergency=1,
        diag_1="428", diag_2="428", diag_3="250",
        number_diagnoses=11, A1Cresult=">8",
        change="Ch", insulin="Up", metformin="No",
        sejours_anterieurs=3, medical_specialty="Cardiology",
    ),
}

# ── numeric field limits ──────────────────────────────────────────────────────
LIMITS = {
    "age":                 (0,   100),
    "time_in_hospital":    (1,   14),
    "num_lab_procedures":  (1,   130),
    "num_procedures":      (0,   6),
    "num_medications":     (1,   80),
    "number_outpatient":   (0,   40),
    "number_emergency":    (0,   40),
    "number_inpatient":    (0,   20),
    "number_diagnoses":    (1,   16),
    "sejours_anterieurs":  (0,   50),
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

# ── class display helpers ─────────────────────────────────────────────────────
CLASS_LABEL = {
    "Domicile":            "Home",
    "Domicile avec soins": "Home with care",
    "Transfert":           "Transfer",
    "Autre":               "Other / Unknown",
}
CLASS_CSS = {
    "Domicile":            "b-domicile",
    "Domicile avec soins": "b-soins",
    "Transfert":           "b-transfert",
    "Autre":               "b-autre",
}

# ── model performance figures (from README / professor_QA) ───────────────────
METRICS = {
    "f1_macro":           0.520,
    "transfert_recall":   0.705,
    "transfert_precision":0.462,
    "transfert_auc":      0.834,
    "ci_low":             0.510,
    "ci_high":            0.528,
}

MODEL_COMPARISON = [
    ("Ensemble + class weights (final)", 0.520, 0.545),
    ("Ensemble HGB + RF",                0.501, 0.563),
    ("Gradient boosting",                0.498, 0.545),
    ("Random forest",                    0.479, 0.571),
    ("Logistic regression",              0.453, 0.507),
    ("Decision tree",                    0.424, 0.516),
    ("Baseline (always Domicile)",        0.186, 0.000),
]


# ── helpers ───────────────────────────────────────────────────────────────────
def _clean(raw):
    """Validate form/dict -> (row dict suitable for build_features, list of errors)."""
    row, errors = {}, []

    for k, (lo, hi) in LIMITS.items():
        try:
            v = int(raw.get(k, DEFAULTS[k]))
            if not lo <= v <= hi:
                errors.append(f"{LABELS[k]} must be between {lo} and {hi}.")
            row[k] = v
        except (TypeError, ValueError):
            errors.append(f"{LABELS[k]} must be a whole number.")

    # categorical validations
    checks = {
        "gender":           GENDERS,
        "race":             RACES,
        "payer_code":       PAYER,
        "medical_specialty":SPECIALTY,
        "insulin":          INSULIN,
        "metformin":        METFORMIN,
        "A1Cresult":        A1C,
        "max_glu_serum":    GLU,
        "diabetesMed":      YESNO,
    }
    for k, ok in checks.items():
        v = str(raw.get(k, DEFAULTS[k]))
        if v not in ok:
            v = DEFAULTS[k]   # silently fall back rather than hard-error for select fields
        row[k] = v

    # change: accept "Yes"/"No" from old form, store as "Ch"/"No"
    ch = str(raw.get("change", DEFAULTS["change"]))
    row["change"] = "Ch" if ch in ("Ch", "Yes") else "No"

    # admission type / source (integers)
    try:
        row["admission_type_id"] = int(raw.get("admission_type_id", DEFAULTS["admission_type_id"]))
    except (TypeError, ValueError):
        row["admission_type_id"] = DEFAULTS["admission_type_id"]
    try:
        row["admission_source_id"] = int(raw.get("admission_source_id", DEFAULTS["admission_source_id"]))
    except (TypeError, ValueError):
        row["admission_source_id"] = DEFAULTS["admission_source_id"]

    # diag codes: accept empty as "NA"
    for c in ("diag_1", "diag_2", "diag_3"):
        v = str(raw.get(c, DEFAULTS[c])).strip()
        row[c] = v if v else "NA"

    # medications: everything defaults to "No"; only insulin and metformin come from the form
    for m in MED_COLS:
        row[m] = str(raw.get(m, "No"))

    return row, errors


def _predict(row):
    """dict -> (prediction str, probabilities dict, transfer_prob float, transfer_alert bool)."""
    r = dict(row)
    r["age"] = age_to_bracket(r["age"])

    X, _ = build_features(pd.DataFrame([r]), art["params"])
    X    = align(X, art["all_columns"])

    proba   = (art["hgb"].predict_proba(X) + art["rf"].predict_proba(X))[0] / 2
    pred    = CLASSES[int(np.argmax(proba * W))]
    p_tr    = float(art["hgb_binary"].predict_proba(X)[0, 1])
    probs   = {c: round(float(p), 4) for c, p in zip(CLASSES, proba)}
    return pred, probs, p_tr, p_tr >= THR


def _ctx_form(d, errors):
    """Build the template context for the input form."""
    return dict(
        d=d, errors=errors,
        limits=LIMITS, examples=EXAMPLES,
        active="form",
        genders=GENDERS, races=RACES,
        adm_types=ADM_TYPE, adm_srcs=ADM_SRC,
        payers=PAYER, specialties=SPECIALTY,
        insulins=INSULIN, metformins=METFORMIN,
        a1c_opts=A1C, glu_opts=GLU,
        change_opts=CHANGE, yesno=YESNO,
    )


# ── routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html", **_ctx_form(DEFAULTS, []))


@app.route("/predict", methods=["POST"])
def predict():
    raw = request.form.to_dict()
    row, errors = _clean(raw)
    if errors:
        merged = {**DEFAULTS, **{k: raw.get(k, DEFAULTS.get(k)) for k in raw}}
        merged.update({k: v for k, v in row.items()})
        return render_template("index.html", **_ctx_form(merged, errors)), 400

    pred, probs, p_tr, alert = _predict(row)
    return render_template(
        "result.html",
        pred=pred, probs=probs, p_tr=round(p_tr, 4), alert=alert,
        thr=round(THR, 3),
        class_label=CLASS_LABEL, class_css=CLASS_CSS,
        row=row, labels=LABELS,
        active="form",
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    payload = request.get_json(silent=True) or {}
    row, errors = _clean(payload)
    if errors:
        return jsonify(error=errors), 400
    pred, probs, p_tr, alert = _predict(row)
    return jsonify(
        prediction=pred,
        probabilities=probs,
        transfer_probability=round(p_tr, 4),
        transfer_alert=alert,
        threshold=round(float(THR), 3),
    )


@app.route("/model")
def model_page():
    return render_template(
        "model.html",
        active="model",
        metrics=METRICS,
        comparison=MODEL_COMPARISON,
        thr=round(float(THR), 3),
        class_label=CLASS_LABEL,
        class_css=CLASS_CSS,
    )


# ── CSV batch ─────────────────────────────────────────────────────────────────
CSV_COLUMNS = [
    "id", "age", "gender", "race", "admission_type_id", "admission_source_id",
    "time_in_hospital", "payer_code", "medical_specialty",
    "num_lab_procedures", "num_procedures", "num_medications",
    "number_outpatient", "number_emergency", "number_inpatient",
    "diag_1", "diag_2", "diag_3", "number_diagnoses",
    "max_glu_serum", "A1Cresult", "change", "diabetesMed",
    "sejours_anterieurs", "insulin", "metformin",
]


@app.route("/batch", methods=["GET", "POST"])
def batch():
    if request.method == "GET":
        return render_template("batch.html", active="batch", result=None, error=None, columns=CSV_COLUMNS)

    f = request.files.get("file")
    if not f or not f.filename.lower().endswith(".csv"):
        return render_template("batch.html", active="batch", result=None, columns=CSV_COLUMNS,
                               error="Please choose a .csv file."), 400
    try:
        df = pd.read_csv(f)
    except Exception:
        return render_template("batch.html", active="batch", result=None, columns=CSV_COLUMNS,
                               error="The file could not be read as CSV."), 400

    missing = [c for c in CSV_COLUMNS if c not in df.columns]
    if missing:
        return render_template("batch.html", active="batch", result=None, columns=CSV_COLUMNS,
                               error="Missing columns: " + ", ".join(missing)), 400
    if len(df) > 5000:
        return render_template("batch.html", active="batch", result=None, columns=CSV_COLUMNS,
                               error="Maximum 5 000 patients per file."), 400

    rows, preds_list, bad = [], [], []
    for i, r in df.iterrows():
        raw = {k: ("" if pd.isna(r.get(k)) else r[k]) for k in CSV_COLUMNS}
        raw["diag_3"] = raw["diag_3"] or "NA"
        raw["diag_2"] = raw["diag_2"] or "NA"
        row, errs = _clean(raw)
        if errs:
            bad.append(f"Row {i + 2}: {errs[0]}")
        rows.append(row)

    if bad:
        return render_template("batch.html", active="batch", result=None, columns=CSV_COLUMNS,
                               error="; ".join(bad[:5]) + (" …" if len(bad) > 5 else "")), 400

    # batch predict
    preds_out, alerts = [], []
    for row in rows:
        pred, probs, p_tr, alert = _predict(row)
        preds_out.append((pred, p_tr, alert))

    out = df[["id"]].copy()
    out["prediction"] = [p[0] for p in preds_out]
    out["transfer_prob"] = np.round([p[1] for p in preds_out], 4)
    out["transfer_alert"] = ["yes" if p[2] else "no" for p in preds_out]
    out = out.sort_values("transfer_prob", ascending=False).reset_index(drop=True)
    out.insert(0, "rank", out.index + 1)

    app.config["LAST_BATCH"] = out
    summary = dict(
        n=len(out),
        transfer_alerts=int((out["transfer_alert"] == "yes").sum()),
        pct_transfer=round((out["transfer_alert"] == "yes").mean() * 100, 1),
    )
    return render_template("batch.html", active="batch", error=None, columns=CSV_COLUMNS,
                           result=dict(summary=summary, rows=out.head(30).to_dict("records"),
                                       class_label=CLASS_LABEL, class_css=CLASS_CSS))


@app.route("/batch/download")
def batch_download():
    out = app.config.get("LAST_BATCH")
    if out is None:
        return "No file scored yet.", 404
    buf = io.BytesIO(out.to_csv(index=False).encode())
    return send_file(buf, mimetype="text/csv", as_attachment=True, download_name="discharge_predictions.csv")


@app.route("/batch/template")
def batch_template():
    ex = EXAMPLES["transfer"]
    ex2 = EXAMPLES["home"]
    rows = []
    for i, e in enumerate([ex, ex2], 1):
        rows.append([i] + [e.get(c, "") for c in CSV_COLUMNS[1:]])
    df = pd.DataFrame(rows, columns=CSV_COLUMNS)
    return send_file(io.BytesIO(df.to_csv(index=False).encode()), mimetype="text/csv",
                     as_attachment=True, download_name="patients_template.csv")


if __name__ == "__main__":
    app.run(debug=True, port=5001)
