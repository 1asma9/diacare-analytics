"""DiaCare Analytics - early readmission risk platform (Flask).
Run:  python app.py   then open http://127.0.0.1:5000
"""
import io
import json
import os

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request, send_file

BASE = os.path.dirname(os.path.abspath(__file__))
B = joblib.load(os.path.join(BASE, "model", "bundle.joblib"))
M = json.load(open(os.path.join(BASE, "model", "metrics.json")))
MODEL, LR, NUM, CAT, THR = B["model"], B["lr"], B["NUM"], B["CAT"], B["thr"]
HIGH = M["hi"]
BASE_RATE = M["base_rate"]

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

AGES = [f"{i}-{i + 10}" for i in range(0, 100, 10)]
DISCHARGE = ["Home", "Home with home-health care", "Skilled nursing facility", "Other short-term hospital",
             "Rehabilitation facility", "Long-term care hospital", "Other inpatient care", "Other"]
ADMISSION = ["Emergency", "Urgent", "Elective", "Other"]
DIAG = {"Circulatory": "Circulatory (heart, vessels)", "Diabetes": "Diabetes", "Respiratory": "Respiratory",
        "Digestive": "Digestive", "Injury": "Injury / poisoning", "Musculoskeletal": "Musculoskeletal",
        "Genitourinary": "Genitourinary", "Neoplasms": "Neoplasms (cancer)", "Other": "Other", "Unknown": "Unknown"}
INSULIN = ["No", "Steady", "Up", "Down"]
A1C = ["None", "Norm", ">7", ">8"]
YESNO = ["Yes", "No"]

# field -> (label shown in "why this score")
LABELS = {"age": "Age", "time_in_hospital": "Length of stay", "num_lab_procedures": "Number of lab tests",
          "num_procedures": "Number of procedures", "num_medications": "Number of medications",
          "number_outpatient": "Outpatient visits (last year)", "number_emergency": "Emergency visits (last year)",
          "number_inpatient": "Inpatient stays (last year)", "number_diagnoses": "Number of diagnoses",
          "n_meds_used": "Diabetes drugs prescribed", "n_med_changes": "Drug dose changes",
          "discharge": "Discharge destination", "admission": "Type of admission", "diag_1": "Main diagnosis",
          "insulin": "Insulin", "A1Cresult": "HbA1c result", "change": "Change of medication",
          "diabetesMed": "Diabetes medication"}

LIMITS = {"time_in_hospital": (1, 14), "num_lab_procedures": (1, 130), "num_procedures": (0, 6),
          "num_medications": (1, 80), "number_outpatient": (0, 40), "number_emergency": (0, 40),
          "number_inpatient": (0, 20), "number_diagnoses": (1, 16), "n_meds_used": (0, 10), "n_med_changes": (0, 6)}

DEFAULTS = {"age": 7, "time_in_hospital": 4, "num_lab_procedures": 43, "num_procedures": 1, "num_medications": 15,
            "number_outpatient": 0, "number_emergency": 0, "number_inpatient": 0, "number_diagnoses": 8,
            "n_meds_used": 1, "n_med_changes": 0, "discharge": "Home", "admission": "Emergency",
            "diag_1": "Circulatory", "insulin": "No", "A1Cresult": "None", "change": "No", "diabetesMed": "Yes"}

EXAMPLES = {
    "lower": dict(DEFAULTS, age=5, time_in_hospital=2, num_lab_procedures=30, num_medications=9, number_diagnoses=5,
                  diabetesMed="No", insulin="No"),
    "higher": dict(DEFAULTS, age=8, time_in_hospital=7, num_lab_procedures=60, num_procedures=2, num_medications=26,
                   number_outpatient=2, number_emergency=2, number_inpatient=3, number_diagnoses=9,
                   n_meds_used=3, n_med_changes=2, discharge="Skilled nursing facility", insulin="Up",
                   change="Yes", A1Cresult=">8"),
}


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
    choices = {"discharge": DISCHARGE, "admission": ADMISSION, "diag_1": list(DIAG), "insulin": INSULIN,
               "A1Cresult": A1C, "change": YESNO, "diabetesMed": YESNO}
    for k, ok in choices.items():
        v = raw.get(k, DEFAULTS[k])
        if v not in ok:
            errors.append(f"{LABELS[k]} is not valid.")
        row[k] = v
    # the model was trained with Ch/No and Yes/No coded as in the original data
    return row, errors


def score(rows):
    X = pd.DataFrame(rows)[NUM + CAT].copy()
    X["change"] = X["change"].map({"Yes": "Ch", "No": "No"})
    X["A1Cresult"] = X["A1Cresult"].replace("None", np.nan)
    return MODEL.predict_proba(X)[:, 1], X


def band(p):
    if p >= HIGH:
        return "high"
    if p >= THR:
        return "elevated"
    return "low"


def explain(X):
    """Contribution of each field to the score, compared with an average patient (logistic regression view)."""
    pre, lr = LR.named_steps["p"], LR.named_steps["m"]
    z = pre.transform(X.iloc[[0]])
    z = z.toarray().ravel() if hasattr(z, "toarray") else np.ravel(z)
    names, coef = pre.get_feature_names_out(), lr.coef_[0]
    out = {}
    for n, c, v in zip(names, coef, z):
        if n.startswith("n__"):
            out[n[3:]] = out.get(n[3:], 0) + c * v
        else:
            f = next(f for f in CAT if n.startswith("c__" + f + "_"))
            out[f] = out.get(f, 0) + c * v
    for f in CAT:
        out[f] = out.get(f, 0) - B["cat_ref"][f]
    items = sorted(out.items(), key=lambda kv: -abs(kv[1]))
    return [dict(field=LABELS[k], value=display(X.iloc[0][k], k), effect=round(float(v), 3)) for k, v in items[:6]
            if abs(v) > 0.02]


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


def ctx(d, errors):
    t = lambda xs: [(x, x) for x in xs]
    return dict(d=d, errors=errors, limits=LIMITS, examples=EXAMPLES, active="form",
                ages=[(i + 1, a) for i, a in enumerate(AGES)], admission=t(ADMISSION), discharge=t(DISCHARGE),
                diag=list(DIAG.items()), insulin=t(INSULIN), a1c=t(A1C), yesno=t(YESNO))


@app.route("/")
def index():
    return render_template("index.html", **ctx(DEFAULTS, []))


@app.route("/predict", methods=["POST"])
def predict():
    row, errors = clean(request.form)
    if errors:
        return render_template("index.html", **ctx({**DEFAULTS, **request.form.to_dict()}, errors)), 400
    p, X = score([row])
    p = float(p[0])
    b = band(p)
    B_ = M["bands"][b]
    return render_template("result.html", p=p, band=b, thr=THR, high=HIGH, base=BASE_RATE, rate=B_["rate"],
                           share=B_["share"], factors=explain(X), row=row, active="form",
                           shown={k: display(X.iloc[0][k], k) for k in NUM + CAT}, labels=LABELS)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    payload = request.get_json(silent=True) or {}
    row, errors = clean(payload)
    if errors:
        return jsonify(error=errors), 400
    p, _ = score([row])
    return jsonify(risk=round(float(p[0]), 4), band=band(float(p[0])), follow_up=bool(p[0] >= THR),
                   threshold=round(THR, 4), average_risk=round(BASE_RATE, 4))


CSV_COLUMNS = ["id", "age_group", "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
               "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses", "n_meds_used",
               "n_med_changes", "discharge", "admission", "diag_1", "insulin", "A1Cresult", "change", "diabetesMed"]


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
                               error="Maximum 5000 patients per file."), 400
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
        return render_template("batch.html", active="batch", result=None, columns=CSV_COLUMNS,
                               error="; ".join(bad[:5]) + (" ..." if len(bad) > 5 else "")), 400
    p, _ = score(rows)
    out = df[["id"]].copy()
    out["risk"] = np.round(p, 4)
    out["level"] = [band(x) for x in p]
    out["follow_up"] = np.where(p >= THR, "yes", "no")
    out = out.sort_values("risk", ascending=False).reset_index(drop=True)
    out.insert(0, "rank", out.index + 1)
    app.config["LAST_BATCH"] = out
    summary = dict(n=len(out), flagged=int((out["follow_up"] == "yes").sum()),
                   high=int((out["level"] == "high").sum()))
    return render_template("batch.html", active="batch", error=None, columns=CSV_COLUMNS,
                           result=dict(summary=summary, rows=out.head(25).to_dict("records")))


@app.route("/batch/download")
def batch_download():
    out = app.config.get("LAST_BATCH")
    if out is None:
        return "No file scored yet.", 404
    buf = io.BytesIO(out.to_csv(index=False).encode())
    return send_file(buf, mimetype="text/csv", as_attachment=True, download_name="risk_scores.csv")


@app.route("/batch/template")
def batch_template():
    ex = EXAMPLES["higher"]
    row = [1, ex["age"], ex["time_in_hospital"], ex["num_lab_procedures"], ex["num_procedures"], ex["num_medications"],
           ex["number_outpatient"], ex["number_emergency"], ex["number_inpatient"], ex["number_diagnoses"],
           ex["n_meds_used"], ex["n_med_changes"], ex["discharge"], ex["admission"], ex["diag_1"], ex["insulin"],
           ex["A1Cresult"], ex["change"], ex["diabetesMed"]]
    low = EXAMPLES["lower"]
    row2 = [2, low["age"], low["time_in_hospital"], low["num_lab_procedures"], low["num_procedures"],
            low["num_medications"], low["number_outpatient"], low["number_emergency"], low["number_inpatient"],
            low["number_diagnoses"], low["n_meds_used"], low["n_med_changes"], low["discharge"], low["admission"],
            low["diag_1"], low["insulin"], low["A1Cresult"], low["change"], low["diabetesMed"]]
    df = pd.DataFrame([row, row2], columns=CSV_COLUMNS)
    return send_file(io.BytesIO(df.to_csv(index=False).encode()), mimetype="text/csv", as_attachment=True,
                     download_name="patients_template.csv")


COMPARISON = [  # name, AUC-ROC, AUC-PR (test set, notebook)
    ("Stacking ensemble (final)", 0.660, 0.192), ("Extra trees", 0.652, 0.190),
    ("Average of 8 models", 0.659, 0.186), ("XGBoost", 0.656, 0.182), ("Gradient boosting (tuned)", 0.654, 0.178),
    ("Random forest (tuned)", 0.650, 0.180), ("Logistic regression", 0.647, 0.173), ("Decision tree", 0.623, 0.150),
    ("Neural network (MLP)", 0.616, 0.142), ("GB + SMOTE", 0.591, 0.122)]


@app.route("/model")
def model_page():
    return render_template("model.html", m=M, active="model", comparison=COMPARISON, thr=THR)


if __name__ == "__main__":
    app.run(debug=False, port=int(os.environ.get("PORT", 5000)))
