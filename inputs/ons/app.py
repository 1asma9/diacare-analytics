"""DiaCare Analytics - early readmission risk platform (Flask), built on the HbA1c notebook.
Run:  python app.py   then open http://127.0.0.1:5000
"""
import io
import json
import os

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request, send_file

BASE = os.path.dirname(os.path.abspath(__file__))
MD = json.load(open(os.path.join(BASE, "model", "model.json")))
M = json.load(open(os.path.join(BASE, "model", "metrics.json")))
COLS = MD["columns"]
IDX = {c: i for i, c in enumerate(COLS)}
COEF = np.array(MD["coef"])
INTERCEPT = MD["intercept"]
COL_MEAN = np.array([MD["col_mean"][c] for c in COLS])
THR, HIGH, BASE_RATE = M["threshold"], M["hi"], M["base_rate"]

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

AGES = [f"{i}-{i + 10}" for i in range(0, 100, 10)]
from labels import GENDER, RACE, ADMISSION, SOURCE, DISCHARGE, DIAG

INSULIN = ["No", "Steady", "Up", "Down"]
INSULIN_CODE = {"No": 0, "Steady": 1, "Down": 2, "Up": 3}      # same coding as the notebook (insulin_enc)
YESNO = ["Yes", "No"]

# field of the form -> (label shown in "why this score")
LABELS = {"age": "Age", "gender": "Gender", "race": "Race", "admission": "Type of admission",
          "source": "Admission source", "discharge": "Discharge destination", "diag_1": "Main diagnosis",
          "time_in_hospital": "Length of stay", "number_diagnoses": "Number of diagnoses",
          "num_lab_procedures": "Number of lab tests", "num_procedures": "Number of procedures",
          "number_inpatient": "Inpatient stays (last year)", "number_emergency": "Emergency visits (last year)",
          "number_outpatient": "Outpatient visits (last year)", "num_medications": "Number of medications",
          "insulin": "Insulin", "a1c_tested": "HbA1c measured", "glu_tested": "Glucose serum test",
          "change": "Change of medication", "diabetesMed": "Diabetes medication"}

# field of the form -> name of the variable in the model
TO_MODEL = {"age": "age", "gender": "gender", "race": "race", "admission": "admission_type_id",
            "source": "admission_source_id", "discharge": "discharge_disposition_id", "diag_1": "diag_1_group",
            "time_in_hospital": "time_in_hospital", "number_diagnoses": "number_diagnoses",
            "num_lab_procedures": "num_lab_procedures", "num_procedures": "num_procedures",
            "number_inpatient": "number_inpatient", "number_emergency": "number_emergency",
            "number_outpatient": "number_outpatient", "num_medications": "num_medications",
            "insulin": "insulin_enc", "a1c_tested": "a1c_tested", "glu_tested": "glu_tested", "change": "change",
            "diabetesMed": "diabetesMed"}

LIMITS = {"time_in_hospital": (1, 14), "num_lab_procedures": (1, 132), "num_procedures": (0, 6),
          "num_medications": (1, 81), "number_outpatient": (0, 42), "number_emergency": (0, 42),
          "number_inpatient": (0, 12), "number_diagnoses": (1, 16)}

DEFAULTS = {"age": 7, "gender": "Female", "race": "Caucasian", "admission": "1", "source": "7", "discharge": "1",
            "diag_1": "Circulatory", "time_in_hospital": 3, "number_diagnoses": 8, "num_lab_procedures": 44,
            "num_procedures": 1, "number_inpatient": 0, "number_emergency": 0, "number_outpatient": 0,
            "num_medications": 14, "insulin": "No", "a1c_tested": "No", "glu_tested": "No", "change": "No",
            "diabetesMed": "Yes"}

EXAMPLES = {
    "lower": dict(DEFAULTS, age=5, time_in_hospital=2, num_lab_procedures=30, num_medications=9,
                  number_diagnoses=5, diabetesMed="No"),
    "higher": dict(DEFAULTS, age=8, time_in_hospital=7, num_lab_procedures=60, num_procedures=2,
                   num_medications=26, number_outpatient=2, number_emergency=2, number_inpatient=3,
                   number_diagnoses=9, discharge="3", insulin="Up", change="Yes", a1c_tested="Yes"),
}

CHOICES = {"gender": GENDER, "race": RACE, "admission": ADMISSION, "source": SOURCE, "discharge": DISCHARGE,
           "diag_1": [(k, v) for k, v in DIAG.items()], "insulin": [(x, x) for x in INSULIN],
           "a1c_tested": [(x, x) for x in YESNO], "glu_tested": [(x, x) for x in YESNO],
           "change": [(x, x) for x in YESNO], "diabetesMed": [(x, x) for x in YESNO]}


LOOKUP = {k: {**{str(v).lower(): v for v, _ in opts}, **{str(t).lower(): v for v, t in opts}}
          for k, opts in CHOICES.items()}


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
        ok = LOOKUP[k].get(v.lower())          # the value (e.g. "3") or the label (e.g. "Skilled nursing facility")
        if ok is None:
            errors.append(f"{LABELS[k]} is not valid.")
        row[k] = ok if ok is not None else v
    return row, errors


def vectorize(row):
    """Form values -> the 55 columns of the notebook's model 3 (one-hot, numeric columns standardised on the train set)."""
    x = np.zeros(len(COLS))
    for f, name in TO_MODEL.items():
        if name in MD["flag_cols"]:
            x[IDX[name]] = 1.0 if row[f] == "Yes" else 0.0
        elif name in MD["num_model"]:
            v = INSULIN_CODE[row[f]] if f == "insulin" else float(row[f])
            x[IDX[name]] = (v - MD["scale_mean"][name]) / MD["scale_sd"][name]
        else:                                                    # categorical -> one-hot, reference level = all zeros
            level = f"[{(row[f] - 1) * 10}-{row[f] * 10})" if f == "age" else str(row[f])
            col = f"{name}_{level}"
            if col in IDX:
                x[IDX[col]] = 1.0
    return x


def score(rows):
    X = np.array([vectorize(r) for r in rows])
    z = X @ COEF + INTERCEPT
    return 1 / (1 + np.exp(-z)), X


def band(p):
    if p >= HIGH:
        return "high"
    if p >= THR:
        return "elevated"
    return "low"


def explain(x, row):
    """Contribution of each field to the score, compared with an average patient (coefficient x (value - average))."""
    contrib = COEF * (x - COL_MEAN)
    out = {}
    for f, name in TO_MODEL.items():
        out[f] = float(sum(contrib[IDX[c]] for c in MD["fields"][name]))
    items = sorted(out.items(), key=lambda kv: -abs(kv[1]))
    return [dict(field=LABELS[k], value=display(row[k], k), effect=round(v, 3)) for k, v in items[:6]
            if abs(v) > 0.02]


def display(v, k):
    if k == "age":
        return AGES[int(v) - 1]
    if k in CHOICES:
        return dict(CHOICES[k]).get(str(v), str(v))
    if isinstance(v, (int, float, np.integer, np.floating)):
        return str(int(v)) if float(v).is_integer() else str(round(float(v), 1))
    return str(v)


def ctx(d, errors):
    return dict(d=d, errors=errors, limits=LIMITS, examples=EXAMPLES, active="form",
                ages=[(i + 1, a) for i, a in enumerate(AGES)], gender=GENDER, race=RACE, admission=ADMISSION,
                source=SOURCE, discharge=DISCHARGE, diag=list(DIAG.items()),
                insulin=[(x, x) for x in INSULIN], yesno=[(x, x) for x in YESNO])


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
                           share=B_["share"], factors=explain(X[0], row), row=row, active="form", m=M,
                           shown={k: display(row[k], k) for k in LABELS}, labels=LABELS)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    payload = request.get_json(silent=True) or {}
    row, errors = clean(payload)
    if errors:
        return jsonify(error=errors), 400
    p, _ = score([row])
    return jsonify(risk=round(float(p[0]), 4), band=band(float(p[0])), follow_up=bool(p[0] >= THR),
                   threshold=round(THR, 4), average_risk=round(BASE_RATE, 4))


CSV_COLUMNS = ["id", "age_group", "gender", "race", "admission", "source", "discharge", "diag_1", "time_in_hospital",
               "num_lab_procedures", "num_procedures", "num_medications", "number_outpatient", "number_emergency",
               "number_inpatient", "number_diagnoses", "insulin", "a1c_tested", "glu_tested", "change", "diabetesMed"]


@app.route("/batch", methods=["GET", "POST"])
def batch():
    if request.method == "GET":
        return render_template("batch.html", active="batch", result=None, error=None, columns=CSV_COLUMNS)
    f = request.files.get("file")
    if not f or not f.filename.lower().endswith(".csv"):
        return render_template("batch.html", active="batch", result=None, columns=CSV_COLUMNS,
                               error="Please choose a .csv file."), 400
    try:
        df = pd.read_csv(f, dtype=str, keep_default_na=False)
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
        raw = {k: r[k].strip() for k in CSV_COLUMNS}
        raw["age"] = raw.pop("age_group")
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
    rows = []
    for i, ex in enumerate([EXAMPLES["higher"], EXAMPLES["lower"]], start=1):
        ex = {k: (display(v, k) if k in CHOICES else v) for k, v in ex.items()}
        rows.append([i, ex["age"], ex["gender"], ex["race"], ex["admission"], ex["source"], ex["discharge"],
                     ex["diag_1"], ex["time_in_hospital"], ex["num_lab_procedures"], ex["num_procedures"],
                     ex["num_medications"], ex["number_outpatient"], ex["number_emergency"],
                     ex["number_inpatient"], ex["number_diagnoses"], ex["insulin"], ex["a1c_tested"],
                     ex["glu_tested"], ex["change"], ex["diabetesMed"]])
    df = pd.DataFrame(rows, columns=CSV_COLUMNS)
    return send_file(io.BytesIO(df.to_csv(index=False).encode()), mimetype="text/csv", as_attachment=True,
                     download_name="patients_template.csv")


@app.route("/model")
def model_page():
    return render_template("model.html", m=M, active="model", comparison=M["comparison"], thr=THR,
                           odds=M.get("odds", []), labels=LABELS)


if __name__ == "__main__":
    app.run(debug=False, port=int(os.environ.get("PORT", 5000)))
