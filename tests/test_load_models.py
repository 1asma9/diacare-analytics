"""
Phase-0 load test: opens all six models in the shared virtual environment
and runs one (or two) example patient(s) through each.

For Seif and Ons the results are compared against the owner's own
scoring code imported directly from inputs/.

Run from the project root:
    .venv\\Scripts\\python tests\\test_load_models.py
"""

import importlib.util
import json
import os
import re
import sys
import warnings

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

# ── helpers ──────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_mod(name: str, filepath: str):
    """Load an owner's app.py without executing its __main__ block."""
    folder = os.path.dirname(os.path.abspath(filepath))
    if folder not in sys.path:
        sys.path.insert(0, folder)
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PASS = "PASS"
FAIL = "FAIL"
results: list[dict] = []


def check(label: str, actual, expected, tol: float = 0.001):
    ok = abs(float(actual) - float(expected)) <= tol
    status = PASS if ok else FAIL
    results.append({"label": label, "status": status, "actual": actual, "expected": expected})
    print(f"  [{status}] {label}: got {actual:.4f}, expected ~{expected:.4f}")
    return ok


# ── 1. Seif - readmission stacking model ─────────────────────────────────────
print("\n=== 1. Seif (readmission stacking) ===")
seif = load_mod("seif_app", os.path.join(ROOT, "inputs", "seif", "app.py"))

p_default, _ = seif.score([seif.DEFAULTS])
check("Seif default patient", p_default[0], 0.0702, tol=0.005)

p_higher, _ = seif.score([seif.EXAMPLES["higher"]])
check("Seif higher-risk patient", p_higher[0], 0.280, tol=0.010)

print(f"  THR={seif.THR:.4f}  HIGH={seif.HIGH:.4f}  AUC~0.65")

# ── 2. Ons - HbA1c logistic regression ───────────────────────────────────────
print("\n=== 2. Ons (HbA1c logistic regression) ===")
ons = load_mod("ons_app", os.path.join(ROOT, "inputs", "ons", "app.py"))

p_o_default, _ = ons.score([ons.DEFAULTS])
p_o_higher, _ = ons.score([ons.EXAMPLES["higher"]])
print(f"  Default:  {p_o_default[0]*100:.2f}%")
print(f"  Higher:   {p_o_higher[0]*100:.2f}%")
print(f"  THR={ons.THR:.4f}  AUC={ons.M['auc_roc']:.4f}")
results.append({"label": "Ons loads and scores", "status": PASS, "actual": p_o_default[0], "expected": "n/a"})

# HbA1c odds ratio (from metrics.json)
for o in ons.M.get("odds", []):
    if o["var"] == "a1c_tested":
        check("Ons HbA1c OR 0.89-0.90", o["OR"], 0.904, tol=0.010)

# ── 3. Dhie - discharge destination ──────────────────────────────────────────
print("\n=== 3. Dhie (discharge destination) ===")
dhie = load_mod("dhie_app", os.path.join(ROOT, "inputs", "dhie", "app.py"))

row_home, _ = dhie._clean(dhie.EXAMPLES["home"])
pred_home, probs_home, p_tr_home, alert_home = dhie._predict(row_home)
print(f"  Home example -> pred={pred_home}, p_transfer={p_tr_home*100:.2f}%, alert={alert_home}")
assert pred_home == "Domicile", f"Expected 'Domicile' got '{pred_home}'"
assert not alert_home, "Home example should NOT trigger transfer alert"
results.append({"label": "Dhie home example predicts Domicile", "status": PASS, "actual": pred_home, "expected": "Domicile"})

row_tr, _ = dhie._clean(dhie.EXAMPLES["transfer"])
pred_tr, probs_tr, p_tr_tr, alert_tr = dhie._predict(row_tr)
print(f"  Transfer example -> pred={pred_tr}, p_transfer={p_tr_tr*100:.2f}%, alert={alert_tr}")
assert pred_tr == "Transfert", f"Expected 'Transfert' got '{pred_tr}'"
assert alert_tr, "Transfer example SHOULD trigger transfer alert"
results.append({"label": "Dhie transfer example predicts Transfert", "status": PASS, "actual": pred_tr, "expected": "Transfert"})

# ── 4. Roua - medication load XGBoost ────────────────────────────────────────
print("\n=== 4. Roua (medication load XGBoost) ===")
sys.path.insert(0, os.path.join(ROOT, "app", "loaders"))
from xgb_compat import load as compat_load  # noqa: E402

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    b_roua = compat_load(os.path.join(ROOT, "models", "roua", "model_bundle.joblib"))

ROUA_COLS = b_roua["columns"]
ROUA_SCALER = b_roua["scaler"]
ROUA_TOP_SPEC = b_roua["top_specialties"]
ROUA_MEANS = b_roua["train_means"]
ROUA_MODEL = b_roua["model"]
ROUA_CLASS_NAMES = b_roua["class_names"]

def _famille(code):
    c = str(code)
    if c.startswith("250"): return "Diabete"
    if len(c) > 0 and c[0] in "VE": return "Autre"
    try: n = float(c)
    except: return "Autre"
    if 390<=n<=459 or n==785: return "Circulatoire"
    if 460<=n<=519 or n==786: return "Respiratoire"
    if 520<=n<=579 or n==787: return "Digestif"
    if 800<=n<=999: return "Lesions"
    if 710<=n<=739: return "Musculo"
    if 580<=n<=629 or n==788: return "Genito_urinaire"
    if 140<=n<=239: return "Tumeurs"
    return "Autre"

def prepare_roua(raw_df: pd.DataFrame) -> pd.DataFrame:
    d = raw_df.copy()
    for c in ["diag_1", "diag_2", "diag_3"]:
        if c in d.columns:
            d[c] = d[c].apply(_famille)
        else:
            d[c] = "Autre"
    if "medical_specialty" in d.columns:
        d["medical_specialty"] = d["medical_specialty"].where(d["medical_specialty"].isin(ROUA_TOP_SPEC), "Autre")
    else:
        d["medical_specialty"] = "Autre"
    d["gender"]      = (d.get("gender", pd.Series(["Female"])).iloc[0] == "Male") * 1
    d["change"]      = d["change"].isin(["Ch", "Yes"]).astype(int)
    d["diabetesMed"] = (d["diabetesMed"] == "Yes").astype(int)
    for c in ["admission_type_id", "admission_source_id"]:
        if c in d.columns:
            d[c] = d[c].astype(str)
    cont = list(ROUA_MEANS.keys())
    for c in cont:
        if c not in d.columns:
            d[c] = ROUA_MEANS[c]
    d_enc = pd.get_dummies(d, drop_first=True)
    d_enc.columns = [re.sub(r"[\[\]<>,:\"{}\s]", "_", col) for col in d_enc.columns]
    X = d_enc.reindex(columns=ROUA_COLS, fill_value=0).copy()
    X[cont] = ROUA_SCALER.transform(d[cont])
    return X

df_sample = pd.read_csv(os.path.join(ROOT, "inputs", "data", "df_clean.csv"), nrows=3)
X_roua = prepare_roua(df_sample)
preds_roua = ROUA_MODEL.predict(X_roua)
print(f"  Roua predictions for 3 patients: {[ROUA_CLASS_NAMES[p] for p in preds_roua]}")
results.append({"label": "Roua loads and scores", "status": PASS, "actual": "ok", "expected": "ok"})

# ── 5. Moez - patient clustering ─────────────────────────────────────────────
print("\n=== 5. Moez (patient clustering) ===")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    b_moez = compat_load(os.path.join(ROOT, "models", "moez", "cluster_bundle.joblib"))

with open(os.path.join(ROOT, "models", "moez", "cluster_profiles.json")) as f:
    profiles = json.load(f)

# Verify the known cluster split ~ 62/38
pct0 = float(profiles["0"]["pct"])
pct1 = float(profiles["1"]["pct"])
print(f"  Cluster 0: {pct0:.1f}%  Cluster 1: {pct1:.1f}%")
check("Moez cluster 0 ~ 62%", pct0, 62.0, tol=2.0)
check("Moez cluster 1 ~ 38%", pct1, 38.0, tol=2.0)

# ── 6. Asma - long stay prediction ───────────────────────────────────────────
print("\n=== 6. Asma (long stay > 3 days) ===")
asma_bundle = compat_load(os.path.join(ROOT, "models", "asma", "model_asma_longstay.joblib"))

sys.path.insert(0, os.path.join(ROOT, "models", "asma"))
import asma_longstay_predict as asma_pred  # noqa: E402

# Admission version (3 inputs)
p_adm = asma_pred.predict_proba(asma_bundle,
    {"number_diagnoses": 8, "num_procedures": 1, "change": "No"},
    version="admission")
print(f"  Admission version proba: {p_adm[0]:.4f}  label: {asma_pred.label(asma_bundle, p_adm[0])}")
results.append({"label": "Asma admission version scores", "status": PASS, "actual": p_adm[0], "expected": "n/a"})

# After-stay version (5 inputs)
p_stay = asma_pred.predict_proba(asma_bundle,
    {"num_medications": 15, "num_lab_procedures": 43, "number_diagnoses": 8, "num_procedures": 1, "change": "No"},
    version="after_stay")
print(f"  After-stay version proba: {p_stay[0]:.4f}  label: {asma_pred.label(asma_bundle, p_stay[0])}")
results.append({"label": "Asma after-stay version scores", "status": PASS, "actual": p_stay[0], "expected": "n/a"})

# ── summary ──────────────────────────────────────────────────────────────────
print("\n" + "="*60)
n_pass = sum(1 for r in results if r["status"] == PASS)
n_fail = sum(1 for r in results if r["status"] == FAIL)
print(f"TOTAL: {n_pass} PASS  {n_fail} FAIL")
if n_fail > 0:
    print("\nFailed checks:")
    for r in results:
        if r["status"] == FAIL:
            print(f"  {r['label']}: got {r['actual']}, expected {r['expected']}")
    sys.exit(1)
