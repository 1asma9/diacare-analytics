"""Rebuilds the model used by the platform from the notebook's data.
Same cleaning, same 80/20 split (random_state=42), same logistic regression ("model 3") and same
threshold rule as the notebook (cells 46-54). Writes:
    model/model.json          coefficients of the logistic regression (no pickle, no version warning)
    model/metrics.json        test-set metrics, risk bands, model comparison, odds ratios
    sample_patients.csv       300 patients of the TEST set, ready for the "Several patients" page
Usage:  python train_model.py path/to/df_clean.csv       (needs scikit-learn, xgboost, statsmodels)
"""
import json, os, re, sys, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from labels import RACE, ADMISSION, SOURCE, DISCHARGE, DIAG
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score, recall_score,
                             precision_score, balanced_accuracy_score, confusion_matrix)
warnings.filterwarnings("ignore")

CSV = sys.argv[1] if len(sys.argv) > 1 else "df_clean.csv"
HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- cleaning (notebook cells 3-22, 46)
df = pd.read_csv(CSV, keep_default_na=False)          # keep "None" as text (test not done)
df["early_readmit"] = (df["readmitted"] == "<30").astype(int)
df["a1c_tested"] = (df["A1Cresult"] != "None").astype(int)
df["glu_tested"] = (df["max_glu_serum"] != "None").astype(int)
med_cols = ["metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride", "acetohexamide", "glipizide",
            "glyburide", "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose", "miglitol", "troglitazone",
            "tolazamide", "examide", "citoglipton", "insulin", "glyburide-metformin", "glipizide-metformin",
            "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone"]
med_cols = [c for c in med_cols if c in df.columns]
df["insulin_enc"] = df["insulin"].map({"No": 0, "Steady": 1, "Down": 2, "Up": 3})
df = df.drop(columns=med_cols + ["readmitted"])
df["change"] = df["change"].map({"No": 0, "Ch": 1})
df["diabetesMed"] = df["diabetesMed"].map({"No": 0, "Yes": 1})


def group_diag(code):
    code = str(code).strip()
    if code.startswith(("V", "E")) or code in ("?", "", "Unknown", "nan"):
        return "Other"
    try:
        v = float(code)
    except ValueError:
        return "Other"
    n = int(v)
    if 250 <= v < 251: return "Diabetes"
    if (390 <= n <= 459) or n == 785: return "Circulatory"
    if (460 <= n <= 519) or n == 786: return "Respiratory"
    if (520 <= n <= 579) or n == 787: return "Digestive"
    if 800 <= n <= 999: return "Injury"
    if 710 <= n <= 739: return "Musculoskeletal"
    if (580 <= n <= 629) or n == 788: return "Genitourinary"
    if 140 <= n <= 239: return "Neoplasms"
    return "Other"


df["diag_1_group"] = df["diag_1"].apply(group_diag)

# ---------------------------------------------------------------- model 3 (notebook cells 48-54)
flag_cols = ["a1c_tested", "change", "diabetesMed", "glu_tested"]
num_model = ["insulin_enc", "num_medications", "time_in_hospital", "number_inpatient", "number_emergency",
             "number_diagnoses", "number_outpatient", "num_lab_procedures", "num_procedures"]
cat_model = ["age", "gender", "race", "admission_type_id", "discharge_disposition_id", "admission_source_id",
             "diag_1_group"]
data = df[flag_cols + num_model + cat_model + ["early_readmit"]].copy()
for c in ["admission_type_id", "discharge_disposition_id", "admission_source_id"]:
    data[c] = data[c].astype(str)


def merge_rare(s, min_frac=0.01):
    freq = s.value_counts(normalize=True)
    rare = freq[freq < min_frac].index
    return s.where(~s.isin(rare), "Rare")


for c in ["race", "admission_type_id", "discharge_disposition_id", "admission_source_id"]:
    data[c] = merge_rare(data[c])
X_all = pd.get_dummies(data.drop(columns="early_readmit"), columns=cat_model, drop_first=True, dtype=int)
y_all = data["early_readmit"]
Xtr, Xte, ytr, yte = train_test_split(X_all, y_all, test_size=0.2, stratify=y_all, random_state=42)
Xtr, Xte = Xtr.copy(), Xte.copy()
sc3 = StandardScaler()
Xtr[num_model] = sc3.fit_transform(Xtr[num_model])
Xte[num_model] = sc3.transform(Xte[num_model])

clf = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
p_te = clf.predict_proba(Xte)[:, 1]

# threshold: chosen on the TRAIN set (5-fold probabilities), best recall with accuracy >= 62 %
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
p_cv = cross_val_predict(LogisticRegression(max_iter=2000), Xtr, ytr, cv=cv, method="predict_proba")[:, 1]
best = None
for q in np.arange(0.40, 0.96, 0.05):
    t = float(np.quantile(p_cv, q))
    pr = (p_cv >= t).astype(int)
    acc, rec = accuracy_score(ytr, pr), recall_score(ytr, pr)
    if acc >= 0.62 and (best is None or rec > best[1]):
        best = (t, rec, acc)
thr = best[0]
hi = float(np.quantile(p_cv, 0.80))                    # "high risk" = top 20 % of the training scores
pred = (p_te >= thr).astype(int)

m = dict(auc_roc=roc_auc_score(yte, p_te), auc_pr=average_precision_score(yte, p_te),
         accuracy=accuracy_score(yte, pred), recall=recall_score(yte, pred), precision=precision_score(yte, pred),
         balanced_accuracy=balanced_accuracy_score(yte, pred), flagged=float(pred.mean()), threshold=thr, hi=hi,
         base_rate=float(ytr.mean()), accuracy_always_no=float(1 - yte.mean()),
         confusion=confusion_matrix(yte, pred).tolist(), n_total=int(len(df)), n_train=int(len(ytr)),
         n_test=int(len(yte)), n_features=int(X_all.shape[1]))
m["bands"] = {}
for name, mask in [("low", p_te < thr), ("elevated", (p_te >= thr) & (p_te < hi)), ("high", p_te >= hi)]:
    m["bands"][name] = dict(share=float(mask.mean()), rate=float(yte[mask].mean()),
                            capture=float(yte[mask].sum() / yte.sum()))

# ---------------------------------------------------------------- model comparison (AUC on the same test patients)
comparison = [dict(name="Logistic regression - model 3, full (used by the platform)",
                   auc_roc=m["auc_roc"], auc_pr=m["auc_pr"], final=True)]
try:  # models 1 and 2 (notebook cells 36 and 43)
    d = df.copy()
    base_cols = ["a1c_tested", "change", "diabetesMed", "insulin_enc", "num_medications", "glu_tested"]
    for name, cols in [("Logistic regression - model 2, adjusted for severity",
                        base_cols + ["time_in_hospital", "number_inpatient", "number_emergency", "number_diagnoses"]),
                       ("Logistic regression - model 1, study variables only", base_cols)]:
        A, y_ = d[cols], d["early_readmit"]
        a_tr, a_te, b_tr, b_te = train_test_split(A, y_, test_size=0.2, stratify=y_, random_state=42)
        a_tr, a_te = a_tr.copy(), a_te.copy()
        sc_cols = [c for c in cols if c in ("insulin_enc", "num_medications", "time_in_hospital",
                                            "number_inpatient", "number_emergency", "number_diagnoses")]
        s = StandardScaler()
        a_tr[sc_cols] = s.fit_transform(a_tr[sc_cols])
        a_te[sc_cols] = s.transform(a_te[sc_cols])
        pp = LogisticRegression(max_iter=1000).fit(a_tr, b_tr).predict_proba(a_te)[:, 1]
        comparison.append(dict(name=name, auc_roc=roc_auc_score(b_te, pp), auc_pr=average_precision_score(b_te, pp)))
except Exception as e:
    print("models 1/2 skipped:", e)
try:  # XGBoost with the parameters found in the notebook (cell 60)
    from xgboost import XGBClassifier
    clean = lambda cols: [re.sub(r"[\[\]<]", "_", str(c)) for c in cols]
    Xtr_x, Xte_x = Xtr.copy(), Xte.copy()
    Xtr_x.columns, Xte_x.columns = clean(Xtr.columns), clean(Xte.columns)
    ratio = (ytr == 0).sum() / (ytr == 1).sum()
    xgb = XGBClassifier(scale_pos_weight=ratio, eval_metric="logloss", random_state=42, n_jobs=1, subsample=0.8,
                        reg_lambda=1, n_estimators=300, min_child_weight=5, max_depth=3, learning_rate=0.05,
                        gamma=1, colsample_bytree=0.8).fit(Xtr_x, ytr)
    px = xgb.predict_proba(Xte_x)[:, 1]
    comparison.append(dict(name="XGBoost (tuned)", auc_roc=roc_auc_score(yte, px),
                           auc_pr=average_precision_score(yte, px)))
except Exception as e:
    print("xgboost skipped:", e)
m["comparison"] = sorted(comparison, key=lambda r: -r["auc_roc"])

# ---------------------------------------------------------------- odds ratios of the study variables (statsmodels, model 3)
try:
    import statsmodels.api as sm
    lg = sm.Logit(ytr, sm.add_constant(Xtr)).fit(disp=0)
    ci = lg.conf_int()
    m["odds"] = [dict(var=v, OR=float(np.exp(lg.params[v])), lo=float(np.exp(ci.loc[v, 0])),
                      hi=float(np.exp(ci.loc[v, 1])), p=float(lg.pvalues[v])) for v in flag_cols]
except Exception as e:
    print("odds ratios skipped:", e)

# ---------------------------------------------------------------- what the app needs to score a patient
cols = list(X_all.columns)
fields = {}
for f in flag_cols + num_model:
    fields[f] = [f]
for f in cat_model:
    fields[f] = [c for c in cols if c.startswith(f + "_")]
ref = {f: sorted(set(data[f].astype(str)) - {c[len(f) + 1:] for c in fields[f]})[0] for f in cat_model}
model = dict(columns=cols, coef=clf.coef_[0].tolist(), intercept=float(clf.intercept_[0]),
             num_model=num_model, flag_cols=flag_cols, cat_model=cat_model,
             scale_mean=dict(zip(num_model, sc3.mean_.tolist())), scale_sd=dict(zip(num_model, sc3.scale_.tolist())),
             col_mean=dict(zip(cols, Xtr.mean().tolist())), fields=fields, reference=ref,
             levels={f: sorted(set(data[f].astype(str))) for f in cat_model})
os.makedirs(os.path.join(HERE, "model"), exist_ok=True)
json.dump(model, open(os.path.join(HERE, "model", "model.json"), "w"))
json.dump(m, open(os.path.join(HERE, "model", "metrics.json"), "w"), indent=1)

# 300 real patients of the TEST set for the "Several patients" page (same columns as the CSV template of the app)
raw = pd.read_csv(CSV, keep_default_na=False).loc[Xte.index].sample(300, random_state=1)
keep = lambda s, ok: s.astype(str).where(s.astype(str).isin(ok), "Rare")
sample = pd.DataFrame({
    "id": raw.index, "age_group": raw["age"].str.extract(r"\[(\d+)-")[0].astype(int) // 10 + 1,
    "gender": raw["gender"], "race": raw["race"].replace({"Asian": "Rare"}).map(dict(RACE)),
    "admission": keep(raw["admission_type_id"], ["1", "2", "3", "5", "6"]).map(dict(ADMISSION)),
    "source": keep(raw["admission_source_id"], ["7", "1", "2", "4", "6", "17"]).map(dict(SOURCE)),
    "discharge": keep(raw["discharge_disposition_id"], ["1", "6", "3", "2", "22", "5", "18", "25"]).map(dict(DISCHARGE)),
    "diag_1": raw["diag_1"].apply(group_diag).map(DIAG),
    "time_in_hospital": raw["time_in_hospital"], "num_lab_procedures": raw["num_lab_procedures"],
    "num_procedures": raw["num_procedures"], "num_medications": raw["num_medications"],
    "number_outpatient": raw["number_outpatient"], "number_emergency": raw["number_emergency"],
    "number_inpatient": raw["number_inpatient"], "number_diagnoses": raw["number_diagnoses"],
    "insulin": raw["insulin"], "a1c_tested": np.where(raw["A1Cresult"] != "None", "Yes", "No"),
    "glu_tested": np.where(raw["max_glu_serum"] != "None", "Yes", "No"),
    "change": np.where(raw["change"] == "Ch", "Yes", "No"), "diabetesMed": raw["diabetesMed"]})
sample.to_csv(os.path.join(HERE, "sample_patients.csv"), index=False)
print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items() if not isinstance(v, (list, dict))})
print([(c["name"][:42], round(c["auc_roc"], 3), round(c["auc_pr"], 3)) for c in m["comparison"]])
