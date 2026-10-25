"""Trains the model used by the platform and saves it in model/bundle.joblib.
Same cleaning and same 80/20 split as the notebook (random_state=42); the platform model only uses
the fields a clinician can fill in at discharge."""
import json, joblib, numpy as np, pandas as pd, warnings
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier, StackingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score
warnings.filterwarnings("ignore")

df = pd.read_csv("../SEIF_own/df_model_seif.csv")
y = df.pop("early_readmit")

DISP = {"1": "Home", "6": "Home with home-health care", "3": "Skilled nursing facility", "2": "Other short-term hospital",
        "22": "Rehabilitation facility", "23": "Long-term care hospital", "5": "Other inpatient care"}
ADM = {"1": "Emergency", "2": "Urgent", "3": "Elective"}
df["discharge"] = df["discharge_disposition_id"].astype(str).map(DISP).fillna("Other")
df["admission"] = df["admission_type_id"].astype(str).map(ADM).fillna("Other")

NUM = ["age", "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications", "number_outpatient",
       "number_emergency", "number_inpatient", "number_diagnoses", "n_meds_used", "n_med_changes"]
CAT = ["discharge", "admission", "diag_1", "insulin", "A1Cresult", "change", "diabetesMed"]
X = df[NUM + CAT]

idx_tr, idx_te = train_test_split(X.index, test_size=0.2, stratify=y, random_state=42)
Xtr, Xte, ytr, yte = X.loc[idx_tr], X.loc[idx_te], y.loc[idx_tr], y.loc[idx_te]

pre = ColumnTransformer([("n", StandardScaler(), NUM), ("c", OneHotEncoder(handle_unknown="ignore"), CAT)])
lr = Pipeline([("p", pre), ("m", LogisticRegression(C=0.05, class_weight="balanced", max_iter=2000))])
hgb = Pipeline([("p", pre), ("m", HistGradientBoostingClassifier(learning_rate=0.05, max_iter=200, max_depth=4,
                 min_samples_leaf=40, l2_regularization=5, class_weight="balanced", random_state=0))])
et = Pipeline([("p", pre), ("m", ExtraTreesClassifier(n_estimators=300, min_samples_leaf=30, max_features=0.4,
                 class_weight="balanced_subsample", n_jobs=-1, random_state=0))])
stack = StackingClassifier([("lr", lr), ("hgb", hgb), ("et", et)], final_estimator=LogisticRegression(C=0.1),
                           cv=StratifiedKFold(5, shuffle=True, random_state=7), stack_method="predict_proba", n_jobs=1)
stack.fit(Xtr, ytr)

# threshold: from out-of-fold probabilities of the training set, target recall 0.65
from sklearn.model_selection import cross_val_predict
oof = cross_val_predict(stack, Xtr, ytr, cv=StratifiedKFold(5, shuffle=True, random_state=42), method="predict_proba")[:, 1]
order = np.argsort(-oof); cum = np.cumsum(ytr.values[order]) / ytr.sum()
k = np.searchsorted(cum, 0.65) + 1
thr = float(oof[order][k - 1])
p = stack.predict_proba(Xte)[:, 1]; pred = p >= thr
res = dict(auc_roc=roc_auc_score(yte, p), auc_pr=average_precision_score(yte, p), recall=recall_score(yte, pred),
           precision=precision_score(yte, pred), flagged=float(pred.mean()), threshold=thr, base_rate=float(ytr.mean()),
           oof_cdf=np.sort(oof).tolist()[::50])
for q in (0.5, 0.65, 0.8):
    kk = np.searchsorted(cum, q) + 1; res[f"thr_{q}"] = float(oof[order][kk - 1])
# risk bands from test-set quantiles of the score
res["band_med"] = float(np.quantile(p, 0.6)); res["band_high"] = float(np.quantile(p, 0.8))
print({k: round(v, 3) for k, v in res.items() if not isinstance(v, list)})
# group rates on the test set for the "how common is this" panel
res["top20_rate"] = float(yte[p >= res["band_high"]].mean()); res["bottom40_rate"] = float(yte[p < res["band_med"]].mean())
res["mid_rate"] = float(yte[(p >= res["band_med"]) & (p < res["band_high"])].mean())
print(res["top20_rate"], res["mid_rate"], res["bottom40_rate"])
# defaults and ranges for the form
res["defaults"] = {c: float(Xtr[c].median()) for c in NUM}
res["ranges"] = {c: [float(Xtr[c].min()), float(Xtr[c].quantile(.995)) if c not in ("age",) else 10.0] for c in NUM}
# linear model for per-patient explanation: contribution of each field
lr.fit(Xtr, ytr)
# average effect of each categorical field over the training patients (reference for the explanation)
Z = pre.transform(Xtr); names = pre.get_feature_names_out(); coef = lr.named_steps["m"].coef_[0]
freq = np.asarray(Z.mean(axis=0)).ravel()
cat_ref = {c: float(sum(coef[i] * freq[i] for i, n in enumerate(names) if n.startswith("c__" + c + "_"))) for c in CAT}
import os; os.makedirs("model", exist_ok=True)
joblib.dump(dict(model=stack, lr=lr, NUM=NUM, CAT=CAT, thr=thr, DISP=DISP, ADM=ADM, train_mean=Xtr[NUM].mean().to_dict(),
                 diag_levels=sorted(Xtr["diag_1"].unique()), cat_ref=cat_ref), "model/bundle.joblib")
hi = res["band_high"]
res["hi"] = hi; res["bands"] = {}
for n, mask in [("low", p < thr), ("elevated", (p >= thr) & (p < hi)), ("high", p >= hi)]:
    res["bands"][n] = dict(share=float(mask.mean()), rate=float(yte[mask].mean()), capture=float(yte[mask].sum() / yte.sum()))
res.pop("oof_cdf", None)
json.dump(res, open("model/metrics.json", "w"))
