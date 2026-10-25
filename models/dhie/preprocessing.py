"""Shared preprocessing = exact logic of lab_modelisation_v3 / lab_evaluation notebooks.
Used by train_model.py (training) and app.py (single-patient inference)."""
import numpy as np
import pandas as pd

CLASSES = ["Domicile", "Domicile avec soins", "Transfert", "Autre"]
TRANSFERT = {2, 3, 4, 5, 15, 22, 23, 24, 27, 28, 29, 30}

MED_COLS = ["metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
            "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
            "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
            "insulin", "glyburide-metformin", "glipizide-metformin",
            "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone"]

CAT_COLS = ["race", "gender", "max_glu_serum", "A1Cresult", "medical_specialty", "payer_code",
            "diag_1_grp", "diag_2_grp", "diag_3_grp", "diag_1_3d", "diag_2_3d",
            "admission_type", "admission_source"]


def groupe_sortie(c):
    if c == 1: return "Domicile"
    if c in (6, 8): return "Domicile avec soins"
    if c in TRANSFERT: return "Transfert"
    return "Autre"


def famille_cim9(code):
    s = str(code)
    if s == "NA" or s.startswith(("V", "E")): return "autres"
    if s.startswith("250"): return "diabete"
    n = float(s)
    if 390 <= n <= 459 or n == 785: return "circulatoire"
    if 460 <= n <= 519 or n == 786: return "respiratoire"
    if 520 <= n <= 579 or n == 787: return "digestif"
    if 800 <= n <= 999: return "lesions"
    if 710 <= n <= 739: return "musculo_squelettique"
    if 580 <= n <= 629 or n == 788: return "genito_urinaire"
    if 140 <= n <= 239: return "tumeurs"
    return "autres"


def age_to_bracket(years):
    """Website input (years) -> dataset bracket string, e.g. 73 -> '[70-80)'."""
    b = min(max(int(years) // 10 * 10, 0), 90)
    return f"[{b}-{b + 10})"


def load_clean(path):
    """CSV -> cleaned dataframe `d` (still raw-valued) and target y. Mirrors notebook section 1."""
    raw = pd.read_csv(path, na_values="?", keep_default_na=False, low_memory=False)
    # number of PRIOR stays of the patient (rank of encounter_id); the total would leak the future
    raw["sejours_anterieurs"] = raw.groupby("patient_nbr")["encounter_id"].rank(method="first") - 1
    d = raw.drop(columns=["weight", "encounter_id", "patient_nbr", "examide", "citoglipton"])
    for c in ["race", "medical_specialty", "payer_code"]:
        d[c] = d[c].fillna("Inconnu")
    d = d[d["gender"] != "Unknown/Invalid"].dropna(subset=["diag_1"]).copy()
    d[["diag_2", "diag_3"]] = d[["diag_2", "diag_3"]].fillna("NA")
    d = d.reset_index(drop=True)
    y = d["discharge_disposition_id"].map(groupe_sortie)
    return d, y


def build_features(d, params=None):
    """Raw-valued dataframe -> numeric features.
    params=None  : training mode, learns the 'top-N categories' lists and returns them.
    params=dict  : inference mode, reuses the lists learned on the training data."""
    fit = params is None
    params = {} if fit else params
    # anti-leakage: the raw target and the future event never enter X
    X = d.drop(columns=[c for c in ["discharge_disposition_id", "readmitted"] if c in d.columns]).copy()

    for c in ["diag_1", "diag_2", "diag_3"]:
        X[c + "_grp"] = X[c].map(famille_cim9)
    for c in ["diag_1", "diag_2"]:
        c3 = X[c].astype(str).str.split(".").str[0]
        if fit:
            params["top_" + c] = c3.value_counts().head(40).index.tolist()
        X[c + "_3d"] = c3.where(c3.isin(params["top_" + c]), "rare")
    X["diag23_manquants"] = (X["diag_2"] == "NA").astype(int) + (X["diag_3"] == "NA").astype(int)
    X = X.drop(columns=["diag_1", "diag_2", "diag_3"])

    X["admission_type"] = X["admission_type_id"].map(
        lambda c: "urgence" if c in (1, 2, 7) else ("elective" if c == 3 else "inconnu"))
    X["admission_source"] = X["admission_source_id"].map(
        lambda c: "reference_medicale" if c in (1, 2, 3) else ("urgences" if c == 7 else
                  ("transfert_etablissement" if c in (4, 5, 6, 10, 18, 22, 25, 26) else "autre_inconnu")))
    X = X.drop(columns=["admission_type_id", "admission_source_id"])

    for col, n in [("medical_specialty", 12), ("payer_code", 8)]:
        if fit:
            params["top_" + col] = X[col].value_counts().head(n).index.tolist()
        X[col] = X[col].where(X[col].isin(params["top_" + col]), "Autre")

    if not pd.api.types.is_numeric_dtype(X["age"]):
        X["age"] = X["age"].str.extract(r"\[(\d+)-")[0].astype(int) + 5

    # constructed variables (computed while medications are still strings)
    X["n_meds_actifs"] = (X[MED_COLS] != "No").sum(axis=1)
    X["n_meds_modifies"] = X[MED_COLS].isin(["Up", "Down"]).sum(axis=1)
    X["visites_prec"] = X["number_outpatient"] + X["number_emergency"] + X["number_inpatient"]
    X["labs_par_jour"] = X["num_lab_procedures"] / X["time_in_hospital"]
    X["meds_par_jour"] = X["num_medications"] / X["time_in_hospital"]
    X["age_x_duree"] = X["age"] * X["time_in_hospital"]

    X[MED_COLS] = (X[MED_COLS] != "No").astype(int)
    X["change"] = (X["change"] == "Ch").astype(int)
    X["diabetesMed"] = (X["diabetesMed"] == "Yes").astype(int)
    X = pd.get_dummies(X, columns=CAT_COLS, dtype=int)
    return X, params


def align(X, all_columns):
    """Force the exact training columns/order (dummy columns absent for one patient = 0)."""
    return X.reindex(columns=all_columns, fill_value=0)
