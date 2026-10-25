"""
Unified Patient Summary Service — Phase 3.

Orchestrates all 6 models for a single patient in one call.
Returns a structured dict consumed by patient_summary/index.html.

Role access:
  - Doctor:    all 6 models
  - Reception: LOS (Asma) + Discharge (Dhie) + Medication (Roua) — 3 models
"""

from __future__ import annotations
import traceback

# ── lazy-import each service to avoid loading all models at import time ──────
def _seif():
    import app.services.seif_service as s
    return s

def _ons():
    import app.services.ons_service as s
    return s

def _dhie():
    import app.services.dhie_service as s
    return s

def _roua():
    import app.services.roua_service as s
    return s

def _moez():
    import app.services.moez_service as s
    return s

def _asma():
    import app.services.asma_service as s
    return s


# ──────────────────────────────────────────────────────────────────────────────
#  FIELD CATALOGUE
#  Merged superset of all 6 models' fields that must appear in the form.
#  Fields are grouped into sections for rendering.
# ──────────────────────────────────────────────────────────────────────────────

# Age groups (1-indexed) shared by Seif & Ons
AGE_GROUPS = [(i, f"{(i-1)*10}–{i*10} yrs") for i in range(1, 11)]

# Shared option lists
GENDERS      = ["Female", "Male"]
YESNO        = ["Yes", "No"]
INSULIN_OPTS = ["No", "Steady", "Up", "Down"]
METFORMIN_OPTS = ["No", "Steady", "Up", "Down"]
A1C_OPTS_SEIF = ["None", "Norm", ">7", ">8"]        # Seif / Dhie (no "NotTested")
A1C_OPTS_ONS  = ["Yes", "No"]                        # Ons uses binary a1c_tested
GLU_OPTS     = ["None", "Norm", ">200", ">300"]

DIAG_GROUPS_SEIF = {
    "Circulatory": "Circulatory (heart, vessels)",
    "Diabetes": "Diabetes",
    "Respiratory": "Respiratory",
    "Digestive": "Digestive",
    "Injury": "Injury / poisoning",
    "Musculoskeletal": "Musculoskeletal",
    "Genitourinary": "Genitourinary",
    "Neoplasms": "Neoplasms (cancer)",
    "Other": "Other",
    "Unknown": "Unknown",
}
RACES = [
    ("Caucasian", "Caucasian"),
    ("AfricanAmerican", "African American"),
    ("Hispanic", "Hispanic"),
    ("Rare", "Asian / Other"),
    ("Other", "Other"),
    ("Unknown", "Not recorded"),
]
ADM_TYPES = [(1, "Emergency"), (2, "Urgent"), (3, "Elective"), (4, "Other")]
ADM_SRCS  = [
    (7, "Emergency Room"), (1, "Physician Referral"), (2, "Clinic Referral"),
    (4, "Transfer from hospital"), (5, "Transfer from SNF"),
    (6, "Transfer from another facility"), (17, "Other / Unknown"),
]
DISCHARGE_DHIE_OPTS = [
    (1, "Home"), (6, "Home with home-health care"),
    (3, "Skilled nursing facility"), (2, "Other short-term hospital"),
    (22, "Rehabilitation facility"), (5, "Other inpatient care"),
    (18, "Not recorded"),
]
PAYERS = ["MC", "HM", "SP", "BC", "MD", "CP", "UN", "CM", "Inconnu"]
SPECIALTIES = [
    "Unknown", "InternalMedicine", "Family/GeneralPractice", "Emergency/Trauma",
    "Cardiology", "Surgery-General", "Orthopedics", "Orthopedics-Reconstructive",
    "Radiologist", "Nephrology", "Gastroenterology", "Pulmonology", "Other",
]

DEFAULTS = {
    # Identity / demographics
    "age":               7,          # age group index (Seif/Ons)
    "age_num":           65,         # numeric age years (Dhie/Moez)
    "gender":            "Female",
    "race":              "Caucasian",
    "payer_code":        "MC",
    "medical_specialty": "InternalMedicine",
    # Stay
    "time_in_hospital":  4,
    # Utilisation
    "num_lab_procedures":  43,
    "num_procedures":       1,
    "num_medications":     15,
    "number_outpatient":    0,
    "number_emergency":     0,
    "number_inpatient":     0,
    "number_diagnoses":     8,
    # Diabetes metrics
    "n_meds_used":    1,
    "n_meds_changed": 0,
    # Diagnoses (ICD-9 free text for Roua/Dhie, grouped for Seif/Ons/Moez)
    "diag_1":     "428",
    "diag_1_grp": "Circulatory",      # for Seif / Ons / Moez
    "diag_2":     "250",
    "diag_3":     "NA",
    # Categorical clinical
    "admission_type_id":       1,
    "admission_source_id":     7,
    "discharge_disposition_id": 1,
    "A1Cresult":     "None",
    "max_glu_serum": "None",
    "insulin":       "No",
    "metformin":     "No",
    "change":        "No",
    "diabetesMed":   "Yes",
    "a1c_tested":    "No",
    "glu_tested":    "No",
    # Seif-only
    "n_med_changes": 0,
    # Dhie-only
    "sejours_anterieurs": 0,
    # Asma-specific (LOS)
    "los_version": "after_stay",
}

EXAMPLES = {
    "routine": {
        "age": 6, "age_num": 55, "gender": "Female", "race": "Caucasian",
        "payer_code": "MC", "medical_specialty": "InternalMedicine",
        "time_in_hospital": 2, "num_lab_procedures": 28, "num_procedures": 0,
        "num_medications": 9, "number_outpatient": 0, "number_emergency": 0,
        "number_inpatient": 0, "number_diagnoses": 4,
        "n_meds_used": 1, "n_meds_changed": 0, "n_med_changes": 0,
        "diag_1": "250", "diag_2": "401", "diag_3": "NA",
        "diag_1_grp": "Diabetes",
        "admission_type_id": 3, "admission_source_id": 1,
        "discharge_disposition_id": 1,
        "A1Cresult": "Norm", "max_glu_serum": "None",
        "insulin": "No", "metformin": "Steady",
        "change": "No", "diabetesMed": "Yes",
        "a1c_tested": "Yes", "glu_tested": "No",
        "sejours_anterieurs": 0,
        "discharge_seif": "Home", "admission_seif": "Elective",
    },
    "complex": {
        "age": 9, "age_num": 82, "gender": "Male", "race": "Caucasian",
        "payer_code": "MC", "medical_specialty": "Cardiology",
        "time_in_hospital": 9, "num_lab_procedures": 68, "num_procedures": 3,
        "num_medications": 24, "number_outpatient": 2, "number_emergency": 2,
        "number_inpatient": 3, "number_diagnoses": 11,
        "n_meds_used": 3, "n_meds_changed": 2, "n_med_changes": 2,
        "diag_1": "428", "diag_2": "428", "diag_3": "250",
        "diag_1_grp": "Circulatory",
        "admission_type_id": 1, "admission_source_id": 7,
        "discharge_disposition_id": 3,
        "A1Cresult": ">8", "max_glu_serum": ">300",
        "insulin": "Up", "metformin": "No",
        "change": "Yes", "diabetesMed": "Yes",
        "a1c_tested": "No", "glu_tested": "No",
        "sejours_anterieurs": 3,
        "discharge_seif": "Skilled nursing facility", "admission_seif": "Emergency",
    }
}


# ──────────────────────────────────────────────────────────────────────────────
#  CLEAN — build per-model row dicts from the single unified form dict
# ──────────────────────────────────────────────────────────────────────────────

def _build_seif_raw(f: dict) -> dict:
    """Map unified form → Seif's expected raw dict."""
    return {
        "age":               f.get("age", DEFAULTS["age"]),
        "time_in_hospital":  f.get("time_in_hospital", DEFAULTS["time_in_hospital"]),
        "num_lab_procedures":f.get("num_lab_procedures", DEFAULTS["num_lab_procedures"]),
        "num_procedures":    f.get("num_procedures", DEFAULTS["num_procedures"]),
        "num_medications":   f.get("num_medications", DEFAULTS["num_medications"]),
        "number_outpatient": f.get("number_outpatient", DEFAULTS["number_outpatient"]),
        "number_emergency":  f.get("number_emergency", DEFAULTS["number_emergency"]),
        "number_inpatient":  f.get("number_inpatient", DEFAULTS["number_inpatient"]),
        "number_diagnoses":  f.get("number_diagnoses", DEFAULTS["number_diagnoses"]),
        "n_meds_used":       f.get("n_meds_used", DEFAULTS["n_meds_used"]),
        "n_med_changes":     f.get("n_med_changes", DEFAULTS["n_med_changes"]),
        "discharge":         f.get("discharge_seif", "Home"),
        "admission":         f.get("admission_seif", "Emergency"),
        "diag_1":            f.get("diag_1_grp", DEFAULTS["diag_1_grp"]),
        "insulin":           f.get("insulin", DEFAULTS["insulin"]),
        "A1Cresult":         f.get("A1Cresult", DEFAULTS["A1Cresult"]),
        "change":            f.get("change", DEFAULTS["change"]),
        "diabetesMed":       f.get("diabetesMed", DEFAULTS["diabetesMed"]),
    }


def _build_ons_raw(f: dict) -> dict:
    """Map unified form → Ons's expected raw dict."""
    return {
        "age":               f.get("age", DEFAULTS["age"]),
        "gender":            f.get("gender", DEFAULTS["gender"]),
        "race":              f.get("race", DEFAULTS["race"]),
        "admission":         str(f.get("admission_type_id", DEFAULTS["admission_type_id"])),
        "source":            str(f.get("admission_source_id", DEFAULTS["admission_source_id"])),
        "discharge":         str(f.get("discharge_disposition_id", DEFAULTS["discharge_disposition_id"])),
        "diag_1":            f.get("diag_1_grp", DEFAULTS["diag_1_grp"]),
        "time_in_hospital":  f.get("time_in_hospital", DEFAULTS["time_in_hospital"]),
        "number_diagnoses":  f.get("number_diagnoses", DEFAULTS["number_diagnoses"]),
        "num_lab_procedures":f.get("num_lab_procedures", DEFAULTS["num_lab_procedures"]),
        "num_procedures":    f.get("num_procedures", DEFAULTS["num_procedures"]),
        "number_inpatient":  f.get("number_inpatient", DEFAULTS["number_inpatient"]),
        "number_emergency":  f.get("number_emergency", DEFAULTS["number_emergency"]),
        "number_outpatient": f.get("number_outpatient", DEFAULTS["number_outpatient"]),
        "num_medications":   f.get("num_medications", DEFAULTS["num_medications"]),
        "insulin":           f.get("insulin", DEFAULTS["insulin"]),
        "a1c_tested":        f.get("a1c_tested", DEFAULTS["a1c_tested"]),
        "glu_tested":        f.get("glu_tested", DEFAULTS["glu_tested"]),
        "change":            f.get("change", DEFAULTS["change"]),
        "diabetesMed":       f.get("diabetesMed", DEFAULTS["diabetesMed"]),
    }


def _build_dhie_raw(f: dict) -> dict:
    """Map unified form → Dhie's expected raw dict."""
    from app.services.dhie_service import MED_COLS
    raw = {
        "age":                  f.get("age_num", DEFAULTS["age_num"]),
        "gender":               f.get("gender", DEFAULTS["gender"]),
        "race":                 f.get("race", DEFAULTS["race"]),
        "admission_type_id":    f.get("admission_type_id", DEFAULTS["admission_type_id"]),
        "admission_source_id":  f.get("admission_source_id", DEFAULTS["admission_source_id"]),
        "time_in_hospital":     f.get("time_in_hospital", DEFAULTS["time_in_hospital"]),
        "payer_code":           f.get("payer_code", DEFAULTS["payer_code"]),
        "medical_specialty":    f.get("medical_specialty", DEFAULTS["medical_specialty"]),
        "num_lab_procedures":   f.get("num_lab_procedures", DEFAULTS["num_lab_procedures"]),
        "num_procedures":       f.get("num_procedures", DEFAULTS["num_procedures"]),
        "num_medications":      f.get("num_medications", DEFAULTS["num_medications"]),
        "number_outpatient":    f.get("number_outpatient", DEFAULTS["number_outpatient"]),
        "number_emergency":     f.get("number_emergency", DEFAULTS["number_emergency"]),
        "number_inpatient":     f.get("number_inpatient", DEFAULTS["number_inpatient"]),
        "diag_1":               f.get("diag_1", DEFAULTS["diag_1"]),
        "diag_2":               f.get("diag_2", DEFAULTS["diag_2"]),
        "diag_3":               f.get("diag_3", DEFAULTS["diag_3"]),
        "number_diagnoses":     f.get("number_diagnoses", DEFAULTS["number_diagnoses"]),
        "max_glu_serum":        f.get("max_glu_serum", DEFAULTS["max_glu_serum"]),
        "A1Cresult":            f.get("A1Cresult", DEFAULTS["A1Cresult"]),
        "change":               f.get("change", DEFAULTS["change"]),
        "diabetesMed":          f.get("diabetesMed", DEFAULTS["diabetesMed"]),
        "sejours_anterieurs":   f.get("sejours_anterieurs", DEFAULTS["sejours_anterieurs"]),
        "insulin":              f.get("insulin", DEFAULTS["insulin"]),
        "metformin":            f.get("metformin", DEFAULTS["metformin"]),
    }
    for m in MED_COLS:
        raw[m] = f.get(m, "No")
    return raw


def _build_roua_raw(f: dict) -> dict:
    """Map unified form → Roua's expected raw dict."""
    return {
        "time_in_hospital":   f.get("time_in_hospital", DEFAULTS["time_in_hospital"]),
        "num_lab_procedures": f.get("num_lab_procedures", DEFAULTS["num_lab_procedures"]),
        "num_procedures":     f.get("num_procedures", DEFAULTS["num_procedures"]),
        "number_diagnoses":   f.get("number_diagnoses", DEFAULTS["number_diagnoses"]),
        "number_inpatient":   f.get("number_inpatient", DEFAULTS["number_inpatient"]),
        "number_emergency":   f.get("number_emergency", DEFAULTS["number_emergency"]),
        "number_outpatient":  f.get("number_outpatient", DEFAULTS["number_outpatient"]),
        "gender":             f.get("gender", DEFAULTS["gender"]),
        "admission_type_id":  f.get("admission_type_id", DEFAULTS["admission_type_id"]),
        "admission_source_id":f.get("admission_source_id", DEFAULTS["admission_source_id"]),
        "medical_specialty":  f.get("medical_specialty", DEFAULTS["medical_specialty"]),
        "diag_1":             f.get("diag_1", DEFAULTS["diag_1"]),
        "diag_2":             f.get("diag_2", DEFAULTS["diag_2"]),
        "diag_3":             f.get("diag_3", DEFAULTS["diag_3"]),
        "change":             f.get("change", DEFAULTS["change"]),
        "diabetesMed":        f.get("diabetesMed", DEFAULTS["diabetesMed"]),
    }


def _build_moez_raw(f: dict) -> dict:
    """Map unified form → Moez's expected raw dict."""
    return {
        "age_num":              f.get("age_num", DEFAULTS["age_num"]),
        "gender":               f.get("gender", DEFAULTS["gender"]),
        "time_in_hospital":     f.get("time_in_hospital", DEFAULTS["time_in_hospital"]),
        "num_lab_procedures":   f.get("num_lab_procedures", DEFAULTS["num_lab_procedures"]),
        "num_procedures":       f.get("num_procedures", DEFAULTS["num_procedures"]),
        "num_medications":      f.get("num_medications", DEFAULTS["num_medications"]),
        "number_outpatient":    f.get("number_outpatient", DEFAULTS["number_outpatient"]),
        "number_emergency":     f.get("number_emergency", DEFAULTS["number_emergency"]),
        "number_inpatient":     f.get("number_inpatient", DEFAULTS["number_inpatient"]),
        "number_diagnoses":     f.get("number_diagnoses", DEFAULTS["number_diagnoses"]),
        "n_meds_used":          f.get("n_meds_used", DEFAULTS["n_meds_used"]),
        "n_meds_changed":       f.get("n_meds_changed", DEFAULTS["n_meds_changed"]),
        "insulin":              f.get("insulin", DEFAULTS["insulin"]),
        "metformin":            f.get("metformin", DEFAULTS["metformin"]),
        "change":               f.get("change", DEFAULTS["change"]),
        "diabetesMed":          f.get("diabetesMed", DEFAULTS["diabetesMed"]),
        "diag_1":               f.get("diag_1", DEFAULTS["diag_1"]),
        "admission_type_id":    f.get("admission_type_id", DEFAULTS["admission_type_id"]),
        "admission_source_id":  f.get("admission_source_id", DEFAULTS["admission_source_id"]),
        "discharge_disposition_id": f.get("discharge_disposition_id", DEFAULTS["discharge_disposition_id"]),
        "A1Cresult":            f.get("A1Cresult", DEFAULTS["A1Cresult"]),
        "max_glu_serum":        f.get("max_glu_serum", DEFAULTS["max_glu_serum"]),
    }


def _build_asma_raw(f: dict) -> dict:
    """Map unified form → Asma's expected raw dict (after_stay version)."""
    return {
        "number_diagnoses":  f.get("number_diagnoses", DEFAULTS["number_diagnoses"]),
        "num_procedures":    f.get("num_procedures", DEFAULTS["num_procedures"]),
        "change":            f.get("change", DEFAULTS["change"]),
        "num_medications":   f.get("num_medications", DEFAULTS["num_medications"]),
        "num_lab_procedures":f.get("num_lab_procedures", DEFAULTS["num_lab_procedures"]),
    }


# ──────────────────────────────────────────────────────────────────────────────
#  PREDICT — run one model safely, return result or error dict
# ──────────────────────────────────────────────────────────────────────────────

def _safe(fn):
    """Wrap a prediction call; return error dict on failure."""
    try:
        return fn(), None
    except Exception:
        return None, traceback.format_exc(limit=4)


def run_all(form: dict, role: str = "doctor") -> dict:
    """
    Run all applicable models for the unified form dict.

    Parameters
    ----------
    form : dict  — cleaned form data (field names as in DEFAULTS)
    role : str   — 'doctor' or 'reception'

    Returns
    -------
    dict with keys:
        los, discharge, medication,          # always present (reception + doctor)
        readmission, hba1c, profile,         # doctor-only (None for reception)
        combined_rules,                      # list of combined advisory strings
        errors                               # dict[model_name, tb_str]
    """
    errors = {}
    results = {
        "los": None, "discharge": None, "medication": None,
        "readmission": None, "hba1c": None, "profile": None,
    }

    # ── 1. LOS (Asma) — reception + doctor ───────────────────────────────────
    asma = _asma()
    asma_raw = _build_asma_raw(form)
    asma_row, asma_errs = asma.clean(asma_raw, version="after_stay")
    if asma_errs:
        errors["los"] = "; ".join(asma_errs)
    else:
        res, err = _safe(lambda: asma.predict_single(asma_row, version="after_stay"))
        if err:
            errors["los"] = err
        else:
            results["los"] = res

    # ── 2. Discharge Destination (Dhie) — reception + doctor ─────────────────
    dhie = _dhie()
    dhie_raw = _build_dhie_raw(form)
    dhie_row, dhie_errs = dhie.clean(dhie_raw)
    if dhie_errs:
        errors["discharge"] = "; ".join(dhie_errs)
    else:
        res, err = _safe(lambda r=dhie_row: dhie.predict_single(r))
        if err:
            errors["discharge"] = err
        else:
            results["discharge"] = res

    # ── 3. Medication Load (Roua) — reception + doctor ───────────────────────
    roua = _roua()
    roua_raw = _build_roua_raw(form)
    roua_row, roua_errs = roua.clean(roua_raw)
    if roua_errs:
        errors["medication"] = "; ".join(roua_errs)
    else:
        res, err = _safe(lambda r=roua_row: roua.predict_single(r))
        if err:
            errors["medication"] = err
        else:
            results["medication"] = res

    if role == "doctor":
        # ── 4. Readmission (Seif) ────────────────────────────────────────────
        seif = _seif()
        seif_raw = _build_seif_raw(form)
        seif_row, seif_errs = seif.clean(seif_raw)
        if seif_errs:
            errors["readmission"] = "; ".join(seif_errs)
        else:
            def _run_seif(r=seif_row):
                probs, X = seif.score([r])
                p = float(probs[0])
                b = seif.band(p)
                factors = seif.explain(X)
                return {"probability": round(p, 4), "percentage": round(p * 100, 1),
                        "band": b, "factors": factors}
            res, err = _safe(_run_seif)
            if err:
                errors["readmission"] = err
            else:
                results["readmission"] = res

        # ── 5. HbA1c (Ons) ───────────────────────────────────────────────────
        ons = _ons()
        ons_raw = _build_ons_raw(form)
        ons_row, ons_errs = ons.clean(ons_raw)
        if ons_errs:
            errors["hba1c"] = "; ".join(ons_errs)
        else:
            def _run_ons(r=ons_row):
                probs, X = ons.score([r])
                p = float(probs[0])
                b = ons.band(p)
                factors = ons.explain(X[0], r)
                return {"probability": round(p, 4), "percentage": round(p * 100, 1),
                        "band": b, "factors": factors,
                        "a1c_tested": r.get("a1c_tested", "No")}
            res, err = _safe(_run_ons)
            if err:
                errors["hba1c"] = err
            else:
                results["hba1c"] = res

        # ── 6. Patient Profile (Moez) ────────────────────────────────────────
        moez = _moez()
        moez_raw = _build_moez_raw(form)
        moez_row, moez_errs = moez.clean(moez_raw)
        if moez_errs:
            errors["profile"] = "; ".join(moez_errs)
        else:
            res, err = _safe(lambda r=moez_row: moez.assign_patient(r))
            if err:
                errors["profile"] = err
            else:
                results["profile"] = res

    results["combined_rules"] = _build_combined_rules(results, role)
    results["errors"] = errors
    return results


# ──────────────────────────────────────────────────────────────────────────────
#  COMBINED CLINICAL RULES
# ──────────────────────────────────────────────────────────────────────────────

def _build_combined_rules(results: dict, role: str) -> list[dict]:
    """
    Generate integrated advisory messages from cross-model signals.
    Returns a list of dicts: {level, icon, title, body}
    """
    rules = []

    los     = results.get("los")
    disc    = results.get("discharge")
    med     = results.get("medication")
    readm   = results.get("readmission")
    hba1c   = results.get("hba1c")
    profile = results.get("profile")

    # Rule 1: High medication + Long stay → escalate pharmacist review
    if med and los:
        if med.get("pred") == "high" and los.get("is_long"):
            rules.append({
                "level": "danger",
                "icon": "bi-exclamation-triangle-fill",
                "title": "High Polypharmacy + Prolonged Stay",
                "body": (
                    "High medication load combined with a predicted stay >3 days significantly "
                    "increases adverse drug event risk. Consider pharmacist-led reconciliation "
                    "at admission and a formal medication review before discharge."
                )
            })

    # Rule 2: Transfer destination + High readmission risk → early care coordination
    if disc and readm and role == "doctor":
        if disc.get("transfer_alert") and readm.get("band") in ("high", "elevated"):
            rules.append({
                "level": "danger",
                "icon": "bi-arrow-left-right",
                "title": "Transfer Alert + Elevated Readmission Risk",
                "body": (
                    "This patient is flagged for potential transfer AND shows elevated readmission risk. "
                    "Consider early social work consultation and confirm receiving-facility care plan "
                    "before initiating transfer to reduce bounce-back admissions."
                )
            })

    # Rule 3: HbA1c not tested + elevated readmission → recommend A1c measurement
    if hba1c and readm and role == "doctor":
        if hba1c.get("a1c_tested") == "No" and readm.get("band") in ("high", "elevated"):
            rules.append({
                "level": "warning",
                "icon": "bi-droplet-half",
                "title": "HbA1c Not Measured — Readmission Risk Elevated",
                "body": (
                    "Inpatient HbA1c measurement is associated with a ~10% reduction in readmission odds "
                    "(OR ≈ 0.904). Consider ordering HbA1c before discharge for this high-risk patient."
                )
            })

    # Rule 4: Profile 1 (complex) + Long stay → multidisciplinary plan
    if profile and los and role == "doctor":
        if profile.get("cluster") == 1 and los.get("is_long"):
            rules.append({
                "level": "warning",
                "icon": "bi-diagram-3",
                "title": "Complex Patient Profile + Long Stay Predicted",
                "body": (
                    "Patient assigned to Profile 1 (heavy polypharmacy, high-acuity) with a predicted "
                    "stay exceeding 3 days. Recommend multidisciplinary team review including internal "
                    "medicine, pharmacy, and discharge planning from day 1."
                )
            })

    # Rule 5: Home discharge + Low medication → reassure / standard pathway
    if disc and med:
        if disc.get("pred") in ("Domicile",) and med.get("pred") == "low":
            rules.append({
                "level": "success",
                "icon": "bi-house-check",
                "title": "Standard Discharge Pathway Suitable",
                "body": (
                    "Home discharge is predicted with low medication complexity. "
                    "Standard discharge checklist and routine outpatient follow-up appear appropriate."
                )
            })

    # Rule 6: Transfer alert alone
    if disc and not any(r["title"].startswith("Transfer Alert") for r in rules):
        if disc.get("transfer_alert"):
            rules.append({
                "level": "warning",
                "icon": "bi-send",
                "title": "Transfer Likely — Coordinate Early",
                "body": (
                    f"Transfer probability: {round(disc.get('transfer_prob', 0) * 100, 1)}%. "
                    "Consider initiating transfer logistics and notifying the receiving facility early "
                    "to minimise care transition delays."
                )
            })

    # Rule 7: Readmission high risk alone (if no combined rule fired)
    if readm and role == "doctor":
        if readm.get("band") == "high" and not any(
            "Readmission" in r["title"] for r in rules
        ):
            rules.append({
                "level": "danger",
                "icon": "bi-arrow-repeat",
                "title": "High Readmission Risk",
                "body": (
                    f"Readmission probability: {readm.get('percentage', 0)}%. "
                    "Consider scheduling a post-discharge phone call within 48 hours, "
                    "ensuring PCP follow-up within 7 days, and reviewing discharge instructions carefully."
                )
            })

    # If no rules fired, add a neutral advisory
    if not rules:
        rules.append({
            "level": "info",
            "icon": "bi-info-circle",
            "title": "No High-Priority Alerts",
            "body": "No critical cross-model alerts triggered. Review individual model results below for detailed guidance."
        })

    return rules


# ──────────────────────────────────────────────────────────────────────────────
#  FORM CONTEXT — provide metadata for rendering the unified form
# ──────────────────────────────────────────────────────────────────────────────

def get_form_context(form_data: dict | None = None, errors: list | None = None) -> dict:
    d = form_data or DEFAULTS
    errors = errors or []
    seif_s = _seif()
    return {
        "d": d,
        "errors": errors,
        "examples":        EXAMPLES,
        # Shared option lists
        "age_groups":      AGE_GROUPS,
        "genders":         GENDERS,
        "yesno":           YESNO,
        "races":           RACES,
        "adm_types":       ADM_TYPES,
        "adm_srcs":        ADM_SRCS,
        "discharge_opts":  DISCHARGE_DHIE_OPTS,
        "payers":          PAYERS,
        "specialties":     SPECIALTIES,
        "insulin_opts":    INSULIN_OPTS,
        "metformin_opts":  METFORMIN_OPTS,
        "a1c_opts":        A1C_OPTS_SEIF,
        "glu_opts":        GLU_OPTS,
        "diag_groups":     DIAG_GROUPS_SEIF,
        # Seif-specific option lists (discharge text / admission text versions)
        "seif_discharge":  seif_s.DISCHARGE,
        "seif_admission":  seif_s.ADMISSION,
        "seif_ages":       seif_s.AGES,
    }
