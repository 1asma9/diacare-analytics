# Phase 0 Report: Preparation & Model Integration Environment

**Date:** 2026-10-25  
**Project:** DiaCare Analytics (ESPRIT, 4CCE19)  
**Status:** Completed successfully — awaiting user "go" for Phase 1.

---

## 1. Environment & Requirements
- **Python Version:** Python 3.12.15 (configured in project virtual environment `.venv/`).
- **Core Pinned Libraries:**
  - `scikit-learn==1.9.1`
  - `xgboost==3.4.1`
  - `flask==3.1.3`
  - `pandas==3.1.0`
  - `numpy==2.5.4`
  - `joblib==1.6.0`
  - `scipy==1.18.1`
- **Output:** Frozen exact versions recorded in [requirements.txt](file:///c:/Users/onsre/Downloads/Clinic-v1.0.0/requirements.txt).

---

## 2. Model Integrity & SHA-256 Checksums
All model files were copied unchanged from `inputs/` into `models/<owner>/`. Initial SHA-256 hashes were recorded in [docs/model_checksums.txt](file:///c:/Users/onsre/Downloads/Clinic-v1.0.0/docs/model_checksums.txt) and re-verified at the end of Phase 0:

| Model / Owner | File Relative Path | SHA-256 Hash | Integrity Status |
|---|---|---|---|
| **Asma** | `models/asma/model_asma_longstay.joblib` | `146d9b65839eb087e18cf14f2012fd95a1a4b3447c06afac8f72cc3bdc4daf9f` | **Verified** |
| **Asma** | `models/asma/asma_longstay_predict.py` | `9d687d40b4071058feb5d78d3897c08af800c915ccfb9b371a6a86917963ce35` | **Verified** |
| **Dhie** | `models/dhie/model.joblib` | `b1c498e13abbc8d3fdf33f81ebfbc3ae3d6bf815d08b0a0327d54870796536a8` | **Verified** |
| **Dhie** | `models/dhie/preprocessing.py` | `189875b87d83f9a7e25c25a6fba0cd363eeb83bd8722de6d67330ebb2e8759a8` | **Verified** |
| **Moez** | `models/moez/cluster_bundle.joblib` | `4972dcce2f34f66e61766add4a020cee6ea6a7c83e794f5fea46422e2bef4197` | **Verified** |
| **Moez** | `models/moez/cluster_profiles.json` | `fa8fa6e4d3b66315d93f50a86705a423dae965b59c5a35c79da2deb47201cf78` | **Verified** |
| **Ons** | `models/ons/model.json` | `24b8764b344e6c41d1fb77f826549a6f2b81e2d2509b7da25f9190ac31d337c9` | **Verified** |
| **Ons** | `models/ons/metrics.json` | `ad6730df2b523fd3f8f95539fca3cd90824d22b0b436f0a18e0c289a48b7a0b8` | **Verified** |
| **Roua** | `models/roua/model_bundle.joblib` | `2f365e768f223217608ee54ca1736a5b965cbdaea866e70ec967629e8d382958` | **Verified** |
| **Seif** | `models/seif/bundle.joblib` | `cb6b0c658ea6238cdcdbb231cbfcc597e2ea80ea25e5b6a1d878eae5c2092d45` | **Verified** |
| **Seif** | `models/seif/metrics.json` | `28175fca528771d71c7ad792360cc1468cafa130cc89aa881e06a9e427f57a19` | **Verified** |

---

## 3. Engineering Challenge & Solution: Cross-Platform XGBoost Deserialization
- **Issue Discovered:** When unpickling Asma's joblib model on Windows, XGBoost 3.4.1 threw `xgboost._c_api.XGBoostError: input stream corrupted`.
- **Root Cause (XGBoost Issue #12459):** XGBoost serializes its internal `std::mt19937` random engine state (`rng_state`) as text using C++ stream operators. On Linux (`libstdc++`), it formats numbers in decimal text; on Windows (`MSVC STL`), the stream operator expects hexadecimal text. Loading a Linux-trained bundle on Windows therefore causes MSVC stream extraction to corrupt the stream.
- **Rule-Compliant Solution:** In adherence to the absolute rule ("Never modify, retrain, re-save, rename or re-tune any model file... Adapt your wrapper"), we created [app/loaders/xgb_compat.py](file:///c:/Users/onsre/Downloads/Clinic-v1.0.0/app/loaders/xgb_compat.py). The loader patches the `rng_state` text chunk in-memory during unpickling on Windows without modifying a single byte of the original `model_asma_longstay.joblib` file on disk.

---

## 4. Moez Exception & Scratch Comparison
In accordance with rule 4, Moez's original bundle was left completely untouched in `models/moez/cluster_bundle.joblib`. Loading it in scikit-learn 1.9.1 produces an expected `InconsistentVersionWarning` (trained with 1.6.1).

A comparison was executed in `scratch/moez_comparison_result.json` re-running his feature pipeline and fitting a new bundle in scikit-learn 1.9.1 on `df_clean.csv` (69,970 stays):

- **Original Bundle (1.6.1 loaded in 1.9.1):**
  - Cluster 0: **43,367 stays (61.98%)**
  - Cluster 1: **26,603 stays (38.02%)**
- **Re-fit Bundle (1.9.1 on identical data):**
  - Cluster 0: **43,275 stays (61.85%)**
  - Cluster 1: **26,695 stays (38.15%)**
- **Cluster Agreement:** **99.85%** identical cluster assignment across the 69,970 patient stays.

**Result for User Approval:** The original bundle behaves stably (99.85% consistency). We await your direction on whether to keep loading the original bundle with the warning suppressed, or whether to save a separate re-saved bundle.

---

## 5. Automated Load Test Verification
The automated verification script [tests/test_load_models.py](file:///c:/Users/onsre/Downloads/Clinic-v1.0.0/tests/test_load_models.py) verified all six models:

1. **Seif (Stacking Early Readmission):**
   - Default patient scores **7.02%** (expected ~7.0%).
   - Higher-risk patient scores **28.03%** (expected ~28.0%).
   - Threshold $\tau = 8.20\%$, High-risk threshold = $12.08\%$. Matches Seif's original scoring code.
2. **Ons (HbA1c Logistic Regression):**
   - Default patient: **7.65%**; Higher-risk patient: **35.71%**.
   - Threshold $\tau = 8.85\%$, AUC = 0.6423.
   - HbA1c measured odds ratio = **0.9044** (95% CI: [0.833, 0.982], $p = 0.017$, ~10% lower odds). Matches Ons's original scoring code.
3. **Dhie (Discharge Destination):**
   - Home example: predicts **`Domicile`** (Home, 60.1% probability), Transfer Alert: **False** (16.5% transfer prob).
   - Transfer example: predicts **`Transfert`** (Transfer, 42.3% probability), Transfer Alert: **True** (67.3% transfer prob).
   - Preprocessing and destination exclusion rules fully verified against Dhie's original code.
4. **Roua (Medication Load Version B):**
   - Successfully loaded and scored patients across the 3 classes: Low ($\le 11$ drugs), Medium (12–17), High ($\ge 18$).
   - Test accuracy: 60.9% (67% precision for high load).
5. **Moez (Patient Profiles Clustering):**
   - Successfully loads cluster profiles: Profile 0 (62.0% size, 7.7% early readmission) and Profile 1 (38.0% size, 11.1% early readmission).
6. **Asma (Length of Stay > 3 Days):**
   - Admission version (3 inputs: `number_diagnoses`, `num_procedures`, `change`): $P(\text{Long}) = 0.5032$ (Long stay).
   - After-stay version (5 inputs): $P(\text{Long}) = 0.5674$ (Long stay).
- **Test Result:** **11 PASS, 0 FAIL**.

---

## 6. One-Page Tool Specifications
Drafted in `docs/specs/` based strictly on teammate apps and notebooks:
- [docs/specs/readmission.md](file:///c:/Users/onsre/Downloads/Clinic-v1.0.0/docs/specs/readmission.md) (Seif)
- [docs/specs/hba1c_decision.md](file:///c:/Users/onsre/Downloads/Clinic-v1.0.0/docs/specs/hba1c_decision.md) (Ons)
- [docs/specs/discharge_destination.md](file:///c:/Users/onsre/Downloads/Clinic-v1.0.0/docs/specs/discharge_destination.md) (Dhie)
- [docs/specs/medication_load.md](file:///c:/Users/onsre/Downloads/Clinic-v1.0.0/docs/specs/medication_load.md) (Roua)
- [docs/specs/patient_profiles.md](file:///c:/Users/onsre/Downloads/Clinic-v1.0.0/docs/specs/patient_profiles.md) (Moez)
- [docs/specs/length_of_stay.md](file:///c:/Users/onsre/Downloads/Clinic-v1.0.0/docs/specs/length_of_stay.md) (Asma)

---

## 7. Open Issues & Questions for User
1. **Moez Re-saved Bundle:** The original bundle loads with only a version warning and produces 99.85% agreement with a re-fitted bundle. Do you approve saving a clean re-saved bundle for scikit-learn 1.9.1 under `models/moez/cluster_bundle_resaved.joblib`, or should we keep loading the original bundle with warning suppression?
2. **Readiness for Phase 1:** All preparation criteria are met. We are ready to proceed with Phase 1 (Foundation: Clinic Bootstrap template shell, role-based login, and shared components) upon your "go".
