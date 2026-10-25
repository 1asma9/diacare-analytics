# Tool Specification: HbA1c Testing & Treatment Decision Support

**Owner:** Ons  
**Model Family:** Multivariable Logistic Regression (Model 3 with 55 standardized/one-hot features, stored as coefficients in `model.json`)  
**Bundle Location:** `models/ons/model.json`, `models/ons/metrics.json`  
**User Role:** Doctors only (Clinical tool)

---

## 1. Clinical Question & Target Audience
- **Question Answered:** How does measuring HbA1c and modifying diabetes medications during the hospital stay associate with early readmission risk (< 30 days) for this patient?
- **Primary Users:** Hospital physicians, endocrinologists, and diabetologists.
- **Business Objective:** Support clinical decision-making regarding glycemic monitoring and treatment adaptation. Testing HbA1c provides a longitudinal window on glycemic control; this tool quantifies its statistical association alongside patient-specific risk.

---

## 2. Model Performance & Clinical Odds Ratios
*(Extracted directly from `models/ons/metrics.json` and `model.json`)*
- **AUC-ROC:** 0.6423 (0.642)
- **AUC-PR:** 0.1663 (baseline prevalence is ~8.97%)
- **Accuracy:** 66.5% (balanced accuracy 60.8%)
- **Base Rate:** 8.97% (0.0897)
- **Total Patients in Cohort:** 69,970 (train: 55,976, test: 13,994)
- **Adjusted Odds Ratios (Associations, NOT Direct Causes):**
  1. **HbA1c Measured (`a1c_tested`):** **OR = 0.904** (95% CI: [0.833, 0.982], $p = 0.017$).  
     *Interpretation:* Having an HbA1c test recorded during the stay is associated with approximately **10% lower odds of early readmission** after adjusting for all patient severity covariates.
  2. **Medication Change (`change`):** **OR = 0.970** (95% CI: [0.897, 1.049], $p = 0.442$).  
     *Interpretation:* In-hospital treatment change alone has no statistically significant association with early readmission ($p = 0.44$).
  3. **Prescribed Diabetes Medication (`diabetesMed`):** **OR = 1.230** ($p = 2.45 \times 10^{-6}$).
  4. **Serum Glucose Tested (`glu_tested`):** **OR = 1.233** ($p = 0.015$).

---

## 3. Thresholds & Risk Bands
- **Decision Threshold ($\tau$):** `8.85%` (0.08846)
  - Scores $\ge 8.85\%$: Elevated early readmission risk.
  - Scores $< 8.85\%$: Lower readmission risk.
- **High-Risk Threshold:** `11.37%` (0.11365, top quintile)
- **Risk Bands:**
  1. **Low Risk (< 8.85%):** 65.9% of cohort, 6.30% observed readmission rate.
  2. **Elevated Risk (8.85% - 11.37%):** 14.3% of cohort, 11.19% observed readmission rate.
  3. **High Risk ($\ge$ 11.37%):** 19.8% of cohort, 16.23% observed readmission rate.

---

## 4. Input Fields, Defaults, and Valid Ranges
From `inputs/ons/app.py`:

| Field Name | Display Label | Options / Range | Default |
|---|---|---|---|
| `age` | Age group | 1–10 (1: [0-10) to 10: [90-100)) | `7` (60–70) |
| `gender` | Gender | Female, Male | `Female` |
| `race` | Race / ethnicity | Caucasian, AfricanAmerican, Hispanic, Asian, Other, Inconnu | `Caucasian` |
| `admission` | Admission type ID | 1 (Emergency), 2 (Urgent), 3 (Elective), 0/other | `1` |
| `source` | Admission source ID | 7 (Emergency room), 1 (Referral), etc. | `7` |
| `discharge` | Discharge disposition ID | 1 (Home), 3 (SNF), 6 (Home health), etc. | `1` |
| `diag_1` | Primary diagnosis | Circulatory, Diabetes, Respiratory, Digestive, etc. | `Circulatory` |
| `time_in_hospital` | Length of stay (days) | Numeric: 1 to 14 | `3` |
| `number_diagnoses` | Number of diagnoses | Numeric: 1 to 16 | `8` |
| `num_lab_procedures` | Number of lab tests | Numeric: 1 to 132 | `44` |
| `num_procedures` | Number of procedures | Numeric: 0 to 6 | `1` |
| `number_inpatient` | Prior inpatient stays | Numeric: 0 to 12 | `0` |
| `number_emergency` | Prior emergency visits | Numeric: 0 to 42 | `0` |
| `number_outpatient` | Prior outpatient visits | Numeric: 0 to 42 | `0` |
| `num_medications` | Number of medications | Numeric: 1 to 81 | `14` |
| `insulin` | Insulin status | No, Steady, Up, Down | `No` |
| `a1c_tested` | HbA1c measured during stay | Yes, No | `No` |
| `glu_tested` | Glucose serum test ordered | Yes, No | `No` |
| `change` | Medication change | Yes, No | `No` |
| `diabetesMed` | Diabetes medication prescribed | Yes, No | `Yes` |

---

## 5. Result Display & Explainability
- **Predicted Probability Card:** Patient's probability of early readmission with level badge (`Low`, `Elevated`, `High`).
- **Clinical Association Panel:** Displays the odds ratios for HbA1c and treatment modification alongside patient values:
  - If HbA1c is **not measured**, highlights the opportunity for measurement (associated with ~10% lower readmission odds).
  - Clarifies that treatment changes should be driven by clinical targets, as broad statistical data show neutral overall odds (OR 0.97).
- **Contributing Features Bar Chart:**
  - $\beta_j \times (x_j - \bar{x}_j)$ breakdown comparing patient values to the cohort mean.

---

## 6. Recommendations & Clinical Banner
- **Careful Interpretation Banner:**  
  *"Read this with care: These figures represent statistical multivariable associations from historical observational data (AUC 0.642). They do not represent causal certainty or prescriptive treatment mandates."*
- **Recommendation Rules:**
  - **If HbA1c was not tested:** *"No HbA1c test recorded during this stay. Consider ordering an HbA1c test to assess baseline glycemic control (associated with ~10% lower readmission odds in hospital data)."*
  - **If patient is High Risk ($\ge 11.4\%$):** *"High early readmission risk detected. Consider a focused glycemic review prior to discharge and arranging outpatient follow-up within 7 days."*
  - **Treatment guidance:** Always framed neutrally: *"Review medication plan against current blood glucose trends. In observational data, medication changes show no significant independent reduction in readmission without coordinated outpatient follow-up."*

---

## 7. Example Patients
- **Lower-Risk Patient:**  
  - Age 40–50 (`age=5`), Stay 2 days, 30 lab tests, 9 medications, 5 diagnoses, no prior hospitalizations, no diabetes medication prescribed.  
  - **Output:** Predicted risk `~4.5%` (Low Risk, well below 8.8% threshold).
- **Higher-Risk Patient:**  
  - Age 70–80 (`age=8`), Stay 7 days, 60 lab tests, 26 medications, 2 outpatient, 2 ER, 3 inpatient stays past year, 9 diagnoses, discharge to SNF (`discharge=3`), Insulin Up, Change Yes, HbA1c measured.  
  - **Output:** Predicted risk `~35.7%` (High Risk).

---

## 8. Batch Mode (CSV Upload)
- **Max Rows:** 5,000 patients.
- **Required Columns:** `id`, `age`, `gender`, `race`, `admission_type_id`, `admission_source_id`, `discharge_disposition_id`, `diag_1`, `time_in_hospital`, `number_diagnoses`, `num_lab_procedures`, `num_procedures`, `number_inpatient`, `number_emergency`, `number_outpatient`, `num_medications`, `insulin`, `a1c_tested`, `glu_tested`, `change`, `diabetesMed`.
- **Output Table:** ID, Readmission Risk (%), Risk Band, HbA1c Tested (Yes/No), Order Test Recommended (Yes/No), with CSV export.
