# Tool Specification: Discharge Destination & Transfer Alert

**Owner:** Dhie  
**Model Family:** Weighted Ensemble (HistGradientBoosting + Random Forest) + Dedicated Binary Transfer Classifier (`hgb_binary`)  
**Bundle Location:** `models/dhie/model.joblib`, `models/dhie/preprocessing.py`  
**User Role:** Doctors AND Reception / Admission Staff

---

## 1. Clinical Question & Target Audience
- **Question Answered:** Where is this diabetic patient most likely to be discharged (Home, Home with care, Transfer, Other), and should hospital bed management prepare for an external facility transfer?
- **Primary Users:** Admission/reception coordinators, bed management, social workers, discharge planning nurses, and ward doctors.
- **Business Objective:** Prevent delayed discharges and transfer bottlenecks. Arranging home healthcare or external skilled nursing facility (SNF) beds takes 24–72 hours; anticipating these destinations early in the stay avoids costly bed-blocking.

---

## 2. Absolute Integrity Rule
> [!IMPORTANT]
> The discharge destination (`discharge_disposition_id`) is the prediction target. The model **must never** receive the discharge destination as an input under any circumstances.

---

## 3. Model Performance & Metrics
*(Extracted directly from `inputs/dhie/app.py` and `inputs/dhie/README.md`)*
- **Macro F1 Score:** 0.520 (95% CI: [0.510, 0.528])
- **Transfer Detection Recall:** 70.5% (0.705)
- **Transfer Detection Precision:** 46.2% (0.462)
- **Binary Transfer AUC:** 0.834
- **Comparison to Baseline:** Pure majority baseline (always Home) achieves Macro F1 of only 0.186; the weighted ensemble achieves 0.520.

---

## 4. Destination Classes & Transfer Alert Threshold
- **Four Destination Classes:**
  1. **Home (`Domicile`):** Routine discharge home without formal medical services.
  2. **Home with care (`Domicile avec soins`):** Discharge home with home-health nurse visits or specialized care.
  3. **Transfer (`Transfert`):** Transfer to skilled nursing facility (SNF), rehab clinic, or another acute inpatient hospital.
  4. **Other / Unknown (`Autre`):** Left AMA, specialized hospice, or unmapped discharge code.
- **Binary Transfer Alert Threshold ($\tau$):** Defined in model artifacts (`art['threshold']` $\approx 0.35$).
  - If predicted $P(\text{Transfer}) \ge \tau$: A prominent **Transfer Alert** badge is triggered to alert bed coordination.

---

## 5. Input Fields, Defaults, and Valid Ranges
From `inputs/dhie/app.py` and `preprocessing.py`:

| Field Name | Display Label | Type / Options | Default | Validation Range |
|---|---|---|---|---|
| `age` | Age (years) | Numeric | `70` | 0 to 100 |
| `gender` | Gender | Female, Male | `Female` | Female / Male |
| `race` | Race / ethnicity | Caucasian, AfricanAmerican, Hispanic, Asian, Other, Inconnu | `Caucasian` | Allowed list |
| `admission_type_id` | Admission type | 1 (Emergency), 2 (Urgent), 3 (Elective), 0 (Other) | `1` | 0 to 8 |
| `admission_source_id` | Admission source | 7 (Emergency room), 1 (Referral), 4 (Transfer), etc. | `7` | Allowed list |
| `time_in_hospital` | Length of stay (days) | Numeric | `4` | 1 to 14 |
| `payer_code` | Insurance / Payer | MC, HM, SP, BC, MD, CP, UN, CM, Inconnu | `MC` | Allowed list |
| `medical_specialty` | Medical specialty | InternalMedicine, Cardiology, Emergency/Trauma, etc. | `Inconnu` | Allowed list |
| `num_lab_procedures` | Lab procedures | Numeric | `40` | 1 to 130 |
| `num_procedures` | Procedures | Numeric | `0` | 0 to 6 |
| `num_medications` | Medications count | Numeric | `15` | 1 to 80 |
| `number_outpatient` | Outpatient visits (past year) | Numeric | `0` | 0 to 40 |
| `number_emergency` | Emergency visits (past year) | Numeric | `0` | 0 to 40 |
| `number_inpatient` | Inpatient stays (past year) | Numeric | `0` | 0 to 20 |
| `sejours_anterieurs` | Prior stays (this patient) | Numeric | `0` | 0 to 50 |
| `diag_1` | Primary diagnosis (ICD-9) | Text code (e.g. `428`, `250`, or NA) | `428` | ICD-9 string |
| `diag_2` | Secondary diagnosis 2 | Text code | `250` | ICD-9 string |
| `diag_3` | Secondary diagnosis 3 | Text code | `NA` | ICD-9 string |
| `number_diagnoses` | Total diagnoses | Numeric | `7` | 1 to 16 |
| `max_glu_serum` | Max glucose serum | None, Norm, >200, >300 | `None` | Allowed list |
| `A1Cresult` | HbA1c result | None, Norm, >7, >8 | `None` | Allowed list |
| `change` | Medication changed | Yes, No | `No` | Yes / No |
| `diabetesMed` | Diabetes medication | Yes, No | `Yes` | Yes / No |
| `insulin` | Insulin status | No, Steady, Up, Down | `No` | Allowed list |
| `metformin` | Metformin status | No, Steady, Up, Down | `No` | Allowed list |

---

## 6. Result Display & Explainability
- **Primary Predicted Class Badge:** Large badge highlighting the predicted destination (`Home`, `Home with care`, `Transfer`, `Other`).
- **Destination Probabilities Breakdown:** Progress bars showing probabilities for all four destinations summing to 100%.
- **Transfer Alert Indicator:** If $P(\text{Transfer}) \ge \tau$, displays an amber alert banner:  
  *"Transfer Alert: Transfer probability is high. Early coordination with post-acute facilities recommended."*

---

## 7. Recommendations & Clinical Banner
- **Decision Support Banner:**  
  *"Read this with care: Destination probabilities are planning aids derived from clinical and admission variables. They support timely bed planning and care coordination, not mandatory administrative decisions."*
- **Recommendation Rules:**
  - **Predicted Transfer or Transfer Alert Active:** *"Predicted transfer or high transfer probability: consider arranging transfer paperwork and contacting receiving facilities (SNF / rehab) early in the stay."*
  - **Predicted Home with Care:** *"Predicted discharge home with care: consider early liaison with community nursing or home healthcare coordinators."*
  - **Predicted Home:** *"Routine discharge home expected. Standard discharge instructions and medication summary recommended."*

---

## 8. Example Patients
- **Home Example Patient:**  
  - Inputs: Age 65, Elective admission (`admission_type_id=3`), Physician referral (`admission_source_id=1`), 2 days stay, 25 lab tests, 8 medications, 5 diagnoses, 0 prior stays, diabetes diagnosis (`diag_1=250`), Insulin Steady.  
  - **Output:** Predicted: `Home` (`Domicile`, 60.1% probability); Transfer prob: 16.5%; Transfer Alert: **False**.
- **Transfer Example Patient:**  
  - Inputs: Age 82, Emergency admission (`admission_type_id=1`), Stay 9 days, 68 lab tests, 3 procedures, 22 medications, 2 prior inpatients, 11 diagnoses, heart failure (`diag_1=428`, `diag_2=428`), HbA1c >8, Insulin Up, 3 prior stays, Cardiology specialty.  
  - **Output:** Predicted: `Transfer` (`Transfert`, 42.3% probability); Transfer prob: **67.3%**; Transfer Alert: **TRUE**.

---

## 9. Batch Mode (CSV Upload)
- **Max Rows:** 5,000 patients.
- **Required Columns:** Form fields excluding target `discharge_disposition_id`.
- **Output Table:** ID, Predicted Destination, P(Home), P(Home with care), P(Transfer), P(Other), Transfer Alert (Yes/No), with CSV download.
