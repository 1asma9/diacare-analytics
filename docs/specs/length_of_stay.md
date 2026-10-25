# Tool Specification: Length of Stay (Long Stay > 3 Days)

**Owner:** Asma  
**Model Family:** Gradient Boosting (XGBoost classifier with 99th percentile capping, log-transformations, and standard scaling; AdaBoost also benchmarked)  
**Bundle Location:** `models/asma/model_asma_longstay.joblib`, `models/asma/asma_longstay_predict.py`  
**User Role:** Doctors AND Reception / Admission Staff (Admission and Planning tool)

---

## 1. Clinical Question & Target Audience
- **Question Answered:** Is this newly admitted diabetic patient likely to experience a prolonged hospital stay (> 3 days), allowing bed management to anticipate capacity?
- **Primary Users:** Admission office, bed managers, ward charge nurses, and attending physicians.
- **Business Objective:** Proactive bed capacity planning. Over 49.8% of diabetic stays exceed 3 days. Identifying long-stay patients at the moment of admission (with just 3 simple variables) enables early discharge planning, social service referrals, and proactive bed turnover management.

---

## 2. Models & Versions
*(Extracted directly from `models/asma/model_asma_longstay.joblib`)*
- **Binary Target:** `0: Short stay` ($\le 3$ days) vs `1: Long stay` ($> 3$ days).
- **Decision Threshold ($\tau$):** `0.50` (50.0%).
- **Two Tool Modes:**
  1. **Admission Version (Primary Mode on Page):**
     - Evaluated at admission using only **3 readily available inputs**: `number_diagnoses`, `num_procedures`, `change`.
     - Intended for admission staff and bed management upon intake.
  2. **After-Stay Version (Optional Toggle):**
     - Evaluated after initial ward tests using **5 inputs**: `num_medications`, `num_lab_procedures`, `number_diagnoses`, `num_procedures`, `change`.
     - Incorporates diagnostic intensity and medication burden for higher accuracy during the stay.

---

## 3. Model Performance & Metrics
*(Extracted directly from `models/asma/model_asma_longstay.joblib`)*

| Metric | Admission Version (3 inputs, Primary) | After-Stay Version (5 inputs, Toggle) |
|---|---|---|
| **Accuracy (Test)** | **61.3%** (0.613) | **69.8%** (0.698) |
| **Precision (Test)** | **62.3%** (0.623) | **71.4%** (0.714) |
| **Recall / Rappel (Test)** | **56.4%** (0.564) | **65.7%** (0.657) |
| **F1 Score (Test)** | **59.2%** (0.592) | **68.4%** (0.684) |
| **AUC-ROC (Test)** | **0.655** (0.655) | **0.772** (0.772) |

---

## 4. Input Fields, Defaults, and Valid Ranges
From `models/asma/asma_longstay_predict.py`:

| Field Name | Display Label | Version Used | Type / Range | Default |
|---|---|---|---|---|
| `number_diagnoses` | Number of diagnoses | Both versions | Numeric: 1 to 16 | `7` |
| `num_procedures` | Number of surgical / clinical procedures | Both versions | Numeric: 0 to 6 | `1` |
| `change` | Medication change expected / performed | Both versions | Select: Yes (`Ch`), No (`No`) | `No` |
| `num_medications` | Number of medications | After-stay toggle only | Numeric: 1 to 81 | `15` |
| `num_lab_procedures` | Number of lab tests | After-stay toggle only | Numeric: 1 to 132 | `43` |

### Preprocessing Applied by Wrapper
1. `number_diagnoses`, `num_procedures`, `num_lab_procedures` clipped at 99th percentile (`cap_p99`), then standardized using bundle means and standard deviations.
2. `num_medications` clipped at 99th percentile (44), log-transformed $\log(1 + x)$, then standardized.
3. `change` mapped to binary: 1 for `Ch`/`Yes`, 0 for `No`.

---

## 5. Result Display & Visualisation
- **Primary Stay Duration Badge:**
  - `Short Stay (≤ 3 days)` (green badge, when $P(\text{Long}) < 0.50$).
  - `Long Stay (> 3 days)` (amber badge, when $P(\text{Long}) \ge 0.50$).
- **Probability Gauge:** Shows estimated probability of stay exceeding 3 days (e.g. 52.4%).
- **Version Switcher:** Seamless UI toggle between "Admission Assessment (3 inputs)" and "Comprehensive Assessment (5 inputs)".

---

## 6. Recommendations & Clinical Banner
- **Decision Support Banner:**  
  *"Read this with care: Length-of-stay predictions are derived from admission and clinical variables to support hospital bed planning. Clinical discharge readiness is always determined by attending physicians."*
- **Recommendation Rules:**
  - **Predicted Long Stay (> 3 days):** *"Long expected stay: consider starting discharge planning early, assessing home support needs, and alerting ward bed management."*
  - **Predicted Short Stay ($\le 3$ days):** *"Routine short stay anticipated. Standard clinical pathway and routine bed scheduling recommended."*

---

## 7. Example Patients
- **Short-Stay Example Patient (Admission):**  
  - Inputs: 3 diagnoses (`number_diagnoses=3`), 0 procedures (`num_procedures=0`), no medication change (`change=No`).  
  - **Output:** Predicted: `Short stay` ($P(\text{Long}) \approx 38\%$).
- **Long-Stay Example Patient (Admission):**  
  - Inputs: 9 diagnoses (`number_diagnoses=9`), 2 procedures (`num_procedures=2`), medication changed (`change=Yes`).  
  - **Output:** Predicted: `Long stay` ($P(\text{Long}) \approx 62\%$).

---

## 8. Batch Mode (CSV Upload)
- **Max Rows:** 5,000 patients.
- **Required Columns:** `id`, `number_diagnoses`, `num_procedures`, `change` (and optionally `num_medications`, `num_lab_procedures` for after-stay mode).
- **Output Table:** ID, Predicted Duration Class, Probability of Long Stay (%), Early Discharge Planning Flag, with CSV export.
