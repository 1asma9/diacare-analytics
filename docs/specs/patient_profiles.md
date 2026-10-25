# Tool Specification: Patient Profiles (Unsupervised Clustering, k = 2)

**Owner:** Moez  
**Model Family:** Unsupervised K-Means ($k=2$) on 18-component PCA space, with StandardScaler and 2D PCA projection  
**Bundle Location:** `models/moez/cluster_bundle.joblib`, `models/moez/cluster_profiles.json`  
**User Role:** Doctors only (Clinical tool)

---

## 1. Clinical Question & Target Audience
- **Question Answered:** Which of the two overarching clinical phenotypes does this patient fit into, and what does this profile indicate about their expected complexity and readmission risk?
- **Primary Users:** Attending physicians, hospital internists, and care coordinators.
- **Business Objective:** Holistic patient segmentation. Beyond isolated risk scores, clustering groups patients by underlying clinical complexity (medication intensity, length of stay, discharge needs, and glycemic destabilization) to guide integrated care pathways.

---

## 2. The Two Clinical Profiles
*(Extracted directly from `models/moez/cluster_profiles.json` and `inputs/moez/moez.ipynb`)*

### Profile 0: Routine / Lower-Complexity Diabetic Stay
- **Cohort Size:** **62.0%** of patients (43,373 stays).
- **All Readmissions Rate:** 37.3%.
- **Early Readmission (< 30 days):** **7.7%**.
- **Distinguishing Clinical Characteristics:**
  - `num_medications`: Mean 12.0 (Global mean 15.67, $z = -0.44$).
  - `time_in_hospital`: Mean 3.05 days (Global mean 4.27 days, $z = -0.42$).
  - `change_bin`: 25% had a medication change (Global mean 45%, $z = -0.39$).
  - `n_meds_changed`: Mean 0.08 dose changes (Global mean 0.26, $z = -0.38$).
  - `n_meds_used`: Mean 0.88 diabetes drugs (Global mean 1.19, $z = -0.33$).
- **Summary:** Shorter, stable hospital stays with low medication turnover and routine discharge directly home.

### Profile 1: Complex / High-Intensity Diabetic Stay
- **Cohort Size:** **38.0%** of patients (26,606 stays).
- **All Readmissions Rate:** 46.4%.
- **Early Readmission (< 30 days):** **11.1%** (~44% relative increase in early readmission compared to Profile 0).
- **Distinguishing Clinical Characteristics:**
  - `num_medications`: Mean 21.64 (Global mean 15.67, $z = +0.72$).
  - `time_in_hospital`: Mean 6.26 days (Global mean 4.27 days, $z = +0.68$).
  - `change_bin`: 77% had a medication change (Global mean 45%, $z = +0.64$).
  - `n_meds_changed`: Mean 0.55 dose changes (Global mean 0.26, $z = +0.61$).
  - `n_meds_used`: Mean 1.70 diabetes drugs (Global mean 1.19, $z = +0.54$).
- **Overrepresented Modalities (Significant Lift):**
  - **Discharge to Facility:** 27.3% in cluster vs 19.5% global (**Lift = 1.40x**).
  - **Severe Hyperglycemia (HbA1c > 8):** 12.7% in cluster vs 8.9% global (**Lift = 1.42x**).
- **Summary:** Longer, therapeutically intensive admissions with substantial medication escalation, frequent post-acute care transfers, and high readmission vulnerability.

---

## 3. Sub-Pages & Interface Layout
The tool is structured into two complementary views:
1. **The Two Profiles:** Overview page showing side-by-side comparison cards, cohort percentages (62% vs 38%), readmission curves, and feature contrast tables.
2. **Where Does This Patient Fit:** Interactive form where clinicians enter a patient's details; the model projects the patient into PCA space, assigns them to Profile 0 or Profile 1, and displays distance to cluster centroids.

---

## 4. Input Fields, Defaults, and Valid Ranges
From `models/moez/cluster_bundle.joblib`:

| Field Name | Description | Type / Range | Default |
|---|---|---|---|
| `time_in_hospital` | Length of stay (days) | Numeric: 1 to 14 | `4` |
| `num_lab_procedures` | Number of lab tests | Numeric: 1 to 132 | `43` |
| `num_procedures` | Number of procedures | Numeric: 0 to 6 | `1` |
| `num_medications` | Number of medications | Numeric: 1 to 81 | `15` |
| `number_outpatient` | Outpatient visits past year | Numeric: 0 to 42 | `0` |
| `number_emergency` | Emergency visits past year | Numeric: 0 to 42 | `0` |
| `number_inpatient` | Inpatient stays past year | Numeric: 0 to 12 | `0` |
| `number_diagnoses` | Number of diagnoses | Numeric: 1 to 16 | `8` |
| `age` | Age bracket | Brackets: [0-10) to [90-100) | `[60-70)` |
| `gender` | Gender | Female, Male | `Female` |
| `insulin` | Insulin status | No, Steady, Up, Down | `No` |
| `metformin` | Metformin status | No, Steady, Up, Down | `No` |
| `change` | Medication change | Yes, No | `No` |
| `diabetesMed` | Diabetes medication prescribed | Yes, No | `Yes` |
| `diag_1` | Primary diagnosis (ICD-9) | ICD-9 code or family | `Circulatory` |
| `admission_type_id` | Admission type | 1 (Emergency), 2 (Urgent), 3 (Elective), etc. | `1` |
| `admission_source_id` | Admission source | 7 (ER), 1 (Referral), 4 (Transfer), etc. | `7` |
| `discharge_disposition_id` | Discharge destination | 1 (Home), 3 (SNF), 6 (Home care), etc. | `1` |
| `A1Cresult` | HbA1c result | None, Norm, >7, >8 | `None` |
| `max_glu_serum` | Glucose serum test | None, Norm, >200, >300 | `None` |

---

## 5. Result Display & Visualisation
- **Assigned Profile Badge:** Prominently displays `Profile 0 (Routine / Lower-Complexity)` or `Profile 1 (Complex / High-Intensity)`.
- **Readmission Context Card:** Shows historical readmission rates for the assigned cluster (Profile 0: 7.7% early / 37.3% total; Profile 1: 11.1% early / 46.4% total).
- **Proximity / Distance to Centroids:** Visual gauge showing how closely the patient aligns with the cluster core.
- **Key Diverging Factors:** Displays patient's values vs cluster averages on the top 5 differentiating features.

---

## 6. Recommendations & Clinical Banner
- **Decision Support Banner:**  
  *"Read this with care: Patient profiles are derived from unsupervised clustering (K-Means k=2) across 69,970 patient stays. They describe clinical phenotypes, not an individual clinical prognosis."*
- **Recommendation Rules:**
  - **Patient fits Profile 1 (Complex / High-Intensity):** *"Patient fits Profile 1 (high therapeutic intensity, 46.4% historical readmission rate). Consider enhanced discharge coordination, comprehensive medication reconciliation, and scheduling post-discharge follow-up within 7 days."*
  - **Patient fits Profile 0 (Routine / Lower-Complexity):** *"Patient fits Profile 0 (routine clinical course, 37.3% historical readmission rate). Standard discharge pathways and routine primary care follow-up recommended."*

---

## 7. Example Patients
- **Profile 0 Example Patient:**  
  - Inputs: Stay 2 days, 9 medications, 0 prior hospital stays, no medication changes, discharge home.  
  - **Output:** Assigned to `Profile 0` (Routine / Lower-Complexity).
- **Profile 1 Example Patient:**  
  - Inputs: Stay 8 days, 24 medications, 2 prior inpatient stays, medication changed (`change=Ch`), Insulin increased, discharge to facility, HbA1c >8.  
  - **Output:** Assigned to `Profile 1` (Complex / High-Intensity).
