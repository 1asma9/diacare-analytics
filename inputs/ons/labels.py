"""Choices shown in the form: (value used by the model, label shown to the user).
The labels of the *_id fields follow IDS_mapping.csv of the UCI dataset; 'Rare' groups the codes that have
less than 1 % of the patients (the notebook merges them in merge_rare)."""
GENDER = [("Female", "Female"), ("Male", "Male")]
RACE = [("Caucasian", "Caucasian"), ("AfricanAmerican", "African American"), ("Hispanic", "Hispanic"),
        ("Rare", "Asian"), ("Other", "Other"), ("Unknown", "Not recorded")]
ADMISSION = [("1", "Emergency"), ("2", "Urgent"), ("3", "Elective"), ("5", "Not available"),
             ("6", "Not recorded"), ("Rare", "Other")]
SOURCE = [("7", "Emergency room"), ("1", "Physician referral"), ("2", "Clinic referral"),
          ("4", "Transfer from a hospital"), ("6", "Transfer from another health care facility"),
          ("17", "Not recorded"), ("Rare", "Other")]
DISCHARGE = [("1", "Home"), ("6", "Home with home-health care"), ("3", "Skilled nursing facility"),
             ("2", "Other short-term hospital"), ("22", "Rehabilitation facility"), ("5", "Other inpatient care"),
             ("18", "Not recorded"), ("25", "Not mapped"), ("Rare", "Other")]
DIAG = {"Circulatory": "Circulatory (heart, vessels)", "Diabetes": "Diabetes", "Respiratory": "Respiratory",
        "Digestive": "Digestive", "Injury": "Injury / poisoning", "Musculoskeletal": "Musculoskeletal",
        "Genitourinary": "Genitourinary", "Neoplasms": "Neoplasms (cancer)", "Other": "Other"}
