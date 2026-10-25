import joblib, re, warnings
import pandas as pd
import numpy as np

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    b = joblib.load('models/roua/model_bundle.joblib')

cols = b['columns']
scaler = b['scaler']
top_spec = b['top_specialties']
means = b['train_means']
model = b['model']

df = pd.read_csv('inputs/data/df_clean.csv', nrows=5)

def famille(code):
    c = str(code)
    if c.startswith('250'): return 'Diabete'
    if len(c) > 0 and c[0] in 'VE': return 'Autre'
    try: n = float(c)
    except: return 'Autre'
    if 390<=n<=459 or n==785: return 'Circulatoire'
    if 460<=n<=519 or n==786: return 'Respiratoire'
    if 520<=n<=579 or n==787: return 'Digestif'
    if 800<=n<=999: return 'Lesions'
    if 710<=n<=739: return 'Musculo'
    if 580<=n<=629 or n==788: return 'Genito_urinaire'
    if 140<=n<=239: return 'Tumeurs'
    return 'Autre'

def prepare_roua(raw_df):
    d = raw_df.copy()
    for c in ['diag_1','diag_2','diag_3']:
        if c in d.columns:
            d[c] = d[c].apply(famille)
        else:
            d[c] = 'Autre'
    if 'medical_specialty' in d.columns:
        d['medical_specialty'] = d['medical_specialty'].where(d['medical_specialty'].isin(top_spec), 'Autre')
    else:
        d['medical_specialty'] = 'Autre'
        
    d['gender'] = (d.get('gender', 'Female') == 'Male').astype(int)
    d['change'] = d.get('change', 'No').isin(['Ch', 'Yes', 1]).astype(int)
    d['diabetesMed'] = (d.get('diabetesMed', 'No') == 'Yes').astype(int)
    
    for c in ['admission_type_id','admission_source_id']:
        if c in d.columns:
            d[c] = d[c].astype(str)
        else:
            d[c] = '1'
            
    cont = ['num_procedures','number_outpatient','number_emergency','number_inpatient','number_diagnoses',
            'time_in_hospital','num_lab_procedures']
    for c in cont:
        if c not in d.columns:
            d[c] = means[c]
            
    d_encoded = pd.get_dummies(d, drop_first=True)
    noms = [re.sub(r'[\[\]<>,:\"{}\s]', '_', col) for col in d_encoded.columns]
    d_encoded.columns = noms
    
    X = d_encoded.reindex(columns=cols, fill_value=0).copy()
    X[cont] = scaler.transform(d[cont])
    return X

X = prepare_roua(df)
preds = model.predict(X)
probs = model.predict_proba(X)
print('Roua predictions for first 5 patients:', preds)
print('Class names:', b['class_names'])
for i, (pred, prob) in enumerate(zip(preds, probs)):
    print(f'Patient {i}: class={b["class_names"][pred]} (prob={prob.round(3)})')
