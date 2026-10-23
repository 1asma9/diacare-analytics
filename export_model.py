"""Rebuilds the StayCast model from df_clean.csv and writes model.json (used by index.html).
Usage: python export_model.py [path/to/df_clean.csv]   (needs pandas, numpy, scikit-learn)"""
import sys
import json
import numpy as np, pandas as pd, json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
df=pd.read_csv(sys.argv[1] if len(sys.argv)>1 else 'df_clean.csv')
df['long_stay']=(df.time_in_hospital>3).astype(int)
dm=df.drop(columns=['readmitted'])
y=dm.long_stay; X=dm.drop(columns=['time_in_hospital','long_stay'])
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
cv=["num_lab_procedures","num_procedures","num_medications","number_outpatient","number_emergency","number_inpatient","number_diagnoses"]
p99=Xtr[cv].quantile(.99); print(p99.to_dict())
for c in cv: Xtr[c]=Xtr[c].clip(upper=p99[c]); Xte[c]=Xte[c].clip(upper=p99[c])
for c in ["number_outpatient","number_emergency","number_inpatient","num_medications"]:
    Xtr['log_'+c]=np.log1p(Xtr[c]); Xte['log_'+c]=np.log1p(Xte[c])
def cat(code):
    if pd.isna(code): return "Missing"
    code=str(code)
    if code[0] in "VE": return "Other"
    try: v=float(code)
    except: return "Other"
    if 250<=v<251: return "Diabetes"
    if 390<=v<=459 or v==785: return "Circulatory"
    if 460<=v<=519 or v==786: return "Respiratory"
    if 520<=v<=579 or v==787: return "Digestive"
    if 800<=v<=999: return "Injury"
    if 710<=v<=739: return "Musculoskeletal"
    if 580<=v<=629 or v==788: return "Genitourinary"
    if 140<=v<=239: return "Neoplasms"
    return "Other"
for c in ["diag_1","diag_2","diag_3"]:
    Xtr[c+'_cat']=Xtr[c].apply(cat); Xte[c+'_cat']=Xte[c].apply(cat)
top=Xtr.loc[Xtr.medical_specialty!="Unknown","medical_specialty"].value_counts().head(10).index
f=lambda s:"Unknown" if (pd.isna(s) or s=="Unknown") else (s if s in top else "Other")
Xtr['medical_specialty_grp']=Xtr.medical_specialty.apply(f); Xte['medical_specialty_grp']=Xte.medical_specialty.apply(f)
for S in (Xtr,Xte):
    S['gender_bin']=S.gender.map({"Female":0,"Male":1}); S['change_bin']=S.change.map({"No":0,"Ch":1}); S['diabetesMed_bin']=S.diabetesMed.map({"No":0,"Yes":1})
cc=["diag_1_cat","diag_2_cat","diag_3_cat","medical_specialty_grp"]
Otr=pd.get_dummies(Xtr[cc],prefix=cc); Ote=pd.get_dummies(Xte[cc],prefix=cc).reindex(columns=Otr.columns,fill_value=0)
sc=["num_lab_procedures","num_procedures","number_diagnoses","log_number_outpatient","log_number_emergency","log_number_inpatient","log_num_medications"]
sk=StandardScaler(); Str=pd.DataFrame(sk.fit_transform(Xtr[sc]),columns=[c+'_scaled' for c in sc],index=Xtr.index); Ste=pd.DataFrame(sk.transform(Xte[sc]),columns=Str.columns,index=Xte.index)
fb=["gender_bin","change_bin","diabetesMed_bin"]
Ftr=pd.concat([Str,Xtr[fb],Otr],axis=1); Fte=pd.concat([Ste,Xte[fb],Ote],axis=1)
corr=Ftr.assign(y=ytr).corr(numeric_only=True)['y'].drop('y')
cand=corr[corr.abs()>.1].sort_values(key=abs,ascending=False).index.tolist(); 
pc=Ftr[cand].corr().abs(); drop=set()
for i,a in enumerate(cand):
    for b in cand[i+1:]:
        if a in drop or b in drop: continue
        if pc.loc[a,b]>.8: drop.add(a if abs(corr[a])<abs(corr[b]) else b)
sel=[c for c in cand if c not in drop]
adm=[c for c in sel if not c.startswith(("num_lab_procedures","log_num_medications","num_medications"))]


from sklearn.linear_model import LogisticRegression
mu=dict(zip(sc,sk.mean_)); sd=dict(zip(sc,sk.scale_))
def z(col,x):
    if col=="num_medications": return (np.log1p(x)-mu["log_num_medications"])/sd["log_num_medications"]
    return (x-mu[col])/sd[col]
RAW={"number_diagnoses":(1,9),"num_procedures":(0,6),"change_bin":(0,1),"num_medications":(1,44),"num_lab_procedures":(1,85)}
full=df.copy(); full["change_bin"]=full.change.map({"No":0,"Ch":1})
for c,(lo,hi) in RAW.items(): full[c]=full[c].clip(lo,hi)
model_json={}
for name,cols in (("admission",adm),("after",sel)):
    m=AdaBoostClassifier(n_estimators=200,learning_rate=.5,random_state=42).fit(Ftr[cols],ytr)
    raw=[c.replace("log_","").replace("_scaled","") for c in cols]; W=m.estimator_weights_.sum()
    # AdaBoost stumps are additive: sum them per variable into one lookup table each
    tabs=[]
    for c,r in zip(cols,raw):
        t=[]
        for x in range(RAW[r][0],RAW[r][1]+1):
            zz=x if r=="change_bin" else z(r,x); s=0
            for est,w in zip(m.estimators_,m.estimator_weights_):
                tr=est.tree_
                if cols[tr.feature[0]]!=c: continue
                cls=tr.value[1][0].argmax() if zz<=tr.threshold[0] else tr.value[2][0].argmax()
                s+=(1 if cls==1 else -1)*w/W
            t.append(round(s,4))
        tabs.append(t)
    off=[RAW[r][0] for r in raw]
    score=lambda df_: sum(np.array(tabs[i])[(df_[r].astype(int)-off[i]).values] for i,r in enumerate(raw))
    D=score(full); lr=LogisticRegression(C=1e6).fit(D.reshape(-1,1),full.long_stay)   # probability calibration
    a,b=float(lr.coef_[0][0]),float(lr.intercept_[0]); P=1/(1+np.exp(-(a*D+b)))
    edges=np.quantile(P,np.linspace(.1,.9,9)); bi=np.searchsorted(edges,P)
    dist,med,q1,q3=[],[],[],[]
    for k in range(10):
        t=full.time_in_hospital[bi==k]
        dist.append([round(float((t==d).mean()),3) for d in range(1,15)])
        q=t.quantile([.25,.5,.75]).round().astype(int).tolist(); q1.append(q[0]); med.append(q[1]); q3.append(q[2])
    mean=[round(float(np.mean(tabs[i])),4) for i in range(len(raw))]
    mean=[round(float(np.mean(np.array(tabs[i])[(full[r].astype(int)-off[i]).values])),4) for i,r in enumerate(raw)]
    model_json[name]=dict(cols=raw,t=tabs,off=off,a=round(a,3),b=round(b,3),mean=mean,edges=[round(float(e),4) for e in edges],dist=dist,med=med,q1=q1,q3=q3)
json.dump(model_json,open("model.json","w"),separators=(",",":"))
print("model.json written")
