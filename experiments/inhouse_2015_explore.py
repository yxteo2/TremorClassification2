"""2015-first nested-CV screen, optionally including NewData. No PADS.

Primary model selects representation and classifier in inner folds only.
All repeated recordings/tasks are pooled within the same patient beforehand.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score
from sklearn.base import BaseEstimator, TransformerMixin
from experiments.task_power_benchmark import features

class Columns(TransformerMixin, BaseEstimator):
    def __init__(self, indices=tuple(range(36))):self.indices=indices
    def fit(self,x,y=None):return self
    def transform(self,x):return x[:,self.indices]


def patient_features(records,cohort):
    if not records:raise ValueError("No recordings loaded; finish data retrieval before running")
    bags={}; labels={}; audit={}; lengths=[]
    for r in records:
        p=f'{cohort}:{r.subject}'
        if p in labels and labels[p]!=r.y:raise ValueError('Conflicting patient labels')
        labels[p]=r.y
        if r.x.shape[0]!=9:raise ValueError('Expected three three-axis sensors')
        row=np.concatenate([features(r.x[s:s+3]) for s in (0,3,6)])
        bags.setdefault((p,r.condition),[]).append(row);lengths.append(r.x.shape[1]/100)
    for task in sorted({t for _,t in bags}):
        ps=[p for p,t in bags if t==task]
        audit[task]={'patients':len(ps),'counts_N_PD_ET':np.bincount([labels[p] for p in ps],minlength=3).tolist(),
                     'recordings':sum(len(bags[p,task]) for p in ps)}
    kept=sorted(p for p in labels if all((p,t) in bags for t in ('OUT','REST')))
    x=np.array([np.concatenate([np.mean(bags[p,t],axis=0) for t in ('OUT','REST')]) for p in kept])
    return x,np.array([labels[p] for p in kept]),np.array(kept),dict(tasks=audit,
        duration_range_s=[min(lengths),max(lengths)],excluded_missing_OUT_or_REST=len(labels)-len(kept))


def search(x,y,cv=None):
    representations=[tuple(range(6,12)),tuple(range(18)),tuple(range(36))]
    pipe=Pipeline([('columns',Columns()),('scale',StandardScaler()),('clf',LogisticRegression())])
    grid=[{'columns__indices':representations,'clf':[LogisticRegression(class_weight='balanced',max_iter=3000)],
           'clf__C':[0.01,0.1,1.]},
          {'columns__indices':representations,'clf':[SVC(class_weight='balanced',kernel='rbf',probability=False)],
           'clf__C':[0.1,1.,10.]}]
    g=GridSearchCV(pipe,grid,scoring='f1_macro',cv=cv if cv is not None else StratifiedKFold(3,shuffle=True,random_state=0),n_jobs=1,error_score='raise')
    g.fit(x,y);return g


def metric(y,p):
    pr,re,f,s=precision_recall_fscore_support(y,p,labels=[0,1,2],zero_division=0)
    return dict(accuracy=accuracy_score(y,p),macro_f1=float(f.mean()),precision=pr.tolist(),recall=re.tolist(),support=s.tolist(),confusion=confusion_matrix(y,p,labels=[0,1,2]).tolist())


def evaluate(x,y,patients,out):
    pred=np.full(len(y),-1);base=pred.copy();foldid=pred.copy();selection=[];splits=[]
    for fold,(tr,te) in enumerate(StratifiedKFold(5,shuffle=True,random_state=0).split(x,y)):
        g=search(x[tr],y[tr]);pred[te]=g.predict(x[te])
        b=Pipeline([('scale',StandardScaler()),('clf',LogisticRegression(C=0.1,class_weight='balanced',max_iter=3000))])
        b.fit(x[tr,6:12],y[tr]);base[te]=b.predict(x[te,6:12]);foldid[te]=fold
        selected=g.best_estimator_;selection.append(dict(fold=fold,features=len(selected['columns'].indices),
            classifier=type(selected['clf']).__name__,C=selected['clf'].C,inner_macro_f1=g.best_score_))
        splits.append({'train':patients[tr].tolist(),'test':patients[te].tolist()})
        print(out.name,fold+1,selection[-1],flush=True)
    result={'nested_selected':metric(y,pred),'fixed_middle_OUT_logistic':metric(y,base),'selections':selection,
        'by_cohort':{c:metric(y[m],pred[m]) for c in sorted({p.split(':')[0] for p in patients})
                     for m in [np.array([p.startswith(c+':') for p in patients])]}}
    rng=np.random.default_rng(0);v=[]
    for _ in range(2000):
        ix=rng.integers(len(y),size=len(y));v.append(metric(y[ix],pred[ix])['macro_f1']-metric(y[ix],base[ix])['macro_f1'])
    result['paired_macro_f1_difference_95']=np.percentile(v,[2.5,97.5]).tolist()
    out.mkdir();(out/'splits.json').write_text(json.dumps(splits,indent=2))
    with (out/'predictions.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['patient','fold','label','nested_prediction','baseline_prediction']);w.writerows(zip(patients,foldid,y,pred,base))
    return result


def augmentation(x,y,xn,yn,baseline):
    """Hold 2015 outer tests fixed; NewData enters inner/outer training only."""
    pred=np.full(len(y),-1); selections=[]
    for fold,(tr,te) in enumerate(StratifiedKFold(5,shuffle=True,random_state=0).split(x,y)):
        xx=np.vstack([x[tr],xn]);yy=np.r_[y[tr],yn]
        extra=np.arange(len(tr),len(yy))
        cv=[(np.r_[a,extra],b) for a,b in StratifiedKFold(3,shuffle=True,random_state=0).split(x[tr],y[tr])]
        g=search(xx,yy,cv);pred[te]=g.predict(x[te])
        selections.append(dict(fold=fold,features=len(g.best_estimator_['columns'].indices),
            classifier=type(g.best_estimator_['clf']).__name__,C=g.best_estimator_['clf'].C))
    rng=np.random.default_rng(0);delta=[]
    for _ in range(2000):
        ix=rng.integers(len(y),size=len(y))
        delta.append(metric(y[ix],pred[ix])['macro_f1']-metric(y[ix],baseline[ix])['macro_f1'])
    return dict(metrics=metric(y,pred),selections=selections,
        paired_macro_f1_95=np.percentile(delta,[2.5,97.5]).tolist(),
        note='Compared with 2015-only nested selection on exactly the same 2015 outer tests. Inner validation also restricted to 2015.')


def main():
    from common.quaternion_data import load_quaternion_recordings_multi
    from common.load_2025 import load_2025_all
    a=argparse.ArgumentParser();a.add_argument('--output',required=True);a.add_argument('--include-newdata',action='store_true');args=a.parse_args()
    out=Path(args.output)
    if out.exists():a.error('Choose a new output directory')
    out.mkdir(parents=True)
    print('Loading 2015 OUT/REST/WING',flush=True)
    x,y,p,audit=patient_features(load_quaternion_recordings_multi('Data',['OUT','REST','WING']),'2015')
    np.savez_compressed(out/'features_2015.npz',x=x,y=y,patients=p)
    result={'audit_2015':audit,'2015':evaluate(x,y,p,out/'2015')}
    (out/'2015_checkpoint.json').write_text(json.dumps(result,indent=2)+'\n')
    if args.include_newdata:
        print('Loading NewData OUT/REST',flush=True)
        xn,yn,pn,an=patient_features(load_2025_all(conditions=('OUT','REST')),'NewData')
        np.savez_compressed(out/'features_newdata.npz',x=xn,y=yn,patients=pn)
        result['audit_newdata']=an
        result['pooled']=evaluate(np.vstack([x,xn]),np.r_[y,yn],np.r_[p,pn],out/'pooled')
        g=search(x,y);result['train_2015_test_newdata']=metric(yn,g.predict(xn))
        rows=list(csv.DictReader((out/'2015'/'predictions.csv').open()))
        baseline=np.array([int(r['nested_prediction']) for r in rows])
        result['newdata_training_only']=augmentation(x,y,xn,yn,baseline)
    result['protocol']={'outer_folds':5,'inner_folds':3,'seed':0,'primary':'nested selected versus fixed middle-sensor OUT logistic on 2015',
        'representations':['middle sensor OUT (6)','three sensors OUT (18)','three sensors OUT + REST (36)'],
        'models':'balanced logistic C=.01,.1,1; balanced RBF SVM C=.1,1,10 gamma=scale',
        'notes':'Exploratory historical data. Only OUT+REST complete patients. No cohort ID input. NewData uses existing 10-second tremor epoch selector; 2015 full recording. WING audited but not used in matched comparisons. Bootstrap conditions on fixed OOF predictions.'}
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
