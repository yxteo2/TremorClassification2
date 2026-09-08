"""Paired robust-scaling test. Never remove a patient based on model errors."""
import argparse
import json
import hashlib
from pathlib import Path
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from experiments.inhouse_2015_explore import Columns, metric, patient_features


def fit_model(x, y, cv, robust=False, fixed=False):
    scaler = RobustScaler() if robust else StandardScaler()
    pipe = Pipeline([('columns', Columns()), ('scale', scaler),
        ('clf', LogisticRegression(C=.1, class_weight='balanced', max_iter=3000))])
    if fixed:
        return pipe.fit(x, y)
    representations = [tuple(range(6,12)), tuple(range(18)), tuple(range(36))]
    grid = [{'columns__indices': representations,
             'clf': [LogisticRegression(class_weight='balanced', max_iter=3000)],
             'clf__C': [.01,.1,1.]},
            {'columns__indices': representations,
             'clf': [SVC(class_weight='balanced', kernel='rbf', probability=False)],
             'clf__C': [.1,1.,10.]}]
    model = GridSearchCV(pipe, grid, scoring='f1_macro', cv=cv, n_jobs=1, error_score='raise')
    model.fit(x, y)
    return model.best_estimator_


def inner_splits(y, extra_count):
    extra = np.arange(len(y), len(y)+extra_count)
    return [(np.r_[a,extra],b) for a,b in
        StratifiedKFold(3,shuffle=True,random_state=0).split(np.zeros(len(y)),y)]


def quaternion_quality(q):
    q = np.asarray(q, dtype=float).reshape(-1,3,4)
    if not np.isfinite(q).all():
        return dict(nonfinite=True, near_zero_norm=False, norm_deviation_review=False)
    norms = np.linalg.norm(q,axis=-1)
    return dict(nonfinite=False, near_zero_norm=bool((norms < 1e-6).any()),
        norm_deviation_review=bool((np.abs(norms-1) > .1).any()))


def audit(records):
    from common.load_2025 import _load_h5
    counts = dict(nonfinite=0, near_zero_norm=0, norm_deviation_review=0)
    for r in records:
        p = Path(r.path)
        q = _load_h5(str(p)) if p.suffix == '.h5' else np.loadtxt(p,delimiter=',')
        flags = quaternion_quality(q)
        for k,v in flags.items(): counts[k] += int(v)
    return dict(recordings=len(records), flagged_recording_counts=counts,
        note='Audit of loaded recordings only; >10% norm deviation is review-only, not proof of artifact. No automatic exclusions.')


def main():
    from common.quaternion_data import load_quaternion_recordings_multi
    from common.load_2025 import load_2025_all
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    args=parser.parse_args();out=Path(args.output)
    if out.exists(): parser.error('Choose a new output directory')
    print('Loading 2015 and NewData',flush=True)
    ra=load_quaternion_recordings_multi('Data',['OUT','REST'])
    rb=load_2025_all(conditions=('OUT','REST'))
    x,y,p,a=patient_features(ra,'2015');xn,yn,pn,b=patient_features(rb,'NewData')
    if np.bincount(y,minlength=3).tolist()!=[61,72,15] or np.bincount(yn,minlength=3).tolist()!=[25,22,6]:
        raise ValueError('Patient population differs from PR #5; inspect data before comparing')
    quality={'2015':audit(ra),'NewData':audit(rb)}
    out.mkdir(parents=True)
    protocol=dict(arms=['standard_nested','robust_nested','standard_fixed','robust_fixed'],
        primary='robust_nested minus standard_nested on 148 2015 test patients',
        training='2015 outer training + all 53 NewData; inner validation only 2015',
        seed=0,outer_folds=5,inner_folds=3,fixed_model='all 36 features balanced logistic C=.1',
        removal='No patients, recordings or features removed',quality=quality,
        feature_sha256={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in [('2015',x),('NewData',xn)]})
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    arms=protocol['arms'];pred={k:np.full(len(y),-1) for k in arms};selections=[];splits=[]
    for fold,(tr,te) in enumerate(StratifiedKFold(5,shuffle=True,random_state=0).split(x,y)):
        xx=np.vstack([x[tr],xn]);yy=np.r_[y[tr],yn];cv=inner_splits(y[tr],len(yn));sel={}
        for arm in arms:
            m=fit_model(xx,yy,cv,robust=arm.startswith('robust'),fixed=arm.endswith('fixed'))
            pred[arm][te]=m.predict(x[te])
            sel[arm]=dict(features=len(m['columns'].indices),model=type(m['clf']).__name__,C=m['clf'].C)
        selections.append(sel);splits.append(dict(train=p[tr].tolist(),test=p[te].tolist(),extra_train=pn.tolist()))
        print(f'Fold {fold+1}/5 complete',flush=True)
    result={k:metric(y,v) for k,v in pred.items()};rng=np.random.default_rng(0)
    delta={k:[] for k in ['nested','fixed']}
    for _ in range(2000):
        ix=rng.integers(len(y),size=len(y))
        for k in delta:
            def score(arm):
                m=metric(y[ix],pred[arm][ix]);return np.array([m['macro_f1'],m['precision'][2],m['recall'][2]])
            delta[k].append(score('robust_'+k)-score('standard_'+k))
    result['paired_95']={k:dict(zip(['macro_f1','ET_precision','ET_recall'],np.percentile(v,[2.5,97.5],axis=0).T.tolist())) for k,v in delta.items()}
    result.update(protocol=protocol,selections=selections)
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    (out/'splits.json').write_text(json.dumps(splits,indent=2))
    np.savez_compressed(out/'predictions.npz',patients=p,y=y,**pred)
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
