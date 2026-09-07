"""Fixed PADS rest/posture power comparison; patient-grouped five-fold evaluation.

Unlike task_contrast, power is measured before spectrum normalization.
No task, seed, band or C search. All arms and patient-level intervals reported.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scipy.signal import welch
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

BANDS = ((3, 5), (5, 7), (7, 10), (10, 15.0001))
ARMS = ('reference', 'posture_logistic', 'tasks_logistic', 'tasks_fusion')


def features(x, fs=100):
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[0] != 3 or x.shape[1] < fs * 4 or not np.isfinite(x).all():
        raise ValueError('Expected finite three-axis recording of at least four seconds')
    f, p = welch(x, fs=fs, nperseg=400, noverlap=200, axis=-1)
    p = p.sum(axis=0)  # preserve total vector power, without rectifying waveform
    df = f[1] - f[0]
    powers = [p[(f >= lo) & (f < hi)].sum() * df for lo, hi in BANDS]
    m = (f >= 3) & (f <= 15)
    q = p[m] / max(p[m].sum(), 1e-30)
    peak = f[m][p[m].argmax()]
    entropy = -(q * np.log(q + 1e-30)).sum() / np.log(len(q))
    return np.r_[np.log(np.maximum(powers, 1e-20)), peak, entropy]


def table(records):
    rows, labels = {}, {}
    for r in records:
        if r.subject in labels and labels[r.subject] != r.y:
            raise ValueError('Conflicting patient labels')
        labels[r.subject] = r.y
        rows.setdefault(r.subject, []).append(features(r.x))
    return {p: np.mean(v, axis=0) for p, v in rows.items()}, labels


def align(post, rest, patients, y):
    a, la = table(post)
    b, lb = table(rest)
    keys = [str(p).removeprefix('PADS:') for p in patients]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate patient rows')
    if any(p not in a or p not in b for p in keys):
        raise ValueError('Missing task: do not silently change the benchmark population')
    if any(la[p] != label or lb[p] != label for p, label in zip(keys, y)):
        raise ValueError('Task label mismatch')
    ap, br = np.array([a[p] for p in keys]), np.array([b[p] for p in keys])
    # Six shared levels and six explicit contrasts; no redundant concatenation.
    return ap, np.hstack([(ap + br) / 2, ap - br])


def fit(x, y, tr, va, te):
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1,
        class_weight='balanced', max_iter=3000, solver='lbfgs'))
    clf.fit(x[tr], y[tr])
    return clf.predict_proba(x[va]), clf.predict_proba(x[te])


def main():
    import torch
    from common.loaders import load_pads_extracted
    from common.protocol import tune_offsets
    from experiments.scattering_benchmark import build, make_splits, reference, metrics
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    out = Path(args.output)
    if out.exists():
        parser.error('Choose a new output directory')
    torch.set_num_threads(1)
    d, _ = build('pads')
    y, patients = d['y'], d['patients']
    xp, xt = align(load_pads_extracted('pads_stretchhold'),
        load_pads_extracted('pads_relaxed'), patients, y)
    splits = make_splits(y, patients, y)
    out.mkdir(parents=True)
    protocol = dict(counts=np.bincount(y).tolist(), seed=0, C=0.1, fs=100,
        bands=BANDS, primary='tasks_logistic minus posture_logistic macro-F1',
        secondary='tasks_logistic and tasks_fusion versus reference',
        note='Exploratory previously studied PADS; fixed OOF bootstrap excludes retraining and selection uncertainty')
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2))
    (out/'splits.json').write_text(json.dumps([{k:patients[ix].tolist() for k,ix in
        zip(('train','validation','test'),s)} for s in splits],indent=2))
    probs = {a:np.zeros((len(y),3)) for a in ARMS}
    preds = {a:np.full(len(y),-1) for a in ARMS}
    fold_ids = np.full(len(y),-1)
    offsets=[]
    for fold,(tr,va,te) in enumerate(splits):
        print(f'Fold {fold+1}/5: reference and task classifiers',flush=True)
        rv,rt=reference(d,tr,va,te)
        pv,pt=fit(xp,y,tr,va,te)
        tv,tt=fit(xt,y,tr,va,te)
        vs=(rv,pv,tv,(rv+tv)/2); ts=(rt,pt,tt,(rt+tt)/2)
        off={}
        for a,v,t in zip(ARMS,vs,ts):
            o=tune_offsets(v,y[va]); off[a]=o.tolist()
            probs[a][te]=t; preds[a][te]=(np.log(t+1e-12)+o).argmax(1)
        offsets.append(off); fold_ids[te]=fold
        np.savez_compressed(out/f'fold_{fold}.npz',patients=patients[te],y=y[te],
            **{f'p_{a}':probs[a][te] for a in ARMS},**{f'pred_{a}':preds[a][te] for a in ARMS})
        print(f'Fold {fold+1}/5 complete',flush=True)
    comparisons=(('tasks_logistic','posture_logistic'),('tasks_logistic','reference'),('tasks_fusion','reference'))
    rng=np.random.default_rng(0); delta={f'{a}-minus-{b}':[] for a,b in comparisons}
    for _ in range(2000):
        ix=rng.integers(len(y),size=len(y))
        s={a:f1_score(y[ix],preds[a][ix],labels=[0,1,2],average='macro',zero_division=0) for a in ARMS}
        for a,b in comparisons: delta[f'{a}-minus-{b}'].append(s[a]-s[b])
    result={a:metrics(y,preds[a],probs[a]) for a in ARMS}
    result.update(protocol=protocol,offsets=offsets,paired_macro_f1_95={k:np.percentile(v,[2.5,97.5]).tolist() for k,v in delta.items()})
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    with (out/'predictions.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['patient','fold','label','arm','prediction','pN','pPD','pET'])
        for i,p in enumerate(patients):
            for a in ARMS:w.writerow([p,fold_ids[i],y[i],a,preds[a][i],*probs[a][i]])
    for a in ARMS:print(a,result[a]['macro_f1'],result[a]['precision'][2],result[a]['recall'][2])

if __name__=='__main__':main()
