"""Cross-dataset PD/N/ET: combine local data with PADS to break the ET ceiling.

The local dataset has ~16 ET subjects; PADS adds ~28 -> ~44 pooled. PADS is
wrist-only, so features here are **single-sensor** (hand for local, wrist for
PADS): STFT-256 spectral profile + biomarker + regularity features. Spatial
(3-sensor) features are intentionally excluded — they don't exist in PADS.

Three protocols:
  * P1  train-local / test-PADS (and reverse)   — generalisation
  * P2  pooled leave-one-subject-out            — n-fix (44 ET), + a dataset-
        identity probe to flag domain confounding
  * P3  leave-one-dataset-out                    — strongest generalisation claim

Run: python -m pdetn.pads_experiment --pads-root /path/to/PADS
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from tremor.biomarker import FEATURE_NAMES, recording_features
from tremor.data import CLASS_NAMES, Recording
from tremor.quaternion_data import load_quaternion_recordings
from pdetn.separability import method_features
from pdetn.signal_features import ADVANCED_FEATURE_NAMES, advanced_features

# Single-sensor slices (3 angular-velocity channels each). A wrist smartwatch
# (PADS) sits closest to lower_arm; hand is the most distal.
SENSOR_SLICES = {"hand": slice(0, 3), "lower_arm": slice(3, 6), "upper_arm": slice(6, 9)}


def load_local_sensor(data_root, action="OUT", sensor="lower_arm"):
    """Local recordings restricted to ONE sensor (3 ch) to match PADS wrist."""
    sl = SENSOR_SLICES[sensor]
    recs = load_quaternion_recordings(data_root, action=action, mode="angular_velocity")
    out = []
    for r in recs:
        out.append(Recording(x=r.x[sl], y=r.y, subject=f"LOCAL_{r.subject}",
                             path=r.path, condition=action))
    return out


def load_local_hand(data_root, action="OUT"):
    """Backwards-compatible: hand sensor."""
    return load_local_sensor(data_root, action=action, sensor="hand")


def load_pads_extracted(folder):
    """Load the StretchHold data extracted by pdetn.extract_pads (<cls>_<pid>_<wrist>.txt)."""
    import re
    cmap = {"N": 0, "PD": 1, "ET": 2}
    recs = []
    for f in sorted(Path(folder).glob("*.txt")):
        m = re.match(r"(N|PD|ET)_(\d+)_(\w+)", f.stem)
        if not m:
            continue
        cls, pid, _ = m.groups()
        x = np.loadtxt(f, delimiter=",", ndmin=2).T          # (3, T) gyro
        recs.append(Recording(x=x.astype(np.float32), y=cmap[cls],
                             subject=f"PADS_{pid}", path=f, condition="OUT"))
    return recs


def _hand_feats(x, fs=100.0):
    d = recording_features(x, fs=fs)               # biomarker (uses ch 0-2)
    d.update(advanced_features(x, fs=fs))          # regularity/sharpness
    return [d[f] for f in (*FEATURE_NAMES, *ADVANCED_FEATURE_NAMES)]


def build_features(recs, fs=100.0, f_max=15.0):
    """Per-patient features = STFT-256 spectral profile + hand biomarker/signal.

    Feature dimension is fixed by (3 channels, nfft, f_max), so local and PADS
    tables share columns even with different recording lengths.
    """
    Xtf, ytf, subj = method_features(recs, "stft", fs=fs, f_max=f_max,
                                     nperseg=256, nfft=256, noverlap=192)
    # aggregate TF to patient level
    pats = sorted(set(subj.tolist()))
    tf = {p: Xtf[subj == p].mean(0) for p in pats}
    # hand biomarker/signal per patient
    hb = defaultdict(list); lab = {}
    for r in recs:
        hb[r.subject].append(_hand_feats(r.x, fs)); lab[r.subject] = r.y
    X = np.stack([np.concatenate([tf[p], np.nanmean(hb[p], 0)]) for p in pats])
    y = np.array([lab[p] for p in pats])
    dataset = np.array([p.split("_")[0] for p in pats])   # LOCAL / PADS
    return X, y, np.array(pats), dataset


# --------------------------------------------------------------------------- #
# Protocols
# --------------------------------------------------------------------------- #
def _metrics(y, y_pred, subj):
    from tremor.evaluate import classification_report
    from tremor.stats import bootstrap_subject_ci
    onehot = np.eye(len(CLASS_NAMES))[y_pred]
    rep = classification_report(np.log(onehot + 1e-6), y, CLASS_NAMES)
    ci = bootstrap_subject_ci(y, y_pred, subj, CLASS_NAMES, n_boot=1000)
    return {"macro_f1": rep["macro_f1"],
            "per_class_f1": {c: rep["per_class"][c]["f1"] for c in CLASS_NAMES},
            "ET_ci": {"lo": ci["ET"].lo, "hi": ci["ET"].hi},
            "confusion": rep["confusion_matrix"]}


def _fresh_model():
    from pdetn.model import TwoStageClassifier
    return TwoStageClassifier("logreg", "logreg", tune_et_threshold=True)


def protocol_p1(Xtr, ytr, Xte, yte, subj_te):
    """Train on one dataset, test on the other (external generalisation)."""
    m = _fresh_model().fit(Xtr, ytr)
    return _metrics(yte, m.predict(Xte), subj_te)


def protocol_p2_pooled_loso(X, y, subj):
    """Pooled leave-one-subject-out over the combined cohort (the n-fix)."""
    from pdetn.evaluate import evaluate
    return evaluate(_fresh_model, X, y, subj, n_boot=2000)


def dataset_identity_probe(X, dataset):
    """Can a classifier tell which dataset a recording came from? High AUC =>
    strong domain shift => pooled disease results are confounded."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import cross_val_score
    d = (dataset == "PADS").astype(int)
    pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(max_iter=2000))
    return float(cross_val_score(pipe, X, d, cv=5, scoring="roc_auc").mean())
