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


def load_pads_extracted(folder, strict=True, task=None):
    """Load the StretchHold data extracted by pdetn.extract_pads.

    The filename class token is NOT trusted. An earlier version of
    ``extract_pads`` mapped diagnoses by substring, so the bare token "et"
    matched "etiology", "asymmetric", "Retrocollis" and "hypokinetic": 13 of 41
    files named ``ET_*`` are not Essential Tremor, including parkinsonian cases
    (a hypokinetic-rigid syndrome, a Lewy-Body dementia). 20 ``PD_*`` files are
    Atypical Parkinsonism, which PADS treats as a separate group.

    With ``strict=True`` (default) the class is re-derived from the manifest's
    ``raw_label`` by EXACT match, and every ambiguous or mixed diagnosis is
    dropped. That gives N=79 / PD=276 / ET=28, and the ET count then agrees with
    the published PADS cohort (Varghese 2024: 28 ET).

    ``strict=False`` reproduces the old contaminated behaviour; it exists only
    to re-derive the superseded numbers and should not be used for new results.

    ``task`` filters by PADS task substring (e.g. ``"Relaxed"`` matches both
    Relaxed1 and Relaxed2, ``"StretchHold"`` the postural task). ``None`` loads
    everything in the folder, which is correct for a single-task folder.
    """
    import csv
    import re

    cmap = {"N": 0, "PD": 1, "ET": 2}
    # Accept several folders. PADS repetitions (Relaxed1, Relaxed2) must be
    # extracted separately because each run rewrites manifest.csv, so the
    # natural layout is one folder per repetition -- load them together here.
    if isinstance(folder, (list, tuple)):
        out = []
        for one in folder:
            out.extend(load_pads_extracted(one, strict=strict, task=task))
        return out
    folder = Path(folder)
    manifest = folder / "manifest.csv"
    exact = {"healthy": "N", "parkinson's": "PD", "essential tremor": "ET"}

    true_cls, file_task = {}, {}
    if strict:
        if not manifest.is_file():
            raise FileNotFoundError(
                f"{manifest} is required for strict labelling; pass strict=False "
                "to fall back to the (contaminated) filename labels.")
        for row in csv.DictReader(manifest.open()):
            lab = exact.get(row["raw_label"].strip().lower())
            if lab:
                true_cls[row["file"]] = lab
            file_task[row["file"]] = (row.get("task") or "").strip()

    recs = []
    for f in sorted(folder.glob("*.txt")):
        # two layouts: legacy <cls>_<pid>_<wrist> and current
        # <cls>_<pid>_<task>_<wrist> (the task token was added so repetitions
        # like Relaxed1/Relaxed2 stop overwriting each other)
        m = re.match(r"(N|PD|ET)_(\d+)_(\w+)", f.stem)
        if not m:
            continue
        cls, pid, _ = m.groups()
        if task is not None:
            parts = f.stem.split("_")
            t = file_task.get(f.name) or (parts[2] if len(parts) > 3 else "")
            if task.lower() not in t.lower():
                continue
        if strict:
            cls = true_cls.get(f.name)
            if cls is None:            # ambiguous / non-N-PD-ET diagnosis
                continue
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


# --------------------------------------------------------------------------- #
# Rotation- and scale-invariant features (for cross-dataset work)
# --------------------------------------------------------------------------- #
def invariant_features(recs, n_bins: int = 40, lo: float = 3.0, hi: float = 15.0,
                       fs: float = 100.0):
    """Per-patient features invariant to sensor orientation AND amplitude scale.

    Summing the PSD across the 3 axes gives the **trace of the spectral matrix**,
    which is unchanged by any rotation of the sensor coordinate frame — so this
    achieves what explicit coordinate correction would, without needing either
    dataset's reference frame (which also sidesteps the fact that per-subject
    mounting orientation is not recorded in either dataset). Normalising the
    resulting spectrum to sum 1 additionally removes amplitude scale.

    Measured effect: the dataset-identity probe drops from AUC 0.999 (full
    features) to 0.526 (chance) — i.e. the local/PADS domain shift is largely an
    orientation+scale effect. See reports/crossdataset_results.md.
    """
    from collections import defaultdict
    from scipy.signal import welch
    per = defaultdict(list)
    label = {}
    for r in recs:
        n = int(min(256, r.x.shape[1]))
        f, P = welch(r.x, fs=fs, nperseg=n, axis=-1)
        P = P.sum(axis=0)                         # rotation-invariant (trace)
        m = (f >= lo) & (f < hi)
        p = P[m] / (P[m].sum() + 1e-18)           # scale-invariant (shape)
        per[r.subject].append(np.interp(np.linspace(lo, hi, n_bins), f[m], p))
        label[r.subject] = r.y
    pats = sorted(per)
    X = np.array([np.mean(per[k], axis=0) for k in pats])
    y = np.array([label[k] for k in pats])
    return X, y, np.array(pats)
