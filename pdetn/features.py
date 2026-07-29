"""Per-patient, condition-aware feature vectors for N/PD/ET separation.

Built on this project's findings:
  * N-vs-tremor is easy; PD-vs-ET is the hard axis.
  * PD-vs-ET separates at REST (lower, higher-power tremor for PD), and the
    rest-vs-action power *contrast* captures the PD=rest / ET=action dichotomy.

So each patient becomes ONE feature vector that stacks the interpretable
spectral biomarkers (``tremor.biomarker``) for each available condition plus
explicit cross-condition contrasts. Missing conditions become NaN (a handful
of patients lack one), to be imputed on the training fold only.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from tremor.biomarker import FEATURE_NAMES, recording_features
from tremor.data import CLASS_NAMES

CONDITIONS = ("OUT", "REST", "WING")


def _contrast_names() -> list[str]:
    names = []
    for a, b in (("REST", "WING"), ("REST", "OUT"), ("OUT", "WING")):
        names.append(f"ctr_logpow_{a}_{b}")     # log power ratio
        names.append(f"ctr_domshift_{a}_{b}")    # dominant-frequency shift
    return names


def feature_names(conditions=CONDITIONS) -> list[str]:
    names = [f"{c}__{f}" for c in conditions for f in FEATURE_NAMES]
    return names + _contrast_names()


def build_patient_table(recs, conditions=CONDITIONS, fs: float = 100.0):
    """Return (X, y, subjects, names).

    X : (n_patients, n_features) with NaN for absent conditions.
    y : int label per patient (N/PD/ET). subjects: patient ids.
    """
    # patient -> condition -> list of per-recording feature dicts
    per: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    label: dict[str, int] = {}
    for r in recs:
        per[r.subject][r.condition].append(recording_features(r.x, fs=fs))
        label[r.subject] = r.y

    names = feature_names(conditions)
    patients = sorted(per)
    X = np.full((len(patients), len(names)), np.nan, dtype=float)

    def cond_mean(pid, cond, feat):
        recs_f = per[pid].get(cond)
        if not recs_f:
            return np.nan
        return float(np.mean([rf[feat] for rf in recs_f]))

    col = {n: i for i, n in enumerate(names)}
    for pi, pid in enumerate(patients):
        for c in conditions:
            for f in FEATURE_NAMES:
                X[pi, col[f"{c}__{f}"]] = cond_mean(pid, c, f)
        # contrasts
        for a, b in (("REST", "WING"), ("REST", "OUT"), ("OUT", "WING")):
            pa, pb = cond_mean(pid, a, "pow_total"), cond_mean(pid, b, "pow_total")
            da, db = cond_mean(pid, a, "dom_freq"), cond_mean(pid, b, "dom_freq")
            if np.isfinite(pa) and np.isfinite(pb):
                X[pi, col[f"ctr_logpow_{a}_{b}"]] = np.log((pa + 1e-9) / (pb + 1e-9))
            if np.isfinite(da) and np.isfinite(db):
                X[pi, col[f"ctr_domshift_{a}_{b}"]] = da - db

    y = np.array([label[p] for p in patients])
    return X, y, np.array(patients), names
