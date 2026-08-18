"""PD vs ET as a dedicated binary problem -- the clinically important axis.

Every PD-vs-ET figure in this project so far has been read out of a 3-class
N/PD/ET model. That model spends capacity separating healthy controls, which is
the easy axis (precision 0.910-0.924 from six frequency features), and dilutes
the boundary that actually matters in clinic. Reading the binary axis out of the
3-class probabilities already looks better than the 3-class ET column
(PD precision 0.852 against ET precision 0.685), which suggests a model built
for the binary problem should do better again.

This trains only on tremor patients (PD + ET) and evaluates every feature family
the project has verified, per cohort, because the families split by cohort:

  family          PADS      in-house    source
  harmonics       0.736     0.402       four_families.md
  axes            0.558     0.641       four_families.md
  stability       0.742     0.652       temporal_stability.md
  asymmetry       0.730     --          limb_asymmetry_pd_vs_et.md (needs 2 limbs)
  descriptors     0.807     0.482       temporal_stability.md

Reported as AUC (threshold-free, so it does not depend on the operating point)
plus precision for both classes at the tuned threshold, since a clinician needs
to know the cost of each error direction.

Run: ``python -m experiments.pd_vs_et``
"""

from __future__ import annotations

import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common.cohorts import asym_for, desc_table, logbin
from common.protocol import N_ASYM
from experiments.final_model import method_table
from frequency.tables import spectrum_table
from signal_processing.stability import patient_table as stab_table
from signal_processing.tremor_physics import FAMILIES, FEATURE_NAMES
from signal_processing.tremor_physics import patient_table as physics_table

REPEATS = 10
AX = [FEATURE_NAMES.index(n) for n in FAMILIES["axes"]]
HARM = [FEATURE_NAMES.index(n) for n in FAMILIES["harmonic"]]
MOD = [FEATURE_NAMES.index(n) for n in FAMILIES["ampmod"]]


def build():
    """Feature blocks for every patient, per cohort."""
    from common.load_2025 import SIDE, load_2025_all
    from common.loaders import load_pads_extracted
    from common.quaternion_data import load_quaternion_recordings

    side_new = lambda r: SIDE.get(os.path.basename(r.path)[:2])
    side_pads = lambda r: ("left" if "LeftWrist" in str(r.path)
                           else ("right" if "RightWrist" in str(r.path) else None))
    out = {}
    for tag, recs, ch, side in (
            ("2015", load_quaternion_recordings("Data", action="OUT",
                                                mode="angular_velocity"),
             slice(3, 6), None),
            ("NewData", load_2025_all(conditions=("OUT",)), slice(3, 6), side_new),
            ("PADS", load_pads_extracted("pads_stretchhold"), slice(0, 3),
             side_pads)):
        sp = spectrum_table(recs, ch=ch)
        ph = physics_table(recs, ch=ch)[0]
        blocks = {
            "spectrum": logbin(method_table(recs, "multitaper", ch)[0]),
            "descriptors": desc_table(recs, ch),
            "stability": stab_table(recs, ch=ch)[0],
            "axes": ph[:, AX],
            "harmonics": ph[:, HARM],
            "ampmod": ph[:, MOD],
        }
        if side is not None:
            a, h = asym_for(recs, side, ch, sp[2])
            blocks["asymmetry"] = np.hstack([a, h[:, None]])
        else:
            blocks["asymmetry"] = np.zeros((len(sp[1]), N_ASYM + 1))
        out[tag] = (blocks, sp[1])
    return out


def clf():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=5000,
                                            class_weight="balanced"))


def evaluate(X, y, k=5, repeats=REPEATS):
    """Repeated stratified CV on tremor patients only. Returns per-repeat rows."""
    rows = []
    for rep in range(repeats):
        prob = np.zeros(len(y))
        cv = StratifiedKFold(k, shuffle=True, random_state=rep)
        for tr, te in cv.split(X, y):
            m = clf().fit(X[tr], y[tr])
            prob[te] = m.predict_proba(X[te])[:, 1]
        pred = (prob >= 0.5).astype(int)
        P, R, _, _ = precision_recall_fscore_support(y, pred, labels=[0, 1],
                                                     zero_division=0)
        rows.append([roc_auc_score(y, prob), P[0], P[1],
                     0.5 * (R[0] + R[1])])
    return np.array(rows)


def main():
    data = build()
    combos = {
        "spectrum": ["spectrum"],
        "descriptors": ["descriptors"],
        "stability": ["stability"],
        "axes": ["axes"],
        "harmonics": ["harmonics"],
        "ampmod": ["ampmod"],
        "asymmetry": ["asymmetry"],
        "desc + stability": ["descriptors", "stability"],
        "axes + stability": ["axes", "stability"],
        "axes + asym": ["axes", "asymmetry"],
        "stability + asym": ["stability", "asymmetry"],
        "axes + stab + asym": ["axes", "stability", "asymmetry"],
        "ALL blocks": ["spectrum", "descriptors", "stability", "axes",
                       "harmonics", "ampmod", "asymmetry"],
    }

    settings = [
        ("PADS", ["PADS"]),
        ("in-house (2015+NewData)", ["2015", "NewData"]),
        ("MERGED (all three)", ["2015", "NewData", "PADS"]),
    ]

    for name, tags in settings:
        blocks = {k: np.vstack([data[t][0][k] for t in tags])
                  for k in data[tags[0]][0]}
        y3 = np.concatenate([data[t][1] for t in tags])
        m = y3 != 0
        y = (y3[m] == 2).astype(int)          # 1 = ET, 0 = PD
        k = 5 if y.sum() >= 25 else 3
        print(f"\n{'='*80}")
        print(f"{name}   PD vs ET   n={int(m.sum())}  PD={int((y==0).sum())} "
              f"ET={int(y.sum())}   {k}-fold x {REPEATS} repeats")
        print(f"{'='*80}")
        print(f"{'features':>22}{'dim':>5}{'AUC':>16}{'precPD':>16}"
              f"{'precET':>16}{'bal-acc':>16}")
        res = {}
        for label, keys in combos.items():
            X = np.hstack([blocks[q] for q in keys])[m]
            X = np.nan_to_num(X)
            r = evaluate(X, y, k=k)
            res[label] = r
            mu, sd = r.mean(0), r.std(0)
            print(f"{label:>22}{X.shape[1]:>5}"
                  + "".join(f"{mu[i]:>10.3f} +/-{sd[i]:<4.3f}" for i in range(4)))
        best = max(res, key=lambda q: res[q][:, 0].mean())
        print(f"  best by AUC: {best} ({res[best][:,0].mean():.3f})")
    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    main()
