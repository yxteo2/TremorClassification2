"""Does the catch22 waveform family beat the spectral families on PD vs ET?

Every feature family here is derived from the power spectrum or from the
instantaneous-frequency trajectory. Häring et al. (Movement Disorders, 2025)
report **81.8 % accuracy / 86.4 % sensitivity / 76.6 % specificity** for PD vs ET
from massive time-series feature extraction on 414 patients with hand
accelerometry, against **70.4 %** for the Tremor Stability Index. This repo
implements TSI (`stability.py`) and measures it at AUC 0.757 on PADS, so their
baseline and ours agree -- which makes their headline the interesting number.

Their mechanistic reading is testable: *"different discrete but stable signal
states in PD indicate several central oscillators, while signal characteristics
in ET point towards a singular pacemaker."* Six of the 22 catch22 features
measure exactly that (`STATE_FEATURES` in `signal_processing/catch22_features.py`),
so the `state subset` arm below is a direct test of the claim rather than of the
whole set.

Arms per cohort, PD vs ET only:

  descriptors        the standing best on PADS (AUC 0.794)
  stability          the TSI family, their comparator
  spectrum           log-binned multitaper
  catch22            the new family, 22 features
  catch22 state subset  the 6 features that encode their mechanism
  descriptors + catch22  the union, which the repo's rule predicts will dilute

Both a logistic regression and a **linear SVM** are run: Häring report linear
SVMs reaching 86.1 % and beating random forests, which matches this project's
own finding that simple models win at this n.

Every single-family claim is checked against a **permutation null** with the
whole pipeline refitted per replicate, per invariant 6 -- at 21 in-house ET the
null spans [0.298, 0.655] and a bootstrap will call chance results significant.

Run: ``python -m experiments.catch22_family``
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from signal_processing.catch22_features import (FEATURE_NAMES, STATE_FEATURES,
                                                patient_table as c22_table)

REPEATS, NPERM = 10, 200
COLS = ("AUC", "precPD", "precET", "macroP", "ETsens")
STATE_IDX = [FEATURE_NAMES.index(n) for n in STATE_FEATURES]


def clf(kind):
    if kind == "logreg":
        return make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=5000,
                                                class_weight="balanced"))
    return make_pipeline(StandardScaler(),
                         SVC(kernel="linear", class_weight="balanced",
                             probability=True, random_state=0))


def oof(X, y, k, kind, repeats=REPEATS, seed0=0):
    acc = np.zeros(len(y))
    for rep in range(repeats):
        p = np.zeros(len(y))
        for tr, te in StratifiedKFold(k, shuffle=True,
                                      random_state=seed0 + rep).split(X, y):
            p[te] = clf(kind).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        acc += p
    return acc / repeats


def scores(y, p):
    pr = (p >= np.quantile(p, 1 - y.mean())).astype(int)
    se = recall_score(y, pr, pos_label=1, zero_division=0)
    pPD = precision_score(y, pr, pos_label=0, zero_division=0)
    pET = precision_score(y, pr, pos_label=1, zero_division=0)
    return [roc_auc_score(y, p), pPD, pET, 0.5 * (pPD + pET), se]


def perm_p(X, y, k, kind, obs, n=NPERM):
    """Permutation null with the whole pipeline refitted on each replicate."""
    rng = np.random.default_rng(0)
    null = []
    for i in range(n):
        yp = rng.permutation(y)
        try:
            null.append(roc_auc_score(yp, oof(X, yp, k, kind, repeats=1,
                                              seed0=1000 + i)))
        except ValueError:
            pass
    null = np.array(null)
    lo, hi = np.percentile(null, [2.5, 97.5])
    p = (1 + np.sum(np.abs(null - null.mean()) >= abs(obs - null.mean()))) \
        / (1 + len(null))
    return null.mean(), lo, hi, p


def main():
    import os

    from common.cohorts import desc_table, logbin
    from common.load_2025 import load_2025_all
    from common.loaders import load_pads_extracted
    from common.quaternion_data import load_quaternion_recordings
    from experiments.final_model import method_table
    from frequency.tables import spectrum_table
    from signal_processing.stability import patient_table as stab_table

    cohorts = [("2015", load_quaternion_recordings("Data", action="OUT",
                                                   mode="angular_velocity"),
                slice(3, 6)),
               ("NewData", load_2025_all(conditions=("OUT",)), slice(3, 6))]
    if os.path.isdir("pads_stretchhold"):
        cohorts.append(("PADS", load_pads_extracted("pads_stretchhold"),
                        slice(0, 3)))

    print("building tables ...", flush=True)
    blocks = {}
    for tag, recs, ch in cohorts:
        sp = spectrum_table(recs, ch=ch)
        c22, y22, p22 = c22_table(recs, ch=ch)
        # catch22 can drop a recording; align to the spectrum table's patients
        idx = {p: i for i, p in enumerate(p22)}
        C = np.array([c22[idx[p]] if p in idx else np.zeros(len(FEATURE_NAMES))
                      for p in sp[2]])
        blocks[tag] = {
            "descriptors": desc_table(recs, ch),
            "stability": stab_table(recs, ch=ch)[0],
            "spectrum": logbin(method_table(recs, "multitaper", ch)[0]),
            "catch22": C,
            "catch22 state subset": C[:, STATE_IDX],
            "y": sp[1],
        }
        miss = sum(1 for p in sp[2] if p not in idx)
        print(f"  {tag:>8}: {len(sp[1])} patients, catch22 {C.shape}"
              f"{f', {miss} without catch22' if miss else ''}")

    groups = {"PADS": ["PADS"], "in-house": ["2015", "NewData"],
              "MERGED": ["2015", "NewData", "PADS"]}
    FAMILIES = ("descriptors", "stability", "spectrum", "catch22",
                "catch22 state subset")

    for gname, tags in groups.items():
        tags = [t for t in tags if t in blocks]
        y3 = np.concatenate([blocks[t]["y"] for t in tags])
        m = y3 != 0
        y = (y3[m] == 2).astype(int)
        k = 3 if gname == "in-house" else 5

        def blk(name):
            return np.nan_to_num(np.vstack([blocks[t][name]
                                            for t in tags]))[m]

        print(f"\n{'='*94}")
        print(f"{gname}  PD vs ET  n={len(y)}  ET={int(y.sum())}  "
              f"prevalence {y.mean():.3f}   {REPEATS} repeats, "
              f"{NPERM} permutations")
        print(f"{'='*94}")
        print(f"{'family':>24}{'clf':>8}{'dim':>5}" +
              "".join(f"{c:>9}" for c in COLS) + f"{'null 95%':>18}{'p':>8}")

        store = {}
        for fam in FAMILIES:
            X = blk(fam)
            for kind in ("logreg", "svm"):
                p = oof(X, y, k, kind)
                s = scores(y, p)
                nm, lo, hi, pv = perm_p(X, y, k, kind, s[0])
                store[(fam, kind)] = (s, p)
                flag = "*" if pv < 0.05 else " "
                print(f"{fam:>24}{kind:>8}{X.shape[1]:>5}"
                      + "".join(f"{v:>9.3f}" for v in s)
                      + f"{f'[{lo:.3f},{hi:.3f}]':>18}{pv:>7.3f}{flag}",
                      flush=True)

        # the union the repo's rule predicts will dilute
        U = np.hstack([blk("descriptors"), blk("catch22")])
        for kind in ("logreg", "svm"):
            p = oof(U, y, k, kind)
            s = scores(y, p)
            nm, lo, hi, pv = perm_p(U, y, k, kind, s[0])
            flag = "*" if pv < 0.05 else " "
            print(f"{'descriptors + catch22':>24}{kind:>8}{U.shape[1]:>5}"
                  + "".join(f"{v:>9.3f}" for v in s)
                  + f"{f'[{lo:.3f},{hi:.3f}]':>18}{pv:>7.3f}{flag}", flush=True)

        # which catch22 features separate the classes, on this cohort
        C = blk("catch22")
        d = []
        for j, nm_ in enumerate(FEATURE_NAMES):
            a, b = C[y == 0, j], C[y == 1, j]
            sd = np.sqrt(0.5 * (a.var() + b.var())) + 1e-12
            d.append((abs(a.mean() - b.mean()) / sd, nm_,
                      "state" if nm_ in STATE_FEATURES else ""))
        d.sort(reverse=True)
        print(f"\n  top catch22 features by |Cohen's d| (PD vs ET):")
        for val, nm_, tag in d[:6]:
            print(f"    {val:.3f}  {nm_:<44}{tag}")

    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    main()
