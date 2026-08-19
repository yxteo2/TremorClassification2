"""The welch rebinning discards the top fifth of the band; the multitaper one does not.

``common.cohorts.logbin`` reduces a spectrum to ``nb`` bins by reshaping:

```python
n = X.shape[1] // nb * nb
return L[:, :n].reshape(len(L), nb, -1).mean(2)
```

Whether that loses anything depends entirely on the width of what it is fed, and
this repo feeds it two different widths:

| producer | columns | ``nb=16`` keeps | band seen |
|---|---|---|---|
| ``frequency.tables.spectrum_table`` (welch band mask) | **61** | 48 | 3.13-12.30 Hz |
| ``experiments.final_model.method_table`` (interp onto ``GRID``) | **64** | 64 | 3.00-15.00 Hz |

64 // 16 * 16 = 64, so the multitaper path is exact and every result built on it
is unaffected. 61 // 16 * 16 = 48, so the welch path **silently discards
12.50-14.84 Hz -- 21 % of the band**. That path feeds ``common.cohorts.load_all``
(the merged table) and the unlabelled corpus in ``experiments.masked_pretrain``.

Two things follow, and this measures both.

**1. Does the discarded region carry anything?** Tremor fundamentals are 4-12 Hz
and always inside the retained range, so the question is about the **second
harmonic of a 6.3-7.4 Hz tremor**. Harmonic structure is the strongest of the
four physics families on PADS (0.736, `four_families.md`), so the region is not
obviously empty.

**2. Is welch-vs-multitaper confounded with it?** ``final_model.py`` ranks
multitaper above welch. Those two arms were binned from 64 and 61 columns
respectively, so part of that gap could be band coverage rather than the
estimator. Binning both to full coverage separates them.

Arms, same classifier and folds throughout:

  welch reshape nb=16       current merged-table path        3.13-12.30 Hz
  welch interp nb=16        same width, whole band           3.13-14.84 Hz
  welch raw 61              no reduction                     3.13-14.84 Hz
  multitaper reshape nb=16  current final-model path         3.00-15.00 Hz
  multitaper interp nb=16   identical here (64 divides 16)   3.00-15.00 Hz

``welch interp`` vs ``welch reshape`` is the controlled comparison: identical
dimensionality, differing only in whether the top of the band is represented.

Run: ``python -m experiments.binning``
"""

from __future__ import annotations

import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPEATS = 20
COLS = ("AUC", "precPD", "precET", "macroP", "ETsens")


def reshape_bin(X, nb):
    """The current logbin: log, then reshape-mean, dropping the remainder."""
    L = np.log(X + 1e-8)
    n = X.shape[1] // nb * nb
    return L[:, :n].reshape(len(L), nb, -1).mean(2)


def interp_bin(X, nb):
    """Log, then average over nb equal-width slices spanning the WHOLE band.

    Same output dimensionality as ``reshape_bin``; the only difference is that
    every input column contributes to some output bin.
    """
    L = np.log(X + 1e-8)
    e = np.linspace(0, L.shape[1], nb + 1).round().astype(int)
    return np.stack([L[:, e[i]:e[i + 1]].mean(1) for i in range(nb)], 1)


def raw_log(X, nb=None):
    return np.log(X + 1e-8)


def clf():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=5000,
                                            class_weight="balanced"))


def scores(y, p):
    pr = (p >= np.quantile(p, 1 - y.mean())).astype(int)
    se = recall_score(y, pr, pos_label=1, zero_division=0)
    pPD = precision_score(y, pr, pos_label=0, zero_division=0)
    pET = precision_score(y, pr, pos_label=1, zero_division=0)
    return [roc_auc_score(y, p), pPD, pET, 0.5 * (pPD + pET), se]


def paired(a, b, n=4000):
    d = a - b
    out = []
    for i in range(len(COLS)):
        boot = [np.mean(np.random.default_rng(s).choice(d[:, i], len(d),
                                                        replace=True))
                for s in range(n)]
        out.append((d[:, i].mean(), *np.percentile(boot, [2.5, 97.5])))
    return out


def evaluate(X, y, k=5):
    rows = []
    for rep in range(REPEATS):
        p = np.zeros(len(y))
        for tr, te in StratifiedKFold(k, shuffle=True,
                                      random_state=rep).split(X, y):
            p[te] = clf().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        rows.append(scores(y, p))
    return np.array(rows)


def main():
    from common.load_2025 import load_2025_all
    from common.loaders import load_pads_extracted
    from common.quaternion_data import load_quaternion_recordings
    from experiments.final_model import GRID, method_table
    from frequency.tables import spectrum_table

    f = np.linspace(0, 50, 257)
    fb = f[(f >= 3.0) & (f <= 15.0)]
    n48 = len(fb) // 16 * 16
    print(f"welch band     : {len(fb)} columns {fb[0]:.2f}-{fb[-1]:.2f} Hz")
    print(f"  reshape nb=16: keeps 0:{n48} -> {fb[0]:.2f}-{fb[n48-1]:.2f} Hz, "
          f"DISCARDS {fb[n48]:.2f}-{fb[-1]:.2f} Hz "
          f"({100*(len(fb)-n48)/len(fb):.0f} % of the band)")
    print(f"multitaper GRID: {len(GRID)} columns {GRID[0]:.2f}-{GRID[-1]:.2f} Hz")
    print(f"  reshape nb=16: keeps 0:{len(GRID)//16*16} -> exact, nothing dropped\n")

    cohorts = []
    for tag, fn, ch in (("2015", lambda: load_quaternion_recordings(
                            "Data", action="OUT", mode="angular_velocity"),
                         slice(3, 6)),
                        ("NewData", lambda: load_2025_all(conditions=("OUT",)),
                         slice(3, 6)),
                        ("PADS", lambda: load_pads_extracted("pads_stretchhold"),
                         slice(0, 3))):
        if tag == "PADS" and not os.path.isdir("pads_stretchhold"):
            continue
        try:
            cohorts.append((tag, fn(), ch))
        except Exception as e:
            print(f"  {tag} unavailable: {e}")

    W, M, Y = {}, {}, {}
    for tag, recs, ch in cohorts:
        sw, y3, _ = spectrum_table(recs, ch=ch)
        sm = method_table(recs, "multitaper", ch)[0]
        W[tag], M[tag], Y[tag] = np.nan_to_num(sw), np.nan_to_num(sm), y3
        print(f"  {tag}: welch {sw.shape}  multitaper {sm.shape}  "
              f"N/PD/ET {[int((y3==c).sum()) for c in (0,1,2)]}")

    ARMS = (("welch reshape nb=16 (current)", "W", reshape_bin, 16),
            ("welch interp nb=16 (full band)", "W", interp_bin, 16),
            ("welch raw 61", "W", raw_log, None),
            ("multitaper reshape nb=16", "M", reshape_bin, 16),
            ("multitaper interp nb=16", "M", interp_bin, 16))

    groups = {"PADS": ["PADS"], "in-house": ["2015", "NewData"],
              "MERGED": ["2015", "NewData", "PADS"]}

    for gname, tags in groups.items():
        tags = [t for t in tags if t in Y]
        if not tags:
            continue
        y3 = np.concatenate([Y[t] for t in tags])
        src = {"W": np.vstack([W[t] for t in tags]),
               "M": np.vstack([M[t] for t in tags])}

        for axis in ("PD vs ET", "N vs Tremor"):
            if axis == "PD vs ET":
                m = y3 != 0
                y, pos = (y3[m] == 2).astype(int), "ET"
            else:
                m = np.ones(len(y3), bool)
                y, pos = (y3 != 0).astype(int), "Tremor"
            print(f"\n{'='*82}")
            print(f"{gname}  {axis}  n={len(y)}  {pos}={int(y.sum())}  "
                  f"prevalence {y.mean():.3f}")
            print(f"{'='*82}")
            print(f"{'representation':>32}{'dim':>5}" +
                  "".join(f"{c:>9}" for c in COLS))
            res = {}
            for lab, key, fn, nb in ARMS:
                S = src[key][m]
                X = fn(S, nb) if nb else fn(S)
                res[lab] = evaluate(X, y)
                print(f"{lab:>32}{X.shape[1]:>5}" +
                      "".join(f"{v:>9.3f}" for v in res[lab].mean(0)))
            base = res["welch reshape nb=16 (current)"]
            print("\n  paired vs welch reshape nb=16 (the current merged path):")
            for lab, _, _, _ in ARMS[1:]:
                for (d, lo, hi), c in zip(paired(res[lab], base), COLS):
                    if c in ("AUC", "precET", "macroP"):
                        star = "*" if lo > 0 or hi < 0 else " "
                        print(f"    {lab:>32} {c:>7} {d:+.3f} "
                              f"[{lo:+.3f}, {hi:+.3f}] {star}")
    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    main()
