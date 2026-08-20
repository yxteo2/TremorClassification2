"""Why does the short-window spectrum win? Three explanations, separated.

`tf_variability_screen.py` found the highest PADS PD-vs-ET AUC in this project —
**0.825** for the per-bin median of 0.64 s STFT frames, against 0.794 for the
standing `descriptors` baseline — with a clean monotone decline as the window
lengthens:

    window       0.64 s   1.28 s   2.56 s   5.12 s
    median AUC    0.825    0.775    0.764    0.716

The variability blocks it was built to test are all *weaker* (iqr at 2.56 s is
0.494, chance), and adding them to the median dilutes it (0.761). So the win is
not "how the spectrum moves" — it is something about short windows, and there are
three candidate explanations that the screen cannot tell apart:

  1. **Robust estimation.** A median over ~60 short frames resists transients and
     nonstationarity; a mean over 1-3 long frames does not.
  2. **Spectral smoothing.** At nperseg 64 the resolution is 1.56 Hz, so 3-15 Hz
     holds only ~8 independent points. The short window may simply be smoothing.
  3. **Effective coarseness.** This project's top-ranked lever is already "coarse-
     bin the spectrum to 16-32 bins". ~8 independent points is *coarser* than the
     16-bin multitaper, so the finding may be the known lever pushed further, with
     the time axis irrelevant.

Four arms separate them. Every arm ends in 8 or 16 features and uses the same
classifier, folds and permutation null:

  A  multitaper, 16 bins        the current pipeline — 0.794
  B  multitaper, 8 bins         tests (3): is coarser simply better?
  C  short-window MEDIAN, 16    the winner — 0.825
  D  short-window MEAN, 16      tests (1): same window, same everything, only
                                the across-frame estimator changes

Readings:

* **D ≈ C**  → the estimator is irrelevant; the gain is the window (2 or 3).
* **C > D**  → robust estimation is doing the work (1).
* **B ≈ C**  → coarseness explains it and the time axis is irrelevant (3).
* **B ≈ A**  → coarseness is *not* the explanation.

This is stated before the run because the previous two causal stories in this
session — that a raw-waveform TCN wastes capacity demodulating, and that
median-centring crippled the analytic stream — were asserted rather than tested,
and the first was wrong.

Run: ``python -m experiments.tf_window_control``
"""

from __future__ import annotations

import os

import numpy as np

from experiments.tf_variability_screen import COLS, NPERM, oof, perm_p, scores
from signal_processing.tf_variability import blocks, patient_table

WIN = 64


def logbin_n(X, nb):
    """Bin an un-binned spectrum table to nb bins, covering every column."""
    L = np.log(X + 1e-8)
    e = np.linspace(0, L.shape[1], nb + 1).round().astype(int)
    return np.stack([L[:, e[i]:e[i + 1]].mean(1) for i in range(nb)], 1)


def main():
    from common.load_2025 import load_2025_all
    from common.loaders import load_pads_extracted
    from common.quaternion_data import load_quaternion_recordings
    from experiments.final_model import method_table
    from frequency.tables import spectrum_table

    cohorts = [("2015", load_quaternion_recordings("Data", action="OUT",
                                                   mode="angular_velocity"),
                slice(3, 6)),
               ("NewData", load_2025_all(conditions=("OUT",)), slice(3, 6))]
    if os.path.isdir("pads_stretchhold"):
        cohorts.append(("PADS", load_pads_extracted("pads_stretchhold"),
                        slice(0, 3)))

    print("building tables ...", flush=True)
    B = blocks()
    store = {}
    for tag, recs, ch in cohorts:
        sp = spectrum_table(recs, ch=ch)
        raw = method_table(recs, "multitaper", ch)[0]
        d = {"y": sp[1],
             "mt16": logbin_n(raw, 16), "mt8": logbin_n(raw, 8)}
        for stat in ("median", "mean"):
            X, _, p = patient_table(recs, ch=ch, nperseg=WIN, stat=stat)
            idx = {q: i for i, q in enumerate(p)}
            dim = X.shape[1] if len(X) else 34
            T = np.array([X[idx[q]] if q in idx else np.zeros(dim)
                          for q in sp[2]])
            d[f"w{WIN}_{stat}"] = T[:, B["median"]]
        store[tag] = d
        cov = float((np.abs(d[f"w{WIN}_median"]).sum(1) > 0).mean())
        print(f"  {tag:>8}: {len(sp[1])} patients, coverage {cov:.0%}")

    ARMS = (("A multitaper, 16 bins (current)", "mt16"),
            ("B multitaper, 8 bins", "mt8"),
            ("C short-window MEDIAN, 16", f"w{WIN}_median"),
            ("D short-window MEAN, 16", f"w{WIN}_mean"))

    for gname, tags, k in (("PADS", ["PADS"], 5),
                           ("MERGED", ["2015", "NewData", "PADS"], 5)):
        tags = [t for t in tags if t in store]
        y3 = np.concatenate([store[t]["y"] for t in tags])
        m = y3 != 0
        y = (y3[m] == 2).astype(int)
        print(f"\n{'='*88}")
        print(f"{gname}  PD vs ET  n={len(y)}  ET={int(y.sum())}  "
              f"prevalence {y.mean():.3f}   {NPERM} permutations")
        print(f"{'='*88}")
        print(f"{'arm':>34}{'dim':>5}" + "".join(f"{c:>9}" for c in COLS)
              + f"{'null 95%':>18}{'p':>8}")
        for lab, key in ARMS:
            X = np.nan_to_num(np.vstack([store[t][key] for t in tags]))[m]
            s = scores(y, oof(X, y, k))
            lo, hi, pv = perm_p(X, y, k, s[0])
            print(f"{lab:>34}{X.shape[1]:>5}"
                  + "".join(f"{v:>9.3f}" for v in s)
                  + f"{f'[{lo:.3f},{hi:.3f}]':>18}{pv:>7.3f}"
                  + ("*" if pv < 0.05 else " "), flush=True)
    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    main()
