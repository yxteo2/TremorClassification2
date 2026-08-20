"""Swap the reported model's spectrum for the short-window one.

`tf_window_paired.py` established, paired over 30 repeats with logistic
regression, that a spectrum built as the **mean of 0.64 s STFT frames** beats the
current multitaper representation at equal dimensionality:

    PADS PD vs ET     AUC +0.036 [+0.031, +0.042] *
                   precET +0.088 [+0.073, +0.102] *
                   macroP +0.049 [+0.040, +0.056] *
    MERGED PD vs ET   AUC +0.030 * , precET +0.032 * , macroP +0.018 *

and that on PADS it beats coarse binning alone (D vs B significant on every
column), so it is not merely this project's known "coarser is better" lever.

Two explanations were tested and settled first (`tf_window_control.py`): the
across-frame estimator is irrelevant — a **mean** does as well as a median, which
refutes the robust-estimation story I first proposed — and on MERGED coarseness
explains the precision gain while the short window adds only ranking.

All of that is logistic regression. This is the question that decides whether it
matters: **does the reported two-stream deep model improve when its spectrum
input is swapped?** Nothing else changes — same descriptors, same asymmetry, same
IF trajectory, same architecture, same validation-tuned priors, same folds.

  A. reported model                multitaper, log-binned to 16
  B. short-window spectrum         mean of 0.64 s frames, 16 bins

30 splits rather than the usual 20. A paired macroP difference near 0.02 was
shown to evaporate on doubling in `early_fusion_confirm.md`, so the split count
is raised in advance for a difference of the size expected here.

Run: ``python -m experiments.shortwindow_deep``
"""

from __future__ import annotations

import os

import numpy as np
import torch

from experiments.alltasks_final import evaluate, paired
from experiments.final_model import build
from experiments.tf_window_control import WIN
from frequency.tables import spectrum_table
from signal_processing.tf_variability import blocks, patient_table

NM = ("precN", "precPD", "precET", "macroP", "macroF1")
SPLITS = 30


def short_window_spectrum(order):
    """(patients, 16) mean-of-short-frames log spectrum, on the reported order."""
    from common.load_2025 import load_2025_all
    from common.loaders import load_pads_extracted
    from common.quaternion_data import load_quaternion_recordings

    src = [(load_quaternion_recordings("Data", action="OUT",
                                       mode="angular_velocity"), slice(3, 6)),
           (load_2025_all(conditions=("OUT",)), slice(3, 6))]
    if os.path.isdir("pads_stretchhold"):
        src.append((load_pads_extracted("pads_stretchhold"), slice(0, 3)))

    B = blocks()
    parts, pats = [], []
    for recs, ch in src:
        X, _, p = patient_table(recs, ch=ch, nperseg=WIN, stat="mean")
        parts.append(X[:, B["median"]])
        pats.append(p)
    allX, allp = np.vstack(parts), np.concatenate(pats)
    idx = {q: i for i, q in enumerate(allp)}
    miss = [q for q in order if q not in idx]
    if miss:
        print(f"  WARNING {len(miss)} patients without a short-window spectrum")
    D = allX.shape[1]
    return np.array([allX[idx[q]] if q in idx else np.zeros(D) for q in order])


def main():
    torch.set_num_threads(1)
    d = build()
    y, key = d["y"], d["key"]
    D = np.hstack([d["DESC"], d["ASYM"], d["HAVE"]])
    traj = d["TRAJ"]
    S_mt = d["SPEC"]["multitaper"]

    from common.load_2025 import load_2025_all
    from common.loaders import load_pads_extracted
    from common.quaternion_data import load_quaternion_recordings
    A = spectrum_table(load_quaternion_recordings("Data", action="OUT",
                                                  mode="angular_velocity"),
                       ch=slice(3, 6))
    B_ = spectrum_table(load_2025_all(conditions=("OUT",)), ch=slice(3, 6))
    C = spectrum_table(load_pads_extracted("pads_stretchhold"), ch=slice(0, 3))
    rng = np.random.default_rng(0)
    keep = []
    for cl in (0, 1, 2):
        i = np.flatnonzero(C[1] == cl)
        keep.extend(rng.choice(i, min(90, len(i)), replace=False))
    keep = np.array(sorted(keep))
    order = np.concatenate([A[2], B_[2], C[2][keep]])
    assert np.array_equal(np.concatenate([A[1], B_[1], C[1][keep]]), y), \
        "patient order does not match build()"

    print("building the short-window spectrum ...", flush=True)
    S_sw = np.nan_to_num(short_window_spectrum(order))
    assert S_sw.shape == S_mt.shape, f"{S_sw.shape} vs {S_mt.shape}"
    print(f"n={len(y)}  N={int((y==0).sum())} PD={int((y==1).sum())} "
          f"ET={int((y==2).sum())}   {SPLITS} splits")
    print(f"spectra {S_mt.shape}, identical shape, "
          f"correlation {np.corrcoef(S_mt.ravel(), S_sw.ravel())[0,1]:.3f}\n",
          flush=True)

    ARMS = (("A reported (multitaper 16)", S_mt),
            ("B short-window mean 16", S_sw))
    res = {}
    print(f"{'arm':>28}" + "".join(f"{c:>9}" for c in NM) + "   sd(macroP)")
    for lab, S in ARMS:
        res[lab] = evaluate(S, D, traj, y, key, splits=SPLITS)
        print(f"{lab:>28}" + "".join(f"{v:>9.3f}" for v in res[lab].mean(0))
              + f"{res[lab][:, 3].std():>12.3f}", flush=True)

    print(f"\npaired B - A over {SPLITS} shared splits:")
    for (dd, lo, hi), c in zip(paired(res["B short-window mean 16"],
                                      res["A reported (multitaper 16)"]), NM):
        star = "*" if lo > 0 or hi < 0 else " "
        print(f"    {c:>8} {dd:+.3f}  [{lo:+.3f}, {hi:+.3f}] {star}")

    print("\nsplit-level win rate for B:")
    a, b = res["A reported (multitaper 16)"], res["B short-window mean 16"]
    for i, c in enumerate(NM):
        print(f"    {c:>8} {float((b[:, i] > a[:, i]).mean()):.2f}")
    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    main()
