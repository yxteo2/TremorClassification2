"""The rest-vs-postural AMPLITUDE ratio, which normalisation had been deleting.

`experiments/task_contrast.py` builds rest-vs-postural contrast features and
prints, for every class, a mean ``log total power`` contrast of exactly 0.000.
That is not a null result -- it is a measurement of the pipeline. Every spectrum
in this repo is sum-normalised (``v / v.sum()`` in ``method_table`` and
``spectrum_table``), so total power is 1 by construction and any ratio of totals
is 1 for everyone.

The normalisation is deliberate and correct for its original purpose: absolute
amplitude is not comparable **across** patients or cohorts, because sensor gain,
placement and units differ. But the ratio of two conditions **within one
patient** is a different quantity. The same sensor, on the same limb, in the same
session: gain cancels exactly. That ratio is the one number a neurologist forms
at the bedside, and it is the primary discriminator between the two diseases --

    Parkinson's tremor is a REST tremor, damped by voluntary posture:
        rest power > postural power.
    Essential tremor is a POSTURAL tremor, minimal at rest:
        postural power > rest power.

So the pipeline was normalising away the strongest physiological contrast it had
access to, and this measures what that costs.

Computed here **before** any normalisation: integrated 3-15 Hz power per
recording, averaged per patient per condition, then

  ``log_amp_ratio``   log(postural band power / rest band power)
  ``log_amp_post``    log postural band power, and
  ``log_amp_rest``    log rest band power -- included only as a control. These
                      two are NOT scale-invariant across cohorts and should help
                      much less than their difference; if they help more, the
                      model is reading a cohort signature, not physiology.

Arms on the reported model, changing only the descriptor block:

  postural only (reported)     baseline
  + log amplitude ratio        one scale-invariant number
  + ratio and both raw levels  the control described above
  + ratio and shape contrast   the ratio plus the 16-bin shape contrast from
                               task_contrast.py

Run: ``python -m experiments.amplitude_contrast``
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch

from experiments.alltasks_final import evaluate, paired
from experiments.final_model import SPLITS, build
from experiments.task_contrast import contrast_block, pairs
from frequency.tables import spectrum_table
from signal_processing.transforms import METHODS

NM = ("precN", "precPD", "precET", "macroP", "macroF1")


def band_power(recs, ch, f_lo=3.0, f_hi=15.0):
    """Mean UN-normalised 3-15 Hz power per patient. The point of this module."""
    acc = defaultdict(list)
    fn = METHODS["multitaper"]
    for r in recs:
        x = r.x[ch] if r.x.shape[0] > 3 else r.x
        f, P = fn(x)
        f, P = np.asarray(f, float), np.asarray(P, float)
        m = np.isfinite(P) & (f >= f_lo) & (f <= f_hi)
        if not m.any():
            continue
        acc[r.subject].append(float(np.trapezoid(P[m], f[m])))
    return {p: float(np.mean(v)) for p, v in acc.items() if np.mean(v) > 0}


def amplitude_block(order):
    """(log ratio, log postural, log rest, availability) per patient."""
    ratio, lpost, lrest, have = {}, {}, {}, {}
    for post, rest, ch in pairs():
        bp = band_power(post, ch)
        br = band_power(rest, ch) if len(rest) else {}
        for p, vp in bp.items():
            lpost[p] = float(np.log(vp))
            if p in br:
                lrest[p] = float(np.log(br[p]))
                ratio[p] = lpost[p] - lrest[p]
                have[p] = 1.0
    R = np.array([[ratio.get(p, 0.0)] for p in order])
    L = np.array([[lpost.get(p, 0.0), lrest.get(p, 0.0)] for p in order])
    H = np.array([[have.get(p, 0.0)] for p in order])
    return np.nan_to_num(R), np.nan_to_num(L), H


def main():
    torch.set_num_threads(1)
    d = build()
    y, key = d["y"], d["key"]
    D_post = np.hstack([d["DESC"], d["ASYM"], d["HAVE"]])
    traj, spec = d["TRAJ"], d["SPEC"]["multitaper"]

    ps = pairs()
    A, B_, C = (spectrum_table(ps[0][0], ch=ps[0][2]),
                spectrum_table(ps[1][0], ch=ps[1][2]),
                spectrum_table(ps[2][0], ch=ps[2][2]))
    rng = np.random.default_rng(0)
    keep = []
    for cl in (0, 1, 2):
        i = np.flatnonzero(C[1] == cl)
        keep.extend(rng.choice(i, min(90, len(i)), replace=False))
    keep = np.array(sorted(keep))
    order = np.concatenate([A[2], B_[2], C[2][keep]])
    assert np.array_equal(np.concatenate([A[1], B_[1], C[1][keep]]), y), \
        "patient order does not match build()"

    R, L, H = amplitude_block(order)
    CB, _, _ = contrast_block(order)

    print(f"n={len(y)}  N={int((y==0).sum())} PD={int((y==1).sum())} "
          f"ET={int((y==2).sum())}   {SPLITS} splits")
    print(f"rest recording available for {int(H.sum())}/{len(y)} patients\n")

    print("log(postural band power / rest band power), by class")
    print("  PD should be NEGATIVE (rest tremor), ET POSITIVE (postural tremor)")
    print(f"{'':>6}{'n':>6}{'mean':>10}{'median':>10}{'sd':>9}"
          f"{'frac > 0':>10}")
    have = H[:, 0] > 0
    for cl, nm in ((0, "N"), (1, "PD"), (2, "ET")):
        m = (y == cl) & have
        v = R[m, 0]
        print(f"{nm:>6}{m.sum():>6}{v.mean():>10.3f}{np.median(v):>10.3f}"
              f"{v.std():>9.3f}{(v > 0).mean():>10.3f}")

    # rank-based separation, no model involved
    from sklearn.metrics import roc_auc_score
    mt = have & (y != 0)
    if mt.sum() > 10:
        yt = (y[mt] == 2).astype(int)
        auc = roc_auc_score(yt, R[mt, 0])
        print(f"\n  PD vs ET from this ONE number: AUC {auc:.3f} "
              f"(n={int(mt.sum())}, ET={int(yt.sum())})")
    mn = have
    print(f"  N vs Tremor from this one number: "
          f"AUC {roc_auc_score((y[mn] != 0).astype(int), R[mn, 0]):.3f}\n")

    ARMS = (("postural only (reported)", D_post),
            ("+ log amplitude ratio", np.hstack([D_post, R, H])),
            ("+ ratio and raw levels", np.hstack([D_post, R, L, H])),
            ("+ ratio and shape contrast", np.hstack([D_post, R, CB, H])))

    res = {}
    print(f"{'arm':>30}" + "".join(f"{c:>9}" for c in NM) + "   sd(macroP)")
    for lab, D in ARMS:
        res[lab] = evaluate(spec, D, traj, y, key)
        m = res[lab].mean(0)
        print(f"{lab:>30}" + "".join(f"{v:>9.3f}" for v in m)
              + f"{res[lab][:, 3].std():>12.3f}", flush=True)

    base = res["postural only (reported)"]
    print("\npaired vs postural only, same splits:")
    for lab, _ in ARMS[1:]:
        print(f"  {lab}:")
        for (dd, lo, hi), c in zip(paired(res[lab], base), NM):
            star = "*" if lo > 0 or hi < 0 else " "
            print(f"    {c:>8} {dd:+.3f}  [{lo:+.3f}, {hi:+.3f}] {star}")
    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    main()
