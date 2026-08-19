"""Rest vs postural as a contrast, not an average.

`experiments/alltasks_final.py` found that averaging every task into the
per-patient spectrum **helps** healthy-vs-tremor detection (precN +0.047
[+0.009, +0.088]) and **significantly hurts** ET precision (−0.104
[−0.170, −0.035]) on the reported model. Macro precision nets out flat.

There is a clinical reason, and it is the central fact about these two diseases:

    Parkinson's tremor is characteristically a REST tremor, damped by voluntary
    posture. Essential tremor is a POSTURAL / kinetic tremor, absent or minimal
    at rest.

So rest-vs-postural is not one more task pair among many -- it is the primary
bedside discriminator between PD and ET. Averaging a rest recording into a
postural one destroys exactly the contrast that carries the PD-vs-ET signal,
while adding signal-to-noise for "is there any tremor at all". Both measured
effects follow from that, and so does the older note that averaging PADS
StretchHold with Relaxed cost ET precision (0.585 vs 0.612) -- Relaxed *is* the
rest condition.

The fix is to keep the two conditions apart and give the model their
**difference**. All three cohorts have both:

    cohort     postural            rest
    2015       OUT                 REST
    NewData    OUT                 REST
    PADS       StretchHold         Relaxed

Contrast features per patient, all rotation- and scale-invariant by construction
because they are ratios of normalised spectra:

  ``ratio_band``   log(postural power / rest power) in each of the 16 log-bins.
                   A 16-d contrast spectrum. PD should be negative in the tremor
                   band (more power at rest), ET positive.
  ``ratio_total``  log of the same ratio integrated over 3-15 Hz -- the single
                   number a clinician would form.
  ``peak_shift``   postural peak frequency minus rest peak frequency.
  ``peak_ratio``   log ratio of peak heights.

Arms on the reported model (multitaper + trajectory, soft-voted with ResidualTCN,
validation-tuned priors), changing only the inputs:

  postural only (reported)        the baseline
  + scalar contrasts              3 numbers appended to the descriptor block
  + contrast spectrum             the 16-d contrast appended to the descriptors
  contrast REPLACES descriptors   tests substitution rather than addition, since
                                  13 feature unions in this project have diluted

Note the standing rule: **prefer replacing a feature family over appending one**.
The last arm is the one the rule predicts should do best.

Run: ``python -m experiments.task_contrast``
"""

from __future__ import annotations

import os
import re
from dataclasses import replace

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedShuffleSplit

from common.cohorts import logbin
from common.protocol import TEST_FRAC, VAL_FRAC, train, tune_offsets
from experiments.alltasks_final import evaluate, paired
from experiments.final_model import GRID, NBIN, SPLITS, build, method_table
from frequency.tables import spectrum_table

NM = ("precN", "precPD", "precET", "macroP", "macroF1")
_ACTION = re.compile(r"_(OUT|REST|WING)$")


def norm(recs):
    return [replace(r, subject=_ACTION.sub("", str(r.subject))) for r in recs]


def pairs():
    """Per cohort: (postural recs, rest recs, channel slice)."""
    from common.load_2025 import load_2025_all
    from common.loaders import load_pads_extracted
    from common.quaternion_data import load_quaternion_recordings

    out = [(norm(load_quaternion_recordings("Data", action="OUT",
                                            mode="angular_velocity")),
            norm(load_quaternion_recordings("Data", action="REST",
                                            mode="angular_velocity")),
            slice(3, 6)),
           (norm(load_2025_all(conditions=("OUT",))),
            norm(load_2025_all(conditions=("REST",))),
            slice(3, 6))]
    rest_pads = (load_pads_extracted("pads_relaxed")
                 if os.path.isdir("pads_relaxed") else [])
    out.append((norm(load_pads_extracted("pads_stretchhold")),
                norm(rest_pads), slice(0, 3)))
    return out


def contrast_block(order):
    """(16-d band contrast, 3 scalar contrasts) per patient, on `order`.

    Patients with no rest recording get zeros plus an availability flag, the
    same missing-modality convention the asymmetry block already uses -- a zero
    contrast must not be readable as "measured and equal".
    """
    band, scal, have = {}, {}, {}
    for post, rest, ch in pairs():
        Xp, _, pp = method_table(post, "multitaper", ch)
        ip = {p: i for i, p in enumerate(pp)}
        if len(rest):
            Xr, _, pr = method_table(rest, "multitaper", ch)
            ir = {p: i for i, p in enumerate(pr)}
        else:
            ir = {}
        for p, i in ip.items():
            vp = Xp[i]
            if p not in ir:
                continue
            vr = Xr[ir[p]]
            # logbin already takes the log, so contrast the BINNED log-spectra
            bp = logbin(vp[None, :], NBIN)[0]
            br = logbin(vr[None, :], NBIN)[0]
            band[p] = bp - br
            kp, kr = int(np.argmax(vp)), int(np.argmax(vr))
            scal[p] = np.array([
                float(np.log(vp.sum() + 1e-12) - np.log(vr.sum() + 1e-12)),
                float(GRID[kp] - GRID[kr]),
                float(np.log(vp[kp] + 1e-12) - np.log(vr[kr] + 1e-12)),
            ])
            have[p] = 1.0

    B = np.array([band.get(p, np.zeros(NBIN)) for p in order])
    S = np.array([scal.get(p, np.zeros(3)) for p in order])
    H = np.array([have.get(p, 0.0) for p in order])[:, None]
    return np.nan_to_num(B), np.nan_to_num(S), H


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

    CB, CS, CH = contrast_block(order)
    print(f"n={len(y)}  N={int((y==0).sum())} PD={int((y==1).sum())} "
          f"ET={int((y==2).sum())}   {SPLITS} splits")
    print(f"rest recording available for {int(CH.sum())}/{len(y)} patients\n")

    # what the contrast says about the classes, before any model sees it
    print("mean scalar contrast (postural minus rest), by class:")
    print(f"{'':>6}{'log total power':>18}{'peak shift Hz':>16}"
          f"{'log peak ratio':>17}")
    for cl, nm in ((0, "N"), (1, "PD"), (2, "ET")):
        m = (y == cl) & (CH[:, 0] > 0)
        print(f"{nm:>6}" + "".join(f"{v:>18.3f}" if j == 0 else f"{v:>16.3f}"
                                   for j, v in enumerate(CS[m].mean(0))))
    print()

    ARMS = (("postural only (reported)", spec, D_post),
            ("+ scalar contrasts", spec, np.hstack([D_post, CS, CH])),
            ("+ contrast spectrum", spec, np.hstack([D_post, CB, CS, CH])),
            ("contrast REPLACES desc", spec,
             np.hstack([CB, CS, CH, d["ASYM"], d["HAVE"]])))

    res = {}
    print(f"{'arm':>28}" + "".join(f"{c:>9}" for c in NM) + "   sd(macroP)")
    for lab, S, D in ARMS:
        res[lab] = evaluate(S, D, traj, y, key)
        m = res[lab].mean(0)
        print(f"{lab:>28}" + "".join(f"{v:>9.3f}" for v in m)
              + f"{res[lab][:, 3].std():>12.3f}", flush=True)

    base = res["postural only (reported)"]
    print("\npaired vs postural only, same splits:")
    for lab, _, _ in ARMS[1:]:
        print(f"  {lab}:")
        for (dd, lo, hi), c in zip(paired(res[lab], base), NM):
            star = "*" if lo > 0 or hi < 0 else " "
            print(f"    {c:>8} {dd:+.3f}  [{lo:+.3f}, {hi:+.3f}] {star}")
    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    main()
