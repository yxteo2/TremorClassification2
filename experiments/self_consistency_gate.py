"""The gate: is the ceiling label noise, or is it signal insufficiency?

Two accounts fit every measurement in this project and imply **opposite**
spending (`data_plan.md` §0):

  label noise            the signal is there, the labels are wrong 15-35 % of
                         the time (clinical PD-vs-ET misdiagnosis, and ET has
                         no gold standard even post-mortem). Re-adjudicate the
                         404 patients we hold; more patients with equally noisy
                         labels buy proportionally less.
  signal insufficiency   the 3-15 Hz wrist spectrum simply does not separate PD
                         from ET in ~40 % of patients. Re-labelling changes
                         nothing; collect more and richer data.

## The discriminator

Every patient here has **more than one recording** — 2015 and NewData repeat the
same arm, PADS gives both wrists — and the pipeline averages them away before
the model sees anything. This asks what the model says about each recording
*separately*, using the reported model trained exactly as reported.

    label noise           -> the model is CONSISTENT with itself on a patient
                             and disagrees with the LABEL. Two recordings of a
                             mislabelled patient both say the same wrong thing.
    signal insufficiency  -> the model is INCONSISTENT on the same patient too.
                             It is guessing, and two recordings of one patient
                             disagree as readily as two random patients do.

The statistic is **self-agreement**: do a patient's recordings receive the same
predicted class? Reported separately for patients the model gets right and
patients it gets wrong, because the correct group is the internal reference —
it calibrates how much agreement this model produces when it *is* working.

## The control that makes it interpretable

Two recordings could agree merely because the model has a class prior and
predicts PD most of the time. So every agreement rate is reported against
**agreement between recordings of two DIFFERENT patients of the same true
class**, resampled per split. If within-patient agreement matches that, the
model is tracking class-level statistics and nothing about the individual
patient, and the "consistent" half of the label-noise signature is absent.

## Two kinds of repeat, kept apart

2015 and NewData repeat the **same arm** — that is test-retest reliability.
PADS's pair is **LeftWrist and RightWrist**, a different limb with a mirrored
mounting, so its disagreement includes genuine bilateral asymmetry and is not a
reliability measure. The orientation diagnostic was misread this way once
already; here the two are reported in separate rows and never pooled.

## Predictions, recorded before the run

1. **Neither account will win cleanly.** A_wrong sits clearly above the
   same-class control — there is stable patient-specific signal even where the
   model is wrong — but clearly below A_correct. Stated as the primary
   prediction because a clean win for either account is the less likely
   outcome, and saying so in advance stops a mixed result being narrated
   afterwards as whichever answer is convenient.
2. **PADS L/R self-agreement < 2015/NewData same-arm self-agreement**, because
   the PADS pair spans two limbs and tremor is genuinely asymmetric.
3. If forced to one side: the **label-noise** half will look stronger on the
   *confidence* statistic than on the agreement statistic — misclassified
   patients being confidently wrong is easier to produce than two recordings
   independently landing on the same wrong class.

## How to read the outcome

    A_wrong ~ A_correct, both >> control   -> LABEL NOISE. Fund adjudication.
    A_wrong ~ control                      -> SIGNAL INSUFFICIENCY. Fund
                                              collection and richer acquisition.
    in between                             -> both; the split between them is
                                              the ratio to spend on.

20 splits, checkpointed. Run: ``python -m experiments.self_consistency_gate``
"""

from __future__ import annotations

import dataclasses
from collections import defaultdict

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit

import experiments.final_model as FM
from common.cohorts import desc_table, logbin
from common.protocol import NBIN, TEST_FRAC, VAL_FRAC, train, tune_offsets
from experiments._resume import resume_load, resume_save
from experiments.final_model import TL, method_table
from models.architectures import TRUNKS, ResidualTCN, Spectrum1DCNN, TwoStreamNet
from signal_processing.stability import trajectory_table

SPLITS = 20
SEEDS = (0, 1, 2)
SEP = "##"


def explode(recs):
    """One pseudo-patient per recording, so the patient-level table builders
    return one row per recording in a recoverable order."""
    out, n = [], defaultdict(int)
    for r in recs:
        k = n[r.subject]
        n[r.subject] += 1
        out.append(dataclasses.replace(r, subject=f"{r.subject}{SEP}{k:02d}"))
    return sorted(out, key=lambda r: r.subject)


def rec_tables(recs, ch):
    """(spec, desc, traj, parent_subject) per recording, sorted by subject."""
    ex = explode(recs)
    spec = method_table(ex, "multitaper", ch)[0]
    desc = desc_table(ex, ch)
    traj = trajectory_table(ex, ch=ch, n_out=TL)[0]
    parent = np.array([r.subject.split(SEP)[0] for r in ex])
    return spec, desc, traj.reshape(len(traj), -1), parent


def fit_and_predict(spec, desc, traj, y, tr, va, te, Rspec, Rdesc, Rtraj):
    """The reported six members, additionally scoring per-RECORDING rows.

    Training, normalisation and seeds are identical to `pooling_rules.fit_members`
    — the recording rows are simply a third output matrix, standardised with the
    same train-fold statistics, so this measures the reported model rather than
    a new one.
    """
    nd = desc.shape[1]
    packed = np.hstack([spec, desc, traj])
    Rpacked = np.hstack([Rspec, Rdesc, Rtraj])
    mk1 = lambda: TwoStreamNet(Spectrum1DCNN(NBIN, 3, ch=8), TRUNKS["cnn"],
                               8 * 2 * 4, NBIN, nd, TL)
    mk2 = lambda: ResidualTCN(NBIN, num_classes=3, ch=16)
    V, T, R = [], [], []
    for X, XR, mk in ((packed, Rpacked, mk1), (spec, Rspec, mk2)):
        mu = X[tr].mean(0, keepdims=True)
        sd = X[tr].std(0, keepdims=True) + 1e-8
        for s in SEEDS:
            pv, pt, pr = train(mk, (X[tr] - mu) / sd, y[tr], (X[va] - mu) / sd,
                               y[va], [(X[va] - mu) / sd, (X[te] - mu) / sd,
                                       (XR - mu) / sd], seed=s)
            V.append(pv); T.append(pt); R.append(pr)
    return np.stack(V).mean(0), np.stack(T).mean(0), np.stack(R).mean(0)


def main():
    torch.set_num_threads(1)
    d = FM.build()
    y, key = d["y"], d["key"]
    SPEC = d["SPEC"]["multitaper"]
    A = np.hstack([d["ASYM"], d["HAVE"]])
    D = np.hstack([d["DESC"], A])
    TR = d["TRAJ"]

    from common.loaders import load_pads_extracted
    from common.load_2025 import load_2025_all
    from common.quaternion_data import load_quaternion_recordings
    from frequency.tables import spectrum_table

    rA = load_quaternion_recordings("Data", action="OUT",
                                    mode="angular_velocity")
    rB = load_2025_all(conditions=("OUT",))
    rC = load_pads_extracted("pads_stretchhold")
    C0 = spectrum_table(rC, ch=slice(0, 3))
    rng = np.random.default_rng(0)
    keep = []
    for c in (0, 1, 2):
        i = np.flatnonzero(C0[1] == c)
        keep.extend(rng.choice(i, min(90, len(i)), replace=False))
    kept_pads = set(C0[2][np.array(sorted(keep))].tolist())
    rC = [r for r in rC if r.subject in kept_pads]

    # per-recording tables, cohort by cohort, stacked in build()'s patient order
    parts = [rec_tables(rA, slice(3, 6)), rec_tables(rB, slice(3, 6)),
             rec_tables(rC, slice(0, 3))]
    Rspec = logbin(np.vstack([p[0] for p in parts]))
    Rdesc = np.vstack([p[1] for p in parts])
    Rtraj = np.vstack([p[2] for p in parts])
    parent = np.concatenate([p[3] for p in parts])
    Rcoh = np.concatenate([np.full(len(p[3]), c) for p, c in
                           zip(parts, ("2015", "NewData", "PADS"))])

    # patient row order from build(): 2015, NewData, PADS(kept), each sorted
    pat_ids = np.concatenate([
        np.array(sorted({r.subject for r in rA})),
        np.array(sorted({r.subject for r in rB})),
        np.array(sorted({r.subject for r in rC}))])
    assert len(pat_ids) == len(y), f"{len(pat_ids)} ids vs {len(y)} patients"
    idx_of = {p: i for i, p in enumerate(pat_ids)}
    rec2pat = np.array([idx_of[p] for p in parent])
    # ASYM/HAVE are patient properties; broadcast them onto the patient's rows
    Rdesc = np.hstack([Rdesc, A[rec2pat]])
    assert Rdesc.shape[1] == D.shape[1], "recording desc width != patient"

    n_rec = np.bincount(rec2pat, minlength=len(y))
    print(f"n={len(y)} patients, {len(rec2pat)} recordings; "
          f"{int((n_rec >= 2).sum())} patients have >=2\n", flush=True)

    ARMS = ("stats",)
    res, done = resume_load("self_consistency_gate", ARMS)
    for sp in range(SPLITS):
        if sp in done:
            continue
        tv, te = next(StratifiedShuffleSplit(1, test_size=TEST_FRAC,
                                             random_state=sp).split(y, key))
        t0, v0 = next(StratifiedShuffleSplit(1, test_size=VAL_FRAC,
                                             random_state=sp).split(y[tv],
                                                                    key[tv]))
        tr, va = tv[t0], tv[v0]
        pv, pt, pr = fit_and_predict(SPEC, D, TR, y, tr, va, te,
                                     Rspec, Rdesc, Rtraj)
        off = tune_offsets(pv, y[va])
        pat_pred = (np.log(pt + 1e-12) + off).argmax(1)
        rec_pred = (np.log(pr + 1e-12) + off).argmax(1)
        rec_conf = pr.max(1)

        te_set = {int(i) for i in te}
        pos_of = {int(p): j for j, p in enumerate(te)}
        by_pat = defaultdict(list)
        for k, p in enumerate(rec2pat):
            if int(p) in te_set:
                by_pat[int(p)].append(k)

        row = {}
        for tag, cohs in (("same-arm", ("2015", "NewData")), ("PADS", ("PADS",))):
            agree_c, agree_w, conf_c, conf_w = [], [], [], []
            pool = []                      # (true class, predicted class) pairs
            for p, ks in by_pat.items():
                ks = [k for k in ks if Rcoh[k] in cohs]
                if len(ks) < 2:
                    continue
                same = len({int(rec_pred[k]) for k in ks}) == 1
                correct = int(pat_pred[pos_of[p]]) == int(y[p])
                (agree_c if correct else agree_w).append(float(same))
                (conf_c if correct else conf_w).append(
                    float(np.mean([rec_conf[k] for k in ks])))
                for k in ks:
                    pool.append((int(y[p]), int(rec_pred[k]), p))
            # control: two recordings from DIFFERENT patients of the SAME class
            ctrl = []
            g = np.random.default_rng(5000 + sp)
            by_cls = defaultdict(list)
            for cls, prd, p in pool:
                by_cls[cls].append((prd, p))
            for cls, v in by_cls.items():
                if len(v) < 4:
                    continue
                for _ in range(400):
                    (a, pa), (b, pb) = (v[g.integers(len(v))],
                                        v[g.integers(len(v))])
                    if pa != pb:
                        ctrl.append(float(a == b))
            row[tag] = [np.mean(agree_c) if agree_c else np.nan,
                        np.mean(agree_w) if agree_w else np.nan,
                        np.mean(ctrl) if ctrl else np.nan,
                        np.mean(conf_c) if conf_c else np.nan,
                        np.mean(conf_w) if conf_w else np.nan,
                        len(agree_c), len(agree_w)]
        # flat row of 14: same-arm block then PADS block, 7 each.
        # resume_save stores rows of floats, not nested lists.
        res["stats"].append(list(row["same-arm"]) + list(row["PADS"]))
        resume_save("self_consistency_gate", res, sp)
        print(f"  split {sp + 1}/{SPLITS} done", flush=True)

    S = np.array(res["stats"], dtype=float).reshape(-1, 2, 7)
    print(f"\n{'repeat kind':>12}{'A_correct':>11}{'A_wrong':>10}"
          f"{'control':>10}{'conf_cor':>10}{'conf_wrong':>12}"
          f"{'n_cor':>8}{'n_wrong':>9}")
    for j, tag in enumerate(("same-arm", "PADS L/R")):
        m = np.nanmean(S[:, j, :], 0)
        print(f"{tag:>12}{m[0]:>11.3f}{m[1]:>10.3f}{m[2]:>10.3f}"
              f"{m[3]:>10.3f}{m[4]:>12.3f}{m[5]:>8.1f}{m[6]:>9.1f}")

    print("\nverdict inputs (same-arm rows are the reliability ones):")
    for j, tag in enumerate(("same-arm", "PADS L/R")):
        aw = np.nanmean(S[:, j, 1]); ac = np.nanmean(S[:, j, 0])
        ct = np.nanmean(S[:, j, 2])
        d_ctrl, d_cor = aw - ct, ac - aw
        print(f"  {tag}: A_wrong - control = {d_ctrl:+.3f}   "
              f"A_correct - A_wrong = {d_cor:+.3f}")
    print("\n  A_wrong >> control and ~ A_correct  -> LABEL NOISE")
    print("  A_wrong ~ control                   -> SIGNAL INSUFFICIENCY")
    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    main()
