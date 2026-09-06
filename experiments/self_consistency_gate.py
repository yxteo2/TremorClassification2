"""Descriptive recording agreement under the reported patient-level model.

Within-patient and between-patient statistics both use pairwise agreement.
Controls match true-label class, cohort and patient-prediction correctness;
patients have equal weights regardless of their recording count. Unmatched
strata are excluded from both matched columns and their counts are reported.
Same-arm repeats and PADS left/right comparisons remain separate.

Agreement cannot distinguish incorrect labels from systematic model errors,
atypical presentations or insufficient features. It does not identify a
label-noise fraction, upper bound, or allocation of research spending.

Run: ``python -m experiments.self_consistency_gate``. Old all-recording
agreement checkpoints are incompatible and use a different checkpoint name.
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
from experiments._agreement import agreement_summary
from experiments.final_model import TL, method_table
from models.architectures import TRUNKS, ResidualTCN, Spectrum1DCNN, TwoStreamNet
from signal_processing.stability import trajectory_table

SPLITS = 20
SEEDS = (0, 1, 2)
SEP = "##"
CHECKPOINT = "self_consistency_pairwise_v2"


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
    pat_coh = np.array([k.rsplit("_", 1)[0] for k in key])
    identities = list(zip(pat_coh, pat_ids))
    assert len(set(identities)) == len(y), "duplicate cohort/patient identity"
    assert np.array_equal(d["patient_ids"], np.array(
        [f"{c}::{p}" for c, p in identities])), "patient order mismatch"
    idx_of = {identity: i for i, identity in enumerate(identities)}
    rec2pat = np.array([idx_of[(c, p)] for c, p in zip(Rcoh, parent)])
    # ASYM/HAVE are patient properties; broadcast them onto the patient's rows
    Rdesc = np.hstack([Rdesc, A[rec2pat]])
    assert Rdesc.shape[1] == D.shape[1], "recording desc width != patient"

    n_rec = np.bincount(rec2pat, minlength=len(y))
    print(f"n={len(y)} patients, {len(rec2pat)} recordings; "
          f"{int((n_rec >= 2).sum())} patients have >=2\n", flush=True)

    ARMS = ("stats",)
    res, done = resume_load(CHECKPOINT, ARMS)
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
            patients = []
            for p, ks in by_pat.items():
                ks = [k for k in ks if Rcoh[k] in cohs]
                if len(ks) < 2:
                    continue
                patients.append(dict(
                    predictions=rec_pred[ks], cohort=str(Rcoh[ks[0]]),
                    label=int(y[p]),
                    correct=int(pat_pred[pos_of[p]]) == int(y[p]),
                    confidence=float(rec_conf[ks].mean())))
            summary = agreement_summary(patients)
            fields = ("agreement", "matched_agreement", "control",
                      "confidence", "n", "n_matched")
            row[tag] = [summary[c][f] for c in (True, False) for f in fields]
        res["stats"].append(row["same-arm"] + row["PADS"])
        resume_save(CHECKPOINT, res, sp)
        print(f"  split {sp + 1}/{SPLITS} done", flush=True)

    S = np.array(res["stats"], dtype=float).reshape(-1, 2, 2, 6)
    print("\nPairwise agreement, equal patient weights; means across splits")
    print("repeat / subgroup: all_A matched_A control confidence n n_matched")
    for j, tag in enumerate(("same-arm", "PADS L/R")):
        for k, label in enumerate(("correct", "wrong")):
            values = S[:, j, k, :]
            m = np.array([np.mean(v[np.isfinite(v)]) if np.isfinite(v).any()
                          else np.nan for v in values.T])
            print(f"{tag} / {label}: " + " ".join(f"{v:.3f}" for v in m))
            print(f"  matched_A - matched control = {m[1] - m[2]:+.3f}")
    print("\nDescriptive agreement only: no label-noise estimate or bound.")
    print("Missing matched strata are excluded from BOTH matched columns.")
    print("Confidence is unadjusted max probability, not calibrated correctness.")
    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    main()

