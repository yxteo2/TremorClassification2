"""Give the model the cohort label, at proper power and with a matched control.

`cohort_strategies.md` tested four ways of combining 2015 / NewData / PADS and
concluded none beat the existing handling. Three arms were significantly worse.
**One was not** — feeding the cohort identity to the network as an input scored
precET 0.690 against the baseline's 0.639 and macroP 0.668 against 0.649, the
only positive row in that table:

    cohort-ID input   precET +0.051 [-0.040, +0.168]   macroP +0.019 [-0.020, +0.064]

It was reported as "not better", correctly, because neither interval clears zero.
But look at how it was measured: **10 splits, on the welch baseline**, not the
reported multitaper + trajectory model. Ten splits gives a precET interval ±0.10
wide here, which cannot resolve an effect of +0.05 either way. That arm was never
really tested.

## Why it is worth the retest now, which is new information

`ensemble_diversity.md` supplies a mechanism that did not exist when that table
was made. The patients the ensemble is unsure about are **not spread evenly
across cohorts**:

    contested rate    2015 0.307    PADS 0.432    NewData 0.573

and the obvious confound is controlled — the three cohorts have nearly identical
class composition, so all three *expect* ~0.40 from the per-class rates alone.
The in-house 2025 cohort is where the model is least sure, by a factor of 1.9
over 2015. If that is domain shift rather than under-fitting, a cohort indicator
is exactly the input that lets the network condition on it.

## The prediction, recorded before the run

**Adding cohort ID should reduce NewData's contested rate more than 2015's, and
any precision gain should concentrate in NewData.** If precision moves but the
contested rates do not, the mechanism proposed here is wrong even if the number
goes up.

This is deliberately the *measurement-derived* kind of prediction rather than the
mechanism-story kind. `failed_predictions.md` records that the first kind has a
much better track record in this project than the second.

## The control that decides it

A cohort one-hot is three extra input columns, and extra input capacity is itself
a change. The **random-ID** arm gives each patient a fixed, randomly assigned
3-level label and one-hots it identically. Same dimensionality, same everything,
no cohort information. If random-ID does as well, the gain is capacity and not
cohort knowledge.

## Two caveats that belong in any writeup of this

**This is a mixed-cohort result only, and can never support a transfer claim.**
A model given the cohort label cannot be applied to a cohort it has never seen.
It is legitimate for deployment — the recording site and device are always known
— but it is invisible to the leave-one-cohort-out question.

**The descriptors already leak partial cohort identity.** `HAVE` is 0 for every
2015 patient and 1 for NewData and PADS, because 2015 has no limb-asymmetry
modality. So the network can already separate 2015 from the rest; what this arm
adds is the NewData/PADS distinction and an explicit rather than incidental
encoding. The increment is what is measured.

**Learned conditioning is not per-cohort priors.** `cohort_strategies.md` found
fitting logit offsets *per cohort* significantly worse (macroP −0.023 *). That is
a different operation — post-hoc thresholds fitted on ~20 validation patients per
cohort — and its failure does not predict this one.

## Half the ensemble cannot see the cohort ID, so both halves are reported

The reported model is two families. `TwoStreamNet` takes the descriptor vector
and therefore receives the cohort one-hot; `ResidualTCN` takes the spectrum
alone, and appending three indicator columns to a spectrum would have them read
as extra frequency bins by a 1-D convolution over that axis — wrong, so it is not
done.

Three of six members are therefore unchanged by construction, which **dilutes any
effect by roughly half and would make a null ambiguous**: no gain could mean the
cohort ID does not help, or that it helps in a place the other half drowns out.
So the TwoStream family is also scored on its own, with and without the ID. That
arm is where the input actually arrives, and it is the one that tests the
mechanism cleanly. The full-ensemble arm answers the different, practical
question of whether it improves *the reported model*.

20 splits, paired, reported model. Run: ``python -m experiments.cohort_id_input``
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedShuffleSplit

from common.protocol import TEST_FRAC, VAL_FRAC, tune_offsets
from experiments.alltasks_final import paired
from experiments.final_model import build
from experiments.pooling_rules import fit_members

NM = ("precN", "precPD", "precET", "macroP", "macroF1")
SPLITS = 20
COHORTS = ("2015", "NewData", "PADS")


def onehot(idx, k=3):
    Z = np.zeros((len(idx), k))
    Z[np.arange(len(idx)), idx] = 1.0
    return Z


def score(pt, off, yte):
    pred = (np.log(pt + 1e-12) + off).argmax(1)
    P, _, F, _ = precision_recall_fscore_support(yte, pred, labels=[0, 1, 2],
                                                 zero_division=0)
    return [P[0], P[1], P[2], P.mean(), F.mean()]


def main():
    torch.set_num_threads(1)
    d = build()
    y, key = d["y"], d["key"]
    A = np.hstack([d["ASYM"], d["HAVE"]])
    desc, traj, spec = d["DESC"], d["TRAJ"], d["SPEC"]["multitaper"]
    coh = np.array([k.rsplit("_", 1)[0] for k in key])
    cid = np.array([COHORTS.index(c) for c in coh])

    rng = np.random.default_rng(7)
    rid = rng.integers(0, 3, len(y))          # fixed random 3-level label

    ARMS = {
        "baseline": np.hstack([desc, A]),
        "+ cohort ID": np.hstack([desc, A, onehot(cid)]),
        "+ random ID": np.hstack([desc, A, onehot(rid)]),
    }

    print(f"n={len(y)}  {SPLITS} splits   "
          + "  ".join(f"{c} {int((coh==c).sum())}" for c in COHORTS))
    print("prediction on record: cohort ID should cut NewData's contested rate")
    print("more than 2015's, and any precision gain should sit in NewData\n",
          flush=True)

    res = {a: [] for a in ARMS}
    solo = {a: [] for a in ARMS}          # TwoStream family alone
    con = {a: {c: [] for c in COHORTS} for a in ARMS}
    csolo = {a: {c: [] for c in COHORTS} for a in ARMS}
    pcoh = {a: {c: [] for c in COHORTS} for a in ARMS}

    for sp in range(SPLITS):
        tv, te = next(StratifiedShuffleSplit(1, test_size=TEST_FRAC,
                                             random_state=sp).split(spec, key))
        t0, v0 = next(StratifiedShuffleSplit(1, test_size=VAL_FRAC,
                                             random_state=sp).split(spec[tv],
                                                                    key[tv]))
        tr, va = tv[t0], tv[v0]
        line = []
        for a, D in ARMS.items():
            V, T = fit_members(spec, D, traj, y, tr, va, te)
            pv, pt = V.mean(0), T.mean(0)
            off = tune_offsets(pv, y[va])
            res[a].append(score(pt, off, y[te]))
            arg = np.stack([T[i].argmax(1) for i in range(len(T))])
            unan = (arg == arg[0]).all(0)
            pred = (np.log(pt + 1e-12) + off).argmax(1)

            # the TwoStream family alone -- the half that actually receives the
            # cohort one-hot. fit_members returns family 1 first, then family 2.
            nf = len(T) // 2
            sv, st = V[:nf].mean(0), T[:nf].mean(0)
            solo[a].append(score(st, tune_offsets(sv, y[va]), y[te]))
            sarg = np.stack([T[i].argmax(1) for i in range(nf)])
            sunan = (sarg == sarg[0]).all(0)

            for c in COHORTS:
                m = coh[te] == c
                con[a][c].append(float((~unan)[m].mean()) if m.any() else np.nan)
                csolo[a][c].append(float((~sunan)[m].mean()) if m.any()
                                   else np.nan)
                pcoh[a][c].append(float((pred[m] == y[te][m]).mean())
                                  if m.any() else np.nan)
            line.append(f"{a} con2015 {con[a]['2015'][-1]:.2f} "
                        f"newd {con[a]['NewData'][-1]:.2f}")
        print(f"  split {sp+1}/{SPLITS}  " + " | ".join(line), flush=True)

    for a in res:
        res[a] = np.array(res[a])

    print(f"\n{'arm':>14}" + "".join(f"{c:>9}" for c in NM) + "   sd(macroP)")
    for a in ARMS:
        print(f"{a:>14}" + "".join(f"{v:>9.3f}" for v in res[a].mean(0))
              + f"{res[a][:, 3].std():>12.3f}")

    base = res["baseline"]
    print("\npaired vs baseline:")
    for a in ("+ cohort ID", "+ random ID"):
        print(f"  {a}:")
        for (dd, lo, hi), c in zip(paired(res[a], base), NM):
            star = "*" if lo > 0 or hi < 0 else " "
            print(f"    {c:>8} {dd:+.3f}  [{lo:+.3f}, {hi:+.3f}] {star}")

    print("\nTHE CONTROL -- cohort ID vs random ID at the same input width:")
    for (dd, lo, hi), c in zip(paired(res["+ cohort ID"], res["+ random ID"]),
                               NM):
        star = "*" if lo > 0 or hi < 0 else " "
        print(f"    {c:>8} {dd:+.3f}  [{lo:+.3f}, {hi:+.3f}] {star}")

    for a in solo:
        solo[a] = np.array(solo[a])
    print("\nTwoStream family ALONE -- the half that receives the cohort ID.")
    print("A null in the full ensemble is ambiguous; this arm is not:")
    print(f"{'arm':>14}" + "".join(f"{c:>9}" for c in NM))
    for a in ARMS:
        print(f"{a:>14}" + "".join(f"{v:>9.3f}" for v in solo[a].mean(0)))
    for a in ("+ cohort ID", "+ random ID"):
        print(f"  {a} vs baseline, TwoStream only:")
        for (dd, lo, hi), c in zip(paired(solo[a], solo["baseline"]), NM):
            star = "*" if lo > 0 or hi < 0 else " "
            print(f"    {c:>8} {dd:+.3f}  [{lo:+.3f}, {hi:+.3f}] {star}")

    print("\nTHE PREDICTION -- contested rate per cohort, and its change:")
    print(f"{'cohort':>9}{'baseline':>11}{'+cohortID':>11}{'change':>9}"
          f"{'acc base':>10}{'acc +ID':>9}{'solo base':>11}{'solo +ID':>10}")
    for c in COHORTS:
        b = np.nanmean(con["baseline"][c])
        k = np.nanmean(con["+ cohort ID"][c])
        ab = np.nanmean(pcoh["baseline"][c])
        ak = np.nanmean(pcoh["+ cohort ID"][c])
        sb = np.nanmean(csolo["baseline"][c])
        sk = np.nanmean(csolo["+ cohort ID"][c])
        print(f"{c:>9}{b:>11.3f}{k:>11.3f}{k-b:>+9.3f}{ab:>10.3f}{ak:>9.3f}"
              f"{sb:>11.3f}{sk:>10.3f}")
    print("\nthe prediction holds only if NewData's contested rate falls by")
    print("more than 2015's. Precision moving without that is the mechanism")
    print("being wrong even if the number goes up.")
    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    main()
