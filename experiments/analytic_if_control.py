"""Did median-centring the IF channel cause the analytic stream to fail?

`time_domain_deep.md` reports that the analytic TCN (log envelope + instantaneous
frequency) was the worst deep input tried — macroP −0.076 [−0.107, −0.047] * and
precET −0.192 * against the reported model — and *worse than the raw waveform*
(0.584 vs 0.626), which refuted the hypothesis it was built on (that a raw-waveform
TCN wastes capacity learning to demodulate).

That report then offers an explanation: the IF channel was centred on each
patient's own median **on purpose**, to avoid duplicating the absolute frequency
the spectrum stream already carries — and absolute tremor frequency is the single
most discriminative quantity available (max + mean frequency alone give AUC 0.786
on PADS). On that account the stream was deprived of the strongest signal by
construction.

**That explanation is untested, and this project has just been reminded what
untested explanations are worth**: the demodulation hypothesis behind the analytic
stream was wrong in the same experiment. So it gets a control rather than an
assertion.

Two arms, identical in every respect except one line of signal processing:

  B  IF centred on the patient's median   the published configuration
  C  IF left absolute                     the same stream with frequency restored

Envelopes are bit-identical between arms (verified), so the only difference is
whether channel 1 carries absolute frequency.

Prediction if the explanation holds: **C beats B by a wide margin**, roughly the
0.042 macroP that separates the analytic stream from the raw waveform, since the
raw waveform also carries absolute frequency. If C ≈ B, the explanation is wrong
and the analytic representation simply is not learnable at this n.

The reported model is not re-run here: the question is B vs C, and both are
already known to sit far below it. If C turns out competitive, the vote against
the reported model becomes worth testing separately.

Run: ``python -m experiments.analytic_if_control``
"""

from __future__ import annotations

import os

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedShuffleSplit

from common.protocol import TEST_FRAC, VAL_FRAC, train, tune_offsets
from experiments.alltasks_final import paired
from experiments.analytic_deep import AnalyticTCN, N_CH, N_REC
from experiments.final_model import SPLITS, build
from frequency.tables import spectrum_table
from signal_processing.waveform import LENGTH, patient_analytic

NM = ("precN", "precPD", "precET", "macroP", "macroF1")
SEEDS = (0, 1, 2)


def build_pack(order, centre_if):
    from common.load_2025 import load_2025_all
    from common.loaders import load_pads_extracted
    from common.quaternion_data import load_quaternion_recordings

    src = [(load_quaternion_recordings("Data", action="OUT",
                                       mode="angular_velocity"), slice(3, 6)),
           (load_2025_all(conditions=("OUT",)), slice(3, 6))]
    if os.path.isdir("pads_stretchhold"):
        src.append((load_pads_extracted("pads_stretchhold"), slice(0, 3)))
    packs, pats = [], []
    for recs, ch in src:
        X, M, _, p = patient_analytic(recs, ch=ch, n_rec=N_REC,
                                      centre_if=centre_if)
        packs.append(np.hstack([X.reshape(len(X), -1), M]))
        pats.append(p)
    allX, allp = np.vstack(packs), np.concatenate(pats)
    idx = {p: i for i, p in enumerate(allp)}
    D = allX.shape[1]
    return np.array([allX[idx[p]] if p in idx else np.zeros(D) for p in order],
                    dtype=np.float32)


def main():
    torch.set_num_threads(1)
    d = build()
    y, key = d["y"], d["key"]

    from common.load_2025 import load_2025_all
    from common.loaders import load_pads_extracted
    from common.quaternion_data import load_quaternion_recordings
    A = spectrum_table(load_quaternion_recordings("Data", action="OUT",
                                                  mode="angular_velocity"),
                       ch=slice(3, 6))
    B_ = spectrum_table(load_2025_all(conditions=("OUT",)), ch=slice(3, 6))
    C_ = spectrum_table(load_pads_extracted("pads_stretchhold"), ch=slice(0, 3))
    rng = np.random.default_rng(0)
    keep = []
    for cl in (0, 1, 2):
        i = np.flatnonzero(C_[1] == cl)
        keep.extend(rng.choice(i, min(90, len(i)), replace=False))
    keep = np.array(sorted(keep))
    order = np.concatenate([A[2], B_[2], C_[2][keep]])
    assert np.array_equal(np.concatenate([A[1], B_[1], C_[1][keep]]), y), \
        "patient order does not match build()"

    print("building both analytic packs ...", flush=True)
    Wc = build_pack(order, centre_if=True)
    Wa = build_pack(order, centre_if=False)

    k = N_REC * N_CH * LENGTH
    envc = Wc[:, :k].reshape(len(Wc), N_REC, N_CH, LENGTH)[:, :, 0, :]
    enva = Wa[:, :k].reshape(len(Wa), N_REC, N_CH, LENGTH)[:, :, 0, :]
    ifc = Wc[:, :k].reshape(len(Wc), N_REC, N_CH, LENGTH)[:, 0, 1, :]
    ifa = Wa[:, :k].reshape(len(Wa), N_REC, N_CH, LENGTH)[:, 0, 1, :]
    print(f"n={len(y)}   {SPLITS} splits")
    print(f"  envelopes identical across arms: {np.allclose(envc, enva)}")
    print(f"  IF channel  centred: mean {ifc.mean():+.3f} Hz   "
          f"absolute: mean {ifa.mean():+.3f} Hz")
    print(f"  absolute IF by class:", end="")
    for cl, nm in ((0, "N"), (1, "PD"), (2, "ET")):
        print(f"  {nm} {ifa[y == cl].mean():.2f}", end="")
    print("\n", flush=True)

    ARMS = (("B IF centred (published)", Wc), ("C IF absolute", Wa))
    res = {a: [] for a, _ in ARMS}

    for sp in range(SPLITS):
        tv, te = next(StratifiedShuffleSplit(1, test_size=TEST_FRAC,
                                             random_state=sp).split(Wc, key))
        t0, v0 = next(StratifiedShuffleSplit(1, test_size=VAL_FRAC,
                                             random_state=sp).split(Wc[tv],
                                                                    key[tv]))
        tr, va = tv[t0], tv[v0]
        for lab, W in ARMS:
            r = [train(lambda: AnalyticTCN(), W[tr], y[tr], W[va], y[va],
                       [W[va], W[te]], seed=s) for s in SEEDS]
            pv = np.mean([a[0] for a in r], 0)
            pt = np.mean([a[1] for a in r], 0)
            pred = (np.log(pt + 1e-12) + tune_offsets(pv, y[va])).argmax(1)
            P, _, F, _ = precision_recall_fscore_support(
                y[te], pred, labels=[0, 1, 2], zero_division=0)
            res[lab].append([P[0], P[1], P[2], P.mean(), F.mean()])
        print(f"  split {sp+1}/{SPLITS} done", flush=True)

    for a, _ in ARMS:
        res[a] = np.array(res[a])

    print(f"\n{'arm':>26}" + "".join(f"{c:>9}" for c in NM) + "   sd(macroP)")
    for a, _ in ARMS:
        print(f"{a:>26}" + "".join(f"{v:>9.3f}" for v in res[a].mean(0))
              + f"{res[a][:, 3].std():>12.3f}")
    print("\n  (reported model for reference: 0.639 0.655 0.685 0.660 0.593)")

    print("\npaired C - B, same splits — the test of the explanation:")
    for (dd, lo, hi), c in zip(paired(res["C IF absolute"],
                                      res["B IF centred (published)"]), NM):
        star = "*" if lo > 0 or hi < 0 else " "
        print(f"    {c:>8} {dd:+.3f}  [{lo:+.3f}, {hi:+.3f}] {star}")
    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    main()
