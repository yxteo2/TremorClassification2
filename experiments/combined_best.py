"""Do the two winners beat the REPORTED model, paired?

Two changes measured positive in isolation this round:

  early input channels   descriptors broadcast along frequency and fed as extra
                         input channels instead of joining at the classifier.
                         macroP +0.036 [+0.004, +0.071] * against late concat
                         (`tcn_fusion.py`).
  log-frequency bins     16 bins equally spaced in log f instead of linear f, so
                         harmonics sit a fixed distance apart for every patient.
                         precN +0.031 * , macroP +0.019 [-0.005, +0.043]
                         (`spectral_representation.py`).

Neither has yet been compared against the **reported** model. Both were measured
against baselines internal to their own experiment, and in the fusion study that
baseline was a re-implementation: `FusionTCN` in "late" mode uses a residual
dilated trunk, while the reported model's spectrum stream is `Spectrum1DCNN`.
Its late-concat arm scored macroP 0.645 where the reported model scores 0.660, so
**the fusion gain was measured from a lower starting point** and could be
recovering ground the reported architecture already holds.

This settles it on the same splits:

  A. reported model                      TwoStreamNet + ResidualTCN, linear bins
  B. reported model, LOG-freq bins       the binning change alone, on the
                                         reported architecture
  C. early channels, linear bins         the fusion change alone
  D. early channels, LOG-freq bins       both

Everything is paired against A, and C is additionally paired against B so the two
changes can be read separately rather than only in combination.

Run: ``python -m experiments.combined_best``
"""

from __future__ import annotations

import numpy as np
import torch

from common.cohorts import logbin
from experiments.alltasks_final import evaluate as eval_reported
from experiments.alltasks_final import paired
from experiments.final_model import NBIN, SPLITS, build
from experiments.spectral_representation import logfreq_bin
from experiments.tcn_fusion import evaluate as eval_fusion

NM = ("precN", "precPD", "precET", "macroP", "macroF1")


def main():
    torch.set_num_threads(1)
    d = build()
    y, key = d["y"], d["key"]
    D = np.hstack([d["DESC"], d["ASYM"], d["HAVE"]])
    traj = d["TRAJ"]

    # The reported SPEC entry is already logbin'd. Rebuild both binnings from the
    # same un-binned multitaper table so the two differ in binning alone.
    from experiments.final_model import method_table
    from frequency.tables import spectrum_table
    from common.load_2025 import load_2025_all
    from common.loaders import load_pads_extracted
    from common.quaternion_data import load_quaternion_recordings

    rA = load_quaternion_recordings("Data", action="OUT",
                                    mode="angular_velocity")
    rB = load_2025_all(conditions=("OUT",))
    rC = load_pads_extracted("pads_stretchhold")
    C0 = spectrum_table(rC, ch=slice(0, 3))
    rng = np.random.default_rng(0)
    keep = []
    for cl in (0, 1, 2):
        i = np.flatnonzero(C0[1] == cl)
        keep.extend(rng.choice(i, min(90, len(i)), replace=False))
    keep = np.array(sorted(keep))
    raw = np.vstack([method_table(rA, "multitaper", slice(3, 6))[0],
                     method_table(rB, "multitaper", slice(3, 6))[0],
                     method_table(rC, "multitaper", slice(0, 3))[0][keep]])
    assert len(raw) == len(y), f"{len(raw)} vs {len(y)}"

    S_lin, S_log = logbin(raw), logfreq_bin(raw)
    assert np.allclose(S_lin, d["SPEC"]["multitaper"]), \
        "rebuilt linear binning does not match the reported SPEC table"
    print(f"n={len(y)}  N={int((y==0).sum())} PD={int((y==1).sum())} "
          f"ET={int((y==2).sum())}   {SPLITS} splits")
    print(f"spectrum {S_lin.shape[1]} bins, {D.shape[1]} descriptors\n",
          flush=True)

    ARMS = (("A reported model (linear bins)", "rep", S_lin),
            ("B reported model, LOG bins", "rep", S_log),
            ("C early channels, linear bins", "early", S_lin),
            ("D early channels, LOG bins", "early", S_log))

    res = {}
    print(f"{'arm':>32}" + "".join(f"{c:>9}" for c in NM) + "   sd(macroP)")
    for lab, kind, S in ARMS:
        S = np.nan_to_num(S)
        res[lab] = (eval_reported(S, D, traj, y, key) if kind == "rep"
                    else eval_fusion("early", S, D, traj, y, key))
        m = res[lab].mean(0)
        print(f"{lab:>32}" + "".join(f"{v:>9.3f}" for v in m)
              + f"{res[lab][:, 3].std():>12.3f}", flush=True)

    base = res["A reported model (linear bins)"]
    print("\npaired vs A, the reported model:")
    for lab, _, _ in ARMS[1:]:
        print(f"  {lab}:")
        for (dd, lo, hi), c in zip(paired(res[lab], base), NM):
            star = "*" if lo > 0 or hi < 0 else " "
            print(f"    {c:>8} {dd:+.3f}  [{lo:+.3f}, {hi:+.3f}] {star}")

    print("\npaired D vs B (fusion change, holding LOG bins fixed):")
    for (dd, lo, hi), c in zip(paired(res["D early channels, LOG bins"],
                                      res["B reported model, LOG bins"]), NM):
        star = "*" if lo > 0 or hi < 0 else " "
        print(f"    {c:>8} {dd:+.3f}  [{lo:+.3f}, {hi:+.3f}] {star}")

    print("\npaired D vs C (binning change, holding early fusion fixed):")
    for (dd, lo, hi), c in zip(paired(res["D early channels, LOG bins"],
                                      res["C early channels, linear bins"]), NM):
        star = "*" if lo > 0 or hi < 0 else " "
        print(f"    {c:>8} {dd:+.3f}  [{lo:+.3f}, {hi:+.3f}] {star}")
    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    main()
