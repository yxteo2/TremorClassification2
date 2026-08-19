"""Does the 3-class merged deep model convert the recovered band?

`reports/band_truncation.md` shows the old `logbin` discarded 12.50-14.84 Hz
from 61-column welch input, and that covering it is worth precET +0.032
[+0.014, +0.050] on PADS PD-vs-ET **for a logistic regression**.

Re-running `python -m common.cohorts` after the fix moved the 3-class merged
numbers by less than their standard deviation:

    strategy                      old macroP   new macroP   sd
    cap 90 + global priors            0.649        0.642    0.064
    cap 90 + per-cohort priors        0.626        0.610    0.071
    cap 90 + cohort-ID input          0.668        0.654    0.063

That is not evidence of no effect. Those are two **unpaired** 10-split runs, and
precET there has sd 0.192 -- a difference of +0.03 is far inside the noise, so
the comparison has essentially no power. The question is still open and needs the
paired design the repo uses everywhere else.

This runs both binnings through the identical model on the **same splits**, so
the split-to-split variance that swamps the unpaired comparison cancels.

Run: ``python -m experiments.binning_deep``
"""

from __future__ import annotations

import numpy as np
import torch

from common.cohorts import load_all
from common.protocol import run

SPLITS = 20
NM = ("precN", "precPD", "precET", "macroP", "macroF1")


def logbin_truncating(X, nb=16):
    """`common.cohorts.logbin` as it was before the fix: drops the remainder."""
    L = np.log(X + 1e-8)
    n = X.shape[1] // nb * nb
    return L[:, :n].reshape(len(L), nb, -1).mean(2)


def logbin_full(X, nb=16):
    """The fixed version: every input column lands in some bin."""
    L = np.log(X + 1e-8)
    e = np.linspace(0, L.shape[1], nb + 1).round().astype(int)
    return np.stack([L[:, e[i]:e[i + 1]].mean(1) for i in range(nb)], 1)


def main():
    torch.set_num_threads(1)

    # load_all already applies the (now fixed) logbin, so rebuild the spectrum
    # block from the raw welch tables to get both binnings on the same patients.
    from common.loaders import load_pads_extracted
    from common.load_2025 import load_2025_all
    from common.quaternion_data import load_quaternion_recordings
    from frequency.tables import spectrum_table

    sb_fixed, dc, y, key, coh = load_all(cap=90)

    rA = load_quaternion_recordings("Data", action="OUT",
                                    mode="angular_velocity")
    rB = load_2025_all(conditions=("OUT",))
    rC = load_pads_extracted("pads_stretchhold")
    A = spectrum_table(rA, ch=slice(3, 6))
    B = spectrum_table(rB, ch=slice(3, 6))
    C = spectrum_table(rC, ch=slice(0, 3))

    rng = np.random.default_rng(0)
    keep = []
    for c in (0, 1, 2):
        i = np.flatnonzero(C[1] == c)
        keep.extend(rng.choice(i, min(90, len(i)), replace=False))
    keep = np.array(sorted(keep))
    raw = np.vstack([A[0], B[0], C[0][keep]])

    sb_trunc, sb_full = logbin_truncating(raw), logbin_full(raw)
    assert np.allclose(sb_full, sb_fixed), \
        "rebuilt spectrum block does not match load_all output"

    f = np.linspace(0, 50, 257)
    fb = f[(f >= 3.0) & (f <= 15.0)]
    n48 = raw.shape[1] // 16 * 16
    print(f"welch spectrum {raw.shape}, band {fb[0]:.2f}-{fb[-1]:.2f} Hz")
    print(f"  truncating: {fb[0]:.2f}-{fb[n48-1]:.2f} Hz "
          f"({100*(raw.shape[1]-n48)/raw.shape[1]:.0f} % discarded)")
    print(f"  full      : {fb[0]:.2f}-{fb[-1]:.2f} Hz")
    print(f"  n={len(y)}  ET={int((y == 2).sum())}   {SPLITS} shared splits\n")

    print(f"{'binning':>36}{'precN':>9}{'precPD':>9}{'precET':>9}"
          f"{'macroP':>9}{'macroF1':>9}")
    a = run("truncating (3.12-12.30 Hz)", sb_trunc, dc, y, key, coh,
            splits=SPLITS)
    b = run("full band (3.12-14.84 Hz)", sb_full, dc, y, key, coh,
            splits=SPLITS)

    print("\npaired, same splits (bootstrap 95 % CI):")
    d = b - a
    for i, nm in enumerate(NM):
        boot = [np.mean(np.random.default_rng(s).choice(d[:, i], len(d),
                                                        replace=True))
                for s in range(4000)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        star = "*" if lo > 0 or hi < 0 else " "
        print(f"  {nm:>8} {d[:, i].mean():+.3f}  [{lo:+.3f}, {hi:+.3f}] {star}"
              f"   (unpaired sd {a[:, i].std():.3f})")
    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    main()
