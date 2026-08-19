"""Two changes to how the spectrum is built, both with a mechanism.

Everything in this project feeds the models a spectrum built the same way:
average the three gyroscope axes' power spectra, resample onto a **linear**
3-15 Hz grid, sum-normalise, then coarse-bin. Two of those steps are choices
that were never tested, and both have a physical argument against them.

**1. Averaging the axes dilutes a linear oscillation.**

``spectrum_table`` and ``method_table`` both do ``P.mean(0)`` over the three
gyroscope channels. Tremor is close to a *linear* oscillation -- this repo
measured linearity 0.997 on real recordings (`four_families.md`) -- so the
oscillation lives on roughly one spatial axis while the other two carry mostly
noise. Averaging three axes when one carries the signal costs a factor of ~3 in
power SNR.

The rotation-invariant alternative is the **largest eigenvalue of the
cross-spectral matrix at each frequency**. Form

    S(f) = mean over STFT segments of  X(f) X(f)^H       (3x3, Hermitian)

then take lambda_1(f) instead of trace(S(f))/3, which is exactly the current
axis mean. lambda_1(f) is the power along the dominant oscillation direction at
that frequency. Under a rotation R, S -> R S R^T has identical eigenvalues, so
this is as rotation-invariant as the mean is -- it just does not throw the
signal into the noise axes.

**This only differs from the mean if S is averaged over multiple segments.** A
single FFT frame gives S = x x^H, which is rank 1, so lambda_1 = trace exactly
and the two are identical. Welch-style segment averaging is what makes the
eigenvalue meaningful, and it is used here.

Also computed: ``lambda_1 / trace`` per frequency, the degree of linear
polarisation as a function of frequency -- a pure shape quantity, unaffected by
amplitude.

**2. A linear frequency axis hides harmonic structure from a convolution.**

`ResidualTCN` convolves along the frequency axis with dilations (1, 2, 4). On a
**linear** axis a tremor's harmonics sit at f, 2f, 3f -- distances that change
with the fundamental, so no fixed kernel can match them across patients whose
tremor frequencies differ. On a **log** axis the same harmonics sit at fixed
offsets (log 2f - log f = log 2 for everyone), so one kernel detects harmonic
structure for every patient, and convolution becomes equivariant to a *scaling*
of frequency rather than a shift.

Harmonic structure is real here: it beats a permutation null on PADS at AUC
0.726 (`permutation_null.md`). Log-frequency binning also spends resolution
where the classes actually sit -- PD 4-6 Hz, ET 4-12 Hz -- instead of spreading
16 uniform bins across 3-15 Hz.

Arms: a 2x2 of {axis mean, principal eigenvalue} x {linear bins, log bins}, plus
the polarisation spectrum as a replacement, on the reported model.

Run: ``python -m experiments.spectral_representation``
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.signal import stft

from common.cohorts import logbin
from experiments.alltasks_final import evaluate, paired
from experiments.final_model import GRID, NBIN, SPLITS, build
from frequency.tables import spectrum_table

NM = ("precN", "precPD", "precET", "macroP", "macroF1")
F_LO, F_HI, FS = 3.0, 15.0, 100.0


# --------------------------------------------------------------------------- #
# Cross-spectral matrix per frequency
# --------------------------------------------------------------------------- #
def cross_spectra(x, fs=FS, nperseg=256):
    """(freqs, trace/3, lambda_1, lambda_1/trace) for a (3, T) recording.

    ``trace/3`` reproduces the current axis-averaged spectrum exactly, so the
    arms differ in one step only.
    """
    x = np.atleast_2d(np.asarray(x, float))
    if x.shape[0] < 3:                       # pad degenerate inputs
        x = np.vstack([x] + [x[-1:]] * (3 - x.shape[0]))
    x = x[:3] - x[:3].mean(1, keepdims=True)
    n = min(nperseg, x.shape[-1])
    f, _, Z = stft(x, fs=fs, nperseg=n, noverlap=n // 2, axis=-1)
    # Z: (3, F, T). Average the outer product over time segments.
    S = np.einsum("aft,bft->fab", Z, Z.conj()) / Z.shape[-1]
    S = np.real(S + S.conj().transpose(0, 2, 1)) / 2.0     # Hermitian part
    w = np.linalg.eigvalsh(S)                              # ascending
    tr = np.clip(np.trace(S, axis1=1, axis2=2), 1e-30, None)
    lam1 = np.clip(w[:, -1], 0, None)
    return f, tr / 3.0, lam1, lam1 / tr


def patient_table(recs, ch, kind):
    """(patients, len(GRID)) on the shared grid, sum-normalised, per patient."""
    from collections import defaultdict
    rows, lab = defaultdict(list), {}
    for r in recs:
        x = r.x[ch] if r.x.shape[0] > 3 else r.x
        f, mean_p, lam1, pol = cross_spectra(x)
        v = {"mean": mean_p, "lam1": lam1, "pol": pol}[kind]
        m = np.isfinite(v)
        g = np.clip(np.interp(GRID, f[m], v[m], left=0.0, right=0.0), 0, None)
        s = g.sum()
        if s <= 0:
            continue
        rows[r.subject].append(g / s)
        lab[r.subject] = r.y
    pats = sorted(rows)
    return (np.nan_to_num(np.array([np.mean(rows[p], 0) for p in pats])),
            np.array([lab[p] for p in pats]), np.array(pats))


# --------------------------------------------------------------------------- #
# Frequency-axis binning
# --------------------------------------------------------------------------- #
def logfreq_bin(X, nb=NBIN, grid=None):
    """Bin log-power into nb bins equally spaced in LOG frequency.

    Harmonics become fixed offsets, so a convolution kernel can match them
    regardless of the fundamental.
    """
    g = GRID if grid is None else grid
    L = np.log(X + 1e-8)
    edges = np.exp(np.linspace(np.log(g[0]), np.log(g[-1]), nb + 1))
    out = np.zeros((len(L), nb))
    for i in range(nb):
        m = (g >= edges[i]) & (g <= edges[i + 1] if i == nb - 1
                               else g < edges[i + 1])
        out[:, i] = L[:, m].mean(1) if m.any() else L[:, 0]
    return out


def main():
    torch.set_num_threads(1)
    from common.load_2025 import load_2025_all
    from common.loaders import load_pads_extracted
    from common.quaternion_data import load_quaternion_recordings

    d = build()
    y, key = d["y"], d["key"]
    D = np.hstack([d["DESC"], d["ASYM"], d["HAVE"]])
    traj = d["TRAJ"]

    cohorts = [(load_quaternion_recordings("Data", action="OUT",
                                           mode="angular_velocity"),
                slice(3, 6)),
               (load_2025_all(conditions=("OUT",)), slice(3, 6)),
               (load_pads_extracted("pads_stretchhold"), slice(0, 3))]

    # PADS cap, reproducing build()'s selection exactly
    C = spectrum_table(cohorts[2][0], ch=cohorts[2][1])
    rng = np.random.default_rng(0)
    keep = []
    for cl in (0, 1, 2):
        i = np.flatnonzero(C[1] == cl)
        keep.extend(rng.choice(i, min(90, len(i)), replace=False))
    keep = np.array(sorted(keep))

    def table(kind):
        parts = []
        for j, (recs, ch) in enumerate(cohorts):
            X, yy, pp = patient_table(recs, ch, kind)
            parts.append(X[keep] if j == 2 else X)
        return np.vstack(parts)

    print(f"n={len(y)}  N={int((y==0).sum())} PD={int((y==1).sum())} "
          f"ET={int((y==2).sum())}   {SPLITS} splits", flush=True)
    print("building cross-spectral tables ...", flush=True)
    T_mean, T_lam1, T_pol = table("mean"), table("lam1"), table("pol")
    for nm, T in (("mean", T_mean), ("lam1", T_lam1), ("pol", T_pol)):
        assert len(T) == len(y), f"{nm}: {len(T)} vs {len(y)}"

    # sanity: with multi-segment averaging lambda_1 must differ from the mean
    rel = np.abs(T_lam1 - T_mean).mean() / (np.abs(T_mean).mean() + 1e-12)
    print(f"  lambda_1 vs axis-mean, mean relative difference {rel:.3f}"
          f"  ({'OK' if rel > 0.01 else 'DEGENERATE -- check segment averaging'})")
    print(f"  mean lambda_1/trace over the band: {T_pol.mean():.3f}\n", flush=True)

    ARMS = (("axis mean, linear bins (current)", logbin(T_mean)),
            ("axis mean, LOG-freq bins", logfreq_bin(T_mean)),
            ("principal eigenvalue, linear bins", logbin(T_lam1)),
            ("principal eigenvalue, LOG-freq bins", logfreq_bin(T_lam1)),
            ("polarisation spectrum, linear bins", logbin(T_pol)))

    res = {}
    print(f"{'arm':>36}" + "".join(f"{c:>9}" for c in NM) + "   sd(macroP)")
    for lab, S in ARMS:
        res[lab] = evaluate(np.nan_to_num(S), D, traj, y, key)
        m = res[lab].mean(0)
        print(f"{lab:>36}" + "".join(f"{v:>9.3f}" for v in m)
              + f"{res[lab][:, 3].std():>12.3f}", flush=True)

    base = res["axis mean, linear bins (current)"]
    print("\npaired vs the current representation, same splits:")
    for lab, _ in ARMS[1:]:
        print(f"  {lab}:")
        for (dd, lo, hi), c in zip(paired(res[lab], base), NM):
            star = "*" if lo > 0 or hi < 0 else " "
            print(f"    {c:>8} {dd:+.3f}  [{lo:+.3f}, {hi:+.3f}] {star}")
    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    main()
