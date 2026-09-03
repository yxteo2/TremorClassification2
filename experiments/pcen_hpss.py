"""PCEN and harmonic-percussive separation — the two audio front-ends that need the time axis.

`audio_techniques.py` already brought four ideas over from the audio literature:
frequency-aware convolution (+0.010, n.s.), SpecAugment frequency masking
(−0.021), multi-resolution concatenation (diluted, like every union here), and a
**PCEN-style trainable exponent**. On that last one it recorded the reason it
could only do half the method:

> PCEN beats the fixed pointwise log of a log-mel front-end. **Its gain-control
> term needs a time axis**, so what carries over is the per-band trainable
> exponent.

That is the gap this closes. Every transform here computes a time–frequency
surface and *then* collapses it with `P.mean(0)` before anything downstream sees
it, so the two strongest audio front-ends were unreachable — not refuted.

## The two methods, and why each has a tremor-specific reason

**PCEN** (Lostanlen et al., *IEEE Signal Processing Letters* 26(1), 2019;
Wang et al. 2017) replaces static log compression with adaptive per-band gain
control:

    M(t,f) = (1-s)·M(t-1,f) + s·E(t,f)          # IIR smoother along time
    PCEN   = (E / (eps + M)^alpha + delta)^r - delta^r

It divides each band by its own recent history, so **stationary background in a
band is suppressed while modulation survives**. The paper's framing is that this
Gaussianises magnitude distributions and decorrelates bands.

The tremor reason: this pipeline's only amplitude normalisation is a *global*
sum-normalisation per recording, which cannot tell a band carrying steady
voluntary motion from a band carrying tremor. PCEN can. The smoother's time
constant is set to **1.5 s**, chosen from this project's own measurement that
tremor amplitude modulation lives at 0.05–5 Hz (`ampmod_features`), so
stationary background is suppressed and waxing-and-waning is not.

**HPSS** (median-filter formulation, Fitzgerald 2010) splits the surface into a
**harmonic** part — horizontal, frequency-sparse, temporally sustained — and a
**percussive** part — vertical, broadband, time-localised — via median filters
along each axis and a soft Wiener mask.

The tremor reason is stronger than the PCEN one. ET is a sustained near-tonal
oscillation; voluntary movement, limb repositioning and the class-ordered PADS
arm-raising onset measured in `pads_onset_trim.md` are broadband transients.
HPSS is a principled separator for exactly that, and unlike the onset trim it
acts **throughout** the recording rather than on the first 1.5 s.

## The constraint that shapes the design

At the reported model's hop (nperseg 256, noverlap 192 → 0.64 s) a PADS
recording yields **13 frames**. That is too few for a median filter along time
and marginal for an IIR smoother. Denser sampling is required — noverlap 240
(hop 0.16 s) gives 49 frames on PADS and 81 on 2015 — and that is a change in
its own right.

So the hop change gets **its own arm**. Without it, any PCEN or HPSS effect
would be confounded with simply sampling the surface more densely.

  reported (hop 0.64 s)      must reproduce build() bit-exactly
  dense hop 0.16 s           the control that isolates the resampling
  dense + PCEN               adaptive per-band gain control
  dense + HPSS harmonic      keep the sustained tonal part
  dense + HPSS percussive    the attribution control -- if the *transient* part
                             classifies as well, the physics story is wrong

Frequency resolution, band, grid, log-binning, architecture, seeds, splits and
priors are identical across every arm; only the time-axis treatment differs.

## The prediction, recorded before the run

**HPSS-harmonic > dense-hop control > HPSS-percussive on precET.** The ordering
is the claim, not the magnitude: tremor is the sustained component, so keeping it
should help and keeping only transients should hurt. If percussive matches
harmonic, the separation is not doing what its physics says.

**PCEN: small and uncertain in sign.** Its gain control targets *stationary*
background, and this pipeline's recordings are short, already band-passed to
3–15 Hz, and sum-normalised — three reasons the thing PCEN removes may largely
be gone already.

Neither is predicted to beat the reported model. The spectrum has been a
near-sufficient statistic at this n, and the honest question is whether a
principled denoiser reaches the contested 40 % that every reshuffling method has
failed to move.

20 splits, paired, checkpointed. Run: ``python -m experiments.pcen_hpss``
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
from scipy.ndimage import median_filter
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedShuffleSplit

import experiments.final_model as FM
from common.cohorts import logbin
from common.protocol import TEST_FRAC, VAL_FRAC, tune_offsets
from experiments._resume import resume_load, resume_save
from experiments.alltasks_final import paired
from experiments.estimator_smoothing import load_cohorts
from experiments.pooling_rules import fit_members
from signal_processing.tfd import apply_multitaper
from signal_processing.transforms import F_MAX, _kept_rfftfreq

NM = ("precN", "precPD", "precET", "macroP", "macroF1", "recET", "nETpred")
SPLITS = 20
FS, NPERSEG = 100.0, 256
HOP_BASE, HOP_DENSE = 64, 16          # 0.64 s (reported) and 0.16 s
TAU_S = 1.5                           # PCEN gain-control time constant


def pcen(E, hop_s, tau_s=TAU_S, alpha=0.98, delta=2.0, r=0.5, eps=1e-6):
    """Per-channel energy normalisation over the time axis of ``E`` (..., time)."""
    s = min(hop_s / max(tau_s, 1e-6), 1.0)
    M = np.empty_like(E)
    M[..., 0] = E[..., 0]
    for t in range(1, E.shape[-1]):
        M[..., t] = (1 - s) * M[..., t - 1] + s * E[..., t]
    return (E / (eps + M) ** alpha + delta) ** r - delta ** r


def hpss(E, k_time=17, k_freq=17, power=2.0, eps=1e-10):
    """Soft-mask harmonic/percussive split of one (freq, time) power surface."""
    H = median_filter(E, size=(1, k_time), mode="nearest")
    P = median_filter(E, size=(k_freq, 1), mode="nearest")
    Hp, Pp = H ** power, P ** power
    mh = Hp / (Hp + Pp + eps)
    return E * mh, E * (1.0 - mh)


def surface(x, hop):
    """Multitaper POWER surface as (n_ch, n_freq, n_time), plus its axis."""
    n = min(NPERSEG, x.shape[-1])
    S = apply_multitaper(x, fs=FS, nperseg=n, nfft=n, noverlap=n - hop,
                         f_max=F_MAX)
    n_ch = np.atleast_2d(x).shape[0]
    n_freq = np.asarray(S).shape[0] // n_ch
    E = np.asarray(S, float).reshape(n_ch, n_freq, -1) ** 2
    return E, _kept_rfftfreq(n, FS)


def spectrum(x, mode):
    """One recording's 3-15 Hz spectrum on FM.GRID under one time-axis rule."""
    hop = HOP_BASE if mode == "base" else HOP_DENSE
    E, f = surface(x, hop)
    if mode == "pcen":
        E = pcen(E, hop / FS)
    elif mode in ("harm", "perc"):
        out = np.empty_like(E)
        for c in range(E.shape[0]):
            h, p = hpss(E[c])
            out[c] = h if mode == "harm" else p
        E = out
    P = E.mean(axis=(0, 2))
    k = (f >= 3.0) & (f <= F_MAX)
    v = np.clip(np.interp(FM.GRID, f[k], P[k], left=0.0, right=0.0), 0, None)
    return v / (v.sum() + 1e-20)


def spec_for(mode, recs, keep):
    def table(rs, ch):
        rows = defaultdict(list)
        for r in rs:
            xx = r.x[ch] if r.x.shape[0] > 3 else r.x
            rows[r.subject].append(spectrum(xx, mode))
        return np.array([np.mean(rows[k], 0) for k in sorted(rows)])
    rA, rB, rC = recs
    return logbin(np.nan_to_num(np.vstack(
        [table(rA, slice(3, 6)), table(rB, slice(3, 6)),
         table(rC, slice(0, 3))[keep]])))


def score(pt, off, yte):
    pred = (np.log(pt + 1e-12) + off).argmax(1)
    P, R, F, _ = precision_recall_fscore_support(yte, pred, labels=[0, 1, 2],
                                                 zero_division=0)
    return [P[0], P[1], P[2], P.mean(), F.mean(), R[2], float((pred == 2).sum())]


def main():
    torch.set_num_threads(1)
    d = FM.build()
    y, key = d["y"], d["key"]
    D = np.hstack([d["DESC"], d["ASYM"], d["HAVE"]])
    traj = d["TRAJ"]
    recs, keep = load_cohorts()

    ARMS = {"reported (hop 0.64s)": "base", "dense hop 0.16s": "dense",
            "dense + PCEN": "pcen", "dense + HPSS harmonic": "harm",
            "dense + HPSS percussive": "perc"}
    SPEC = {}
    for a, m in ARMS.items():
        print(f"building spectra: {a} ...", flush=True)
        S = spec_for(m, recs, keep)
        assert len(S) == len(y), f"{a}: {len(S)} rows, expected {len(y)}"
        SPEC[a] = S
    dev = float(np.abs(SPEC["reported (hop 0.64s)"]
                       - d["SPEC"]["multitaper"]).max())
    print(f"\nbaseline arm vs build(): max|diff| = {dev:.2e} "
          f"{'OK' if dev < 1e-6 else 'MISMATCH -- comparisons invalid'}")
    assert dev < 1e-6
    for a in list(ARMS)[1:]:
        print(f"  {a:>26} moved the input by "
              f"{float(np.abs(SPEC[a] - SPEC['reported (hop 0.64s)']).max()):.3f} "
              f"log units")
    print(flush=True)

    res, done = resume_load("pcen_hpss", list(ARMS))
    for sp in range(SPLITS):
        if sp in done:
            continue
        tv, te = next(StratifiedShuffleSplit(1, test_size=TEST_FRAC,
                                             random_state=sp).split(y[:, None],
                                                                    key))
        t0, v0 = next(StratifiedShuffleSplit(1, test_size=VAL_FRAC,
                                             random_state=sp).split(
                                                y[tv][:, None], key[tv]))
        tr, va = tv[t0], tv[v0]
        for a in ARMS:
            V, T = fit_members(SPEC[a], D, traj, y, tr, va, te)
            res[a].append(score(T.mean(0), tune_offsets(V.mean(0), y[va]),
                                y[te]))
        resume_save("pcen_hpss", res, sp)
        print(f"  split {sp+1}/{SPLITS}", flush=True)

    for a in res:
        res[a] = np.array(res[a])

    print(f"\n{'arm':>26}" + "".join(f"{c:>9}" for c in NM) + "   sd(macroP)")
    for a in ARMS:
        print(f"{a:>26}" + "".join(f"{v:>9.3f}" for v in res[a].mean(0))
              + f"{res[a][:, 3].std():>12.3f}")

    base = res["reported (hop 0.64s)"]
    print("\npaired vs the reported model (ADOPTION):")
    for a in list(ARMS)[1:]:
        print(f"  {a}:")
        for (dd, lo, hi), c in zip(paired(res[a], base), NM):
            star = "*" if lo > 0 or hi < 0 else " "
            print(f"    {c:>8} {dd:+.3f}  [{lo:+.3f}, {hi:+.3f}] {star}")

    dense = res["dense hop 0.16s"]
    print("\npaired vs the DENSE-HOP control (ATTRIBUTION -- isolates the "
          "front-end from the resampling):")
    for a in ("dense + PCEN", "dense + HPSS harmonic", "dense + HPSS percussive"):
        print(f"  {a}:")
        for (dd, lo, hi), c in zip(paired(res[a], dense), NM):
            star = "*" if lo > 0 or hi < 0 else " "
            print(f"    {c:>8} {dd:+.3f}  [{lo:+.3f}, {hi:+.3f}] {star}")

    print("\nTHE PREDICTION -- harmonic > dense control > percussive on precET:")
    for a in ("dense + HPSS harmonic", "dense hop 0.16s",
              "dense + HPSS percussive"):
        print(f"  {a:>26}  precET {res[a][:,2].mean():.3f}  "
              f"macroP {res[a][:,3].mean():.3f}")
    print("\nsplit-level win rate vs the reported model:")
    for a in list(ARMS)[1:]:
        print(f"  {a:>26}: " + "  ".join(
            f"{c} {float((res[a][:, i] > base[:, i]).mean()):.2f}"
            for i, c in enumerate(NM[:5])))
    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    main()
