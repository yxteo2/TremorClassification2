"""Was HHT closed fairly? Audit the removed EMD/HHT estimator.

HHT was dropped on 2026-09-28 as "worst of the estimators" (closed_families:
`signal_processing_summary.md`), while an earlier line of the same summary called
HHT-7/8 the best PD-vs-ET separator. A method's ranking is only as good as its
implementation, so this checks the implementation before the verdict.

## The removed estimator (loaded from git history, not retyped)

`signal_processing/tfd.py` at `57fd8a72^` -- `_emd_imfs`, `apply_hht` -- is
executed from the committed source with its one pruned import stubbed, and
summarised exactly as the old `transforms.m_hht` did: 3-15 Hz in 0.25 Hz bins,
`decim=4`, squared, averaged over time and channels.

Suspected weaknesses, read from that code:

1. instantaneous frequency is a **one-sample phase difference** -- the noisiest
   possible derivative -- binned at 0.25 Hz, so jitter smears energy across bins;
2. no end-effect handling before the Hilbert transform;
3. plain EMD, which mode-mixes under noise.

## The corrected estimators

* **HHT edges** -- the old instantaneous frequency, only the band edges made
  finite (found by this audit's first run, below).
* **HHT full fix** -- edges, plus per IMF a 1 s mirror pad before the Hilbert
  transform and a Savitzky-Golay derivative of the unwrapped phase (11 samples,
  0.11 s) for the instantaneous frequency; energy = amplitude^2 per sample.

**Found on the first run:** the removed `apply_hht` bins with outer edges of
-inf / +inf, so its out-of-band mask never excludes anything. Every noisy
synthetic signal -- and pure noise -- peaked at exactly 15.00 Hz with sharpness
18-31, under the old estimator and under a first "fix" that kept those edges.
The prediction below was written before that run.

## Synthetic tests (known answers, 10 noise draws each, 3 channels, 15 s)

clean 6 Hz | 6 Hz + noise | jittered + amplitude-modulated 6 Hz + noise |
6 Hz + 12 Hz harmonic | 4.5 + 7.5 Hz | noise only. Reported: peak error and
peak sharpness (peak / band mean, the repo's `peak_sharp`) for old HHT, fixed
HHT and the pipeline's multitaper.

## Real data: 2015 OUT, lower arm (the user's scope)

Four spectral descriptors per method (max_freq, mean_freq, bandwidth,
peak_sharp), PD vs ET and N vs tremor, logistic regression 5-fold x 20 repeats
against a 200-permutation null.

## Prediction, recorded before the run

**Synthetic:** old HHT has the bluntest peak on jittered and noisy tremor
(sharpness well below multitaper), and the fix recovers most of it; all three
find the frequency of a clean tone. **Real:** no method separates PD from ET on
2015 OUT above its null (spectral shape carries none there); on N vs tremor the
fixed HHT is at least as good as the old one -- i.e. "HHT measured worst" was
partly the implementation.

Run: ``python -m experiments.hht_audit``
"""

from __future__ import annotations

import subprocess
import sys
import types

import numpy as np
from scipy.signal import hilbert, savgol_filter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FS = 100.0
FREQS = np.arange(3.0, 15.0 + 1e-9, 0.25)
#: FINITE outer edges. The removed apply_hht used -inf / +inf, so its "valid"
#: mask was always true and every out-of-band frequency was binned into the
#: 3 Hz or 15 Hz edge bin -- sensor noise above 15 Hz (EMD's first IMF) piled up
#: at 15 Hz. That, not the method, is why HHT "measured worst".
EDGES = np.concatenate([[FREQS[0] - 0.125], (FREQS[:-1] + FREQS[1:]) / 2,
                        [FREQS[-1] + 0.125]])
REPEATS, NPERM = 20, 200


def load_old():
    """The removed tfd.py, executed from git history with its dead import stubbed."""
    src = subprocess.run(["git", "show", "57fd8a72^:signal_processing/tfd.py"],
                         capture_output=True, text=True, check=True).stdout
    stub = types.ModuleType("signal_processing.preprocessing")
    stub.apply_stft = stub._frame = stub._hann_periodic = None
    sys.modules.setdefault("signal_processing.preprocessing", stub)
    mod = types.ModuleType("old_tfd")
    exec(compile(src, "old_tfd.py@57fd8a72^", "exec"), mod.__dict__)
    return mod


OLD = load_old()


def old_hht(x):
    """Exactly the old transforms.m_hht summary of the old apply_hht."""
    x = np.atleast_2d(x)
    S = OLD.apply_hht(x, fs=FS, freqs=FREQS, max_imfs=8, decim=4)
    S = np.abs(S) ** 2
    return FREQS, S.reshape(x.shape[0], len(FREQS), -1).mean(axis=(0, 2))


def hht(x, smooth_if=True, pad_s=1.0, sg=11):
    """HHT marginal energy spectrum with finite band edges.

    ``smooth_if=False, pad_s=0`` keeps the old estimator's instantaneous
    frequency (one-sample phase difference, no padding) and changes ONLY the
    band edges -- the edges-only fix. ``smooth_if=True`` adds mirror padding
    and a Savitzky-Golay phase derivative.
    """
    x = np.atleast_2d(np.asarray(x, float))
    P = np.zeros(len(FREQS))
    pad = int(pad_s * FS)
    for ch in x:
        for imf in OLD._emd_imfs(ch, max_imfs=8):
            imf = imf.astype(float)
            if np.allclose(imf, 0):
                continue
            if pad:
                z = np.concatenate([imf[pad:0:-1], imf, imf[-2:-pad - 2:-1]])
            else:
                z = imf
            an = hilbert(z)
            ph = np.unwrap(np.angle(an))
            if smooth_if:
                f = savgol_filter(ph, sg, 2, deriv=1) * FS / (2 * np.pi)
            else:
                f = np.empty_like(ph)
                f[1:] = np.diff(ph) * FS / (2 * np.pi)
                f[0] = f[1]
            sl = slice(pad, pad + len(imf))
            e, f = np.abs(an[sl]) ** 2, f[sl]
            b = np.digitize(f, EDGES) - 1
            ok = (b >= 0) & (b < len(FREQS))
            np.add.at(P, b[ok], e[ok])
    return FREQS, P / x.shape[0]


def hht_edges(x):
    return hht(x, smooth_if=False, pad_s=0)


def hht_fixed(x):
    return hht(x)


def multitaper(x):
    from signal_processing.transforms import METHODS
    return METHODS["multitaper"](np.atleast_2d(x), fs=FS)


METHODS = {"old HHT": old_hht, "HHT edges": hht_edges, "HHT full fix": hht_fixed,
           "multitaper": multitaper}


def peak(f, P):
    b = (f >= 3) & (f <= 15)
    f, P = f[b], P[b]
    return float(f[np.argmax(P)]), float(P.max() / (P.mean() + 1e-20))


def synth(kind, rng, T=15.0):
    t = np.arange(0, T, 1 / FS)
    if kind == "clean 6 Hz":
        s, sd = np.sin(2 * np.pi * 6 * t), 0.0
    elif kind == "6 Hz + noise":
        s, sd = np.sin(2 * np.pi * 6 * t), 0.6
    elif kind == "jittered + AM":
        f_inst = 6 + np.cumsum(rng.normal(0, 0.02, len(t)))
        f_inst = 6 + 0.3 * (f_inst - f_inst.mean()) / (f_inst.std() + 1e-12)
        am = 1 + 0.3 * np.sin(2 * np.pi * 0.3 * t)
        s, sd = am * np.sin(2 * np.pi * np.cumsum(f_inst) / FS), 0.6
    elif kind == "6 + 12 Hz harmonic":
        s, sd = np.sin(2 * np.pi * 6 * t) + 0.3 * np.sin(2 * np.pi * 12 * t), 0.6
    elif kind == "4.5 + 7.5 Hz":
        s, sd = np.sin(2 * np.pi * 4.5 * t) + 0.7 * np.sin(2 * np.pi * 7.5 * t), 0.6
    else:
        s, sd = 0 * t, 1.0
    u = rng.normal(size=3)
    u /= np.linalg.norm(u)
    return np.outer(u, s) + sd * rng.normal(size=(3, len(t)))


def synthetic():
    print("SYNTHETIC  (median over 10 draws; true peak 6 Hz unless noted)")
    print(f"{'signal':<20}" + "".join(f"{m + ' peak':>18}{'sharp':>8}"
                                       for m in METHODS))
    for kind in ("clean 6 Hz", "6 Hz + noise", "jittered + AM",
                 "6 + 12 Hz harmonic", "4.5 + 7.5 Hz", "noise only"):
        cells = []
        for m, fn in METHODS.items():
            r = np.array([peak(*fn(synth(kind, np.random.default_rng(s))))
                          for s in range(10)])
            cells.append(f"{np.median(r[:, 0]):>18.2f}{np.median(r[:, 1]):>8.2f}")
        print(f"{kind:<20}" + "".join(cells), flush=True)


def descriptors(f, P):
    b = (f >= 3) & (f <= 15)
    f, P = f[b], P[b]
    w = P / (P.sum() + 1e-20)
    mf = float((f * w).sum())
    return [float(f[np.argmax(P)]), mf, float(np.sqrt(((f - mf) ** 2 * w).sum())),
            float(P.max() / (P.mean() + 1e-20))]


def model():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=5000, class_weight="balanced"))


def cv_auc(X, y, seed0=0):
    return float(np.mean([roc_auc_score(y, cross_val_predict(
        model(), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=seed0 + r),
        method="predict_proba")[:, 1]) for r in range(REPEATS)]))


def with_null(X, y):
    real = cv_auc(X, y)
    rng = np.random.default_rng(0)
    null = np.array([cv_auc(X, rng.permutation(y), 1000 + i * REPEATS)
                     for i in range(NPERM)])
    return real, *np.quantile(null, [0.025, 0.975]), (1 + np.sum(null >= real)) / (1 + NPERM)


def real():
    from common.quaternion_data import load_quaternion_recordings
    recs = load_quaternion_recordings("Data", action="OUT", mode="angular_velocity")
    print(f"\nREAL  2015 OUT, lower arm, {len(recs)} recordings")
    tabs = {}
    for m, fn in METHODS.items():
        rows, lab = {}, {}
        for r in recs:
            rows.setdefault(r.subject, []).append(descriptors(*fn(r.x[3:6])))
            lab[r.subject] = r.y
        pats = sorted(rows)
        tabs[m] = (np.array([np.mean(rows[p], 0) for p in pats]),
                   np.array([lab[p] for p in pats]))
        print(f"  {m}: done", flush=True)
    print(f"\n{'method':<12}{'axis':<13}{'CV AUC':>8}{'null 95%':>20}{'p':>8}"
          "   univariate AUC: max_freq mean_freq bandwidth peak_sharp")
    for m, (X, y) in tabs.items():
        for axis in ("PD_vs_ET", "N_vs_Tremor"):
            if axis == "PD_vs_ET":
                k = y != 0; Xa, ya = X[k], (y[k] == 2).astype(int)
            else:
                Xa, ya = X, (y != 0).astype(int)
            a, lo, hi, p = with_null(Xa, ya)
            uni = " ".join(f"{roc_auc_score(ya, Xa[:, j]):.2f}" for j in range(4))
            print(f"{m:<12}{axis:<13}{a:>8.3f}   [{lo:.3f}, {hi:.3f}]{p:>8.3f}   {uni}",
                  flush=True)


if __name__ == "__main__":
    synthetic()
    real()
    print("\nMARKER_DONE", flush=True)
