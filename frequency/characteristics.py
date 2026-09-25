"""Goal 1: tremor characteristics and classification from mean / max frequency.

Two things:

1. **Describe** the tremor. Per class and per cohort, the quantities clinicians
   actually name -- dominant (max-power) frequency, power-weighted mean
   frequency, bandwidth, in-band power fraction, harmonic ratio, and the
   temporal stability of the instantaneous frequency.

2. **Classify** from those alone, starting with just mean and max frequency, so
   the contribution of each added quantity is visible rather than buried in a
   30-dimensional feature vector.

Reference expectations from the clinical literature: PD rest tremor is
~4-6 Hz, ET ~6-12 Hz and more symmetric between limbs, and ET's
cycle-to-cycle frequency is the more stable of the two.

Run: ``python -m frequency.characteristics``
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.signal import welch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CLASS_NAMES = ("N", "PD", "ET")
FEATURES = ("max_freq", "mean_freq", "bandwidth", "inband_frac", "harm_ratio",
            "peak_sharp")


def spectrum_characteristics(x, fs=100.0, f_lo=3.0, f_hi=15.0):
    """Frequency-domain characteristics of one recording (channels averaged).

    ``max_freq``    dominant frequency -- the bin carrying most power.
    ``mean_freq``   power-weighted mean frequency (spectral centroid).
    ``bandwidth``   power-weighted standard deviation about that centroid.
    ``inband_frac`` fraction of 0.5-40 Hz power inside the tremor band; a
                    tremor-vs-noise measure rather than a which-tremor one.
    ``harm_ratio``  power near 2 x the dominant frequency, relative to the
                    dominant peak.
    ``peak_sharp``  peak power divided by in-band mean power -- how sharply
                    peaked the tremor is.
    """
    x = np.atleast_2d(np.asarray(x, dtype=float))
    n = int(min(512, x.shape[-1]))
    f, P = welch(x, fs=fs, nperseg=n, axis=-1)
    P = P.mean(0)
    band = (f >= f_lo) & (f <= f_hi)
    wide = (f >= 0.5) & (f <= 40.0)
    fb, pb = f[band], P[band]
    if pb.sum() <= 0 or not band.any():
        return {k: float("nan") for k in FEATURES}
    w = pb / pb.sum()
    fmax = float(fb[np.argmax(pb)])
    fmean = float((fb * w).sum())
    bw = float(np.sqrt(((fb - fmean) ** 2 * w).sum()))

    def near(c, half=0.75):
        m = (f >= c - half) & (f <= c + half)
        return float(P[m].sum()) if m.any() else 0.0

    p1 = near(fmax)
    return {
        "max_freq": fmax,
        "mean_freq": fmean,
        "bandwidth": bw,
        "inband_frac": float(pb.sum() / (P[wide].sum() + 1e-20)),
        # The harmonic window is TWICE the fundamental's. A true 2nd harmonic is
        # phase-locked at exactly 2x the instantaneous frequency, so frequency
        # jitter widens its peak twice as much; an equal-width window cut the
        # ratio from 0.25 to 0.18 at 0.4 Hz of jitter (0.16 at 0.6 Hz).
        "harm_ratio": float(near(2 * fmax, half=1.5) / (p1 + 1e-20)) if p1 > 0 else 0.0,
        "peak_sharp": float(pb.max() / (pb.mean() + 1e-20)),
    }


def patient_table(recs, ch=slice(0, 3), fs=100.0, **kw):
    """(patients, 6) characteristics, averaged over a patient's recordings."""
    rows, lab = defaultdict(list), {}
    for r in recs:
        sig = r.x[ch] if r.x.shape[0] > 3 else r.x
        d = spectrum_characteristics(sig, fs=fs, **kw)
        rows[r.subject].append([d[k] for k in FEATURES])
        lab[r.subject] = r.y
    pats = sorted(rows)
    X = np.array([np.nanmean(rows[p], axis=0) for p in pats])
    return (np.nan_to_num(X), np.array([lab[p] for p in pats]), np.array(pats))


def describe(X, y, tag=""):
    """Print per-class mean +/- sd for every characteristic."""
    print(f"\n### {tag}   n={len(y)}  "
          + "  ".join(f"{c}={int((y == i).sum())}"
                      for i, c in enumerate(CLASS_NAMES)))
    print(f"{'characteristic':>14}" + "".join(f"{c:>20}" for c in CLASS_NAMES))
    for j, name in enumerate(FEATURES):
        cells = []
        for i in range(3):
            v = X[y == i, j]
            cells.append(f"{np.nanmean(v):8.2f} +/-{np.nanstd(v):<7.2f}"
                         if len(v) else f"{'--':>18}")
        print(f"{name:>14}" + "".join(f"{c:>20}" for c in cells))


def classify(X, y, names=FEATURES, tag="", axis="PD_vs_ET", n_splits=5,
             n_perm=0, min_pos=5, n_repeats=1):
    """Cumulative feature-set classification, one characteristic at a time.

    ``n_perm`` > 0 adds a shuffled-label null for the full feature set, run
    through the identical pipeline: the AUC that labels carrying no information
    reach, as a 95 % range, plus a p-value. Without it a small-cohort AUC is
    uninterpretable. 2015 PD-vs-ET reads 0.31, which looks like a model worse
    than random, and sits inside its own null of about [0.29, 0.66].

    ``min_pos`` skips a contrast whose smaller class is below it. With 6 ET
    (NewData), each test fold holds about one ET patient and the pooled
    out-of-fold AUC is dominated by small-sample effects. Default 5 keeps the
    old behaviour; the notebook uses 10.

    ``n_repeats`` > 1 averages AUC, precision and recall over that many
    re-shuffled 5-fold partitions and prints the AUC range across them. One
    partition is not enough at 15 ET: 2015 OUT ``max_freq`` alone scores
    0.32-0.52 depending only on the CV seed, and seed 0 is the lowest -- the
    0.31 the notebook used to print. The null is then built from the same
    repeat-averaged statistic, so the two stay comparable. Default 1 keeps the
    old output.
    """
    if axis == "PD_vs_ET":
        m = y != 0
        Xa, ya = X[m], (y[m] == 2).astype(int)
        pos = "ET"
    else:
        Xa, ya = X, (y != 0).astype(int)
        pos = "Tremor"
    if ya.sum() < min_pos or (1 - ya).sum() < min_pos:
        print(f"\n  {tag}  {axis}: only {int(min(ya.sum(), (1 - ya).sum()))} in "
              f"the smaller class (< {min_pos}) -- not evaluable, skipped")
        return
    print(f"\n  {tag}  {axis}   n={len(ya)}  {pos}={int(ya.sum())}")
    rng_col = "  AUC range over CV seeds" if n_repeats > 1 else ""
    print(f"{'features used':>44}{'AUC':>8}{'prec':>8}{'rec':>8}{rng_col}")

    def mdl():
        return make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=5000,
                                                class_weight="balanced"))

    def cv_auc(Xc, yc, seed0):
        """Mean AUC / precision / recall over n_repeats CV partitions."""
        a, pp, rr = [], [], []
        for r in range(n_repeats):
            cv = StratifiedKFold(n_splits, shuffle=True, random_state=seed0 + r)
            pr = cross_val_predict(mdl(), Xc, yc, cv=cv,
                                   method="predict_proba")[:, 1]
            P, R, _, _ = precision_recall_fscore_support(
                yc, (pr >= 0.5).astype(int), labels=[1], zero_division=0)
            a.append(roc_auc_score(yc, pr)); pp.append(P[0]); rr.append(R[0])
        return np.array(a), float(np.mean(pp)), float(np.mean(rr))

    for k in range(1, len(names) + 1):
        cols = [FEATURES.index(n) for n in names[:k]]
        a, P, R = cv_auc(Xa[:, cols], ya, 0)
        extra = f"   [{a.min():.3f}, {a.max():.3f}]" if n_repeats > 1 else ""
        print(f"{' + '.join(names[:k]):>44}{a.mean():>8.3f}{P:>8.3f}{R:>8.3f}"
              f"{extra}")
    if n_perm > 0:
        real = a.mean()
        rng = np.random.default_rng(0)
        null = []
        for i in range(n_perm):
            yp = rng.permutation(ya)
            null.append(cv_auc(Xa[:, cols], yp, i * n_repeats)[0].mean())
        null = np.array(null)
        lo, hi = np.quantile(null, [0.025, 0.975])
        p = float(np.mean(null >= real))
        verdict = ("above chance" if p < 0.05 else
                   "NOT distinguishable from chance")
        print(f"{'shuffled-label null, all features':>44}   95 % range "
              f"[{lo:.3f}, {hi:.3f}]   p={p:.3f}   -> {verdict}")


def main():
    from common.loaders import load_pads_extracted
    from common.load_2025 import load_2025_all
    from common.quaternion_data import load_quaternion_recordings

    cohorts = [
        ("2015 OUT", load_quaternion_recordings("Data", action="OUT",
                                                mode="angular_velocity"),
         slice(3, 6)),
        ("NewData OUT", load_2025_all(conditions=("OUT",)), slice(3, 6)),
        ("PADS StretchHold", load_pads_extracted("pads_stretchhold"),
         slice(0, 3)),
    ]
    print("=" * 78)
    print("TREMOR CHARACTERISTICS  (angular velocity, 3-15 Hz band)")
    print("=" * 78)
    tables = {}
    for name, recs, ch in cohorts:
        X, y, pats = patient_table(recs, ch=ch)
        tables[name] = (X, y)
        describe(X, y, tag=name)

    print("\n" + "=" * 78)
    print("CLASSIFICATION FROM FREQUENCY CHARACTERISTICS ALONE")
    print("features are added cumulatively, starting from max frequency")
    print("=" * 78)
    for name, (X, y) in tables.items():
        classify(X, y, tag=name, axis="N_vs_Tremor")
        classify(X, y, tag=name, axis="PD_vs_ET")
    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    main()
