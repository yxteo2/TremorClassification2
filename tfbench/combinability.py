"""Can two cohorts be pooled? A pre-merge check, run before any training.

Pooling is only defensible when the cohorts agree on the thing being measured
AND cannot be told apart by the model. This checks both, on the only class the
2015 and NewData cohorts share (**ET**), because that is where a disagreement
would silently become a label:

1. **Frequency agreement** — do max_freq / mean_freq distributions overlap?
   Reported as effect size with a bootstrap CI on the median difference, plus a
   two-one-sided-test style verdict against a tolerance in Hz. A non-significant
   difference is NOT evidence of agreement; the CI has to be narrow too.
2. **Device-identity probe** — can a classifier tell which cohort a patient came
   from? Judged on ``|AUC - 0.5|``, since an AUC of 0.000 is maximally
   separable (LOO-inverted labels), not safe.

Both must pass. NewData is ET-only, so a device signature would let a pooled
model read "new device => ET" and inflate everything downstream.

Run:  python -m tfbench.combinability
"""

from __future__ import annotations

import numpy as np
from scipy.stats import mannwhitneyu

from tfbench.frequency_report import patient_freqs, WRIST_2015
from tfbench.merged import descriptor_table, device_probe

TOLERANCE_HZ = 1.0     # median difference we are willing to call "the same"


def frequency_agreement(V_a, V_b, n_boot=5000, seed=0, tol=TOLERANCE_HZ):
    """Compare two cohorts' [max_freq, mean_freq] arrays. Returns per-measure dict."""
    rng = np.random.default_rng(seed)
    out = {}
    for j, nm in ((0, "max_freq"), (1, "mean_freq")):
        a, b = V_a[:, j], V_b[:, j]
        u, p = mannwhitneyu(a, b)
        eff = 2 * u / (len(a) * len(b)) - 1
        d = np.array([np.median(rng.choice(a, len(a), True))
                      - np.median(rng.choice(b, len(b), True)) for _ in range(n_boot)])
        lo, hi = np.percentile(d, [2.5, 97.5])
        # equivalent only if the WHOLE CI sits inside +/- tol
        out[nm] = dict(med_a=float(np.median(a)), med_b=float(np.median(b)),
                       diff=float(np.median(a) - np.median(b)), eff=float(eff),
                       p=float(p), lo=float(lo), hi=float(hi),
                       equivalent=bool(lo > -tol and hi < tol))
    return out


def check(cohort_a, cohort_b, ch_a=WRIST_2015, ch_b=WRIST_2015,
          shared_class=2, method="welch", tol=TOLERANCE_HZ, label=("A", "B")):
    """Full pre-merge check on the shared class. Returns (verdict, details)."""
    Va, ya, ga = patient_freqs(cohort_a, ch_a)
    Vb, yb, gb = patient_freqs(cohort_b, ch_b)
    Va, ga = Va[ya == shared_class], ga[ya == shared_class]
    Vb, gb = Vb[yb == shared_class], gb[yb == shared_class]

    freq = frequency_agreement(Va, Vb, tol=tol)

    Xa, yA, gA = descriptor_table(cohort_a, method, ch_a)
    Xb, yB, gB = descriptor_table(cohort_b, method, ch_b)
    auc, dev = device_probe(Xa[yA == shared_class], gA[yA == shared_class],
                            Xb[yB == shared_class], gB[yB == shared_class])

    freq_ok = all(v["equivalent"] for v in freq.values())
    probe_ok = dev < 0.25
    print(f"\n=== {label[0]}  vs  {label[1]}   (shared class n={len(Va)} / {len(Vb)}) ===")
    for nm, v in freq.items():
        flag = "EQUIVALENT" if v["equivalent"] else "not equivalent"
        print(f"  {nm:>10}: {label[0]} {v['med_a']:.2f} Hz | {label[1]} {v['med_b']:.2f} Hz"
              f" | diff {v['diff']:+.2f} CI [{v['lo']:+.2f},{v['hi']:+.2f}]"
              f" | eff {v['eff']:+.3f} p={v['p']:.3f}  -> {flag} (tol +/-{tol}Hz)")
    print(f"  device probe ({method}): AUC {auc:.3f}, |dev| {dev:.3f}  -> "
          f"{'pass' if probe_ok else 'CONFOUNDED'}")
    verdict = "COMBINE" if (freq_ok and probe_ok) else "DO NOT COMBINE"
    why = [] if verdict == "COMBINE" else (
        ([] if freq_ok else ["frequency distributions differ"]) +
        ([] if probe_ok else ["cohorts are separable by device"]))
    print(f"  VERDICT: {verdict}" + (f"  ({'; '.join(why)})" if why else ""))
    return verdict, {"frequency": freq, "identity_auc": auc, "deviation": dev}


def main(data_root="Data"):
    from pdetn.load_2025 import load_2025
    from tremor.quaternion_data import load_quaternion_recordings
    results = {}
    for cond in ("REST", "OUT", "WING"):
        loc = load_quaternion_recordings(data_root, action=cond, mode="angular_velocity")
        new_cond = cond if cond in ("REST", "OUT") else None
        if new_cond is None:
            print(f"\n=== 2015 {cond}: NewData has no matching task, cannot combine ===")
            continue
        new = load_2025(mode="angular_velocity", conditions=(new_cond,))
        results[cond] = check(loc, new, label=(f"2015 {cond}", f"NewData {cond}"))
    print("\n" + "=" * 62)
    for cond, (v, _) in results.items():
        print(f"  {cond:>5}: {v}")
    return results


if __name__ == "__main__":
    main()
