# Quaternion representation work — consolidated verdict

Everything from this line of work, tested to the same standard, in one place.
Written after several intermediate claims in `quaternion_representations.md`
turned out not to hold.

## The tests that matter

All use **patient-level LOSO** and a **paired subject-level bootstrap** of the
difference (same patients scored by both models on each resample), which is the
correct test for "is B better than A" and is what I should have run first.

| claim | measured | paired 95% CI of difference | verdict |
|---|---|---|---|
| upper-arm handedness biomarker | eff −0.588, raw p 0.00035 | Bonferroni thr 2.5e-4 (n=198); BH q 0.070 | **retracted** |
| — cross-condition | OUT −0.588 / WING **+0.179** / REST −0.138 | sign reverses at WING | **retracted** |
| log map improves N-vs-tremor @OUT | 0.861 → 0.914 | **[+0.000, +0.106]**, p = 0.036 | marginal |
| — same, WING | 0.847 → 0.883 | [−0.029, +0.102], p = 0.148 | n.s. |
| — same, REST | 0.803 → 0.783 | [−0.092, +0.046], p = 0.743 | absent |
| hybrid (logmap→gchir) macro-F1 | 0.651 → 0.718 | **[−0.027, +0.157]**, p = 0.077 | **n.s.** |
| — same, ET-F1 | 0.378 → 0.462 | [−0.109, +0.273], p = 0.201 | **n.s.** |

## What this means

**Nothing in the quaternion representation work reaches significance.** The
macro-F1 0.651 → 0.718 headline, which read as the largest single-model gain in
the project, has a paired CI spanning zero. The log-map N-vs-tremor gain is the
strongest survivor and its CI lower bound is **+0.000** at the one condition
where it appears, with the direction consistent at WING but absent at REST.

This does not overturn the project's existing conclusion — it **reproduces** it.
`signal_processing_summary.md` already said that across 8 TF methods, spatial,
nonlinear, higher-order and parametric features, everything lands within CI and
the binding constraint is the ~15 ET subjects. Quaternion-aware representations
are one more family that behaves the same way.

## What the work did produce

1. **A real bug fix.** PADS ET was 32 % contaminated — 13 of 41 "ET" patients
   were not Essential Tremor, including parkinsonian cases, because diagnoses
   were substring-matched and "et" occurs in "etiology"/"asymmetric"/
   "hypokinetic". Fixed to exact matching (N=79/PD=276/ET=28, matching the
   publication). This corrected numbers in two reports.
   See `reports/pads_label_bug.md`.

2. **Correct, validated infrastructure.** `pdetn/quaternion_tf.py` and the new
   `tremor/quaternion.py` modes are exact on synthetic orbits (circularity
   1.000 vs 0.000, rotation-invariance to 2e-16; signed handedness ±1.000,
   mount-invariance to 8e-16). The code is right; the biology is not there at
   this n.

3. **A genuinely device-agnostic feature block.** Gravity-chirality is the only
   representation that does not identify the recording device (probe AUC 0.567
   vs 1.000 for everything else), which is why NewData can be pooled on it at
   all. That property is real and independent of whether it classifies well.

4. **Two confirmed negatives.** PADS remains unusable as training data after the
   label fix (pooling costs ET-F1 0.538 → 0.350, identity probe still 1.000),
   and orbit geometry does not transfer between cohorts (AUC 0.372 PADS→LOCAL).

## Method note

Three claims in this line of work were stated before the test that would refute
them had been run — the handedness biomarker (no multiple-comparison
correction, no cross-condition check), the log-map gain (called "not a CI
argument" before any paired CI existed), and the hybrid (reported as the
project's best macro-F1 with no paired test). Each was retracted or downgraded
within a few steps. The fix is procedural: for any "B beats A" claim, the paired
CI is part of the same run that produces the point estimate, not a follow-up.
