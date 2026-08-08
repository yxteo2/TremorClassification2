# The upper-arm handedness finding does NOT survive correction

Recorded because I built this up across several turns before running the checks
that undo it. `reports/quaternion_representations.md` presents
`upper_gchir_mean` as "the strongest single PD-vs-ET feature found"
(effect −0.588, p = 0.0004). That framing is wrong, for three independent
reasons found afterwards.

## 1. It fails multiple-comparison correction

The screen was never one test. Across 3 conditions x all quaternion features
(polarization, QSTFT, gravity-chirality) there are **n = 198 tests**.

| | value |
|---|---|
| `upper_gchir_mean` raw p | 0.00035 |
| Bonferroni threshold (α=0.05, n=198) | **0.000253** |
| verdict | **fails** |
| Benjamini-Hochberg q | **0.070** |
| verdict at FDR 0.05 | **fails** |

A subject-level permutation test (20 000 permutations) returns p = 0.00025,
which confirms the *raw* p is correctly computed — it does not rescue the
feature, because that p is still unadjusted for having been picked as the best
of 198.

Features at raw p<0.05: **26 of 198**, against ~10 expected by chance. So there
is probably *some* real signal in the family — but roughly 2x the chance rate is
weak, not the decisive picture the top-line number suggested.

## 2. It is OUT-only, and reverses at WING

Same subjects, different motor task — a partially independent test, and one PADS
cannot provide since it has no upper-arm sensor:

| condition | effect | p | PD median | ET median |
|---|---|---|---|---|
| **OUT** | **−0.588** | **0.0004** | −0.056 | +0.023 |
| WING | **+0.179** | 0.314 | −0.036 | −0.064 |
| REST | −0.138 | 0.390 | −0.020 | −0.011 |

OUT and WING are both postural/action tasks. A genuine physiological difference
in which direction the upper arm orbits should not vanish and **invert** between
them. This is the single most damaging check.

## 3. The limb-side confound was never excluded

4 of 6 subjects in the 2025 cohort flip handedness sign between limbs (binomial
p = 0.34 — indistinguishable from chance in either direction), and the local
cohort has no limb-side labels.

## What survives

* **Not** the biomarker claim. "PD and ET orbit the upper arm in opposite
  senses" is a **hypothesis for a future cohort**, not a result. It should not
  go in a paper as a finding.
* The three top features (`upper_gchir_mean`, `upper_q_chirality_peak`,
  `upper_q_chirality`) are ranks 1–3 of 198 and measure the same quantity
  through two independent estimators, at the same condition. That coherence is
  why it is worth *retesting*, not why it is true.
* The **classifier** results are a separate kind of evidence and are not
  invalidated here: the hybrid (logmap stage 1 → chirality stage 2, macro-F1
  0.718 vs 0.651 baseline) is a cross-validated measurement, not a screened
  p-value. But ~14 feature configurations were compared to find it, every ET-F1
  CI overlaps the baseline, and the same limb-side confound applies to stage 2 —
  so it warrants the same caution.

## Process lesson

I ran the univariate screen, reported the top hit as a headline with an
uncorrected p-value, and only ran the correction and the cross-condition check
several steps later, after building further conclusions on top. The screen was
labelled "screening only, uncorrected" in the report — and then its top result
was used as though it were confirmatory anyway. Correction and an
independent-condition check belong in the same run as the screen, before
anything is called a finding.
