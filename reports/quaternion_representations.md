# Quaternion-aware time-frequency representations

**Question.** Every representation tried so far reduces each sensor to a
*scalar* spectrum (per-axis power, or the vector magnitude). That discards the
relative phase **between** the three axes — the shape and handedness of the 3-D
orbit the limb segment traces at each tremor frequency. Since we established
that PD and ET overlap ~70 % along the **frequency** axis in two independent
cohorts (`reports/signal_processing_summary.md`), finer frequency resolution has
nothing left to resolve — but orbit *geometry* is a genuinely different axis.

There is also a correctness argument. Unit quaternions live on S³, so a
component-wise STFT of (w, x, y, z) is not a well-defined spectral operation: it
mixes the scalar and vector parts and ignores the norm constraint. We were never
doing that (we differentiate to angular velocity first, which is the standard
kinematic route), but it motivates asking what the *other* principled routes off
the manifold give us.

## Representations implemented

Code: `tremor/quaternion.py` (new modes) and `pdetn/quaternion_tf.py` (new).

| representation | what it keeps | mount-invariant? |
|---|---|---|
| `angular_velocity` ω = 2 q̇ q⁻¹, + STFT | frequency / amplitude | yes (magnitude) |
| `log_map` θ = 2 ln(q), + STFT | relative pose trajectory | yes, if referenced |
| **polarization** circularity, planarity | cross-axis phase, orbit **shape** | **yes** |
| **QSTFT** simplex/perplex + chirality | orbit **handedness** | no — axis-dependent |
| **gravity-referenced chirality** | signed handedness vs gravity | **yes** |

Two design decisions worth stating, because both are places this could have gone
quietly wrong:

1. **The log map is referenced, not absolute.** An absolute log map encodes the
   sensor's mounting pose — subject-specific nuisance a classifier will happily
   learn instead of tremor. `mode='log_map'` measures each sample relative to the
   recording's own median orientation, `θ(t) = 2 ln(q_ref* ⊗ q(t))`, so a fixed
   re-mounting rotation cancels exactly.

2. **Handedness needs an anatomical reference.** Circularity `‖Im(Z* × Z)‖/|Z|²`
   is invariant to mounting but discards the sign. The hypercomplex QSTFT keeps
   the sign but measures it against an axis `μ` fixed in the *sensor* frame, so
   it changes if the strap rotates. Referencing the spin pseudovector to the
   body-frame gravity direction — `χ = (s · ĝ)/|Z|²` — keeps the sign **and** is
   mount-invariant, because `s` and `ĝ` rotate together.

Verified on synthetic orbits: circularity 1.000 (circle) vs 0.000 (line), exact
rotation-invariance to 2e-16; signed handedness +1.000 / −1.000 for CCW / CW,
mount-invariance to 8e-16; QSTFT chirality ±0.866 for CCW/CW, 0.000 for linear.

## Finding 1 — orbit handedness is the strongest single PD-vs-ET feature found

Univariate screen, patient-level, OUT condition, PD (n=75) vs ET (n=15),
Mann-Whitney with rank-biserial effect. Uncorrected p-values; **screening only**,
nothing here was used to select classifier features.

| feature | effect | p | PD median | ET median |
|---|---|---|---|---|
| **upper_gchir_mean** (gravity-referenced handedness) | **−0.588** | **0.0004** | −0.056 | +0.023 |
| upper_q_chirality_peak (QSTFT) | +0.547 | 0.0009 | +0.085 | −0.100 |
| upper_q_chirality (QSTFT) | +0.534 | 0.0012 | +0.086 | −0.057 |
| upper_gchir_sign_frac | −0.468 | 0.0044 | 0.350 | 0.547 |
| upper_gchir_weighted | −0.449 | 0.0064 | −0.043 | −0.003 |
| hand_q_balance | −0.342 | 0.038 | −0.210 | −0.132 |
| lower_q_balance | −0.328 | 0.046 | −0.292 | −0.126 |

This is a **larger effect than any biomarker previously found** (best prior:
Higuchi fractal dimension at REST, −0.52). Two independent estimators — the
mount-invariant gravity-referenced one and the mount-dependent hypercomplex one
— agree in direction, which is what you would expect if the effect is real
rather than an artifact of one particular contraction.

Three properties of the effect:

* **It lives entirely in the sign.** Signed: effect −0.588, p = 0.0004.
  Sign-blind `|χ|`: effect +0.196, p = 0.234. So this is handedness, not
  amplitude — PD and ET orbit the **upper arm in opposite senses** during
  outstretch.
* **It is not driven by any one subject.** Leave-one-subject-out effect size
  stays in [−0.640, −0.558] across all 90 removals.
* **It is proximal.** The upper arm was previously the *worst* sensor
  (`reports/sensor_selection.md`, ET-F1 0.250). That is consistent: the proximal
  segment barely moves, so amplitude features are useless there — but the
  *direction* of that small rotation is clean.

### The confound I could not rule out

Handedness flips when you swap limbs. If PD and ET subjects differed
systematically in which arm was recorded, this effect would appear with no
physiology behind it. The local dataset does not label limb side, so I tested it
on the 2025 cohort, which records both limbs per subject (action 02 = right,
09 = left):

| subject | right (02) | left (09) | sign flip |
|---|---|---|---|
| ET_19 | −0.076 | +0.023 | yes |
| ET_20 | +0.059 | +0.089 | no |
| ET_21 | −0.079 | −0.068 | no |
| ET_22 | +0.039 | −0.044 | yes |
| ET_23 | −0.052 | +0.087 | yes |
| ET_26 | −0.071 | +0.126 | yes |

**4 / 6 flip — which is not significantly different from chance** (binomial
p = 0.34 for ≥4/6). So this neither confirms nor excludes a limb-side confound;
n = 6 is simply too small. It does show handedness is not a stable within-subject
trait across limbs, which weakens (but does not kill) the physiological reading —
tremor asymmetry is itself well documented.

One piece of reassurance: the upper-arm gravity direction, which captures how the
segment was held, is nearly identical for the two tremor groups
(PD [0.006, 0.795, −0.167] vs ET [−0.266, 0.761, −0.107]), so gross posture is
matched between exactly the classes being compared. N differs, as expected.

**Status: promising, not established.** Before this goes in a paper it needs
limb-side labels for the local cohort, or replication on a cohort where side is
known and balanced.

## Finding 2 — the classifier effect

Same two-stage LOSO, tuned ET threshold, subject bootstrap CI as everywhere else,
so these rows are directly comparable to `reports/decomposition_study.md`.

<!-- RESULTS_TABLE -->

## Finding 3 — the log map helps stage 1, hurts stage 2

`logmap_stft` reaches **N-vs-tremor 0.914** against the angular-velocity
baseline's 0.861 — the best stage-1 number in the study — while its ET-F1 drops
(0.216 vs 0.378). That makes sense: the log map retains low-frequency pose
excursion, which separates "is there tremor at all" well, but adds nuisance
variance to the already data-starved PD-vs-ET stage. It is a candidate for the
**stage-1 representation in a hybrid two-stage**, which the repo already supports
via `HybridTwoStage`.

## Reproduce

```python
from pdetn.quaternion_repr import univariate_screen, compare
univariate_screen(top=14)     # the biomarker screen
compare(n_boot=1000, n_perm=1000)   # the classifier comparison
```
Or run sections 8–8.3 of `pdetn/two_stage_comparison.ipynb`.
