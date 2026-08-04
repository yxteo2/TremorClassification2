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

Patient-level LOSO, OUT condition, 151 patients (N=61, PD=75, ET=15).

| feature set | n_feat | macro-F1 | ET-F1 [95% CI] | N-vs-T | PD-vs-ET acc |
|---|---|---|---|---|---|
| omega_stft (baseline) | 702 | 0.651 | 0.378 [0.16, 0.56] | 0.861 | 0.756 |
| logmap_stft | 702 | 0.618 | 0.216 [0.06, 0.41] | **0.914** | 0.683 |
| polarization | 36 | 0.493 | 0.200 [0.07, 0.33] | 0.815 | 0.473 |
| qstft | 15 | 0.486 | 0.438 [0.21, 0.67] | 0.570 | 0.796 |
| **grav_chirality** | **15** | 0.664 | **0.471 [0.24, 0.65]** | 0.795 | 0.855 |
| polar + qstft + gchir | 66 | **0.696** | 0.412 [0.19, 0.61] | 0.887 | 0.815 |
| omega_stft + qstft | 717 | 0.652 | 0.412 [0.18, 0.60] | 0.848 | 0.766 |
| omega_stft + gchir | 717 | 0.648 | 0.359 [0.15, 0.54] | 0.874 | 0.731 |
| omega_stft + polar + qstft | 753 | 0.669 | 0.424 [0.19, 0.62] | 0.868 | 0.772 |
| omega_stft + polar + qstft + gchir | 768 | 0.686 | 0.471 [0.24, 0.65] | 0.861 | 0.795 |

**Read the ET-F1 column honestly: every CI overlaps the baseline's.** 0.471
[0.24, 0.65] against 0.378 [0.16, 0.56] is a real point improvement that is
*not* separated at 15 ET subjects — the same verdict every other feature-
engineering axis in this project has received. What is new is the efficiency:
`grav_chirality` reaches it with **15 features instead of 702**, and it is the
only feature block whose gain concentrates on the PD-vs-ET axis rather than
being diluted by stage-1 routing.

**Do not read the PD-vs-ET accuracy column as a win.** With 75 PD and 15 ET the
majority-class baseline is **0.833**, so 0.855 is barely above always-guess-PD.
Balanced accuracy is reported for the hybrids below instead.

### The hybrid that follows from the table

The two representations are specialists in opposite directions: `logmap_stft`
is the best stage-1 (N-vs-tremor 0.914, the best that number has been) and the
worst stage-2; `grav_chirality` is the reverse. `HybridTwoStage` uses a different
feature matrix per stage, so:

| stage 1 → stage 2 | macro-F1 | ET-F1 [95% CI] | N-vs-T | PD-vs-ET **bal-acc** | p |
|---|---|---|---|---|---|
| **logmap_stft → grav_chirality** | **0.718** | 0.462 [0.24, 0.64] | **0.914** | **0.745** | 0.0010 |
| logmap_stft → polar+qstft+gchir | 0.714 | 0.432 [0.22, 0.62] | 0.914 | 0.721 | 0.0010 |
| omega_stft → grav_chirality | 0.671 | 0.419 [0.21, 0.60] | 0.861 | 0.738 | 0.0010 |
| omega_stft → polar+qstft+gchir | 0.656 | 0.378 [0.17, 0.56] | 0.861 | 0.677 | 0.0010 |
| omega_stft → omega_stft (baseline) | 0.651 | 0.378 [0.16, 0.56] | 0.861 | 0.677 | 0.0010 |

**macro-F1 0.651 → 0.718** on the same patients, with stage 2 using 15 features.
This is the highest macro-F1 in the project (previous best: lower_arm single
sensor, 0.704 — though that config still holds the ET-F1 record at 0.516).

The gain decomposes cleanly, which is the part worth trusting:
* **stage 1**: 0.861 → 0.914, entirely attributable to the log map. Not a CI
  argument — it is a different, better-conditioned representation for "is there
  tremor at all", because it retains pose excursion that ω differentiates away.
* **stage 2**: balanced accuracy 0.677 → 0.745, from orbit handedness.

Caveat on stage 1: 0.914 is the OUT-condition, patient-level LOSO number and is
comparable **only** to the 0.861 baseline in the same table. It is not
comparable to the headline 0.884 [0.832, 0.929] in `reports/final_results.md`,
which is a different evaluation over 155 recordings.

Caveat on stage 2: it inherits the unresolved limb-side confound above.

## Reproduce the hybrid

```python
from pdetn.quaternion_repr import load_repr, patient_gchir_table
from pdetn.separability import patient_decomp_features
from pdetn.model import HybridTwoStage
from pdetn.evaluate import evaluate_hybrid

lm = load_repr(action="OUT", mode="log_map")
om = load_repr(action="OUT", mode="angular_velocity")
gr = load_repr(action="OUT", mode="gravity")
X1, y, pats = patient_decomp_features(lm, "stft", nperseg=256, nfft=256, noverlap=192)[:3]
X2 = patient_gchir_table(om, gr)[0]
evaluate_hybrid(lambda: HybridTwoStage("logreg", "logreg", tune_et_threshold=True),
                X1, X2, y, pats)
```

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

## Finding 4 — all three cohorts

| cohort | sensors | signal | classes | ET | orbit geometry | log map / gravity |
|---|---|---|---|---|---|---|
| LOCAL 2015 | 3 (hand/lower/upper) | quaternion | N/PD/ET | 15 | yes | yes |
| NewData 2025 | 3, both limbs | quaternion | **ET only** | 6 | yes | yes |
| PADS | 1 wrist | **gyro only** | N/PD/ET | 41 | yes | **no** |

PADS ships raw gyroscope with no orientation, so gravity-chirality and the log
map are not computable there. Polarization/QSTFT need only a 3-axis rate signal.
**That portability is itself a result:** orbit *shape* transfers to any gyro
dataset; orbit *handedness* requires orientation.

### The device-identity probe decides what may be pooled

NewData is ET-only, so if its device is identifiable a pooled model can learn
"new device ⇒ ET" and report a fake ET-F1 gain. Probe = predict which cohort a
patient came from, ET subjects only, subject-level LOSO:

| feature block | LOCAL-ET vs NewData-ET | verdict |
|---|---|---|
| stft-702 | **AUC 1.000** | confounded — must not pool |
| orbit geometry-66 | **AUC 1.000** | confounded — must not pool |
| **gravity-chirality-15** | **AUC 0.567** | at chance — safe to pool |

The mount-invariant chirality block is the **only** representation that carries
no device signature, which is exactly what its construction predicts: it is
built from rotation-invariant contractions referenced to gravity, so it encodes
neither device scale, nor axis convention, nor mounting.

### Pooling NewData (gravity-chirality only), PD vs ET

| cohort | PD | ET | AUC | bal-acc | ET-F1 [95% CI] |
|---|---|---|---|---|---|
| LOCAL only | 75 | 15 | 0.761 | 0.787 | 0.545 [0.35, 0.71] |
| + NewData right limb | 75 | 21 | 0.721 | 0.707 | 0.519 [0.34, 0.67] |
| + NewData left limb | 75 | 21 | 0.752 | 0.690 | 0.500 [0.32, 0.65] |
| **+ NewData both limbs** | 75 | **21** | 0.756 | 0.768 | **0.593 [0.42, 0.73]** |

Pooling does not hurt (all within CI) and the ET-F1 CI **narrows** from width
0.36 to 0.31 — which is what 6 extra ET subjects should buy. This is the first
legitimate ET cohort expansion in the project: unlike PADS, it survives the
device probe.

### PADS — independent replication, and what it says

| model | PD | ET | AUC | bal-acc | ET-F1 [95% CI] |
|---|---|---|---|---|---|
| LOCAL lower_arm, orbit geometry | 75 | 15 | 0.668 | 0.513 | 0.233 [0.06, 0.41] |
| **PADS wrist, orbit geometry** | 296 | **41** | 0.715 | 0.690 | **0.381 [0.27, 0.48]** |

Orbit geometry carries real PD-vs-ET signal on PADS, with 41 ET subjects and a
tight CI — and it beats the previously reported PADS-only ET-F1 of 0.262
(`reports/track3_external_data.md`). That is genuine independent support for the
*general* claim that orbit geometry is informative.

**But the specific features do not replicate.** Not one feature reaches p<0.05
in both cohorts with the same sign, and the clearest shared one is *inverted*:

| feature | LOCAL effect | LOCAL p | PADS effect | PADS p |
|---|---|---|---|---|
| plan_bandmean | +0.319 | 0.053 | **−0.193** | 0.046 |
| plan_peak | +0.140 | 0.398 | −0.350 | 0.0003 |
| peak_share | −0.054 | 0.745 | −0.455 | <0.0001 |
| q_balance | −0.328 | 0.046 | −0.043 | 0.657 |

And the device probe on ET subjects gives **AUC 0.959** — LOCAL and PADS are
strongly separable, so pooling them remains confounded, consistent with every
earlier PADS result in this project.

**The honest reading.** Orbit geometry is informative in both cohorts, but
through different features with different signs. The strongest LOCAL finding —
upper-arm handedness — is **not testable on PADS at all**, because PADS has no
upper-arm sensor and no orientation. So PADS neither confirms nor refutes the
headline claim; it only shows the general family of features has signal.

### What would settle it
1. **Limb-side labels for the local cohort** — the one unresolved confound on
   the headline result.
2. **More NewData** — it is the only cohort that passes the device probe, and
   ET is still the binding constraint at 21.
3. An external cohort with **3 sensors and orientation**, which no public
   dataset currently provides.
