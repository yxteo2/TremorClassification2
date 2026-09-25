# In-house PD vs ET: no difference at OUT, a textbook one at REST, and PADS points the other way

> **Update (follow-up audit, below).** Steps 1-3 describe the **OUT** task only.
> The signal chain was then checked against independent code and found correct;
> the 0.314 turned out to be one unlucky CV partition; and **2015 REST does
> separate PD from ET** (AUC ~0.65-0.70 on all three sensors, above chance), in
> the textbook direction -- opposite to PADS. See "Follow-up".

## The complaint

In `01_tremor_characteristics.ipynb`, PD vs ET from six frequency characteristics
reads AUC **0.314** on 2015 (15 ET) and **0.065** on NewData (6 ET). Both look
worse than random. PADS (28 ET) reads **0.793**.

## Step 1 — the in-house numbers are no signal, not a reversed one

Each AUC compared with the same pipeline run on shuffled labels:

| cohort | ET | AUC | shuffled-label 95 % range | p |
|---|---|---|---|---|
| 2015 | 15 | 0.314 | [0.291, 0.659] | 0.94 |
| NewData | 6 | 0.065 | [0.156, 0.797] | 0.997 |
| PADS | 28 | **0.793** | [0.320, 0.646] | **< 0.003** |

2015 sits inside its own chance range. NewData falls *below* its range: with 6 ET,
each test fold holds about one ET patient, each held-out patient drags the
training ET mean away from itself, and a heterogeneous group of six is pushed
toward the PD side. That is a small-sample effect, not a model that is reliably
backwards. The project already treats NewData as training-only.

## Step 2 — no combination recovers a signal in-house

| PD vs ET, 6 characteristics | ET | AUC | null 95 % | p |
|---|---|---|---|---|
| in-house pooled, raw | 21 | 0.320 | [0.271, 0.639] | 0.94 |
| in-house pooled, z-scored within cohort | 21 | 0.304 | [0.274, 0.660] | 0.96 |
| all three pooled, z-scored within cohort | 49 | 0.595 | [0.362, 0.592] | 0.025 |
|   scored on in-house patients only | 21 | 0.378 | [0.331, 0.641] | 0.91 |
|   scored on PADS patients only | 28 | 0.751 | [0.364, 0.646] | < 0.005 |
| PADS-trained model applied to in-house | 21 | 0.472 | [0.362, 0.636] | 0.65 |

The pooled result is significant only through its PADS patients.

**Sensor choice does not explain it.** Both in-house cohorts carry hand, lower-arm
and upper-arm sensors (the pipeline uses lower arm, roughly wrist-equivalent). In-house pooled PD vs ET: hand 0.491 (p = 0.49), lower arm 0.304 (p = 0.96), upper arm
0.517 (p = 0.40).

**Nor is it only sample size.** At 21 ET the chance range tops out near 0.64-0.66.
A PADS-strength signal (0.75-0.79) would clear that easily. The in-house point
estimates are 0.30-0.38, so the signal is genuinely weaker, not just unconfirmed.

## Step 3 — the reason: in-house ET patients do not have ET tremor

Model-free, per patient, ET vs PD within each cohort (medians, Mann-Whitney p):

| cohort | tremor RMS (log10) | peak sharpness | peak frequency (Hz) |
|---|---|---|---|
| **PADS** | **−1.38 vs −1.72** (p < 0.01) | **10.4 vs 4.8** (p < 0.01) | **5.8 vs 6.9** (p < 0.01) |
| **2015** | −1.69 vs −1.66 (p = 0.46) | 4.9 vs 4.9 (p = 0.87) | 6.6 vs 6.6 (p = 0.64) |
| NewData | −1.78 vs −1.83 (p = 0.51) | 4.1 vs 6.4 (p = 0.38) | 7.3 vs 7.7 (p = 0.63) |

PADS ET patients have the textbook ET tremor: about **twice the amplitude** of PD,
twice as sharply peaked, and slower. **In-house ET patients match in-house PD on
all three.** PD is similar across cohorts; it is the ET groups that differ, with
PADS ET tremor roughly twice as strong as in-house ET tremor.

That also explains why PADS does not transfer (AUC 0.578 in `own_data_reality_check.md`,
0.472 here): a model learns "ET = strong, sharp, slow tremor", and the in-house ET
patients do not have it.

## What would change this

Not a model, at OUT: the postural recordings contain no difference to find.
(At REST they do -- see Follow-up.) Candidate reasons for
the phenotype gap, none measurable from these data: milder or earlier-stage
in-house ET, medication state at recording, recruitment (PADS may favour
pronounced tremor), or diagnostic protocol.

* **Record ET severity** (e.g. TETRAS) for in-house patients. If in-house ET is
  mild, the honest claim becomes "PD vs ET is separable for pronounced ET", which
  is clinically accurate and defensible.
* **Recruit ET patients with pronounced tremor** when collecting toward the
  `data_plan.md` target, and record medication timing.
* **Report in-house PD vs ET with its chance range**, never as a bare point
  estimate. The notebook now does this, and skips cohorts with fewer than 10 ET.

## Changes made

* `frequency.characteristics.classify` gained `n_perm` (shuffled-label null for
  the full feature set) and `min_pos` (skip a contrast with too few positives).
  Defaults keep the script's output unchanged.
* `01_tremor_characteristics.ipynb` uses both for PD vs ET, adds a cell showing
  the ET-vs-PD phenotype comparison per cohort, and corrects its conclusions:
  the stale PADS precision (0.924 → 0.916), and a bullet claiming
  instantaneous-frequency stability "works" on 2015 at AUC 0.652, which is
  inside the chance range at 15 ET and predates the trajectory end-point fix.

## Follow-up: is the signal processing wrong? No. Is OUT the right task? No.

Asked to "check the whole process" because 0.31 looked implausible.

**Signal chain, checked against code it does not share**

| check | result |
|---|---|
| angular velocity vs independent SciPy `Rotation` (central difference, 80 files x 3 sensors) | relative error median 0.00009, max 0.0011 |
| quaternion convention `xyzw` vs `wxyz` (column 0 is the largest component in 45 % of rows, column 3 in 50 %) | band power changes by < 0.5 %; dominant frequency differs in 0.4 % of files |
| duplicate files (whole-file and 2 s window hashes, all 799 2015 files) | none |
| repeated samples (sample-and-hold) | none |
| unit norm of every raw quaternion file, all tasks | **4 failures: `N 2` REST and WING, \|q\| ~ 9.7-9.8** -- gravity in m/s^2, i.e. accelerometer data saved as quaternions |

The four `N 2` files are controls in REST/WING; OUT is clean. They are reported
by the new `verify_data` check 13, not dropped; the fix is to re-export them.

**Why 0.31.** On 2015 OUT every feature has a raw, no-CV AUC of 0.46-0.54: ET
and PD have the same median on all six. The notebook printed one 5-fold
partition, seed 0. The same one-feature model reads **0.32-0.52 depending only on
the CV seed**, and seed 0 was the lowest. With 10 partitions averaged, the full
feature set reads 0.288 against a null of [0.334, 0.662] on the lower-arm
sensor, but 0.385 and 0.537 on hand and upper arm (both inside their nulls). The
below-0.5 drift is the known cross-validation behaviour with no signal when the
smaller class sits inside the larger one's spread (ET's ranges are narrower than
PD's on max_freq and peak_sharp); the in-sample AUC is only 0.604.
`classify` now takes `n_repeats`; the notebook uses 10.

**At rest the in-house cohort separates PD from ET**

PD vs ET, six characteristics, CV averaged over 10 partitions, 200-permutation
null built from the same averaged statistic (16 ET, 75 PD):

| 2015 REST sensor | AUC | null 95 % | p |
|---|---|---|---|
| hand | 0.683 | [0.278, 0.663] | 0.015 |
| lower arm | 0.650 | [0.301, 0.649] | 0.025 |
| upper arm | 0.670 | [0.287, 0.675] | 0.035 |

The best subsets reach ~0.70-0.73. The driver is frequency: PD rest tremor is
slower than ET's, the textbook ordering. **PADS goes the other way**:

| | PD max_freq | ET max_freq | raw AUC, ET higher |
|---|---|---|---|
| 2015 OUT | 6.64 | 6.64 | 0.54 |
| 2015 REST | 5.57 | 6.05 | 0.64 |
| PADS StretchHold | 6.93 | 5.81 | 0.33 |
| PADS Relaxed | 6.05 | 4.83 | 0.30 |

NewData REST (6 ET, not evaluable) leans the in-house way: PD 5.27, ET 5.52.

So the two sources teach **opposite frequency rules**: in-house, ET is faster
than PD at rest; in PADS, ET is slower at both tasks (older or more severe ET,
whose frequency falls with severity, is one candidate). That is consistent with
the PADS-trained model scoring chance in-house (0.47), and it means pooling
cannot help the in-house axis through frequency.

**What this changes**

* The in-house pipeline (`common/cohorts.py`) uses OUT only for 2015 and
  NewData -- the task where in-house ET and PD are indistinguishable.
  `rest_postural_contrast.md` and `task_averaging.md` found REST hurt the
  *merged* model, but that is dominated by PADS, where REST runs the other way.
  In-house REST was then tested in the deep model (`inhouse_rest.md`): a
  +0.022 in-house ranking gain that a shuffled control mostly matches; not adopted.
* 16 ET is thin. Three sensors all above chance is better than one, but this
  needs confirmation on more in-house ET before it is a claim.
