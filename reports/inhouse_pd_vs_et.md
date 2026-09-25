# In-house PD vs ET: the ET patients lack the ET tremor, so no method recovers it

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

Not a model. The recordings contain no difference to find. Candidate reasons for
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
