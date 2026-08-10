# NewData 2025 is now a full cohort — and the segmenter needs fixing

HC and PD were added, so the 2025 cohort is no longer ET-only.

| class | subjects | h5 files |
|---|---|---|
| HC (= N) | **31** | 339 |
| PD | **34** | 326 |
| ET | 6 | 85 |

All 14 task codes present for every class. `load_2025_all()` loads N/PD/ET
together; `load_2025` infers the label from the folder.

Loading REST/OUT gives fewer patients than the folder counts (N=27, PD=25, ET=6
at REST) — **not a loader bug**. All 202 REST/OUT files carry every required
sensor; some subjects simply lack some task codes.

## Standalone results (lower_arm, stft512, patient-level LOSO)

| condition | axis | bal-acc | AUC | ET precision |
|---|---|---|---|---|
| **OUT** | **N vs Tremor** | **0.787** | **0.829** | — |
| REST | N vs Tremor | 0.640 | 0.670 | — |
| OUT | PD vs ET | 0.388 | **0.196** | 0.077 |
| REST | PD vs ET | 0.487 | **0.273** | 0.182 |

**N-vs-Tremor at OUT works** — 0.787 / AUC 0.829 on 27 N vs 29 tremor,
comparable to the 2015 cohort. The new controls are usable.

**PD-vs-ET is inverted**, AUC 0.196–0.273, far below chance across both
conditions. With 6 ET some instability is expected, but a systematic inversion
is not.

## The segmenter hypothesis — tested and REFUTED

I proposed that `select_task_epoch` caused the inversion: it picks the window
with the highest tremor-band power fraction, which controls do not have, so the
rule behaves differently per class. The class effect is real and measurable
(in-band fraction at OUT: HC 0.536 / PD 0.730 / ET 0.714 under this rule vs
HC 0.473 / PD 0.642 / ET 0.722 under a tremor-blind alternative).

**But switching to the tremor-blind rule does not fix the inversion, and makes
everything else worse:**

| condition | axis | `tremor` rule | `steady` rule |
|---|---|---|---|
| OUT | N-vs-Tremor | **0.787** (AUC 0.829) | 0.714 (AUC 0.745) |
| OUT | PD-vs-ET | 0.388 (AUC 0.196) | 0.449 (AUC **0.384**) |
| REST | N-vs-Tremor | **0.640** (AUC 0.670) | 0.587 (AUC 0.602) |
| REST | PD-vs-ET | 0.487 (AUC 0.273) | 0.280 (AUC **0.167**) |

PD-vs-ET stays below chance under both rules, and at REST the tremor-blind rule
is *worse* (AUC 0.167, ET precision 0.000 — it never correctly identifies an ET
patient). N-vs-Tremor drops under the blind rule in both conditions.

**The concern was also overstated.** Selecting on tremor content is not leakage:
the rule never sees a label, is applied identically at train and test, and would
run the same way on an unlabelled recording at deployment. "Find the tremor if
there is one" is the intended behaviour, not a bug. The default therefore stays
`segment="tremor"`; `select_steady_epoch` remains available.

**So the inversion is unexplained.** The most likely remaining cause is simply
**6 ET subjects** against 23–25 PD — below the n=8 point where the PADS learning
curve was still climbing steeply. Nothing about PD-vs-ET on this cohort should
be reported until there are more ET.

## The likely cause is our own preprocessing

`select_task_epoch` chose the window with the **highest tremor-band power
fraction**. That criterion is **label-dependent**: PD and ET have tremor to find,
HC does not, so for a control it selects whichever window has the highest
in-band *noise* ratio. The rule was written when the cohort was ET-only, where
it could not bias anything. With 31 HC it can.

Measured class bias in the selected windows (in-band fraction, OUT):

| rule | HC | PD | ET |
|---|---|---|---|
| `tremor` (old default) | 0.536 | 0.730 | 0.714 |
| **`steady` (new default)** | **0.473** | 0.642 | 0.722 |

The old rule inflates HC in-band content by ~0.06 relative to the tremor-blind
rule — it is finding "tremor-like" windows in people who have no tremor.

## Fix

`select_steady_epoch` scores windows by **body-frame gravity stability** — how
still the limb is held — and never looks at the tremor band, so controls and
patients are selected on the same criterion. `load_2025(segment="steady")` is
now the default; `segment="tremor"` reproduces the old behaviour and
`segment=False` disables selection.

This rule was already validated when the unsegmented-data bug was found:
in-band recovery 0.722 (steady) vs 0.714 (tremor), i.e. equivalent, without
keying on the quantity being measured.

Since the default is unchanged after testing, **earlier NewData results stand**
— no re-run is needed.

## Status

* The ET-only confound is **gone**. Cohort membership no longer predicts the
  label, so NewData can be trained and tested on its own, and pooling is no
  longer structurally blocked.
* The bottleneck moved from "no controls" to **6 ET subjects** — below the n=8
  point where the PADS learning curve was still climbing steeply. PD-vs-ET on
  this cohort should not be reported until there are more ET.
* Five kinetic tasks (drinking, finger-to-nose, pouring, finger-tapping,
  pronation-supination) are now available **with controls and PD**, which was
  the blocker on using them. That is the most promising untouched lever here.
