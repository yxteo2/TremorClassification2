# NewData's epoch selection is unnecessary, and that is worth more than a gain

## What was tested

`verify_data.py` found the three cohorts consistent on everything it can check —
no leakage, no duplicate recordings, no id collisions, amplitude scales within
1.4x. **One preprocessing step is applied to a single cohort.** NewData's ~38 s
`Free_Form` exports are cropped to a 10 s window by `select_task_epoch`, which
picks the window with the **most 3-15 Hz power** — a criterion correlated with
the thing being classified.

That deserved a measurement, because this project has explicitly refused a
competitor's number for a stronger version of the same idea
(`ceiling_and_preprocessing.md`: *"a 2026 preprint claiming 87 % on PADS uses
class-dependent window overlap — preprocessing that reads the label; not a
comparator"*). The loader offers a tremor-blind alternative, `select_steady_epoch`,
which scores windows by how still the limb is held and never looks at the tremor
band.

The justification for preferring tremor-selection was *"better on both axes
(N-vs-Tremor 0.787 vs 0.714 at OUT)"* — measured **on NewData alone, on the
binary axis, before the three preprocessing fixes**, and never checked in the
merged 3-class model the project reports.

## Result — 20 splits, 2015 and PADS untouched in every arm

| arm | precN | precPD | precET | macroP | macroF1 | recET | nETpred |
|---|---|---|---|---|---|---|---|
| reported (tremor-selected) | 0.642 | 0.649 | 0.648 | 0.646 | 0.590 | 0.475 | 8.95 |
| steady (tremor-blind) | 0.637 | 0.622 | **0.684** | **0.648** | 0.586 | 0.425 | **6.85** |
| **none (whole 38 s recording)** | 0.645 | **0.651** | 0.648 | **0.648** | **0.598** | 0.475 | 8.40 |

**Paired vs the reported arm — null on every performance column:**

| arm | precET | macroP | nETpred |
|---|---|---|---|
| steady | +0.036 [−0.024, +0.102] | +0.002 [−0.026, +0.029] | **−2.10 [−4.05, −0.55]** \* |
| none | +0.000 [−0.060, +0.054] | +0.002 [−0.022, +0.026] | −0.55 [−2.30, +1.10] |

The single significant cell is `steady`'s **nETpred**: it makes 2.1 fewer ET
predictions per split, with recET −0.050 and precET +0.036. That is a **threshold
shift, not an improvement** — predicting ET less often raises precision and
lowers recall, which is what invariant 5's companion column exists to expose.

## The finding: a 6x change in signal purity is invisible to the model

The three arms differ enormously in how tremor-dominated their input is:

    none (whole 38 s)      in-band 3-15 Hz fraction   0.118
    steady-selected                                   0.571
    tremor-selected                                   0.697

**Six times the in-band purity, and macroP moves by 0.002.**

The explanation is that the pipeline **already band-limits to 3-15 Hz** before
binning. The 9.9-11.8 % figure counts power across the whole spectrum, but the
model never sees anything outside the band. Set-up and settling motion is
low-frequency and is filtered out regardless — so epoch selection is largely
*redundant with band-limiting*, and was solving a problem the pipeline had
already solved.

This also adds to the picture from `spectral_representation.md`, where a 2x SNR
improvement (39.7 → 80.5) produced macroP −0.000 because sum-normalisation
discards it. **The model reads spectral shape, and is close to indifferent to
signal purity.**

## Predictions, scored

1. *"Null on the merged model."* — **held.** Every performance column null.
2. *"`none` will be worst of the three, because a 9.9 % in-band fraction is
   mostly set-up motion."* — **failed.** `none` is not worst; it ties on macroP
   and is marginally best on precPD and macroF1. The reasoning ignored that the
   pipeline band-limits before the model sees anything, so out-of-band motion
   was never a threat.
3. *"If tremor and steady tie, prefer steady on principle."* — they do tie, and
   the conclusion is now stronger than the prediction anticipated: **`none` ties
   too**, so the whole selection step can go rather than merely being swapped
   for a defensible one.

## Standing

* **Drop the epoch selection.** `segment=False` ties the current default on every
  performance column while removing the only cohort-specific preprocessing step
  in the project. That is a simplification, not a trade.
* **It removes a methodological objection at zero cost.** Selecting the most
  tremor-dominated window is not label-reading, but it is selection on a
  criterion correlated with the class, and a reviewer is entitled to ask. Not
  doing it is strictly easier to defend than doing it and arguing it was
  harmless.
* **Do not read this as "the earlier NewData result was wrong."** That was
  NewData alone on the binary axis; this is the merged 3-class model. Both can
  hold — the merged model is dominated by PADS and simply cannot see a change
  confined to 14 % of patients.
* **Keep the change out of the headline.** macroP 0.646 → 0.648 is inside noise;
  the reason to adopt is defensibility and simplicity, not the number.
