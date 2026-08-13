# Bilateral (left/right limb) modelling

Motivated by the observation that PD signs typically begin unilaterally and stay
more severe on that side, so the **relationship between limbs** carries
information a single-limb model discards. NewData records both limbs per
subject (action codes 01–07 right, 08–14 left), so this is testable here.

`tfbench.small_nets.BilateralAttention` implements an interleaved-attention
encoder: both limb spectra are linearly projected to `d`, given sinusoidal
positional encodings and a **learned modality embedding** identifying the limb,
concatenated into one `2F` sequence, and passed through standard transformer
blocks. The `2F × 2F` attention map decomposes into within-limb diagonal blocks
and left–right off-diagonal blocks, giving the same interactions an explicit
dual-stream cross-attention would compute with one tied set of projection
weights (19 k params at d=32/N=2; 154 k at d=64/N=3).

**One deliberate deviation from the source design: interleaving is over
FREQUENCY, not time.** That is forced by what this data shows — tremor is
quasi-stationary over a 10 s window, and a BiLSTM over the time axis of a
spectrogram sits at chance (0.513) while the same family over the frequency axis
reaches 0.913. Attention over time would attend to an axis with little
structure.

## The limbs are wildly asymmetric

DRINK, PD-vs-ET, 19 PD / 6 ET (patients with both limbs recorded):

| model | bal-acc | AUC | precision | recall |
|---|---|---|---|---|
| BiLSTM, **left limb only** | 0.421 | 0.526 | 0.000 | 0.000 |
| BiLSTM, **right limb only** | **0.724** | 0.754 | **0.750** | 0.500 |
| BiLSTM, limbs averaged | 0.671 | **0.886** | 0.500 | 0.500 |
| BiLSTM, limbs concatenated | 0.557 | 0.877 | 0.500 | 0.167 |
| TCN, limbs concatenated | 0.478 | 0.763 | 0.200 | 0.167 |
| BilateralAttention d=32/64 | *pending — slow on CPU* | | | |

**The right limb carries the signal and the left is at chance.** That is a large
effect and it validates the premise: limb identity matters, and treating the two
as interchangeable is wrong.

## This has a consequence for every previous NewData result

`load_2025` defaults to `sides=("right", "left")`, and patient-level
aggregation **averages the two limbs**. So every NewData number reported in this
project — the task sweep, the kinetic-task finding, the merge tests — averaged an
informative limb with an uninformative one. Averaging still gave the best AUC
here (0.886), so it is not simply harmful, but right-limb-only gives much better
balanced accuracy and precision (0.724/0.750 vs 0.671/0.500).

Re-running the main NewData results with `sides=("right",)` is a cheap and
well-motivated next step.

## Caveats

* **6 ET.** Same constraint as everywhere; the left/right gap could partly be
  which limb happened to be more affected in these six.
* We do not have per-subject **more-affected-side** labels. The clinical CSV
  (`Clinical Study Subjects - Non-Identifiable Data.csv`, not uploaded) has
  `Tremor More Affected Hand` — with it, limbs could be ordered by severity
  rather than by anatomical side, which is what the asymmetry argument actually
  calls for.
* Whether attention beats simple averaging is still unmeasured.
