# The win is the discarded recordings, not the learned pooling

**What prompted this.** Every model in this project reduces a patient to one
spectrum before the network runs (`np.mean(rows[p], 0)` in
`frequency/tables.py`), and merges on a single postural task per cohort. Two
things followed from that:

1. The aggregator is hard-coded, and there was evidence it was the wrong one —
   the standing note *"averaging two PADS tasks | precET 0.585 vs 0.612"* says a
   uniform mean over tasks measured worse than one task alone.
2. The pipeline discards most of what it loads. Of ~3,000 recordings, the merged
   table uses **768**.

Multiple-instance learning addresses both: the bag is the patient, the instances
are that patient's recordings, and attention learns the weights instead of
assuming a uniform mean.

Run: `python -m experiments.mil_recordings`. Merged cohort, n=404 (167 / 188 /
49), 20 splits, 3 seeds, identical 16-bin log multitaper representation and
identical instance encoder in every arm, so **only the aggregation differs**.
All-task bags hold **2,291** recordings against the postural 768 — a 3× increase.

## Result

| arm | precN | precPD | precET | macroP | macroF1 |
|---|---|---|---|---|---|
| avg spectrum, postural only (current) | 0.658 | 0.630 | 0.510 | 0.599 | 0.551 |
| **avg spectrum, ALL tasks** | 0.683 | **0.687** | **0.547** | **0.639** | **0.600** |
| MIL mean-pool, ALL tasks | 0.686 | 0.681 | 0.524 | 0.631 | 0.589 |
| MIL max-pool, ALL tasks | 0.680 | 0.621 | 0.400 | 0.567 | 0.537 |
| MIL gated attention, ALL tasks | 0.679 | 0.635 | 0.430 | 0.581 | 0.551 |

**vs the current pipeline:**

| arm | precPD | precET | macroP | macroF1 |
|---|---|---|---|---|
| avg spectrum, ALL tasks | **+0.057 [+0.000, +0.106]** * | +0.037 [−0.068, +0.145] | +0.040 [−0.006, +0.083] | **+0.049 [+0.023, +0.076]** * |
| MIL mean-pool | **+0.052 [+0.013, +0.088]** * | +0.015 | +0.032 | **+0.037** * |
| MIL max-pool | −0.009 | **−0.110 [−0.216, −0.008]** * | −0.032 | −0.015 |
| MIL gated attention | +0.006 | −0.080 | −0.018 | −0.000 |

**vs uniform averaging of all tasks — the comparison that isolates learned
pooling from extra data:**

| arm | precPD | precET | macroP |
|---|---|---|---|
| MIL mean-pool | −0.005 | −0.022 | −0.008 |
| MIL max-pool | **−0.065** * | **−0.147 [−0.263, −0.018]** * | **−0.072** * |
| MIL gated attention | **−0.051** * | **−0.117 [−0.242, −0.002]** * | **−0.057** * |

## Reading it

* **Using the discarded tasks helps.** Uniform averaging over every recording
  beats the postural-only pipeline on macro F1 (+0.049 *) and PD precision
  (+0.057 *), with macro precision +0.040 whose interval only just includes zero.
  Nothing about the model changed — only which recordings were averaged.
* **Learned pooling does not.** Attention and max-pooling are both *significantly
  worse* than the uniform mean they were meant to improve on (precET −0.117 and
  −0.147). Mean-pooling over embeddings is indistinguishable from mean-pooling
  over spectra, so even moving the average from before the encoder to after it
  buys nothing.
* **Max-pooling fails hardest on ET**, which is the diagnostic detail. "The
  patient is as abnormal as their most abnormal recording" is a reasonable prior
  for detecting *presence* of tremor; it is a bad one for distinguishing two
  tremor types, where the informative thing is the shape of the typical
  recording, not the extreme one.

This is the fourth mechanism in this project to fail for the same reason. At 49
ET patients, anything with parameters to fit on top of the aggregation — 
attention weights here, encoder weights under fine-tuning, extra feature columns
under concatenation — costs more than it returns. **The one lever that keeps
working is giving the existing model more data.**

## A tension with a standing note worth flagging

The skill file records *"averaging two PADS tasks | precET 0.585 vs 0.612"* as
evidence that averaging tasks hurts. Here averaging **all** tasks helps. The two
are not directly comparable — that was PADS-only with two tasks, this is the
merged cohort with up to 18 recordings per patient — but the directions differ,
and this measurement is the better-powered of the two (20 splits, paired, n=404).
The earlier note should be read as specific to its setting, not as a general
result about task averaging.

## Scope

This was measured on a **spectrum-only** model with no descriptors, asymmetry or
trajectory stream, so its baseline is macroP 0.599 rather than the reported
0.660. A gain on a stripped-down model need not survive on the full one — the
extra tasks may supply information the descriptor and trajectory streams already
carry. `experiments/alltasks_final.py` tests exactly that on the reported model.

## A leakage bug caught while building this

2015 encodes the action into the subject id: `ET 10_OUT`, `ET 10_REST` and
`ET 10_WING` are one person. Keying bags on the raw subject made each patient
three rows — n inflated from 151 to 440 for that cohort — and would have put the
same patient in both train and test, violating the project's first invariant.

**No previously reported result is affected**: every other experiment loads a
single action, where the raw ids are already one row per patient. The bug only
becomes reachable when tasks are combined, which nothing did before this.
