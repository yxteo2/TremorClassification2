# Dropping the hardest majority patients — RETRACTED, now null in both directions

> ## RETRACTION — the significant results below do not survive the preprocessing fixes
>
> Re-run on the corrected pipeline (axis, contiguous Q-factor, guarded IF
> trajectory), with `difficulty()`, `prune()`, `fit_eval()`, the splits and the
> seeds **bit-identical to the original run** — only the upstream features
> changed:
>
> | | precET, then | precET, now | macroP, then | macroP, now |
> |---|---|---|---|---|
> | hard-drop 5 vs baseline | **−0.081 \*** | +0.024 n.s. | **−0.032 \*** | +0.009 n.s. |
> | hard-drop 15 vs baseline | −0.061 | +0.056 n.s. | **−0.030 \*** | +0.015 n.s. |
> | hard vs random, k=5 | **−0.065 \*** | −0.002 n.s. | **−0.030 \*** | −0.005 n.s. |
> | hard vs random, k=15 | −0.017 | +0.067 n.s. | −0.013 | +0.021 n.s. |
>
> **Every significant result reverses sign and becomes null.** The baseline moved
> too (precET 0.685 → 0.648), which is the signature of the three fixes rather
> than of this experiment.
>
> **What can no longer be claimed:** that the hardest majority patients are
> boundary-defining, or that dropping them is worse than dropping random ones.
> **What cannot be claimed instead:** that dropping them helps — hard-drop 15 is
> precET +0.056 [−0.021, +0.136] with a split-level **win rate of 0.45**, the
> exact positive-mean/sub-0.5-win-rate pattern invariant 5 exists to catch.
> **The experiment is now uninformative in both directions.**
>
> A methodological note worth more than the result. The retracted effect was
> significant but *small* — macroP −0.032, against the ~0.04 that 20 splits
> resolves. The project already knew three ~0.03 differences had flipped sign on
> doubling the splits; this is a fourth, flipped by **fixing a defect upstream
> instead**. A significant result sitting at the resolution floor is fragile to
> anything that moves the pipeline, not just to more splits.
>
> The pre-fix analysis is kept below unchanged, as the record of what was
> believed and why.


## The idea and why it looked promising

Remove the N and PD patients the model finds hardest — presumed mislabelled,
atypical or uninformative — so they stop dragging the boundary across the
minority class. N and PD are abundant (167 and 188); ET (49) is never touched.

**CORRECTION.** This experiment's premise cited "20 PADS records labelled
parkinsonian are Atypical Parkinsonism" as live contamination in the PD class.
**That is wrong.** The extracted manifest's `raw_label` takes exactly three
values — "Parkinson's" (552 rows), "Healthy" (158), "Essential Tremor" (56) — so
`extract_pads.py`'s strict exact-match has *already* excluded the atypical
records, which live in PADS's differential-diagnoses group and are dropped except
for ET. There is no known majority-class label contamination left to remove, which
weakens the motivation for pruning but does not change any measured result here.

There was a specific reason to expect majority-class label noise here.
`common/extract_pads.py` records that **20 PADS records labelled parkinsonian are
Atypical Parkinsonism** — PSP, MSA, vascular parkinsonism, not idiopathic PD.
They carry the PD label without PD's tremor, and are exactly what such a method
should find.

## Protocol

Difficulty = 1 − p(true class) from **5-fold inner CV on the training fold
alone**. Validation is left intact because it tunes the class priors; test is
never touched. ET is never dropped. Merged 3-class protocol, 20 splits, paired.

Every hard-drop arm is matched by a **random-drop** arm removing the same count
from the same classes, because dropping the hardest *k* confounds **which**
patients leave with the fact that **k majority patients left at all** — and class
balance already moves ET precision hard here (uncapped PADS drives precET from
0.612 to 0.221).

## Result

| arm | precN | precPD | precET | macroP | macroF1 |
|---|---|---|---|---|---|
| **k=0 (baseline)** | 0.639 | 0.655 | **0.685** | **0.660** | 0.593 |
| hard-drop 5 | 0.633 | 0.647 | 0.604 | 0.628 | 0.585 |
| hard-drop 15 | 0.626 | 0.640 | 0.624 | 0.630 | 0.586 |
| random-drop 5 | 0.645 | 0.658 | 0.669 | 0.657 | 0.587 |
| random-drop 15 | 0.630 | 0.658 | 0.641 | 0.643 | 0.594 |

paired vs baseline:

| arm | precET | macroP |
|---|---|---|
| hard-drop 5 | **−0.081 [−0.165, −0.009]** * | **−0.032 [−0.057, −0.011]** * |
| hard-drop 15 | −0.061 [−0.129, +0.001] | **−0.030 [−0.055, −0.008]** * |
| random-drop 5 | −0.016 [−0.056, +0.026] | −0.002 [−0.016, +0.011] |
| random-drop 15 | −0.044 [−0.102, +0.007] | −0.017 [−0.039, +0.003] |

**hard vs random at the same k — the comparison that decides it:**

| | precET | macroP |
|---|---|---|
| k=5 | **−0.065 [−0.134, −0.008]** * | **−0.030 [−0.053, −0.011]** * |
| k=15 | −0.017 [−0.093, +0.050] | −0.013 [−0.044, +0.015] |

## Reading it

**Removing patients is not the problem; removing the *right* ones is.** Random
removal of 10 majority patients costs essentially nothing (macroP −0.002, precET
−0.016, neither significant). Removing the 10 *hardest* costs precET −0.081 and
macroP −0.032, both significant — and is **significantly worse than random at the
same k**.

So the hypothesis is not merely unsupported, it is **inverted**. The hardest N and
PD patients are not mislabelled noise; they are the **boundary-defining**
examples. They are hard precisely because they sit near the PD/ET frontier, and
deleting them lets the boundary drift into ET territory — which is where ET
precision is lost.

The census supports that. Across 20 splits the rule spreads its choices over all
three cohorts and both classes — 5 PADS, 3 from 2015, 2 NewData in the top ten,
mixing N and PD — rather than concentrating on the PADS Atypical Parkinsonism
subgroup it was hypothesised to find. It is selecting *borderline* patients, not
*mislabelled* ones, and this data gives the method no way to tell those apart.

Note also that the effect **shrinks** from k=5 to k=15 (hard-vs-random precET
−0.065 → −0.017). Once 30 majority patients are gone, the undersampling term
starts to dominate and the two rules converge. The damage is concentrated in the
first, most boundary-adjacent patients removed.

## Standing

* **Do not prune majority-class training patients by difficulty.** Significantly
  worse than both keeping them and removing random ones.
* Removing a *small* number of majority patients at random is free (macroP
  −0.002 at k=5), which is worth knowing if training cost ever matters — but it
  buys nothing either.
* The known Atypical Parkinsonism contamination in PADS is real, but **difficulty
  scoring does not find it**. Removing it would need the diagnosis field, not a
  model-derived score.
* The mirror experiment — dropping the *easiest* majority patients — follows
  directly, and the prediction stands: **if hard examples are boundary-defining,
  dropping easy ones should be harmless.** An earlier version of this line
  promised it in `prune_training_easy.md`; **no such report was ever written**,
  and the reference is removed rather than left dangling. The easy-drop arms are
  implemented in `prune_training.py` itself and are reported below.


## The post-fix run, in full (20 splits, 7 arms, one baseline, shared controls)

Both directions now run together, which the pre-fix version could not do: an
earlier revision had replaced the hard-drop arms with easy-drop ones and never
run them, so the script had stopped reproducing this report.

| arm | precN | precPD | precET | macroP | macroF1 |
|---|---|---|---|---|---|
| k=0 (baseline) | 0.642 | 0.649 | 0.648 | 0.646 | 0.590 |
| hard-drop 5 | 0.636 | 0.657 | 0.671 | 0.655 | 0.589 |
| hard-drop 15 | 0.637 | 0.644 | **0.703** | **0.661** | 0.593 |
| easy-drop 5 | 0.642 | 0.639 | 0.665 | 0.649 | 0.587 |
| easy-drop 15 | 0.655 | 0.645 | 0.670 | 0.657 | **0.596** |
| random-drop 5 | 0.650 | 0.657 | 0.673 | 0.660 | 0.590 |
| random-drop 15 | 0.642 | 0.642 | 0.636 | 0.640 | 0.586 |

**Nothing is significant on any column, in either direction, against either the
baseline or the matched random control.** Every win rate sits at or below 0.60,
and the largest mean (hard-drop 15, precET +0.056) carries a win rate of 0.45.

### The easy-drop prediction, on record since this report was first written

> if hard examples are boundary-defining, **dropping easy ones should be
> harmless** — at worst a mild undersampling cost the matched random-drop arm
> also pays.

**Held**: easy vs its matched random control is macroP −0.011 at k=5 and +0.016
at k=15, null both times. But it held **vacuously**. The contrast it was designed
to complete has evaporated: with the hard-drop half now null too, "easy is
harmless while hard is harmful" has become "both are null", and the experiment
no longer separates the accounts. A prediction can hold and still tell you
nothing once its premise is withdrawn.

### The census still says what it said

The most-dropped patients at k=15 spread across all three cohorts and both
majority classes — NewData, PADS and 2015, mixing N and PD — rather than
concentrating anywhere. Difficulty scoring selects *borderline* patients, not
*mislabelled* ones, and this data still gives the method no way to tell those
apart. That reading never depended on the retracted significance.
