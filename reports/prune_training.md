# Dropping the hardest majority patients makes things worse, not better

## The idea and why it looked promising

Remove the N and PD patients the model finds hardest — presumed mislabelled,
atypical or uninformative — so they stop dragging the boundary across the
minority class. N and PD are abundant (167 and 188); ET (49) is never touched.

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
  directly and is tested in `prune_training_easy.md`. If hard examples are
  boundary-defining, dropping easy ones should be harmless.
