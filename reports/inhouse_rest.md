# In-house REST in the deep model: a small in-house ranking gain that its control also gets

Run: `python -m experiments.inhouse_rest` (20 splits, adopted 9-member ensemble)

## Why this was run

`inhouse_pd_vs_et.md` (follow-up) found in-house PD vs ET **absent at OUT** —
the only task the in-house cohorts feed the model — but **present at REST**:
six frequency characteristics read AUC ~0.65-0.70 on 2015 REST, above chance on
all three sensors, with PD slower (the textbook direction). PADS runs the other
way. Earlier task studies averaged REST *into* the postural spectrum on the
merged cohort; none gave the model in-house REST as a separate input.

## Design

REST enters as its own descriptor block (10 repo descriptors on the REST
recordings + a have-REST flag), zeros where absent, exactly like asymmetry.
Spectrum, trajectory and the TCN member are unchanged.

| arm | REST block |
|---|---|
| reported9 | none (adopted model) |
| + REST in-house | 2015 + NewData; PADS flag 0 |
| + REST all | all three cohorts (PADS Relaxed) |
| CONTROL shuffled | in-house block, rows permuted within cohort, re-drawn every split |

Coverage: 2015 147/151 patients (15/15 ET), NewData 53/56 (6/6 ET), PADS all.
`N 2`'s two REST files are accelerometer data (`verify_data` check 13) and were
treated as missing; the patient stays in.

**Baseline check:** the reported9 arm's 20-split means reproduce
`diverse_ensemble_2.md` exactly (0.656 / 0.655 / 0.715 / 0.675 / 0.611 / 0.460),
on a freshly reinstalled environment.

`inAUC`, `inPrecET` and `inRecET` score in-house test patients only, about 41 per
split and **about 4 ET**, so they are noisy per split. Paired over splits.

## Result

| arm | precN | precPD | precET | macroP | macroF1 | recET | inAUC | inPrecET | inRecET |
|---|---|---|---|---|---|---|---|---|---|
| reported9 | 0.656 | 0.655 | **0.715** | **0.675** | 0.611 | 0.460 | 0.588 | 0.414 | 0.325 |
| + REST in-house | 0.652 | 0.653 | 0.686 | 0.664 | 0.600 | 0.420 | **0.609** | 0.353 | 0.263 |
| + REST all | 0.667 | 0.659 | 0.653 | 0.660 | 0.612 | 0.465 | 0.597 | 0.417 | 0.325 |
| CONTROL shuffled | 0.645 | 0.648 | 0.667 | 0.653 | 0.595 | 0.430 | 0.598 | 0.414 | 0.275 |

**Adoption, + REST in-house vs reported9:** inAUC **+0.022 [+0.004, +0.042] \***
(win 0.60), but in-house ET recall **−0.062 [−0.113, −0.013] \*** and merged recET
**−0.040 [−0.080, −0.010] \***. precET −0.029, macroP −0.012, both null.

**Attribution, + REST in-house vs the shuffled control:** inAUC +0.012
[−0.006, +0.029], **null**. The control, with no patient-label link, itself
moves inAUC +0.010 over the baseline — so most of the ranking gain is the extra
columns, not what REST says. The control is also significantly **worse** than
the baseline on the merged fold (precET −0.049 \*, macroP −0.022 \*, macroF1
−0.016 \*): eleven uninformative columns cost the ensemble real precision, and
`+ REST in-house` recovers only part of that (+0.019 precET over the control,
null).

**PADS REST, + REST all vs + REST in-house:** inAUC −0.013 [−0.032, +0.006],
right direction, null. Against the control it gains precN +0.023 \* and macroF1
+0.017 \*, but against the adopted model nothing, and precET −0.062
[−0.125, +0.000].

## Verdict

**Not adopted.** In-house REST nudges the in-house PD-vs-ET *ranking* up by
+0.02, but the shuffled control gets half of that without the REST information,
the decision-level ET metrics fall (in-house ET recall −0.062 \*), and nothing
on the merged fold improves.

This matches the standing rule that **sub-component gains do not compose**
(`method_rules.md`): a signal visible to six hand-made features at 16 ET does
not survive becoming 11 extra columns among ~120 inputs, trained on 404
patients of whom ~21 in-house ET, and read by an ensemble a third of which (TCN)
never sees it. The notebook's REST finding stands as a property of the data; the
model cannot use it at this n.

The prediction failed (#29 in `failed_predictions.md`): it expected +0.03 to
+0.08 inAUC over *both* the baseline and the control.

## What would test it properly

* **More in-house ET.** The notebook signal is AUC ~0.65-0.70 at 16 ET; the
  model needs the ~4 ET per test fold to become ~10+ before a +0.02-0.05 effect
  on the in-house axis is resolvable.
* **A dedicated in-house PD-vs-ET model on REST features**, evaluated with the
  notebook's repeated-CV null, if a separate in-house result is wanted for the
  paper. It sidesteps the dilution by ~120 other inputs, but it is a second
  model and should be reported as such.
