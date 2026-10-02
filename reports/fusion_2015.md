# OUT + REST late fusion on 2015: better for PD, worse for ET -- a trade, not a gain

Run: `REPS=a-b python -m experiments.fusion_2015` in parts, then
`python -m experiments.fusion_2015 report` (40 repeats, CPU ~7 min/repeat/process).

## Why

`stack_2015.md`: 5 of `ft`'s consistent errors are controls with tremor at OUT
but none at REST, and REST tremor amplitude raised precN / precPD +0.02 \* in a
second stage. Here each model sees one action -- `ft` on OUT (PADS transfer),
`rest_2015._scratch` on REST (its PADS transfer reverses the ET ranking) -- and
only their log-probabilities are averaged, on `transfer_2015`'s exact partitions.
147 of 151 patients have REST; the other 4 keep the OUT prediction.

## Result (40 repeats)

| arm | precN | precPD | precET | macroP | macroF1 | PD-vs-ET AUC |
|---|---|---|---|---|---|---|
| ft | 0.723 | 0.737 | **0.350** | 0.603 | 0.590 | 0.625 |
| ft_x2 (6 seeds, size control) | 0.718 | 0.736 | 0.354 | 0.603 | 0.586 | 0.623 |
| rest alone | 0.638 | 0.726 | 0.112 | 0.492 | 0.481 | 0.600 |
| **fuse** (REST weight 0.5) | **0.741** | **0.772** | 0.288 | 0.601 | 0.591 | **0.662** |
| fuse_w25 (REST weight 0.25) | 0.734 | 0.748 | 0.326 | 0.603 | 0.591 | 0.646 |

| contrast | precN | precPD | precET | macroP | AUC |
|---|---|---|---|---|---|
| fuse − ft_x2 (attribution) | **+0.023 \*** | **+0.036 \*** | **−0.066 \*** | −0.002 | **+0.039 \*** |
| fuse − ft (adoption) | **+0.019 \*** | **+0.036 \*** | **−0.062 \*** | −0.003 | **+0.037 \*** |
| fuse_w25 − ft_x2 | **+0.016 \*** | +0.012 | −0.028 | +0.000 | **+0.023 \*** |
| fuse_w25 − fuse | −0.007 | **−0.024 \*** | **+0.038 \*** | +0.002 | −0.016 \* |
| ft_x2 − ft | −0.004 | −0.001 | +0.004 | −0.000 | −0.002 |

ET precision among the k patients ranked most ET-like (15 ET of 151):

| arm | k = 3 | 5 | 10 | 15 |
|---|---|---|---|---|
| ft | **0.533** | **0.495** | 0.405 | 0.363 |
| fuse | 0.358 (−0.175 \*) | 0.380 (−0.115 \*) | 0.390 | 0.340 (−0.023 \*) |
| fuse_w25 | 0.500 (−0.033) | 0.475 (−0.020) | 0.408 | 0.367 |

## Reading

* **REST fixes the N/PD side, as the diagnostic said**: precPD +0.036 \* and
  precN +0.019-0.023 \* against both `ft` and the size-matched 6-seed control.
  The control itself is null (6 seeds = 3 seeds here), so the gain is REST's.
* **It costs ET** at every operating point: thresholded precET −0.062 \*, and at
  the top of the ranking (k = 3: 0.533 -> 0.358 \*). REST has no 2015 ET signal
  (alone: precET 0.112, chance 0.099), so averaging dilutes the most confident
  ET calls. The PD-vs-ET AUC gain (+0.037 \*) is in the middle of the ranking,
  not at the top where ET is flagged.
* **Macro scores do not move** (macroP −0.003, macroF1 +0.001): a trade between
  classes, not an overall improvement.
* **REST weight 0.25** keeps the top-k ET precision (−0.02 to −0.03, n.s.), keeps
  a smaller N gain (+0.016 \*) and AUC +0.023 \*, and loses the PD gain.

## Verdict

* **ET is the target -> keep `ft`.** No fusion weight improves ET precision.
* **PD / control precision is the target -> `fuse` (equal weight)**, +0.036 PD
  and +0.019 N precision, at −0.06 ET precision.
* `fuse_w25` is the low-risk middle: better N precision and ranking, ET
  statistically unchanged, macro unchanged.

## Prediction (in the docstring) -- held

"fuse beats ft_x2 on precPD (+0.02 to +0.05) and precN, with precET flat or
lower; fuse_w25 keeps more ET precision than fuse." precPD +0.036 \*, precN
+0.023 \*, precET −0.066 \*; fuse_w25 − fuse precET +0.038 \*. Registered as AC.
Derived from a measurement of this dataset (`stack_2015.md`), like most held
predictions in the register.
