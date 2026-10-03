# WING as a third task: a small PD / control gain over OUT + 0.25 REST

Run: `REPS=a-b python -m experiments.fusion3_2015` in parts, then
`python -m experiments.fusion3_2015 report` (40 repeats, CPU ~10 min/repeat/process).

## Why

The recommended 2015 model is `ft` (OUT) + 0.25 x REST (`fusion_2015.md`). 2015
also records WING. Feature level: WING has no PD-vs-ET signal (AUC 0.526, null
p = 0.39) but detects tremor as well as OUT (N vs tremor 0.893); the controls
`ft` calls PD are at control level in WING. One action per model; fixed weights
0.5 OUT / 0.25 REST / 0.25 third model. The size-matched control's third model is
a 2015 OUT scratch model.

**Assert first:** `base` reproduces `fusion_2015`'s `fuse_w25` predictions bit
for bit on every selection repeat.

## Selection partitions (seeds 0-39)

| arm | precN | precPD | precET | macroP | macroF1 | PD-vs-ET AUC |
|---|---|---|---|---|---|---|
| base (OUT + 0.25 REST) | 0.734 | 0.748 | 0.326 | 0.603 | 0.591 | 0.646 |
| **+ WING** | **0.745** | **0.780** | **0.328** | **0.618** | **0.599** | 0.648 |
| + OUT2 (size control) | 0.729 | 0.768 | 0.319 | 0.606 | 0.590 | 0.642 |

| contrast | precN | precPD | precET | macroP | macroF1 |
|---|---|---|---|---|---|
| + WING − base (adoption) | +0.010 | **+0.032 \*** | +0.003 | **+0.015 \*** | +0.008 |
| + WING − + OUT2 (attribution) | **+0.015 \*** | +0.012 [−0.000, +0.024] | +0.009 | +0.012 | +0.009 |
| + OUT2 − base | −0.005 | **+0.020 \*** | −0.007 | +0.003 | −0.000 |

* **Adoption:** PD precision +0.032 \* and macroP +0.015 \* over the recommended
  model, ET unchanged.
* **Attribution:** about two-thirds of the PD gain comes from adding a third
  model at all (+ OUT2: +0.020 \*); WING's own contribution is control precision
  (+0.015 \*) and PD +0.012 at the edge of its CI.
* **The 15-repeat interim overstated it**: WING − OUT2 read precET +0.056 \*,
  macroP +0.028 \* at 15 repeats and +0.009 / +0.012 (n.s.) at 40.

## Prediction (in the docstring) -- held, weakly

"+ WING beats + OUT2 on precPD (+0.01 to +0.03) and precN, ET flat": precN
+0.015 \*, precPD +0.012 (in range, CI touching 0), ET +0.009. (AF)

## Confirmation on fresh partitions

Running (seeds 100-139); verdict pending.
