# WING as a third task: PD precision +0.03, replicated -- at the standard decision point only

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

## Confirmation on fresh partitions (seeds 100-139)

`base` reproduces `fusion_2015_confirm`'s `fuse_w25` bit for bit in 40 of 40
repeats (checked after the run).

| arm | precN | precPD | precET | macroP | macroF1 | AUC |
|---|---|---|---|---|---|---|
| base | 0.740 | 0.751 | 0.357 | 0.616 | 0.606 | 0.656 |
| **+ WING** | 0.744 | **0.778** | 0.345 | **0.623** | **0.609** | 0.663 |
| + OUT2 | 0.732 | 0.763 | 0.328 | 0.607 | 0.595 | 0.656 |

| contrast (fresh) | precN | precPD | precET | macroP | macroF1 |
|---|---|---|---|---|---|
| + WING − base | +0.004 | **+0.027 \*** | −0.012 | +0.006 | +0.003 |
| + WING − + OUT2 | +0.012 | **+0.015 \*** | +0.018 | **+0.015 \*** | **+0.013 \*** |

**The PD-precision gain replicates** (+0.032 \* / +0.027 \* over base) with ET
unchanged (+0.003 / −0.012), and on fresh partitions it is attributable to WING
(+0.015 \* over the size control). macroP over base: +0.015 \* / +0.006.

## Under the high-confidence rule it does not help (`high_precision_2015.py`)

Nested rule asking 0.90, held-out precision (recall):

| partitions | base N | + WING N | base PD | + WING PD |
|---|---|---|---|---|
| selection | **0.846** (0.45) | 0.814 (0.37) | **0.872** (0.36) | 0.864 (0.38) |
| fresh | **0.826** (0.40) | 0.799 (0.36) | 0.860 (0.38) | **0.866** (0.41) |

## Verdict

* **Standard 3-class decision (macro-F1 offsets): use OUT + 0.25 REST + 0.25
  WING** -- PD precision ~0.78 vs ~0.75, ET unchanged, replicated.
* **High-confidence >= 80 % rule: keep OUT + 0.25 REST** -- WING does not raise
  PD there and lowers control precision (0.80-0.81 vs 0.83-0.85).
* Neither moves ET precision.
