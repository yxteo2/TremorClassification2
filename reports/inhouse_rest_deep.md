# In-house REST in the deep model: carries PD information, not ET

**Question.** `inhouse_pd_vs_et.md` closed OUT for in-house PD vs ET and left one
lever open: "in-house REST in the deep model is untested". This tests it on the
cohort the clinic actually has -- 2015 + NewData only, PADS nowhere.

**Answer.** REST does not raise in-house ET precision on any sensor, alone or
fused. Fusing OUT and REST scores gives a PD-precision gain on the hand sensor
(+0.059 \*) and nothing significant on macroP on either sensor. In-house ET
precision stays at **0.16 at prevalence 0.105** -- about 1.5x chance.

Run: `python -m experiments._inhouse_rest_diagnostic` (pre-check, linear),
`TREMOR_DEVICE=cuda SENSOR=hand|lower python -m experiments.inhouse_rest_deep`.

## 1. The pre-check (linear, no deep fits)

PD vs ET, in-house pooled (21-22 ET), features = 16-bin multitaper + ten
descriptors z-scored within cohort, L2 logistic regression, 5-fold CV averaged
over 10 partitions, 100-permutation null. The four `N 2` accelerometer files
(verify_data check 13) are dropped.

| sensor | OUT | REST | OUT+REST (union) |
|---|---|---|---|
| hand | 0.547 (p 0.28) | **0.661 (p 0.03)** | **0.712 (p 0.01)** |
| lower arm | 0.482 (p 0.60) | 0.556 (p 0.23) | 0.587 (p 0.17) |
| upper arm | 0.519 (p 0.46) | 0.598 (p 0.09) | 0.537 (p 0.30) |
| all three, REST | | 0.618 (p 0.07) | |

N vs tremor is above chance everywhere (0.72-0.89; best OUT+REST lower arm 0.890).

Only the hand sensor clears its null on PD vs ET, and it is the best of nine
arms -- nothing survives Bonferroni. So hand was pre-registered as the primary
deep arm and lower arm (the pipeline's sensor) as secondary.

## 2. The deep run

`own_data_10et` protocol: 200 patients with both tasks (85 N / 94 PD / 21 ET),
every test set 40 N / 45 PD / **10 ET (prevalence 0.105)**, 20 repeats, the
reported recipe (two-stream CNN + ResidualTCN, 3 seeds, validation-tuned
offsets), trained on the GPU. `OUT+REST` is a soft vote of the per-task models
(tasks never averaged, `task_averaging.md`); `OUT+shufREST` is the invariant-11
control, REST tables permuted across patients within cohort, redrawn per repeat.

| sensor | arm | precN | precPD | precET | macroP | macroF1 |
|---|---|---|---|---|---|---|
| lower | OUT (baseline) | 0.657 | 0.719 | **0.162** | 0.513 | 0.469 |
| lower | REST | 0.599 | 0.736 | 0.105 | 0.480 | 0.411 |
| lower | OUT+REST | **0.667** | **0.753** | 0.161 | **0.527** | **0.477** |
| lower | OUT+shufREST | 0.655 | 0.703 | 0.086 | 0.481 | 0.428 |
| hand | OUT | 0.617 | 0.714 | 0.094 | 0.475 | 0.425 |
| hand | REST | 0.523 | 0.705 | 0.053 | 0.427 | 0.381 |
| hand | OUT+REST | 0.592 | 0.773 | 0.087 | 0.484 | 0.416 |
| hand | OUT+shufREST | 0.589 | 0.612 | 0.110 | 0.437 | 0.396 |

Paired, 20 repeats, bootstrap 95 % CI, split win rate:

| sensor | contrast | precPD | precET | macroP | macroF1 |
|---|---|---|---|---|---|
| lower | OUT+REST − OUT | +0.034 [−0.005, +0.080] (0.60) | −0.001 [−0.055, +0.055] (0.45) | +0.014 [−0.009, +0.039] (0.60) | +0.008 (0.65) |
| lower | OUT+REST − shufREST | +0.050 \* (0.85) | +0.075 \* (0.50) | +0.046 \* (0.85) | +0.049 \* (0.80) |
| lower | shufREST − OUT | −0.016 | **−0.076 \*** (0.25) | −0.032 \* | −0.042 \* |
| hand | OUT+REST − OUT | **+0.059 [+0.016, +0.109] \*** (0.70) | −0.007 (0.30) | +0.009 (0.60) | −0.009 |
| hand | OUT+REST − shufREST | +0.160 \* (0.80) | −0.024 (0.25) | +0.047 \* (0.75) | +0.020 |
| hand | REST − OUT | −0.009 | −0.040 (0.35) | −0.048 \* | −0.044 \* |

## Reading it

* **Attribution vs adoption disagree, and adoption decides.** Against the
  shuffled control, OUT+REST looks significant on four columns. But the control
  is itself significantly *worse* than the plain baseline (a noise member hurts:
  precET −0.076 \* on lower arm), so most of that gap is the control being
  damaged. REST does carry real patient-specific information -- the control
  proves that -- but against the plain OUT model it buys macroP +0.014 (n.s.)
  and precET −0.001.
* **The information REST carries is PD information.** The one baseline-significant
  gain is precPD on the hand sensor (+0.059 \*, win 0.70); on lower arm it is
  +0.034 (win 0.60). That fits the physiology (PD is a rest tremor) and not the
  ET hope.
* **The linear pre-check's hand-sensor PD-vs-ET AUC did not carry into the
  3-class deep model.** The hand sensor is worse than lower arm on almost every
  column. A binary ranking AUC of 0.66-0.71 at 21 ET is not enough to move a
  3-class precision at 10 test ET, where one ET prediction is worth ~0.07.
* **Baseline consistency.** The lower-arm OUT arm (0.657 / 0.719 / 0.162 / 0.513)
  is below the published in-house figure (0.652 / 0.769 / 0.193 / 0.538). The
  test sets differ (200 patients with both tasks vs 207 with OUT; different
  draws) and training ran on the GPU, so the two are not paired. Quote the
  paired differences above, not the absolute levels, across the two reports.

## Standing

* In-house ET is not improvable with REST, OUT+REST fusion, or a different
  sensor. This closes the last modelling lever `inhouse_pd_vs_et.md` listed.
  The binding constraint in-house is the ET patients themselves (21, and at OUT
  indistinguishable from PD), per `data_plan.md`.
* OUT+REST score fusion on the lower arm is the best in-house arm on every
  column but ET, with nothing significant against baseline. It is a reasonable
  default if a REST recording exists, not a demonstrated improvement.
* If the in-house PD-precision gain matters clinically, re-run the hand-sensor
  OUT+REST arm at 40 repeats before quoting it -- +0.059 at 20 repeats is
  exactly the size that has flipped sign on doubling here before.
