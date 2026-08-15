# Combining 2015 / NewData / PADS: four strategies, none better

**Conclusion: the existing handling is already the best of those tested.** Cap
PADS at 90/class, pool the three cohorts, fit one global set of
validation-tuned priors. Two alternatives are significantly *worse*.

Mixed-cohort protocol (all three sources in train/val/test), asymmetry carried
as a missing modality, welch, 10 splits, per-class precision.

| strategy | precN | precPD | precET | macroP | macroF1 |
|---|---|---|---|---|---|
| **cap 90 + global priors (baseline)** | 0.653 | 0.655 | 0.639 | **0.649** | **0.600** |
| cap 90 + per-cohort priors | 0.641 | 0.640 | 0.597 | 0.626 | 0.573 |
| cap 90 + PADS-pretrain/finetune | 0.658 | 0.639 | 0.451 | 0.583 | 0.548 |
| cap 90 + cohort-ID input | 0.657 | 0.656 | 0.690 | 0.668 | 0.589 |
| uncapped, unweighted | 0.556 | 0.742 | 0.221 | 0.506 | 0.511 |
| uncapped + sample weights | 0.554 | 0.728 | 0.195 | 0.492 | 0.486 |
| uncapped + weights + per-cohort priors | 0.549 | 0.752 | 0.169 | 0.490 | 0.488 |

Paired against the baseline on the same splits:

| strategy | precET | macroP | macroF1 |
|---|---|---|---|
| per-cohort priors | -0.042 [-0.139, +0.019] | **-0.023 [-0.051, -0.002]** * | **-0.028 [-0.049, -0.006]** * |
| PADS pretrain/finetune | **-0.188 [-0.315, -0.088]** * | **-0.066 [-0.110, -0.032]** * | **-0.052 [-0.090, -0.017]** * |
| cohort-ID input | +0.051 [-0.040, +0.168] | +0.019 [-0.020, +0.064] | -0.012 [-0.043, +0.020] |

## Transfer learning from PADS fails

Pretraining on the PADS patients of the training split then fine-tuning on
2015 + NewData is the standard way to use a large auxiliary cohort. It produces
the **largest negative effect measured in this repo**: ET precision
-0.188 [-0.315, -0.088].

The fine-tuning set is ~130 patients with ~21 ET, so the fine-tune phase
overwrites what the PADS phase learned about the minority class. PADS is not an
auxiliary cohort to be transferred *from* -- it is where most of the ET signal
lives (`merge_design.md`: dropping it takes ET precision from 0.519 to 0.065).
Pooling is the correct way to use it.

## Discarding data beats reweighting it

Sample weights of 1/cohort-size were expected to substitute for capping while
keeping all 383 PADS patients. They do not: macroP 0.492 against 0.649 capped,
and plain uncapped (0.506) is *better* than uncapped+weighted.

The class pattern shows why. Uncapped, precPD rises to 0.742 while precET
collapses to 0.221 -- with PADS at 72 % PD, keeping every patient makes the
model a PD detector. Weighting equalises each cohort's contribution to the
**loss** but not its influence on the learned **representation**, and under
full-batch training the gradient direction is still set by PADS's covariance.

## Per-cohort priors overfit

Fitting logit offsets per cohort rather than globally is significantly worse
(macroP -0.023 [-0.051, -0.002]). NewData contributes ~11 validation patients
per split, which is far too few to fit a 2-parameter offset. Global prior
tuning -- the largest single gain of the session -- should stay global.

## Cohort-ID input: best mean, not significant, and destabilising

macroP +0.019 [-0.020, +0.064] and precET +0.051 [-0.040, +0.168]: the highest
means in the table but both CIs span zero. More telling, its sd nearly doubles
(macroP 0.095 against 0.052; precET 0.239 against 0.185), so it helps on some
splits and hurts on others.

**It is also protocol-restricted.** Cohort identity is legitimate under the
mixed protocol, where the target sites are known in advance. Under
leave-one-cohort-out the test site is unseen and the feature leaks. Do not carry
it into any generalisation claim.

Reproduce: `python -m tfbench.cohort_strategies`.
