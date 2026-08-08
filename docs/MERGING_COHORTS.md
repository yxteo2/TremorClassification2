# How to combine 2015, NewData and PADS

Short answer: **do not merge them into one training pool.** Every pooling
attempt this session either failed a device-identity probe or produced no
measurable benefit. Use a three-tier design instead.

## What the three cohorts actually are

| | 2015 | NewData 2025 | PADS |
|---|---|---|---|
| subjects | N 61 / PD 75 / ET 15–16 | **ET only, 6** | N 79 / PD 276 / ET 28 |
| sensors | 3 (hand, lower_arm, upper_arm) | 3, both limbs | **1 wrist** |
| signal | quaternion → ω | quaternion → ω | **raw gyro only** |
| tasks | OUT, WING, REST | REST, OUT + **5 unused** | StretchHold, Relaxed, … |
| log_map / gravity features | yes | yes | **impossible** |
| best PD-vs-ET | REST + stft512 **0.730** | n/a (one class) | StretchHold + cwt **0.774** |

PADS has no orientation stream, so `log_map` and gravity-referenced chirality
cannot be computed there at all. That alone rules out a single shared feature
space across all three.

## The evidence against pooling

**Device-identity probes** (criterion is `|AUC − 0.5|`, not `AUC` — an AUC of
0.000 is *maximally* separable with LOO-inverted labels, not "safe"):

| pair | features | identity AUC | verdict |
|---|---|---|---|
| 2015 ET vs NewData ET, **REST** | welch descriptors | 0.688 | borderline OK |
| 2015 ET vs NewData ET, REST | stft512 | 0.729 | borderline |
| 2015 ET vs NewData ET, REST | multitaper | 0.760 | confounded |
| 2015 ET vs NewData ET, **OUT** | any | 0.000–0.122 | **maximally separable** |
| 2015 ET vs PADS ET | tfbench 10 descriptors | 0.629 | best available |
| 2015 ET vs PADS ET | orbit geometry (66) | 0.959 | confounded |
| 2015 ET vs PADS ET | STFT-702 profile | 1.000 | confounded |

**Measured effect of pooling:**

| merge | result |
|---|---|
| 2015 + NewData @ REST | every paired CI spans zero; direction inconsistent (welch +0.062, stft512 −0.051) |
| 2015 + NewData @ OUT | apparent gains, but probe is degenerate → untrustworthy |
| 2015 + PADS | **hurts**: ET-F1 0.538 → 0.350 |
| train PADS → test 2015 | chance (AUC 0.636) |
| train 2015 → test PADS | chance (AUC 0.637) |

An empirical power curve (`docs/IMPLEMENTATION_PLAN.md`) independently shows
PD-vs-ET plateaus by n≈15 ET, so extra ET subjects were never going to help
regardless of the domain shift.

## The recommended design

### Tier 1 — 2015 is the primary cohort. Train and report here.
**REST condition, full descriptor set, lower_arm.** Best result: stft512,
bal-acc **0.730**, AUC 0.729, ET-F1 0.500 [0.30, 0.67], 75 PD vs 16 ET.
Use OUT/WING for the N-vs-tremor stage, where they are stronger.

### Tier 2 — PADS is an independent external cohort. Never training data.
Run the identical pipeline on it and report side by side. It is the more
powerful cohort (28 ET, tight CIs) and it is where method differences become
detectable at all — `cwt` beats welch there by +0.040, CI [+0.009, +0.085],
p=0.0029, the only Bonferroni-passing method result in the project.

Restrict to what PADS can support: wrist sensor, gyro-derived descriptors, no
log_map, no gravity-chirality.

### Tier 3 — NewData is a *platform*, not a data source (yet).
Its 6 ET subjects add nothing measurable. Its value is that it is a device on
which you can still collect **N and PD**, and it already has **5 unused tasks**:
drinking, finger-to-nose, pouring, finger-tapping, pronation-supination.
Finger-to-nose and pouring are the classic kinetic-tremor manoeuvres, and
kinetic tremor is the clinical discriminator for ET; pronation-supination is the
counterpart for PD pill-rolling.

If you pool NewData at all: **REST only**, welch-family descriptors only,
segmented (`segment=True`, now the default), and rerun the probe first.

## Reporting

Report the three cohorts **side by side, not merged**. Three independent
cohorts run through one honest pipeline is a stronger result than one pooled
number, and it is what the domain-shift evidence supports. The cross-dataset
transfer failures are themselves a finding worth reporting.

## Non-negotiables when combining anything

1. Run the device-identity probe **first**, on the minority class only, judged
   by `|AUC − 0.5|`.
2. Match sensor position (lower_arm ↔ wrist) and task (REST ↔ Relaxed,
   OUT ↔ StretchHold) before comparing anything.
3. Score on the held-out cohort's patients, subject-level, and report the
   **paired** CI against the un-pooled baseline — not two separate point
   estimates.
4. Balanced accuracy on PD-vs-ET always; the majority baseline is 0.833 (2015)
   and 0.908 (PADS).
