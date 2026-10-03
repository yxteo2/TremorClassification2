# Reaching 80 % precision on 2015 OUT: yes for N and PD, not for ET

Run: `python -m experiments.high_precision_2015` (seconds; saved out-of-fold
probabilities from `segments_2015` and `fusion_2015`, same 40 partitions).

## Rule

Flag a patient as class c only when P(c) clears a threshold; leave the rest for
clinical review (no forced 3-class call). **Nested**: for each held-out fold the
threshold is the lowest one at which the other four folds reach the requested
precision (>= 3 flags), applied unchanged to the held-out fold. An ORACLE line
(threshold chosen on the scored patients) is printed for contrast only.

## Result (40 repeats; mean held-out precision, bootstrap 95 % CI)

**Requested precision 0.90 on the training folds:**

| class | score | held-out precision | recall | flagged / 151 |
|---|---|---|---|---|
| N | ft | 0.817 [0.784, 0.845] | 0.311 | 22.6 |
| N | **fuse** | **0.849** [0.827, 0.867] | 0.443 | 31.4 |
| PD | ft | 0.852 [0.835, 0.867] | 0.251 | 22.0 |
| PD | **fuse** | **0.873** [0.864, 0.881] | 0.407 | 35.0 |
| tremor vs none | fuse | **0.894** [0.890, 0.897] | 0.781 | 78.6 |
| ET | ft | 0.333 [0.133, 0.533] | 0.008 | 0.3 (none in 35 / 40) |

**Requested 0.80:** held-out N 0.789 (recall 0.752), PD 0.789 (0.671), tremor
0.803 (0.912), ET 0.298 on 0.5 flags (none in 33 of 40) -- a threshold chosen
on training patients always gives back a little on new ones, so the rule must
ask for more than the goal.

The ORACLE line claims ET precision 0.90-1.00 -- on 0.1-0.4 flags per repeat.
That is the number a non-nested report would publish; it is noise.

## Reading

* **N and PD exceed 80 % precision on held-out patients** with the fused
  OUT + REST model and a high-confidence rule: 0.849 / 0.873, flagging about 40 %
  of each class; the rest go to review. `fuse` beats `ft` here because REST
  sharpens exactly the N/PD boundary (`fusion_2015.md`).
* **Tremor vs none reaches 0.89 precision at 0.78 recall** -- the screening use.
* **ET does not reach 80 % at any coverage on this data.** A P(ET) threshold
  learned from ~12 training ET does not transfer to 3 held-out ET; the best
  ranking-based ET precision is ~0.5 among the 3-5 most ET-like patients
  (`fusion_2015.md`, `ft`). The binding constraint is the 15 ET, 6 of whom show
  no OUT tremor (`tremor_present_2015.md`).

## Recommended reporting

Report the high-confidence rule for N and PD with its coverage (precision 0.85 /
0.87 at ~40 % recall, the rest referred), the tremor screen (0.89 / 0.78), and
ET as a ranked "flag for review" list with its measured precision (~0.5 at the
top 3-5), not as a thresholded class with an 80 % claim.

## Prediction (in the docstring)

* N and PD reach >= 0.80 held-out precision: **failed at the 0.80 request**
  (0.789 / 0.789), held when the rule asks for 0.90 (0.849 / 0.873). #40.
* ET does not reach 0.80 reliably: **held** (AD).
