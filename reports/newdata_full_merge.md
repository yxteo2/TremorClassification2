# Merging the FULL NewData cohort with 2015

Every earlier pooling test added NewData's **ET only**, because that was all the
cohort had. With 31 HC and 34 PD it can now be merged whole — cohort membership
no longer predicts the ET label, which was the structural objection.

Scored two ways: on all patients, and on the **same 2015 patients** as the
2015-only baseline (the only fair comparison).

## REST

| model | n | pos | bal-acc | AUC | precision | recall |
|---|---|---|---|---|---|---|
| **PD vs ET** | | | | | | |
| 2015 only | 91 | 16 | **0.730** | **0.729** | **0.393** | 0.688 |
| NewData only | 31 | 6 | 0.487 | 0.273 | 0.182 | 0.333 |
| merged, scored on 2015 | 91 | 16 | 0.648 | 0.720 | 0.310 | 0.562 |
| **N vs Tremor** | | | | | | |
| 2015 only | 152 | 91 | 0.753 | 0.844 | 0.842 | 0.703 |
| NewData only | 58 | 31 | 0.640 | 0.670 | 0.679 | 0.613 |
| **merged, scored on 2015** | 152 | 91 | **0.761** | **0.853** | **0.871** | 0.670 |

## OUT

| model | n | pos | bal-acc | AUC | precision |
|---|---|---|---|---|---|
| PD vs ET, 2015 only | 90 | 15 | 0.460 | 0.496 | 0.139 |
| PD vs ET, merged on 2015 | 90 | 15 | 0.427 | 0.495 | 0.114 |
| N vs Tremor, 2015 only | 151 | 90 | 0.812 | 0.902 | 0.877 |
| **N vs Tremor, merged on 2015** | 151 | 90 | **0.821** | **0.905** | **0.887** |

## The answer differs by axis

**N-vs-Tremor: merging helps, slightly and consistently.** REST 0.753 → 0.761
(AUC 0.844 → 0.853, precision 0.842 → 0.871); OUT 0.812 → 0.821 (AUC 0.902 →
0.905, precision 0.877 → 0.887). Small, and no paired CI has been run, but the
direction is the same across balanced accuracy, AUC *and* precision, in both
conditions. This is the first merge in the project that has not hurt.

**PD-vs-ET: merging still hurts.** REST 0.730 → 0.648, precision 0.393 → 0.310.
Adding 6 ET and 25 PD from a cohort whose own PD-vs-ET is at chance (0.487, AUC
0.273) drags the combined model down.

## Device probes, now measurable on every class

Only ET could be probed before. With all three classes present:

| condition | N | PD | ET |
|---|---|---|---|
| REST | 0.880 **conf.** | 0.903 **conf.** | 0.688 pass |
| OUT | 0.879 **conf.** | 0.788 **conf.** | 0.200 **conf.** |

**The cohorts are more separable on N and PD than on ET** — the opposite of what
the ET-only probes suggested. Every earlier "safe to pool" verdict rested on the
ET probe alone and was therefore measuring the least device-separable class.

That the N-vs-Tremor merge helps *despite* N and PD being strongly
device-separable is worth noting: with both cohorts contributing all three
classes, the device cue is orthogonal to the label rather than aligned with it,
so the extra patients help even though the domains differ.

## Recommendation

* **N-vs-Tremor: merge.** Best result becomes **OUT, merged, scored on 2015 —
  bal-acc 0.821, AUC 0.905, precision 0.887.** Confirm with a paired CI before
  reporting.
* **PD-vs-ET: do not merge.** 2015 REST alone (0.730 / precision 0.393) remains
  best; NewData's contribution on this axis is negative.

# Domain correction works — and does not help

If merging fails because of device shift, removing the shift should fix it.
Per-cohort standardisation (z-score features within each cohort before pooling)
does remove it:

| | device probe on ET |
|---|---|
| raw | AUC 0.688 (\|dev\| 0.188) |
| **per-cohort z-score** | **AUC 0.490 (\|dev\| 0.010)** |

The cohorts become statistically inseparable. Performance does not follow:

| model (REST, PD-vs-ET) | ET | bal-acc | AUC | precision |
|---|---|---|---|---|
| **2015 only** | 16 | **0.730** | **0.729** | **0.393** |
| plain merge, scored on 2015 | 16 | 0.648 | 0.720 | 0.310 |
| per-cohort z-score, scored on 2015 | 16 | 0.639 | 0.721 | 0.278 |
| per-cohort z-score, scored on all | 22 | 0.630 | 0.667 | 0.283 |

**So merging does not fail because of domain shift.** The shift was removed
completely and nothing changed. That rules out the technical explanation and
leaves the substantive one: the 6 NewData ET patients do not carry usable
PD-vs-ET information — consistent with NewData's own internal PD-vs-ET sitting
at chance (AUC 0.273 at REST). Adding uninformative ET dilutes rather than
strengthens.

Merging has now been measured five ways — plain, ET-only, all-three-cohort,
full-cohort, and domain-corrected — and is negative for PD-vs-ET every time.
More ET helps only if the ET are discriminable to begin with.
