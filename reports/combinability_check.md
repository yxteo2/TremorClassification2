# Can 2015 and NewData be combined? — pre-merge check

Run before any pooled training, on the only class the two cohorts share (**ET**),
because that is where a disagreement silently becomes a label. Code:
`python -m tfbench.combinability`.

Two conditions must both hold:
1. **Frequency agreement** — max_freq and mean_freq distributions equivalent
   within ±1 Hz (whole bootstrap CI inside the tolerance).
2. **Device-identity probe** — cohorts not separable, judged on `|AUC − 0.5|`.

## Result: neither condition passes both

| condition | measure | 2015 ET | NewData ET | diff [95% CI] | p | equivalent? |
|---|---|---|---|---|---|---|
| **REST** | max_freq | 6.15 | 5.86 | +0.29 [−1.07, +1.66] | 0.459 | no (CI too wide) |
| **REST** | **mean_freq** | **7.45** | **6.59** | **+0.86 [+0.13, +1.76]** | **0.017** | **no — genuinely differ** |
| OUT | max_freq | 6.45 | 6.93 | −0.49 [−3.52, +1.17] | 0.459 | no (CI too wide) |
| OUT | mean_freq | 7.25 | 7.11 | +0.14 [−2.04, +0.94] | 0.677 | no (CI too wide) |

| condition | device probe | verdict |
|---|---|---|
| REST | AUC 0.698, \|dev\| 0.198 — **pass** | **DO NOT COMBINE** — frequencies differ |
| OUT | AUC 0.211, \|dev\| 0.289 — **confounded** | **DO NOT COMBINE** — device-separable *and* underpowered |

The two conditions fail for **opposite** reasons: at OUT the cohorts agree on
frequency (p=0.68) but a model can tell them apart; at REST they are
indistinguishable by device but genuinely differ in mean frequency (p=0.017).

**Read the "not equivalent" verdicts carefully.** With only 6 NewData ET
subjects, three of the four are simply *underpowered* — the CI is too wide to
certify equivalence, which is absence of evidence, not evidence of difference.
Only **REST mean_freq** is a real, significant disagreement.

## Why this matters more than the CI width: the merged ET class straddles PD

At REST the 2015 cohort has ET **above** PD (7.45 vs 7.09 Hz) — the textbook
direction. NewData ET sits **below** PD at 6.59 Hz, i.e. on the *other side*.

| | fraction of ET patients below the PD median |
|---|---|
| 2015 ET alone | 38 % |
| 2015 + NewData ET | **50 %** |

Merging pushes the ET class from leaning above PD to **evenly straddling it**.
On the frequency axis the merged ET group becomes bimodal, with half of it now
on the PD side. That is the opposite of what adding data should do.

## The uncomfortable part

The merged REST model produced the **best ET-F1 in the project** (0.557
[0.40, 0.70] vs 0.417 for 2015 alone) — see `reports/merged_design_results.md`.
That gain now looks suspicious rather than encouraging: the cohorts differ
significantly on the very feature the model uses, so some of the improvement may
be the classifier exploiting a **cohort** difference that happens to align with
the ET label, not learning better tremor discrimination. NewData is ET-only, so
any cohort cue is perfectly confounded with the ET class.

The REST device probe passing (0.698) argues against the crudest version of
this — the model cannot trivially identify the cohort from the descriptors — but
the probe uses all 10 descriptors while the disagreement is concentrated in one.

## Recommendation

**Do not merge for the headline result.** Report 2015 alone as primary:
REST + stft512, bal-acc 0.730, ET-F1 0.500 [0.30, 0.67].

If the merged number is reported at all, it needs:
* the frequency disagreement stated (2015 ET 7.45 vs NewData ET 6.59 Hz, p=0.017);
* a leave-one-cohort-out check — does the merged model still beat 2015-alone
  when scored **only** on 2015 patients? (`tfbench.merged` already scores this
  way, and the paired CI spanned zero: +0.062 [−0.013, +0.160]);
* six extra ET subjects acknowledged as too few to move a plateau anyway.

**What would make them combinable:** collect **N and PD on the NewData device**.
That removes the ET-only confound entirely — cohort membership would no longer
predict the label — and lets the device effect be estimated rather than assumed
away.
