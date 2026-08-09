# PADS `Relaxed` result — both pre-registered predictions failed

`docs/EXTRACT_PADS_RELAXED.md` recorded a prediction **before** the data was
extracted:

> On the 2015 cohort REST beat OUT by 0.15–0.28 balanced accuracy for PD-vs-ET.
> If that is a property of the *condition* rather than of our cohort, PADS
> Relaxed should beat PADS StretchHold's 0.774. If it does not, the REST
> advantage is cohort-specific.

Extraction verified first: 766 recordings, 383 patients, **N=79 / PD=276 /
ET=28** — matching the published cohort, so the label fix held.

## Test 1 — PADS Relaxed is WORSE than StretchHold, for every method

PD-vs-ET balanced accuracy, PADS only, patient-level LOSO:

| method | StretchHold (postural) | **Relaxed (rest)** | diff |
|---|---|---|---|
| welch | 0.734 | 0.675 | **−0.060** |
| cwt | **0.774** | 0.641 | **−0.133** |
| stft512 | 0.749 | 0.671 | −0.078 |
| multitaper | 0.736 | 0.689 | −0.047 |

**Prediction refuted.** Rest is not the better condition in PADS — it is the
worse one, consistently, by 0.05–0.13.

**Therefore the REST advantage is cohort-specific to the 2015 data.** On our
cohort REST beat OUT by 0.15–0.28; on PADS the ordering reverses. "Use REST for
PD-vs-ET" is a fact about our recordings, not about rest tremor, and must not be
generalised in the write-up.

## Test 2 — task-matched external validation still fails

Merged 2015+NewData trained at REST, scored once on PADS **Relaxed** (matched
task, matched sensor position):

| method | PD-vs-ET internal | **external (PADS Relaxed)** | AUC |
|---|---|---|---|
| welch | **0.740** | 0.432 | **0.420** |
| cwt | 0.674 | 0.475 | 0.424 |
| stft512 | 0.685 | 0.434 | 0.412 |
| multitaper | 0.724 | 0.445 | **0.387** |

**This kills the task-mismatch explanation.** Earlier the below-chance transfer
could be blamed on training at REST and testing on a postural task. Now the
tasks match — rest → rest — and external AUC is **0.387–0.424**, still
consistently **below** 0.5 across all four methods.

An AUC reliably under chance is not noise. It means the PD/ET decision boundary
learned on our cohorts is **inverted** on PADS. The frequency analysis already
showed why: PADS has PD **faster** than ET (7.71 vs 6.55 Hz, p<0.0001), while
the 2015 cohort has them unseparated at OUT and PD **slower** at REST
(5.47 vs 6.15 Hz). Opposite directions, so a model trained on one is
anti-predictive on the other.

N-vs-Tremor also transfers worse to Relaxed (0.539–0.608) than to StretchHold
(0.691–0.736), consistent with rest being the harder condition in PADS.

## Consequences

1. **PD-vs-ET cannot be externally validated on PADS**, under any task pairing
   tried. It is an internal result: merged 2015+NewData at REST, welch,
   **bal-acc 0.740, ET-F1 0.557 [0.40, 0.70]**, and it must be reported as such.
2. **N-vs-Tremor keeps its external validation**, but from the OUT-trained model
   against PADS StretchHold: internal 0.814 → external **0.736, AUC 0.783**.
   That pairing stands; the REST pairing does not.
3. **The transfer failure is a finding, not a gap.** Two independent cohorts
   disagree on the *direction* of the PD/ET frequency contrast. That is a
   substantive claim about wearable tremor data — one that anyone pooling public
   tremor datasets needs to know — and it is now demonstrated on matched tasks,
   matched sensors and corrected labels, which is a much stronger form of the
   claim than the earlier mismatched version.
4. The best condition for PD-vs-ET is **cohort-dependent**: REST for 2015,
   StretchHold for PADS. Any recommendation about task choice has to be scoped
   to the cohort it was measured on.

## Housekeeping

`pads_stretchhold/` now holds 1598 files — 766 newly extracted with the task
token plus 832 legacy-named leftovers. Loading is unaffected (`strict=True`
takes classes from the manifest, which lists only the 766), but the legacy files
are dead weight and will break `strict=False`. Safe to delete any
`pads_stretchhold/*.txt` whose name has only three underscore-separated fields.
