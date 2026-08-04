# PADS label contamination — a real bug, and what it invalidates

**Found by the user's instinct that "something is wrong with the PADS
processing." It was.** The signal processing was fine; the *labelling* was not.

## The bug

`pdetn/extract_pads.py` mapped PADS's free-text diagnosis strings to N/PD/ET by
**substring** match, and the map contained the bare tokens `"et"` and `"pd"`:

```python
LABEL_MAP = {"healthy": "N", "control": "N", "hc": "N",
             "parkinson": "PD", "pd": "PD",
             "essential tremor": "ET", "essential-tremor": "ET", "et": "ET"}
```

PADS diagnoses are clinical free text. `"et"` occurs inside **etiology**,
**asymm*et*ric**, **R*et*rocollis** and **hypokin*et*ic**. So:

**13 of 41 "ET" patients (32 %) were not Essential Tremor**, including two
*parkinsonian* cases sitting inside the ET class:

| patient | actual diagnosis | matched on |
|---|---|---|
| 414 | Starting **hypokinetic**-rigid syndrome | "hypokin*et*ic" |
| 404 | Tremor of unknown **etiology**. Lewy-Body-Dementia | "*et*iology" |
| 111 | Tremor of unknown etiology, DD, SWEDD syndrome | "*et*iology" |
| 130 | **Asymmetric** tremor of the hands of unknown etiology | "asymm*et*ric" |
| 196 | Cervical dystonia … **Retrocollis**. THS | "R*et*rocollis" |
| 209 | Leukoencephalopathy of unknown etiology | "*et*iology" |
| 273 | Dystonia of unknown etiology | "*et*iology" |
| 287 | Adult multifocal dystonia of unknown etiology | "*et*iology" |
| 439 | Spinocerebellar syndrome … asymmetric | "asymm*et*ric" |
| 035, 123, 207, 276 | mixed/differential diagnoses containing "Essential Tremor" | ambiguous |

**20 of 296 "PD" patients** were Atypical Parkinsonism / vascular Parkinson
syndrome / Dystonia-Parkinson-Syndrome — which PADS treats as a **separate
differential-diagnosis group**, and `reports/track3_external_data.md` had
explicitly said to filter out. It was never filtered.

A 32 % contamination rate in the minority class, with parkinsonian disorders
mixed *into* ET, is the worst possible direction for this project: it blurs
exactly the PD-vs-ET boundary the whole study is about.

**The count was checkable all along.** `track3_external_data.md` records the
published PADS cohort as **28 ET** (Varghese 2024). The extraction produced 41.
That discrepancy was in the repo and went unquestioned.

## The fix

Exact matching on the normalised diagnosis, everything else dropped:

```python
LABEL_MAP_EXACT = {"healthy": "N", "parkinson's": "PD", "essential tremor": "ET"}
```

N=79 / PD=276 / ET=**28** — the ET count now agrees with the publication.
Mixed diagnoses ("Essential Tremor, DD functional Tremor") are excluded
deliberately: they are ambiguous by construction and cannot support a clean
PD-vs-ET claim.

`load_pads_extracted(folder, strict=True)` re-derives the class from the
manifest's `raw_label` and no longer trusts the filename token. The extracted
signal files are unaffected, so no re-extraction is needed.
`strict=False` reproduces the old behaviour, for re-deriving superseded numbers
only.

## What the correction changes — all in the favourable direction

| result | contaminated | **corrected** |
|---|---|---|
| PADS cohort | 41 ET | **28 ET** |
| PADS PD-vs-ET, orbit geometry — AUC | 0.715 | **0.758** |
| — balanced accuracy | 0.690 | **0.747** |
| — ET-F1 | 0.381 [0.27, 0.48] | **0.414 [0.27, 0.53]** |
| PD vs ET peak-frequency overlap | 0.66 | **0.61** |
| — effect size / p | +0.248 / 2.7e-4 | **+0.274 / 7.2e-4** |
| PD / ET median peak frequency | 7.03 / 6.05 Hz | 7.03 / **5.86** Hz |

Every PADS number improves once the parkinsonian cases are removed from ET,
which is exactly what should happen if the contamination was blurring the
boundary.

## Claims that must be revised

1. **"PD and ET are not separated along the frequency axis."** On PADS with
   correct labels, PD 7.03 Hz vs ET 5.86 Hz, MWU **p = 7.2e-4**, effect +0.274.
   The distributions still overlap heavily (0.61), so "overlap is large" stands,
   but "the classes share the same dominant frequency" does **not** hold on
   PADS. That specific 6.64-vs-6.64 Hz identity was a **LOCAL lower_arm**
   observation and must be attributed to the local cohort only, not presented as
   replicated across two cohorts.

2. **"PADS-only, 41 ET, tight CI, ET-F1 0.262 ⇒ the difficulty is intrinsic,
   not a small-cohort artifact."** The cohort was 28, not 41, and a third of it
   was mislabelled. This is the load-bearing evidence for the project's central
   conclusion and is being re-run — see `artifacts/pads_relabel.txt`.

3. Any cross-dataset result in `reports/crossdataset_results.md` computed from
   PADS labels — including the domain-shift and "PADS cannot be training data"
   conclusions — inherits the contamination and needs re-running before it is
   cited.

## Lesson for the rest of the pipeline

Substring matching on clinical free text is unsafe, full stop. Where a published
cohort size exists, **assert it**: had the extractor checked `n_ET == 28`, this
would have failed loudly at extraction time instead of silently propagating into
every downstream result and two reports.
