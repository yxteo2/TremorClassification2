# Verification sweep — cohorts, filtering, and a duration confound

Run after the DD/atypical-parkinsonism filtering question.

## 1. Diagnosis filtering — clean in both PADS folders

| folder | diagnoses present | N / PD / ET |
|---|---|---|
| `pads_relaxed` | `Healthy`, `Parkinson's`, `Essential Tremor` — nothing else | **79 / 276 / 28** |
| `pads_stretchhold` | same | **79 / 276 / 28** |

Both match **Varghese et al. 2024** exactly, which is the independent check that
the filter is right. Screened against every parkinsonian-mimic keyword
(atypical, hypokinetic, vascular parkinson, Lewy, PSP, MSA, spinocerebellar,
dystonia, SWEDD, leukoencephalopathy, unknown etiology, functional): **zero
hits**. The 21 mimics that the old substring mapping had put inside **PD** are
gone.

Both folders cover the **identical 383 patients**, so Relaxed-vs-StretchHold is
a within-subject comparison.

## 2. Borderline ET — excluding them is free

Three DD diagnoses contain the string "Essential Tremor":
`Essential Tremor, DD functional Tremor`;
`Cervical dystonia with Torticollis with tremor. Essential Tremor`;
`Starting IPS DD, age dependent tremor DD, asymmetric Essential Tremor`.

| variant | n ET | welch | cwt | stft512 | ET-F1 |
|---|---|---|---|---|---|
| **strict** (exact match) | 28 | **0.734** | **0.774** | **0.749** | 0.458 |
| loose (+3) | 31 | 0.710 | 0.766 | 0.739 | 0.457 |

Including them is slightly worse everywhere and leaves ET-F1 unchanged. The
third case lists **IPS (idiopathic Parkinson syndrome)** as a differential —
calling that ET would be actively wrong. Strict stays.

## 3. A duration confound, found and ruled out

The two PADS tasks are **not the same length**:

| cohort | samples | duration |
|---|---|---|
| PADS **Relaxed** | 2048 | 20.5 s |
| PADS **StretchHold** | 1024 | 10.2 s |
| 2015 REST | ~1544 | 15.4 s |

Longer records give smoother Welch estimates, so Relaxed had an *advantage* in
the earlier comparison — and still lost. Tested directly by truncating Relaxed
to 1024 samples:

| variant | welch | cwt | stft512 |
|---|---|---|---|
| Relaxed, full 2048 | 0.675 | 0.641 | 0.671 |
| Relaxed, truncated to 1024 | 0.667 | 0.628 | 0.642 |
| **StretchHold, 1024** | **0.734** | **0.774** | **0.749** |

Truncation makes Relaxed slightly *worse*, not better. **StretchHold genuinely
beats Relaxed on PADS by 0.07–0.13, independent of recording length.** The
earlier conclusion — that the REST advantage is specific to the 2015 cohort —
survives duration control.

## 4. Local cohorts unchanged

| cohort | condition | recordings | patients |
|---|---|---|---|
| 2015 | REST | 275 | 152 |
| 2015 | OUT | 274 | 151 |
| NewData (segmented) | REST | 12 | 6 |
| NewData (segmented) | OUT | 12 | 6 |

## Housekeeping still outstanding

`pads_stretchhold/` holds 1598 `.txt` files: 766 correctly task-named plus 832
legacy-named leftovers from the pre-fix extraction. Loading is unaffected
(`strict=True` takes classes from the manifest, which lists only the 766), but
they are dead weight and would corrupt any `strict=False` load. Safe to delete
every file there whose name has only three underscore-separated fields.
