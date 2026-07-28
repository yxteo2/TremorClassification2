# Track 4 — The Drive cohort (`Tremor Classification IMU`): what is in it, and what it costs to use

**Bottom line.** The Drive folder holds **116 subject exports that are not in the
repo** — 6 ET, 51 PD, 59 HC — and their study IDs continue exactly where
`Data/raw_quaternion/` stops. Pooled, that takes **ET from 16 to 22 subjects**
(+38%) with no external dataset and no license constraint. But the exports are
**not** the representation the pipeline trains on, and they carry **no
OUT/REST/WING labels**, so the cohort is not usable as a drop-in. One answer
from the user (how trials map to conditions) decides whether this is a week of
work or a dead end.

## Inventory

Folder: `Tremor Classification IMU/` (Drive, owner `teoyx@tarc.edu.my`), three
class subfolders, one directory per subject:

`Moveo_Explorer_Subject_Export_<GROUP>_<ID>_<yyyymmdd-hhmmssTZ>/`

| group | export dirs | study-ID range | already in repo | new |
|---|---|---|---|---|
| **ET** | 6 | 19, 20, 21, 22, 23, 26 | ET 1–18 (16 subjects) | **all 6** |
| **PD** | 51 | 88–151 (gaps) | PD 1–87 (76 subjects) | **all 51** |
| **HC** → `N` | 59 | 68–126 (contiguous) | N 2–67 (61 subjects) | **all 59** |

No ID in Drive overlaps an ID already in `Data/raw_quaternion/`. Spot-checked
trial counts per subject: ET 21 → 14 trials, PD 88 → ≥13, HC 100 → ≥9, each
~14–39 s. So the cohort is roughly **1.5k trials / ~10 h of signal**.

**Effect on the binding constraint.** Track 3 named 16 ET subjects as the limit
that no model change can move. This folder is the cheapest lever on it:

| class | repo | + Drive | pooled |
|---|---|---|---|
| ET | 16 | 6 | **22** |
| PD | 76 | 51 | 127 |
| N / HC | 61 | 59 | 120 |

It does not replace PADS (22 is still small, and it is still one site, one
device, one clinic), but it is in-house, already consented, and needs no
cross-dataset generalisation argument.

## What an export actually contains

Verified by downloading and opening three exports end-to-end — ET 21 trial 03,
PD 88 trial 09, HC 100 trial 09. All three are structurally identical.

Per subject directory:

| file | content |
|---|---|
| `NN_1_<stamp>_<Condition>_<GROUP>_<localid>_Analysis.h5` | the signal (below) |
| `NN_1_<stamp>SGT_<Condition>_Trial.csv` | ~0.4 kB per-trial summary: subject, `Condition`, `Duration` |
| `NN_1_<stamp>SGT_<Condition>_Trial_Joint_Angles.csv` | the same series as CSV (~0.4 MB) |
| `<Condition>_trials.csv`, `<Condition>_DTSv3.csv`, `..._verbose.csv` | roll-ups |
| `SubjectMetadata.xml` | subject + per-trial manifest, `startDelay`, md5s |
| `DataTransferSpecificationNotes.DTSv3.txt` | exporter format notes |

Inside `*_Analysis.h5`:

```
Measures/Duration                                 (1, 1)    seconds
Processed/Joint Angles          attrs: sampleRate=128.0, nSamples
Processed/Joint Angles/{Elbow,Wrist}/{Left,Right}/
    Quaternion                                    (T, 4)    unit, scalar-first (w,x,y,z)
    X, Y, Z                                       (T, 1)    Euler degrees (group attr `order`)
```

That is **four joint-angle streams** — elbow and wrist, both sides — and nothing
else. There is no accelerometer, no gyroscope, and no per-sensor orientation.

## The five things that will bite

1. **It is joint angles, not the three sensor orientations.** `raw_quaternion`
   is `(T, 12)` = absolute orientation of `(hand, lower_arm, upper_arm)`. The
   export is `(T, 16)` = relative orientation across `(elbow, wrist) x (L, R)`.
   These are different quantities; feeding them to the same model as if the
   channels were interchangeable would train on a dataset-identity artefact, not
   on tremor.
2. **fs = 128 Hz, not 100 Hz.** The exporter declares `sampleRate=128.0`. The
   package defaults `fs=100.0` for quaternion data. Running the export at 100 Hz
   misplaces every frequency by 1.28x — a 5 Hz ET peak reads as 3.9 Hz — and it
   fails silently, exactly the fs trap the skill warns about for the 60 Hz
   amplitude files.
3. **Quaternions are scalar-first `(w, x, y, z)`.** The package convention is
   scalar-last. Confirmed, not assumed: reconstructing intrinsic-YZX Euler
   angles from ET 21's `Wrist/Right` quaternion under the scalar-first reading
   reproduces the stored `X, Y, Z` degrees to 4 decimal places
   (`0.4053, 23.1347, 15.0295`, up to the exporter's sign convention); the
   scalar-last reading gives `-0.3862, -15.1885, -156.9659`, i.e. nonsense.
4. **The first 3 s of every trial is the calibration pose, not the task.**
   `SubjectMetadata.xml` has `startDelay="3"`, and `nSamples/sampleRate -
   Duration` = 3.0 s on all three exports (33.73−30.72, 35.65−32.64,
   38.87−35.86). Left in, every spectrogram starts with a few hundred still
   samples.
5. **The h5 filename carries the wrong subject ID.** PD 88's export directory
   contains files named `..._Free_Form_PD_1_Analysis.h5` — the acquisition
   station's local ID, not the study ID. `PD 1` already exists in the repo, so
   parsing the filename would merge two different patients into one subject and
   break subject-level splitting while every disjointness assertion still
   passes. IDs must come from the directory name or `SubjectMetadata.xml`.

## Measured profile of the signal (19 trials, 14 subjects)

`tremor/moveo_profile.py` computes, per joint per trial: angular-velocity RMS,
where the **whole** spectrum peaks (`global_peak_hz`, 0.5–20 Hz), the strongest
narrowband peak inside 3–12 Hz, and how far that peak rises above a fitted
**1/f^a background** (`peak_excess`; 1.0 = nothing there). Aggregation is
subject-median first, then across subjects.

**A correction to an earlier version of this report.** The first pass took the
raw-power `argmax` inside 3–12 Hz and scored it against the band median. Movement
spectra here fall off monotonically from ~0.5 Hz, so that `argmax` lands on the
**window's lower edge** whenever no real peak exists, and the peak-to-median ratio
*rewards* a steep low-frequency tail. It reported confident "tremor" peaks at
exactly 3.00 Hz — including a 3.25 Hz peak with prominence 28.8 for ET 21 — that
were pure boundary artefacts. `_background()` now divides the spectrum by the
fitted 1/f^a trend before peak-finding, and `peak_at_edge` flags any peak still
sitting on a boundary. The numbers below are the corrected ones. Two conclusions
in the earlier version were wrong and are retracted explicitly further down.

Sample: **19 trials from 14 subjects** — all 6 ET subjects in Drive (ET 19, 20,
21, 22, 23, 26), 5 PD (88, 101, 103, 104, 106) and 3 HC (100, 101, 102). Pulled one
file per API request, so most PD/HC subjects have a single trial. Not a class
comparison; the task confound below is unresolved.

### Where the power actually is

**In 14 of 19 trials the global spectral peak is below 1.5 Hz** (0.50–0.88 Hz).
The exceptions are ET 22 t02 (9.75 Hz), PD 101 (4.25), PD 104 (3.62), PD 106
(2.38) and HC 101 (1.75). So in most recordings the dominant content is voluntary
movement and any tremor is a *secondary* narrowband feature riding on it. Notably
**3 of 5 PD subjects are among the exceptions** — their tremor is a larger share of
total movement — which is what you would expect if those trials were closer to a
rest condition, but with no task labels that stays a guess.

### Per trial, right elbow

| class | subject | trial | global Hz | peak Hz | excess | RMS rad/s | 3–12 Hz frac |
|---|---|---|---|---|---|---|---|
| ET | 19 | 08 | 0.62 | 5.75 | 9.0 | 0.215 | 0.218 |
| ET | 19 | 09 | 0.62 | 6.38 | 10.5 | 0.143 | 0.185 |
| ET | 20 | 09 | 0.88 | 8.00 | 8.3 | 0.071 | 0.289 |
| ET | 21 | 03 | 0.50 | 8.62 | 5.6 | 0.670 | 0.065 |
| ET | 21 | 08 | 0.75 | 5.75 | 7.5 | 0.119 | 0.237 |
| ET | 22 | 02 | **9.75** | 9.75 | **63.6** | 0.064 | **0.758** |
| ET | 22 | 08 | 0.50 | 8.00 | 4.9 | 0.151 | 0.085 |
| ET | 23 | 09 | 0.62 | 4.88 | 4.9 | 0.111 | 0.345 |
| ET | 23 | 10 | 0.50 | 4.00 | 2.9 | 0.541 | 0.092 |
| ET | 26 | 05 | 0.88 | 6.88 | 4.8 | 0.699 | 0.109 |
| ET | 26 | 08 | 0.50 | 11.88 (edge) | 4.2 | 0.155 | 0.221 |
| PD | 88 | 09 | 0.50 | 10.88 | 18.3 | 0.184 | 0.478 |
| PD | 101 | 13 | **4.25** | 8.50 | **22.6** | 0.391 | 0.746 |
| PD | 103 | 12 | 0.62 | 4.50 | 3.0 | 0.371 | 0.190 |
| PD | 104 | 09 | **3.62** | 7.88 | 13.7 | 0.100 | 0.811 |
| PD | 106 | 09 | **2.38** | 5.88 | 15.1 | 0.136 | 0.366 |
| HC | 100 | 09 | 0.50 | 8.38 | 8.5 | 0.113 | 0.244 |
| HC | 101 | 09 | 1.75 | 10.88 | 3.8 | 0.089 | 0.570 |
| HC | 102 | 09 | 0.50 | 7.62 | 5.4 | 0.157 | 0.564 |

### Class level, subject as the unit — median [IQR]

| metric | ET (n=6) | PD (n=5) | N/HC (n=3) |
|---|---|---|---|
| peak Hz | 7.59 [6.34, 8.66] | 7.88 [5.88, 8.50] | 8.38 [8.00, 9.62] |
| **peak excess** | 7.45 [5.02, 9.39] | **15.11 [13.70, 18.32]** | 5.35 [4.58, 6.93] |
| RMS rad/s | 0.25 [0.13, 0.38] | 0.18 [0.14, 0.37] | 0.11 [0.10, 0.14] |
| 3–12 Hz frac | 0.21 [0.17, 0.27] | 0.48 [0.37, 0.75] | 0.56 [0.40, 0.57] |

### What holds up

**The adapter reads the data correctly.** ET 19 trial 08 peaks at **5.75 Hz on all
four joints** (spread 0.00 Hz, excess 6.8–9.9); ET 22 trial 02 at **9.62–9.75 Hz on
all four** with excess up to 63.6, the one trial where tremor is also the global
maximum; PD 88 at **10.88 Hz on all four** (spread 0.00). A single frequency shared
across both elbows and both wrists is a whole-limb oscillation, which a mis-parsed
quaternion or a wrong sample rate does not produce. The 128 Hz / scalar-first
handling is sound.

**Peak frequency does not separate the classes — at all.** ET 7.59, PD 7.88, HC
8.38 Hz, with IQRs almost entirely overlapping. This is now on 14 subjects rather
than 4. Consistent with the clinical overlap and with why the repo's ET-vs-PD
separation has been hard: no peak-frequency feature is going to carry it.

**`tremor_frac` is confounded by how much the subject moved, and must not be read
as tremor severity.** HC scores *highest* on it (0.56) and ET lowest (0.21) — not
because controls tremor more, but because they move least overall (RMS 0.11 vs ET
0.25), so the low-frequency voluntary component that sits in the denominator is
smaller. Any severity or screening score built on a band-power *ratio* over these
recordings would rank healthy controls as the most tremulous group. Use
`peak_excess`, which is normalised against the recording's own 1/f background, not
`tremor_frac`.

**The one descriptor that does show group structure is `peak_excess`** — how far
the narrowband peak rises above each recording's own background. PD sits at 15.11
[13.70, 18.32] against ET 7.45 [5.02, 9.39] and HC 5.35 [4.58, 6.93], and 4 of the
5 PD subjects fall in 13.7–22.6 while 5 of 6 ET subjects fall in 3.9–9.8. Treat
this as a lead worth testing, **not a result**: n=14, most PD/HC subjects contribute
one trial, and the task confound is completely unresolved — 3 of 5 PD subjects also
have their global peak above 2 Hz, so "PD has sharper peaks" and "these particular
PD trials happened to be closer to rest" are indistinguishable in this sample.

**Retracted: "PD 88 shows 5.00 Hz on both elbows, squarely PD rest-tremor
frequency."** That was the boundary artefact. Background-corrected, PD 88's
strongest narrowband excess is at **10.88 Hz (×18.3, all four joints)**, with a
secondary at 9.50 Hz and nothing prominent near 5 Hz.

**Retracted: "the two ET subjects straddle the PD subject in frequency."** That
rested on ET 21's bogus 3.25 Hz. With 14 subjects there is no frequency ordering
between the groups to speak of — see the class table.

**Tremor frequency is only partly a stable subject trait here.** Across their two
trials, ET 19 (0.62 Hz) and ET 23 (0.88 Hz) hold their peak frequency; ET 22
(1.75 Hz), ET 21 (2.88 Hz) and ET 26 (5.00 Hz) do not.

### The part that blocks everything

RMS and band-fraction do not depend on the peak-finding, so this conclusion is
unaffected by the correction above. Both subjects with two trials disagree with
*themselves* more than the groups disagree with each other. Right wrist, same
subject, same session, minutes apart:

| subject | trial | RMS rad/s | 3–12 Hz frac |
|---|---|---|---|
| ET 21 | 03 | 0.222 | 0.119 |
| ET 21 | 08 | **0.073** | **0.354** |
| ET 19 | 08 | 0.213 | 0.250 |
| ET 19 | 09 | **0.064** | **0.511** |

Both drop ~3× in amplitude and double or triple their tremor-band fraction from
one trial to the next. On the left elbow ET 21 goes 0.784 → 0.190 rad/s. Across the
ET cohort peak RMS per trial ranges 0.17–1.15 rad/s — a 7× span *within one
diagnosis*. The low-amplitude/high-band-fraction trials look like a held posture
with the tremor exposed; the high-amplitude ones look like gross voluntary movement
burying it. That is the signature of two *different tasks* under one `Free Form`
label, and it also explains why peak frequency fails to reproduce across trials
for four of five subjects.

So the missing condition labels are not a bookkeeping inconvenience to work around
later. **The unlabelled task is the dominant source of variance in this data** —
larger than the between-group contrasts anyone would want to measure. Any class
comparison, any pooled training run, and any ET-vs-PD frequency claim built on
these trials before they are mapped to tasks would be measuring task, not
pathology. The retractions above are a live demonstration: two plausible-looking
class findings, both artefacts.

## The same profile on the repo's own data — n = 15/75/61, and tasks are labelled

The Drive cohort caps at **6 ET subjects**, so no amount of downloading reaches
20 ET from there. The repo's own `raw_quaternion` data does have the numbers, and
it has the one thing the exports lack: **task labels**. Harmonised to elbow/wrist
(`joint_quaternions_from_sensors`) it can be profiled with the identical
descriptors, so the two cohorts are directly comparable:

```bash
python -m tremor.moveo_profile --local-root Data --action OUT
```

Subjects per condition: **OUT** ET 15 / PD 75 / N 61 · **REST** ET 16 / PD 75 /
N 61 · **WING** ET 13 / PD 63 / N 61. Median [IQR] of subject medians, elbow:

| condition | metric | ET | PD | N | ET-vs-PD |
|---|---|---|---|---|---|
| OUT | peak Hz | 7.38 [6.22, 8.47] | 7.81 [5.75, 9.22] | **9.50 [8.50, 10.62]** | p=0.90 |
| OUT | peak excess | 19.82 [10.2, 46.6] | 17.68 [9.7, 46.3] | **6.51 [4.8, 10.2]** | p=0.85 |
| OUT | 3–12 Hz frac | 0.73 [0.60, 0.79] | 0.74 [0.61, 0.87] | **0.50 [0.44, 0.59]** | p=0.61 |
| REST | peak Hz | 6.84 [6.38, 8.30] | 6.62 [5.25, 7.47] | 7.50 [6.75, 9.19] | p=0.15 |
| REST | peak excess | 15.97 [10.9, 22.0] | 17.04 [8.4, 33.8] | **8.68 [5.6, 13.0]** | p=0.82 |
| WING | peak excess | 15.43 [8.0, 54.2] | 9.73 [6.1, 36.5] | **5.25 [4.2, 6.5]** | p=0.33 |

Mann–Whitney U on subject-level values, ET vs PD, two-sided.

**Patients separate cleanly from controls.** In every condition and at both
joints, ET and PD sit at 2–3× the controls' `peak_excess` and ~0.65–0.75 vs ~0.50
tremor-band fraction. And controls' peak frequency is *higher* — 9.50 Hz in OUT
against 7.4–7.8 for patients — which is exactly the textbook split between normal
physiological tremor (8–12 Hz) and pathological tremor (4–8 Hz). The descriptors
are measuring something real.

**ET vs PD does not separate on any of them, in any condition.** Every ET-vs-PD
p-value is ≥0.09 across 24 comparisons (2 joints × 3 conditions × 4 metrics), and
the medians are nearly identical — OUT elbow 7.38 vs 7.81 Hz, excess 19.8 vs 17.7,
frac 0.73 vs 0.74. The one nominal hit (REST wrist RMS, p=0.026) does not survive
any correction for 24 tests. With **15–16 ET and 63–75 PD subjects** this is no
longer a small-sample excuse: these spectral summaries carry the N-vs-patient
distinction and essentially none of the ET-vs-PD distinction. That is a direct,
quantified explanation of why the repo's ET F1 has been the hard number, and it
says the discriminative information — if it is there — is not in peak frequency,
peak sharpness, or band-power ratio.

**The labelled data is also much cleaner than Free Form.** Only **14–18%** of these
recordings have their global spectral peak below 1.5 Hz, against **74%** of the
Drive Free Form trials, and band-edge peaks are 2–4% against 5%. Held postures and
rest put the tremor where it can be measured; unlabelled free movement buries it.
That is the cost of the missing condition labels, in one number.

Two caveats on these figures. Frequencies scale linearly with the assumed
**fs=100 Hz**, which is still unverified (see below) — if the true rate is 128 Hz
every frequency here is 1.28× low, though the ET-vs-PD null and the patient-vs-N
contrast are unaffected because both would shift together. And RMS is 0.01–0.04
rad/s here against 0.06–0.70 in the exports, a ~10× gap that is partly task
(held posture vs free movement) and partly whatever undocumented processing the
`.txt` files went through.

### What this means for "20 subjects per class"

| class | repo `raw_quaternion` | Drive exports | pooled ceiling |
|---|---|---|---|
| ET | **16** (15 OUT, 16 REST, 13 WING) | 6 | **22** |
| PD | 76 | 51 | 127 |
| N / HC | 61 | 59 | 120 |

PD and N clear 20 in either source alone. **ET does not exist in 20-subject
quantity in either source** — 16 locally, 6 in Drive. Reaching 20+ ET means
pooling both, which needs the elbow/wrist harmonisation above *and* an answer on
the Free Form task labels, or it means external data (Track 3's PADS route, which
adds 28 ET). Those are the only two paths to n≥20 ET; more downloading is not one
of them.

## Two blockers

### 1. No condition labels — the one that needs an answer from you

Every trial in every export is `conditionName="Free Form"`, `testName="Free
Form"`, with `<notes/>` and `Trial Notes` empty. There is nothing in the export
that says which trial was outstretched, at rest, or wing-beating. The repo's
entire structure is per-condition (`OUT/`, `REST/`, `WING/`), and Track 1's
results are per-condition.

The gap is not cosmetic, and the section above quantifies it: two trials from the
same ET subject differ 4× in RMS and 11× in peak sharpness. Separately, ET 21's
trial 03 runs 0.22–0.78 rad/s per joint against ET 10's `REST` recording at
0.03–0.10 rad/s — different subjects, so not controlled, but an order of magnitude
apart. These trials are not REST under another name, and they are not all the same
thing as each other either.

**What we need:** the session log or protocol that says what the 9–14 trials per
subject were. Three plausible shapes, each with a different cost:

- *A fixed protocol order* (e.g. trials 1–4 = OUT, 5–8 = REST, …) → the numeric
  prefix `NN_1_` gives the mapping directly, and the cohort becomes usable
  immediately.
- *An external log* (spreadsheet, notebook, the Moveo web app's session notes)
  → same outcome, one join away.
- *No record* → the trials can still be used for condition-agnostic work
  (self-supervised pretraining, or a "any-task" classifier), but they cannot
  join the per-condition results, and they cannot be pooled into the
  cross-condition runs without inventing labels.

### 2. The raw per-sensor h5 is not in the export

`SubjectMetadata.xml` references the raw trial file
(`20250508-091102_Free_Form_ET_21.h5`) for every trial, but the export only ships
`*_Analysis.h5`. If those raw files can be re-exported from Moveo Explorer, they
would contain per-sensor orientation and remove blocker 1's representation
mismatch entirely — that is the single highest-value thing to ask the recording
site for, and it is worth asking before building anything on joint angles.

## Harmonisation, if the raw files cannot be recovered

Joint angles are derivable from segment orientations, so the conversion runs
*old → new*, not the other way:

```
elbow = conj(upper_arm) * lower_arm
wrist = conj(lower_arm) * hand
```

`tremor.moveo_data.joint_quaternions_from_sensors` implements this and turns the
existing `(T, 12)` files into `(T, 8)` = `(elbow, wrist) x 4`, which lines up
with the export's elbow/wrist streams for one arm. That gives a common
representation both cohorts can express.

Caveat, stated plainly: the parent-relative direction above is the standard
definition, but the exporter's own sign and axis conventions for joint angles
could not be validated from the exports, because the two cohorts share no
subject — there is nothing to cross-check against. Before trusting any pooled
result, run the **dataset-identity probe** from Track 3: train a classifier to
predict which cohort a recording came from. If it succeeds easily, the pooled
disease result is measuring provenance, not pathology.

Also unresolved: the export instruments **both arms**, the repo's data is one
arm. Which side to use (dominant, more-affected, or higher tremor-band power)
is an empirical choice that has to be made once and documented.

## Worth verifying: is the existing data really 100 Hz?

The package hard-codes `fs=100.0` for `raw_quaternion`. The Moveo/APDM exporter
that produced this study's recordings declares **128 Hz**. If the existing
`.txt` files came off the same system, `fs=100` would be wrong by 1.28x, and
every tremor-band frequency reported so far would be shifted with it.

This is a question, not a finding — the `.txt` files carry no header, no
timestamps and no sample rate, so nothing in the repo can settle it, and the
file lengths are consistent with either reading (1484 samples = 14.8 s at 100 Hz,
11.6 s at 128 Hz). It is cheap to settle against the original source files or the
export script, and expensive to leave open. Worth doing before the next round of
frequency-domain claims.

Secondary observation: the local `.txt` quaternions are not exactly unit norm
(mean ‖q‖ = 1.0055 / 1.0060 / 0.9976 per sensor on `ET 10_REST 1.txt`), while the
export's are 1.000000. Harmless for angular velocity, since the package
normalises, but it means the `.txt` files went through some processing step that
is not documented anywhere in the repo.

## Privacy — read before syncing anything into the repo

The exports contain **directly identifying patient data**: `SubjectMetadata.xml`
carries first name, last name, date of birth and height per subject, and each
`_Trial.csv` repeats the name and DOB. Two real names and two DOBs were visible
in the three files opened for this assessment.

- **Do not commit any part of the export tree to git.** Not the XML, not the
  `_Trial.csv` files, not "just one sample". This repo is on GitHub.
- `tremor.moveo_data` reads **only** the `*_Analysis.h5` signal arrays and the
  directory name. It never opens `SubjectMetadata.xml` or the trial CSVs, and it
  never writes a name, DOB or height anywhere.
- Subjects are namespaced `MV-<GROUP><ID>` (e.g. `MV-ET21`) — study IDs only,
  and distinct from repo IDs so pooling cannot collide.
- If the export tree is synced locally, put it **outside** the repo, or add the
  path to `.gitignore` first. A separate de-identified copy (drop the XML and
  the `_Trial.csv` files) is the safer thing to hand to anyone else.

## The adapter

`tremor/moveo_data.py`, tested end-to-end on the three downloaded trials:

```bash
# summarise a synced tree without loading signals
python -m tremor.moveo_data --root "/path/to/Tremor Classification IMU" --inventory
python -m tremor.moveo_data --root ... --inventory --out moveo_inventory.csv
python -m tremor.moveo_data --root ... --groups ET      # load, report shapes
```

```python
from tremor.moveo_data import load_moveo_recordings, moveo_inventory

recs = load_moveo_recordings(root, groups=["ET"])   # -> list[Recording]
recs[0].subject, recs[0].condition                  # 'MV-ET21', 'FREEFORM'
recs[0].x.shape                                     # (12, T) = 4 joints x 3 rad/s
```

and `tremor/moveo_profile.py` for the descriptive sweep:

```bash
python -m tremor.moveo_profile --root ... --per-subject          # full profile
python -m tremor.moveo_profile --root ... --groups ET --out et.csv
```

The loader defaults to `fs` read from the file (128 Hz), reorders scalar-first →
scalar-last, trims the 3 s calibration pose, drops trials under 5 s, derives the
subject ID from the directory, and labels every trial `condition='FREEFORM'`
because that is the only honest label available. `moveo_inventory()` reports
trials, total seconds, sample rates and joint streams per subject so the real
per-subject counts can be checked once the folder is synced.

Getting the data local: the folder is ~1.5k files of ~1.1–1.4 MB (≈2 GB), which
is not practical to pull one file at a time through the Drive API. Sync it with
Drive for Desktop or `rclone copy`, then point `--root` at it.

## Next actions, in order

1. **Ask the recording site for the raw per-sensor trial h5 files** (the ones
   `SubjectMetadata.xml` names). If they exist, blockers 1 and 2 both shrink to
   an fs conversion.
2. **Find the condition mapping** for the Free Form trials — protocol order,
   session log, or a definitive "there isn't one". Nothing per-condition can be
   built until this is answered.
3. **Settle the 100 vs 128 Hz question** for the existing `raw_quaternion`
   files.
4. Sync the tree outside the repo, run `--inventory`, and confirm the real
   per-subject trial counts against the table above.
5. Only then: harmonise to elbow/wrist, run the dataset-identity probe, and if it
   comes back clean, re-run ET-LOSO at n=22 with the subject bootstrap CI.
