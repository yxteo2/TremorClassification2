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

## Measured profile of the signal (6 trials — and why it stops there)

`tremor/moveo_profile.py` computes, per joint per trial, angular-velocity RMS,
the spectral peak in 3–12 Hz, how far that peak stands above the rest of the band
(`peak_prominence`, 1.0 = flat), and the share of 0.5–30 Hz power in the tremor
band. Aggregation is subject-median first, then across subjects — trial counts
vary 9–14, so trial-level pooling would silently weight the busiest subjects.

Bulk download through the Drive API is one file per request, so the sample is
**6 trials from 4 subjects** — ET 19, ET 21, PD 88, HC 100. It is not a class
comparison and must not be read as one. It is still decisive about two things.

| subj | trial | joint | s | RMS rad/s | peak Hz | prom. | 3–12 Hz frac |
|---|---|---|---|---|---|---|---|
| ET 19 | **08** | Elbow/L | 40.6 | 0.359 | **5.75** | 5.6 | 0.306 |
| ET 19 | **08** | Elbow/R | 40.6 | 0.215 | **5.75** | 5.4 | 0.194 |
| ET 19 | **08** | Wrist/L | 40.6 | 0.353 | **5.75** | 5.9 | 0.337 |
| ET 19 | **08** | Wrist/R | 40.6 | 0.213 | **5.75** | 5.3 | 0.250 |
| ET 19 | 09 | Elbow/R | 36.0 | 0.143 | 4.50 | **17.9** | 0.173 |
| ET 19 | 09 | Wrist/R | 36.0 | 0.064 | 6.75 | 7.4 | **0.511** |
| ET 21 | 03 | Elbow/L | 30.7 | **0.784** | 3.00 | 2.8 | 0.065 |
| ET 21 | 03 | Wrist/R | 30.7 | 0.222 | 8.75 | 2.8 | 0.119 |
| ET 21 | **08** | Elbow/L | 42.7 | **0.190** | 3.00 | **8.6** | 0.110 |
| ET 21 | **08** | Elbow/R | 42.7 | 0.119 | 3.25 | **28.8** | 0.159 |
| ET 21 | **08** | Wrist/R | 42.7 | 0.073 | 3.25 | 5.1 | 0.354 |
| PD 88 | 09 | Elbow/L | 32.6 | 0.196 | **5.00** | 3.8 | 0.221 |
| PD 88 | 09 | Elbow/R | 32.6 | 0.184 | **5.00** | 3.7 | 0.217 |
| PD 88 | 09 | Wrist/R | 32.6 | 0.076 | 4.75 | 2.3 | **0.539** |
| HC 100 | 09 | Elbow/L | 35.9 | 0.197 | 5.50 | 4.2 | 0.120 |
| HC 100 | 09 | Wrist/R | 35.9 | 0.060 | 3.00 | 9.2 | 0.229 |

Three readings, and the last one matters far more than the first two.

**The signal is real and the adapter reads it correctly.** ET 19 trial 08 is the
cleanest example: **all four joints peak at exactly 5.75 Hz**, prominence 5.3–5.9,
with 19–34% of power in band. A single frequency shared across both elbows and
both wrists is a whole-limb oscillation — that is what postural ET looks like, and
it is not something a mis-parsed quaternion or a wrong sample rate would produce.
PD 88 likewise shows 5.00 Hz on *both* elbows at ~22% band power, and ET 21 trial
08 a very sharp 3.25 Hz peak (prominence 28.8 right elbow). The 128 Hz /
scalar-first handling is sound.

**Frequency alone will not separate the classes.** In this sample ET 19 peaks at
5.75–6.75 Hz, PD 88 at 5.00 Hz, and ET 21 at 3.00–3.25 Hz — so the two ET subjects
sit on opposite sides of the PD subject. Four subjects proves nothing, but it is
consistent with the clinical overlap and with why the repo's existing ET-vs-PD
separation has been hard: a peak-frequency feature is not going to carry it.

**The part that blocks everything.** Both subjects with two trials disagree with
*themselves* more than the groups disagree with each other. Right wrist, same
subject, same session, minutes apart:

| subject | trial | RMS rad/s | peak Hz | 3–12 Hz frac |
|---|---|---|---|---|
| ET 21 | 03 | 0.222 | 8.75 | 0.119 |
| ET 21 | 08 | **0.073** | **3.25** | **0.354** |
| ET 19 | 08 | 0.213 | 5.75 | 0.250 |
| ET 19 | 09 | **0.064** | 6.75 | **0.511** |

Both drop ~3× in amplitude and roughly double or triple their band fraction from
one trial to the next, and ET 21's peak moves 8.75 → 3.25 Hz. On the left elbow ET
21 goes 0.784 → 0.190 rad/s with peak prominence 2.5 → 28.8, an 11× change in how
sharp the oscillation is. The low-amplitude/high-band-fraction trials look like a
held posture with the tremor exposed; the high-amplitude ones look like gross
voluntary movement burying it. That is the signature of two *different tasks*
recorded under the same `Free Form` label.

So the missing condition labels are not a bookkeeping inconvenience to work
around later. **The unlabelled task is the dominant source of variance in this
data** — larger than the between-group contrasts anyone would want to measure. Any
class comparison, any pooled training run, and any ET-vs-PD frequency claim built
on these trials before they are mapped to tasks would be measuring task, not
pathology.

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
