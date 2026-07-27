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

## Two blockers

### 1. No condition labels — the one that needs an answer from you

Every trial in every export is `conditionName="Free Form"`, `testName="Free
Form"`, with `<notes/>` and `Trial Notes` empty. There is nothing in the export
that says which trial was outstretched, at rest, or wing-beating. The repo's
entire structure is per-condition (`OUT/`, `REST/`, `WING/`), and Track 1's
results are per-condition.

The gap is not cosmetic. Spot-check: ET 21's Free Form trial 03 has per-joint
angular-velocity RMS of 0.19–1.05 rad/s, while ET 10's `REST` recording sits at
0.03–0.10 rad/s. Different subjects, so not a controlled comparison — but an
order of magnitude apart is consistent with the Free Form trials containing
gross voluntary movement rather than a held posture. They are not REST by
another name.

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

It defaults to `fs` read from the file (128 Hz), reorders scalar-first →
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
