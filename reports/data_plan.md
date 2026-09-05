# A plan for the data, derived from what the modelling has closed

Seventy-two experiments have not moved the reported model past precET 0.654 /
macroP 0.652. The negatives are not all equal, and read together they say
something specific: **the constraint is the data, and within the data it is ET
patients and label quality — in that order of certainty, though not necessarily
of cost.** This is the plan that follows from them.

## 0. The gate: one measurement decides where the money goes

Two accounts of the ceiling are both consistent with everything measured, and
they imply **opposite** spending:

| account | implication |
|---|---|
| **label noise** — clinical PD-vs-ET misdiagnosis runs 15–35 % and ET has no gold standard even post-mortem; the measured ceiling coincides with that rate | adjudicate the labels on the 404 patients we already have; more patients with equally noisy labels buys proportionally less |
| **signal insufficiency** — the 3–15 Hz wrist spectrum is a near-sufficient statistic and simply does not separate PD from ET in ~40 % of patients | more patients and richer acquisition; re-labelling changes nothing |

**The discriminating test is already named in `SKILL.md` as untested:** does the
model agree with *itself* across a patient's own recordings far more than it
agrees with the label? Label noise predicts high self-consistency with wrong
labels. Signal insufficiency predicts the model is inconsistent on the same
patient too.

It costs about one experiment — one fit set per split, scoring each recording
separately instead of averaging to the patient. **Run this before committing to
either branch below.** Every patient in this project has ≥ 1 repeat recording
(2015 and NewData repeat the same arm; PADS gives both wrists), so the
measurement is available today at no collection cost.

## 1. ET patients — the number, and why this number

Everything downstream is set by **49 ET patients**, ~10 of which reach a test
fold. `_sample_size_planning.py` computes what that costs us:

| ET total | PD-vs-ET null p95 | precET step size | resolution, 40 splits |
|---|---|---|---|
| 21 (in-house only) | **0.757** | 0.317 | 0.084 |
| **49 (today)** | **0.676** | **0.136** | **0.055** |
| 75 | 0.640 | 0.089 | 0.045 |
| **100** | 0.638 | 0.067 | 0.039 |
| 150 | 0.618 | 0.044 | 0.032 |
| 300 | 0.597 | 0.022 | 0.022 |

Three things fall out of this table.

**We cannot currently detect the improvements we are looking for.** Every
candidate measured since the headline lands at +0.02 to +0.04 precET. At 49 ET,
40 splits resolves 0.055. **The instrument is coarser than the effects.** That
is not a modelling failure; it is a sample-size fact, and it means some of the
21 recorded failures may be real gains we could not see.

**In-house PD-vs-ET is unmeasurable, not merely hard.** At 21 ET the chance
ceiling is AUC 0.757 — higher than almost any effect in the literature. No
in-house PD-vs-ET claim can be made at this n, and none should be attempted.

**More ET beats more splits, decisively.** Going 40 → 80 splits at 49 ET buys
0.055 → 0.039 for 2× the compute and *no new information*. Going 49 → 150 ET at
40 splits buys 0.055 → 0.032, and the extra information also raises the model
rather than only sharpening the ruler.

**Target: +100 ET patients (to ~150).** That is the point where precET's step
size (0.044) finally drops below the effect sizes we keep chasing, and where the
null falls to 0.618. Below ~75 the table barely moves; above ~150 returns
flatten.

**Do not collect more N or PD.** At 167 and 188 they are not the constraint, and
cropped training settled the related question: **9.2× more training rows was
null**, so rows are not scarce — *patients of the minority class* are.

## 2. Label adjudication — what to record, on old and new patients alike

Conditional on the gate in §0 pointing at labels, and cheap enough to specify
now either way:

* **Multi-rater diagnosis with the agreement retained**, not collapsed. A
  patient two neurologists disagree on is data, not noise — `prune_training.md`
  showed the hardest patients are *boundary-defining* and dropping them is
  significantly worse than dropping random ones.
* **Diagnosis stability at follow-up.** ET→PD revision is common; a label taken
  at recording time and never revisited is the single most likely source of the
  irreducible 40 %.
* **Supporting evidence where it exists** — levodopa response, DaTSCAN. These
  are what "gold standard" means in practice for this contrast.
* **Per-patient confidence**, so severity- and confidence-stratified reporting
  becomes possible. `SKILL.md` lists severity-stratified reporting as open
  purely because the covariate was never recorded.

## 3. Acquisition protocol for anything collected from here

These cannot be retrofitted, so they are worth fixing before the next patient is
recorded, independent of §1 and §2.

* **Fixed, marked sensor orientation.** The orientation work measured a
  patient's tremor axis as reproducible to **15.8°** within-patient against
  40.2° between — so mounting consistency is achievable. It is not currently
  consistent *across cohorts*, which is the only reason the whole tangent-space
  analysis had to be run within cohort. Consistent mounting would make that
  information poolable. (It is redundant with the descriptors on PD-vs-ET, so
  this is hygiene, not a new feature — see `riemann_axes.md`.)
* **≥ 30 s of sustained postural recording.** 2015's median is 15.5 s and PADS
  is a fixed 10.24 s. HPSS established directly that **the class information
  lives in the sustained component**, so recording more sustained signal is
  on-target in a way that adding channels is not. It also equalises the
  cohort-dependent frame averaging (21 / 13 / 12) that is currently an open
  known issue.
* **Both wrists, always.** Only PADS is bilateral today, so `ASYM` is
  structurally zero for all of 2015. PADS's two wrists differ in tremor axis as
  much as two random patients do (39.5° vs 38.3°), i.e. there is genuine
  per-limb information, and bilateral asymmetry is a classical PD-vs-ET
  contrast.
* **Rest, postural and kinetic in one fixed battery.** `task_averaging.md`
  measured that extra tasks help precN (+0.047 \*) and hurt precET (−0.104 \*);
  `axis_specific_inputs.md` got macroP +0.011 by giving each decision its own
  input. That design is only usable if every patient has every task. A rest
  recording long enough to show **re-emergent tremor latency** (5–10 s) would
  add a genuinely PD-specific marker the current data cannot express.
* **Clinical covariates**: UPDRS/TETRAS, age, disease duration, medication
  state. Amplitude is deliberately normalised away by the pipeline
  (`spectral_representation.md`), so severity is currently invisible to us
  *and* unrecorded — we cannot even check whether the contested 40 % are the
  mild cases.

## 4. Fix in the data we already hold

* **NewData resamples quaternions before the sign-continuity fix.** Latent today
  — no flips exist in the raw streams — but it is a landmine for any future
  NewData recording. Cheap to reorder.
* **Frame-count harmonisation** (21 / 13 / 12 across cohorts) is untested. A
  matched arm equalising it would say whether the cohort-dependent contested
  rate (2015 0.307, PADS 0.432, NewData 0.573, class mix controlled) is partly
  an artefact of how many frames each cohort averages.

## 5. What not to spend on

Each of these is closed by a measurement, and each would be a natural thing to
buy:

* **More recordings per existing patient.** Cropped training: 9.2× rows, macroP
  +0.016 while losing 11 of 20 splits.
* **Higher sampling rate or more sensor channels.** The band is 3–15 Hz sampled
  at 100 Hz — oversampled by an order of magnitude. The spectrum is
  near-sufficient: orientation turned out redundant with the descriptors, and
  every richer representation tried has been null or worse.
* **More external heterogeneous cohorts, expecting them to help in-house.**
  PADS→in-house PD-vs-ET transfer is AUC 0.578, *below* the 0.655 chance floor,
  yet dropping PADS collapses the merged precET from 0.519 to 0.065. External
  data props up the merged headline while doing nothing for the clinical target.
  Adding a fourth cohort would likely repeat that pattern — a better merged
  number and no better in-house prediction. **Decide which of those two numbers
  the paper is actually claiming before buying more of either.**

## Ordering

1. **Run the self-consistency gate (§0).** One experiment, existing data, decides §2's priority.
2. **Fix the acquisition protocol (§3)** before the next patient is recorded — free now, impossible later.
3. **Collect ET patients toward ~150 (§1)**, with §2's label protocol applied to new and existing patients alike.
4. **Fix §4** opportunistically; both are cheap.

The honest framing for a Transactions submission is unchanged and is
strengthened, not weakened, by this plan: *where the ceiling is, why it
coincides with label reliability, and what it would take to move it* — with a
sample-size analysis showing the field's usual "we tried another architecture"
cannot work at n = 49 ET, because the effects sought are smaller than the
measurement resolution.
