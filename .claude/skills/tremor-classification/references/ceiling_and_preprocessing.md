# The ceiling, the preprocessing audit, transfer, and the literature

## Where the ceiling is and what shape it has

Splitting the test fold by whether all six ensemble members agree
(`ensemble_diversity.md`, 10 splits):

| | fraction | accuracy | balanced acc | top-2 margin |
|---|---|---|---|---|
| unanimous | 0.595 | 0.688 | — | 0.369 |
| contested | 0.405 | 0.485 | **0.443** | 0.094 |

Always-PD scores 0.465, so the contested 40 % are collectively almost
uninformative. The members are not copies — r(p(ET)) 0.859, argmax disagreement
20.5 %, 23.7 % across the two architectures — so this is a real boundary. It
predicts that any method which only reshuffles the contested set ties, and ten
have (see `closed_families.md`). Even the unanimous 60 % are wrong 31 % of the
time, so the ceiling is not purely a boundary problem.

**Contested rate by cohort** (class composition near-identical, all expect
~0.40): 2015 0.307, PADS 0.432, NewData 0.573. A genuine site effect. Cohort-ID
input does not reduce it.

**Contested rate by mean-frequency tercile**: slow 0.515, mid 0.416, fast 0.253
(20 splits, balanced groups). Class composition explains a spread of 0.049 of the
observed 0.262 — but the within-class effect sizes (rho −0.385 N / −0.241 PD /
−0.051 ET) fall monotonically with each class's own mean frequency (8.16 / 7.51 /
7.04 Hz), which is the signature of class confusion through a monotone class
ordering. The claim that this was a non-circular physical handle is
**retracted**. The one clean physical test — widening the band below 3 Hz —
changed nothing for slow patients.

**Contestedness is predictable from descriptors** (AUC 0.725, not via the class
label), but most of that is circular: descriptors that flip sign between N and
ET are restating the decision boundary. Signal *does* remain in the contested
region — six of eight feature blocks clear chance there, DESC at +0.187
[+0.121, +0.253] balanced accuracy — but nothing tried sees structure the deep
model does not already see.

## The label-noise account (hypothesis, with its test)

Clinical PD-vs-ET misdiagnosis runs 15–35 %; specialist accuracy ~80 %,
non-specialist 74 %; **essential tremor has no gold standard, not even at
post-mortem** (DaTscan meta-analysis, npj Parkinson's Disease 2021; TSI, Brain
2017). The unanimous-60 % accuracy of 68.8 % and the overall macroP ~0.66 sit
where a perfect classifier scored against ~78 %-accurate labels would land.

This is the most seductive excuse in ML and must be measured, not asserted. The
test, using data already in hand: **model-vs-model agreement across a patient's
own recordings** (both PADS wrists, 2015 trials, NewData tasks) against
**model-vs-label agreement**. Signal clean + labels noisy → within-patient
agreement far above 0.66. Within-patient agreement also ~0.66 → the signal is
ambiguous and the account is wrong. Untried; it is the single most decisive
cheap experiment left.

## Preprocessing audit — what each stage keeps and what was checked

Pipeline: quaternion → body-frame angular velocity (central difference, sign
continuity, renormalisation) → `lower_arm` sensor (2015/NewData) or wrist gyro
(PADS) → multitaper nw 2.5 K 4 nperseg 256, 75 % overlap, frames averaged →
interpolated onto a 64-point 3–15 Hz grid → sum-normalised per recording →
mean over the patient's recordings → 16 log-spaced log-power bins. Side
branches: DESC (10 descriptors from `stft512`, **un-normalised**, so
`total_power` is the one place amplitude survives), TRAJ (IF + envelope, 64
points, mean-normalised), ASYM/HAVE. Computed but unused by the reported model:
STAB (TSI), harmonic, axis, ampmod, reemergence features.

Axes each stage discards: proximal–distal gradient (sensor selection), harmonics
above 15 Hz for tremors above 7.5 Hz (band), amplitude (normalisation),
bilateral and task structure (patient mean), everything temporal except the IF
trajectory (frequency marginal).

**Found and fixed.** Frequency axis stretched 1.05 % in `m_multitaper` / `m_sst`
(`axis_fix_audit.md`): performance effect −0.008 [−0.029, +0.010] macroP;
headline re-derived and marginally stronger; welch row bit-identical as it must
be. `_kept_rfftfreq` now asserts.

Two more from the synthetic sweep (`descriptor_trajectory_fix.md`).
`describe()`'s Q-factor spanned every supra-half-max bin rather than the
contiguous peak — a tone with a 0.8-amplitude harmonic read Q 0.94 instead of
15, and the supra-half set is non-contiguous for 85 % of PADS N, 74 % of PD and
30 % of ET, so the old `q_factor` measured "no clear peak" and "secondary
content" in a class-ordered way. Under the contiguous definition **PADS has no
ET-vs-PD Q gap** (ET 21.0 / PD 22.1 / N 22.8; 2015 ratio 1.44 → 1.18). The
headline `peak_sharp` (peak over mean power) is a different quantity and
stands. And the IF trajectory's points 0 and 63 were band-pass transients — a
steady tone read 0.36 Hz of wander at point 0, a ±0.5 Hz FM tone read 2.7 Hz at
point 63, interior correct to 0.06 Hz; a 0.25 s guard removes them. Paired
against the reconstructed pre-fix model: Q fix +0.012 [−0.003, +0.034] macroP,
guard −0.004 [−0.027, +0.021], both −0.006 [−0.026, +0.019]. The headline
survives at +0.044 / +0.104, but **the trajectory stream is no longer
significant on its own** (precET +0.057 * → +0.026 [−0.009, +0.068]); the story
that its old gain was the transients reading the PADS onset was measured and is
wrong (point-0 magnitude ~1.0 Hz for every class, rho +0.03 with the onset).
`experiments/verify_preprocessing.py` is the regression test: 41 checks, exit
code = failures.

**Checked and cleared.** All cohorts are gyroscope angular velocity — no unit
mismatch. Missing detrend in the multitaper path: 5×10⁻⁵ effect, because
quaternion differentiation removes DC. NewData's epoch selection is documented
and its tremor-blind alternative was worse. No sign flips exist in raw NewData
quaternions, so resampling before the sign fix is a latent hazard only.
Noise-dominated recordings: ≤ 8 % of any class. Wrist averaging: between-wrist
peak mismatch is a median 0.78 Hz for N/PD and 0.39 Hz for ET against a 0.39 Hz
bin, but the model sits below the misalignment knee — aligned 0.650, plain
0.652, doubled jitter 0.581 macroP. (The "33 % of the ET–PD sharpness gap"
that motivated the test was measured with the pre-fix span-Q and is withdrawn;
under the contiguous definition PADS shows no ET-vs-PD Q gap at all: ET 21.0,
PD 22.1, N 22.8.)

**Found and open.** Frames averaged differ by cohort (2015 21, PADS 13, NewData
12) purely from recording length. PADS carries an untrimmed arm-raising onset in
the first ~1.5 s that is absent from the other cohorts and class-ordered — first-
1.5 s in-band RMS over the remainder: N 1.39, PD 1.33, ET 1.06 (31 / 18 / 11 % of
recordings above 2×). `extract_pads --trim-start` defaults to 0 on one early
unpaired PADS-only comparison. Trimming collapses the ordering (1.10 / 1.04 /
0.96) and sharpens N and PD peaks slightly, but PADS→in-house PD-vs-ET transfer
is at chance with or without it (0.578 / 0.563 / 0.592, all below the 0.655
floor). The onset is a second-order contamination on top of a cohort gap much
larger than it. Mixed-cohort headline effect of trimming it: macroP −0.006
[−0.031, +0.022], null on every column, while the length-matched *trim-end*
control significantly costs N precision (−0.037 \*) — the tail of a PADS
recording carries something the model uses for N; the head does not. Leave
`--trim-start` at 0 and record the onset as a cohort inconsistency, not a
confound that was fixed (`pads_onset_trim.md`).

## Estimator resolution, measured on a pure tone

| estimator | Q ceiling | 3-class macroP |
|---|---|---|
| ar16 | 31.00 | 0.629 |
| welch n512 | 15.00 | 0.636 |
| **multitaper nw 2.5 K 4 (reported)** | 5.33 | **0.660** |
| multitaper nw 4 K 7 | 2.14 | 0.649 |
| multitaper nw 6 K 11 | 1.36 | 0.644 |

An interior optimum. The headline transform gain is not "smoother is better";
multitaper at nw 2.5 sits at the peak and welch is on the wrong side of it. The
optimal 2W = 1.95 Hz happens to match ET's measured bandwidth of 2.04 Hz —
offered as a coincidence, not a finding.

## Transfer

PD-vs-ET does not transfer between cohorts in either direction
(`pd_vs_et_transfer.md`): fit on PADS, test in-house, no family's CI excludes
0.5; descriptors fall from AUC 0.794 within PADS to 0.519 in-house. Adding PADS
to in-house training degrades in-house PD precision (−0.082). The in-house
detection floor is AUC 0.655 at 21 ET; nothing in-house clears it. Never present
the PADS PD-vs-ET number as a method that works — it is a result about PADS.
Report LOCO for any generalisation claim; a 98 % internal / 79 % external gap is
the published norm for this task.

## The rest-tremor axis is absent from these recordings

The within-patient postural/rest band-power ratio is *positive* for PD (+0.837),
the opposite of the textbook; every contrast feature built on it makes the model
worse; onset latency is ≈ 0 for every class. No protocol here uses a
distraction-based rest condition and PD cohorts are typically ON medication. The
most specific data-collection recommendation this project can make is a proper
rest condition — it may be cheaper than more ET patients.

## Literature positioning

* PADS published baseline: 72.42 % balanced accuracy PD vs DD, 91.16 % PD vs HC
  (Varghese 2024, multimodal). This repo's PADS PD-vs-ET AUC 0.794 is the same
  regime.
* Häring (Mov Disord 2025): 81.8 % PD vs ET on 414 patients from massive
  feature extraction, vs 70.4 % for TSI; their TSI matches ours (0.757 on PADS).
* A 2026 preprint claiming 87 % on PADS uses class-dependent window overlap —
  preprocessing that reads the label; not a comparator.
* For a Transactions submission, the defensible framing is "where the ceiling
  is and why it coincides with label reliability", supported by ~70 disciplined
  negatives with matched controls, rather than "a classifier that reaches 0.66".
  Feature-level cohort harmonisation (ComBat, fitted on train only) is the one
  untried methods contribution with a real chance of moving a number.
