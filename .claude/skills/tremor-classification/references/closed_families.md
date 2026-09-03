# Closed families — do not re-try without new evidence

Every row was measured against the reported model with a paired bootstrap over
≥ 20 patient-level splits and, where the idea had an obvious confound, a matched
control that isolates it. Numbers are macroP / precET deltas unless stated;
`*` means the interval excludes zero. The report named in each row has the full
table.

## Decision rule and ensemble (all null — the members already disagree on 20.5 % of patients and the contested ones are at chance)

| tried | result | report |
|---|---|---|
| 7 pooling rules over the six members (geometric, median, trimmed, temperature-scaled, family-weighted) | all within ±0.012 macroP; geometric = arithmetic to three decimals | `pooling_rules.md` |
| 6 seeds instead of 3 | macroP −0.006, precET −0.024 | `balanced_bagging.md` |
| balanced bagging, 6 bags keeping all ET | +0.001 [−0.015, +0.019]; +0.007 vs the matched 6-seed control | `balanced_bagging.md` |
| one-vs-rest with a dedicated ET detector | **precET −0.162 [−0.249, −0.073] \***; worse as a pure ranker (AUC 0.750 vs 0.770). The softmax's shared normalisation is load-bearing | `one_vs_rest.md` |
| gate contested patients to a second model | +0.001 [−0.066, +0.063] precET vs uniform fusion — the gate is decorative | `contested_gating.md` |
| tune priors for macro precision (the target) | −0.049 macroP, sd doubles; F1's recall term regularises the search | `prior_objective.md` |
| per-cohort priors | significantly worse | `cohort_strategies.md` |
| temperature scaling before priors | null; `tune_offsets` already re-fits with more freedom | `pooling_rules.md` |

## Training set

| tried | result | report |
|---|---|---|
| drop the hardest N/PD patients | **worse than nothing and worse than random** — precET −0.081 \*, hard-vs-random −0.065 \*. The hardest are boundary-defining | `prune_training.md` |
| drop by Monte-Carlo influence (Data-Shapley) | no better than random, trending worse | `influence_prune.md` |
| drop by exact leave-one-out harm, 20 inner splits | same — LOO-vs-random precET −0.023 [−0.084, +0.032]; ranking overlap 0.15 vs 0.04 chance, against 0.60 when a harmful set is planted | `influence_stable.md` |
| cohort-ID input | precN +0.024 [+0.009, +0.041] \* over a *valid* per-split random control; precET +0.007; macroP +0.011 [−0.000, +0.021]. Does not reduce NewData's contested rate | `cohort_id_input.md` |
| mixup / SpecAugment / shift+noise | worse or null | `deep_model_improvement.md` |
| SMOTE | plain hurts; boundary variants help only where minority n suffices | `resampling.md` |
| PADS pretrain → in-house finetune | precET −0.188 \*, the worst thing tried | `cohort_strategies.md` |
| masked-spectrum SSL on 3,081 recordings | no transferable benefit; the reported gain was frozen-vs-finetuned confounding | `ssl_retraction.md` |

## Representation and preprocessing

| tried | result | report |
|---|---|---|
| low band edge 2.0 / 1.5 Hz | −0.008 / −0.004 macroP; slow patients unmoved; model ignores sub-3 Hz | `low_band_edge.md` |
| estimator sharpness sweep, ar16 → welch → nw 2.5 → nw 4 → nw 6 (Q ceiling 31 → 1.4) | inverted U peaking at the current nw 2.5; ar16 −0.031 \*, welch −0.024 \* | `estimator_smoothing.md` |
| peak-aligned averaging of a patient's recordings | +0.007 precET vs plain mean; vs random-shift +0.119 \* — mechanism real, data below the knee | `peak_aligned_average.md` |
| trim the PADS arm-raising onset | mechanism confirmed (N 1.39 → 1.10); PADS→in-house transfer unchanged, all arms below the 0.655 floor | `pads_onset_trim.md` |
| log-frequency binning | precET −0.086 \* against the real baseline (was measured against a weaker re-implementation) | `spectral_representation.md` |
| principal-eigenvalue / polarisation spectrum | null / worst; sum-normalisation discards amplitude gains | `spectral_representation.md` |
| short-window (0.64 s) spectrum in the 3-class model | macroP −0.033 \*; wins on the PD-vs-ET binary axis but does not compose | `tf_window_length.md` |
| spectral variability features (IQR, flux, wander) | weakest arms everywhere; dilute when appended | `tf_window_length.md` |
| TCN on the raw waveform / on analytic channels | −0.034 \* / −0.076 \* (precET −0.192 \*) | `time_domain_deep.md` |
| HHT / hht_imf2plus | worst of the estimators | `signal_processing_summary.md` |
| ImageNet backbones, frozen ViT | at chance | `frozen_vit.md`, `pretrained_backbones.md` |
| catch22 hybrid | AUC +0.014 \* but precET −0.028 \* — rank-averaging dilutes the top of the ranking | `catch22_waveform_features.md` |
| MiniRocket/ROCKET on the waveform | macroP −0.088 \* and −0.085 \*; **worse than the learned TCN on the same input** (0.555 vs 0.626); fusion null with validation choosing weight 0 in 10/20 splits | `rocket_waveform.md` |
| training-time logit adjustment (Menon et al.) | null at 40 splits — τ=0.5 macroP +0.012 [−0.003, +0.027], and the effect halved on doubling from 20 splits while the precET win rate fell to 0.42 | `logit_adjustment.md` |
| **PCEN** (Lostanlen, *IEEE SPL* 2019) on a dense-hop surface | **precET −0.233 [−0.351, −0.121] \*, macroP −0.101 \*** — the largest negative measured. Structural: dividing each band by a smoothed copy of itself is exactly what destroys *which band* has energy. No α fixes it | `pcen_hpss.md` |
| **HPSS**, harmonic component | +0.018 [−0.056, +0.093] precET — null for adoption, but the ordering held (see below) | `pcen_hpss.md` |
| dense 0.16 s hop (13 → 49 frames) | free: +0.004 macroP, null on every column. Resample to it if anything ever needs the time axis | `pcen_hpss.md` |

## Task and structure

| tried | result | report |
|---|---|---|
| average non-postural tasks into the spectrum | precN +0.047 \*, precET −0.104 \* | `task_averaging.md` |
| rest-vs-postural contrast features, any form | −0.061 to −0.106 precET | `rest_postural_contrast.md` |
| MIL attention / max over recordings | significantly worse than the uniform mean (−0.117, −0.147) | `mil_recordings.md` |
| two-stage hierarchy, same inputs | null; with axis-specific inputs macroP +0.011 n.s. and precET −0.075 | `axis_specific_inputs.md` |
| feature unions (16 instances) | underperform the best member; dimensionality binds at n = 404 | many |
| FiLM / channel gating / early fusion | null at 40 splits | `tcn_fusion.md`, `early_fusion_confirm.md` |
| STAB (TSI) appended or replacing descriptors | +0.030 / +0.060 precET, both n.s.; **halves precET variance** — the one null worth 40 splits | `temporal_stability.md` |

## The two unions that work

`axes + stability`, and `logreg + one-class Mahalanobis` rank-averaged — both
two-member, both members individually decent, combined at the score level.
Combine only when members are comparable and you cannot tell which will win.

## The time-domain rule, repaired

The old rule — "time-domain information is only reachable through estimators
that do not have to be learned from this cohort" — is **wrong as stated**.
MiniRocket is unlearned and lost to the learned TCN. The repaired rule is:

> **few features, selected for classification.** catch22 satisfies both (22
> statistics chosen on 93 datasets); MiniRocket satisfies neither. Supplying
> only the low-dimensional half via PCA to 22 dims is worth macroP +0.046 \* —
> significant, and still short of both the TCN and the reported model.

Dimensionality remains a first-order constraint, now measured at its most
extreme: 9 996 features on ~260 training patients.

## What HPSS established even though it was null

Adoption was null, but the attribution control was not. The recorded ordering
held exactly — harmonic 0.660 > dense control 0.639 > **percussive 0.523**, the
percussive arm significantly worse than the control it is matched against
(precET −0.117 \*, macroP −0.046 \*). So:

> **Class information sits in the sustained, tonal component; the broadband
> transient component carries significantly less of it.**

That is the first direct confirmation of the assumption the whole front-end
rests on. It also explains the null: `P.mean(0)` is already a weak
harmonic–percussive separator — a transient occupies a few frames out of 49 and
gets divided by the frame count, while a sustained oscillation contributes in
every frame. HPSS removes explicitly what averaging removes implicitly. Same
shape as `pads_onset_trim.md` from the other direction: a real, class-ordered
artifact that changed nothing once the pipeline averaged over time.

## Standing rule on sub-component gains

A gain measured on a component — a descriptor, a binary axis, a sharpness gap —
is **not** evidence about the 3-class model. Failed prediction #5, then the
short-window composition, then peak alignment (33 % sharpness recovery →
+0.007 precET). Treat such gains as motivation to test, never as a result.

**The rule is asymmetric.** Gains have failed to compose three times; a
descriptor-level *destruction* composed perfectly the first time it was tried.
PCEN's damage was visible in two label-free statistics of the 16-bin spectrum
before any model was fitted (`_pcen_alpha_diagnostic.py`: entropy 0.905 → 0.992,
peak/mean 3.08 → 1.14), and that diagnostic called both the direction and the
rough magnitude where the reasoned prediction called neither. **Run the cheap
label-free measurement of what a transform does to the representation before
spending the fits** — it cannot promise a gain, but it can rule one out.
