# Register of predictions made before the run, and how they turned out

This project writes the prediction into the experiment's docstring *before*
launching it, so that a null cannot be reinterpreted afterwards as the expected
outcome. This file is the register. It exists because the count was previously
asserted in the README as a bare number that nothing on disk could check.

The pattern it documents is the useful part: **mechanism stories about why a
method ought to work have a poor record here, and predictions derived from
measurements of this dataset have a better one.**

## Failed

| # | prediction | where | what happened |
|---|---|---|---|
| 1 | robust estimation should help | `tf_window_length.md` | failed |
| 2 | demodulation cost explains the gap | `tf_window_length.md` | failed |
| 3 | median-centring should help | `tf_window_length.md` | failed |
| 4 | N-vs-Tremor blurring explains the gap | `tf_window_length.md` | failed |
| 5 | measured sub-component gains should compose in two stages | `tf_window_length.md` | failed |
| 6 | the hardest majority patients are mislabelled noise; dropping them sharpens the boundary | `prune_training.md` | failed — and the verdict itself was later **downgraded**. It read *inverted* (precET −0.081 \*, hard-vs-random −0.065 \*), but on the corrected pipeline, with the experiment's own code bit-identical, every significant column reverses sign and becomes null. The honest verdict is **failed and now uninformative**: the data does not support "hard patients are boundary-defining" *or* its opposite. |
| 7 | subjects measurably harmful in training can be identified and removed | `influence_prune.md` | failed — no better than random, and trending worse, with an unstable ranking |
| 8 | geometric pooling should buy ET precision by vetoing over-confident members | `pooling_rules.md` | failed — reproduces arithmetic pooling to three decimals on macroP |
| 9 | the pooling null is explained by the six members being near-copies | `ensemble_diversity.md` | **refuted by measurement** — they disagree on 20.5 % of patient pairs |
| 10 | a dedicated ET detector beats ET's column in the 3-class softmax | `one_vs_rest.md` | **inverted** — precET −0.162 [−0.249, −0.073], and worse as a pure ranker (AUC 0.750 vs 0.770) |
| 11 | routing contested patients to a second model should help | `contested_gating.md` | failed — +0.001 against the fusion control that ignores the gate |
| 12 | slow patients are contested because their signal is entangled with drift at the 3 Hz band edge; extending the band down should help them specifically | `low_band_edge.md` | failed — slow-tercile contested rate rose slightly (+0.005 at 2 Hz, +0.019 at 1.5 Hz) and precision was null on every column |
| 13 | the frequency-contested gradient cannot be class confusion, because confusion produces opposing signs | `contested_profile.md` | **refuted by re-reading my own numbers** — that holds only when the class means straddle the range. They are monotonically ordered (8.16 / 7.51 / 7.04 Hz), so confusion gives the same sign for N and PD and none for ET, and the measured effect sizes (−0.385 / −0.241 / −0.051) fall monotonically with each class's own mean frequency. |
| 14 | cohort ID should cut NewData's contested rate more than 2015's, absorbing a domain shift | `cohort_id_input.md` | failed — contested rate moved −0.002 / −0.009 / −0.007 across cohorts, no differential effect; the precN gain it did produce is unexplained by this mechanism |
| 15 | model performance should rise monotonically as the spectral estimator gets smoother | `estimator_smoothing.md` | **still failed, but the recorded reason was wrong.** It read "an inverted U peaking at the current nw 2.5 (ar16 −0.031 \*, nw6 −0.016)". That sweep ran on a **duplicated copy of the frequency-axis bug**; corrected and at 40 splits there is no inverted U and no peak — only ar16 is distinguishable (macroP −0.018, win 0.38) and the top four arms span 0.007 against a resolution of 0.025. The prediction fails because the sweep is **flat**, not because it turns over. |
| 16 | trimming the PADS arm-raising onset should raise PADS-to-in-house PD-vs-ET transfer, because the onset is a PADS-only class-ordered signature the in-house cohorts lack | `pads_onset_trim.md` | failed — transfer AUC 0.578 → 0.563, −0.015 [−0.042, +0.009]; every arm below the 0.655 floor. The onset is real (mechanism held) but is second-order on a cohort gap far larger than it. |
| 17 | the old IF trajectory's significant gain lived in its transient end points reading the class-ordered PADS onset (point 0 sits on the onset) | `descriptor_trajectory_fix.md` | failed — the pre-fix point-0 magnitude is ~1.0 Hz for every class on PADS (N 1.01 / PD 0.89 / ET 1.01) and correlates with the onset ratio at +0.032. Class-agnostic noise. |
| 18 | training-time logit adjustment would lean NEGATIVE on precET, because prior_objective measured precET −0.236 when this project's imbalance correction was aimed at balanced accuracy and logit adjustment is Fisher-consistent for that objective | `logit_adjustment.md` | **wrong in direction, right in magnitude** — τ = 0.5 came out +0.034 precET / +0.012 macroP at 40 splits, positive but null. The flaw: post-hoc offset tuning and training-time adjustment share an objective but not a failure mode; the offset search moves a threshold on a fixed representation, adjustment shapes the representation. The sub-prediction τ = 0.5 > τ = 1.0 held. |
| 19 | MiniRocket, being unlearned like catch22, should beat the learned TCN on the same waveform | `rocket_waveform.md` | **refuted** — macroP 0.555/0.558 against the TCN's 0.626, and significantly worse than the reported model (−0.088 *, −0.085 *). It broke the standing rule it was built on: "unlearned" is the wrong abstraction; the repaired rule is "few features, selected for classification". |
| 20 | PCEN would be small and uncertain in sign, since the recordings are short, band-passed and sum-normalised so the stationary background it removes is largely gone | `pcen_hpss.md` | **wrong — large and significantly negative**: precET −0.233 [−0.351, −0.121] *, macroP −0.101 *, winning macroP in 3/20 splits. A pre-run label-free diagnostic had already measured PCEN flattening the spectrum (peak/mean 3.08 → 1.14) and correctly anticipated the failure. |
| 21 | cropped training would be NEGATIVE on precET, because window-to-patient aggregation was already measured to *help* and training on windows makes the network fit un-denoised rows | `window_training.md` | failed — +0.026 precET, positive and null on every column. The denoising argument does not transfer from evaluation to training: the window arm still aggregates at inference, so only its gradient signal is noisier. Its designed-to-be-informative sub-prediction (gain largest for the scarcest class if rows bind) also failed to pay out — ET does gain most, but the ordering is not monotone in class size and nothing is significant. |

Prediction 23 failed in the most useful way available: it named a statistic that
turned out to have **no control**, and the failure is what surfaced that. The
agreement columns were controlled against different-patient-same-class pairs and
carried the entire result; the confidence columns were reported bare and could
not have decided anything. **A statistic without a baseline is not evidence, and
writing the prediction down first is what made that visible** — had the
confidence numbers happened to separate widely, they would have been reported as
a finding.

Prediction M is the register's first *deliberately* mixed prediction. Both
accounts of the ceiling were live, the experiment was built to apportion rather
than to choose, and predicting "neither wins cleanly" in advance is what stops a
54/46 split being written up afterwards as whichever answer suits the spending
decision it feeds.

Prediction 22 is the first failure in this register that a **shuffled control**,
rather than an interval, decided. Every column of the fusion arm was null anyway,
but the finding is that six columns of permuted noise reproduced the real
feature exactly (+0.003 macroP against +0.003). Had the arm come out at +0.03
the control would still have said the same thing. **Report a shuffled control on
every appended feature block** — without one this would have been written up as
"+0.003 macroP, +0.016 precPD, promising".

It also shows the limit of the pre-run diagnostic that invariant 10 asks for. The
diagnostic was right — the orientation information is real, AUC 0.702 / 0.713
against permutation nulls — and the method still failed, because *existing* is
not *being new*. The missing measurement was complementarity against the features
the model already has, and on PD-vs-ET the union beat the descriptors by +0.002.
**Ask whether a feature is redundant, not only whether it is informative.**

Prediction 21 is the first time the **win rate**, not the interval, carried the
verdict. Every column came out positive and null, which on the means alone reads
as "promising, needs more splits". macroP was positive while losing 11 of 20
splits. Invariant 5 was added to the skill three experiments earlier, after
logit adjustment showed the same shape; this is it working.

Prediction 20 is the clearest case yet for running a cheap diagnostic before the
fits. The reasoned prediction ("PCEN small and uncertain") was wrong; a
label-free measurement of what PCEN does to the spectrum, taken before the run,
called both the direction and the rough magnitude. It also marks an
**asymmetry in standing rule #5**: descriptor-level *gains* have failed to
compose to the model three times, but a descriptor-level *destruction* composed
perfectly.

Prediction 5 was the most sobering of the early batch because it was the
disciplined kind — built from measured sub-component gains rather than a story —
and it still failed. **Sub-component gains on this dataset are not evidence about
the composite task.**

Prediction 13 is the second wrong *explanation* in this register, after #9, and
both were caught the same way: by checking whether a competing account predicted
the numbers **including their magnitudes**, not just their signs. #9 was caught
by measuring ensemble disagreement; #13 by noticing that the three effect sizes
were ordered exactly as the rival account required. A sign test is weak evidence
when a magnitude ordering is available.

Prediction 12 is worth separating from the rest. It failed, but the experiment
was built so that its failure *eliminated* one of two named accounts rather than
just returning nothing — the rival account (cycle count) is untouched by a band
change, so the null moved it from one-of-two to the live explanation. A
prediction designed so that either outcome is informative is worth more than one
that only pays out when it holds.

Prediction 18 adds a distinct failure mode to the two above: **sharing an
objective does not make two methods share a failure mode.** Post-hoc offset
tuning and training-time logit adjustment both target balanced error, and the
first had failed badly here, so the second was predicted to fail too. It did not
— it came out positive, just not significantly. The methods differ in what they
can overfit: the offset search sees ~11 validation ET patients, adjustment sees
none.

Predictions 6 and 10 were written up as sharing a shape — both identified
something that *looked* like a handicap for the minority class (hard patients
dragging the boundary; ET's logit diluted by two majority columns) and both
appeared to be **load-bearing**. **Half of that pairing is now withdrawn.**
Prediction 6's inversion does not survive the preprocessing fixes; only
prediction 10 (one-vs-rest, precET −0.162 \*) still stands as a measured case.
The general lesson — that a structure looking wasteful at 404 patients with 49 ET
is more often doing work than not — is now supported by **one** instance rather
than two, and should be held that much more loosely.

| 22 | the fusion gain would be larger on precPD **and** precET than on precN, because the model-free diagnostic separated PD from ET (AUC 0.702 / 0.713) far better than it separated N from tremor on PADS (0.579) | `riemann_axes.md` | failed on the half that mattered — precN −0.000, precPD +0.016, precET **−0.006**. The mechanism reasoning was sound and the pre-run measurement was real; what it could not anticipate is that the tangent vector is **redundant with the ten descriptors** on PD-vs-ET (PADS union 0.797 vs descriptors 0.795), so there was nothing new for the model to compose on that contrast. |

| 23 | the label-noise half of the ceiling would show up more strongly in the *confidence* statistic than in the agreement statistic, since being confidently wrong is easier to produce than two recordings independently landing on the same wrong class | `self_consistency_gate.md` | failed — agreement carried the whole result (A_wrong +0.179 above its control) while confidence separated by only ~0.05. It also exposed a design gap in my own experiment: the confidence columns were reported **without a matched control**, unlike the agreement columns, so they could not have been decisive in either direction. |

| 24 | the whole-recording arm would be worst of the three, because NewData's raw 38 s captures are only 9.9 % in-band and mostly set-up motion | `newdata_epoch_choice.md` | failed — it **ties** (macroP 0.648 vs 0.646) and is marginally best on precPD and macroF1. The reasoning ignored that the pipeline **band-limits to 3-15 Hz before the model sees anything**, so out-of-band set-up motion was never a threat. A 6x range in in-band purity (0.118 / 0.571 / 0.697) moves macroP by 0.002. |

| 25 | the conv+time-transformer would tie the time-averaged CNN, and the frame-shuffle control would tie it too | `time_axis_transformer.md` | **half right, and the wrong half was the interesting one.** B tied A to three decimals (macroP 0.608 both) — temporal order carries nothing, as predicted. But C, with order destroyed, **beat** both: macroP +0.027 [+0.005, +0.052] \*, precET +0.052 [+0.013, +0.093] \*. The shuffle was designed as a sanity check and became the result. |

| 26 | precN would reach 0.90 at high coverage under abstention, since N-vs-Tremor already exceeds 0.90 at full coverage (0.910 / 0.924) | `selective_90.md` | failed — precN tops out at **0.838 at 20 % coverage**, i.e. 16 patients. The error was extrapolating a **binary screen** result to **3-class precN**: in three classes, precN is diluted by PD and ET patients misassigned to N, which the binary screen never sees. Different quantities with the same name. |

| 27 | adding a structurally different architecture as an ensemble member would be **null**, and the seed control would match it — because `pooling_rules` found combination does not matter within ±0.012 and the members already disagree on 20.5 % of patients | `diverse_ensemble.md` | **failed — the first gain in this session to survive 40 splits.** `+ transformer` reads precET +0.040 [+0.004, +0.074] \*, macroP +0.016 \*, macroF1 +0.016 \* against the reported model. The control *did* capture about half (precET 0.662 on its own), so the reasoning was right that member count matters — but wrong that diversity adds nothing: macroF1 +0.017 \* survives against the control. |

| 28 | the dominant confusion among consistently-wrong patients would be **PD↔ET**, not either-vs-N, since N-vs-Tremor already reaches 0.91–0.92 | `readjudication_list.md` | **failed — N↔PD is 43 of 62.** The reasoning confused a screen's overall precision with the make-up of its residual errors. And a model-free amplitude test then showed those 43 are *phenotype*, not mislabels: PD→N patients have Normal-level tremor (and are tremor-free at rest too), N→PD controls have PD-level tremor. |

| 29 | in-house REST, given to the deep model as its own descriptor block, would lift in-house PD-vs-ET AUC by +0.03 to +0.08 over **both** the adopted model and a shuffled-REST control, because 2015 REST separates PD from ET at AUC ~0.65-0.70 on six features | `inhouse_rest.md` | failed — +0.022 \* over the adopted model, but +0.012 (null) over the control, which itself moved inAUC +0.010 with no patient-label link; in-house ET recall fell −0.062 \*. The side predictions held weakly: merged metrics stayed flat except recET −0.040 \*, and adding PADS REST was lower on inAUC (−0.013) but not significantly. Same failure as #5 and #22: a feature-level signal did not compose into the model. |
| 30 | pooling NewData REST with 2015 REST (z-scored within cohort) would stay above its shuffled-label null, only weaker than 2015 alone | `rest_replication.md` | failed — 0.573 against a null of [0.334, 0.644], p = 0.162. NewData's 6 ET diluted the 2015 signal rather than adding to it; together with 2015's own p = 0.038 on a sensor chosen among three, the in-house REST signal is not established. |
| 31 | in-house REST would raise ET precision in the deep model (REST and OUT+REST precET > OUT by > +0.05), on the hand sensor where the linear pre-check found PD-vs-ET AUC 0.661 / 0.712 above the null, and not on the lower arm | `inhouse_rest_deep.md` | failed — precET REST − OUT −0.040 (hand), −0.057 (lower); OUT+REST − OUT −0.007 / −0.001. The only baseline-significant gain was **precPD** +0.059 \* on the hand sensor. The pre-check was a measurement, but of a binary *ranking* at 21 ET; the prediction extrapolated it to 3-class *precision* at 10 test ET, where one prediction is worth ~0.07 -- the same screen-to-3-class slip as #26. |
| 32 | 2015 OUT transfer: precET null for every transfer arm (\|mean\| < 0.05, CI spanning 0) because PD-vs-ET transferred only weakly (0.589); precN up for `ft` **and `ft_NT`** because N-vs-tremor transferred fully | `transfer_2015.md` | **failed, in the useful direction.** `ft` precET **+0.084 [+0.035, +0.135] \*** at 40 repeats (win 0.72), +0.122 \* over the permuted-label control. `ft_NT` did not raise precN (−0.010) and collapsed ET (−0.173 \* at 20). A weak linear PD-vs-ET transfer AUC understated what a pretrained *representation* carries once the decision rule is re-fitted on 2015. |
| 33 | `ens` (soft vote of the transfer model and the 2015-only model) would tie its ensemble-size control `ft_x2` on macroP, trading PD for ET; and fine-tuning longer (`ft_long`) would raise precPD as it lowered precET | `transfer_2015.md` | **failed on both PD halves.** `ens` − `ft_x2`: macroP −0.020 \*, precET −0.063 \*, precPD +0.006 (n.s.) -- mixing in the scratch model costs ET without buying PD. `ft_long` lowered precET (−0.021 \*) as predicted but left precPD unchanged (+0.002): past 80 epochs the fine-tune erodes PADS's ET knowledge without recovering 2015's PD rule. |
| 34 | L2-SP fine-tuning would not beat `ft` at either strength; strong L2-SP (λ 1) would drift toward head-only -- precPD below `ft`, precET not above -- because head-only was the λ → ∞ limit and was worse; and pretraining 400 epochs would stay within ±0.03 precET of `ft` | `transfer_2015.md` | **failed on the main claim.** L2-SP λ 1 improved ET-vs-PD ranking (AUC +0.018 \*, win 0.80; replicated +0.017 \* in the sweep) with precET +0.031 (n.s.) -- the PD half held (−0.014 \*). Head-only is not the limit of the anchor: it also freezes BatchNorm and the whole backbone, while L2-SP lets every weight move a little. Pretrain 400: precET −0.038 (n.s.), just outside the band. |
| 35 | the L2-SP ranking gain would peak at λ 3 or 10, at least +0.005 AUC above λ 1 | `transfer_2015.md` | failed -- the peak is flat between λ 1 (0.645) and λ 3 (0.647); λ 10 already falls (0.640). The λ 30 half held (0.633, precPD −0.051 \*). |
| 36 | (2015 REST pre-check) PADS -> 2015 N-vs-tremor transfer AUC > 0.70, as at OUT; and 2015's own PD-vs-ET REST CV AUC above its null | `rest_2015.md` | failed on both -- N-vs-tremor transferred at 0.637 (above its null, well below OUT's 0.850), and 2015's own PD-vs-ET REST AUC read 0.605-0.620, p 0.10-0.12, inside the null across four feature sets. The earlier 0.650 / p 0.025 (`inhouse_pd_vs_et.md`) does not survive a change of regularisation and CV seeds. |
| 37 | multi-segment coupling in the 2015 OUT model would beat its shuffled-row control on PD-vs-ET AUC by +0.02 to +0.05 (significant) but not on ET precision, since feature-level signals have moved ranking before thresholded precision here | `multisegment_2015.md` | **inverted** — AUC +0.007 (n.s.) over the control, while macroP +0.018 \*, macroF1 +0.014 \*, precPD +0.012 \* and precET +0.035 (borderline). The control itself lifted AUC +0.012 \*: extra columns move ranking, the real content moved decisions. |
| 38 | the removed HHT would show the bluntest peak on noisy and jittered tremor, from its one-sample instantaneous-frequency estimate, and smoothing would recover it | `hht_audit.md` | **failed — the defect was the opposite.** The old estimator peaked *sharply* at exactly 15.00 Hz on every noisy signal and on pure noise: its bin edges were -inf / +inf, so out-of-band noise was binned into the edge bin. Smoothing the instantaneous frequency was worth only +0.4 sharpness. Fixed, HHT still trails multitaper on 2015 OUT N vs tremor (0.868 vs 0.891). |
| 39 | a second stage adding hand-forearm coherence to the 2015 `ft` probabilities would beat both the plain second stage and a shuffled control on precET by +0.03-0.06, and all 8 coupling features would do no better than coherence alone | `stack_2015.md` | **half failed** — vs shuffled +0.049 \* (held) but vs the plain second stage −0.003; and coupling8 beat coh_hl (precN +0.035 \*, macroF1 +0.020 \*). The diagnostic picked the strongest single feature, not the only informative one. |
| 40 | a nested high-confidence rule asking for 80 % precision would deliver >= 0.80 held-out precision for N and PD on 2015 OUT | `high_precision_2015.md` | failed narrowly — 0.789 / 0.789: a threshold chosen on training folds gives back ~0.01-0.03 on held-out ones. Asking for 0.90 delivers 0.849 / 0.873. |
| 41 | (post hoc) giving the PD-vs-ET second stage tremor amplitude and its products with the coupling features would fix its weak top-1 pick, since coherence is meaningless without tremor | `pdet_2015.md` | failed on both partition sets — fold AUC −0.012 \* / −0.013 \*, top-1 −0.070 \* / −0.050 |
| 42 | re-assigning the recommended model's ET labels (same count per fold) by the coupling second stage's PD-vs-ET ranking would raise precET +0.03-0.08 on both partition sets | `pdet_2015.md` (`rerank_2015.py`) | failed — −0.033 / −0.031 \*; +0.084 \* / +0.101 \* over a shuffled-coupling rerank. The AUC gain sits in the middle of the ranking, not in the ~3 top calls per fold |

## Held

| # | prediction | where | what happened |
|---|---|---|---|
| A | thirteen feature families would not beat the best single one | `tcn_fusion.md` | held |
| B | balanced bagging would be null, because the members already disagree on 20.5 % of patients so data diversity is not the binding constraint | `balanced_bagging.md` | held — macroP +0.001 [−0.015, +0.019], and +0.007 against the matched seed control |
| C | fixing the multitaper frequency axis would change little or nothing, because the distortion was uniform across every recording and the model was fitted and evaluated on it consistently | `axis_fix_audit.md` | held — macroP −0.008 [−0.029, +0.010], nothing significant on any column |
| D | peak-aligned averaging should beat the random-shift control on precET if the misalignment mechanism is real (deliberately NOT a prediction of improvement over the plain mean) | `peak_aligned_average.md` | held — +0.119 [+0.018, +0.226] * — while adoption was null (+0.007 precET vs the plain mean). A narrow prediction that held and still produced no gain. |
| E | trimming the PADS onset collapses the N > PD > ET first-1.5 s excess and the length-matched trim-end control does not | `pads_onset_trim.md` | held — 1.39/1.33/1.06 → 1.10/1.04/0.96; control 1.40/1.33/1.05 |
| F | the mixed-cohort headline effect of trimming the onset is small with uncertain sign | `pads_onset_trim.md` | held — macroP −0.006 [−0.031, +0.022], null on every column |
| G | the 0.25 s guard on the IF trajectory would be null, because the corrupted points are the same two positions for every patient and their magnitude does not depend on class | `descriptor_trajectory_fix.md` | held — macroP −0.004 [−0.027, +0.021], precET −0.004 [−0.068, +0.071] |
| H | the contiguous Q-factor fix would be small with uncertain sign, because the mislabelled feature nevertheless carried class-correlated information | `descriptor_trajectory_fix.md` | held — macroP +0.012 [−0.003, +0.034], precET +0.036 [−0.012, +0.097]; trended positive rather than the leaned-toward negative, which was explicitly not the claim |
| I | MiniRocket's failure is dimensionality, so reducing 9 996 features to ~22 should help substantially, with the optimum nearer 22–64 than 9 996 | `rocket_waveform.md` | held — PCA 22 gives macroP +0.046 [+0.013, +0.076] * and precET +0.086 [+0.006, +0.155] *, optimum exactly at 22. It does not rescue the method (0.605 vs the reported 0.643), so dimensionality is necessary but not sufficient. |
| K | patient-level Euclidean Alignment collapses PD-vs-ET AUC to chance while cohort-level alignment does not, because unlike the BCI setting EA was designed for, each patient here carries exactly one label | `euclidean_alignment.md` | held on both halves — patient-level 0.702 → 0.558 (inside its null, p = 0.300) on 2015 and 0.713 → 0.574 (p = 0.050) on PADS; cohort-level holds 0.679 and 0.708, both p < 0.001. Closed a method before any fits were spent. |
| L | the tangent-space arm loses to the reported model, because six band-limited numbers cannot match a 16-bin spectrum | `riemann_axes.md` | held — macroP −0.009, precET −0.032. Weakly, though: the arm turned out to *substitute* for the descriptors rather than stand alone, and swapping ten descriptors for six covariance numbers costs only −0.009. |
| M | neither account of the ceiling would win cleanly — A_wrong clearly above the same-class control but clearly below A_correct | `self_consistency_gate.md` | held, and closely — +0.179 above the control, +0.149 below A_correct, the two gaps nearly equal. Recorded in advance precisely so a mixed result could not be narrated afterwards as whichever answer suited the spending decision. |
| N | PADS L/R self-agreement < 2015/NewData same-arm self-agreement, because the PADS pair spans two limbs and tremor is genuinely asymmetric | `self_consistency_gate.md` | held — 0.773 vs 0.882 on correctly classified patients, 0.643 vs 0.733 on misclassified ones |
| O | the epoch rule would be null on the merged model, since NewData is 56 of 404 patients and 6 of 49 ET | `newdata_epoch_choice.md` | held — every performance column null. The one significant cell is `steady`'s nETpred −2.10 [−4.05, −0.55] \*, which is a threshold shift (recET −0.050, precET +0.036), not a gain. |
| P | small attention on the current input would be null or slightly negative, but **not catastrophic** the way the 85 M backbones were, because 17 k parameters sits in the band this cohort peaks in | `attention_fair_test.md` | held on both halves — SpectrumTransformer macroP −0.011 [−0.046, +0.021], CrossStreamAttention −0.013, both null; and 17.3 k reaches 0.635 where 85.8 M reached chance (AUC 0.540). Attention is not the problem; scale was. |
| Q | `pos_enc=False` (arm D) would match the frame-shuffle arm to within noise, since the shuffle's only effect should be to disable position | `time_axis_transformer.md` | **satisfied in the letter, empty in substance.** D vs C is null on every column (macroP −0.014 [−0.042, +0.015]) — but the two are 0.014 apart against a 0.040 resolution, so "within noise" holds only because the noise exceeds the difference under test. The experiment cannot separate "removing order" from "shuffle as augmentation". Same shape as the easy-drop prediction: a prediction can hold and tell you nothing. |
| R | ET's precision-vs-coverage curve would be **non-monotone**, because ranking by confidence removes ET patients faster than the PD patients they are confused with | `selective_90.md` | held, strikingly — precET rises to 0.753 at 0.6 coverage then **collapses to 0.448 at 0.2**, a 0.305 fall. The surviving errors are high-confidence, which is the same population `self_consistency_gate.md` found to be consistently wrong. Two instruments, one conclusion: **abstention cannot fix label noise because the model is confident on mislabelled patients.** |
| S | the transformer's halved sd(precET) is **conservatism**, not stabilisation — it should show up as a lower and steadier `nETpred` | `diverse_ensemble.md` | **half held, and the control separated the halves.** Against the reported model the arm does predict ET less often (nETpred −2.000 \* at 20 splits); against the matched seed control that shift is **null** (−0.400), so the conservatism comes from adding members at all, not from the transformer. The companion quantisation-floor reading held: sd(precET) tracks member count (0.189 → 0.188 → 0.182 → 0.162), not architecture. |
| T | the re-adjudication list would flag 40–70 patients with ET over-represented relative to its 12 % share | `readjudication_list.md` | held — 62 flagged, ET 29 % of them |
| U | a fourth ensemble family would not beat the family-count control — diversity saturates — with logistic regression the better of the two candidates | `diverse_ensemble_2.md` | held — both null or worse (CrossStreamAttention macroF1 −0.018 \* vs control), control itself null vs the adopted model; logistic regression was the better of the two, though not positive on macroF1 as hoped |
| J | HPSS should order harmonic > dense-hop control > percussive on precET, because tremor is the sustained component and movement artifacts are transients | `pcen_hpss.md` | held exactly — 0.660 / 0.639 / 0.523, and the percussive arm is significantly worse than its matched control (precET −0.117 *, macroP −0.046 *). Adoption is null (+0.021 precET n.s.), so the physics is confirmed while the separation is unnecessary. |
| V | a 2015-REST PD-vs-ET model, frozen, would transfer to NewData above 0.5 but not significantly (6 ET), and would be significantly **reversed** on PADS Relaxed | `rest_replication.md` | held on both halves — NewData 0.533 (p = 0.41); PADS 0.280 (p < 0.001), and a one-feature max_freq rule 0.300. Not blind: univariate medians had been printed earlier; the multivariate transfer had not. |
| W | on the 2015 transfer model: head-only fine-tuning would behave like the gentle fine-tune (PD below `ft`); pretraining on all 383 PADS would stay within ±0.03 precET of `ft` with precPD no higher; adding NewData OUT to pretraining would be null | `transfer_2015.md` | held on all three -- `ft_head` precPD −0.052 \* / precET −0.094 \* (PD 0.679, ET 0.256 vs gentle's 0.671 / 0.286); `ft_uncap` precET −0.029, precPD −0.008 (both n.s.); `ft_pn` null on precPD / precET / macroP (only precN −0.015 \*). |
| X | L2-SP λ 0.1 would match `ft` on every column (\|d\| < 0.02) -- the anchor too weak to bind in 80 epochs; per-recording PADS pretraining within ±0.03 precET of `ft` | `transfer_2015.md` | held -- λ 0.1 all columns within 0.004 (only AUC +0.003 \*); per-recording +0.012 precET (n.s.). |
| Y | 2015 REST deep model: PADS-Relaxed transfer would NOT beat scratch on precET or PD-vs-ET AUC (PADS teaches the opposite rest-frequency rule), would beat its permuted-label control on precN (tremor presence transfers), and scratch precET would be above chance but modest | `rest_2015.md` | held -- ft precET −0.010, AUC −0.105 \* (below chance, 0.484); ft − ft_shuf precN +0.041 \*; scratch precET 0.124 at prevalence 0.106 (1.2x, barely above chance). |
| Z | on 40 fresh partitions, L2-SP λ 1 would keep a significant PD-vs-ET AUC gain over `ft`, while its ~+0.03 precET gain from the selection partitions might not reach significance | `transfer_2015.md` | held -- AUC +0.012 \* (win 0.72); precET −0.007 (n.s.), so the selection-partition gain was selection bias plus noise. |
| AA | 2015 OUT multi-segment coupling separates N from tremor above its null (tremor propagates coherently through the arm) | `multisegment_2015.md` | held — CV AUC 0.786, null [0.379, 0.614], p = 0.005; REST 0.693, p = 0.005 |
| AB | adding REST tremor amplitude to the 2015 `ft` probabilities raises precPD and precN (the N->PD controls have tremor at OUT only) and leaves precET flat | `stack_2015.md` | held — precN +0.021 \*, precPD +0.019 \*, precET +0.009 |
| AC | OUT + REST late fusion on 2015 beats the 6-seed size control on precPD (+0.02 to +0.05) and precN with precET flat or lower, and REST weight 0.25 keeps more ET precision than 0.5 | `fusion_2015.md` | held — precPD +0.036 \*, precN +0.023 \*, precET −0.066 \*; weight 0.25 vs 0.5 precET +0.038 \*. Derived from the second-stage measurement in `stack_2015.md` |
| AD | ET does not reach 80 % precision on 2015 OUT under a nested threshold, though the non-nested (oracle) line will claim it | `high_precision_2015.md` | held — nested 0.10-0.33 on 0.2-0.5 flags per repeat (no flags in 31-36 of 40); oracle 0.90-1.00 on 0.1-0.4 flags |
| AE | a PD-vs-ET second stage adding the 8 coupling features to the OUT + REST fusion logit beats fusion and a shuffled-coupling control by +0.02 to +0.05 AUC; averaging fusion with ft_seg gains < +0.015 | `pdet_2015.md` | held and **replicated on fresh partitions** — within-fold AUC 0.712 / 0.710 (+0.051 \* / +0.033 \* over fusion, +0.148 \* / +0.136 \* over the control); averaging −0.006 |
| AF | a 2015 WING model as a third fused task beats a size-matched third OUT model on precPD (+0.01-0.03) and precN, with ET flat | `fusion3_2015.md` | held weakly — precN +0.015 \*, precPD +0.012 (CI touching 0), precET +0.009; a 15-repeat interim had read precET +0.056 \* |

Prediction K is the cheapest thing in this register. It closed a published
method **without fitting a single model**, by naming in advance the one
structural assumption the method makes — that the unit it normalises over
contains every class — and then measuring whether this data satisfies it. It
does not: a patient here carries one label. The same shape had already sunk PCEN
(a band divided by its own time-average) and per-cohort priors, which makes it a
family rather than a coincidence.

Prediction B is the contrast that makes the register worth keeping. It was
recorded in writing in two reports before the run finished, and unlike the failed
ones it was derived from a **measurement of this dataset** (the ensemble's actual
internal disagreement) rather than from an argument about why a method ought to
work.

## How to use this file

Append a row when an experiment's docstring commits to a direction before the
run. Do not edit a row after the fact. If a prediction is later shown to have
been right for the wrong reason, add a row rather than amending one — that is
what happened to #9, which was a wrong explanation of a correct null.
