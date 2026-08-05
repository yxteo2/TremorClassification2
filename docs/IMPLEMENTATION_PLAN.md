# Implementation plan

Written after the quaternion line of work closed out with no significant result,
and after an empirical power curve that contradicts the advice I had been giving.

## The finding that reframes everything

Using PADS (28 clean ET, the largest clean ET cohort available) as a donor, I
subsampled ET and measured PD-vs-ET balanced accuracy as the cohort grows:

| n_ET | n_PD | mean bal-acc | sd | 95% spread |
|---|---|---|---|---|
| 8 | 60 | 0.636 | 0.061 | [0.544, 0.731] |
| 12 | 60 | 0.665 | 0.059 | [0.557, 0.743] |
| 15 | 60 | 0.678 | 0.060 | [0.543, 0.737] |
| 20 | 60 | 0.687 | 0.078 | [0.552, 0.807] |
| 28 | 60 | 0.677 | 0.053 | [0.609, 0.757] |

**The curve plateaus at ~0.68 by n=15 and does not rise after.** n=20 (0.687)
and n=28 (0.677) are indistinguishable.

I told you repeatedly that "more ET data is the only lever." **That is wrong for
the point estimate.** More subjects buys a *tighter CI* (sd 0.061 → 0.053,
spread narrows), not a *better model*. The ~0.68 ceiling with conventional
spectral features is an **information limit, not a sample-size limit**.

Caveats, stated so this isn't over-read: one cohort, one feature set
(STFT+biomarker), 12 repetitions per point, PD subsampled to 60. It is
suggestive, not definitive — but it is the only direct evidence available on the
question and it points the opposite way to the assumption the project has been
running on.

Corroborating: on that same PADS cohort, **feature choice moves the number more
than n does** — orbit geometry reaches bal-acc 0.747 where spectral features
reach 0.615–0.677. Different information helps; more of the same does not.

---

## Phase 0 — Lock the methodology (1 day, do first)

Non-negotiable after this session, where three claims were asserted before the
test that refuted them.

- **0.1** `pdetn/paired.py`: one function `paired_compare(predsA, predsB, y,
  subjects)` returning the paired subject-level bootstrap CI of the difference.
  Every "B beats A" claim goes through it.
- **0.2** Any univariate screen returns **BH-corrected q-values** alongside raw
  p, and reports the test count. No screen output is quotable without it.
- **0.3** Any screened effect is checked on **≥2 conditions** before it is
  written down. WING/REST cost nothing and would have killed the handedness
  claim on day one.
- **0.4** Write `docs/ANALYSIS_PLAN.md` fixing, in advance: primary metric
  (PD-vs-ET balanced accuracy + N-vs-tremor accuracy), CV scheme, and the exact
  model configs to be compared. Anything added later is exploratory and labelled
  as such.

## Phase 1 — Establish the honest ceiling (1 week)

Goal: know what the IMU data can and cannot support, before spending on
collection or writing.

- **1.1 Extend the power curve** to your local cohort and to orbit-geometry
  features. Does geometry plateau at the same n, or later? If geometry keeps
  improving past n=28, more data *is* worth collecting for it specifically.
- **1.2 Information-source ablation.** Which of these moves PD-vs-ET, measured
  with paired CIs: (a) 3 sensors vs 1; (b) multi-condition (OUT+WING+REST
  concatenated) vs OUT alone; (c) recording length; (d) adding clinical
  covariates (age, sex, disease duration) if available.
  **This is the highest-value experiment in the plan** — it tells you which axis
  actually carries unexploited information.
- **1.3 Ceiling estimate.** Fit the best model you have on PADS (n=276/28) and
  report its bal-acc. That is roughly what this modality supports with a large
  clean cohort. If it is ~0.75, then 90% is not achievable from wrist/arm IMU
  alone and the target needs revising.

## Phase 2 — Data collection, scoped by Phase 1 (ongoing)

Only collect what Phase 1 says will help.

- **2.1** NewData currently has **ET only (6 subjects)**. Collect **N and PD on
  the same device** — this matters more than more ET, because it turns NewData
  from an ET-only annex (poolable on one feature block) into a self-contained
  cohort you can train and test within, with no domain shift at all.
- **2.2** Record **limb side** and per-subject clinical metadata from now on.
  Cheap, and it closes a confound permanently.
- **2.3** More ET only if 1.1 shows the curve still rising for your best feature
  family. Otherwise ET collection buys CI width, which is worth something for the
  paper but will not hit 90%.

## Phase 3 — Paper (start now, in parallel)

Do not wait for a number that Phase 1 may show is unreachable.

- **3.1 Reframe the contribution.** What you have that is genuinely publishable:
  - a **methodological result**: the device-identity probe as a gating criterion
    for cross-dataset pooling, with the concrete finding that only
    mount-invariant features are device-agnostic (probe AUC 0.567 vs 1.000);
  - a **negative result with unusual rigor**: PD-vs-ET does not separate from
    arm IMU across 9+ feature families, two cohorts, with paired CIs and
    multiple-comparison correction — including a power curve showing it is not a
    sample-size artifact;
  - a **solid positive**: N-vs-tremor at ~0.88–0.91;
  - a **data-quality contribution**: the PADS label contamination (32 % of ET),
    which affects anyone else using that dataset.
- **3.2** Report PD-vs-ET as **balanced accuracy**, never raw accuracy (majority
  baseline is 0.833 locally, 0.908 on PADS).
- **3.3** Include the retraction trail. A paper that shows a promising effect and
  then correctly kills it is more credible, not less.

## Phase 4 — Deep learning / XAI (gated, and the gate should move)

- **4.1** Your stated gate was 90 % on both axes. Phase 1.3 will likely show that
  is unreachable for PD-vs-ET from this modality. **Recommend re-gating** to:
  N-vs-tremor ≥0.90 (already close) and PD-vs-ET significantly above the
  majority baseline with a paired CI excluding zero.
- **4.2** If deep learning is pursued, use a **CNN/ResNet on spectrograms**, not
  the BiLSTM — GradCAM needs conv feature maps. `pdetn/deep_eval.py` already
  does patient-grouped CV with fine-tuning support.
- **4.3** XAI on a model that is not significantly better than chance on its hard
  axis will produce confident-looking attributions for noise. Do not do it until
  4.1 passes.

---

## Order of work

1. Phase 0 (1 day) — cheap, prevents repeat of this session's failures.
2. Phase 1.2 ablation (highest information value) and 1.3 ceiling.
3. Phase 3.1 paper reframe, in parallel.
4. Phase 2 collection, scoped by what 1.1/1.2 find.
5. Phase 4 only after the re-gate in 4.1.

## What I would drop

- Further TF/feature engineering on the existing data. Ten families have now
  landed within CI. An eleventh will too.
- Limb-side labels as a *priority* — they were needed for the handedness claim,
  which is retracted. Still worth recording going forward (2.2), not worth a
  retrospective chase.
- PADS as training data. Confirmed unusable twice, before and after the label
  fix. Keep it as an independent test set and as the donor for power curves.
