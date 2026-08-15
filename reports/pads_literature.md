# What others do with PADS, and what we are missing

Literature check prompted by "discover how other people do with PADS to increase
the performance". Full-text fetches were proxy-blocked here, so the details
below come from abstracts and search summaries; the numbers are quoted, the
mechanisms are as described by the authors.

## The published results

| work | PD vs HC | PD vs DD | modalities |
|---|---|---|---|
| [Varghese et al. 2024, npj Parkinson's Disease](https://www.nature.com/articles/s41531-023-00625-7) (the dataset paper) | 91.16 % bal-acc | 72.42 % bal-acc | IMU **+ questionnaire + medical history** |
| [Self-Supervised Dual-Channel Cross-Attention, arXiv 2604.18372](https://arxiv.org/abs/2604.18372) (Apr 2026) | 93.12 % acc | **87.04 %** acc | bilateral IMU + SSL pretraining |

Dataset: 469 participants (291 PD, 79 HC, 99 DD), bilateral wrist IMU at 100 Hz,
10 neurologist-designed motor tasks, 5159 measurement steps. PADS ships
**recommended 5-fold train/test splits** for comparability.

## 1. The biggest gap: we are not using the questionnaire

The dataset paper's pipeline is **multimodal**. Alongside the smartwatch signal
it uses a **30-item yes/no non-motor symptom questionnaire** (MDS PDNMS) plus
medical history. Our pipeline is IMU-only.

Non-motor symptoms -- hyposmia, REM sleep behaviour disorder, constipation,
urinary dysfunction -- are strongly PD-specific and largely absent in essential
tremor. They separate PD from ET on clinical grounds that **no spectral analysis
of wrist motion can recover**. This is very likely worth more than every
architecture change measured in `deep_model_improvement.md` combined, and it
needs no modelling work.

**Action:** the questionnaire and patient metadata are in the original PADS
download. Only the movement files were extracted here (`pads_stretchhold/`,
`pads_relaxed/`). Extract the questionnaire and fuse it at the patient level.

Caveat to check when doing so: our 2015 and NewData cohorts have **no**
questionnaire, so a questionnaire-fused model cannot be trained on the merged
cohort. It would be a PADS-only model, or would need a missing-modality design.

## 2. Bilateral asymmetry is independently validated

The cross-attention paper's stated motivation for coupling the two wrist streams
is **"the motor asymmetry of PD"** -- PD begins and stays unilateral, ET is more
symmetric.

This is the same mechanism found independently here in
`limb_asymmetry_pd_vs_et.md`: four unsigned between-limb dissimilarity features
reach AUC 0.730 on PADS PD-vs-ET where the single-limb spectrum sits at 0.527,
with a paired bootstrap dAUC of +0.183 [+0.031, +0.343] and a clean permutation
null.

Good position for a write-up: same mechanism as the state of the art, reached
from the data, with a 4-feature logistic regression instead of a transformer.

## 3. Self-supervised pretraining -- a correction

`deep_model_improvement.md` and the session record contain a fairly confident
argument that contrastive / InfoNCE objectives are contraindicated here, because
every natural positive-pair choice destroys a measured biomarker (amplitude
scaling removes tremor amplitude, time warping removes tremor frequency,
left-vs-right positives remove the asymmetry finding).

A published paper reports SSL pretraining working on this exact dataset. The
objective is not visible in the abstract, and the full text could not be fetched
here:

* if it is **masked reconstruction**, the objection does not apply at all -- no
  invariance has to be chosen, so the argument was aimed at the wrong target;
* if it is **contrastive**, they found the opposite of what was argued.

Either way the earlier claim was overstated. **Masked-spectrum pretraining**
(predict held-out frequency bins) is the version worth trying: it uses the
unlabelled recordings without committing to any invariance.

## 4. Two caveats before comparing our numbers to theirs

**PD vs DD is not PD vs ET.** DD in PADS is essential tremor *plus* atypical
parkinsonism, secondary parkinsonism and multiple sclerosis -- 99 patients. That
is a broader and plausibly easier boundary than the 28-ET axis worked on here.
Our own `pads_label_bug.md` found 20 records labelled parkinsonian that are
Atypical Parkinsonism, which is exactly the population DD absorbs.

**Use the official splits.** PADS provides recommended 5-fold splits; a PADS
number computed on custom splits is not comparable to either published result.

## Ranked recommendations

1. **Extract and fuse the PADS questionnaire.** Largest expected gain, no
   modelling risk. PADS-only model, or missing-modality handling for the merge.
2. **Report on the official PADS 5-fold splits** whenever quoting a PADS number.
3. **Try masked-spectrum self-supervised pretraining** -- the SSL variant that
   survives the objection raised earlier.
4. **Keep the bilateral asymmetry features**; they are the cheap version of the
   state-of-the-art's central mechanism.
5. Do **not** frame PD-vs-DD results as comparable to PD-vs-ET.
