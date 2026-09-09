# OUT-only contrastive transfer with a matched control

## Question and fixed protocol

Does source-only contrastive learning improve a frozen encoder over the exact
same random initialization? This corrects a limitation of the preceding masked
reconstruction experiment: the frozen random control was missing.

OUT only, all three sensors. Target = 151 2015 patients (61 N,75 PD,15 ET).
Source = 56 NewData patients (27 N,23 PD,6 ET). No REST, WING or PADS. Use
the same five outer folds, inner validation reservation and labelled training
patients as PR #7. No data from target validation or test patients enter source
pretraining; the reserved validation patients are unused by the fixed heads.

Three fixed seeds (0,1,2), averaging probabilities, never selecting a best seed.
For each seed retain a copy of the randomly initialized encoder before source
pretraining. Both encoders share the same architecture, source-only RMS scaling,
32-d mean/log-variance pooling and balanced logistic C=.1 head with fold-local
StandardScaler. The head trains on target outer-train plus all source patients.

Pretraining: 40 epochs x five batches, at most 32 distinct source patients per
batch, one sampled window per patient. Two independently 20%-masked views of
the same window; normalized contrastive logits at temperature .2. Pair matching
across patients and time positions, averaged across max-pooled temporal scales.
No diagnosis labels, time-warping or amplitude augmentation in pretraining.

Motivation: [TS2Vec, AAAI 2022](https://ojs.aaai.org/index.php/AAAI/article/view/20881)
and its [author implementation](https://github.com/zhihanyue/ts2vec/blob/main/models/losses.py).
This is an independently implemented small adaptation: different encoder,
temperature-normalized similarity, masking-only contexts, pooling and schedule.
It is not an exact TS2Vec reproduction or transfer from a large external checkpoint.
Periodic signals can also make temporal negatives imperfect; benefit is empirical.

Shared waveform processing from PR #7: 3–15 Hz, 40 Hz, real four-second windows,
at most eight windows per patient and no padded input sequences. Temporal mean
and log-variance are pooled per window then averaged per patient. The spectral
control uses full-recording features. These differences limit cross-method
interpretation but are identical between random and pretrained encoders.

Primary: pretrained-minus-random frozen macro-F1. Secondary: pretrained versus
spectral and fixed 50:50 pretrained/spectral probability fusion versus spectral.
All four arms and all individual seeds are reported. No parameter search or
outlier removal. Patient identities are cohort-qualified; lack of real cross-year
participant overlap is assumed, not verified by a clinical linkage register.

## Reproduce

Finish retrieving Data/raw_quaternion and NewData first. Uses dependencies from
PR #7, with no additional package.

```bash
python -m unittest discover -s tests -p test_out_contrastive.py -v
python -m experiments.out_contrastive --output artifacts/out_contrastive
```

Three tests verify positive-pair indexing, loss symmetry and finite gradients,
and independent patient embeddings. Source losses, per-seed metrics, aggregate
metrics and 2000 paired patient-bootstrap intervals are recorded. Fixed-prediction
intervals exclude full retraining and historical model-selection uncertainty.
This remains exploratory on previously investigated patients. No production
model is changed. Source checkpoints and patient-level outputs stay uncommitted.
