# Robust scaling without patient removal

## Fixed comparison

This follows PR #5: 148 complete OUT+REST 2015 patients (61 N, 72 PD, 15 ET),
with 53 NewData patients (25 N, 22 PD, 6 ET) added only to training. No PADS.
The same five stratified outer folds use seed 0. Inner validation contains
2015 patients only; NewData stays in inner training.

Primary: replace StandardScaler with default RobustScaler (median/IQR) in
the same nested representation/model selection. Secondary: compare scalers
in a fixed all-sensor OUT+REST balanced logistic C=.1 model. All scaling is
fitted within training folds. No clipping, patient removal, seed sweep,
rejection threshold or test-label-based quality selection is used.

The shared features are six spectral power/shape features per sensor/task,
with repeated recordings averaged within each patient and task. Nested
selection covers 6/18/36 feature representations and balanced logistic
C=.01,.1,1 or RBF SVM C=.1,1,10, exactly as in PR #5.

RobustScaler protects estimates of center and scale from extreme values;
it does not bound an outlier's influence on the fitted classifier.
[Official documentation](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html).

## Quality audit

Check source quaternions for nonfinite values, norms below 1e-6, and norm
deviations over 10%. The last is a review flag, not a validated corruption
threshold. Audit only records returned by existing loaders; NewData can skip
missing-sensor, short or unconvertible files before this stage. These checks
cannot verify labels, placement, timing, task adherence, every movement
artifact, or cross-cohort participant overlap. No exclusion is justified by
high feature values or prediction disagreement alone.

## Reproduce

Retrieve Data/raw_quaternion and NewData completely first. Dependencies match
PR #5; no new package is required.

```bash
python -m unittest discover -s tests -p test_robust_inhouse.py -v
python -m experiments.robust_inhouse --output artifacts/robust_inhouse
```

Three tests cover malformed quaternions, NewData remaining inside inner training,
and scaler fitting being unaffected by test prediction calls. Patient-level
predictions and split manifests remain uncommitted research artifacts.
Intervals use 2000 paired fixed-prediction patient bootstraps; they exclude
retraining and historical model-selection uncertainty. This is exploratory
evaluation on previously investigated cohorts, not prospective validation.
