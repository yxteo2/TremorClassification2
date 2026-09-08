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

## Complete results

All rows test the same 148 2015 patients; all 53 NewData patients are training-only.

| Method | Accuracy | Macro-F1 | ET precision | ET recall |
|---|---:|---:|---:|---:|
| Standard scaling, nested selection | .703 | .629 | .276 | .533 (8/15) |
| Robust scaling, nested selection | .662 | .578 | .231 | .400 (6/15) |
| Standard scaling, fixed logistic | .716 | .642 | .308 | .533 (8/15) |
| Robust scaling, fixed logistic | .682 | .613 | .286 | .533 (8/15) |

Primary robust-minus-standard macro-F1 change: -.051, paired 95% interval
[-.113,+.005]. Fixed-model secondary change: -.029, interval [-.055,-.006].
These results do not support replacing standard scaling with robust scaling.
Fixed logistic is a practical candidate, but its .642 is an exploratory result
after earlier model search and should not be advertised as externally validated.
The standard nested control exactly reproduces PR #5's aggregate metrics and
confusion matrix. This is not verification of identical patient-level decisions,
because the prior augmentation predictions were not persisted remotely.

Across 549 loaded 2015 and 202 loaded NewData OUT/REST recordings, no nonfinite
source quaternion values or near-zero norms were found. Two 2015 REST recordings
from one normal participant had norms around 9.4–10.1 throughout all sensors.
No NewData recordings triggered the >10% norm-deviation review flag. The existing
conversion already normalizes quaternions to unit length; a scale discrepancy
alone is not evidence of unusable orientation data. No records were excluded.
Dividing those source quaternions by 10 before the existing conversion produced
finite angular velocities with maximum absolute differences below 1.4e-5 rad/s
and RMS differences below 2.5e-6 rad/s versus the unchanged input. This supports
the normalization explanation, not the overall validity of every motion sample.
This limited audit does not prove all recordings are artifact-free.

Recommendation: retain patients, standard scaling and the all-sensor OUT+REST
candidate. Investigate acquisition scaling metadata for the flagged recordings
before any correction beyond existing normalization. Quality-based exclusion
was not benchmarked because these checks did not establish unusable recordings;
deleting difficult patients or choosing an exclusion threshold from their test
errors is not warranted.

Complete aggregate metrics, intervals, feature hashes and selected models are in
[robust_inhouse_summary.json](robust_inhouse_summary.json).
