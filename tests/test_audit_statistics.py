"""Small synthetic regressions; no clinical data or model training needed."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from experiments._agreement import agreement_summary, pairwise_agreement
from experiments.patient_level_ci import metrics_on, patient_bootstrap, save_predictions


def patient(predictions, cohort='A', label=1, correct=False):
    return dict(predictions=predictions, cohort=cohort, label=label,
                correct=correct, confidence=0.5)


class AgreementTests(unittest.TestCase):
    def test_three_and_four_recordings_use_pairs(self):
        self.assertAlmostEqual(pairwise_agreement([0, 0, 1]), 1 / 3)
        self.assertAlmostEqual(pairwise_agreement([0, 0, 1, 1]), 1 / 3)
        self.assertEqual(pairwise_agreement([2, 2]), 1)
        with self.assertRaises(ValueError):
            pairwise_agreement([0])

    def test_singleton_excluded_from_both_matched_columns(self):
        s = agreement_summary([patient([0, 0]), patient([1, 1]),
                               patient([0, 1], cohort='B')])[False]
        self.assertEqual(s['n'], 3)
        self.assertEqual(s['n_matched'], 2)
        self.assertAlmostEqual(s['agreement'], 2 / 3)
        self.assertEqual(s['matched_agreement'], 1)
        self.assertEqual(s['control'], 0)

    def test_no_pooling_across_cohort_class_or_correctness(self):
        ps = [patient([0, 0]), patient([0, 0], cohort='B'),
              patient([0, 0], label=2), patient([0, 0], correct=True)]
        s = agreement_summary(ps)
        for correct in (True, False):
            self.assertEqual(s[correct]['n_matched'], 0)
            self.assertTrue(np.isnan(s[correct]['control']))

    def test_patient_weights_not_recording_weights(self):
        ps = [patient([0, 0]), patient([1] * 20),
              patient([0, 1], cohort='B'), patient([0, 1], cohort='B')]
        s = agreement_summary(ps)[False]
        self.assertEqual(s['matched_agreement'], 0.5)
        self.assertEqual(s['control'], 0.25)

    def test_empty_groups(self):
        s = agreement_summary([])
        self.assertEqual(s[False]['n'], 0)
        self.assertTrue(np.isnan(s[True]['agreement']))


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        self.y = np.array([0, 1, 2, 0, 1, 2])
        self.a = [(np.array([0, 1, 2, 3]), np.array([0, 0, 2, 1])),
                  (np.array([2, 3, 4, 5]), np.array([1, 0, 1, 0]))]
        self.b = [(te, self.y[te]) for te, _ in self.a]

    def test_identical_arms_zero_and_swapping_reverses_sign(self):
        z = patient_bootstrap(self.y, self.a, self.a, 6, n=12)
        np.testing.assert_array_equal(z, np.zeros((12, 5)))
        ab = patient_bootstrap(self.y, self.a, self.b, 6, n=12)
        ba = patient_bootstrap(self.y, self.b, self.a, 6, n=12)
        np.testing.assert_allclose(ab, -ba)

    def test_patient_multiplicity_shared_across_overlapping_splits(self):
        expected = []
        rng = np.random.default_rng(4)
        for _ in range(8):
            draw = rng.integers(0, 6, 6)
            diffs = []
            for (te, pa), (_, pb) in zip(self.a, self.b):
                lookup = {int(p): i for i, p in enumerate(te)}
                positions = [lookup[int(p)] for p in draw if int(p) in lookup]
                if positions:
                    diffs.append(metrics_on(self.y[te][positions], pb[positions]) -
                                 metrics_on(self.y[te][positions], pa[positions]))
            expected.append(np.mean(diffs, axis=0))
        np.testing.assert_allclose(patient_bootstrap(
            self.y, self.a, self.b, 6, n=8, seed=4), expected)

    def test_missing_class_draws_are_retained(self):
        y = np.array([0])
        a = [(np.array([0]), np.array([1]))]
        b = [(np.array([0]), np.array([0]))]
        result = patient_bootstrap(y, a, b, 1, n=3)
        np.testing.assert_allclose(result, np.tile([1, 0, 0, 1/3, 1/3], (3, 1)))

    def test_misaligned_truncated_or_duplicate_splits_rejected(self):
        bad = [(self.b[0][0][::-1], self.b[0][1][::-1]), self.b[1]]
        for other in (bad, self.b[:1]):
            with self.assertRaises(ValueError):
                patient_bootstrap(self.y, self.a, other, 6, n=1)
        dup = [(np.array([0, 0]), np.array([0, 0]))]
        with self.assertRaises(ValueError):
            patient_bootstrap(self.y, dup, dup, 6, n=1)

    def test_prediction_export_round_trip(self):
        d = dict(y=self.y, patient_ids=np.array([f'A::{i}' for i in range(6)]),
                 key=np.array(['A'] * 6), SPEC={'welch': np.zeros((6, 2))})
        for name in ('DESC', 'ASYM', 'HAVE', 'TRAJ'):
            d[name] = np.zeros((6, 2))
        with tempfile.TemporaryDirectory() as tmp, patch(
                'subprocess.check_output', return_value='test-revision\n'):
            out = Path(tmp) / 'audit.json'
            save_predictions(out, d, {'base': (np.zeros((2, 5)), self.a)})
            payload = json.loads(out.read_text())
            self.assertEqual(payload['revision'], 'test-revision')
            self.assertEqual(payload['patient_ids'], d['patient_ids'].tolist())
            self.assertEqual(payload['arms']['base']['splits'][0]['test_indices'],
                             self.a[0][0].tolist())
            self.assertEqual(len(payload['feature_sha256']), 64)


if __name__ == '__main__':
    unittest.main()
