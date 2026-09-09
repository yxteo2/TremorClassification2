"""Synthetic invariants and split isolation; no patient files needed."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from signal_processing.scattering_features import from_waveform, patient_table
from signal_processing.waveform import waveform
from experiments.scattering_benchmark import make_splits, fit_scattering, metrics


class ScatteringTests(unittest.TestCase):
    def setUp(self):
        t = np.arange(1600) / 100
        self.x = np.array([np.sin(2 * np.pi * 6 * t),
                           0.5 * np.sin(2 * np.pi * 6 * t + 0.4),
                           0.1 * np.cos(2 * np.pi * 6 * t)])

    def record(self, subject="p1", label=1, x=None):
        return SimpleNamespace(subject=subject, y=label, path=subject,
                               x=self.x if x is None else x)

    def test_dimensions_and_first_order_prefix(self):
        a, b = from_waveform(waveform(self.x))
        self.assertGreater(len(a), 0)
        self.assertGreater(len(b), len(a))
        np.testing.assert_array_equal(a, b[:len(a)])
        self.assertTrue(np.isfinite(b).all())

    def test_sign_invariance(self):
        w = waveform(self.x)
        np.testing.assert_allclose(from_waveform(w)[1], from_waveform(-w)[1], atol=1e-10)

    def test_rotation_and_scale_invariance(self):
        q, _ = np.linalg.qr(np.random.default_rng(0).normal(size=(3, 3)))
        a = from_waveform(waveform(self.x))[1]
        b = from_waveform(waveform(3 * q @ self.x))[1]
        np.testing.assert_allclose(a, b, atol=2e-4)

    def test_modulation_changes_second_order(self):
        t = np.arange(384) / 40
        carrier = np.sin(2 * np.pi * 6 * t)
        modulated = (1 + 0.7 * np.sin(2 * np.pi * 2 * t)) * carrier
        a, b = from_waveform(carrier)
        _, c = from_waveform(modulated)
        self.assertGreater(np.linalg.norm(b[len(a):] - c[len(a):]), 0.5)

    def test_patient_aggregation_uses_every_record(self):
        recs = [self.record(), self.record(), self.record()]
        fake = [(np.array([i]), np.array([i, i])) for i in (1., 2., 6.)]
        with patch("signal_processing.scattering_features.from_waveform", side_effect=fake):
            a, b, y, p, excluded = patient_table(recs)
        np.testing.assert_allclose(a, [[3.]])
        self.assertEqual(len(p), 1)
        self.assertEqual(excluded, [])

    def test_conflicting_labels_rejected(self):
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            patient_table([self.record(label=1), self.record(label=2)])

    def test_short_and_nonfinite_recordings_excluded(self):
        recs = [self.record(), self.record("short", x=self.x[:, :500]),
                self.record("bad", x=np.full_like(self.x, np.nan))]
        *_, p, excluded = patient_table(recs)
        self.assertEqual(p.tolist(), ["p1"])
        self.assertEqual(excluded, ["short", "bad"])

    def test_invalid_waveform_rejected(self):
        for w in (np.zeros(10), np.full(384, np.nan)):
            with self.assertRaises(ValueError):
                from_waveform(w)


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.y = np.repeat([0, 1, 2], 30)
        self.p = np.array([f"p{i}" for i in range(90)])

    def test_no_overlap_and_one_test_prediction_per_patient(self):
        splits = make_splits(self.y, self.p, self.y)
        seen = []
        for tr, va, te in splits:
            self.assertFalse(set(tr) & set(va) or set(tr) & set(te) or set(va) & set(te))
            self.assertEqual(len(tr) + len(va) + len(te), len(self.y))
            seen.extend(te)
        self.assertEqual(sorted(seen), list(range(len(self.y))))

    def test_duplicate_patients_rejected(self):
        p = self.p.copy()
        p[1] = p[0]
        with self.assertRaisesRegex(ValueError, "one row"):
            make_splits(self.y, p, self.y)

    def test_test_data_cannot_change_selection_or_validation_predictions(self):
        tr, va, te = make_splits(self.y, self.p, self.y)[0]
        X = np.random.default_rng(0).normal(size=(90, 5))
        X[:, 0] += self.y
        a, _, c = fit_scattering(X, self.y, tr, va, te)
        changed_x, changed_y = X.copy(), self.y.copy()
        changed_x[te] = 1e6
        changed_y[te] = (self.y[te] + 1) % 3
        b, _, d = fit_scattering(changed_x, changed_y, tr, va, te)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(c, d)

    def test_missing_prediction_class_has_zero_precision(self):
        s = metrics(np.array([0, 1, 2]), np.array([0, 0, 0]), np.ones((3, 3)) / 3)
        self.assertEqual(s["precision"][2], 0)
        self.assertEqual(s["recall"][2], 0)


if __name__ == "__main__":
    unittest.main()
