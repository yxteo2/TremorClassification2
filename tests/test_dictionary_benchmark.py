import unittest
import numpy as np
from scipy import sparse
from experiments.dictionary_benchmark import pool_words, stack_records, fit_dictionary


class ToyDictionary:
    def fit_transform(self, X, y):
        self.scale = np.abs(X).mean() + 1e-6
        return self.transform(X)

    def transform(self, X):
        v = np.abs(X[:, 0]) / self.scale
        return sparse.csr_matrix(np.column_stack([v[:, :4].mean(1), v[:, 4:].mean(1)]))


class DictionaryTests(unittest.TestCase):
    def test_pooling(self):
        got = pool_words([[1., 3.], [3., 1.], [0., 2.]], np.array([0, 0, 1]), 2).toarray()
        np.testing.assert_allclose(got, [[np.sqrt(.5), np.sqrt(.5)], [0., 1.]])

    def test_duplicate_recordings_do_not_reweight_patient(self):
        a = pool_words([[1., 2.], [3., 4.]], np.array([0, 1]), 2)
        b = pool_words([[1., 2.], [1., 2.], [3., 4.]], np.array([0, 0, 1]), 2)
        np.testing.assert_allclose(a.toarray(), b.toarray())

    def test_empty_bag_rejected(self):
        with self.assertRaises(ValueError):
            stack_records([np.empty((0, 8))], [0])
        with self.assertRaises(ValueError):
            pool_words([[1., 2.]], np.array([0]), 2)

    def test_invalid_counts_rejected(self):
        for value in (-1., np.nan):
            with self.assertRaises(ValueError):
                pool_words([[value, 2.]], np.array([0]), 1)

    def test_stack_preserves_patient_order(self):
        x, owners = stack_records([np.ones((2, 8)), np.zeros((1, 8))], [1, 0])
        self.assertEqual(x.shape, (3, 1, 8))
        self.assertEqual(owners.tolist(), [0, 1, 1])
        self.assertEqual(x[0].sum(), 0.)

    def test_test_changes_cannot_change_validation_predictions(self):
        rng = np.random.default_rng(0)
        waves = [rng.normal(size=(2, 8)) for _ in range(18)]
        y = np.tile([0, 1, 2], 6)
        tr, va, te = np.arange(12), np.arange(12, 15), np.arange(15, 18)
        a, _, _ = fit_dictionary(waves, y, tr, va, te, ToyDictionary)
        changed = list(waves)
        for i in te:
            changed[i] = np.full((2, 8), 1e6)
        yy = y.copy()
        yy[te] = 0
        b, _, _ = fit_dictionary(changed, yy, tr, va, te, ToyDictionary)
        np.testing.assert_array_equal(a, b)


if __name__ == "__main__":
    unittest.main()
