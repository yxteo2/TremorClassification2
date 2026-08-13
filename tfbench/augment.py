"""Crop + noise augmentation with a patient-level held-out test set.

Two things this module is careful about, both of which are easy to get wrong:

1. **Augment every class, not just the minority.** Augmenting only ET makes
   "looks like a crop" a cue for ET. The model can learn that instead of
   tremor, and a test set of un-augmented recordings will not reveal it.
   Imbalance is handled with class weights; augmentation is applied uniformly.

2. **Crops are not new patients.** Several crops of one recording are highly
   correlated: they add within-patient variation and act as a regulariser, but
   they cannot add between-patient variability -- which is what 16 ET subjects
   actually limits. Expect a regularisation effect, not a sample-size fix.

The split is patient-level and made ONCE, before any augmentation, so no crop
of a test patient can appear in training.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from tremor.data import Recording


def holdout_split(recs, n_per_class=10, seed=0):
    """Hold out ``n_per_class`` PATIENTS of each class. Returns (train, test)."""
    by_cls = defaultdict(set)
    for r in recs:
        by_cls[r.y].add(r.subject)
    rng = np.random.default_rng(seed)
    test_subj = set()
    for cls, subs in by_cls.items():
        subs = np.array(sorted(subs))
        k = min(n_per_class, len(subs))
        test_subj.update(rng.choice(subs, k, replace=False).tolist())
    train = [r for r in recs if r.subject not in test_subj]
    test = [r for r in recs if r.subject in test_subj]
    return train, test


def augment(recs, n_crops=4, crop_frac=0.6, noise_sd=0.0, seed=0,
            classes=None, min_len=512):
    """Random time-crops (+ optional noise) of each recording.

    ``classes=None`` augments everything -- the correct default. Passing a
    subset reproduces the class-conditional version, which leaks.

    Noise is scaled to each recording's own std, so it is a relative
    perturbation rather than a fixed absolute level across very different
    amplitudes.
    """
    rng = np.random.default_rng(seed)
    out = list(recs)
    for r in recs:
        if classes is not None and r.y not in classes:
            continue
        T = r.x.shape[1]
        n = max(int(T * crop_frac), min_len)
        if n >= T:
            continue
        for j in range(n_crops):
            s = int(rng.integers(0, T - n))
            x = r.x[:, s:s + n].copy()
            if noise_sd > 0:
                x = x + rng.normal(0, noise_sd * (x.std() + 1e-12), x.shape
                                   ).astype(x.dtype)
            out.append(Recording(x=x, y=r.y, subject=r.subject, path=r.path,
                                 condition=r.condition))
    return out
