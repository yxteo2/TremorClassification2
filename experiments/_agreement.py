"""Descriptive agreement; no inference about whether clinical labels are wrong."""

from itertools import combinations

import numpy as np


def pairwise_agreement(predictions):
    """Fraction of unordered recording pairs agreeing, within one patient."""
    p = np.asarray(predictions)
    if p.ndim != 1 or len(p) < 2:
        raise ValueError("At least two recording predictions are required")
    _, counts = np.unique(p, return_counts=True)
    return float(np.sum(counts * (counts - 1)) / (len(p) * (len(p) - 1)))


def agreement_summary(patients):
    """Summarise eligible patients with equal patient weights.

    Each item has predictions, cohort, label, correct, and confidence. Controls
    pair distinct patients within EXACT cohort/label/correctness strata, then
    average over all cross-patient recording pairs. Stratum means are weighted
    by eligible patient count, matching the observed subgroup composition.
    Singleton strata are excluded from BOTH matched columns, but retained in
    the all-patient descriptive column. Counts make this missingness explicit.
    Correctness is conditional on the patient-level model prediction, not a
    claim about the clinical truth. This control is descriptive, not causal.
    """
    result = {}
    for correct in (True, False):
        group = [p for p in patients if p["correct"] == correct]
        strata = {}
        for p in group:
            strata.setdefault((p["cohort"], p["label"]), []).append(p)
        observed, controls = [], []
        for members in strata.values():
            if len(members) < 2:
                continue
            cross = [float(np.mean(np.asarray(a["predictions"])[:, None] ==
                                   np.asarray(b["predictions"])[None, :]))
                     for a, b in combinations(members, 2)]
            observed.extend(pairwise_agreement(p["predictions"]) for p in members)
            controls.extend([float(np.mean(cross))] * len(members))
        mean = lambda values: float(np.mean(values)) if values else float("nan")
        result[correct] = dict(
            agreement=mean([pairwise_agreement(p["predictions"]) for p in group]),
            matched_agreement=mean(observed), control=mean(controls),
            confidence=mean([p["confidence"] for p in group]),
            n=len(group), n_matched=len(observed))
    return result
