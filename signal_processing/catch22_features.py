"""catch22: the canonical 22 time-series features, on the tremor waveform itself.

Every feature family in this project is derived from the **power spectrum** --
descriptors, harmonics, axis shape, amplitude modulation -- or from the
instantaneous-frequency trajectory. None reads the waveform's *temporal
structure* directly.

Häring et al. (Movement Disorders, 2025) did exactly that on 414 patients with
hand accelerometry and reported **81.8 % accuracy / 86.4 % sensitivity / 76.6 %
specificity** for PD vs ET from massive time-series feature extraction, against
**70.4 %** for the Tremor Stability Index -- which this repo already implements
(`stability.py`) and measures at AUC 0.757 on PADS. Their interpretation is
mechanistic and testable:

    "different discrete but stable signal states in PD indicate several central
     oscillators, while signal characteristics in ET point towards a singular
     pacemaker"

catch22 is the canonical 22-feature distillation of the hctsa library, and
several of its features measure precisely that claim:

    SB_TransitionMatrix_3ac_sumdiagcov   transitions between discretised states
    SB_BinaryStats_mean_longstretch1     how long the signal stays in one state
    SB_BinaryStats_diff_longstretch0     the same for the differenced signal
    SB_MotifThree_quantile_hh            entropy of 3-symbol motifs
    DN_HistogramMode_5 / _10             multimodality of the amplitude
                                         distribution -- several oscillators
                                         should give several modes
    SC_FluctAnal_2_dfa / _rsrangefit      long-range scaling
    PD_PeriodicityWang_th0_01            periodicity strength

**Why 22 and not tsfresh's thousands.** This project's most robust finding is
that dimensionality binds harder than information at 404 patients with 49 ET:
thirteen feature unions have underperformed their best member. tsfresh would emit
hundreds of features and need a selection stage fitted inside every fold.
catch22 is a *fixed* 22-dimensional set chosen once, offline, on 93 different
datasets -- comparable in size to the 10 spectral descriptors already in use, and
with no selection to leak.

## Making it rotation-invariant

catch22 takes one time series; the sensor gives three axes. Two reductions are
rotation-invariant:

* the **magnitude** ||w(t)||, which is wrong here -- for a linear oscillation
  ||A sin(2 pi f t)|| has fundamental **2f**, so every frequency-related feature
  is distorted;
* the **principal-axis projection**, s(t) = v . w(t) where v is the leading
  eigenvector of the time-domain covariance. Tremor is close to a linear
  oscillation (linearity 0.997), so this recovers the waveform almost intact and
  is invariant to how the sensor was worn.

The projection is used. Its sign is arbitrary under rotation, so it is fixed by
forcing non-negative skewness, which makes the pipeline deterministic and keeps
the sign-sensitive features (`CO_trev_1_num`, the outlier features) meaningful.

The signal is band-passed to 3-15 Hz first, matching the band every other family
uses, and z-scored per recording -- amplitude is not comparable across cohorts,
and catch22 features are defined for z-scored input.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.signal import butter, filtfilt

try:
    import pycatch22
    HAVE_CATCH22 = True
except ImportError:                                    # pragma: no cover
    HAVE_CATCH22 = False

FS, F_LO, F_HI = 100.0, 3.0, 15.0

FEATURE_NAMES = [
    "DN_HistogramMode_5", "DN_HistogramMode_10", "CO_f1ecac", "CO_FirstMin_ac",
    "CO_HistogramAMI_even_2_5", "CO_trev_1_num", "MD_hrv_classic_pnn40",
    "SB_BinaryStats_mean_longstretch1", "SB_TransitionMatrix_3ac_sumdiagcov",
    "PD_PeriodicityWang_th0_01", "CO_Embed2_Dist_tau_d_expfit_meandiff",
    "IN_AutoMutualInfoStats_40_gaussian_fmmi", "FC_LocalSimple_mean1_tauresrat",
    "DN_OutlierInclude_p_001_mdrmd", "DN_OutlierInclude_n_001_mdrmd",
    "SP_Summaries_welch_rect_area_5_1", "SB_BinaryStats_diff_longstretch0",
    "SB_MotifThree_quantile_hh", "SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1",
    "SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1", "SP_Summaries_welch_rect_centroid",
    "FC_LocalSimple_mean3_stderr",
]

# the subset that measures the "several oscillators vs one pacemaker" claim
STATE_FEATURES = ["SB_TransitionMatrix_3ac_sumdiagcov",
                  "SB_BinaryStats_mean_longstretch1",
                  "SB_BinaryStats_diff_longstretch0",
                  "SB_MotifThree_quantile_hh",
                  "DN_HistogramMode_5", "DN_HistogramMode_10"]


def principal_projection(x, fs=FS, f_lo=F_LO, f_hi=F_HI):
    """(3, T) -> (T,) rotation-invariant 1-D waveform, band-passed and z-scored.

    Returns ``None`` when the recording is too short or degenerate.
    """
    x = np.asarray(x, float)
    if x.ndim == 1:
        x = x[None, :]
    x = x[:3]
    if x.shape[-1] < 64:
        return None
    x = x - x.mean(-1, keepdims=True)

    nyq = fs / 2.0
    try:
        b, a = butter(4, [f_lo / nyq, min(f_hi / nyq, 0.99)], btype="band")
        x = filtfilt(b, a, x, axis=-1)
    except Exception:
        return None

    if x.shape[0] >= 2:
        C = np.cov(x)
        if not np.isfinite(C).all():
            return None
        v = np.linalg.eigh(C)[1][:, -1]
        s = v @ x
    else:
        s = x[0]

    sd = s.std()
    if not np.isfinite(sd) or sd <= 1e-12:
        return None
    s = (s - s.mean()) / sd
    # the eigenvector sign is arbitrary; fix it deterministically
    m3 = float((s ** 3).mean())
    if m3 < 0:
        s = -s
    return s


def features(x, fs=FS):
    """catch22 of one recording's principal-axis waveform, or None."""
    if not HAVE_CATCH22:
        raise ImportError("pycatch22 is required: pip install pycatch22")
    s = principal_projection(x, fs=fs)
    if s is None:
        return None
    try:
        out = pycatch22.catch22_all(list(map(float, s)), catch24=False)
    except Exception:
        return None
    v = np.asarray(out["values"], float)
    if v.shape[0] != len(FEATURE_NAMES) or not np.isfinite(v).all():
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        if v.shape[0] != len(FEATURE_NAMES):
            return None
    return v


def patient_table(recs, ch=slice(3, 6), fs=FS):
    """(patients, 22) averaged over each patient's recordings, plus labels/ids."""
    rows, lab = defaultdict(list), {}
    for r in recs:
        x = r.x[ch] if r.x.shape[0] > 3 else r.x
        v = features(x, fs=fs)
        if v is None:
            continue
        rows[r.subject].append(v)
        lab[r.subject] = r.y
    pats = sorted(rows)
    X = np.array([np.mean(rows[p], 0) for p in pats]) if pats else \
        np.zeros((0, len(FEATURE_NAMES)))
    return (np.nan_to_num(X), np.array([lab[p] for p in pats]),
            np.array(pats))
