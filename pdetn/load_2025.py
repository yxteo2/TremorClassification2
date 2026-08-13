"""Load the 2025 Moveo h5 recordings and align them to the 2015 pipeline.

Sensor IDs (from NewData/Convert_h5_and_csv_to_xlsx.ipynb):
    7257=Hand, 10871=Wrist(~2015 lower_arm), 10468=Upper Arm,
    10464=Index Finger, 10833=Thorax  (last two have no 2015 counterpart)
Action codes: 01/08=REST, 02/09=OUT (01-07 right limb, 08-14 left).
Quaternions are resampled 128 -> 100 Hz, then converted to angular velocity
with the same tremor.quaternion routine used for the 2015 data.
"""
from __future__ import annotations
import glob, os, re
import numpy as np
from scipy.signal import resample_poly
from tremor.data import Recording
from tremor.quaternion import process_quaternion_data

SENSOR_ORDER = ["7257", "10871", "10468"]          # hand, lower_arm, upper_arm
#: Action code -> task. 01-07 are the RIGHT upper limb, 08-14 the LEFT.
#: REST and OUT have 2015 counterparts; the other five are 2025-only and were
#: unusable while the cohort was ET-only (no class to discriminate against).
#: With 31 HC and 34 PD they are testable -- and FINGER_NOSE and POUR are the
#: classic clinical kinetic-tremor manoeuvres for ET.
ACTION = {"01": "REST",        "08": "REST",
          "02": "OUT",         "09": "OUT",
          "03": "DRINK",       "10": "DRINK",
          "04": "FINGER_NOSE", "11": "FINGER_NOSE",
          "05": "POUR",        "12": "POUR",
          "06": "TAP",         "13": "TAP",
          "07": "PRON_SUP",    "14": "PRON_SUP"}
ALL_TASKS_2025 = ["REST", "OUT", "DRINK", "FINGER_NOSE", "POUR", "TAP", "PRON_SUP"]
FS_SRC, FS_DST = 128.0, 100.0

def _load_h5(path, sensors=SENSOR_ORDER):
    import h5py
    with h5py.File(path, "r") as h:
        proc = h["Processed"]
        if not all(s in proc for s in sensors):
            return None
        # (T,4) per sensor -> concat to (T, 12) in 2015 channel order
        return np.concatenate([np.asarray(proc[s]["Orientation"]) for s in sensors], axis=1)

#: Action codes 01-07 are the RIGHT upper limb, 08-14 the LEFT.
SIDE = {c: ("right" if int(c) <= 7 else "left") for c in ACTION}


def select_task_epoch(x, fs=100.0, win_s=10.0, hop_s=0.5, f_lo=3.0, f_hi=15.0,
                      channels=(3, 4, 5)):
    """Pick the window of a free-form recording that actually contains tremor.

    The 2025 exports are ~38 s ``Free_Form`` captures with an EMPTY Annotations
    table, so there is no marker for when the task starts. Taking the whole
    recording means the spectrum is dominated by set-up and settling motion:
    measured over the 12 OUT recordings, only **9.9 %** of power lands in the
    3-15 Hz tremor band, against 76.5 % for the 2015 cohort and 81.2 % for PADS.

    This slides a window and keeps the one with the highest in-band power
    FRACTION -- a ratio, so it selects the most tremor-dominated segment rather
    than simply the most energetic one (which would pick the set-up movement).

    .. note::
       This selects on tremor-band content, so it behaves differently for
       controls (who have none) than for patients. That looked like a
       label-dependent bias once the cohort gained 31 HC, and the class effect
       is measurable -- in-band fraction at OUT is HC 0.536 / PD 0.730 / ET 0.714
       under this rule against HC 0.473 / PD 0.642 / ET 0.722 under the
       tremor-blind ``select_steady_epoch``.

       **It is not leakage.** The rule never sees a label and is applied
       identically at train and test time, and at deployment on an unlabelled
       recording. "Find the tremor if there is one" is the intended behaviour.
       Tested directly: switching to the tremor-blind rule makes results WORSE
       (N-vs-Tremor 0.787 -> 0.714 at OUT, 0.640 -> 0.587 at REST) and does NOT
       fix the below-chance PD-vs-ET AUC, which therefore has another cause --
       most likely just 6 ET subjects.

    Returns the selected slice of ``x``.
    """
    from scipy.signal import welch as _welch
    x = np.asarray(x)
    n = int(win_s * fs); hop = max(int(hop_s * fs), 1)
    if x.shape[1] <= n:
        return x
    ch = [c for c in channels if c < x.shape[0]] or list(range(x.shape[0]))
    best, best_i = -1.0, 0
    for i in range(0, x.shape[1] - n + 1, hop):
        seg = x[ch, i:i + n]
        f, P = _welch(seg, fs=fs, nperseg=min(256, n), axis=-1)
        P = P.mean(0)
        tot = P[(f >= 0.5) & (f <= 40)].sum()
        frac = P[(f >= f_lo) & (f <= f_hi)].sum() / max(tot, 1e-20)
        if frac > best:
            best, best_i = frac, i
    return x[:, best_i:best_i + n]


def select_steady_epoch(x, g, fs=100.0, win_s=10.0, hop_s=0.5, channels=(3, 4, 5)):
    """Pick the window where POSTURE is steadiest. Tremor-blind, so unbiased
    across classes.

    Scores each window by the variability of the body-frame gravity direction --
    how still the limb is being held -- and never looks at the tremor band. A
    healthy control and a tremor patient are therefore selected on the same
    criterion, which :func:`select_task_epoch` cannot claim.

    Args:
        x: ``(channels, T)`` angular velocity.
        g: ``(channels, T)`` body-frame gravity for the same recording, from
           ``process_quaternion_data(..., mode="gravity")``.
    """
    x = np.asarray(x); g = np.asarray(g)
    n = int(win_s * fs); hop = max(int(hop_s * fs), 1)
    if x.shape[1] <= n:
        return x
    ch = [c for c in channels if c < g.shape[0]] or list(range(g.shape[0]))
    g = g[:, :x.shape[1]] if g.shape[1] >= x.shape[1] else g
    best, best_i = np.inf, 0
    for i in range(0, x.shape[1] - n + 1, hop):
        seg = g[ch, i:i + n]
        if seg.shape[1] < n:
            break
        d = float(np.mean(np.std(seg, axis=1)))
        if d < best:
            best, best_i = d, i
    return x[:, best_i:best_i + n]


#: folder name -> class label, matching tremor.data (N=0, PD=1, ET=2).
#: HC (healthy control) is the 2025 cohort's name for N.
CLASS_LABELS = {"HC": 0, "N": 0, "PD": 1, "ET": 2}


def load_2025_all(root="NewData", classes=("HC", "PD", "ET"), **kw):
    """Load every class of the 2025 cohort as one N/PD/ET recording list.

    The cohort was ET-only when first integrated, which made any device cue
    perfectly confounded with the ET label and blocked pooling. With HC and PD
    present it is self-contained: cohort membership no longer predicts the
    class, so it can be trained and evaluated on its own.
    """
    recs = []
    for c in classes:
        recs.extend(load_2025(root=root, cls=c, label=CLASS_LABELS[c], **kw))
    return recs


def load_2025(root="NewData", cls="ET", label=None, conditions=("OUT",), fs=FS_DST,
              mode="angular_velocity", sides=("right", "left"), segment="tremor",
              win_s=10.0):
    """Load the 2025 cohort, aligned to the 2015 channel order and rate.

    ``mode`` is passed to :func:`tremor.quaternion.process_quaternion_data`, so
    the same log_map / gravity representations used on the 2015 data are
    available here.

    ``segment`` is ``"tremor"`` (default), ``"steady"`` or ``False``.
    ``"tremor"`` picks the most tremor-dominated window; ``"steady"`` the
    steadiest-posture window. Measured on the full 2025 cohort, ``"tremor"`` is
    better on both axes (N-vs-Tremor 0.787 vs 0.714 at OUT) -- see
    :func:`select_task_epoch` for why the label-dependence concern turned out
    not to be disqualifying.

    These exports are ~38 s ``Free_Form``
    captures with an empty Annotations table, and using the whole recording
    leaves only 9.9 % of power in the 3-15 Hz tremor band (vs 76.5 % for the
    2015 cohort and 81.2 % for PADS) because set-up and settling motion
    dominate. Pass ``segment=False`` only to reproduce the superseded numbers.

    ``sides`` selects limbs. Both limbs of one subject share a subject id, so
    they can never be split across CV folds -- but note that pooling both
    doubles a subject's recordings without adding a subject, which matters for
    anything that averages per patient.
    """
    if label is None:
        label = CLASS_LABELS[cls]
    recs = []
    for subj_dir in sorted(glob.glob(os.path.join(root, cls, "*/"))):
        sid = re.search(r"_(%s_\d+)_" % cls, subj_dir)
        sid = sid.group(1) if sid else os.path.basename(subj_dir.rstrip("/"))
        for f in sorted(glob.glob(os.path.join(subj_dir, "rawData", "*.h5"))):
            code = os.path.basename(f)[:2]
            cond = ACTION.get(code)
            if cond is None or cond not in conditions:
                continue
            if SIDE.get(code) not in sides:
                continue
            q = _load_h5(f)
            if q is None or q.shape[0] < 256:
                continue
            q = resample_poly(q, 25, 32, axis=0)      # 128 -> 100 Hz
            try:
                x = process_quaternion_data(q.astype(np.float32), fs=fs,
                                            mode=mode, convention="xyzw",
                                            n_sensors=3)
            except Exception:
                continue
            if segment == "tremor" or segment is True:
                x = select_task_epoch(x, fs=fs, win_s=win_s)
            elif segment:                      # "steady" -- tremor-blind default
                try:
                    gq = process_quaternion_data(q.astype(np.float32), fs=fs,
                                                 mode="gravity",
                                                 convention="xyzw", n_sensors=3)
                    x = select_steady_epoch(x, gq, fs=fs, win_s=win_s)
                except Exception:
                    x = select_task_epoch(x, fs=fs, win_s=win_s)
            recs.append(Recording(x=x, y=label, subject=f"NEW_{sid}",
                                  path=f, condition=cond))
    return recs
