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
ACTION = {"01": "REST", "08": "REST", "02": "OUT", "09": "OUT"}
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


def load_2025(root="NewData", cls="ET", label=2, conditions=("OUT",), fs=FS_DST,
              mode="angular_velocity", sides=("right", "left"), segment=True,
              win_s=10.0):
    """Load the 2025 cohort, aligned to the 2015 channel order and rate.

    ``mode`` is passed to :func:`tremor.quaternion.process_quaternion_data`, so
    the same log_map / gravity representations used on the 2015 data are
    available here.

    ``segment`` defaults to **True**. These exports are ~38 s ``Free_Form``
    captures with an empty Annotations table, and using the whole recording
    leaves only 9.9 % of power in the 3-15 Hz tremor band (vs 76.5 % for the
    2015 cohort and 81.2 % for PADS) because set-up and settling motion
    dominate. Pass ``segment=False`` only to reproduce the superseded numbers.

    ``sides`` selects limbs. Both limbs of one subject share a subject id, so
    they can never be split across CV folds -- but note that pooling both
    doubles a subject's recordings without adding a subject, which matters for
    anything that averages per patient.
    """
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
            if segment:
                x = select_task_epoch(x, fs=fs, win_s=win_s)
            recs.append(Recording(x=x, y=label, subject=f"NEW_{sid}",
                                  path=f, condition=cond))
    return recs
