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

def load_2025(root="NewData", cls="ET", label=2, conditions=("OUT",), fs=FS_DST):
    recs = []
    for subj_dir in sorted(glob.glob(os.path.join(root, cls, "*/"))):
        sid = re.search(r"_(%s_\d+)_" % cls, subj_dir)
        sid = sid.group(1) if sid else os.path.basename(subj_dir.rstrip("/"))
        for f in sorted(glob.glob(os.path.join(subj_dir, "rawData", "*.h5"))):
            code = os.path.basename(f)[:2]
            cond = ACTION.get(code)
            if cond is None or cond not in conditions:
                continue
            q = _load_h5(f)
            if q is None or q.shape[0] < 256:
                continue
            q = resample_poly(q, 25, 32, axis=0)      # 128 -> 100 Hz
            try:
                x = process_quaternion_data(q.astype(np.float32), fs=fs,
                                            mode="angular_velocity",
                                            convention="xyzw", n_sensors=3)
            except Exception:
                continue
            recs.append(Recording(x=x, y=label, subject=f"NEW_{sid}",
                                  path=f, condition=cond))
    return recs
