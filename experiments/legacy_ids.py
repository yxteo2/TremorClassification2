"""Row order of runs saved before the 2015 subject-id fix (151 rows).

`common.quaternion_data.canonical_subject` (2026-10-05) merged "PD 12_OUT N
LOAD" into PD 12 and upper-cased "PD 1_out" / "PD 23_out": 2015 OUT has 150
patients. Every run saved before that has 151 rows in the old sorted-id order,
so analyses that join features onto those saved predictions must rebuild the
old ids. `recordings_for(n_rows, action)` does that: for 151-row runs it
re-applies the pre-fix rule (strip a trailing trial number only); otherwise it
returns the canonical recordings unchanged.
"""

from __future__ import annotations

import os
import re

_OLD = re.compile(r"[_\s]+\d+$")


def recordings_for(n_rows, action="OUT"):
    from common.quaternion_data import load_quaternion_recordings
    recs = load_quaternion_recordings("Data", action=action, mode="angular_velocity")
    if n_rows == 151 and action == "OUT":
        for r in recs:
            r.subject = _OLD.sub("", os.path.splitext(os.path.basename(str(r.path)))[0])
    return recs


def n_rows(directory):
    """Row count of the runs saved in `directory` (151 = pre-fix)."""
    import glob
    import numpy as np
    files = sorted(glob.glob(f"{directory}/rep*.npz"))
    return len(np.load(files[0])["y"]) if files else 150
