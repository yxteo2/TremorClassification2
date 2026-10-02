"""2015 OUT model performance on all patients vs patients with measurable tremor.

`stack_2015.md`: 14 of `ft`'s 27 consistent errors are patients with no
measurable tremor in OUT, REST or WING (PD->N 11, ET->N 3). A tremor classifier
cannot see them, so a headline over all 151 patients mixes "the model is wrong"
with "there is no tremor to classify". This reports both, from the saved
out-of-fold predictions -- no refit.

**Tremor present** = log10 OUT tremor RMS (lower arm, 3-15 Hz band-pass, mean
over recordings) above the 90th percentile of the controls'. It is fixed in
advance, uses only the controls' distribution, and is applied to every patient
whatever their label (so controls with tremor stay in the subset). It is a
**reporting stratification, not a model input**: the model was trained and
scored on all 151 patients; only the scoring is split.

Run: ``python -m experiments.tremor_present_2015`` (needs `segments_2015_runs/`).
"""

from __future__ import annotations

import glob

import numpy as np
from scipy.signal import butter, sosfiltfilt
from sklearn.metrics import precision_recall_fscore_support

ARMS = ("scratch", "ft")
NAMES = ("precN", "precPD", "precET", "macroP", "recPD", "recET")


def tremor_rms(ids):
    from common.quaternion_data import load_quaternion_recordings
    sos = butter(4, [3 / 50, 15 / 50], btype="band", output="sos")
    d = {}
    for r in load_quaternion_recordings("Data", action="OUT", mode="angular_velocity"):
        x = np.asarray(r.x[3:6], float)
        d.setdefault(r.subject, []).append(
            np.sqrt(np.mean(sosfiltfilt(sos, x, axis=-1) ** 2)))
    return np.log10([np.mean(d[p]) for p in ids])


def main():
    from common.quaternion_data import load_quaternion_recordings
    from frequency.tables import spectrum_table
    files = sorted(glob.glob("segments_2015_runs/rep*.npz"))
    assert files, "run experiments.segments_2015 first"
    recs = load_quaternion_recordings("Data", action="OUT", mode="angular_velocity")
    ids = spectrum_table(recs, ch=slice(3, 6))[2]
    D = [np.load(f) for f in files]
    y = D[0]["y"]
    R = tremor_rms(ids)
    thr = np.percentile(R[y == 0], 90)
    pres = R > thr
    print(f"tremor present: log10 RMS > {thr:.2f} (controls' 90th percentile)")
    for c, nm in enumerate(("N", "PD", "ET")):
        print(f"  {nm:>2}: {int(pres[y == c].sum())} of {int((y == c).sum())} have tremor")
    for subset, m in (("all 151 patients", np.ones(len(y), bool)),
                      (f"tremor present ({int(pres.sum())})", pres),
                      (f"tremor absent ({int((~pres).sum())})", ~pres)):
        print(f"\n{subset}")
        print(f"{'arm':>10}" + "".join(f"{c:>9}" for c in NAMES)
              + "   ET prevalence")
        for k in ARMS:
            rows = []
            for d in D:
                P, Rc, _, _ = precision_recall_fscore_support(
                    y[m], d[k][m], labels=[0, 1, 2], zero_division=0)
                rows.append([P[0], P[1], P[2], P.mean(), Rc[1], Rc[2]])
            v = np.mean(rows, 0)
            print(f"{k:>10}" + "".join(f"{x:>9.3f}" for x in v)
                  + f"   {(y[m] == 2).mean():.3f}")
    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    main()
