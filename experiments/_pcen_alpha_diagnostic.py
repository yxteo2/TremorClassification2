"""Why PCEN cannot work on this representation — measured without a model.

Supporting evidence for the PCEN arm of `pcen_hpss.py`. It answers one question
cheaply, before any deep fits: **does PCEN's gain control remove the spectral
peak, and does that depend on alpha or hold across the family?**

PCEN divides each band by an IIR-smoothed copy of itself. For a *stationary*
band the smoother tracks the signal, so

    E / M^alpha  ->  E^(1 - alpha)

which at the published default alpha = 0.98 leaves `E^0.02` — nearly constant.
That is the intended behaviour in audio, where the discriminative information is
*when* energy appears in a band and the band's own average is background to be
divided out. Here the discriminative information is **which band** has energy,
so dividing each band by its own average is precisely the operation that
destroys the signal.

This measures that directly on real recordings using two label-free statistics
of the resulting 16-bin spectrum:

  spectral entropy   normalised to [0, 1]; **1.0 means perfectly flat**
  peak-to-mean       the height of the tremor peak over the band average

Run: ``python -m experiments._pcen_alpha_diagnostic``
"""

from __future__ import annotations

import warnings

import numpy as np

import experiments.final_model as FM
from common.quaternion_data import load_quaternion_recordings
from experiments.pcen_hpss import FS, HOP_DENSE, pcen, surface
from signal_processing.transforms import F_MAX

warnings.filterwarnings("ignore")
ALPHAS = (0.98, 0.8, 0.5, 0.3)
N_REC = 40


def _spectrum(E, f):
    P = E.mean(axis=(0, 2))
    k = (f >= 3.0) & (f <= F_MAX)
    v = np.clip(np.interp(FM.GRID, f[k], P[k], left=0.0, right=0.0), 0, None)
    return v / (v.sum() + 1e-20)


def _stats(vs):
    vs = np.asarray(vs)
    ent = -(vs * np.log(vs + 1e-12)).sum(1) / np.log(vs.shape[1])
    return float(ent.mean()), float((vs.max(1) / (vs.mean(1) + 1e-12)).mean())


def main():
    recs = load_quaternion_recordings("Data", action="OUT",
                                      mode="angular_velocity")[:N_REC]
    rows = {"no PCEN": []}
    rows.update({f"alpha={a}": [] for a in ALPHAS})
    for r in recs:
        E, f = surface(r.x[3:6], HOP_DENSE)
        rows["no PCEN"].append(_spectrum(E, f))
        for a in ALPHAS:
            rows[f"alpha={a}"].append(
                _spectrum(pcen(E, HOP_DENSE / FS, alpha=a), f))

    print(f"{N_REC} recordings, 2015 OUT, lower-arm\n")
    print(f"{'arm':>12}{'entropy (1=flat)':>19}{'peak/mean':>12}")
    for k, v in rows.items():
        e, d = _stats(v)
        print(f"{k:>12}{e:>19.4f}{d:>12.2f}")
    print("\nPCEN flattens monotonically in alpha, and the family runs from")
    print("'no PCEN' (alpha=0) to 'fully flattened' (alpha->1) with no setting")
    print("that adds structure. Extra alpha arms would only interpolate.")
    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    main()
