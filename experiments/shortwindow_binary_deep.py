"""Resolving a confound I flagged: deep model, PD-vs-ET, short-window spectrum.

`tf_window_length.md` reports two results that point opposite ways:

    logistic regression, PD vs ET binary    PADS precET +0.088 [+0.073, +0.102] *
    reported deep model, 3-class merged     macroP -0.033 [-0.057, -0.007] *

and states plainly that they differ in **task** as well as in **model**, so
"the short-window spectrum helps linear models and hurts deep ones" is not
actually established — the deep arm was never run on the binary axis.

This runs it. Same binary PD-vs-ET framing as `pd_vs_et_deep.py`, tremor patients
only, folds shared across arms by construction (`run` seeds `StratifiedKFold` on
the repeat index), so every comparison is paired.

  logreg  multitaper 16      the linear baseline
  logreg  short-window 16    reproduces the reported linear gain
  CNN     multitaper 16      Spectrum1DCNN, 2 outputs
  CNN     short-window 16    the same, only the representation swapped

`Spectrum1DCNN` alone rather than the two-stream model, so the spectrum
representation is isolated: no descriptors, no trajectory, nothing else that
could absorb or mask the difference.

Three outcomes and what each would mean:

* **CNN gains like logreg** → the 3-class result was about the *task*, not the
  model, and the representation is simply better.
* **CNN flat while logreg gains** → the gain is specific to linear models, as the
  smoothing/variance-reduction reading suggests.
* **CNN loses like the 3-class run** → the representation genuinely suits linear
  models only, and the earlier conclusion stands on firmer ground.

Note this module's precision convention differs from the rest of the session:
`run` thresholds at 0.5, not at the prevalence quantile, so precET values are not
comparable with the 3-class tables. The paired differences are what matter here.

Run: ``python -m experiments.shortwindow_binary_deep``
"""

from __future__ import annotations

import os

import numpy as np
import torch

from experiments.pd_vs_et_deep import paired, run
from experiments.tf_window_control import WIN, logbin_n
from models.architectures import Spectrum1DCNN
from signal_processing.tf_variability import blocks, patient_table

REPEATS = 20
NBIN = 16


def main():
    torch.set_num_threads(1)
    from common.load_2025 import load_2025_all
    from common.loaders import load_pads_extracted
    from common.quaternion_data import load_quaternion_recordings
    from experiments.final_model import method_table
    from frequency.tables import spectrum_table

    cohorts = [("2015", load_quaternion_recordings("Data", action="OUT",
                                                   mode="angular_velocity"),
                slice(3, 6)),
               ("NewData", load_2025_all(conditions=("OUT",)), slice(3, 6))]
    if os.path.isdir("pads_stretchhold"):
        cohorts.append(("PADS", load_pads_extracted("pads_stretchhold"),
                        slice(0, 3)))

    print("building tables ...", flush=True)
    B = blocks()
    store = {}
    for tag, recs, ch in cohorts:
        sp = spectrum_table(recs, ch=ch)
        mt = logbin_n(method_table(recs, "multitaper", ch)[0], NBIN)
        X, _, p = patient_table(recs, ch=ch, nperseg=WIN, stat="mean")
        idx = {q: i for i, q in enumerate(p)}
        dim = X.shape[1] if len(X) else 34
        T = np.array([X[idx[q]] if q in idx else np.zeros(dim) for q in sp[2]])
        store[tag] = {"y": sp[1], "mt": mt, "sw": T[:, B["median"]]}
        print(f"  {tag:>8}: {len(sp[1])} patients")

    for gname, tags, k in (("PADS", ["PADS"], 5),
                           ("MERGED", ["2015", "NewData", "PADS"], 5)):
        tags = [t for t in tags if t in store]
        y3 = np.concatenate([store[t]["y"] for t in tags])
        m = y3 != 0
        y = (y3[m] == 2).astype(int)
        X = {q: np.nan_to_num(np.vstack([store[t][q] for t in tags]))[m]
             for q in ("mt", "sw")}

        print(f"\n{'='*86}")
        print(f"{gname}  PD vs ET  n={len(y)}  ET={int(y.sum())}   "
              f"{REPEATS} repeats, threshold 0.5")
        print(f"{'='*86}")
        print(f"{'model':>26}{'dim':>5}{'AUC':>16}{'precPD':>16}"
              f"{'precET':>16}{'bal-acc':>16}")

        mk = lambda: Spectrum1DCNN(NBIN, num_classes=2, ch=8)
        res = {}
        res["lr_mt"] = run("logreg  multitaper 16", X["mt"], y, None, k=k,
                           repeats=REPEATS, deep=False)
        res["lr_sw"] = run("logreg  short-window 16", X["sw"], y, None, k=k,
                           repeats=REPEATS, deep=False)
        res["cnn_mt"] = run("CNN     multitaper 16", X["mt"], y, mk, k=k,
                            repeats=REPEATS, deep=True)
        res["cnn_sw"] = run("CNN     short-window 16", X["sw"], y, mk, k=k,
                            repeats=REPEATS, deep=True)

        print("\npaired short-window - multitaper, same folds:")
        paired(res["lr_sw"], res["lr_mt"], "logistic regression")
        paired(res["cnn_sw"], res["cnn_mt"], "Spectrum1DCNN")
    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    main()
