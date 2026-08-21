"""Do the kinetic tasks really separate ET? Auditing the project's lever #3.

The skill file lists, as the third-ranked route to better ET performance:

    "PADS's 8 unextracted tasks -- the kinetic ones (DrinkGlas, TouchNose) are
     where ET separates best (NewData DRINK AUC 0.812 vs 0.20-0.27 at REST)."

That recommendation has driven real decisions, including an attempt this session
to download the remaining PADS tasks. But the evidence for it is a single AUC on
**NewData, which has 6 ET patients**, and `permutation_null.md` later established
that the permutation null for PD-vs-ET AUC at 6 ET spans **[0.195, 0.819]**.

0.812 sits inside that interval.

So the claim may never have been distinguishable from chance, and it was recorded
before the machinery existed to check. This runs the check it needs: every
NewData task, both axes, with a permutation null that refits the whole pipeline
per replicate.

  PD vs ET      29 patients, 6 ET -- the axis the claim is about, and hopeless
  N vs Tremor   56 patients, 29 tremor -- better powered, included so the tasks
                can be compared on an axis where something is measurable

If no task clears its null on PD-vs-ET, the honest conclusion is not "kinetic
tasks do not help" but "**this cohort cannot tell**", and lever #3 must be
restated as a hypothesis rather than a measurement.

Run: ``python -m experiments.kinetic_task_audit``
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from experiments.tf_variability_screen import NPERM, oof, perm_p

KINETIC = {"DRINK", "FINGER_NOSE", "POUR", "TAP", "PRON_SUP"}


def main():
    from common.cohorts import desc_table
    from common.load_2025 import ALL_TASKS_2025, load_2025_all
    from frequency.tables import spectrum_table

    print(f"NewData, {len(ALL_TASKS_2025)} tasks, {NPERM} permutations per cell")
    print("kinetic tasks marked [K]\n", flush=True)

    rows = []
    for task in ALL_TASKS_2025:
        try:
            recs = load_2025_all(conditions=(task,))
        except Exception as e:
            print(f"  {task}: unavailable ({e})")
            continue
        if not recs:
            continue
        sp = spectrum_table(recs, ch=slice(3, 6))
        y3 = sp[1]
        D = np.nan_to_num(desc_table(recs, slice(3, 6)))
        rows.append((task, D, y3))
        print(f"  {task:>12}: {len(y3)} patients  "
              f"N/PD/ET {[int((y3 == c).sum()) for c in (0, 1, 2)]}")

    for axis in ("PD vs ET", "N vs Tremor"):
        print(f"\n{'='*84}")
        print(f"NewData  {axis}  (descriptors, 10 features)")
        print(f"{'='*84}")
        print(f"{'task':>14}{'':>4}{'n':>5}{'pos':>5}{'AUC':>9}"
              f"{'null 95%':>20}{'p':>8}   verdict")
        for task, D, y3 in rows:
            if axis == "PD vs ET":
                m = y3 != 0
                X, y = D[m], (y3[m] == 2).astype(int)
            else:
                X, y = D, (y3 != 0).astype(int)
            if y.sum() < 4 or (len(y) - y.sum()) < 4:
                print(f"{task:>14}{'':>4}{len(y):>5}{int(y.sum()):>5}"
                      "   too few in one class")
                continue
            k = 3
            try:
                obs = roc_auc_score(y, oof(X, y, k))
            except ValueError:
                continue
            lo, hi, pv = perm_p(X, y, k, obs)
            tag = "[K]" if task in KINETIC else "   "
            if pv < 0.05 and obs > (lo + hi) / 2:
                v = "separates"
            elif pv < 0.05:
                v = "separates INVERTED"
            else:
                v = "indistinguishable from chance"
            print(f"{task:>14}{tag:>4}{len(y):>5}{int(y.sum()):>5}{obs:>9.3f}"
                  f"{f'[{lo:.3f}, {hi:.3f}]':>20}{pv:>8.3f}   {v}", flush=True)

    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    main()
