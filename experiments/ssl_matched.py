"""The SSL encoder was pretrained on different frequency bins than it was used on.

``experiments/masked_pretrain.py`` builds its unlabelled corpus with **welch**,
masks the 3-15 Hz band to 61 columns, and calls ``logbin(nb=16)`` -- which keeps
61 // 16 * 16 = 48 of them. Bin *j* of a pretraining input therefore spans

    3.13 + 0.586 j  ...  3.52 + 0.586 j  Hz

The labelled tables it is then applied to come from ``method_table(recs,
"multitaper", ch)``, which interpolates onto ``GRID`` -- 64 columns, so
``logbin(nb=16)`` is exact and bin *j* spans

    3.00 + 0.762 j  ...  3.57 + 0.762 j  Hz

Same vector length, different frequencies. By bin 15 the pretraining input means
11.9-12.3 Hz and the downstream input means 14.4-15.0 Hz. **The encoder was
pretrained on one axis and evaluated on another**, and it still produced the
largest deep-learning gain measured in this project (PADS PD-vs-ET precET
+0.161). That gain is therefore a lower bound, and this measures what the
matched pipeline is worth.

`experiments/binning.py` already shows the band coverage alone is worth precET
+0.032 [+0.014, +0.050] on PADS PD-vs-ET for a logistic regression, so the two
defects are not the same defect: one is *which frequencies are represented*, the
other is *whether pretrain and downstream agree about it*.

Arms, all with the encoder **frozen** and a linear head, which is the
configuration that won:

  1. random init, frozen        -- the honest control. The published comparison
                                   used a random init that was *fine-tuned*, so
                                   it confounded pretraining with freezing;
                                   this does not.
  2. SSL, mismatched corpus     -- welch-61 -> logbin, reproduces the current
                                   result
  3. SSL, matched corpus        -- multitaper -> GRID -> logbin, byte-identical
                                   construction to the labelled features
  4. SSL, matched, cohort held out -- the evaluated cohort contributes nothing to
                                   pretraining, so the encoder has never seen any
                                   recording of any patient it is tested on

Run: ``python -m experiments.ssl_matched``
"""

from __future__ import annotations

import os

import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from common.cohorts import logbin
from experiments.masked_pretrain import finetune, pretrain
from experiments.pd_vs_et import build as build_labelled

REPEATS, COLS = 10, ("AUC", "precPD", "precET", "macroP", "ETsens")


def _recordings():
    """Every unlabelled recording, tagged by cohort group."""
    from common.load_2025 import ALL_TASKS_2025, load_2025_all
    from common.loaders import load_pads_extracted
    from common.quaternion_data import load_quaternion_recordings

    out = []
    for action in ("OUT", "REST", "WING"):
        try:
            out.append(("in-house", load_quaternion_recordings(
                "Data", action=action, mode="angular_velocity"), slice(3, 6)))
        except Exception:
            pass
    for task in ALL_TASKS_2025:
        try:
            r = load_2025_all(conditions=(task,))
            if r:
                out.append(("in-house", r, slice(3, 6)))
        except Exception:
            pass
    for folder in ("pads_stretchhold", "pads_relaxed"):
        if os.path.isdir(folder):
            try:
                out.append(("PADS", load_pads_extracted(folder), slice(0, 3)))
            except Exception:
                pass
    return out


def corpus(matched):
    """(X, cohort) for every unlabelled recording.

    ``matched=True`` reproduces the labelled pipeline exactly: multitaper,
    interpolated onto ``GRID``, sum-normalised, then ``logbin``. ``matched=False``
    reproduces the current one: welch, band-masked to 61 columns, then ``logbin``
    -- which truncates at 12.3 Hz.
    """
    from scipy.signal import welch

    from experiments.final_model import GRID, METHODS

    mt = METHODS["multitaper"]
    rows, coh = [], []
    for tag, recs, ch in _recordings():
        for r in recs:
            x = r.x[ch] if r.x.shape[0] > 3 else r.x
            if matched:
                f, P = mt(x)
                f, P = np.asarray(f, float), np.asarray(P, float)
                m = np.isfinite(P)
                v = np.clip(np.interp(GRID, f[m], P[m], left=0.0, right=0.0),
                            0, None)
            else:
                f, P = welch(np.atleast_2d(x), fs=100.0,
                             nperseg=min(512, x.shape[-1]), axis=-1)
                v = P.mean(0)[(f >= 3.0) & (f <= 15.0)]
            if not np.isfinite(v).all() or v.sum() <= 0:
                continue
            rows.append(v / v.sum())
            coh.append(tag)
    X = np.nan_to_num(logbin(np.array(rows)).astype(np.float32))
    return X, np.array(coh)


def scores(y, p):
    pr = (p >= np.quantile(p, 1 - y.mean())).astype(int)
    se = recall_score(y, pr, pos_label=1, zero_division=0)
    pPD = precision_score(y, pr, pos_label=0, zero_division=0)
    pET = precision_score(y, pr, pos_label=1, zero_division=0)
    return [roc_auc_score(y, p), pPD, pET, 0.5 * (pPD + pET), se]


def paired(a, b, n=4000):
    d = a - b
    return [(d[:, i].mean(),
             *np.percentile([np.mean(np.random.default_rng(s).choice(
                 d[:, i], len(d), replace=True)) for s in range(n)],
                 [2.5, 97.5]))
            for i in range(len(COLS))]


def main():
    torch.set_num_threads(1)

    print("building corpora ...", flush=True)
    Xmis, _ = corpus(matched=False)
    Xmat, coh = corpus(matched=True)
    print(f"  mismatched (welch-61 -> logbin) : {Xmis.shape}")
    print(f"  matched (multitaper GRID -> logbin): {Xmat.shape}   "
          f"PADS {int((coh=='PADS').sum())}  in-house "
          f"{int((coh=='in-house').sum())}\n", flush=True)

    print("pretraining ...", flush=True)
    enc_mis, nrm_mis = pretrain(Xmis)
    st_mis = {k: v.clone() for k, v in enc_mis.state_dict().items()}
    enc_mat, nrm_mat = pretrain(Xmat)
    st_mat = {k: v.clone() for k, v in enc_mat.state_dict().items()}
    held = {}
    for g in ("PADS", "in-house"):
        e, n_ = pretrain(Xmat[coh != g])
        held[g] = ({k: v.clone() for k, v in e.state_dict().items()}, n_)
        print(f"  held-out-{g} encoder pretrained on "
              f"{int((coh!=g).sum())} recordings", flush=True)
    print()

    data = build_labelled()
    for name, tags, k in (("PADS", ["PADS"], 5),
                          ("MERGED", ["2015", "NewData", "PADS"], 5),
                          ("in-house", ["2015", "NewData"], 3)):
        spec = np.vstack([data[t][0]["spectrum"] for t in tags])
        y3 = np.concatenate([data[t][1] for t in tags])
        m = y3 != 0
        y = (y3[m] == 2).astype(int)
        X = np.nan_to_num(spec[m])

        arms = [("random init, frozen", None, None),
                ("SSL mismatched corpus", st_mis, nrm_mis),
                ("SSL matched corpus", st_mat, nrm_mat)]
        if name in held:
            arms.append((f"SSL matched, {name} held out", *held[name]))

        print(f"\n{'='*84}")
        print(f"{name}  PD vs ET  n={len(y)}  ET={int(y.sum())}  "
              f"prevalence {y.mean():.3f}   frozen encoder + linear head")
        print(f"{'='*84}")
        print(f"{'arm':>30}" + "".join(f"{c:>9}" for c in COLS))

        res = {}
        for lab, st, nrm in arms:
            rows = []
            for rep in range(REPEATS):
                p = np.zeros(len(y))
                for tr, te in StratifiedKFold(k, shuffle=True,
                                              random_state=rep).split(X, y):
                    nm = nrm if nrm is not None else (
                        X[tr].mean(0, keepdims=True),
                        X[tr].std(0, keepdims=True) + 1e-8)
                    p[te] = np.mean([finetune(st, nm, X[tr], y[tr], X[te],
                                              seed=s, freeze=True)
                                     for s in (0, 1)], 0)
                rows.append(scores(y, p))
            res[lab] = np.array(rows)
            print(f"{lab:>30}" + "".join(f"{v:>9.3f}"
                                        for v in res[lab].mean(0)), flush=True)

        base = res["random init, frozen"]
        print("\n  paired vs random init, frozen:")
        for lab, _, _ in arms[1:]:
            for (d, lo, hi), c in zip(paired(res[lab], base), COLS):
                if c in ("AUC", "precET", "macroP"):
                    star = "*" if lo > 0 or hi < 0 else " "
                    print(f"    {lab:>30} {c:>7} {d:+.3f} "
                          f"[{lo:+.3f}, {hi:+.3f}] {star}")
        print("\n  paired vs the mismatched corpus (what the fix is worth):")
        for lab, _, _ in arms[2:]:
            for (d, lo, hi), c in zip(
                    paired(res[lab], res["SSL mismatched corpus"]), COLS):
                if c in ("AUC", "precET", "macroP"):
                    star = "*" if lo > 0 or hi < 0 else " "
                    print(f"    {lab:>30} {c:>7} {d:+.3f} "
                          f"[{lo:+.3f}, {hi:+.3f}] {star}")
    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    main()
