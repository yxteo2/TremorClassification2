"""Route contested patients to a different model. The gate is the hypothesis.

Two measurements set this up, and neither is a hope:

1. `ensemble_diversity.md` — the six members agree on 59.5 % of patients and are
   68.8 % correct there; on the contested 40.5 % they fall to 0.443 balanced
   accuracy with a top-2 margin four times narrower.
2. `contested_specialists.md` — those contested patients are **not** a
   random-label region. Six of eight feature blocks clear chance there
   significantly with a plain logistic regression, best DESC at +0.187
   [+0.121, +0.253].

So the model is confident and good on 60 % of patients, uncertain and weak on
40 %, and there is structure left in the 40 % that a linear model on hand-built
descriptors can see. The architecture that follows is a **gate**: predict with
the deep ensemble where its members agree, and with something else where they do
not.

**The gate is legitimate, not an oracle.** Unanimity is computed from the six
members' own outputs — no labels, no test-set statistics — so it is available at
prediction time for a single unseen patient. This is the property that makes the
idea deployable rather than a diagnostic.

## The confound this experiment exists to control

`contested_specialists.md` states the trap explicitly: the contested subset is
*defined* by the deep ensemble's uncertainty, so any second model looks good
there by construction. That report was therefore careful to claim only that the
LR clears **chance**, never that it beats the deep model.

The same trap applies to the gate itself, so the decisive arm here is not
"gated beats baseline" — it is **gated vs uniform fusion**:

  baseline          the reported model
  LR everywhere     the descriptor logistic regression alone, for scale
  uniform fusion    deep and LR mixed on EVERY patient, gate ignored
  gated hard        deep where unanimous, LR where contested
  gated fusion      deep where unanimous, deep+LR mixed where contested

If **uniform fusion** matches **gated fusion**, the gate is decorative and any
gain is ordinary score fusion — which `score_vs_feature_fusion.md` has already
characterised. The gate only earns its place by beating the same fusion applied
without it.

Mixing weight and class priors are both chosen on the untouched validation split,
exactly as the reported model chooses its priors. Test is never touched.

20 splits, paired. Run: ``python -m experiments.contested_gating``
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common.protocol import TEST_FRAC, VAL_FRAC, tune_offsets
from experiments.alltasks_final import paired
from experiments.final_model import build
from experiments.pooling_rules import fit_members

NM = ("precN", "precPD", "precET", "macroP", "macroF1")
SPLITS = 20
WGRID = np.linspace(0.0, 1.0, 11)


def _norm(P):
    return P / np.clip(P.sum(1, keepdims=True), 1e-12, None)


def _mix(a, b, w):
    return _norm(np.exp((1 - w) * np.log(a + 1e-12) + w * np.log(b + 1e-12)))


def score(pt, off, yte):
    pred = (np.log(pt + 1e-12) + off).argmax(1)
    P, _, F, _ = precision_recall_fscore_support(yte, pred, labels=[0, 1, 2],
                                                 zero_division=0)
    return [P[0], P[1], P[2], P.mean(), F.mean()]


def _val_f1(pv, off, yva):
    _, _, F, _ = precision_recall_fscore_support(
        yva, (np.log(pv + 1e-12) + off).argmax(1), labels=[0, 1, 2],
        zero_division=0)
    return F.mean()


def pick_w(fn_v, yva):
    """Weight maximising validation macro F1, priors re-tuned at each w."""
    best = (-1.0, 0.0)
    for w in WGRID:
        pv = fn_v(w)
        f = _val_f1(pv, tune_offsets(pv, yva), yva)
        if f > best[0]:
            best = (f, w)
    return best[1]


def main():
    torch.set_num_threads(1)
    d = build()
    y, key = d["y"], d["key"]
    A = np.hstack([d["ASYM"], d["HAVE"]])
    desc, traj, spec = d["DESC"], d["TRAJ"], d["SPEC"]["multitaper"]
    Dfull = np.hstack([desc, A])

    print(f"n={len(y)}  N={int((y==0).sum())} PD={int((y==1).sum())} "
          f"ET={int((y==2).sum())}   {SPLITS} splits")
    print("gate = the six deep members do not agree; computed from their")
    print("outputs alone, so it is available at prediction time\n", flush=True)

    ARMS = ("baseline (deep)", "LR everywhere", "uniform fusion",
            "gated hard", "gated fusion")
    res = {a: [] for a in ARMS}
    gate_v, gate_t, ws = [], [], []

    for sp in range(SPLITS):
        tv, te = next(StratifiedShuffleSplit(1, test_size=TEST_FRAC,
                                             random_state=sp).split(spec, key))
        t0, v0 = next(StratifiedShuffleSplit(1, test_size=VAL_FRAC,
                                             random_state=sp).split(spec[tv],
                                                                    key[tv]))
        tr, va = tv[t0], tv[v0]

        V, T = fit_members(spec, Dfull, traj, y, tr, va, te)
        dv, dt = _norm(V.mean(0)), _norm(T.mean(0))
        gv = ~(np.stack([V[i].argmax(1) for i in range(len(V))])
               == V[0].argmax(1)).all(0)
        gt = ~(np.stack([T[i].argmax(1) for i in range(len(T))])
               == T[0].argmax(1)).all(0)
        gate_v.append(float(gv.mean()))
        gate_t.append(float(gt.mean()))

        m = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=5000,
                                             class_weight="balanced"))
        m.fit(desc[tr], y[tr])
        lv, lt = m.predict_proba(desc[va]), m.predict_proba(desc[te])

        res["baseline (deep)"].append(score(dt, tune_offsets(dv, y[va]), y[te]))
        res["LR everywhere"].append(score(lt, tune_offsets(lv, y[va]), y[te]))

        w_u = pick_w(lambda w: _mix(dv, lv, w), y[va])
        pv = _mix(dv, lv, w_u)
        res["uniform fusion"].append(
            score(_mix(dt, lt, w_u), tune_offsets(pv, y[va]), y[te]))

        hv, ht = dv.copy(), dt.copy()
        hv[gv], ht[gt] = lv[gv], lt[gt]
        res["gated hard"].append(score(ht, tune_offsets(hv, y[va]), y[te]))

        def gfv(w):
            p = dv.copy()
            p[gv] = _mix(dv[gv], lv[gv], w)
            return p
        w_g = pick_w(gfv, y[va])
        gt_p = dt.copy()
        gt_p[gt] = _mix(dt[gt], lt[gt], w_g)
        res["gated fusion"].append(
            score(gt_p, tune_offsets(gfv(w_g), y[va]), y[te]))
        ws.append((w_u, w_g))

        print(f"  split {sp+1}/{SPLITS}  contested val {gv.mean():.3f} "
              f"test {gt.mean():.3f}   w uniform {w_u:.1f} gated {w_g:.1f}",
              flush=True)

    for a in res:
        res[a] = np.array(res[a])

    print(f"\n{'arm':>20}" + "".join(f"{c:>9}" for c in NM) + "   sd(macroP)")
    for a in ARMS:
        print(f"{a:>20}" + "".join(f"{v:>9.3f}" for v in res[a].mean(0))
              + f"{res[a][:, 3].std():>12.3f}")

    base = res["baseline (deep)"]
    print("\npaired vs the reported deep model:")
    for a in ARMS[1:]:
        print(f"  {a}:")
        for (dd, lo, hi), c in zip(paired(res[a], base), NM):
            star = "*" if lo > 0 or hi < 0 else " "
            print(f"    {c:>8} {dd:+.3f}  [{lo:+.3f}, {hi:+.3f}] {star}")

    print("\nTHE ARM THAT DECIDES IT -- gated fusion vs uniform fusion.")
    print("If this is null, the gate is decorative and the effect is fusion:")
    for (dd, lo, hi), c in zip(paired(res["gated fusion"],
                                      res["uniform fusion"]), NM):
        star = "*" if lo > 0 or hi < 0 else " "
        print(f"    {c:>8} {dd:+.3f}  [{lo:+.3f}, {hi:+.3f}] {star}")

    print("\nsplit-level win rate vs the deep baseline:")
    for a in ARMS[1:]:
        print(f"  {a:>20}: " + "  ".join(
            f"{c} {float((res[a][:, i] > base[:, i]).mean()):.2f}"
            for i, c in enumerate(NM)))

    w = np.array(ws)
    print(f"\ncontested rate: val {np.mean(gate_v):.3f}, "
          f"test {np.mean(gate_t):.3f}")
    print(f"mixing weight (0 = deep only, 1 = LR only): "
          f"uniform {w[:,0].mean():.2f}, gated {w[:,1].mean():.2f}")
    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    main()
