"""Fine-tuning MOMENT-1-small for 2015 OUT (LP-FT), vs frozen MOMENT and `ft`.

`pretrained_moment` / `moment_ensemble`: frozen MOMENT embeddings + a logistic
head robustly improve PD-vs-ET ranking (+0.06 AUC over `ft`) and N / PD
precision, but not ET precision. This fine-tunes the encoder itself.

Method, fixed before any run (no tuning on test folds):
  LP-FT (linear probe, then fine-tune; Kumar et al., ICLR 2022) -- on small
  data it distorts pretrained features less than fine-tuning with a random head.
  * LP: linear head on frozen window embeddings, 300 full-batch Adam steps,
    lr 1e-2, wd 1e-3.
  * FT: all weights, AdamW (encoder 1e-5, head 1e-4, wd 1e-2), up to 15 epochs,
    batches of 32 windows; early stopping on the validation patients'
    class-weighted CE, the LP state counting as epoch 0 (so if fine-tuning never
    helps validation, the arm falls back to the linear probe).
  * Every loss weights each patient equally (1 / its window count) and each
    class by inverse frequency. Patient probability = mean of its windows'.
  * Input: band-passed 3-15 Hz lower-arm windows (512 samples, hop 256), as in
    `pretrained_moment`. One seed per fold.

Arms, on 20 FRESH partitions (seeds 200-219, never used before) with the same
folds, validation splits, PADS draws and validation-tuned offsets as
`transfer_2015`:

``ft``            adopted model (PADS pretrain -> fine-tune)
``mom_lr``        frozen MOMENT, patient-mean embedding + logistic (as before)
``mom_ft``        MOMENT fine-tuned with LP-FT
``mom_ft_perm``   control: identical LP-FT from weight-PERMUTED MOMENT (seed 0)
``ens_lr``        ft + mom_lr soft vote        ``ens_ft``   ft + mom_ft soft vote

PREDICTIONS (before the run):
  * mom_ft vs mom_lr: PD-vs-ET AUC within +-0.03 and precET within +-0.04 --
    ~90 training patients (~9 ET) per fold cannot usefully move a ~35 M
    parameter encoder; LP-FT keeps it near the probe. Fine-tuning is null.
  * mom_ft beats mom_ft_perm by > +0.05 PD-vs-ET AUC: the pretrained start
    still matters when every weight trains.
  * ens_ft vs ens_lr: |d AUC| < 0.02.

Stages (the ``train`` stage runs in the ``moment`` conda env, on the GPU):
    python -m experiments.moment_finetune folds                 (project env)
    <moment env> -m experiments.moment_finetune train           (ARM=mom_ft|mom_ft_perm, REPS=a-b)
    TREMOR_DEVICE=cuda python -m experiments.moment_finetune eval   (project env; REPS=a-b)
    python -m experiments.moment_finetune report
Env: WORK (folder with windows.npz from `pretrained_moment`), MOMENT_DIR.
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np

WORK = os.environ.get("WORK", "moment_runs")
MODEL_DIR = os.environ.get("MOMENT_DIR", "")
FT_DIR = f"{WORK}/finetune"
REP_RANGE = tuple(map(int, os.environ.get("REPS", "200-220").split("-")))
CAP = 90


def _patients():
    """Patient keys (sorted = pipeline order), labels, and window -> patient index."""
    w = np.load(f"{WORK}/windows.npz")
    subj, lab, rec_of = w["subj"], w["lab"], w["rec_of"]
    keys = np.array(sorted(set(subj)))
    pidx = {k: i for i, k in enumerate(keys)}
    ylab = {s: l for s, l in zip(subj, lab)}
    y = np.array([ylab[k] for k in keys])
    win_pat = np.array([pidx[subj[r]] for r in rec_of])
    return w, keys, y, win_pat


# --------------------------------------------------------------------------- folds
def folds():
    from sklearn.model_selection import StratifiedKFold, train_test_split
    _, keys, y, _ = _patients()
    os.makedirs(FT_DIR, exist_ok=True)
    out = {}
    for rep in range(*REP_RANGE):
        for fold, (rest, te) in enumerate(
                StratifiedKFold(5, shuffle=True, random_state=rep).split(np.zeros(len(y)), y)):
            tr, va = train_test_split(rest, test_size=0.25, stratify=y[rest],
                                      random_state=rep * 10 + fold)
            out[f"{rep}_{fold}_tr"], out[f"{rep}_{fold}_va"] = np.sort(tr), np.sort(va)
            out[f"{rep}_{fold}_te"] = te
    np.savez(f"{FT_DIR}/folds.npz", y=y, keys=keys, **out)
    print(f"folds for reps {REP_RANGE[0]}-{REP_RANGE[1] - 1} -> {FT_DIR}/folds.npz")


# --------------------------------------------------------------------------- train (moment env)
def train():
    import torch
    import torch.nn as nn
    from momentfm import MOMENTPipeline
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arm = os.environ.get("ARM", "mom_ft")
    w, keys, y, win_pat = _patients()
    F = np.load(f"{FT_DIR}/folds.npz")
    assert (F["keys"] == keys).all()
    X = torch.tensor(w["bp"])                                     # (n_win, 3, 512)
    nwin = np.bincount(win_pat, minlength=len(y)).astype(float)

    def load_model():
        # momentfm FREEZES the encoder and patch embedder by default
        # (freeze_encoder / freeze_embedder = True): without these flags
        # "fine-tuning" silently trains only the head.
        m = MOMENTPipeline.from_pretrained(
            MODEL_DIR or "AutonLab/MOMENT-1-small",
            model_kwargs={"task_name": "embedding", "freeze_encoder": False,
                          "freeze_embedder": False, "enable_gradient_checkpointing": False})
        m.init()
        if arm == "mom_ft_perm":
            g = torch.Generator().manual_seed(0)
            with torch.no_grad():
                for p in m.parameters():
                    flat = p.data.flatten()
                    p.data.copy_(flat[torch.randperm(flat.numel(), generator=g)].view_as(p))
        return m.to(dev)

    base = load_model()
    base_state = {k: v.detach().clone() for k, v in base.state_dict().items()}

    def embed(m, idx, grad=False, bs=128):
        outs = []
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            for i in range(0, len(idx), bs):
                xb = X[idx[i:i + bs]].to(dev)
                outs.append(m.embed(x_enc=xb, reduction="mean").embeddings)
        return torch.cat(outs)

    base.eval()
    E_all = embed(base, np.arange(len(X)))                        # frozen embeddings, all windows

    def weights(win_idx, pats):
        cls = np.bincount(y[pats], minlength=3).astype(float)
        cw = cls.sum() / (3 * np.maximum(cls, 1))
        wt = cw[y[win_pat[win_idx]]] / nwin[win_pat[win_idx]]
        return torch.tensor(wt / wt.mean(), dtype=torch.float32, device=dev)

    def patient_probs(logits, win_idx, pats):
        pr = torch.softmax(logits, 1).detach().cpu().numpy()
        out = np.zeros((len(pats), 3))
        pos = {p: i for i, p in enumerate(pats)}
        cnt = np.zeros(len(pats))
        for k, wi in enumerate(win_idx):
            j = pos[win_pat[wi]]; out[j] += pr[k]; cnt[j] += 1
        return out / cnt[:, None]

    def val_ce(P, pats):
        cls = np.bincount(y[pats], minlength=3).astype(float)
        cw = cls.sum() / (3 * np.maximum(cls, 1))
        wts = cw[y[pats]]
        return float(-(wts * np.log(P[np.arange(len(pats)), y[pats]] + 1e-12)).sum() / wts.sum())

    os.makedirs(FT_DIR, exist_ok=True)
    for rep in range(*REP_RANGE):
        res = {}
        for fold in range(5):
            tr, va, te = (F[f"{rep}_{fold}_{s}"] for s in ("tr", "va", "te"))
            wtr = np.flatnonzero(np.isin(win_pat, tr))
            wva = np.flatnonzero(np.isin(win_pat, va))
            wte = np.flatnonzero(np.isin(win_pat, te))
            torch.manual_seed(1000 * rep + fold)
            ytr = torch.tensor(y[win_pat[wtr]], device=dev)
            wt = weights(wtr, tr)
            # ---- LP: linear head on frozen embeddings
            head = nn.Linear(E_all.shape[1], 3).to(dev)
            opt = torch.optim.Adam(head.parameters(), lr=1e-2, weight_decay=1e-3)
            for _ in range(300):
                opt.zero_grad()
                loss = (nn.functional.cross_entropy(head(E_all[wtr]), ytr, reduction="none") * wt).mean()
                loss.backward(); opt.step()
            m = base
            m.load_state_dict(base_state)
            with torch.no_grad():
                Pva = patient_probs(head(E_all[wva]), wva, va)
                Pte = patient_probs(head(E_all[wte]), wte, te)
            best = (val_ce(Pva, va), Pva, Pte, 0)
            # ---- FT: all weights, small lr, early stopping on validation patients
            opt = torch.optim.AdamW([{"params": m.parameters(), "lr": 1e-5},
                                     {"params": head.parameters(), "lr": 1e-4}], weight_decay=1e-2)
            rng = np.random.default_rng(1000 * rep + fold)
            for ep in range(1, 16):
                m.train(); head.train()
                order = rng.permutation(len(wtr))
                for i in range(0, len(order), 32):
                    b = order[i:i + 32]
                    opt.zero_grad()
                    z = m.embed(x_enc=X[wtr[b]].to(dev), reduction="mean").embeddings
                    loss = (nn.functional.cross_entropy(head(z), ytr[b], reduction="none") * wt[b]).mean()
                    loss.backward()
                    if ep == 1 and i == 0:
                        enc = [q for n_, q in m.named_parameters() if n_.startswith("encoder.")]
                        assert enc and all(q.requires_grad for q in enc), "encoder frozen"
                        assert sum(float(q.grad.abs().sum()) for q in enc if q.grad is not None) > 0,                             "encoder received no gradient"
                    opt.step()
                m.eval(); head.eval()
                with torch.no_grad():
                    Pva = patient_probs(head(embed(m, wva)), wva, va)
                    v = val_ce(Pva, va)
                    if v < best[0]:
                        Pte = patient_probs(head(embed(m, wte)), wte, te)
                        best = (v, Pva, Pte, ep)
            res[f"{fold}_pv"], res[f"{fold}_pt"] = best[1], best[2]
            res[f"{fold}_epoch"] = np.array(best[3])
            print(f"{arm} rep {rep} fold {fold}: best epoch {best[3]}  val CE {best[0]:.3f}", flush=True)
        np.savez(f"{FT_DIR}/{arm}_rep{rep:03d}.npz", **res)


# --------------------------------------------------------------------------- eval (project env)
def evaluate():
    import torch
    from sklearn.metrics import precision_recall_fscore_support
    from common.protocol import DEVICE, tune_offsets
    from experiments.moment_ensemble import _lr
    from experiments.own_data_10et import build
    from experiments.pretrained_moment import _patient_mean
    from experiments.transfer_2015 import fit
    torch.set_num_threads(1)
    A, _, C = build()
    spec, desc, traj, y = A
    own = {"packed": np.hstack([spec, desc, traj]), "spec": spec, "nd": desc.shape[1]}
    Cp = np.hstack([C[0], C[1], C[2]])
    w, keys, yk, _ = _patients()
    assert (yk == y).all(), "patient order/labels differ from build()"
    E, _ = _patient_mean(np.load(f"{WORK}/embeddings.npz")["bp"], w["rec_of"], w["subj"])
    F = np.load(f"{FT_DIR}/folds.npz")
    print(f"device={DEVICE}", flush=True)
    for rep in range(*REP_RANGE):
        rng = np.random.default_rng(rep)
        kp = np.sort(np.concatenate([rng.choice(np.flatnonzero(C[3] == c),
                                                min(CAP, int((C[3] == c).sum())),
                                                replace=False) for c in (0, 1, 2)]))
        pads = {"packed": Cp[kp], "spec": C[0][kp]}
        M = {a: np.load(f"{FT_DIR}/{a}_rep{rep:03d}.npz") for a in ("mom_ft", "mom_ft_perm")}
        arms = ("ft", "mom_lr", "mom_ft", "mom_ft_perm", "ens_lr", "ens_ft")
        prob = {k: np.zeros((len(y), 3)) for k in arms}
        pred = {k: np.full(len(y), -1) for k in arms}
        epochs = {a: [] for a in M}
        for fold in range(5):
            tr, va, te = (F[f"{rep}_{fold}_{s}"] for s in ("tr", "va", "te"))
            ft = fit("ft", own, pads, y, tr, va, te, C[3][kp])
            lr = _lr(E, y, tr, va, te)
            mf = (M["mom_ft"][f"{fold}_pv"], M["mom_ft"][f"{fold}_pt"])
            mp = (M["mom_ft_perm"][f"{fold}_pv"], M["mom_ft_perm"][f"{fold}_pt"])
            for a in M:
                epochs[a].append(int(M[a][f"{fold}_epoch"]))
            out = {"ft": ft, "mom_lr": lr, "mom_ft": mf, "mom_ft_perm": mp,
                   "ens_lr": ((ft[0] + lr[0]) / 2, (ft[1] + lr[1]) / 2),
                   "ens_ft": ((ft[0] + mf[0]) / 2, (ft[1] + mf[1]) / 2)}
            for k, (pv, pt) in out.items():
                prob[k][te] = pt
                pred[k][te] = (np.log(pt + 1e-12) + tune_offsets(pv, y[va])).argmax(1)
        np.savez(f"{FT_DIR}/eval_rep{rep:03d}.npz", y=y,
                 **{f"pred_{k}": v for k, v in pred.items()},
                 **{f"prob_{k}": v for k, v in prob.items()},
                 **{f"epochs_{a}": np.array(v) for a, v in epochs.items()})
        P = {k: precision_recall_fscore_support(y, pred[k], labels=[0, 1, 2],
                                                zero_division=0)[0] for k in arms}
        print(f"rep {rep}  " + "  ".join(f"{k} ET {P[k][2]:.2f}" for k in arms), flush=True)


def report():
    from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
    files = sorted(glob.glob(f"{FT_DIR}/eval_rep*.npz"))
    arms = ("ft", "mom_lr", "mom_ft", "mom_ft_perm", "ens_lr", "ens_ft")
    cols = ("precN", "precPD", "precET", "macroP", "macroF1", "PDvsET AUC")
    out = {k: [] for k in arms}
    ep = {"mom_ft": [], "mom_ft_perm": []}
    for f in files:
        d = np.load(f); y = d["y"]; m = y != 0
        for a in ep:
            ep[a].extend(d[f"epochs_{a}"].tolist())
        for k in arms:
            P, _, F1, _ = precision_recall_fscore_support(y, d[f"pred_{k}"], labels=[0, 1, 2],
                                                          zero_division=0)
            p = d[f"prob_{k}"]
            a = roc_auc_score(y[m] == 2, np.log(p[m, 2] + 1e-12) - np.log(p[m, 1] + 1e-12))
            out[k].append([P[0], P[1], P[2], P.mean(), F1.mean(), a])
    n = len(files)
    print(f"2015 OUT, {n} fresh CV repeats, all 151 patients, ET prevalence 0.099\n")
    for a, v in ep.items():
        v = np.array(v)
        print(f"{a}: early-stopping epoch -- LP kept (0) in {np.mean(v == 0):.0%} of folds, "
              f"median {np.median(v):.0f}, max {v.max()}")
    print(f"\n{'arm':>12}" + "".join(f"{c:>11}" for c in cols))
    for k in arms:
        out[k] = np.array(out[k])
        print(f"{k:>12}" + "".join(f"{v:>11.3f}" for v in out[k].mean(0)))
    pairs = (("mom_lr", "mom_ft"), ("mom_ft_perm", "mom_ft"), ("ft", "mom_ft"),
             ("ens_lr", "ens_ft"), ("ft", "ens_ft"))
    for base, k in pairs:
        print(f"\n{k} - {base}  (paired, {n} repeats)")
        for i, c in enumerate(cols):
            dlt = out[k][:, i] - out[base][:, i]
            b = [np.random.default_rng(s).choice(dlt, len(dlt)).mean() for s in range(4000)]
            lo, hi = np.percentile(b, [2.5, 97.5])
            print(f"  {c:>10} {dlt.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] "
                  f"{'*' if lo > 0 or hi < 0 else ' '}  win {np.mean(dlt > 0):.2f}")
    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    {"folds": folds, "train": train, "eval": evaluate, "report": report}[sys.argv[1]]()
