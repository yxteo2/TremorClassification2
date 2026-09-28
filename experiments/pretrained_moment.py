"""Transfer from an online pretrained time-series foundation model: MOMENT-1-small.

MOMENT (AutonLab/MOMENT-1-small, MIT licence) is a T5-encoder foundation model
pretrained on a large, diverse corpus of time series. It embeds a (channels,
512) window by normalising each channel (RevIN: zero mean, unit variance),
patching it (8 samples) and averaging the encoder output over patches and
channels. So it sees waveform *shape*, not amplitude.

Scope: 2015 OUT, lower-arm angular velocity (the pipeline's input), one model per
action. This is the cheap, frozen-feature test first: if MOMENT embeddings do not
beat the current spectrum + descriptor features under the same linear classifier,
fine-tuning a 40 M-parameter encoder on 121 training patients is not justified.

Three stages; `embed` runs in the separate conda env ``moment`` (momentfm pins
numpy 1.25 / transformers 4.33), the others in the project env:

``export``  2015 OUT recordings -> 512-sample windows (hop 256, 5.12 s at 100 Hz),
            3 axes, band-passed 3-15 Hz (primary) and raw (secondary) -> npz
``embed``   MOMENT embeddings per window (reduction "mean") -> npz
``eval``    per-patient mean embedding; 40 CV partitions, L2 multinomial
            logistic regression, OOF on all 151 patients; arms:
              spec   16-bin multitaper + 10 descriptors (baseline)
              mom    MOMENT, band-passed        mom_raw  MOMENT, raw
              union  spec + mom                 u_shuf   spec + mom permuted
                                                         across patients
                                                         (invariant 11)

PREDICTIONS (before any stage runs):
  * mom < spec on PD-vs-ET AUC and macroP (a generic pretrained encoder with
    amplitude normalised away; the frozen ImageNet ViT reached macroP 0.501 vs
    0.652 here).
  * mom within 0.05 of spec on N-vs-tremor AUC (oscillation presence and
    sharpness are generic waveform properties).
  * union - u_shuf null on PD-vs-ET AUC (|d| < 0.02): MOMENT adds nothing the
    spectrum lacks.

OUTCOME of the first eval (40 partitions): the first and third predictions
FAILED -- mom beat spec on PD-vs-ET AUC (0.701 vs 0.621, +0.080, win 1.00),
precET (+0.059) and macroP (+0.034), and union beat u_shuf on every column.

Follow-up controls, predictions recorded before running them:
  * ``WEIGHTS=perm<seed>`` embeds with every MOMENT weight tensor randomly
    permuted (same values, scale and architecture; learned structure
    destroyed). If the gain is transfer, mom_perm falls well below mom on
    PD-vs-ET AUC (> 0.03). Prediction: it does -- but held weakly, since
    random features have beaten hand-made ones before in this repo.
  * A label-permutation null for mom's PD-vs-ET AUC: 0.701 sits just above the
    ~0.66-0.69 tops seen at 15 ET. Prediction: p < 0.05.

Run:
    python -m experiments.pretrained_moment export            (project env)
    <moment env python> -m experiments.pretrained_moment embed
    python -m experiments.pretrained_moment eval              (project env)
Env var WORK = folder for the npz files (default: moment_runs/, git-ignored).
"""
from __future__ import annotations

import os
import sys

import numpy as np

WORK = os.environ.get("WORK", "moment_runs")
WIN, HOP, CH = 512, 256, slice(3, 6)
MODEL_DIR = os.environ.get("MOMENT_DIR", "")   # local dir with config.json + model.safetensors


def export():
    from scipy.signal import butter, sosfiltfilt
    from common.quaternion_data import load_quaternion_recordings
    recs = load_quaternion_recordings("Data", action="OUT", mode="angular_velocity")
    sos = butter(4, [3, 15], btype="band", fs=100.0, output="sos")
    bp, raw, rec_of, subj, lab = [], [], [], [], []
    for i, r in enumerate(recs):
        x = r.x[CH].astype(np.float64)
        xb = sosfiltfilt(sos, x, axis=-1)
        for s in range(0, x.shape[1] - WIN + 1, HOP):
            bp.append(xb[:, s:s + WIN]); raw.append(x[:, s:s + WIN]); rec_of.append(i)
        subj.append(r.subject); lab.append(r.y)
    os.makedirs(WORK, exist_ok=True)
    np.savez(f"{WORK}/windows.npz", bp=np.array(bp, np.float32),
             raw=np.array(raw, np.float32), rec_of=np.array(rec_of),
             subj=np.array(subj), lab=np.array(lab))
    print(f"{len(recs)} recordings -> {len(bp)} windows of {WIN} samples; "
          f"patients {len(set(subj))}")


def embed():
    import torch
    from momentfm import MOMENTPipeline
    torch.set_num_threads(max(1, os.cpu_count() - 2))
    model = MOMENTPipeline.from_pretrained(MODEL_DIR or "AutonLab/MOMENT-1-small",
                                           model_kwargs={"task_name": "embedding"})
    model.init()
    model.eval()
    tag = os.environ.get("WEIGHTS", "pretrained")
    if tag.startswith("perm"):
        g = torch.Generator().manual_seed(int(tag[4:] or 0))
        with torch.no_grad():
            for prm in model.parameters():
                flat = prm.data.flatten()
                prm.data.copy_(flat[torch.randperm(flat.numel(), generator=g)].view_as(prm))
    d = np.load(f"{WORK}/windows.npz")
    out = {}
    for key in ("bp", "raw"):
        X = torch.tensor(d[key])
        embs = []
        with torch.no_grad():
            for i in range(0, len(X), 64):
                embs.append(model.embed(x_enc=X[i:i + 64], reduction="mean").embeddings.numpy())
        out[key] = np.concatenate(embs)
        print(f"{key}: embeddings {out[key].shape}", flush=True)
    np.savez(f"{WORK}/embeddings.npz" if tag == "pretrained"
             else f"{WORK}/embeddings_{tag}.npz", **out)


def _patient_mean(E, rec_of, subj):
    """Window -> recording mean -> patient mean, patients sorted (pipeline order)."""
    rec = {}
    for e, r in zip(E, rec_of):
        rec.setdefault(r, []).append(e)
    pat = {}
    for r, v in rec.items():
        pat.setdefault(subj[r], []).append(np.mean(v, 0))
    keys = sorted(pat)
    return np.array([np.mean(pat[k], 0) for k in keys]), np.array(keys)


def evaluate():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from common.cohorts import desc_table, logbin
    from common.quaternion_data import load_quaternion_recordings
    from experiments.final_model import method_table

    w = np.load(f"{WORK}/windows.npz"); e = np.load(f"{WORK}/embeddings.npz")
    subj, lab = w["subj"], w["lab"]
    Mbp, keys = _patient_mean(e["bp"], w["rec_of"], subj)
    Mraw, _ = _patient_mean(e["raw"], w["rec_of"], subj)
    recs = load_quaternion_recordings("Data", action="OUT", mode="angular_velocity")
    S, y, pats = method_table(recs, "multitaper", CH)
    assert (pats == keys).all(), "patient order mismatch"
    ylab = {s: l for s, l in zip(subj, lab)}
    assert all(ylab[k] == yy for k, yy in zip(keys, y))
    spec = np.hstack([logbin(S), desc_table(recs, CH)])
    print(f"2015 OUT: {len(y)} patients N/PD/ET={np.bincount(y).tolist()}; "
          f"MOMENT dim {Mbp.shape[1]}; spectrum+descriptors dim {spec.shape[1]}\n")

    import glob as _glob
    perm_arms = {}
    for f in sorted(_glob.glob(f"{WORK}/embeddings_perm*.npz")):
        t = os.path.basename(f)[len("embeddings_"):-4]
        perm_arms[f"mom_{t}"] = _patient_mean(np.load(f)["bp"], w["rec_of"], subj)[0]

    def arms(rng):
        perm = rng.permutation(len(y))
        return {"spec": spec, "mom": Mbp, "mom_raw": Mraw,
                "union": np.hstack([spec, Mbp]), "u_shuf": np.hstack([spec, Mbp[perm]]),
                **perm_arms}

    names = ("precN", "precPD", "precET", "macroP", "macroF1", "PDvsET AUC", "NvsT AUC")
    res = {k: [] for k in ("spec", "mom", "mom_raw", "union", "u_shuf", *perm_arms)}
    for rep in range(40):
        rng = np.random.default_rng(rep)
        A = arms(rng)
        for k, X in A.items():
            P = np.zeros((len(y), 3))
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=rep).split(X, y):
                sc = StandardScaler().fit(X[tr])
                m = LogisticRegression(C=0.1, max_iter=5000, class_weight="balanced")
                P[te] = m.fit(sc.transform(X[tr]), y[tr]).predict_proba(sc.transform(X[te]))
            pred = P.argmax(1)
            pr, _, f1, _ = precision_recall_fscore_support(y, pred, labels=[0, 1, 2],
                                                           zero_division=0)
            mk = y != 0
            a_pe = roc_auc_score(y[mk] == 2, np.log(P[mk, 2] + 1e-12) - np.log(P[mk, 1] + 1e-12))
            a_nt = roc_auc_score(y != 0, 1 - P[:, 0])
            res[k].append([pr[0], pr[1], pr[2], pr.mean(), f1.mean(), a_pe, a_nt])
    print(f"{'arm':>8}" + "".join(f"{n:>11}" for n in names))
    for k in res:
        res[k] = np.array(res[k])
        print(f"{k:>8}" + "".join(f"{v:>11.3f}" for v in res[k].mean(0)))
    pairs = [("spec", "mom"), ("spec", "mom_raw"), ("spec", "union"), ("u_shuf", "union")]
    pairs += [(k, "mom") for k in perm_arms]
    for base, k in pairs:
        print(f"\n{k} - {base}  (paired, 40 repeats)")
        for i, n in enumerate(names):
            d = res[k][:, i] - res[base][:, i]
            b = [np.random.default_rng(s).choice(d, len(d)).mean() for s in range(4000)]
            lo, hi = np.percentile(b, [2.5, 97.5])
            print(f"  {n:>10} {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] "
                  f"{'*' if lo > 0 or hi < 0 else ' '}  win {np.mean(d > 0):.2f}")

    # label-permutation null for MOMENT's binary PD-vs-ET AUC (same pipeline)
    mk = y != 0
    Xm, ym = Mbp[mk], (y[mk] == 2).astype(int)

    def pdet_auc(yy, seed0, reps):
        a = []
        for r in range(reps):
            s_ = np.zeros(len(yy))
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed0 + r).split(Xm, yy):
                sc = StandardScaler().fit(Xm[tr])
                m = LogisticRegression(C=0.1, max_iter=5000, class_weight="balanced")
                s_[te] = m.fit(sc.transform(Xm[tr]), yy[tr]).decision_function(sc.transform(Xm[te]))
            a.append(roc_auc_score(yy, s_))
        return float(np.mean(a))
    obs = pdet_auc(ym, 0, 10)
    rng = np.random.default_rng(123)
    null = np.array([pdet_auc(rng.permutation(ym), 1000 + 10 * i, 3) for i in range(200)])
    print(f"\nMOMENT binary PD-vs-ET AUC {obs:.3f}; label-permutation null 95 % "
          f"[{np.percentile(null, 2.5):.3f}, {np.percentile(null, 97.5):.3f}], "
          f"p = {(1 + (null >= obs).sum()) / (1 + len(null)):.3f}")
    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    {"export": export, "embed": embed, "eval": evaluate}[sys.argv[1]]()
