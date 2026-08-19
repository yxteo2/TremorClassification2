"""Learn how to pool a patient's recordings instead of averaging them.

Every model in this project reduces a patient to one spectrum before the network
ever runs:

    rows[r.subject].append(s / s.sum())     # frequency/tables.py
    ...
    np.array([np.mean(rows[p], 0) for p in pats])

Two things are wrong with that, and both are measurable.

**1. The aggregator is hard-coded and there is evidence it is the wrong one.**
`averaging two PADS tasks` measured *worse* than using one task alone (precET
0.585 vs 0.612). A uniform mean over recordings of differing quality and
informativeness is a strong assumption, and that result says it does not hold
here. Attention pooling learns the weights instead.

**2. The pipeline throws away most of its recordings.** It merges on a single
postural task per cohort, so of 3,081 loaded recordings roughly 1,140 are used:

    PADS      766 StretchHold used, 766 Relaxed discarded
    NewData   100 OUT used, ~650 across six other tasks discarded
    2015      274 OUT used, ~525 across REST and WING discarded

Multiple-instance learning uses all of them without having to pick one or
average them: the bag is the patient, the instances are that patient's
recordings, and the label is at the bag level, which is exactly how the data is
actually labelled.

**Why this is not the attention that already failed here.** Cross-attention
between streams and attention over frequency bins both measured at or below
their controls. This is attention over a *set of recordings* -- typically 4 to 14
items, one scalar weight each, permutation-invariant. It is a far smaller
hypothesis class than either, and it targets a specific known-wrong assumption
rather than adding generic capacity.

Arms, all sharing the same instance encoder and the same 16-bin log multitaper
representation, so the comparison isolates **aggregation** alone:

  avg spectrum, postural only   the current pipeline -- baseline
  avg spectrum, ALL tasks       uniform mean over every recording. Tests whether
                                the extra tasks help at all under the existing
                                aggregator, and separates "more data" from
                                "better pooling".
  MIL mean-pool                 encode each recording, then average the
                                EMBEDDINGS. Late vs early averaging, nothing
                                learned.
  MIL max-pool                  the classic MIL aggregator: the patient is as
                                abnormal as their most abnormal recording.
  MIL gated attention           Ilse et al. ABMIL -- learned instance weights.

Reading the arms in that order separates three effects: extra recordings,
late-vs-early pooling, and learned weights.

Run: ``python -m experiments.mil_recordings``
"""

from __future__ import annotations

import os
import re
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedShuffleSplit

from common.cohorts import logbin
from common.protocol import TEST_FRAC, VAL_FRAC, tune_offsets
from experiments.final_model import GRID, NBIN
from signal_processing.transforms import METHODS

SPLITS, SEEDS = 20, (0, 1, 2)
NM = ("precN", "precPD", "precET", "macroP", "macroF1")


# --------------------------------------------------------------------------- #
# Data: one bag per patient
# --------------------------------------------------------------------------- #
_ACTION_SUFFIX = re.compile(r"_(OUT|REST|WING)$")


def patient_key(cohort, subject):
    """One key per physical patient, stable across tasks.

    2015 encodes the action INTO the subject id -- ``ET 10_OUT``, ``ET 10_REST``,
    ``ET 10_WING`` are one person. Keying on the raw subject splits them into
    three rows, which inflates n from 151 to 440 for that cohort and, far worse,
    puts the same patient in both train and test. NewData and PADS already use
    task-independent ids, so only 2015 needs stripping.

    No existing result is affected: every other experiment loads a single action,
    where the raw ids are already one-per-patient.
    """
    return f"{cohort}:{_ACTION_SUFFIX.sub('', str(subject))}"


def _spec(x):
    """Multitaper power on the shared 3-15 Hz grid, sum-normalised."""
    f, P = METHODS["multitaper"](x)
    f, P = np.asarray(f, float), np.asarray(P, float)
    m = np.isfinite(P)
    v = np.clip(np.interp(GRID, f[m], P[m], left=0.0, right=0.0), 0, None)
    s = v.sum()
    return v / s if s > 0 else None


def build_bags():
    """(bags, y, cohort) where bags[i] is (n_instances, NBIN) for patient i.

    ``postural`` marks the recordings the current pipeline would have used, so
    the baseline arm can be reconstructed from the identical feature computation
    rather than from a separately-built table.
    """
    from common.load_2025 import ALL_TASKS_2025, load_2025_all
    from common.loaders import load_pads_extracted
    from common.quaternion_data import load_quaternion_recordings

    src = []
    for act in ("OUT", "REST", "WING"):
        try:
            src.append(("2015", act, act == "OUT",
                        load_quaternion_recordings("Data", action=act,
                                                   mode="angular_velocity"),
                        slice(3, 6)))
        except Exception:
            pass
    for task in ALL_TASKS_2025:
        try:
            r = load_2025_all(conditions=(task,))
            if r:
                src.append(("NewData", task, task == "OUT", r, slice(3, 6)))
        except Exception:
            pass
    for folder in ("pads_stretchhold", "pads_relaxed"):
        if os.path.isdir(folder):
            try:
                src.append(("PADS", folder, folder == "pads_stretchhold",
                            load_pads_extracted(folder), slice(0, 3)))
            except Exception:
                pass

    rows, post, lab, coh = (defaultdict(list), defaultdict(list), {}, {})
    for cohort, tag, is_post, recs, ch in src:
        n = 0
        for r in recs:
            x = r.x[ch] if r.x.shape[0] > 3 else r.x
            v = _spec(x)
            if v is None:
                continue
            key = patient_key(cohort, r.subject)
            rows[key].append(v)
            post[key].append(bool(is_post))
            lab[key], coh[key] = r.y, cohort
            n += 1
        print(f"    {cohort:>8} {tag:<18} {n:>4} recordings"
              f"{'   [postural]' if is_post else ''}")

    pats = sorted(rows)
    bags = [np.nan_to_num(logbin(np.array(rows[p]))) for p in pats]
    postm = [np.array(post[p]) for p in pats]
    y = np.array([lab[p] for p in pats])
    c = np.array([coh[p] for p in pats])
    return bags, postm, y, c, np.array(pats)


def cap_pads(y, coh, cap=90, seed=0):
    """The settled merge rule: PADS capped at 90/class, everything else kept."""
    rng = np.random.default_rng(seed)
    keep = list(np.flatnonzero(coh != "PADS"))
    for cl in (0, 1, 2):
        i = np.flatnonzero((coh == "PADS") & (y == cl))
        keep.extend(rng.choice(i, min(cap, len(i)), replace=False))
    return np.array(sorted(keep))


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class InstanceEncoder(nn.Module):
    """Spectrum1DCNN's trunk, applied to one recording."""

    def __init__(self, n_bins=NBIN, ch=8):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, ch, 5, padding=2), nn.BatchNorm1d(ch), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(ch, ch * 2, 3, padding=1), nn.BatchNorm1d(ch * 2),
            nn.ReLU(), nn.AdaptiveAvgPool1d(4), nn.Flatten())
        self.dim = ch * 2 * 4

    def forward(self, x):                      # (B*I, bins) -> (B*I, dim)
        return self.conv(x.unsqueeze(1))


class MILNet(nn.Module):
    """Encode every instance, pool over the bag, classify.

    ``pool`` is "mean", "max" or "attn". The attention branch is gated ABMIL:
    a tanh path and a sigmoid gate, multiplied, then a linear score per
    instance, softmaxed over the bag with padded slots masked to -inf.
    """

    def __init__(self, pool="attn", n_bins=NBIN, ch=8, att=32, dropout=0.3,
                 num_classes=3):
        super().__init__()
        self.pool = pool
        self.enc = InstanceEncoder(n_bins, ch)
        d = self.enc.dim
        if pool == "attn":
            self.V = nn.Sequential(nn.Linear(d, att), nn.Tanh())
            self.U = nn.Sequential(nn.Linear(d, att), nn.Sigmoid())
            self.w = nn.Linear(att, 1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d, num_classes))

    def forward(self, x, mask):                # x (B, I, bins), mask (B, I)
        B, I, F = x.shape
        h = self.enc(x.reshape(B * I, F)).reshape(B, I, -1)
        m = mask.unsqueeze(-1)
        if self.pool == "mean":
            z = (h * m).sum(1) / m.sum(1).clamp(min=1)
        elif self.pool == "max":
            z = h.masked_fill(~m.bool(), -1e9).max(1).values
        else:
            a = self.w(self.V(h) * self.U(h)).squeeze(-1)       # (B, I)
            a = a.masked_fill(~mask.bool(), -1e9).softmax(1)
            z = (h * a.unsqueeze(-1)).sum(1)
        return self.head(z)


def pad(bags, idx):
    """Stack ragged bags into (B, Imax, bins) plus a float mask."""
    sel = [bags[i] for i in idx]
    B, I, F = len(sel), max(len(b) for b in sel), sel[0].shape[1]
    x = np.zeros((B, I, F), np.float32)
    m = np.zeros((B, I), np.float32)
    for j, b in enumerate(sel):
        x[j, :len(b)] = b
        m[j, :len(b)] = 1.0
    return x, m


def train_mil(pool, xtr, mtr, ytr, xva, mva, yva, outs, seed=0,
              epochs=200, lr=3e-3, wd=1e-3):
    """Full-batch, class-weighted, best-validation-loss checkpoint.

    Matches ``common.protocol.train`` in optimiser, schedule, epoch count and
    model selection, so nothing but the pooling differs from the baseline.
    """
    torch.manual_seed(seed)
    T = lambda z: torch.tensor(z, dtype=torch.float32)
    xt, mt = T(xtr), T(mtr)
    yt = torch.tensor(ytr, dtype=torch.long)
    xv, mv = T(xva), T(mva)
    yv = torch.tensor(yva, dtype=torch.long)
    cnt = np.bincount(ytr, minlength=3).astype(float)
    w = torch.tensor(cnt.sum() / (3 * np.maximum(cnt, 1)), dtype=torch.float32)
    lf = nn.CrossEntropyLoss(weight=w)

    m = MILNet(pool=pool)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best, state = np.inf, None
    for _ in range(epochs):
        m.train(); opt.zero_grad()
        lf(m(xt, mt), yt).backward(); opt.step(); sch.step()
        m.eval()
        with torch.no_grad():
            v = float(lf(m(xv, mv), yv))
        if v < best:
            best = v
            state = {k: t.clone() for k, t in m.state_dict().items()}
    if state is not None:
        m.load_state_dict(state)
    m.eval()
    out = []
    with torch.no_grad():
        for xo, mo in outs:
            out.append(torch.softmax(m(T(xo), T(mo)), 1).numpy())
    return out


def paired(a, b, n=4000):
    d = a - b
    return [(d[:, i].mean(),
             *np.percentile([np.mean(np.random.default_rng(s).choice(
                 d[:, i], len(d), replace=True)) for s in range(n)],
                 [2.5, 97.5]))
            for i in range(len(NM))]


def main():
    torch.set_num_threads(1)
    print("building bags ...", flush=True)
    bags, postm, y, coh, pats = build_bags()

    # Keep only patients who HAVE a postural recording. Loading every task turns
    # up 18 patients with e.g. REST but no OUT, whom the current pipeline never
    # sees. Including them would change the patient set as well as the pooling,
    # and the baseline arm would stop being the current pipeline. Restricting
    # here recovers exactly the established n=404.
    has_post = np.array([p.any() for p in postm])
    d = int((~has_post).sum())
    if d:
        print(f"\n  dropping {d} patients with no postural recording, so every "
              f"arm describes the same patients")
    idx = np.flatnonzero(has_post)
    bags = [bags[i] for i in idx]
    postm = [postm[i] for i in idx]
    y, coh, pats = y[idx], coh[idx], pats[idx]

    keep = cap_pads(y, coh)
    bags = [bags[i] for i in keep]
    postm = [postm[i] for i in keep]
    y, coh = y[keep], coh[keep]
    key = np.array([f"{c}_{l}" for c, l in zip(coh, y)])

    # the current pipeline's view: postural recordings only, averaged
    post_bags = [b[p] for b, p in zip(bags, postm)]

    sizes = np.array([len(b) for b in bags])
    psz = np.array([len(b) for b in post_bags])
    print(f"\nn={len(y)}  N={int((y==0).sum())} PD={int((y==1).sum())} "
          f"ET={int((y==2).sum())}")
    print(f"instances per patient -- all tasks: min {sizes.min()} "
          f"med {int(np.median(sizes))} max {sizes.max()}  "
          f"(total {sizes.sum()})")
    print(f"                        postural : min {psz.min()} "
          f"med {int(np.median(psz))} max {psz.max()}  "
          f"(total {psz.sum()})\n")

    ARMS = (("avg spectrum, postural only", "mean", post_bags, True),
            ("avg spectrum, ALL tasks", "mean", bags, True),
            ("MIL mean-pool, ALL tasks", "mean", bags, False),
            ("MIL max-pool, ALL tasks", "max", bags, False),
            ("MIL gated attention, ALL tasks", "attn", bags, False))

    res = {lab: [] for lab, _, _, _ in ARMS}
    for sp in range(SPLITS):
        tv, te = next(StratifiedShuffleSplit(1, test_size=TEST_FRAC,
                                             random_state=sp).split(y, key))
        t0, v0 = next(StratifiedShuffleSplit(1, test_size=VAL_FRAC,
                                             random_state=sp).split(y[tv],
                                                                    key[tv]))
        tr, va = tv[t0], tv[v0]

        for lab, pool, src, collapse in ARMS:
            use = ([m.mean(0, keepdims=True) for m in src] if collapse else src)
            xt, mt = pad(use, tr)
            xv, mv = pad(use, va)
            xe, me = pad(use, te)
            # per-instance standardisation fitted on TRAIN instances only
            flat = xt[mt.astype(bool)]
            mu, sd = flat.mean(0), flat.std(0) + 1e-8
            nz = lambda a, m_: ((a - mu) / sd) * m_[..., None]
            r = [train_mil(pool, nz(xt, mt), mt, y[tr], nz(xv, mv), mv, y[va],
                           [(nz(xv, mv), mv), (nz(xe, me), me)], seed=s)
                 for s in SEEDS]
            pv = np.mean([a[0] for a in r], 0)
            pt = np.mean([a[1] for a in r], 0)
            pred = (np.log(pt + 1e-12) + tune_offsets(pv, y[va])).argmax(1)
            P, _, F, _ = precision_recall_fscore_support(
                y[te], pred, labels=[0, 1, 2], zero_division=0)
            res[lab].append([P[0], P[1], P[2], P.mean(), F.mean()])
        print(f"  split {sp+1}/{SPLITS} done", flush=True)

    for k in res:
        res[k] = np.array(res[k])

    print(f"\n{'arm':>32}" + "".join(f"{c:>9}" for c in NM) + "   sd(macroP)")
    for lab, _, _, _ in ARMS:
        m = res[lab].mean(0)
        print(f"{lab:>32}" + "".join(f"{v:>9.3f}" for v in m)
              + f"{res[lab][:, 3].std():>12.3f}")

    base = res["avg spectrum, postural only"]
    print("\npaired vs the current pipeline (avg spectrum, postural only):")
    for lab, _, _, _ in ARMS[1:]:
        print(f"  {lab}:")
        for (dd, lo, hi), c in zip(paired(res[lab], base), NM):
            star = "*" if lo > 0 or hi < 0 else " "
            print(f"    {c:>8} {dd:+.3f}  [{lo:+.3f}, {hi:+.3f}] {star}")

    b2 = res["avg spectrum, ALL tasks"]
    print("\npaired vs uniform averaging of ALL tasks "
          "(isolates learned pooling from extra data):")
    for lab, _, _, _ in ARMS[2:]:
        print(f"  {lab}:")
        for (dd, lo, hi), c in zip(paired(res[lab], b2), NM):
            star = "*" if lo > 0 or hi < 0 else " "
            print(f"    {c:>8} {dd:+.3f}  [{lo:+.3f}, {hi:+.3f}] {star}")
    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    main()
