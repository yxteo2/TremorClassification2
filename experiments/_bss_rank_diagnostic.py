"""BSS rank diagnostic for transfer_reg_150: rank / dead columns of the penultimate feature matrix after PADS pretraining (rep 0, fold 0, seed 0). Run: python -m experiments._bss_rank_diagnostic"""
import numpy as np, torch, torch.nn as nn
from experiments.transfer_reg_150 import _data, _pads, _folds, _T, _L, _wt, _head, EPOCHS, LR, WD
from experiments.transfer_2015 import members, zfit
torch.set_num_threads(1)
spec, desc, y, own, Cp, C = _data()
mk1, mk2 = members(desc.shape[1]); kp = _pads(C, 0)
pads = {"packed": Cp[kp], "spec": C[0][kp]}
_, tr, va, te = next(_folds(spec, y, 0))
for key, mk in (("packed", mk1), ("spec", mk2)):
    X, P = own[key], pads[key]; f = zfit(X[tr])
    torch.manual_seed(0); m = mk()
    xp, yp = _T(zfit(P)(P)), _L(C[3][kp]); lf = nn.CrossEntropyLoss(weight=_wt(C[3][kp]))
    op = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=WD); sp = torch.optim.lr_scheduler.CosineAnnealingLR(op, EPOCHS)
    m.train()
    for _ in range(EPOCHS):
        op.zero_grad(); lf(m(xp), yp).backward(); op.step(); sp.step()
    feats = {}; _head(m).register_forward_hook(lambda mod, i, o: feats.__setitem__("f", i[0]))
    for mode in ("train", "eval"):
        getattr(m, mode)()
        with torch.no_grad(): m(_T(f(X[tr])))
        F = feats["f"]; sv = torch.linalg.svdvals(F)
        print(key, mode, "shape", tuple(F.shape), "all-zero cols", int((F.abs().sum(0) == 0).sum()),
              "rank", int(torch.linalg.matrix_rank(F)), "sv min/max %.3g/%.3g" % (sv[-1], sv[0]))
