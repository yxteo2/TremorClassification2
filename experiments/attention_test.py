"""Does attention help on the CURRENT input? The fair test.

Attention has been dismissed here twice, and neither test was fair to it:

  * `BilateralAttention` -- 25-patient NewData, 61-bin raw spectra, at chance
  * `vit_b_16` -- 85.8 M parameters, random weights (ImageNet proxy-blocked),
    AUC 0.540 on 58 patients

Neither used the current input (16 log-scaled bins + IF trajectory) or the
current cohort (404 patients). This runs two small attention models that do:

``SpectrumTransformer``   frequency bins as tokens with sinusoidal position
                          encoding -- gives attention the "where in the band"
                          information a plain convolution lacks. 17 k params.
``CrossStreamAttention``  spectrum tokens attend to the trajectory, so the two
                          streams can condition on each other instead of meeting
                          only at the classifier head. 5 k params.

Both are in the 1e4 band every model on this cohort has peaked in.
"""
import numpy as np, torch
from experiments.trajectory_tuning import assemble, evaluate, paired
from experiments.selection_and_calibration import _fit_split, score
from common.protocol import NBIN, TEST_FRAC, VAL_FRAC, train, tune_offsets
from models.architectures import (CrossStreamAttention, DescriptorFusion,
                                  ResidualTCN, Spectrum1DCNN, SpectrumTransformer,
                                  TRUNKS, TwoStreamNet)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import precision_recall_fscore_support

SPLITS, TL = 20, 64

def run(name, spec, desc, traj, y, key, kind):
    nd = desc.shape[1]
    packed = np.hstack([spec, desc, traj])
    if kind == "twostream":
        mk1 = lambda: TwoStreamNet(Spectrum1DCNN(NBIN, 3, ch=8), TRUNKS["cnn"],
                                   8*2*4, NBIN, nd, TL)
    elif kind == "transformer":
        mk1 = lambda: TwoStreamNet(SpectrumTransformer(NBIN, 3, d=32),
                                   TRUNKS["trunk"], 32, NBIN, nd, TL)
    elif kind == "cross":
        mk1 = lambda: CrossStreamAttention(NBIN, nd, TL)
    mk2 = lambda: ResidualTCN(NBIN, num_classes=3, ch=16)
    out = []
    for sp in range(SPLITS):
        tv, te = next(StratifiedShuffleSplit(1, test_size=TEST_FRAC,
                      random_state=sp).split(packed, key))
        t0, v0 = next(StratifiedShuffleSplit(1, test_size=VAL_FRAC,
                      random_state=sp).split(packed[tv], key[tv]))
        tr, va = tv[t0], tv[v0]
        pv_l, pt_l = [], []
        for X, mk in ((packed, mk1), (spec, mk2)):
            mu, sd = X[tr].mean(0, keepdims=True), X[tr].std(0, keepdims=True)+1e-8
            r = [train(mk, (X[tr]-mu)/sd, y[tr], (X[va]-mu)/sd, y[va],
                       [(X[va]-mu)/sd, (X[te]-mu)/sd], seed=s) for s in (0,1,2)]
            pv_l.append(np.mean([a[0] for a in r],0))
            pt_l.append(np.mean([a[1] for a in r],0))
        pv, pt = np.mean(pv_l,0), np.mean(pt_l,0)
        pred = (np.log(pt+1e-12) + tune_offsets(pv, y[va])).argmax(1)
        out.append(score(y[te], pred))
    a = np.array(out); m, s = a.mean(0), a.std(0)
    print(f"{name:>34}" + "".join(f"{m[i]:>9.3f}" for i in range(5))
          + "  |" + "".join(f"{s[i]:>7.3f}" for i in range(5)), flush=True)
    return a

def main():
    torch.set_num_threads(1)
    spec, desc, traj, n_ch, y, key = assemble(axis_mode="mean", n_out=TL)
    npar = lambda m: sum(p.numel() for p in m.parameters())
    print(f"n={len(y)}  params: Spectrum1DCNN "
          f"{npar(Spectrum1DCNN(NBIN,3,ch=8))/1000:.1f}k  SpectrumTransformer "
          f"{npar(SpectrumTransformer(NBIN,3,d=32))/1000:.1f}k  CrossStream "
          f"{npar(CrossStreamAttention(NBIN, desc.shape[1], TL))/1000:.1f}k\n")
    print(f"{'config':>34}{'precN':>9}{'precPD':>9}{'precET':>9}{'macroP':>9}"
          f"{'macroF1':>9}  |{'  sd':>7}")
    base = run("TwoStreamNet CNN (current best)", spec, desc, traj, y, key, "twostream")
    tr_  = run("+ SpectrumTransformer", spec, desc, traj, y, key, "transformer")
    cr_  = run("CrossStreamAttention", spec, desc, traj, y, key, "cross")
    print(f"\npaired vs current best, {SPLITS} splits:")
    paired(tr_, base, "SpectrumTransformer")
    paired(cr_, base, "CrossStreamAttention")
    print("\nMARKER_DONE", flush=True)

if __name__ == "__main__":
    main()
