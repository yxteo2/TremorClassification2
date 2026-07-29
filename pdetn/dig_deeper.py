"""Deeper time-frequency exploration for separability.

Extends the decomposition study with:
  * SST (synchrosqueezed STFT) — a high-resolution reassignment method.
  * finer HHT IMF sweep around the 8-IMF optimum.
  * feature-level FUSION — concatenate reduced features from complementary
    decompositions (STFT best 3-class + HHT best PD-vs-ET) to test whether
    combining representations separates better than either alone.

Model-free: subject-CV LDA (3-class and PD-vs-ET), Fisher, silhouette on OUT.

    python -m pdetn.dig_deeper --data-root Data --action OUT
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from tremor.quaternion_data import load_quaternion_recordings
from pdetn.separability import method_features, separability


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="Data")
    p.add_argument("--action", default="OUT")
    p.add_argument("--f-max", type=float, default=15.0)
    p.add_argument("--output", default="artifacts/dig_deeper")
    args = p.parse_args()

    recs = load_quaternion_recordings(args.data_root, action=args.action,
                                      mode="angular_velocity")
    print(f"[dig] {args.action}: {len(recs)} recordings")

    # Single-method configs to add.
    singles = [
        ("sst", "sst", {}),
        ("hht_imf7", "hht", {"hht_max_imfs": 7}),
        ("hht_imf9", "hht", {"hht_max_imfs": 9}),
        ("hht_imf12", "hht", {"hht_max_imfs": 12}),
        ("stft_win256", "stft", {"nperseg": 256, "nfft": 256, "noverlap": 192}),
        ("cwt_w06_step0.5", "cwt", {"cwt_w0": 6.0, "cwt_freq_step": 0.5}),
    ]
    feats: dict[str, np.ndarray] = {}
    y = subj = None
    res = {}
    for label, method, kw in singles:
        t = time.time()
        try:
            X, y, subj = method_features(recs, method, f_max=args.f_max, **kw)
            feats[label] = X
            s = separability(X, y, subj); s["sec"] = round(time.time() - t, 1)
            res[label] = s
            print(f"  {label:>18}: 3cls={s['subjcv_lda_f1']:.3f} "
                  f"PDvsET={s['pdet_lda_f1']:.3f} ({s['sec']}s)")
        except Exception as e:
            print(f"  {label:>18}: FAILED {type(e).__name__}: {e}")

    # Feature-level fusion of complementary decompositions (z-scored concat).
    def zc(*names):
        from sklearn.preprocessing import StandardScaler
        return np.concatenate([StandardScaler().fit_transform(feats[n]) for n in names], axis=1)

    fusions = [
        ("fuse_stft256+hht8", ["stft_win256"]),   # hht8 added below if present
    ]
    # Build fusions only from what we actually computed (+ reuse decomp_sweep hht8 if present).
    if "stft_win256" in feats and "hht_imf9" in feats:
        for name, combo in [("fuse_stft256+hht9", ["stft_win256", "hht_imf9"]),
                            ("fuse_stft256+cwt", ["stft_win256", "cwt_w06_step0.5"]),
                            ("fuse_cwt+hht9", ["cwt_w06_step0.5", "hht_imf9"]),
                            ("fuse_all3", ["stft_win256", "cwt_w06_step0.5", "hht_imf9"])]:
            if all(c in feats for c in combo):
                Xf = zc(*combo)
                s = separability(Xf, y, subj)
                res[name] = s
                print(f"  {name:>18}: 3cls={s['subjcv_lda_f1']:.3f} "
                      f"PDvsET={s['pdet_lda_f1']:.3f} (fusion, {Xf.shape[1]} feat)")

    outdir = Path(args.output); outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"{args.action}.json").write_text(json.dumps(res, indent=2))
    print(f"\n=== ranked by PD-vs-ET LDA F1 ({args.action}) ===")
    for k, v in sorted(res.items(), key=lambda kv: -(kv[1]["pdet_lda_f1"] if kv[1]["pdet_lda_f1"] == kv[1]["pdet_lda_f1"] else -1)):
        print(f"  {k:>18}  3cls={v['subjcv_lda_f1']:.3f}  PDvsET={v['pdet_lda_f1']:.3f}")
    print(f"\nsaved -> {outdir}/{args.action}.json")


if __name__ == "__main__":
    main()
