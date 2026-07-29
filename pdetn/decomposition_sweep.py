"""Quantitative decomposition study: tune each TF method's parameters for
maximum class separability, model-free, before training any model.

We sweep the knobs that actually change the time-frequency decomposition —
STFT window length, CWT Morlet width & frequency step, HHT IMF count, wavelet
level & family — and score each configuration by subject-CV LDA separability
(3-class and the hard PD-vs-ET axis) plus Fisher / silhouette.

    python -m pdetn.decomposition_sweep --data-root Data --action OUT
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from tremor.quaternion_data import load_quaternion_recordings

from pdetn.separability import method_features, separability


def build_grid() -> list[dict]:
    """(label, method, kwargs) configurations that vary the decomposition."""
    cfgs = []
    # STFT: window length trades time vs frequency resolution.
    for w in (64, 128, 256):
        cfgs.append({"label": f"stft_win{w}", "method": "stft",
                     "kw": {"nperseg": w, "nfft": w, "noverlap": int(w * 0.75)}})
    # CWT: Morlet width w0 (time-freq tradeoff) x frequency step (scale density).
    for w0 in (4.0, 6.0, 8.0, 10.0):
        for step in (0.25, 0.5):
            cfgs.append({"label": f"cwt_w0{w0:g}_step{step}", "method": "cwt",
                         "kw": {"cwt_w0": w0, "cwt_freq_step": step}})
    # HHT: number of IMFs retained (the decomposition depth).
    for imf in (4, 6, 8, 10):
        cfgs.append({"label": f"hht_imf{imf}", "method": "hht",
                     "kw": {"hht_max_imfs": imf}})
    # Wavelet packet: decomposition level x wavelet family.
    for lvl in (4, 5, 6):
        for wav in ("db4", "db6"):
            cfgs.append({"label": f"wp_l{lvl}_{wav}", "method": "wavelet_packet",
                         "kw": {"wp_level": lvl, "wp_wavelet": wav}})
    # Multitaper reference.
    cfgs.append({"label": "multitaper", "method": "multitaper", "kw": {}})
    return cfgs


def run(recs, f_max: float = 15.0) -> dict:
    out = {}
    for cfg in build_grid():
        t = time.time()
        try:
            X, y, subj = method_features(recs, cfg["method"], f_max=f_max, **cfg["kw"])
            s = separability(X, y, subj)
            s["sec"] = round(time.time() - t, 1)
            out[cfg["label"]] = s
            print(f"  {cfg['label']:>18}: 3cls={s['subjcv_lda_f1']:.3f} "
                  f"PDvsET={s['pdet_lda_f1']:.3f} sil={s['silhouette']:.3f} "
                  f"({s['sec']}s)")
        except Exception as e:  # keep going if one config fails
            print(f"  {cfg['label']:>18}: FAILED {type(e).__name__}: {e}")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="Data")
    p.add_argument("--action", default="OUT")
    p.add_argument("--f-max", type=float, default=15.0)
    p.add_argument("--output", default="artifacts/decomp_sweep")
    args = p.parse_args()

    recs = load_quaternion_recordings(args.data_root, action=args.action,
                                      mode="angular_velocity")
    print(f"[decomp] {args.action}: {len(recs)} recordings, "
          f"{len(set(r.subject for r in recs))} subjects")
    res = run(recs, f_max=args.f_max)

    outdir = Path(args.output); outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"{args.action}.json").write_text(json.dumps(res, indent=2))
    # ranked tables
    def table(key):
        rows = sorted(res.items(), key=lambda kv: -(kv[1][key] if kv[1][key] == kv[1][key] else -1))
        return "\n".join(f"  {k:>18}  {v[key]:.3f}" for k, v in rows)
    print(f"\n=== ranked by 3-class LDA F1 ({args.action}) ===\n" + table("subjcv_lda_f1"))
    print(f"\n=== ranked by PD-vs-ET LDA F1 ({args.action}) ===\n" + table("pdet_lda_f1"))
    print(f"\nsaved -> {outdir}/{args.action}.json")


if __name__ == "__main__":
    main()
