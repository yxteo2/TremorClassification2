"""Compare preprocessing recipes and recommend the best one for training.

Run as a module from the repo root:

    python -m tremor.compare_preprocessing --data-root . --action OUT

The script auto-detects which data sources are available under
``<data-root>/ProcessedData/`` (precomputed ``stft``, raw
``filtered_amplitudes``, raw ``raw_quaternion``), runs every sensible
recipe through a leave-one-out LDA evaluator on a mean-spectrogram
feature, prints a ranked table, and tells you the exact
``train.py`` invocation to reproduce the winner.

If no real data is found and ``--synthetic`` is passed, the script
falls back to a synthetic PD / ET / Normal benchmark so you can still
sanity-check that the recipes work.
"""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------
# Recipe definition + evaluation
# ---------------------------------------------------------------------


def _apply_tfd(signal: np.ndarray, tfd_method: str, fs: float, f_max: float,
               cwt_step: float = 0.5) -> np.ndarray:
    """Compute the (F, T) magnitude spectrogram for one (channels, time) signal."""
    from tremor.tfd import apply_stft, apply_cwt, apply_hht

    if tfd_method == "stft":
        return apply_stft(signal, fs=fs, nperseg=128, nfft=256, noverlap=96,
                          f_max=f_max)
    if tfd_method == "cwt":
        freqs = np.arange(3.0, f_max + 1e-9, cwt_step)
        return apply_cwt(signal, fs=fs, freqs=freqs, w0=6.0, decim=8,
                         f_max=f_max)
    if tfd_method == "hht":
        freqs = np.arange(3.0, f_max + 1e-9, cwt_step)
        return apply_hht(signal, fs=fs, freqs=freqs, max_imfs=8, decim=8,
                         f_max=f_max)
    raise ValueError(f"unknown tfd_method {tfd_method!r}")


def _apply_norm(S: np.ndarray, normalize: str, log_compress_on: bool) -> np.ndarray:
    """Apply log + normalization in the same order TremorDataset does."""
    from tremor.spectral import log_compress, per_freq_zscore, per_recording_zscore

    if log_compress_on:
        S = log_compress(S)
    if normalize == "per_recording":
        S = per_recording_zscore(S)
    elif normalize == "per_freq":
        S = per_freq_zscore(S)
    return S


def _flatten_mean_time(S: np.ndarray) -> np.ndarray:
    """Average over time to get a 1-D feature vector for LDA."""
    if S.ndim == 1:
        return S
    return S.mean(axis=-1)


def _fisher_ratio(features: np.ndarray, labels: np.ndarray) -> float:
    classes = np.unique(labels)
    means = {c: features[labels == c].mean(axis=0) for c in classes}
    vars_ = {c: features[labels == c].var(axis=0) for c in classes}
    pair_scores = []
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            ci, cj = classes[i], classes[j]
            num = (means[ci] - means[cj]) ** 2
            den = vars_[ci] + vars_[cj] + 1e-9
            pair_scores.append((num / den).mean())
    return float(np.mean(pair_scores))


def _lda_loo_accuracy(features: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    n = len(features)
    if n < 3:
        return float("nan")
    correct = 0
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        try:
            lda = LinearDiscriminantAnalysis()
            lda.fit(features[mask], labels[mask])
            if lda.predict(features[i : i + 1])[0] == labels[i]:
                correct += 1
        except Exception:
            pass
    return correct / n


def _evaluate_recipe(
    signals: list[np.ndarray], labels: np.ndarray,
    tfd_method: str, normalize: str, log_compress_on: bool,
    fs: float, f_max: float,
) -> tuple[float, float]:
    feats = []
    for x in signals:
        S = _apply_tfd(x, tfd_method=tfd_method, fs=fs, f_max=f_max)
        S = _apply_norm(S, normalize=normalize, log_compress_on=log_compress_on)
        feats.append(_flatten_mean_time(S))
    feats = np.stack(feats)
    return _fisher_ratio(feats, labels), _lda_loo_accuracy(feats, labels)


# ---------------------------------------------------------------------
# Data-source detection / loading
# ---------------------------------------------------------------------


def _detect_sources(data_root: Path, action: str) -> dict[str, Path]:
    """Return the (source_name -> folder) of available preprocessing inputs."""
    pd = data_root / "ProcessedData"
    candidates = {
        "quaternion": pd / "raw_quaternion" / action,
        "raw_amplitudes": pd / "filtered_amplitudes" / action,
        "precomputed_stft": pd / "stft" / action,
    }
    return {name: p for name, p in candidates.items() if p.is_dir()}


def _load_quaternion(data_root: Path, action: str, fs: float):
    from tremor.quaternion_data import load_quaternion_recordings

    recs = load_quaternion_recordings(
        data_root, action=action, fs=fs, mode="angular_velocity",
    )
    return [r.x for r in recs], np.asarray([r.y for r in recs])


def _load_amplitudes(data_root: Path, action: str):
    from tremor.data import load_recordings

    recs = load_recordings(data_root, feature="filtered_amplitudes", action=action)
    return [r.x for r in recs], np.asarray([r.y for r in recs])


def _load_precomputed_stft(data_root: Path, action: str):
    """Each loaded record is already a spectrogram (features, time)."""
    from tremor.stft_data import load_stft_recordings

    recs = load_stft_recordings(data_root, action=action, feature="stft")
    return [r.x for r in recs], np.asarray([r.y for r in recs])


def _synthetic_dataset(seed: int = 0, n_per_class: int = 15, n_samples: int = 500):
    """Quaternion-derived angular velocity for PD / ET / Normal."""
    from tremor.quaternion import process_quaternion_data

    fs = 100.0

    def synth(freq_range, amp_range, env_freq=0.0, noise_q=0.003, rng=None):
        t = np.arange(n_samples) / fs
        f = rng.uniform(*freq_range)
        a = rng.uniform(*amp_range)
        env = 1 + 0.15 * np.sin(2 * np.pi * env_freq * t) if env_freq > 0 else 1
        theta = a * env * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
        Q = np.zeros((n_samples, 4), dtype=np.float32)
        Q[:, rng.integers(0, 3)] = np.sin(theta / 2)
        Q[:, 3] = np.cos(theta / 2)
        Q += rng.normal(0, noise_q, Q.shape).astype(np.float32)
        Q /= np.linalg.norm(Q, axis=1, keepdims=True)
        return np.tile(Q, (1, 3))

    rng = np.random.default_rng(seed)
    sigs, labs = [], []
    for i in range(n_per_class):
        sigs.append(synth((4.0, 7.0), (0.04, 0.07),
                          rng=np.random.default_rng(seed + i)))
        labs.append(1)
        sigs.append(synth((6.0, 10.0), (0.02, 0.05), env_freq=0.4,
                          rng=np.random.default_rng(seed + 100 + i)))
        labs.append(2)
        sigs.append(synth((8.0, 12.0), (0.003, 0.008), noise_q=0.006,
                          rng=np.random.default_rng(seed + 200 + i)))
        labs.append(0)

    av = [process_quaternion_data(s, fs=fs, mode="angular_velocity")
          for s in sigs]
    return av, np.asarray(labs)


# ---------------------------------------------------------------------
# Recipe sweeps
# ---------------------------------------------------------------------

TFD_METHODS = ("stft", "cwt", "hht")
NORM_CHOICES = ("none", "per_recording", "per_freq")
LOG_CHOICES = (True, False)


def _sweep_for_timeseries(signals, labels, fs, f_max):
    """All TFD * log * normalize combos for time-domain inputs."""
    rows = []
    for tfd, log_on, norm in product(TFD_METHODS, LOG_CHOICES, NORM_CHOICES):
        if tfd in ("cwt", "hht") and not log_on and norm == "none":
            # Unnormalized magnitudes are wildly heavy-tailed; skip.
            continue
        try:
            fisher, lda = _evaluate_recipe(
                signals, labels, tfd_method=tfd, normalize=norm,
                log_compress_on=log_on, fs=fs, f_max=f_max,
            )
        except ImportError as e:
            print(f"  skipping {tfd}: {e}")
            break
        rows.append({
            "tfd": tfd, "log": log_on, "normalize": norm,
            "fisher": fisher, "lda": lda,
        })
    return rows


def _sweep_for_precomputed_stft(signals, labels):
    """The TFD is already applied; only log and normalize remain."""
    rows = []
    for log_on, norm in product(LOG_CHOICES, NORM_CHOICES):
        from tremor.spectral import log_compress, per_freq_zscore, per_recording_zscore
        feats = []
        for S in signals:
            X = log_compress(S) if log_on else S
            if norm == "per_recording":
                X = per_recording_zscore(X)
            elif norm == "per_freq":
                X = per_freq_zscore(X)
            feats.append(X.mean(axis=-1))
        feats = np.stack(feats)
        rows.append({
            "tfd": "precomputed", "log": log_on, "normalize": norm,
            "fisher": _fisher_ratio(feats, labels),
            "lda": _lda_loo_accuracy(feats, labels),
        })
    return rows


# ---------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------


def _print_table(source: str, rows: list[dict]) -> None:
    rows = sorted(rows, key=lambda r: (r["lda"], r["fisher"]), reverse=True)
    print(f"\n=== DATA SOURCE: {source} ===")
    print(f"  {'TFD':12s} {'log':>5s} {'normalize':>16s}  "
          f"{'Fisher':>8s}  {'LDA_LOO':>8s}")
    print("  " + "-" * 60)
    for r in rows:
        print(f"  {r['tfd']:12s} {str(r['log']):>5s} {r['normalize']:>16s}  "
              f"{r['fisher']:>8.3f}  {r['lda']:>7.1%}")


def _recommend(all_results: dict[str, list[dict]], data_root: Path,
               action: str) -> None:
    """Print the overall best recipe and the matching train.py command."""
    best_source, best_row = None, None
    for source, rows in all_results.items():
        for r in rows:
            if best_row is None or (r["lda"], r["fisher"]) > (best_row["lda"], best_row["fisher"]):
                best_source, best_row = source, r

    if best_row is None:
        print("\nNo evaluable recipes found.")
        return

    print("\n" + "=" * 65)
    print("RECOMMENDATION")
    print("=" * 65)
    print(f"Best: source={best_source}  tfd={best_row['tfd']}  "
          f"log={best_row['log']}  normalize={best_row['normalize']}")
    print(f"  Fisher = {best_row['fisher']:.3f}   "
          f"LDA-LOO accuracy = {best_row['lda']:.2%}")
    print()
    print("Reproduce in training with:")
    parts = [
        "python -m tremor.train",
        f"--data-root {data_root}",
        f"--action {action}",
    ]
    if best_source in ("quaternion", "synthetic_quaternion_av"):
        parts.append("--data-mode quaternion")
    elif best_source == "raw_amplitudes":
        parts.append("--data-mode raw --feature filtered_amplitudes --apply-bandpass")
    elif best_source == "precomputed_stft":
        parts.append("--data-mode stft")
    else:
        parts.append(f"# unknown source: {best_source}")

    if best_row["tfd"] in ("stft", "cwt"):
        parts.append(f"--tfd-method {best_row['tfd']}")
    if not best_row["log"]:
        parts.append("--no-log-compress")
    parts.append(f"--normalize {best_row['normalize']}")
    parts.extend(["--f-max 15", "--model bilstm"])
    print("  " + " \\\n    ".join(parts))


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark preprocessing recipes and report which one separates "
            "Normal / PD / ET best on a labelled set."
        )
    )
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--action", default="OUT",
                        help="Action folder to evaluate (OUT, REST, DRINK, "
                             "EAT, FNF, WING).")
    parser.add_argument("--fs", type=float, default=100.0)
    parser.add_argument("--f-max", type=float, default=15.0)
    parser.add_argument("--synthetic", action="store_true",
                        help="Fall back to a synthesised PD/ET/N benchmark "
                             "even if real data is available.")
    args = parser.parse_args()

    print(f"Tremor preprocessing benchmark")
    print(f"  data root: {args.data_root}")
    print(f"  action:    {args.action}")
    print(f"  fs:        {args.fs}  Hz, f_max = {args.f_max} Hz")

    sources = _detect_sources(args.data_root, args.action)
    if args.synthetic or not sources:
        if not sources:
            print("  no real data found; using synthetic dataset")
        else:
            print("  --synthetic flag set; using synthetic dataset")
        sigs, labs = _synthetic_dataset()
        print(f"  loaded synthetic: {len(sigs)} recordings  "
              f"classes={dict(zip(*[a.tolist() for a in np.unique(labs, return_counts=True)]))}")
        all_results = {
            "synthetic_quaternion_av": _sweep_for_timeseries(
                sigs, labs, fs=args.fs, f_max=args.f_max,
            )
        }
    else:
        print(f"  detected: {list(sources)}\n")
        all_results: dict[str, list[dict]] = {}
        for source in sources:
            try:
                if source == "quaternion":
                    sigs, labs = _load_quaternion(args.data_root, args.action, args.fs)
                    rows = _sweep_for_timeseries(sigs, labs, fs=args.fs,
                                                 f_max=args.f_max)
                elif source == "raw_amplitudes":
                    sigs, labs = _load_amplitudes(args.data_root, args.action)
                    rows = _sweep_for_timeseries(sigs, labs, fs=args.fs,
                                                 f_max=args.f_max)
                elif source == "precomputed_stft":
                    sigs, labs = _load_precomputed_stft(args.data_root, args.action)
                    rows = _sweep_for_precomputed_stft(sigs, labs)
                else:
                    continue
                print(f"  {source}: {len(sigs)} recordings  "
                      f"classes={dict(zip(*[a.tolist() for a in np.unique(labs, return_counts=True)]))}")
                all_results[source] = rows
            except Exception as e:
                print(f"  {source}: SKIPPED ({type(e).__name__}: {e})")

    for source, rows in all_results.items():
        _print_table(source, rows)

    _recommend(all_results, args.data_root, args.action)


if __name__ == "__main__":
    main()
