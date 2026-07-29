"""Deep biomarker characterization for the N/PD/ET differential.

Produces, from the interpretable feature set in ``tremor.biomarker``:
  1. Per-class, per-condition band-power / dominant-frequency statistics with
     Kruskal-Wallis (3-class) and Mann-Whitney (PD-vs-ET) tests + effect sizes.
  2. Per-frequency PSD breakdown by class and condition (figure).
  3. The rest-vs-action **contrast** biomarker (the physiological discriminator).
  4. Harmonic-structure analysis.
  5. A transparent leave-one-patient-out classifier (LDA + RandomForest) with a
     subject-level bootstrap CI, plus feature importances — the interpretable
     counterpart to the deep model.

Torch-free. Usage:
    python -m tremor.biomarker_analysis --data-root Data \
        --actions OUT,REST,WING --output reports/biomarker
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import kruskal, mannwhitneyu
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from tremor.biomarker import (
    FEATURE_NAMES, TREMOR_HI, TREMOR_LO, mean_band_psd, recording_features,
)
from tremor.data import CLASS_NAMES
from tremor.evaluate import classification_report
from tremor.quaternion_data import load_quaternion_recordings_multi
from tremor.stats import bootstrap_subject_ci

CLASS_COLORS = {"N": "#2c7fb8", "PD": "#d95f02", "ET": "#1b9e77"}


def _rank_biserial(a, b):
    """Effect size for Mann-Whitney (0=no effect, ±1=complete separation)."""
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    u, _ = mannwhitneyu(a, b, alternative="two-sided")
    return float(2 * u / (len(a) * len(b)) - 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True, type=Path)
    p.add_argument("--actions", default="OUT,REST,WING")
    p.add_argument("--fs", type=float, default=100.0)
    p.add_argument("--output", type=Path, default=Path("reports/biomarker"))
    args = p.parse_args()
    actions = [a.strip() for a in args.actions.split(",") if a.strip()]
    fig_dir = args.output / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    recs = load_quaternion_recordings_multi(
        args.data_root, actions=actions, fs=args.fs, mode="angular_velocity",
    )
    name_of = {i: n for i, n in enumerate(CLASS_NAMES)}

    # --- per-recording features + PSD --------------------------------------
    grid = np.arange(TREMOR_LO, TREMOR_HI + 1e-9, 0.25)
    rows = []          # dicts: patient, cond, cls, **features
    psd_by = defaultdict(list)   # (cls, cond) -> list of psd on grid
    for r in recs:
        feats = recording_features(r.x, fs=args.fs)
        cls = name_of[r.y]
        rows.append({"patient": r.subject, "cond": r.condition, "cls": cls, **feats})
        _, psd = mean_band_psd(r.x, fs=args.fs, grid=grid)
        psd_by[(cls, r.condition)].append(psd)

    report: dict = {"n_recordings": len(rows), "actions": actions}

    # --- (1) descriptive stats per condition -------------------------------
    key_feats = ["dom_freq", "et_pd_ratio", "harmonic_ratio", "rel_b_5_7",
                 "rel_b_7_10", "spec_entropy", "pow_total"]
    stats_out = {}
    for cond in actions:
        sub = [r for r in rows if r["cond"] == cond]
        by_cls = {c: [r for r in sub if r["cls"] == c] for c in CLASS_NAMES}
        cond_stats = {}
        for feat in key_feats:
            vals = {c: np.array([r[feat] for r in by_cls[c]]) for c in CLASS_NAMES}
            groups = [v for v in vals.values() if len(v) > 1]
            kw_p = float(kruskal(*groups).pvalue) if len(groups) == 3 else float("nan")
            pd_et_p = float("nan"); eff = float("nan")
            if len(vals["PD"]) > 1 and len(vals["ET"]) > 1:
                pd_et_p = float(mannwhitneyu(vals["PD"], vals["ET"],
                                             alternative="two-sided").pvalue)
                eff = _rank_biserial(vals["PD"], vals["ET"])
            cond_stats[feat] = {
                "median": {c: float(np.median(vals[c])) if len(vals[c]) else None
                           for c in CLASS_NAMES},
                "kruskal_p_3class": kw_p,
                "mannwhitney_p_PD_vs_ET": pd_et_p,
                "effect_PD_vs_ET": eff,
            }
        stats_out[cond] = cond_stats
    report["per_condition_stats"] = stats_out

    # --- (2) per-frequency PSD breakdown figure ----------------------------
    fig, axes = plt.subplots(1, len(actions), figsize=(5 * len(actions), 4),
                             sharey=True)
    if len(actions) == 1:
        axes = [axes]
    for ax, cond in zip(axes, actions):
        for c in CLASS_NAMES:
            stack = psd_by.get((c, cond))
            if not stack:
                continue
            m = np.mean(np.log1p(np.stack(stack)), axis=0)
            ax.plot(grid, m, label=f"{c} (n={len(stack)})", color=CLASS_COLORS[c])
        ax.set_title(f"{cond}: mean log-PSD (hand)")
        ax.set_xlabel("Hz"); ax.legend(fontsize=8)
    axes[0].set_ylabel("log(1+PSD)")
    fig.tight_layout(); fig.savefig(fig_dir / "psd_by_class_condition.png", dpi=110)
    plt.close(fig)

    # --- (3) rest-vs-action contrast biomarker -----------------------------
    # per-patient: mean pow_total & dom_freq per condition, then REST vs action.
    pat_cond = defaultdict(dict)   # patient -> cond -> (pow_total, dom_freq, cls)
    for r in rows:
        d = pat_cond[r["patient"]].setdefault(r["cond"], {"pow": [], "dom": [],
                                                           "cls": r["cls"]})
        d["pow"].append(r["pow_total"]); d["dom"].append(r["dom_freq"])
    action_cond = "WING" if "WING" in actions else ("OUT" if "OUT" in actions else None)
    contrast = {c: [] for c in CLASS_NAMES}
    dom_shift = {c: [] for c in CLASS_NAMES}
    for pid, cd in pat_cond.items():
        if "REST" in cd and action_cond in cd:
            cls = cd["REST"]["cls"]
            rest_pow = np.mean(cd["REST"]["pow"]); act_pow = np.mean(cd[action_cond]["pow"])
            contrast[cls].append(np.log((rest_pow + 1e-9) / (act_pow + 1e-9)))
            dom_shift[cls].append(np.mean(cd[action_cond]["dom"]) - np.mean(cd["REST"]["dom"]))
    if action_cond:
        cr = {c: [float(np.median(contrast[c])) if contrast[c] else None,
                  len(contrast[c])] for c in CLASS_NAMES}
        pd_et_c_p = (float(mannwhitneyu(contrast["PD"], contrast["ET"]).pvalue)
                     if len(contrast["PD"]) > 1 and len(contrast["ET"]) > 1 else None)
        report["rest_action_contrast"] = {
            "definition": f"log(power_REST / power_{action_cond}), hand sensor",
            "median_by_class": cr, "mannwhitney_p_PD_vs_ET": pd_et_c_p,
        }
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        data = [contrast[c] for c in CLASS_NAMES]
        ax[0].boxplot(data, tick_labels=CLASS_NAMES)
        ax[0].axhline(0, ls="--", c="gray", lw=1)
        ax[0].set_title(f"log(REST/{action_cond}) tremor power"); ax[0].set_ylabel("log ratio")
        ax[1].boxplot([dom_shift[c] for c in CLASS_NAMES], tick_labels=CLASS_NAMES)
        ax[1].axhline(0, ls="--", c="gray", lw=1)
        ax[1].set_title(f"dom-freq shift ({action_cond} - REST)"); ax[1].set_ylabel("Hz")
        fig.tight_layout(); fig.savefig(fig_dir / "rest_action_contrast.png", dpi=110)
        plt.close(fig)

    # --- (5) interpretable LOO classifier (single-condition OUT) -----------
    clf_cond = "OUT" if "OUT" in actions else actions[0]
    # per-patient feature vector = mean of their clf_cond recordings
    pat_feats = defaultdict(list); pat_label = {}
    for r in rows:
        if r["cond"] != clf_cond:
            continue
        pat_feats[r["patient"]].append([r[f] for f in FEATURE_NAMES])
        pat_label[r["patient"]] = r["cls"]
    patients = sorted(pat_feats)
    X = np.array([np.mean(pat_feats[p], axis=0) for p in patients])
    y = np.array([CLASS_NAMES.index(pat_label[p]) for p in patients])
    subj = np.array(patients)

    clf_report = {}
    for cname, ctor in [("lda", lambda: LinearDiscriminantAnalysis()),
                        ("rf", lambda: RandomForestClassifier(
                            n_estimators=300, class_weight="balanced",
                            random_state=0))]:
        # leave-one-patient-out: every patient predicted once
        preds = np.empty(len(patients), dtype=int)
        for i in range(len(patients)):
            tr = np.ones(len(patients), bool); tr[i] = False
            pipe = make_pipeline(StandardScaler(), ctor())
            pipe.fit(X[tr], y[tr])
            preds[i] = pipe.predict(X[i:i + 1])[0]
        # metrics via one-hot "logits" so we can reuse classification_report
        onehot = np.eye(len(CLASS_NAMES))[preds]
        rep = classification_report(np.log(onehot + 1e-6), y, CLASS_NAMES)
        ci = bootstrap_subject_ci(y, preds, subj, CLASS_NAMES, n_boot=2000, seed=0)
        clf_report[cname] = {
            "macro_f1": rep["macro_f1"],
            "per_class_f1": {c: rep["per_class"][c]["f1"] for c in CLASS_NAMES},
            "ET_f1_ci": {"point": ci["ET"].point, "lo": ci["ET"].lo, "hi": ci["ET"].hi},
            "macro_f1_ci": {"point": ci["macro_f1"].point, "lo": ci["macro_f1"].lo,
                            "hi": ci["macro_f1"].hi},
            "confusion_matrix": rep["confusion_matrix"],
        }
    report["interpretable_classifier"] = {"condition": clf_cond, **clf_report}

    # --- feature importance (RF fit on all patients) + LDA coefficients -----
    rf = make_pipeline(StandardScaler(),
                       RandomForestClassifier(n_estimators=400,
                                              class_weight="balanced", random_state=0))
    rf.fit(X, y)
    imp = rf.named_steps["randomforestclassifier"].feature_importances_
    order = np.argsort(imp)[::-1]
    report["feature_importance_rf"] = [
        {"feature": FEATURE_NAMES[i], "importance": float(imp[i])} for i in order
    ]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh([FEATURE_NAMES[i] for i in order][::-1], imp[order][::-1], color="#555")
    ax.set_title(f"RandomForest feature importance ({clf_cond})")
    fig.tight_layout(); fig.savefig(fig_dir / "feature_importance.png", dpi=110)
    plt.close(fig)

    # --- (4) harmonic + dom_freq boxplots (clf_cond) -----------------------
    sub = [r for r in rows if r["cond"] == clf_cond]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for j, feat in enumerate(["dom_freq", "harmonic_ratio"]):
        ax[j].boxplot([[r[feat] for r in sub if r["cls"] == c] for c in CLASS_NAMES],
                      tick_labels=CLASS_NAMES)
        ax[j].set_title(f"{feat} ({clf_cond})")
    fig.tight_layout(); fig.savefig(fig_dir / "dom_freq_harmonics.png", dpi=110)
    plt.close(fig)

    (args.output / "biomarker_report.json").write_text(json.dumps(report, indent=2))
    print("=== interpretable classifier (leave-one-patient-out) ===")
    for cn, r in clf_report.items():
        print(f"  {cn.upper():>4}: macroF1={r['macro_f1']:.3f} "
              f"ET_F1={r['per_class_f1']['ET']:.3f} "
              f"[{r['ET_f1_ci']['lo']:.3f},{r['ET_f1_ci']['hi']:.3f}]")
    print("=== top features (RF) ===")
    for e in report["feature_importance_rf"][:6]:
        print(f"  {e['feature']:>14}  {e['importance']:.3f}")
    print(f"\nfigures -> {fig_dir}/   report -> {args.output}/biomarker_report.json")


if __name__ == "__main__":
    main()
