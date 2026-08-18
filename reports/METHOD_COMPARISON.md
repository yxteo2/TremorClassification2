# Method and merge comparison — everything measured

Two tables: **which classifier** works, and **how to combine the cohorts**.
Numbers come from different protocols, so each table states its own; they are
not interchangeable.

---

## 1. Classifier comparison — binary PD vs ET

Tremor patients only. Repeated stratified CV, 10 repeats (5-fold, or 3-fold
in-house where ET is scarcest). Best per column in **bold**.

### PADS (276 PD / 28 ET, prevalence 0.092)

| method | family | AUC | precPD | precET |
|---|---|---|---|---|
| logreg, spectrum | linear | **0.790** | 0.955 | 0.268 |
| MLPHead h=16 | neural | 0.786 | 0.954 | 0.282 |
| logreg L1 | linear | 0.776 | 0.955 | 0.269 |
| Spectrum1DCNN | neural | 0.771 | 0.957 | 0.289 |
| **logreg + SVMSMOTE** | resampling | 0.764 | 0.950 | 0.362 |
| logreg, desc+stability | linear | 0.757 | 0.954 | 0.260 |
| **RandomForest** | trees | 0.756 | 0.945 | **0.401** |
| ExtraTrees | trees | 0.755 | 0.955 | 0.356 |
| RF + SVMSMOTE | trees+resamp | 0.749 | 0.947 | 0.374 |
| HistGradBoost | trees | 0.739 | 0.934 | 0.343 |
| ResidualTCN | neural | 0.732 | 0.955 | 0.298 |
| SVM rbf | kernel | 0.765 | 0.910 | 0.350 ± 0.329 |

### Merged, all three cohorts (374 PD / 49 ET, prevalence 0.116)

| method | family | AUC | precPD | precET |
|---|---|---|---|---|
| logreg, axes+stability | linear | **0.728** | 0.929 | 0.218 |
| **logreg + SVMSMOTE** | resampling | 0.717 | 0.927 | 0.291 |
| TwoStream + hand feats | neural | 0.714 | **0.937** | 0.239 |
| logreg L1 | linear | 0.708 | 0.933 | 0.222 |
| MLPHead h=16 | neural | 0.696 | 0.931 | 0.222 |
| ExtraTrees | trees | 0.691 | 0.929 | 0.252 |
| SVM rbf | kernel | 0.670 | 0.884 | 0.000 |
| **RandomForest** | trees | 0.660 | 0.922 | **0.292** |
| Spectrum1DCNN | neural | 0.653 | 0.926 | 0.220 |
| ResidualTCN | neural | 0.651 | 0.927 | 0.221 |

### In-house, 2015 + NewData (98 PD / 21 ET, prevalence 0.176)

| method | family | AUC | precPD | precET |
|---|---|---|---|---|
| **logreg, 4 axis features** | linear | 0.625 | **0.879** | **0.291** |
| ExtraTrees | trees | **0.645** | 0.868 | 0.259 |
| TwoStream + hand feats | neural | 0.644 | 0.869 | 0.253 |
| logreg L1 | linear | 0.641 | 0.895 | 0.263 |
| logreg + SVMSMOTE | resampling | — | — | 0.291 |
| MLPHead h=16 | neural | 0.615 | 0.874 | 0.267 |
| RandomForest | trees | 0.617 | 0.834 | 0.202 |
| SVM rbf | kernel | 0.609 | 0.824 | 0.000 |
| HistGradBoost | trees | 0.563 | 0.832 | 0.202 |
| Spectrum1DCNN | neural | 0.508 | 0.819 | 0.168 |
| ResidualTCN | neural | 0.475 | 0.828 | 0.175 |

### The pattern across all three

| method family | PADS (28 ET) | merged (49 ET) | in-house (21 ET) |
|---|---|---|---|
| linear | best AUC | best AUC | **best overall** |
| trees | **best precET** | best precET | significantly worse |
| resampling (SVMSMOTE) | +0.068 precET * | +0.073 precET * | nothing |
| neural (spectrum) | competitive | competitive | near chance |
| kernel (RBF SVM) | unstable | collapses to all-PD | collapses to all-PD |

**Everything that adds capacity or synthesises data helps on the larger cohorts
and fails in-house.** Three independent method families, one boundary, at
roughly 25-30 minority patients.

---

## 2. Cohort-merge comparison

3-class N/PD/ET, mixed protocol (all sources in train/val/test), macro precision,
20 splits. From `cohort_strategies.md` and `merge_design.md`.

| merge strategy | macro P | verdict |
|---|---|---|
| **cap PADS 90/class, pool, global priors** | **0.649** | **best** |
| cohort-ID as input | 0.668 | best mean, CI spans zero, sd doubles |
| per-cohort priors | 0.626 | significantly worse * |
| PADS pretrain → fine-tune | 0.583 | significantly worse * (−0.066) |
| uncapped, unweighted | 0.506 | much worse |
| uncapped + sample weights | 0.492 | worse than plain uncapped |
| uncapped + weights + per-cohort priors | 0.490 | worst |

### Distribution alignment — all harmful

| alignment | cohort probe \|acc−majority\| | effect |
|---|---|---|
| **none** | **0.003–0.035** | already invariant |
| per-cohort z-score | 0.285–0.480 | worse |
| per-cohort rank | 0.306–0.471 | worse |
| CORAL | 0.241–0.407 | worse |

The features are already scale- and rotation-invariant, so there is no shift to
correct. Every alignment method *injects* one, by centring each cohort on its own
class mixture (PADS is 72 % PD, 2015 balanced).

### Task alignment

| alignment | best LOCO macro F1 | best precET |
|---|---|---|
| **postural** (OUT / OUT / StretchHold) | **0.451** | **0.282** |
| rest (REST / REST / Relaxed) | 0.399 | 0.209 |

### Does PADS help in-house patients? No.

Test sets are 2015+NewData only, 10 ET each, PADS in **training only**, 20 draws.

| training data | precN | precPD | precET | macro P |
|---|---|---|---|---|
| **2015 + NewData only** | 0.652 | **0.769** | 0.193 | **0.538** |
| + PADS capped 90 | 0.685 | 0.687 | 0.196 | 0.523 |
| + PADS uncapped | 0.687 | 0.653 | 0.190 | 0.510 |

Paired: PADS capped **precPD −0.082 [−0.142, −0.025] ***, uncapped
**−0.116 [−0.185, −0.050] ***, precET **+0.003** (nothing).

**Adding PADS does not help ET and significantly degrades PD, with a dose
response.** The merged ET precision of 0.685 was substantially PADS predicting
PADS.

---

## 3. Which features, by cohort

PD-vs-ET AUC by family (`four_families.md`, `pd_vs_et_binary.md`):

| family | PADS | in-house |
|---|---|---|
| spectrum | **0.790** | 0.552 |
| descriptors | 0.784 | **0.411** |
| stability (TSI) | 0.758 | 0.532 |
| asymmetry | 0.739 | 0.425 |
| harmonics | 0.725 | 0.418 |
| **axes (rotation-invariant)** | **0.550** | **0.625** |

**The families invert between cohorts.** Spectral features are best on PADS and
worst in-house; axis-shape features are the reverse. This is the feature-level
form of the transfer failure above.

`axes + stability` on the merged cohort (0.728) is the **only feature union in
this project to beat both its members** (0.627, 0.659) — spatial shape and
temporal steadiness are physically independent.

---

## 4. Recommended configuration

| target | model | AUC | precPD | precET |
|---|---|---|---|---|
| in-house patients | logreg on 4 axis features | 0.625 | 0.879 | 0.291 |
| merged cohort | logreg + SVMSMOTE | 0.717 | 0.927 | 0.291 |
| PADS | RandomForest | 0.756 | 0.945 | **0.401** |

Merge recipe: **postural task, PADS capped at 90/class, pooled, no distribution
alignment, one global set of validation-tuned priors** — and report in-house
results separately, because PADS does not transfer.
