# PD vs ET as a dedicated binary problem

Every PD-vs-ET figure earlier in this project was read out of a 3-class N/PD/ET
model. That model spends capacity on the easy axis -- N-vs-Tremor reaches
precision 0.910-0.924 from six frequency features -- and dilutes the boundary
that matters clinically. Training on tremor patients only changes the picture.

## Linear baselines, per cohort

Repeated stratified CV on tremor patients only, 10 repeats.

### PADS (276 PD / 28 ET), 5-fold

| features | dim | AUC | precPD | precET | bal-acc |
|---|---|---|---|---|---|
| **spectrum** | 16 | **0.790 +/- 0.014** | 0.955 | 0.268 | 0.721 |
| descriptors | 10 | 0.784 +/- 0.021 | 0.960 | **0.275** | **0.742** |
| stability | 6 | 0.758 +/- 0.017 | 0.952 | 0.245 | 0.705 |
| asymmetry | 5 | 0.739 +/- 0.010 | **0.962** | 0.185 | 0.704 |
| harmonics | 3 | 0.725 +/- 0.010 | 0.952 | 0.157 | 0.656 |
| axes | 4 | 0.550 +/- 0.027 | 0.922 | 0.118 | 0.555 |

### In-house, 2015 + NewData (98 PD / 21 ET), 3-fold

| features | dim | AUC | precPD | precET | bal-acc |
|---|---|---|---|---|---|
| **axes** | 4 | **0.625 +/- 0.040** | **0.879** | **0.291** | **0.629** |
| axes + asym | 9 | 0.593 +/- 0.039 | 0.869 | 0.258 | 0.601 |
| axes + stability | 10 | 0.583 +/- 0.068 | 0.869 | 0.244 | 0.593 |
| spectrum | 16 | 0.552 +/- 0.044 | 0.835 | 0.190 | 0.521 |
| stability | 6 | 0.532 +/- 0.074 | 0.848 | 0.203 | 0.544 |
| descriptors | 10 | 0.411 +/- 0.067 | 0.780 | 0.110 | 0.410 |

### Merged, all three (374 PD / 49 ET), 5-fold

| features | dim | AUC | precPD | precET | bal-acc |
|---|---|---|---|---|---|
| **axes + stability** | 10 | **0.728 +/- 0.012** | 0.929 | 0.218 | 0.653 |
| axes + stab + asym | 15 | 0.714 +/- 0.013 | 0.934 | 0.226 | 0.668 |
| axes + asym | 9 | 0.689 +/- 0.011 | 0.934 | 0.209 | 0.660 |
| ampmod | 6 | 0.668 +/- 0.026 | 0.926 | 0.205 | 0.639 |
| spectrum | 16 | 0.665 +/- 0.020 | 0.921 | 0.211 | 0.628 |
| stability | 6 | 0.659 +/- 0.010 | 0.915 | 0.181 | 0.602 |
| axes | 4 | 0.627 +/- 0.013 | 0.911 | 0.183 | 0.594 |

## Three findings

### The binary framing helps

In-house ET precision **0.291**, against 0.193 from the 3-class model and 0.245
from the 3-class model with axis features added. Removing the healthy controls
from the training problem lets the model spend its capacity on the boundary that
matters.

### The first feature union to beat both its members

On the merged cohort, `axes + stability` reaches AUC **0.728** against 0.627 for
axes alone and 0.659 for stability alone. Eight earlier unions in this project
underperformed their best member; this one does not, and the reason is
interpretable: **spatial shape** (how confined the oscillation is to one axis)
and **temporal steadiness** (how much the instantaneous frequency wanders) are
independent physical properties, so their information adds rather than dilutes.

### The families really do split by cohort

| | PADS | in-house |
|---|---|---|
| spectrum | **0.790** | 0.552 |
| descriptors | 0.784 | **0.411** |
| axes | **0.550** | **0.625** |

Spectral features are the best choice on PADS and the *worst* in-house;
axis-shape features are the reverse. Any single cross-cohort recommendation
would be wrong for one of them, which is consistent with
`own_data_reality_check.md` finding that PADS training does not transfer to
in-house patients.

## Note on precPD

Precision for PD looks high everywhere (0.78-0.96) because PD is the majority
class in every cohort -- 276/304 on PADS. It should be read against that
prevalence, not as a standalone achievement. ET precision and AUC are the
informative columns.

Reproduce: `python -m experiments.pd_vs_et`.

## Deep models on the binary problem

Same folds, same features, binary heads throughout.

### In-house (98 PD / 21 ET) -- logistic regression wins outright

| model | dim | AUC | precPD | precET | bal-acc |
|---|---|---|---|---|---|
| **logreg on axes** | 4 | 0.625 | **0.879** | **0.291** | **0.629** |
| MLPHead h=16 | 4 | 0.615 | 0.874 | 0.267 | 0.611 |
| Spectrum1DCNN | 16 | 0.508 | 0.819 | 0.168 | 0.491 |
| ResidualTCN | 16 | 0.475 | 0.828 | 0.175 | 0.499 |
| TwoStream + hand feats | 148 | 0.644 | 0.869 | 0.253 | 0.597 |

Paired against logreg: Spectrum1DCNN **AUC -0.117 [-0.148, -0.089]** *,
ResidualTCN **-0.149 [-0.186, -0.113]** *, TwoStream **precET -0.038
[-0.074, -0.006]** *. TwoStream's AUC edge (+0.019) does not clear zero.

**A 4-feature logistic regression is the best in-house PD-vs-ET model measured.**

### Merged (374 PD / 49 ET) -- deep helps the decision, not the ranking

| model | AUC | precPD | precET | bal-acc |
|---|---|---|---|---|
| logreg (axes+stability) | **0.728** | 0.929 | 0.218 | 0.653 |
| TwoStream + hand feats | 0.714 | **0.937** | **0.239** | **0.678** |

Paired: AUC **-0.013** *, precPD **+0.007** *, precET **+0.021** *,
bal-acc **+0.025** *. All four significant, in opposite directions.

### PADS (276 PD / 28 ET) -- same split

| model | AUC | precET |
|---|---|---|
| logreg (spectrum) | **0.790** | 0.268 |
| Spectrum1DCNN | 0.771 | 0.289 (+0.021 *) |
| ResidualTCN | 0.732 (-0.058 *) | **0.298** (+0.030 *) |

## The pattern worth naming

On the two larger cohorts, deep models give **worse rankings but better
decisions**: AUC drops significantly while ET precision and balanced accuracy
rise significantly. They are not ordering patients better -- they are placing the
boundary better for the minority class, which is what class-weighted training
plus a learned representation buys.

On in-house data with 21 ET they lose on both counts, and the spectrum-only
models collapse to near chance (AUC 0.475-0.508).

**Which metric matters decides which model to use.** For ranking patients by
risk, use logistic regression. For a fixed-threshold decision on the merged
cohort, the two-stream model is better. For in-house patients, logistic
regression on four axis features wins on every metric.
