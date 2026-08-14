# Left-right limb asymmetry moves PD-vs-ET off chance

**Status: verified.** Survives repeated CV, a paired subject bootstrap, a
label-permutation null, and removal of the side-bias confound. This is the first
feature in this repo to beat chance on PD-vs-ET at meaningful n.

## Claim

Six -- and better, four -- descriptors of how the two wrists' spectra *differ*
reach AUC 0.730 on PADS StretchHold PD-vs-ET (n=304, 28 ET), where the
single-limb spectrum sits at 0.527 and 122 concatenated bilateral spectral bins
reach 0.605.

The clinical premise is standard neurology: Parkinsonian signs begin
unilaterally and remain asymmetric; essential tremor is typically more
symmetric. The contribution here is that stating it directly, as a handful of
between-limb dissimilarity measures, works -- while giving a network both
spectra and asking it to find the relationship does not.

## Features

`tfbench.small_nets.asym_feats`, computed from `bilateral_table` (patients x 2F,
left spectrum concatenated with right, each normalised to sum 1):

| name | definition | signed? |
|---|---|---|
| `corr` | correlation of the two mean-centred spectral shapes | no |
| `cos` | cosine similarity of the raw shapes | no |
| `peak_df` | \|peak-bin difference\| between limbs | no |
| `l1` | L1 distance between the two shapes | no |
| `log_peak_ratio` | log ratio of peak heights | **yes** |
| `log_power_ratio` | log ratio of total in-band power | **yes** |

## Results

PADS StretchHold, wrist gyro, 3-15 Hz, Welch-512 normalised spectra,
5-fold `StratifiedGroupKFold`, `StandardScaler + LogisticRegression(
class_weight="balanced")`. 20 CV seeds.

### Feature-set comparison (n=304, 28 ET)

| feature set | dim | bal-acc | AUC | prec | rec | F1 |
|---|---|---|---|---|---|---|
| single limb (limbs averaged) | 61 | 0.533 | 0.556 | 0.121 | 0.250 | 0.163 |
| concat `[left \| right]` | 122 | 0.556 | 0.605 | 0.156 | 0.250 | 0.192 |
| concat + asym | 128 | 0.549 | 0.554 | 0.158 | 0.214 | 0.182 |
| **asym only** | **6** | **0.696** | **0.709** | 0.183 | 0.714 | 0.292 |

### Stability and significance

```
asym-only  AUC 0.708 +/- 0.015   [min 0.678, max 0.740]   (20 seeds)
single     AUC 0.527 +/- 0.027   [min 0.479, max 0.567]
seeds where asym > single: 20/20
PAIRED subject bootstrap dAUC (asym - single): +0.183  [+0.031, +0.343]  SIGNIFICANT
```

On **N-vs-Tremor** the same comparison is null: dAUC +0.032 [-0.041, +0.104].
That specificity is corroborating rather than disappointing -- limb asymmetry
should separate PD from ET and should *not* separate tremor from no-tremor. A
feature that helped on both axes would look like a domain artifact.

### Is it asymmetry, or is it "which side"?

Two features are signed, and signed left-minus-right encodes limb *dominance*,
which is arbitrary across patients unless the cohort has a handedness bias
tracking diagnosis. PADS does have such a bias:

```
left-dominant fraction:  ET 0.179   PD 0.388
```

So this had to be ruled out. It is:

| feature set | dim | AUC |
|---|---|---|
| all 6 (signed, as first reported) | 6 | 0.708 |
| 6 with \|log ratios\| (magnitude) | 6 | 0.724 |
| **4 shape-only, both ratios dropped** | **4** | **0.730** |
| 2 signed ratios ONLY (side) | 2 | 0.604 |

Removing the side information entirely makes the model **better**. The effect is
genuine between-limb shape dissimilarity, not limb dominance. The side bias is
real but the finding does not rest on it, and the recommended feature set is the
4 unsigned ones.

### Protocol null

```
permutation null AUC 0.487 +/- 0.075 (max 0.615);  real 0.710;  empirical p = 0.000
```

5-fold CV at 28 positives is not leaking on its own.

## Why the 6 beat the 128

`concat+asym` (0.554) is *worse* than `asym only` (0.709). Adding 122
uninformative spectral dimensions buries 6 informative ones at 28 positives.

This is the mechanistic reason not to expect a bilateral transformer to help
here: attention over 2F frequency tokens operates in exactly that diluted
regime, with ~30 k parameters, and must rediscover the relationship from 28 ET
subjects. See `window_vs_patient_level.md` for the rest of that comparison.

## Caveats

* **Precision is still poor** (0.183 at threshold 0.5). The gain is recall-led
  (0.250 -> 0.714). For a usable classifier the threshold and the
  precision/recall trade still need work; this is a *separability* result.
* **Does not replicate on NewData** (PD-vs-ET, 6 ET): asym-only AUC 0.465 at
  OUT, 0.728 at DRINK but below the 0.912 of plain concat. At 6 ET nothing is
  measurable there, so this is neither support nor refutation.
* **PADS only**, one task (StretchHold). The obvious next test is the kinetic
  PADS tasks (DrinkGlas, TouchNose), where ET separates best in every other
  cohort.
* Requires **both limbs**, so it cannot be applied to the 2015 cohort at all.

## Consequence for data collection

The lever is not architecture, it is protocol: record both limbs. The bilateral
paper's real advantage over this repo's 2015 data is not its transformer, it is
that both wrists were instrumented.

Reproduce: `scratch/asym.py`, `scratch/asym_verify.py`, `scratch/asym_robust.py`
(gitignored; features live in `tfbench.small_nets.asym_feats`).
