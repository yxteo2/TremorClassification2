# Contested patients are the slow ones — but the mechanism claim is retracted

## What was asked

`ensemble_diversity.md` found the contested 40 % concentrated by cohort (NewData
0.498 against 2015 0.356) and flagged the follow-up that mattered: whether they
cluster by a **physical** property rather than a provenance label. "NewData
patients are contested" suggests nothing to build. A signal property does.

30 splits, ~6 test appearances per patient, giving a per-patient **contested
rate** rather than a noisy flag; 395 of 404 patients qualify.
`python -m experiments.contested_profile`.

## Contestedness is predictable, and not from the class label

    descriptors -> often-contested    AUC 0.725  (sd 0.043)
    + true class label                AUC 0.722  (sd 0.029)
    descriptors -> contested RATE     Spearman rho +0.450 (out-of-fold)

Patient-level 5-fold CV, no patient in both train and test. **AUC 0.725 is well
clear of chance**, and adding the true class label changes nothing (0.722), so
this is not the class-prevalence effect in disguise.

## The caveat that has to come first

**Much of this is close to circular, and the report would be misleading without
saying so.** The descriptors that predict contestedness are descriptors the model
itself consumes. "The ensemble is unsure when its own inputs are ambiguous" is
nearly a restatement of what a decision boundary is, not a discovery about
tremor.

The within-class correlations show exactly that circular pattern, and the
**sign flip between N and ET** is its signature:

| descriptor | rho \| N | rho \| PD | rho \| ET | mean within |
|---|---|---|---|---|
| spectral_entropy | **−0.610** | −0.172 | **+0.119** | −0.221 |
| spectral_spread | **−0.607** | −0.148 | **+0.178** | −0.193 |
| freq_iqr | **−0.563** | −0.196 | **+0.293** | −0.155 |
| peak_share | **+0.499** | +0.104 | −0.178 | +0.142 |
| q_factor | +0.357 | +0.031 | −0.242 | +0.048 |
| total_power | +0.409 | +0.162 | −0.024 | +0.183 |

Read down the N and ET columns: **an N patient becomes contested when their
spectrum looks tremor-like** (low entropy, narrow, sharp, powerful), and **an ET
patient becomes contested when theirs looks normal** (broad, high-entropy, flat).
That is the model being confused by patients who resemble another class — true,
unsurprising, and not a handle.

## CORRECTION — the "not circular" argument below is wrong

**This section originally claimed the frequency effect could not be class
confusion. That argument does not hold, and the correction is recorded here
rather than by rewriting the section.**

The argument was: class confusion produces *opposing* signs, so a descriptor
correlating in the same direction in all three classes must be describing
something else. **That is only true when the class means straddle the
descriptor's range.** Here they are monotonically ordered — mean frequency N
8.16, PD 7.51, ET 7.04 Hz — so moving *down* in frequency moves an N patient
toward PD and ET, and a PD patient toward ET. Both acquire the same sign. ET,
already the slowest class, has nowhere further to move and should show no effect.

That is exactly what is measured, magnitudes included:

| class | mean frequency | within-class rho |
|---|---|---|
| N | 8.16 Hz | **−0.385** |
| PD | 7.51 Hz | −0.241 |
| ET | 7.04 Hz | **−0.051** |

The effect size falls monotonically with the class's own mean frequency and
vanishes for the slowest class. **That is the signature of class confusion, not
of uniform degradation**, which would predict comparable magnitudes in all three.

What survives, and it is not nothing: the pooled gradient is **much larger than
class composition alone explains**. Expected contested rate from each tercile's
class mix is 0.424 / 0.395 / 0.375 (spread 0.049) against an observed 0.515 /
0.416 / 0.253 (spread 0.262) — composition accounts for under a fifth of it. But
the remainder is consistent with *within-class* confusion, which is what the rho
column measures, so this does not rescue the original claim.

**Standing after the correction:** the frequency gradient is real, large and
reproducible, and there is no longer good evidence that it is anything other than
class confusion expressed through a monotone class ordering. Treat the section
below as the measurement, not the interpretation. The test that would separate
the two accounts is untried: whether contested rate tracks *distance toward the
nearest other class mean* — which for ET means **faster**, not slower — better
than it tracks raw frequency.

## The part that was claimed to be non-circular

Three descriptors break the pattern. **Every frequency-location descriptor is
negatively associated with contestedness in all three classes, with no sign
flip:**

| descriptor | rho \| N | rho \| PD | rho \| ET | mean within |
|---|---|---|---|---|
| **mean_freq** | −0.385 | −0.241 | −0.051 | **−0.226** |
| median_freq | −0.315 | −0.251 | −0.050 | −0.205 |
| max_freq | −0.072 | −0.174 | −0.216 | −0.154 |

**Lower tremor frequency means more contested, regardless of class.** `mean_freq`
is the strongest within-class association in the whole table (−0.226).

~~This one cannot be class confusion, because class confusion produces opposing
signs.~~ **Retracted — see the correction above.** The class means are
monotonically ordered (N 8.16, PD 7.51, ET 7.04 Hz), so confusion produces the
*same* sign for N and PD and none for ET, which is what is measured.

**Nor is it the cohort effect in disguise**, which is the other confound this
project would expect. Mean frequency is essentially identical across the three
cohorts, while contested rate is not:

| cohort | n | mean_freq (sd) | contested rate |
|---|---|---|---|
| 2015 | 151 | 7.714 (1.168) | 0.356 |
| NewData | 56 | 7.791 (1.104) | **0.498** |
| PADS | 197 | 7.706 (1.038) | 0.401 |

The cohorts differ in contested rate by a factor of 1.4 while differing in mean
frequency by 1 %. **Cohort and frequency are independent contributors.** If
anything the cohort effect works *against* the frequency effect here — NewData
has the highest mean frequency and the highest contested rate — so the
within-class frequency correlation is understated rather than inflated by it.

## What that points at, stated as a hypothesis and not a result

Slow tremor is where this representation is weakest, and there are physically
plausible reasons that are worth testing rather than assuming:

* **Voluntary motion and postural drift live at low frequency.** The analysis
  band starts at 3 Hz; a 3.5 Hz tremor sits much closer to that contamination
  than a 9 Hz one, so its effective SNR is worse at the same amplitude.
* **A slow oscillation completes fewer cycles** in a fixed-length recording, so
  every frequency and stability estimate from it rests on fewer periods.

Either would predict that contested patients cluster at the low end, which is
what is measured. Neither is established here. The distinguishing experiment is
straightforward and untried: **hold amplitude fixed and vary frequency**, or
compare contested rate against recording length at matched frequency, which
separates the SNR account from the cycle-count account.

## Standing

* **Contestedness is predictable from spectral descriptors at AUC 0.725**, and
  not via the class label.
* **Most of that signal is circular** — the sign-flipping descriptors are
  measuring "this patient resembles another class", which restates the decision
  boundary. Do not report those correlations as a finding about tremor.
* **The frequency-location effect is real but its interpretation is retracted.**
  Lower mean frequency does mean more contested, in every class, with `mean_freq`
  the strongest within-class association measured (−0.226), and the pooled
  gradient exceeds what class composition explains by a factor of five. But the
  within-class effect sizes fall monotonically with each class's own mean
  frequency and vanish for ET, which is the signature of class confusion. Do not
  build on this as a uniform representation failure.
* Untested follow-ups in order: separate the SNR account from the cycle-count
  account by matching on recording length; check whether widening or shifting the
  3–15 Hz band changes the contested rate for low-frequency patients
  specifically. Both are representation changes, which is the only route
  `contested_gating.md` left open.
