# Contested patients are the slow ones — a uniform effect, not class confusion

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

## The part that is not circular

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

This one cannot be class confusion, because class confusion produces opposing
signs — a patient made ambiguous by looking like another class must move *toward*
that class's typical value, and N, PD and ET have different typical frequencies
(PADS max frequency: N 7.20, PD 7.07, ET 6.16 Hz). A descriptor that predicts
contestedness in the *same direction* for all three classes is describing
something that degrades the representation uniformly, not something that blurs
one class into another.

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
* **The frequency-location effect is not circular** and is the one result here
  worth building on: lower mean frequency, more contested, in every class, with
  `mean_freq` the strongest within-class association measured (−0.226).
* Untested follow-ups in order: separate the SNR account from the cycle-count
  account by matching on recording length; check whether widening or shifting the
  3–15 Hz band changes the contested rate for low-frequency patients
  specifically. Both are representation changes, which is the only route
  `contested_gating.md` left open.
