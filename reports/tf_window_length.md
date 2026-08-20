# Window length matters, but the gain belongs to logistic regression

## The gap

Every transform in `signal_processing/transforms.py` computes a time-frequency
surface and then collapses it to the **frequency marginal** before anything
downstream sees it:

```python
f, P = welch(x, ...);  return _band(f, P.mean(0))          # m_welch
P = _per_freq_mean(S, n_freq, n_ch, square=True)            # m_multitaper
```

A project whose stated line of work is *time-frequency* processing has been using
only the frequency axis, and every downstream family — descriptors, harmonics,
axis shape — inherits that loss. Separately, **window length was never swept**:
every transform is pinned at nperseg 256 or 512, i.e. 2.6 s or 5.1 s, which is
20–60 cycles of a 4–12 Hz tremor.

## What was built

Fixed summary statistics over the discarded time axis — not a learned surface,
since `time_domain_deep.md` had just shown learned time-axis models lose:

    median (16)   per-bin median across frames
    iqr    (16)   per-bin inter-quartile range
    flux    (1)   mean L1 change between consecutive frames
    wander  (1)   sd of the per-frame peak frequency, Hz

Each frame normalised to unit power first, so everything describes spectral
*shape* and its movement, never amplitude. Windows 64 / 128 / 256 / 512.

Validated on synthetics before use: a single stable 6 Hz pacemaker gives IQR 0.42
/ flux 0.073 / wander 0.00 Hz, a signal switching between 5 and 7 Hz states gives
1.47 / 0.195 / 0.91. Rotation and scale invariance to 1e-11.

## The variability hypothesis fails; window length is what matters

The features were built to test the "several oscillators vs one pacemaker"
mechanism. Those are the **weakest** arms everywhere — `iqr` at 2.56 s scores AUC
0.494 on PADS, chance — and adding them to the median *dilutes* it (0.761 vs
0.825), the sixteenth feature union to underperform its best member.

What appeared instead was a clean monotone effect of window length on PADS:

| window | 0.64 s | 1.28 s | 2.56 s | 5.12 s |
|---|---|---|---|---|
| median-block AUC | **0.825** | 0.775 | 0.764 | 0.716 |

On MERGED the same sweep is flat (0.694 / 0.667 / 0.689 / 0.694), so the
monotone trend is PADS-specific.

## Three explanations, separated before believing any of them

Readings were written down before the control ran, because the two previous
causal stories this session were asserted rather than tested and both turned out
wrong.

| arm | PADS AUC | MERGED AUC |
|---|---|---|
| A multitaper, 16 bins (current) | 0.798 | 0.675 |
| B multitaper, **8 bins** | 0.818 | 0.671 |
| C short-window **median**, 16 | 0.825 | 0.694 |
| D short-window **mean**, 16 | **0.830** | **0.701** |

* **D ≥ C → "robust estimation" is refuted.** A mean over frames does as well as
  a median, so resistance to transients is not the mechanism. This was my first
  explanation and it is wrong.
* **B ≈ C on PADS → coarseness explains most of the PADS gain.** Going from 16 to
  8 multitaper bins recovers 0.798 → 0.818 on its own. That is this project's
  already top-ranked lever ("coarse-bin to 16–32 bins") pushed further, not a new
  finding.
* **B ≈ A on MERGED → coarseness explains nothing there**, yet D still gains
  +0.026. On the merged cohort a genuine short-window effect remains.

## Paired, and significant — for logistic regression

30 repeats, every arm scored on the same folds.

**PADS PD vs ET**

| arm | dim | AUC | precET | macroP |
|---|---|---|---|---|
| A multitaper 16 | 16 | 0.789 | 0.398 | 0.668 |
| B multitaper 8 | 8 | 0.811 | 0.429 | 0.685 |
| **D short-window mean 16** | 16 | **0.826** | **0.486** | **0.717** |

D vs A: AUC **+0.036 [+0.031, +0.042]** *, precET **+0.088 [+0.073, +0.102]** *,
macroP **+0.049 [+0.040, +0.056]** *.
D vs B — the short window beyond coarseness — is significant on all four columns
(AUC +0.015 *, precET +0.057 *).

**MERGED PD vs ET**: D vs A gives AUC +0.030 *, precET +0.032 *, macroP +0.018 *.
But D vs B is **AUC only** (+0.033 *) with precET −0.001 and macroP −0.001 — there
the precision gain is coarseness and the short window adds ranking alone.

## And it reverses in the deep model

The test that decides whether it matters: swap only the spectrum input of the
reported two-stream model, keeping descriptors, asymmetry, trajectory,
architecture, priors and folds fixed. 30 splits, raised in advance.

| arm | precN | precPD | precET | macroP |
|---|---|---|---|---|
| A reported (multitaper 16) | 0.655 | 0.652 | **0.658** | **0.655** |
| B short-window mean 16 | 0.660 | 0.628 | 0.578 | 0.622 |

paired: **macroP −0.033 [−0.057, −0.007] \***, precET −0.080 [−0.153, +0.000].
B loses on **77 % of splits** for macro precision.

**That framing was wrong**, and the follow-up says so.

## Resolving the confound: it is the TASK, not the model

The two results above differ in **task** as well as model, so the deep arm was
re-run on the binary axis — `Spectrum1DCNN` alone, tremor patients only, folds
shared across arms (`experiments/shortwindow_binary_deep.py`, 20 repeats).

| paired short-window − multitaper | PADS AUC | PADS precET | MERGED AUC | MERGED precET |
|---|---|---|---|---|
| logistic regression | **+0.034** * | **+0.033** * | **+0.030** * | **+0.030** * |
| Spectrum1DCNN | **+0.016** * | −0.001 | +0.007 | **+0.013** * |

**The CNN gains too**, at roughly half the magnitude, and significantly on
PADS AUC and MERGED precET. So "helps linear models, hurts deep ones" is not
supported. The short-window spectrum is better for **PD-vs-ET under both model
families**; what it hurts is the **3-class merged** model.

## And the obvious mechanism for that is refuted as well

The natural explanation was that heavy smoothing at 1.56 Hz resolution blurs the
N-vs-Tremor boundary, which leans on peak sharpness (`peak_sharp` is the standout
descriptor: ET 12.19, PD 5.80, N 4.08). Tested directly with logistic regression
on all 590 patients:

| representation | N-vs-Tremor AUC | peak sharpness N / PD / ET |
|---|---|---|
| multitaper 16 | 0.774 | 0.98 / 1.51 / 2.03 |
| short-window 16 | **0.810** | 0.71 / 1.19 / 1.68 |

**Short-window is better at N-vs-Tremor too.** Sharpness is compressed in absolute
terms but its class ordering and separation survive. So that explanation fails.

## What is actually established, and what is not

Established, all paired:

* short-window helps **logreg** on PD-vs-ET, both cohorts (AUC +0.030 to +0.034 *,
  precET +0.030 to +0.033 *);
* it helps **logreg** on N-vs-Tremor (0.774 → 0.810);
* it helps the **binary CNN**, about half as much;
* it **hurts the reported 3-class two-stream model** (macroP −0.033 *, losing on
  77 % of splits).

**Why the 3-class model loses is unexplained.** Four mechanisms were proposed and
tested against data today, and all four failed: robust estimation over frames
(a mean does as well as a median), demodulation cost and median-centring in the
time-domain study, and N-vs-Tremor blurring here. A fifth guess is not offered.

That run of four is itself worth recording: **on this dataset my mechanistic
explanations have a poor track record, and the discipline that keeps paying is
running the control rather than writing the sentence.** Each of those four would
have gone into a report as an assertion if it had not been tested.

Remaining candidates, none tested: the 3-class model consumes the spectrum twice
(inside `TwoStreamNet` alongside descriptors and trajectory, and alone in
`ResidualTCN`) and soft-votes them, so the loss may live in an interaction rather
than in the representation; or in the validation-tuned priors, which are fitted on
three classes.

## Two bugs caught in the first run

Both would have produced a confident wrong answer.

1. **All-zero features for PADS and NewData.** At short windows the native
   resolution gives fewer than 16 native bins in 3–15 Hz, so 16 log-bins could not
   be formed at all; frames are now interpolated onto a fixed 64-point grid, which
   decouples output bins from window length — the whole point of sweeping it. And
   at nperseg 512 with 50 % overlap a 1024-sample PADS recording yields 3 frames
   against a 4-frame minimum; overlap is now 75 %.
2. **The coverage print hid it.** It reported a count taken after the window loop,
   so it described only the last window. It now reports per window and warns below
   99 %.

## Standing

* **Do not swap the reported model's spectrum for the short-window one** —
  significantly worse (macroP −0.033 *).
* **Do consider it for linear models on PD-vs-ET**, where it is the best
  representation measured: PADS AUC 0.826 / precET 0.486 against multitaper's
  0.789 / 0.398, at equal dimensionality.
* **Spectral variability features are settled negative** — weakest arms at every
  window on both cohorts, and they dilute the median when appended.
* **Window length is worth carrying as a knob.** It was never swept, and it moves
  PADS PD-vs-ET AUC by 0.11 across the range tested.
