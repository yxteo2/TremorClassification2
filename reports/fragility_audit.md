# Which surviving claims are in the same position as the one that just failed

`prune_training.md` was retracted after a re-run: with the experiment's own code
**bit-identical**, every significant column reversed sign and went null once the
axis / Q-factor / IF-trajectory fixes were in. The retracted effect was
**macroP −0.032**, against the ~0.04 that 20 splits resolves.

That is not a property of that experiment. It is a property of **any small
significant result measured before 2026-09-02**, and this audit asks how many
there are. Reproduce with `_fragility_audit` in the shell history, or by the
recipe below.

## Method

A claim is at risk if all three hold:

1. its report was **last touched before `603b5b4`** (the axis fix, 2026-09-02),
   so its numbers were produced on the defective pipeline;
2. it is asserted as **significant** (an interval excluding zero);
3. the effect is **under 0.04**, i.e. at or below what 20 splits resolves.

Exposure is broad because the three fixes touched `m_multitaper`/`m_sst` (the
axis), `describe()` (the Q-factor, feeding `DESC`) and the IF trajectory
(`TRAJ`) — and the reported model consumes all three. Any experiment built on
the reported model inherits all of them.

## Result

**32 pre-fix reports carry a significant claim; 25 of those have one under
0.04.** Restricting to claims that are *load-bearing* — the ones
`closed_families.md` cites as the reason a family is closed — and reading the
magnitude the family row **actually asserts** rather than the smallest in the
report:

| closed family | asserted effect | at risk? |
|---|---|---|
| **estimator sharpness sweep** | ar16 −0.031 \*, welch −0.024 \* | **yes, both** |
| **catch22 hybrid** | AUC +0.014 \*, precET −0.028 \* | **yes, both** |
| **short-window (0.64 s) spectrum** | macroP −0.033 \* | **yes** |
| **cohort-ID input** | precN +0.024 \* | **yes** |
| TCN on the raw waveform | −0.034 \*, −0.076 \*, precET −0.192 \* | the −0.034 arm only |
| PADS pretrain → finetune | precET −0.188 \* | no — far above the band |
| log-frequency binning | precET −0.086 \* | no |
| average non-postural tasks | precN +0.047 \*, precET −0.104 \* | no |

**Four families rest entirely on effects inside the fragility band**, and one
rests partly on one.

## The one re-run, and why this one

`estimator_smoothing.md` is first in the queue because it is the most
load-bearing of the four: it is the evidence that **nw 2.5 is an interior
optimum**, which `SKILL.md` states as settled ("every component of that recipe
has now been swept and sits at an interior optimum"). Both of its significant
arms sit in the band.

One arm has independent post-fix corroboration and one does not, which is worth
stating before the result arrives:

* **welch −0.024 \*** is corroborated. The post-fix headline audit measures
  welch against the reported model at **macroP +0.044 [+0.020, +0.068] \* over
  40 splits** — larger, tighter, and after all three fixes. If the re-run keeps
  welch behind, that is consistent with an already-verified number.
* **ar16 −0.031 \*** has no such corroboration. It is the arm actually on trial.

## The queue, if the fragility proves systematic

In descending order of how much rests on them:

1. `estimator_smoothing.md` — the interior-optimum claim *(running)*
2. `tf_window_length.md` — short-window macroP −0.033 \*; also the source of
   failed prediction #5, the composition rule
3. `catch22_waveform_features.md` — both asserted effects in the band
4. `cohort_id_input.md` — precN +0.024 \*, and the only positive result among
   the four
5. `time_domain_deep.md` — only its −0.034 arm; the −0.076 and −0.192 stand

## What this does not say

It does **not** say these results are wrong. It says they were measured on a
pipeline with three known defects, at effect sizes below what the protocol
resolves, and that the one case tested so far did not survive. That is a reason
to re-run before building on them — not a reason to withdraw them. Withdrawal
requires a re-run, as `prune_training.md` got.

It also does not cover the largest category of risk, which is unquantifiable
here: **claims whose reports were never re-touched and whose experiments no
longer exist in runnable form.** `prune_training.py` was only re-runnable
because the audit restored its arms first.
