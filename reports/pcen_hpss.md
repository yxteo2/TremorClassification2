# PCEN and HPSS: the physics is confirmed, the separation is unnecessary, PCEN is harmful

## The gap these close

`audio_techniques.py` brought four ideas from the audio literature and closed
them all. On PCEN it recorded why it could only do half the method:

> PCEN beats the fixed pointwise log of a log-mel front-end. **Its gain-control
> term needs a time axis**, so what carries over is the per-band trainable
> exponent.

Every transform here collapses its time–frequency surface with `P.mean(0)`
before anything downstream sees it, so the two strongest audio front-ends —
**PCEN** (Lostanlen et al., *IEEE SPL* 26(1) 2019) and **harmonic–percussive
source separation** — were unreachable rather than refuted. Both act on the time
axis; both are implemented here directly and verified against signals with known
answers before use.

At the reported hop (0.64 s) a PADS recording yields only **13 frames**, too few
for a median filter along time, so the surface is resampled to a 0.16 s hop
(49 frames on PADS, 81 on 2015). **That resampling gets its own arm**, and a
**percussive arm** is the attribution control: if the transient part classifies
as well as the sustained part, the physics story is wrong.

## Result

| arm | precN | precPD | precET | macroP | macroF1 | recET | nETpred |
|---|---|---|---|---|---|---|---|
| **reported (hop 0.64 s)** | 0.641 | 0.647 | 0.642 | 0.643 | 0.590 | 0.465 | 8.75 |
| dense hop 0.16 s | 0.643 | **0.660** | 0.639 | **0.647** | 0.592 | 0.485 | 8.85 |
| dense + PCEN | 0.641 | 0.577 | 0.409 | 0.542 | 0.511 | 0.315 | 7.95 |
| dense + HPSS harmonic | 0.639 | 0.634 | **0.660** | 0.644 | **0.593** | 0.445 | 7.30 |
| dense + HPSS percussive | **0.661** | 0.620 | 0.523 | 0.601 | 0.553 | 0.325 | 7.00 |

**Adoption — paired vs the reported model:**

| arm | precET | macroP |
|---|---|---|
| dense hop alone | −0.002 [−0.044, +0.032] | +0.004 [−0.011, +0.018] |
| **PCEN** | **−0.233 [−0.351, −0.121]** * | **−0.101 [−0.146, −0.058]** * |
| **HPSS harmonic** | +0.018 [−0.056, +0.093] | +0.001 [−0.027, +0.031] |
| **HPSS percussive** | **−0.119 [−0.226, −0.009]** * | **−0.042 [−0.079, −0.002]** * |

**Attribution — paired vs the dense-hop control**, which isolates the front-end
from the resampling:

| arm | precET | macroP |
|---|---|---|
| PCEN | **−0.230 [−0.363, −0.115]** * | **−0.105 [−0.152, −0.062]** * |
| HPSS harmonic | +0.021 [−0.050, +0.093] | −0.003 [−0.030, +0.024] |
| HPSS percussive | **−0.117 [−0.219, −0.009]** * | **−0.046 [−0.082, −0.008]** * |

The dense hop alone is null (+0.004 macroP), so nothing below is the resampling.

## The physics prediction held, and the control earned its place

Recorded before the run: **harmonic > dense control > percussive on precET**,
the ordering being the claim rather than the magnitude.

    dense + HPSS harmonic    precET 0.660
    dense hop 0.16 s         precET 0.639
    dense + HPSS percussive  precET 0.523

Exactly that, and the percussive arm is **significantly worse than the control
it is matched against** (precET −0.117 *, macroP −0.046 *). So the separation
does what its physics says: **the class information lives in the sustained,
tonal component, and the broadband transient component carries significantly
less of it.** That is the first direct confirmation in this project that tremor
discriminability is concentrated in the sustained part of the recording rather
than in movement transients.

Had the percussive arm matched the harmonic one, the separation would have been
doing something other than what it claims. It did not, so the claim stands.

## But separating it buys nothing

HPSS-harmonic against the reported model is precET +0.018 [−0.056, +0.093] and
macroP +0.001 [−0.027, +0.031]; against its own dense-hop control, +0.021 and
−0.003. Null either way, on every column.

The likely reason, offered as an explanation of a shape rather than a tested
claim: **the time-average is already a weak harmonic–percussive separator.** A
transient occupies a few frames out of 49, so `P.mean(0)` already divides its
contribution by the frame count, while a sustained oscillation contributes in
every frame. HPSS removes explicitly what averaging has mostly removed
implicitly. The model was never seeing much transient energy to begin with.

That also explains why `pads_onset_trim.md` found the same thing from the other
direction: an artifact that is real, class-ordered, and measurable in the raw
signal still changed nothing once the pipeline had averaged over time.

## PCEN fails, significantly — and a cheap diagnostic saw it coming

PCEN is the largest negative measured: precET **−0.233 [−0.351, −0.121]** *,
macroP **−0.101 [−0.146, −0.058]** *, and it wins on macroP in **3 of 20
splits**.

The recorded prediction said "small and uncertain in sign". **That was wrong** —
it is large and significantly negative. But before the run, a label-free
diagnostic on 40 real recordings (`_pcen_alpha_diagnostic.py`) measured the
mechanism and anticipated it:

    arm            entropy (1 = flat)   peak/mean
    no PCEN              0.9051            3.08
    alpha = 0.98         0.9920            1.14

PCEN flattens the spectrum monotonically in alpha, removing the peak almost
entirely at the published default. The mismatch is conceptual, not a tuning
problem: for a stationary band `E / M^alpha → E^(1-alpha)`, which at 0.98 is
`E^0.02`. **In audio the discriminative information is *when* energy appears in
a band and the band's own average is background to divide out; here it is
*which band* has energy, and dividing each band by its own average destroys
exactly that.** No alpha adds structure — the family interpolates between "no
PCEN" and "fully flattened".

**This is an exception to standing rule #5 worth recording.** That rule says
descriptor-level gains do not compose to the model, measured three times. It is
*asymmetric*: a descriptor-level **destruction** composed perfectly here. The
diagnostic predicted the direction and the rough magnitude, and the reasoned
prediction did not. Cheap label-free measurements of what a transform does to
the representation are worth running before the fits.

## Standing

* **Do not use PCEN.** Significantly worse on precET, precPD, macroP and macroF1
  against both the reported model and the matched control, for a structural
  reason that no hyperparameter fixes.
* **Do not adopt HPSS**, but keep the finding. The separation is null for
  adoption while confirming that class information sits in the sustained
  component — worth stating in a writeup, and worth revisiting only if a future
  cohort has recordings noisy enough that averaging no longer suppresses
  transients on its own.
* **The dense 0.16 s hop is free** (macroP +0.004, null). If anything ever needs
  the time axis, resampling to it costs nothing.
* **A descriptor-level catastrophe does compose**, even though descriptor-level
  gains do not. Run the label-free diagnostic first; it cost minutes and
  correctly called the largest negative in this report.
