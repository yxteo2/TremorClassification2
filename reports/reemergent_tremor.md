# Re-emergent tremor: not measurable in these recordings

Parkinsonian rest tremor disappears when the arms are first held out and
**re-emerges after a latency of roughly 2-10 s**; essential tremor is postural
and present from the moment the posture is adopted. It is a well-established
clinical sign and a purely temporal one, which nothing in this project could see
-- the spectrum is averaged over the whole recording, `select_task_epoch` picks
the window with the MOST tremor power (skipping any latency), and the IF
trajectory normalises the envelope and resamples it.

`signal_processing/reemergence.py` measures it directly, in absolute time from
the start of each recording.

## Result: at chance

| cohort | n | ET | PD-vs-ET AUC, 5 features |
|---|---|---|---|
| 2015 OUT | 90 | 15 | 0.440 |
| PADS StretchHold | 304 | 28 | 0.504 |
| NewData OUT (unsegmented) | 29 | 6 | 0.543 |

## Why -- and this is the useful part

| onset_latency (s) | N | PD | ET |
|---|---|---|---|
| 2015 OUT | 0.000 | 0.126 | 0.000 |
| PADS StretchHold | 0.000 | 0.009 | 0.045 |

**Onset latency is essentially zero for every class.** The envelope reaches half
its median level within the first sample: the tremor is already present when the
recording starts. PADS StretchHold is uniformly *exactly* 10.24 s, which is a
fixed-length extract cut from a longer task rather than a recording that begins
when the arms go out.

**This is not evidence against re-emergent tremor.** It is evidence that these
recordings are not aligned to posture onset, so the phenomenon is invisible to
any method. No deep-learning architecture can recover information that is not in
the data -- a TCN over the envelope is the right tool for this signature and
would find nothing here.

## A weaker related signal that IS present

| | N | PD | ET | univariate AUC |
|---|---|---|---|---|
| `env_slope` | -0.129 | -0.073 | **-0.044** | 0.611 |
| `late_energy_frac` | 0.322 | 0.378 | **0.423** | 0.572 |

ET sustains its tremor across the recording while PD and N decay -- consistent
with ET being continuous and postural, and with ET's greater
instantaneous-frequency stability (`temporal_stability.md`). But the
multivariate combination is at chance (0.504), so it is not usable as it stands.

## What would make it testable

Recordings that **start at the moment the posture is adopted**, ideally 20-30 s
long. That is a data-collection specification, not a modelling one, and it is
cheap to add to any future protocol given how well established the clinical sign
is.

Reproduce: `signal_processing/reemergence.py`, `scratch/reemerge_test.py`.
