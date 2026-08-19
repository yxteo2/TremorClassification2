# Where the descriptors meet the TCN is worth more than any of them

**The question.** `TwoStreamNet` fuses by **late concatenation** — the spectrum
goes through a conv trunk, the descriptors through a small MLP, and the two
feature vectors meet immediately before the classifier:

```python
parts = [self.spec_feat_fn(self.spec, s), self.traj(t)]
if d is not None: parts.append(self.desc(d))
return self.head(torch.cat(parts, dim=1))
```

That is the arrangement in which the descriptors influence the spectrum
representation the *least* — they never touch it. The trunk extracts the same
features from a 4 Hz peak whether the patient's bandwidth is 1 Hz or 4 Hz,
because bandwidth only arrives after the trunk has finished.

Run: `python -m experiments.tcn_fusion`. Four integration points, same inputs,
same trunk depth, same trajectory stream concatenated at the head in every arm,
so only the descriptor–spectrum interface changes. Merged cohort, n=404, 20
splits.

## Result

| arm | params | precN | precPD | precET | macroP | macroF1 |
|---|---|---|---|---|---|---|
| late concat (current) | 7,731 | 0.633 | 0.655 | 0.648 | 0.645 | 0.596 |
| **early input channels** | 8,115 | **0.650** | 0.654 | **0.739** | **0.681** | 0.590 |
| FiLM conditioning | 8,963 | 0.645 | 0.647 | 0.699 | 0.664 | 0.596 |
| channel gate | 7,683 | 0.636 | 0.641 | 0.645 | 0.641 | 0.594 |

paired vs late concat:

| arm | precN | precET | macroP |
|---|---|---|---|
| **early input channels** | **+0.017 [+0.000, +0.035]** * | +0.091 [−0.003, +0.189] | **+0.036 [+0.004, +0.071]** * |
| FiLM conditioning | +0.012 | +0.051 [−0.050, +0.157] | +0.019 [−0.019, +0.057] |
| channel gate | +0.003 | −0.003 | −0.004 |

**Early fusion beats late concatenation significantly on macro precision.** ET
precision moves +0.091 with a lower bound of −0.003 — it misses significance by
the width of a rounding error, and the point estimate 0.739 is the highest ET
precision seen anywhere in this project.

## What "early channels" actually does

Each descriptor is broadcast to a constant value along the frequency axis and
concatenated as an extra **input channel**, so the first convolution sees

    channel 0    the log spectrum, varying across frequency
    channel 1..k each descriptor, constant across frequency

The first conv layer can therefore form products of a descriptor with a local
spectral pattern — "this peak shape, *given* that bandwidth" — which late
concatenation cannot represent at all, because by the time the descriptor is
available the spectral pattern has been reduced to a pooled vector.

## The prediction held, and it says why

The module docstring recorded a prediction before the run: thirteen feature
unions and every attention mechanism have failed here because they add
parameters that 49 ET patients cannot pay for; `early channels` roughly does not
add parameters, because the descriptor MLP branch disappears as the input widens.
*"If anything wins, the prediction is that it is that one."*

The parameter column supports the mechanism rather than just the guess:

* `channel gate` is the **cheapest** arm (7,683, below the baseline) and does
  nothing. So cheapness alone is not the explanation.
* `FiLM` is the **most expensive** (8,963) and lands in between — a real point
  estimate (+0.019 macroP) that does not reach significance, exactly the profile
  of a mechanism paying for its own parameters.
* `early channels` is +384 parameters over the baseline and wins.

The ordering is not by parameter count. It is by **how early the two information
sources can interact**: input (wins) > per-block modulation (partial) > after
pooling (nothing) > at the classifier (baseline). That is a cleaner statement of
the result than "early fusion is better", and it is the first architectural
change in this project to produce a significant gain.

## RESOLVED: it does not beat the reported model

`FusionTCN` in "late" mode is a **re-implementation**, not the reported
architecture: it uses a residual dilated trunk where the reported model's
spectrum stream is `Spectrum1DCNN`. Its late arm scores macroP 0.645 where the
reported model scores 0.660, so the +0.036 was measured from a lower starting
point.

`combined_best.py` and then `early_fusion_confirm.py` settled it:

| comparison | splits | macroP |
|---|---|---|
| vs late concat, matched trunk | 20 | **+0.036 [+0.004, +0.071]** * |
| vs reported model | 20 | +0.021 [−0.006, +0.048] |
| vs reported model | **40** | **+0.005 [−0.020, +0.028]** |

**The gain against the reported model is split noise** (`early_fusion_confirm.md`).
The `Spectrum1DCNN` trunk already captures what early fusion recovers in the
residual-TCN trunk, leaving nothing to add.

So the finding stands as an **architecture fact and not a model improvement**:
where descriptors meet a convolutional trunk matters, and matters more than what
the fusion costs in parameters. It is not a route to a better model on this
cohort.
