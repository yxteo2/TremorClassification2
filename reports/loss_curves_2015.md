# Loss curves of the 2015 transfer model, and the checkpoint rule they suggested

Run: `REPS=0-5 python -m experiments.loss_curves_2015` then `... plot`
(figure: `reports/figures/loss_curves_2015.png`); checkpoint rule:
`python -m experiments.checkpoint_rule_2015 report`.

The logged loop reproduces the `ft` model exactly (asserted). 150 curves per
stage (5 repeats x 5 folds x 2 members x 3 seeds).

## What the curves show

* **Two-stream pretraining overfits PADS.** Its loss on 2015 (never trained on)
  is lowest at pretraining epoch ~18 (1.035) and rises to 1.183 by epoch 200
  while the PADS loss keeps falling. The TCN's bottoms near epoch 76 and stays
  flat. (`transfer_2015.md` found pretraining 100 epochs mixed: precPD +0.021 \*,
  AUC −0.020 \*; 400 worse.)
* **Fine-tuning does not overfit by loss**: validation and test loss fall, then
  are flat from ~epoch 40.
* **So the best-validation-loss checkpoint is near-random**: chosen epoch IQR
  12-62 (two-stream) and 15-76 (TCN), bimodal.
* **Loss and the decision metric disagree**: TCN test macro-F1 peaks at epochs
  10-20 (0.564) and falls to 0.537 while its loss still improves.

## Checkpoint rule (one trajectory, weights snapshotted)

| arm | repeats 5-39: AUC / top-5 ET / macroP | fresh 100-139: AUC / top-5 ET / macroP |
|---|---|---|
| ckpt (= `ft`) | 0.628 / 0.491 / 0.604 | 0.633 / 0.470 / 0.605 |
| last (both epoch 80) | −0.021 \* / −0.006 / −0.012 | −0.017 \* / +0.075 \* / −0.006 |
| two-stream ckpt + TCN epoch 20 | **+0.012 \* / +0.046 \* / +0.010** | **+0.017 \* / +0.090 \* / +0.011 \*** |

**Not a confirmed result.** Epoch 20 was read off *test* curves; in 5-fold CV
every patient is a test patient in every repeat, so excluding repeats 0-4 and
re-scoring on fresh partitions still evaluates on patients whose labels shaped
the choice (audit, 2026-10-05). The gain is biased upward, by an unknown but
probably small amount (only a couple of discrete options were compared). The
clean test chooses the epoch inside each training fold (pooled inner-CV curves,
then refit on train + validation) -- not run.

The "last epoch" arms show the two-stream member's checkpoint does matter:
taking its final epoch costs AUC (−0.017 to −0.021 \*).
