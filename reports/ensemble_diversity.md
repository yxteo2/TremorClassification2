# Two fifths of patients are a coin flip, and that is where every method dies

## What this was for

`pooling_rules.md` found that **no combination rule beats the arithmetic mean** —
geometric, median, trimmed and temperature-scaled pooling all landed within
±0.012 macro precision of it, every interval spanning zero. That report first
attributed the null to the six ensemble members being near-copies: three seeds
per family differ only in weight initialisation, so perhaps every rule was
averaging over nothing.

**That explanation was wrong.** This measures it instead of assuming it, and the
answer turns out to say something much more useful about the ceiling than about
pooling.

10 splits, the same six members as the reported model (2 families × 3 seeds),
measured on the test fold. `python -m experiments.ensemble_diversity`.

## The members disagree a great deal

| | value | sd |
|---|---|---|
| r(p(ET)) all pairs | 0.859 | 0.044 |
| r within family | 0.928 | 0.031 |
| r across families | 0.812 | 0.057 |
| **argmax disagreement, all pairs** | **0.205** | 0.028 |
| disagreement within family | 0.156 | 0.028 |
| disagreement across families | 0.237 | 0.035 |
| mean sd of p(ET) across members | 0.066 | 0.015 |

One pair of members in five labels a given patient differently, and the two
architectures disagree more than two seeds do (0.237 vs 0.156) — as they should,
since they see different inputs. The ensemble is **not** six copies. There is
ample material for a pooling rule to reorder, and reordering it changed nothing.

## Where the disagreement sits — the row that matters

Split the test fold by whether all six members agree:

| | fraction | accuracy | top-2 margin |
|---|---|---|---|
| **unanimous** | 0.595 | **0.688** | 0.369 |
| **contested** | 0.405 | **0.485** | 0.094 |

*(Accuracy here is the raw argmax of the pooled probabilities, without the
validation-tuned priors, so it is a diagnostic and not the reported operating
point.)*

**Accuracy on contested patients is 0.485. Always guessing PD scores 0.465**
(188 of 404). The 40 % of patients the ensemble argues about are, collectively,
almost uninformative — the pooled model does about as well on them as a constant
prediction. Their top-2 margin is 0.094, four times narrower than on unanimous
patients: they sit directly on the decision boundary.

That resolves the pooling null. A pooling rule can only change the ranking where
members disagree, and where they disagree the answer is near chance. **Reordering
those patients changes which errors are made, not how many.** Every rule must
therefore tie, which is exactly what was measured.

## The consequence, which is bigger than pooling

The model decomposes cleanly into two populations:

* **60 % of patients: confident and 69 % correct.**
* **40 % of patients: contested and 48 % correct — a coin flip on the boundary.**

This predicts, in advance, that **any method which only reshuffles the contested
set cannot improve the headline.** That family is large, and much of it has
already been tested here and failed exactly as this now explains:

| method | result | reshuffles only? |
|---|---|---|
| pooling rules (7 arms) | all null | yes |
| temperature calibration | null | yes |
| prior objective (7 objectives) | nothing beats macro F1 | yes |
| difficulty pruning | significantly **worse** | no — it moves the boundary |
| influence pruning | no better than random | no |

The three post-hoc arms are all null and the two training-set arms are both
harmful. The post-hoc levers are exhausted, and this says why in one number.

**Improvement has to move patients out of the contested set**, which means a
representation that separates them — not a better decision rule over the
representation we have. It is also worth noting that even the *unanimous* 60 %
are wrong 31 % of the time, so the ceiling is not solely a boundary problem.

## Standing

* **The six members are genuinely diverse** (20.5 % pairwise disagreement,
  r = 0.859). Do not describe the 3-seed average as a variance-reduction
  formality; it is combining real disagreement.
* **40.5 % of patients are contested, and on them the model is at 0.485 against
  a 0.465 constant-prediction baseline.** This is the single most compact
  statement of the ceiling in the project and belongs in the writeup.
* **Post-hoc decision rules are done.** Any future arm that is a pure transform
  of the ensemble outputs should be expected to tie, and should be added to
  `pooling_rules.md` rather than given its own experiment.
* This **lowers the prior on `balanced_bagging`** — recorded before that run
  landed. Bagging adds diversity of a kind already present in quantity and
  demonstrably not the binding constraint.
* Untested follow-up worth having: whether contested patients are concentrated in
  a cohort, a class, or a tremor-severity band. If they cluster, that is a
  handle; if they are spread evenly, the ceiling is intrinsic to the task at this
  sample size.
