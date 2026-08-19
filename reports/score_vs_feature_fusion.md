# Score-level fusion beats feature-level, but still loses to the best single family

**What prompted this.** After the one-class hybrid worked (`oneclass_hybrid.md`),
I wrote a rule into the skill file: *"combine at the score level when the models
differ in kind, at the feature level almost never."* That rule was inferred from
two successful unions and eight failed ones. This tests it directly, and **it is
too broad**.

Run: `python -m experiments.score_ensemble`. Seven feature families, one logistic
regression each, PD-vs-ET, 20 repeats. Combiners: uniform average of within-fold
ranks; ranks weighted by an inner-CV AUC fitted on the training fold only;
plain feature concatenation (the control); and the rank average with the
one-class PD-density score added as an eighth member.

## Results

| | PADS (28 ET) | in-house (21 ET) | MERGED (49 ET) |
|---|---|---|---|
| | AUC / precET | AUC / precET | AUC / precET |
| descriptors | 0.794 / 0.448 | 0.399 / 0.071 | 0.640 / 0.303 |
| spectrum | 0.792 / 0.391 | 0.530 / 0.195 | 0.667 / 0.280 |
| stability | 0.753 / **0.450** | 0.542 / 0.169 | 0.653 / 0.330 |
| axes | 0.539 / 0.150 | 0.613 / **0.302** | 0.622 / 0.245 |
| harmonics | 0.725 / 0.352 | 0.404 / 0.090 | 0.633 / 0.267 |
| ampmod | 0.700 / 0.327 | 0.440 / 0.107 | 0.669 / 0.291 |
| asymmetry | 0.735 / 0.332 | 0.430 / 0.142 | 0.662 / 0.224 |
| **concat ALL** (50-d) | 0.753 / 0.316 | 0.556 / 0.229 | 0.691 / 0.287 |
| **rank-avg ALL** | 0.788 / 0.388 | 0.470 / 0.164 | **0.718** / **0.340** |
| rank-avg, AUC-weighted | 0.791 / 0.418 | 0.537 / 0.221 | 0.700 / 0.331 |
| rank-avg + one-class | 0.784 / 0.405 | 0.453 / 0.157 | 0.713 / 0.338 |

Paired against the best single family (chosen with hindsight, which is a handicap
the ensemble does not get):

| combiner | PADS vs `stability` | in-house vs `axes` | MERGED vs `stability` |
|---|---|---|---|
| rank-avg ALL | precET **−0.063** * | precET **−0.138** * | precET +0.011, AUC **+0.065** * |
| rank-avg AUC-weighted | precET **−0.032** * | precET **−0.081** * | precET +0.001, AUC **+0.048** * |
| concat ALL | precET **−0.134** * | precET **−0.074** * | precET **−0.043** * |

## What the rule should actually say

**1. Score-level beats feature-level — usually, not always.** Rank-averaging beats
concatenation on PADS (precET 0.388 vs 0.316) and MERGED (0.340 vs 0.287), but
*loses* in-house (0.164 vs 0.229). "Score level always wins" is not supported.

**2. Neither beats the best single family within one cohort.** On PADS every
combiner is significantly below `stability`; in-house every combiner is
significantly below `axes`. Combination is not free: averaging in members that
are near chance on *this* cohort drags the good member down, and with 7 members
the good one carries only 1/7 of the weight.

**3. Combination pays exactly when no member dominates.** On MERGED — the only
group where the cohorts are mixed and therefore no family is reliably best —
`rank-avg ALL` is the best model in the table, with **AUC +0.065 [+0.059, +0.071]**
over the best family. Its precision advantage (+0.011) is not significant, so the
honest claim is *better ranking, not better precision*.

**4. AUC-weighting recovers about half the loss** where uniform averaging hurts
(PADS −0.063 → −0.032; in-house −0.138 → −0.081) and gives up a little where it
helps (MERGED AUC +0.065 → +0.048). Weighting is the safer default if a single
combiner must be picked, but it does not turn a losing combination into a winning
one.

**5. The one-class member adds nothing here.** It helped as one of *two* members
in `oneclass_hybrid.md` (+0.023 precET in-house) and is neutral-to-negative as
one of *eight*. The active ingredient in that result was two strong, dissimilar
members — not the one-class model itself.

So the corrected rule: **combination helps when the members are individually
comparable in strength and you cannot tell in advance which will win. It hurts
when one member dominates, and the more members you add the more it hurts.**

## An eleventh, twelfth and thirteenth instance of feature-level dilution

`concat ALL` (50 features) is worse than the best single family in all three
groups, decisively so on PADS (precET 0.316 vs 0.450). This is now the strongest
single demonstration of the standing finding: at 404 patients with 49 ET,
**dimensionality binds harder than information**.

## Scope

The in-house comparisons here are between models whose individual AUCs all sit
inside the permutation null for that cohort ([0.298, 0.655] at 21 ET —
`permutation_null.md`). The *paired* differences remain interpretable, because
pairing removes the fold noise the two arms share, but no row of the in-house
column should be read as evidence that the family concerned separates PD from ET.

The paired intervals are bootstrapped over repeats on a fixed patient set, so
they answer "does A beat B on these patients", not "would A beat B on new
patients".
