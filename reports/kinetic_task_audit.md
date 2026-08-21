# Lever #3 audited: the kinetic-task claim is a hypothesis, not a measurement

## Why this was checked

The skill file listed, as the third-ranked route to better ET performance:

> "PADS's 8 unextracted tasks — the kinetic ones (DrinkGlas, TouchNose) are where
> ET separates best (NewData DRINK AUC 0.812 vs 0.20–0.27 at REST)."

That recommendation has driven decisions, including an attempt this session to
download the remaining PADS tasks. But its evidence is a single AUC from
**NewData, which has 6 ET patients** — and `permutation_null.md`, written much
later, established that the permutation null for PD-vs-ET AUC at 6 ET spans
**[0.195, 0.819]**. The cited 0.812 sits inside it.

The claim predates the machinery needed to test it. Run:
`python -m experiments.kinetic_task_audit` — every NewData task, both axes,
permutation null refitting the whole pipeline per replicate, 200 permutations.

## PD vs ET — the axis the claim is about (6 ET)

| task | | n | AUC | null 95 % | p |
|---|---|---|---|---|---|
| REST | | 31 | 0.307 | [0.179, 0.820] | 0.294 |
| OUT | | 29 | 0.225 | [0.195, 0.819] | 0.134 |
| **DRINK** | K | 29 | **0.804** | [0.203, 0.819] | **0.050** |
| **FINGER_NOSE** | K | 30 | **0.826** | [0.173, 0.840] | 0.060 |
| POUR | K | 33 | 0.593 | [0.209, 0.790] | 0.622 |
| TAP | K | 32 | 0.449 | [0.192, 0.757] | 0.826 |
| PRON_SUP | K | 33 | 0.494 | [0.154, 0.784] | 0.975 |

**Neither survives multiplicity.** Seven tasks were tested, so a Bonferroni
threshold is 0.05 / 7 = 0.007. DRINK at p = 0.050 and FINGER_NOSE at p = 0.060 are
exactly the borderline results this project has repeatedly retracted.

The reproduction is faithful — DRINK measures 0.804 here against the 0.812 quoted,
and REST/OUT measure 0.307 / 0.225 against the quoted 0.20–0.27. **The numbers
were right; the inference from them was not.**

## N vs Tremor — the axis that is actually powered (29–33 positives)

| task | | AUC | null 95 % | p |
|---|---|---|---|---|
| **OUT** | | **0.840** | [0.308, 0.720] | **0.005** |
| FINGER_NOSE | K | 0.807 | [0.293, 0.676] | **0.005** |
| DRINK | K | 0.787 | [0.316, 0.679] | **0.005** |
| POUR | K | 0.781 | [0.303, 0.666] | **0.005** |
| PRON_SUP | K | 0.667 | [0.311, 0.654] | 0.045 |
| REST | | 0.652 | [0.306, 0.692] | 0.169 |
| TAP | K | 0.640 | [0.311, 0.662] | 0.134 |

On the axis with enough patients to measure anything, **the postural OUT task is
the best**, and the kinetic tasks are comparable rather than superior. The claim
"kinetic tasks are where the signal is" does not hold where it can be tested.

## The honest reading — and why the recommendation partly survives

Three things are true at once and the report would be wrong to collapse them:

1. **The claim as written is not established.** "The kinetic ones are where ET
   separates best" is stated as a measurement; at 6 ET, after multiplicity, it is
   not one. It must be restated as a hypothesis.
2. **The pattern is not random-looking.** The two hand-to-target tasks — DRINK and
   FINGER_NOSE — sit at AUC 0.804 and 0.826 while REST and OUT sit at 0.307 and
   0.225. That is a large, coherent separation between task *types*, and the two
   tasks that stand out are the two where a tremor that worsens on approach to a
   target would be expected to show. It is suggestive.
3. **Six ET patients cannot settle it, and nothing in this cohort can.** The null
   at 6 ET reaches 0.819. A task would have to be near-perfect to register.

So the *action* the lever recommends — extract PADS's kinetic tasks, which carry
**28 ET** instead of 6 — remains the right next step. What changes is its status:
it is a **test of an open hypothesis**, not the exploitation of a known effect.
That distinction matters for how the result should be described in a paper, and
for how disappointed to be if it comes back null.

## Blocked

`physionet.org` is denied by this environment's network policy (403 at the egress
proxy, confirmed in the proxy's own failure log), so the PADS kinetic tasks cannot
be fetched here. Testing this needs either a network policy that permits
PhysioNet, or the archive uploaded to the repo as `pads_stretchhold` and
`pads_relaxed` were.

## What this says about the reports generally

This is a claim that sat in the project's top-three recommendations, was quoted
with a specific number, and was reproduced exactly — and it still did not mean
what it said. The number was never wrong. What was missing was the null, and the
null did not exist yet when the claim was made.

Worth applying the same check to any other single-cohort NewData claim: at 6 ET
the PD-vs-ET null reaches 0.819, so **no PD-vs-ET result from NewData alone can
be evidence of anything.**
