# Classical baseline: max + mean frequency only

Step 1 of the "classical features → AI features" narrative. Only the **maximum
(peak)** and **mean (spectral centroid)** frequency in the 3–15 Hz band,
computed per-channel (no rectification), subject-grouped 5-fold, balanced-class
logistic regression.

## N vs Tremor — the 70% target is met with 2 features

| config | #feats | accuracy | balanced acc | AUC |
|---|---|---|---|---|
| OUT, lower_arm | **2** | **0.703** | **0.709** | 0.800 |
| OUT, all 3 sensors | 6 | 0.716 | 0.720 | 0.809 |
| **OUT+REST+WING, lower_arm** | 6 | **0.781** | **0.787** | **0.871** |
| OUT+REST+WING, all sensors | 18 | 0.774 | 0.774 | 0.838 |

Majority-class baseline = 0.596. **Max+mean frequency alone reaches 70% with two
features and 78% using the three conditions.**

## PD vs ET — classical frequency features are at chance

| config | balanced acc | AUC |
|---|---|---|
| OUT, lower_arm | 0.488 | 0.527 |
| OUT, all 3 sensors | 0.438 | 0.476 |
| OUT+REST+WING, lower_arm | 0.519 | 0.582 |
| OUT+REST+WING, all sensors | 0.427 | 0.578 |

Chance = 0.50 balanced accuracy. **Max/mean frequency carries essentially no
PD-vs-ET information** — consistent with the distribution analysis (PD and ET
share the same median dominant frequency, 6.64 Hz; distribution overlap 0.5–0.7;
PD-vs-ET tests non-significant, p ≈ 0.25–0.47).

*(Note: raw accuracy is meaningless on this axis — the cohort is 125 PD vs 29 ET,
so "always PD" scores 0.833. Balanced accuracy and AUC are the honest metrics.)*

## The classical → learned comparison

| axis | classical (max+mean freq) | learned/AI features | contribution |
|---|---|---|---|
| **N vs Tremor** | 0.78 acc, AUC 0.871 | 0.854 acc, **AUC 0.937** | refinement |
| **PD vs ET** | **AUC 0.582 (chance)** | **AUC 0.800** | **signal where classical analysis has none** |

This is the core result for the AI-feature narrative: classical spectral analysis
solves the easy axis but **completely fails** on the clinically hard PD-vs-ET
differential, where learned features recover AUC 0.80 from chance. The natural
follow-up for explainable AI is: *what do the learned features capture that
frequency does not?* — current evidence points to spatial/propagation structure
and envelope (amplitude-modulation) properties.
