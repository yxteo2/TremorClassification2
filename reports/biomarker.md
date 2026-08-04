# Interpretable tremor biomarkers — N/PD/ET

A transparent, physiologically-motivated feature analysis to sit alongside the
deep model. Features are computed from the **hand** sensor angular velocity
(`tremor/biomarker.py`); analysis and figures from `tremor/biomarker_analysis.py`.
All tests are on per-recording features unless noted; the classifier is
per-patient, leave-one-patient-out.

## 1. Dominant tremor frequency — the physiology is in the data

Median dominant frequency (Hz) in the 3–15 Hz band, by class and condition:

| condition | N | PD | ET | 3-class KW | PD-vs-ET (MWU) |
|---|---|---|---|---|---|
| OUT (postural) | 8.2 | 7.0 | 6.2 | p=0.083 | p=0.65 (ns) |
| **REST** | 6.6 | **5.5** | **6.2** | **p=2.6e-10** | **p<0.001**, effect −0.41 |
| WING (action) | 3.5 | 4.7 | 5.5 | p=0.009 | p=0.95 (ns) |

**PD-vs-ET separates at REST, and essentially only at REST.** PD's rest tremor
is slower (5.5 Hz — the classic 4–6 Hz Parkinsonian band) and ET's is higher.
The three strongest PD-vs-ET separators are all REST features:

| REST feature | effect (rank-biserial) | p |
|---|---|---|
| dominant frequency | −0.41 | <0.001 |
| total tremor power | +0.37 | 0.002 |
| 7–10 Hz power fraction | −0.35 | 0.003 |

i.e. at rest, PD has **lower-frequency, higher-power** tremor with less high-band
content than ET — exactly the clinical picture.

## 2. The rest-vs-action contrast biomarker

Defined per patient as `log(tremor power at REST / tremor power at WING)` (hand):

| class | median log(REST/WING) | n |
|---|---|---|
| N | −0.96 | 61 |
| PD | −0.72 | 63 |
| **ET** | **−1.61** | 13 |

**ET is strongly action-dominant** (much less power at rest than in action); PD
is the least action-dominant (relatively more rest power). This is the
PD=rest / ET=action dichotomy made quantitative. PD-vs-ET Mann-Whitney p=0.057
— borderline, limited by only 13 ET patients having both conditions.

## 3. Interpretable classifier vs the deep model

Leave-one-patient-out on single-condition (OUT) per-patient features:

| model | macro-F1 | ET-F1 [95% CI] |
|---|---|---|
| LDA (13 features) | 0.522 | 0.231 [0.00, 0.44] |
| RandomForest | 0.524 | 0.143 [0.00, 0.32] |
| deep BiLSTM (threshold-tuned, ref) | ~0.63 | ~0.47 |

The interpretable model reaches macro-F1 ~0.52; the deep model's advantage
(~0.63) is real but modest and concentrated on ET. Top RandomForest features:
spectral entropy, 5–7 Hz (PD-core) power, peak power, 10–15 Hz (ET-high)
fraction, 3–5 Hz power, total power — all physiologically interpretable.

## 4. Actionable insight

All deep-model work so far used the **OUT** condition, but PD-vs-ET separates at
**REST** (dominant-frequency KW p=2.6e-10 vs 0.083 for OUT). **REST is likely the
better condition for the hard PD-vs-ET axis** — a concrete, testable next step:
re-run the ET-LOSO deep model on REST (and REST+contrast) rather than OUT.

## Figures
- `figures/psd_by_class_condition.png` — mean log-PSD by class, per condition.
- `figures/rest_action_contrast.png` — REST/WING power ratio & dom-freq shift.
- `figures/dom_freq_harmonics.png` — dominant frequency & harmonic ratio by class.
- `figures/feature_importance.png` — RandomForest feature importance.

Reproduce: `python -m tremor.biomarker_analysis --data-root Data --actions OUT,REST,WING --output reports/biomarker`
