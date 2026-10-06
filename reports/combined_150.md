# The combined 2015 system on the corrected table: OUT + REST + WING raises PD precision; NewData trades ET for ranking

Run: `python -m experiments.combined_150 check`, `REPS=0-20 python -m
experiments.combined_150`, `python -m experiments.combined_150 report`
(150-patient table, 20 repeats; own `ft` asserted bit-identical to
`common.protocol.train`; agent-written, run resumed and report completed by the
coordinator after the agent hit an API limit).

Arms (OUT models branch from shared PADS-pretrained weights; REST / WING are
2015 scratch models; fusion = weighted mean of log-probabilities, 0.5 OUT /
0.25 REST / 0.25 WING, a term dropped for patients without that task):

* `ft` -- PADS pretrain -> 2015 fine-tune;
* `ft_nd` -- PADS pretrain -> fine-tune on 2015 + all NewData;
* `sys_ft` -- `ft` + 0.25 REST + 0.25 WING;
* `sys` -- `ft_nd` + 0.25 REST + 0.25 WING.

## Standard decision (macro-F1 offsets on validation)

| arm | precN | precPD | precET | macroP | macroF1 | PD-vs-ET AUC | top-5 ET | ET calls |
|---|---|---|---|---|---|---|---|---|
| ft | 0.732 | 0.710 | 0.329 | 0.590 | 0.581 | 0.613 | 0.420 | 14.0 |
| ft_nd | 0.718 | 0.712 | 0.303 | 0.578 | 0.566 | 0.655 | 0.360 | 16.6 |
| **sys_ft** | 0.750 | 0.777 | **0.342** | **0.623** | **0.610** | 0.641 | 0.440 | 13.9 |
| sys | **0.760** | **0.790** | 0.282 | 0.611 | 0.601 | **0.675** | 0.500 | 17.8 |

| contrast | repeat level | patient level |
|---|---|---|
| sys_ft − ft | precPD +0.067 \*, macroP +0.033 \*, macroF1 +0.029 \*, AUC +0.029 \*, precET +0.013 | **precPD +0.067 [+0.029, +0.106] \***; others n.s. |
| sys − ft | precPD +0.081 \*, precN +0.028 \*, AUC +0.062 \*, macroP +0.020 \*, **precET −0.047 \*** | **precPD +0.081 [+0.042, +0.119] \***; AUC [−0.010, +0.137] |
| ft_nd − ft | AUC +0.043 \*, precN −0.014 \* | **AUC +0.043 [+0.002, +0.089] \*** |
| sys − sys_ft | AUC +0.034 \*, **precET −0.060 \*** | AUC [+0.001, +0.071] \*, **precET [−0.114, −0.017] \*** |

## High-confidence rule (nested, request 0.90)

| class | ft | sys_ft | sys |
|---|---|---|---|
| N: held-out precision (recall) | 0.837 (0.30) | **0.848 (0.42)** | 0.815 (0.34) |
| PD | 0.856 (0.21) | **0.867 (0.37)** | 0.861 (0.42) |
| ET | no flags in 20 / 20 repeats | < 1 flag | < 1 flag |
| tremor vs none | 0.893 (0.71) | **0.900 (0.81)** | 0.901 (0.79) |

`sys` − `ft` under the rule: PD recall **+0.207** (patient level [+0.157,
+0.259] \*) at unchanged precision; tremor recall +0.086 (patient [+0.043,
+0.135] \*).

## Verdict

* **Best system for precision: `sys_ft` = `ft` (OUT) + 0.25 x REST + 0.25 x
  WING.** On the corrected table it raises PD precision by +0.067 -- the gain
  that survives patient resampling -- and macroP / macroF1 by ~0.03 at the
  repeat level, with ET precision unchanged (0.342). Under the high-confidence
  rule it keeps PD precision ~0.87 and N ~0.85 while flagging ~40 % of each
  class instead of ~20-30 %.
* **NewData pooling trades ET precision for ranking** inside the fused system
  (precET −0.060, AUC +0.034, both significant at patient level). Use it for
  PD-vs-ET ranking (`sys`: AUC 0.675), not for precision.
* ET precision is unchanged by every component; no ET >= 0.80 claim is possible.

## Prediction (in the docstring)

Held: `ft_nd` AUC +0.03-0.05 at both levels (+0.043); high-confidence PD >= `ft`
at higher recall. Failed (#49): `sys` precET within ±0.04 n.s. (−0.047 \*);
`sys` AUC surviving at patient level (n.s.); `sys` − `sys_ft` precision n.s.
(precET −0.060 \*, patient level).
