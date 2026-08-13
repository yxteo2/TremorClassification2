# Deep models on the merged cohort — N-vs-Tremor

The axis where merging helps, and where there is actually enough data: **207
patients (88 N, 119 tremor)** at OUT, versus 29 for PD-vs-ET on DRINK. If deep
learning is going to beat the linear model anywhere in this project, here is
where it has the best chance.

Scored on the 151 2015 patients, so the numbers are comparable to the
2015-only baseline.

| model | bal-acc | AUC | precision | recall | F1 [95% CI] |
|---|---|---|---|---|---|
| **logreg, 10 descriptors** | **0.821** | **0.905** | **0.887** | 0.789 | **0.835 [0.77, 0.89]** |
| MLP on descriptors h=16 | 0.810 | 0.889 | 0.867 | 0.800 | 0.832 [0.76, 0.89] |
| MLP on descriptors h=32 | 0.777 | 0.881 | 0.828 | 0.800 | 0.814 [0.75, 0.87] |
| BiLSTM over frequency h=16/32/64 | *running* | | | | |

**The MLP does not beat logistic regression even with 207 patients**, and
degrades with width (h=16 → h=32 loses 0.033). Its CI overlaps the baseline's
almost entirely.

This is a useful negative because it removes the obvious excuse. On PD-vs-ET the
deep models could be dismissed as starved at n=29. Here there are seven times as
many patients, the merge is legitimate (device cue orthogonal to label), the
axis is the one that works — and the linear model still wins.

Two readings, and the data does not yet separate them:

* the 10 descriptors already capture what is discriminable for N-vs-Tremor, so
  there is no residual non-linearity to find;
* 207 patients is still small for a learned representation, and the ceiling is
  further out than this.

The BiLSTM-over-frequency result will discriminate between them: it was the
architecture that beat the linear model on PD-vs-ET DRINK (AUC 0.870–0.942 vs
0.812), so if it also fails here the first reading is the likely one.

**Standing best N-vs-Tremor: merged OUT, logreg on 10 descriptors — bal-acc
0.821, AUC 0.905, precision 0.887, F1 0.835 [0.77, 0.89].**
