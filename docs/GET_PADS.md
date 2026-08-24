# Getting PADS

The two extracted task folders in this repo — `pads_stretchhold/` and
`pads_relaxed/` — were produced from the full PADS archive. The archive itself is
**not** in the repo, which limits what can be tested: eight of PADS's ten tasks
have never been extracted, including the kinetic ones
(`reports/kinetic_task_audit.md`).

## Source

PhysioNet, DOI **10.13026/m0w9-zx22** —
https://physionet.org/content/parkinsons-disease-smartwatch/1.0.0/
(a Kaggle mirror also exists). Licence CC BY-NC-SA 4.0, academic use permitted.

Unzip to a folder such as `PADS/`, containing `movement/timeseries/*.txt` and
`patients/patient_*.json`.

## Extraction

```bash
python -m common.extract_pads --pads-root PADS --inspect          # confirm layout
python -m common.extract_pads --pads-root PADS --out pads_stretchhold
```

Labels are re-derived from the manifest by **exact** diagnosis match. A substring
match once put 13 non-ET records into the ET class
(`reports/pads_label_bug.md`); strict mode gives 79 N / 276 PD / 28 ET.

## Network access in this environment

`physionet.org` is **denied by the egress proxy** (403 on CONNECT, confirmed in
the proxy's own failure log), so the archive cannot be fetched from a Claude Code
web session. To extract further tasks, either run the extraction locally and
upload the resulting folders, or enable a network policy that permits PhysioNet.

## Why the remaining tasks are worth having

`reports/kinetic_task_audit.md` found that the claim "the kinetic tasks are where
ET separates best" is an **open hypothesis, not a measurement** — it rests on
NewData's 6 ET patients, where the PD-vs-ET permutation null reaches 0.819.
PADS's kinetic tasks carry **28 ET**, which is the smallest cohort that could
settle it.
