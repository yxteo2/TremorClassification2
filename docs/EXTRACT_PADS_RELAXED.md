# Runbook: extract PADS `Relaxed` (the REST-equivalent task)

**Why:** PD-vs-ET works on our data at **REST** (merged 2015+NewData: bal-acc
0.740, ET-F1 0.557). The only PADS task extracted so far is **StretchHold**,
which is postural — so that result currently has **no task-matched external
test**. `Relaxed` is PADS's rest condition and would provide one.

**I could not run this here:** PhysioNet is blocked by this environment's proxy
(`CONNECT tunnel failed, 403`). Run it locally.

## Steps

```bash
# 1. Download PADS (PhysioNet DOI 10.13026/m0w9-zx22)
#    https://physionet.org/content/parkinsons-disease-smartwatch/1.0.0/
#    -> unpack to ./PADS

# 2. Confirm the layout still matches what the extractor expects
python -m pdetn.extract_pads --pads-root PADS --inspect
python -m pdetn.extract_pads --pads-root PADS --list-tasks

# 3. Extract Relaxed (matches both Relaxed1 and Relaxed2)
python -m pdetn.extract_pads --pads-root PADS --task Relaxed --out pads_relaxed

# 4. Re-extract StretchHold with the current code so both folders share the
#    new filename layout (the task token is now in the filename)
python -m pdetn.extract_pads --pads-root PADS --task StretchHold --out pads_stretchhold
```

## What to check immediately after

```python
from pdetn.crossdataset import load_pads_extracted
r = load_pads_extracted("pads_relaxed", strict=True)
y = [x.y for x in r]
print(len(r), "recordings", len({x.subject for x in r}), "patients")
print("N/PD/ET:", [y.count(k) for k in (0,1,2)])
```

**Expect N=79 / PD=276 / ET=28 patients.** If ET comes out as 41, the strict
label mapping has regressed — see `reports/pads_label_bug.md`.

Relaxed has **two repetitions** per wrist (Relaxed1, Relaxed2), so expect
roughly **4 recordings per patient** vs 2 for StretchHold.

## Two bugs fixed in preparation

1. **Filename collision (silent data loss).** `extract_pads` wrote
   `<cls>_<pid>_<wrist>.txt` with no task token. Extracting `Relaxed` would have
   made Relaxed2 **overwrite** Relaxed1 for every patient — half the data gone,
   no error. Filenames are now `<cls>_<pid>_<task>_<wrist>.txt` and the manifest
   carries a `task` column.
2. **`load_pads_extracted(task=...)`** added, understanding both the legacy and
   new filename layouts.

## Then re-run the design

```bash
python -m tfbench.merged   # or the scratch runner used for reports/merged_design_results.md
```
with PADS `Relaxed` as the external cohort for the **REST-trained stage 2**, and
PADS `StretchHold` kept for the **OUT-trained stage 1**. That is the first
fully task-matched version of the chosen design.

**Prediction worth recording before the test:** on the 2015 cohort REST beat OUT
by 0.15–0.28 balanced accuracy for PD-vs-ET. If that is a property of the
*condition* rather than of our cohort, PADS Relaxed should beat PADS
StretchHold's 0.774. If it does not, the REST advantage is cohort-specific and
the merged 0.740 result should be reported with that caveat.
