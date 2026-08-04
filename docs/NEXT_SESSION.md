# Next session — integrate NewData (6 new ET subjects)

## Status
6 new ET subjects in `NewData/ET/` (ET_19,20,21,22,23,26) -> ET 15 -> 21 (+40%).
Same quaternion modality + same lab protocol as the 2015 data, so unlike PADS
there should be no device domain shift. PD/HC folders are empty (ET-only).

## h5 structure (verified)
    Processed/<sensorID>/Orientation   (N, 4) quaternions   <- use this
    Sensors/<sensorID>/{Gyroscope, Accelerometer, Time}
  * 6 sensor IDs: 7257, 7279, 10464, 10468, 10833, 10871
  * 4910 samples @ 128 Hz = 38.4 s  (2015 data is ~10 s)
  * FileFormatVersion 5

## SENSOR MAPPING — RESOLVED
From NewData/Convert_h5_and_csv_to_xlsx.ipynb:
    # Naming order: 10464 (I), 10468 (U), 10833 (T), 10871 (L), 7257 (H)
    7257  = Hand (H)
    10871 = Wrist (L)        <- matches 2015 "lower_arm" (worn near the wrist)
    10468 = Upper Arm (U)
    10464 = Index Finger (I) -- not in the 2015 set
    10833 = Thorax/Trunk (T) -- not in the 2015 set
    7279  = 6th sensor, role unconfirmed (check a file's column labels)

Channel order to build (matching tremor.data):
    hand=7257 (ch 0-2), lower_arm=10871 (ch 3-5), upper_arm=10468 (ch 6-8)
Note the notebook labels columns 'REST (R)' etc., so column headers in the
converted xlsx also encode condition + side -- useful cross-check.

## Loader plan (pdetn/load_2025.py)
1. Actions: 02/09 -> OUT (the winning condition), 01/08 -> REST
2. Select the 3 sensors for the active limb; order distal->proximal to match
   tremor.data channel order (hand=0-2, lower_arm=3-5, upper_arm=6-8)
3. Resample 128 -> 100 Hz (scipy.signal.resample_poly, up=25, down=32)
4. Length: new recordings are ~38 s vs ~10 s; crop/window to match
5. Quaternion frame correction on the 2015 data: CONFIRM it is the single
   sandwich product v' = q (x) v (x) q* with q=[0,1,0,0] (one 180 deg rotation
   about x). Two full 180 deg rotations would be a no-op.
6. Emit tremor.data.Recording objects (subject ids must not collide with 2015)

## Validate before combining
Run the dataset-identity probe (pdetn.crossdataset.dataset_identity_probe) on
old-vs-new. PADS gave AUC 0.999 and was unusable. If this comes back near chance,
the sets are genuinely poolable.

## Then re-run the canonical PD-vs-ET evaluation with 21 ET
Numbers to beat (reports/final_results.md), lower_arm + OUT:
    AUC 0.848 | ET precision 0.67 | ET recall 0.40 | ET F1 0.50
Caution: ET-only augmentation is exactly the setup that failed with PADS
(precision collapsed to 0.179 as the boundary shifted). Watch ET precision, not
just recall. With no device shift here it should behave, but verify.
