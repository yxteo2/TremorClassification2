"""Per-split checkpointing so a container restart does not cost a whole run.

The container running this project resets without warning and kills background
work — it has done so seven times, most expensively at split 18 of 20. Nothing
here is stateful, so a run that saves its per-split rows can simply skip the
splits it already has.

Usage inside an experiment's split loop::

    res, done = resume_load(TAG, ARMS)
    for sp in range(SPLITS):
        if sp in done:
            continue
        ...
        for a in ARMS:
            res[a].append(score(...))
        resume_save(TAG, res, sp)

Checkpoints live in the scratchpad, **outside the repository**, because the
reset also reverts the working tree. Delete the file to force a clean re-run;
`resume_load` also discards a checkpoint whose arm names no longer match, so
editing the arms of an experiment cannot silently mix old and new rows.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

_DIR = Path(os.environ.get(
    "TREMOR_CKPT_DIR",
    "/tmp/claude-0/-home-user-TremorClassification2/"
    "045373e7-5c30-5c16-b89d-4cbbb5ac8751/scratchpad/ckpt"))


def _path(tag):
    _DIR.mkdir(parents=True, exist_ok=True)
    return _DIR / f"{tag}.json"


def resume_load(tag, arms):
    """Return ``(res, done_splits)``; empty if there is no usable checkpoint."""
    p = _path(tag)
    if not p.exists():
        return {a: [] for a in arms}, set()
    try:
        d = json.loads(p.read_text())
    except Exception:                                    # noqa: BLE001
        return {a: [] for a in arms}, set()
    if sorted(d.get("arms", [])) != sorted(arms):
        print(f"[resume] {tag}: arms changed, ignoring stale checkpoint")
        return {a: [] for a in arms}, set()
    res = {a: [list(r) for r in d["res"][a]] for a in arms}
    done = set(d.get("done", []))
    if done:
        print(f"[resume] {tag}: {len(done)} split(s) recovered, "
              f"resuming after {max(done) + 1}")
    return res, done


def resume_save(tag, res, sp, extra=None):
    """Append split ``sp``'s rows atomically. Cheap enough to call every split."""
    p = _path(tag)
    done = sorted(set(json.loads(p.read_text()).get("done", []))
                  | {int(sp)}) if p.exists() else [int(sp)]
    payload = {"arms": sorted(res), "done": done,
               "res": {a: [list(map(float, r)) for r in v]
                       for a, v in res.items()}}
    if extra is not None:
        payload["extra"] = extra
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload))
    tmp.replace(p)
