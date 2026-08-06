"""Cache patient descriptor tables to disk.

`build_all` recomputes 12 transforms over every recording (~7 min, VMD and the
S-transform dominate). Anything that needs the tables more than once -- reruns,
a second axis, a notebook restart -- should go through here instead.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tfbench.benchmark import build_all

DEFAULT = Path("artifacts/tfbench_tables.npz")


def load_or_build(recs, path=DEFAULT, methods=None, fs=100.0, rebuild=False,
                  verbose=True, **kw):
    """Return {method: (X, y, patients)}, from cache when available."""
    path = Path(path)
    if path.is_file() and not rebuild:
        z = np.load(path, allow_pickle=True)
        names = [k[:-len("__X")] for k in z.files if k.endswith("__X")]
        tables = {n: (z[f"{n}__X"], z[f"{n}__y"], z[f"{n}__p"]) for n in names}
        if verbose:
            print(f"loaded {len(tables)} cached tables from {path}")
        return tables
    tables = build_all(recs, methods=methods, fs=fs, verbose=verbose, **kw)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{f"{n}__{k}": v
                                 for n, t in tables.items()
                                 for k, v in zip("Xyp", t)})
    if verbose:
        print(f"cached {len(tables)} tables to {path}")
    return tables
