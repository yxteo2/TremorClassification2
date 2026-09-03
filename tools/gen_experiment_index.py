"""Regenerate ``experiments/INDEX.md``.

Every experiment module, the date of the commit that last touched it, and the
reports that mention it by name. Run after adding an experiment or a report:

    python tools/gen_experiment_index.py

Modules starting with ``_`` are shared helpers, not studies, and are skipped.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXP, REP = ROOT / "experiments", ROOT / "reports"
OUT = EXP / "INDEX.md"

HEAD = """# Experiment index

Generated, not hand-edited: every experiment, its last commit date, and the \
reports that mention it by name.
An empty reports column means the study was run but never written up, or was \
superseded before writing.

| experiment | last commit | reports that cite it |
|---|---|---|
"""


def last_commit(path: Path) -> str:
    out = subprocess.run(
        ["git", "log", "-1", "--format=%ad", "--date=short", "--", str(path)],
        cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return out or "uncommitted"


def main() -> None:
    names = sorted(p.stem for p in EXP.glob("*.py")
                   if not p.stem.startswith("_"))
    texts = {p.name: p.read_text() for p in sorted(REP.glob("*.md"))}

    rows, orphans = [], []
    for n in names:
        pat = re.compile(rf"\b{re.escape(n)}\b")
        cites = [r for r, t in texts.items() if pat.search(t)]
        rows.append(f"| `{n}` | {last_commit(EXP / f'{n}.py')} | "
                    + (", ".join(f"`{c}`" for c in cites) if cites
                       else "**none**") + " |")
        if not cites:
            orphans.append(n)

    tail = (f"\n**{len(names)} experiments, {len(orphans)} with no report:** "
            + ", ".join(f"`{o}`" for o in orphans) + "\n")
    OUT.write_text(HEAD + "\n".join(rows) + "\n" + tail)
    print(f"{OUT.relative_to(ROOT)}: {len(names)} experiments, "
          f"{len(orphans)} unreported")


if __name__ == "__main__":
    main()
