#!/usr/bin/env python3
"""
The client recipe, read off the payloads instead of off the methods paragraph.

The methods section states how a client trains: Adam, a learning rate, a weight
decay, a batch size, five local epochs, and the local early-stopping rule left
off so that every client runs its whole budget. Those six statements were the
last quantities in the manuscript with no view behind them. They are not
measurements, which is why they sat in the hand-maintained block of
`numbers.tex` for as long as they did - but they are not free either: every one
of them is recorded in `config` inside every stored payload, because the runner
writes the configuration it ran under beside the accuracies it produced. A
constant that is written down in seven thousand files and typed once into a
macro is a constant that can drift without anything failing.

    python tools/export_recipe_view.py --root "$FOA_STUDY_DIR" \\
        --csv "$FOA_STUDY_DIR/tables/paper/"

UNANIMITY IS THE CLAIM. The paper says "the clients train with" and then one
number, so the view has to be able to say that the number is one number. Every
payload under the study tag is read, the six fields are collected, and a field
that holds more than one value across the programme is a FATAL refusal naming
both values rather than a row carrying whichever came first. `payloads` is how
many files agreed, and it is printed so that a run of this tool over half a
study cannot be mistaken for a run over all of it.

NOT EVERY FIELD IS IN EVERY PAYLOAD, AND THE COUNT SAYS SO. The runner began
writing the two client-side early-stopping settings into `config` after the
first stages had run, so those two rows carry a smaller `payloads` than the four
beside them. That is why the count is a column rather than a sentence in this
docstring: a field the programme records less often is still unanimous where it
is recorded, and a reader can see which is which without being told.

THE ROUNDS ARE NOT HERE. The horizon is a property of a stage and not of a
client, `cost_stages.csv` already carries it per stage, and `\\nRounds` has been
read from that file since it landed. Two views of one number are how two numbers
begin.

Standard library only apart from `report_tables`, which decides which folders
belong to this study; deciding that a second way here is how a retired
arrangement walks back into a table.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO / "src"))

import report_tables  # noqa: E402

#: The six fields, where each sits inside ``config``, and the name the view
#: gives it. The nesting is the runner's: the optimiser settings arrive as one
#: ``hyperparameters`` dict because that is what the trainer is constructed
#: from, and the loop's own two sit beside it.
FIELDS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("learning_rate", ("hyperparameters", "learning_rate")),
    ("weight_decay", ("hyperparameters", "weight_decay")),
    ("client_early_stopping", ("hyperparameters", "early_stopping")),
    ("client_patience", ("hyperparameters", "patience")),
    ("batch_size", ("batch_size",)),
    ("local_epochs", ("epochs",)),
)

COLUMNS = ("setting", "value", "payloads")


def _dig(body: dict, path: Tuple[str, ...]) -> Any:
    """One field of a payload's ``config``, or ``None`` where the path stops."""
    node: Any = body.get("config") or {}
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def _spell(value: Any) -> str:
    """
    How a value is written into the view.

    A boolean is written as 0 or 1 rather than as ``False``: every other column
    of every other view in this bundle is a number, and a reader loading them
    with one CSV reader should not have to know which of them spells a flag in
    Python.
    """
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        return repr(value)
    return str(value)


def collect(root: Path, tag: str) -> Dict[str, Dict[str, int]]:
    """``{setting: {value: payloads}}`` over every stored payload of the study."""
    found: Dict[str, Dict[str, int]] = {name: {} for name, _ in FIELDS}
    for name in sorted(glob.glob(f"{root}/{tag}_*/**/summary_0.json", recursive=True)):
        if not report_tables.in_study(name):
            continue
        try:
            payload = json.loads(Path(name).read_text())
        except ValueError:
            continue
        for body in payload.values():
            if not isinstance(body, dict):
                continue
            for setting, path in FIELDS:
                value = _dig(body, path)
                if value is None:
                    continue
                counts = found[setting]
                spelled = _spell(value)
                counts[spelled] = counts.get(spelled, 0) + 1
    return found


def rows(found: Dict[str, Dict[str, int]]) -> List[dict]:
    """One row per setting, or a refusal naming the values that disagreed."""
    out: List[dict] = []
    for setting, _ in FIELDS:
        counts = found[setting]
        if not counts:
            raise SystemExit(
                f"FATAL: no stored payload records {setting!r}. The manuscript "
                "states it as a constant of the protocol, and a constant with "
                "no record behind it is a sentence, not a view."
            )
        if len(counts) > 1:
            spread = ", ".join(f"{value} in {n} payloads"
                               for value, n in sorted(counts.items()))
            raise SystemExit(
                f"FATAL: {setting!r} is not one value across the programme: "
                f"{spread}. The manuscript states one; which of these it means "
                "is not for this tool to choose."
            )
        value, payloads = next(iter(counts.items()))
        out.append({"setting": setting, "value": value, "payloads": payloads})
    return out


def write_csv(path: Path, rows_: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(COLUMNS), restval="")
        writer.writeheader()
        writer.writerows(rows_)
    print(f"  -> {path}  ({len(rows_)} rows)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path,
                    help="The study root ($FOA_STUDY_DIR).")
    ap.add_argument("--csv", required=True, type=Path,
                    help="Directory the CSV view is written into.")
    ap.add_argument("--study-tag", default="d01")
    args = ap.parse_args()

    found = collect(args.root, args.study_tag)
    emitted = rows(found)
    for row in emitted:
        print(f"{row['setting']:>22} = {row['value']:<8} "
              f"({row['payloads']} payloads, all agreeing)")
    write_csv(args.csv / "recipe.csv", emitted)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
