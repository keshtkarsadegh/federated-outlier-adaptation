"""
Reading a writer-counts artefact, and cutting an eligible population from it.

Why this is a module and not three lines in a task file
------------------------------------------------------
It was three lines in a task file, and they were wrong: they read a key named
``counts`` that no writer-counts artefact has ever had.  ``dict.get("counts",
payload)`` then fell back to the whole payload, the loop iterated its metadata
keys, and the run died comparing the string ``"chars80"`` with an integer -
one stage into a six-stage chain, on a GPU, twenty minutes after submission.

Nothing could have caught that.  An inline ``python -c`` in a task line is
invisible to the test suite by construction: it is a string until the moment it
runs.  So the reading of an artefact belongs in a module that a test can call,
and the task line becomes a command with arguments.

The schema, stated once
-----------------------
:func:`~federated_outlier_adaptation.outliers.scoring.writer_counts` writes

    {"classes", "num_classes", "writers", "rows",
     "per_writer_total": {writer: int},
     "per_writer_per_class": {writer: {label: int}},
     "class_totals", "summary"}

and every dataset writes the same keys - the digits artefacts
differ only in their values.  :func:`per_writer_totals` is the one place that
knows this, and it **refuses an unrecognised shape** rather than degrading to a
default, because a silent default is what turned a typo into a dead chain.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping

#: The key holding ``{writer: rows}`` in a writer-counts payload.
TOTALS_KEY = "per_writer_total"


def per_writer_totals(payload: Any) -> Dict[str, int]:
    """
    ``{writer: rows}`` from a writer-counts payload.

    Raises:
        ValueError: when the payload is not a writer-counts artefact, or its
            totals are not integers.  The message names the keys that *were*
            found, because the failure this guards against is reading the right
            file for the wrong key.
    """
    if not isinstance(payload, Mapping):
        raise ValueError(
            f"a writer-counts artefact is a JSON object; got {type(payload).__name__}"
        )
    if TOTALS_KEY not in payload:
        raise ValueError(
            f"no {TOTALS_KEY!r} in the writer-counts artefact; it holds "
            f"{sorted(payload)}. This is the file `foa writer-counts` writes."
        )
    totals = payload[TOTALS_KEY]
    if not isinstance(totals, Mapping) or not totals:
        raise ValueError(f"{TOTALS_KEY!r} is empty or not an object")
    bad = [w for w, n in totals.items() if not isinstance(n, int) or isinstance(n, bool)]
    if bad:
        raise ValueError(
            f"{TOTALS_KEY!r} must map writer -> int; {bad[:3]} are not"
        )
    return {str(w): int(n) for w, n in totals.items()}


def load_totals(path) -> Dict[str, int]:
    """The totals of a writer-counts file on disk."""
    with open(path) as handle:
        return per_writer_totals(json.load(handle))


def eligible(totals: Mapping[str, int], min_samples: int) -> List[str]:
    """
    The writers holding at least ``min_samples`` rows, sorted.

    Sorted so the list is a function of the data and not of dictionary order:
    two runs of this on the same artefact produce the same file, byte for byte.
    """
    if min_samples < 1:
        raise ValueError(f"min_samples must be >= 1, got {min_samples}")
    return sorted(w for w, n in totals.items() if n >= min_samples)


def eligibility_record(totals: Mapping[str, int], min_samples: int) -> Dict[str, Any]:
    """
    The population record: who is eligible, and what the floor cost.

    Carries the population it was cut from, so a reader can see the price of the
    floor - how many writers and how many rows it excluded - without going back
    to the counts artefact.
    """
    kept = eligible(totals, min_samples)
    all_rows = sum(totals.values())
    kept_rows = sum(totals[w] for w in kept)
    return {
        "rule": f"writers holding at least {min_samples} rows",
        "min_samples": int(min_samples),
        "size": len(kept),
        "clients": kept,
        "population": {
            "writers": len(totals),
            "rows": all_rows,
            "writers_kept": len(kept),
            "rows_kept": kept_rows,
            "writers_dropped": len(totals) - len(kept),
            "rows_dropped": all_rows - kept_rows,
            "share_of_writers": (len(kept) / len(totals)) if totals else None,
            "share_of_rows": (kept_rows / all_rows) if all_rows else None,
        },
    }


def write_eligibility(record: Dict[str, Any], path) -> Path:
    """Write the record where the rest of the chain reads it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n")
    return path
