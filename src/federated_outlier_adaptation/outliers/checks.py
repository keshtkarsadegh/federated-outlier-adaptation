"""
Assertions a study chain makes about its own artefacts, as testable functions.

A chain stage is only as good as what it refuses to pass on.  The digit study
made these checks with inline ``python -c`` steps in its task files, and the
second chain inherited the habit; one of them read a key that did not
exist and the chain died on a GPU rather than in the suite.

So the checks live here, where a test can call them with a deliberately broken
artefact and watch them fail.  Every function returns a **list of problems**
rather than raising: a run that is wrong in three ways should say so once, not
three times in three submissions.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence

#: The three partitions a fold book cuts, and that a trainable client needs.
PARTS = ("train", "val", "test")


def load_clients(path) -> List[str]:
    """The client list of any of this pipeline's list-shaped artefacts."""
    from federated_outlier_adaptation.outliers.selection import load_client_pool

    clients, _ = load_client_pool(path)
    return list(clients)


def check_size(clients: Sequence[str], expect: Optional[int]) -> List[str]:
    """The list is the size the design says it is."""
    problems = []
    if expect is not None and len(clients) != expect:
        problems.append(f"holds {len(clients)} clients, expected {expect}")
    duplicates = sorted({c for c in clients if list(clients).count(c) > 1})
    if duplicates:
        problems.append(f"duplicate clients: {duplicates[:5]}")
    return problems


def check_book(book, clients: Sequence[str], folds: Iterable[int]) -> List[str]:
    """
    The book covers every client, and no fold leaves a partition empty.

    An empty train partition is the one that does not announce itself: the
    client trains on nothing, contributes an unchanged update, and still appears
    in the per-client column with a number.
    """
    problems = []
    uncovered = [c for c in clients if not book.covers(c)]
    if uncovered:
        problems.append(f"the book does not cover {uncovered[:5]}")
        clients = [c for c in clients if c not in set(uncovered)]
    empty = [
        f"{client}/fold{fold}/{part}"
        for fold in folds
        for client in clients
        for part in PARTS
        if not book.part(fold, client, part)
    ]
    if empty:
        problems.append(f"empty partitions: {empty[:5]} ({len(empty)} total)")
    return problems


def check_membership(clients: Sequence[str],
                     subset_of: Optional[Dict[str, Sequence[str]]] = None,
                     disjoint_from: Optional[Dict[str, Sequence[str]]] = None) -> List[str]:
    """
    Where these clients must and must not come from.

    The two rules the selection protocol rests on: a cohort writer comes from
    the BAD pool, and no cohort writer is also an old-data writer. Neither is
    visible in a result - a leaked writer just makes preservation look good.
    """
    problems = []
    here = set(clients)
    for name, population in (subset_of or {}).items():
        stray = sorted(here - set(population))
        if stray:
            problems.append(f"not in {name}: {stray[:5]}")
    for name, population in (disjoint_from or {}).items():
        shared = sorted(here & set(population))
        if shared:
            problems.append(f"also in {name}: {shared[:5]}")
    return problems


def describe(clients: Sequence[str], totals: Optional[Dict[str, int]] = None) -> str:
    """A one-line summary a chain log can be read back from."""
    if not totals:
        return f"{len(clients)} clients"
    sizes = sorted(totals.get(c, 0) for c in clients)
    return (
        f"{len(clients)} clients, rows {sizes[0]}..{sizes[-1]} "
        f"(median {sizes[len(sizes) // 2]})"
    )
