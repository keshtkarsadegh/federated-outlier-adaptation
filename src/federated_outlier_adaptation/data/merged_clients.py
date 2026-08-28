"""
One client holding two writers' data.

The extreme cases ask three questions about a federation that has almost no
clients left, and the third one needs an object the rest of the pipeline has no
word for.  ``single`` is a federation of one writer and ``double`` a federation
of two; ``dual`` is a federation of **one client whose data is both writers'
data**.  The three differ in how the same images are distributed, not in which
images they are, which is what makes them a controlled comparison: ``double``
and ``dual`` hold exactly the same rows and differ only in whether the
aggregation ever sees them separately.

The id is the definition
------------------------
A merged client is named by joining its members with ``+``:
``"f1556_19+f0596_26"``.  There is no registry to keep in step and nothing to
persist - the id carries its own meaning, so a merged client survives being
written to a pool file, passed through a worker process and read back out of a
provenance block without anything else having to know it is special.

Partitions are preserved, not re-cut
------------------------------------
The merge is a union **per partition**: train is the union of the members'
train rows, validation of their validation rows, test of their test rows.  It is
not a re-split of the combined data, because a re-split would move rows between
partitions and the whole point of ``dual`` is that it holds precisely what
``double`` holds.  ``Nist28Dataset.build_dataset`` already unions a list of
writers this way when a fold book covers them, so the merge is expressed by
expanding the id into its members and changing nothing else.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence

#: What joins the members of a merged client id.
MERGE_SEPARATOR = "+"


def is_merged(client_id) -> bool:
    """Whether an id names more than one writer."""
    return isinstance(client_id, str) and MERGE_SEPARATOR in client_id


def members(client_id) -> List[str]:
    """
    The writers an id names: its members if merged, else itself.

    Empty components are dropped, so a stray separator cannot silently produce
    a member called ``""`` that matches no writer and contributes no rows.
    """
    if not is_merged(client_id):
        return [client_id]
    return [part for part in str(client_id).split(MERGE_SEPARATOR) if part]


def merged_id(writers: Sequence[str]) -> str:
    """The id of a client holding all of ``writers``, in the given order."""
    writers = [str(writer) for writer in writers]
    if not writers:
        raise ValueError("A merged client needs at least one writer.")
    if len(writers) == 1:
        return writers[0]
    if any(MERGE_SEPARATOR in writer for writer in writers):
        raise ValueError(
            f"A writer id may not contain {MERGE_SEPARATOR!r}; it is what "
            "separates the members of a merged client."
        )
    return MERGE_SEPARATOR.join(writers)


def expand(client_ids) -> List[str]:
    """
    Replace every merged id in a list by its members, keeping the order.

    A writer named by two different merged clients would appear twice; that is
    the caller's business, and no case in the study does it.
    """
    if isinstance(client_ids, str):
        return members(client_ids)
    expanded: List[str] = []
    for client_id in client_ids or []:
        expanded.extend(members(client_id))
    return expanded


def unknown_members(client_ids: Iterable[str], known: Iterable[str]) -> List[str]:
    """
    Members of ``client_ids`` that are not writers of ``known``.

    A merged id is valid exactly when every one of its members is, so this is
    what validation asks instead of looking the merged id itself up in a
    population that will never contain it.
    """
    known = set(known)
    missing: List[str] = []
    for client_id in client_ids:
        for member in members(client_id):
            if member not in known and member not in missing:
                missing.append(member)
    return missing
