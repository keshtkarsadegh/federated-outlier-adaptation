"""
LEAF Shakespeare: next-character prediction with speaking roles as clients.

The task and the preprocessing follow LEAF (S. Caldas, S. M. K. Duddu, P. Wu,
T. Li, J. Konecny, H. B. McMahan, V. Smith, A. Talwalkar, *LEAF: A Benchmark for
Federated Settings*, 2018), specifically
``leaf/data/shakespeare/preprocess_shakespeare.py`` and
``leaf/models/utils/language_utils.py``.  The logic is re-implemented here so
that the package has no dependency on the LEAF distribution:

* the raw corpus is the Project Gutenberg *Complete Works of William
  Shakespeare* (eBook #100, public domain);
* a client ("user") is one speaking role of one play, named
  ``<PLAY>_<CHARACTER>`` with all non-alphanumerics replaced by ``_``;
* every role's utterances are concatenated into a single character stream, runs
  of two or more spaces are collapsed, and the stream is cut into overlapping
  windows of 80 characters (stride 1) whose target is the next character;
* the vocabulary is LEAF's 80-symbol alphabet :data:`ALL_LETTERS`, hence
  ``num_classes == 80``.

**Edition note.**  LEAF downloads ``1994-01-100.zip`` from
``http://www.gutenberg.org/files/100/old/``.  That path has been retired and now
answers 404 on gutenberg.org and on every mirror, so the current plain-text
edition of the very same eBook #100 (``https://www.gutenberg.org/cache/epub/100/
pg100.txt``, release date January 1 1994, most recently updated 2025) is used
instead.  It is the same corpus but a re-typeset edition: speaker cues sit on
their own line in full capitals instead of being indented two spaces, and the
text uses typographic punctuation.  :func:`parse_plays` therefore implements
LEAF's *rules* (play -> role -> utterance) against this layout, and
:func:`normalise_text` folds the typographic characters back to the ASCII
symbols of :data:`ALL_LETTERS`.

Storage: the prepared dataset is a single ``.npz`` plus one JSON index, in line
with the project's rule of never spending inodes on per-sample files.  Because
the 80-character windows overlap with stride 1, the arrays are stored as the
concatenated per-user character stream plus offsets; ``X`` of shape
``[Nseq, 80]`` and ``y`` of shape ``[Nseq]`` are recovered exactly (as strided
views) by :class:`ShakespeareData`, at 1/80 of the storage cost.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from collections import OrderedDict
from pathlib import Path
from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from federated_outlier_adaptation import config
from federated_outlier_adaptation.data.splits import split_indices
from federated_outlier_adaptation.utils.eval_cache import IDENTITY
from federated_outlier_adaptation.utils.seeding import make_generator

# --- LEAF's alphabet (leaf/models/utils/language_utils.py) -------------------
ALL_LETTERS = (
    "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]"
    "abcdefghijklmnopqrstuvwxyz}"
)
NUM_LETTERS = len(ALL_LETTERS)  # 80
SEQ_LENGTH = 80

_LETTER_INDEX = {char: index for index, char in enumerate(ALL_LETTERS)}

#: Works of eBook #100 that contain no dramatic dialogue.  LEAF discards the
#: sonnets explicitly; the remaining poems yield no speaking roles either.
NON_DRAMATIC_WORKS = frozenset(
    {
        "THE SONNETS",
        "A LOVER'S COMPLAINT",
        "THE PASSIONATE PILGRIM",
        "THE PHOENIX AND THE TURTLE",
        "THE RAPE OF LUCRECE",
        "VENUS AND ADONIS",
    }
)

# Typographic characters of the current edition folded onto LEAF's alphabet.
_FOLD = {
    "‘": "'",
    "’": "'",
    "‚": "'",
    "‛": "'",
    "′": "'",
    "“": '"',
    "”": '"',
    "„": '"',
    "″": '"',
    "«": '"',
    "»": '"',
    "‐": "-",
    "‑": "-",
    "‒": "-",
    "–": "-",
    "—": "-",
    "―": "-",
    "−": "-",
    "─": "-",
    "…": "...",
    " ": " ",
    " ": " ",
    " ": " ",
    " ": " ",
    " ": " ",
    "﻿": "",
    "Æ": "AE",
    "æ": "ae",
    "Œ": "OE",
    "œ": "oe",
    "À": "A",
    "Â": "A",
    "Ä": "A",
    "Ç": "C",
    "È": "E",
    "É": "E",
    "Ê": "E",
    "Ë": "E",
    "Î": "I",
    "Ï": "I",
    "Ô": "O",
    "Ö": "O",
    "Û": "U",
    "Ü": "U",
    "à": "a",
    "á": "a",
    "â": "a",
    "ä": "a",
    "ç": "c",
    "è": "e",
    "é": "e",
    "ê": "e",
    "ë": "e",
    "ì": "i",
    "í": "i",
    "î": "i",
    "ï": "i",
    "ñ": "n",
    "ò": "o",
    "ó": "o",
    "ô": "o",
    "ö": "o",
    "ù": "u",
    "ú": "u",
    "û": "u",
    "ü": "u",
    "ÿ": "y",
    "œ": "oe",
}

# Additional invisible marks that carry no text (zero-width space, soft hyphen).
_FOLD.update({"​": "", "‌": "", "‍": "", "­": "", "﻿": ""})

_FOLD_TABLE = {ord(source): target for source, target in _FOLD.items()}

_GUTENBERG_START = re.compile(
    r"^\s*\*\*\*\s*START OF TH(?:IS|E) PROJECT GUTENBERG.*$", re.MULTILINE | re.IGNORECASE
)
_GUTENBERG_END = re.compile(
    r"^\s*\*\*\*\s*END OF TH(?:IS|E) PROJECT GUTENBERG.*$", re.MULTILINE | re.IGNORECASE
)

# A speaker cue: a full-capital role name terminated by a period, either alone
# on its line (current edition) or followed by the first line of the utterance
# (LEAF's 1994 edition).
_CUE = re.compile(r"^([A-Z][A-Z' ]{0,48})\.(?:\s+(\S.*))?$")
_HEADING = re.compile(r"^(ACT|SCENE|PROLOGUE|INDUCTION|EPILOGUE|CHORUS)\b")
# "SCENE I. An open Place." and "ACT IV." are headings, not speaker cues; the
# numeral distinguishes them from the roles that are called PROLOGUE or CHORUS.
_NUMBERED_HEADING = re.compile(r"^(ACT|SCENE)\s+[IVXLCDM0-9]", re.IGNORECASE)
_DRAMATIS = re.compile(r"^Dramatis Person", re.IGNORECASE)
_FRONT_MATTER_LINES = 400
_MULTI_SPACE = re.compile(r"   *")
_NON_WORD = re.compile(r"\W+")


# --------------------------------------------------------------------- text
def normalise_text(text: str) -> str:
    """Fold line endings and typographic characters onto LEAF's alphabet."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text.translate(_FOLD_TABLE)


def strip_gutenberg_boilerplate(text: str) -> str:
    """Drop the Project Gutenberg header and licence footer."""
    start = _GUTENBERG_START.search(text)
    if start is not None:
        text = text[start.end() :]
    end = _GUTENBERG_END.search(text)
    if end is not None:
        text = text[: end.start()]
    return text


def read_raw_text(path: Path) -> str:
    """
    Read the corpus from a ``.txt`` file or, LEAF-style, from a ``.zip``.

    A zip archive is read **in memory**; it is never extracted, not even to a
    temporary directory.
    """
    path = Path(path)
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            members = [n for n in archive.namelist() if n.lower().endswith(".txt")]
            if not members:
                raise ValueError(f"No .txt member inside {path}")
            payload = archive.read(members[0])
    else:
        payload = path.read_bytes()
    return normalise_text(payload.decode("utf-8", errors="replace"))


# ------------------------------------------------------------------ parsing
def _contents_titles(lines: Sequence[str]) -> Tuple[List[str], int]:
    """Return the work titles listed in the table of contents and its end."""
    start = None
    for index, line in enumerate(lines):
        if line.strip() == "Contents":
            start = index
            break
    if start is None:
        return [], 0

    titles: List[str] = []
    blanks = 0
    index = start + 1
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if not stripped:
            blanks += 1
            if titles and blanks >= 3:
                break
            index += 1
            continue
        if not line.startswith("    "):
            break
        blanks = 0
        titles.append(stripped)
        index += 1
    return titles, index


def _work_bounds(
    lines: Sequence[str], titles: Sequence[str], contents_end: int
) -> List[Tuple[str, int, int]]:
    """Locate the body of every listed work as ``(title, start, stop)``."""
    starts: List[Tuple[str, int]] = []
    cursor = contents_end
    for title in titles:
        for index in range(cursor, len(lines)):
            if lines[index].strip() == title:
                starts.append((title, index))
                cursor = index + 1
                break
    bounds = []
    for position, (title, start) in enumerate(starts):
        stop = starts[position + 1][1] if position + 1 < len(starts) else len(lines)
        bounds.append((title, start, stop))
    return bounds


def _play_text_start(body: Sequence[str]) -> int:
    """
    Index of the first line of dramatic text inside a play body.

    The per-play table of contents and the *Dramatis Personae* list precede the
    text; both contain lines that would otherwise be mistaken for speaker cues.
    """
    marker = 0
    for index in range(min(len(body), _FRONT_MATTER_LINES)):
        stripped = body[index].strip()
        if stripped == "Contents" or _DRAMATIS.match(stripped):
            marker = index
    for index in range(marker, len(body)):
        if _HEADING.match(body[index].strip()):
            return index
    return marker


def parse_plays(text: str) -> Tuple[List[Tuple[str, "OrderedDict[str, List[str]]"]], int]:
    """
    Split the corpus into ``(play_title, {role: [utterances]})`` pairs.

    Mirrors LEAF's ``_split_into_plays``: non-dramatic works are dropped, roles
    are upper-cased so that different casings of one name collapse, and plays
    that yield at most one role are discarded as parsing failures.

    Returns:
        The list of plays and the number of characters of text that no rule
        consumed (stage directions, scene headings, front matter).
    """
    text = strip_gutenberg_boilerplate(normalise_text(text))
    lines = text.split("\n")
    titles, contents_end = _contents_titles(lines)
    if not titles:
        raise ValueError("No table of contents found in the corpus")

    plays: List[Tuple[str, "OrderedDict[str, List[str]]"]] = []
    discarded = 0

    for title, start, stop in _work_bounds(lines, titles, contents_end):
        if title.upper() in NON_DRAMATIC_WORKS:
            continue
        body = lines[start:stop]
        roles: "OrderedDict[str, List[str]]" = OrderedDict()
        current: Optional[str] = None

        for line in body[_play_text_start(body) :]:
            stripped = line.strip()
            if not stripped:
                current = None
                continue
            if _NUMBERED_HEADING.match(stripped):
                current = None
                discarded += len(stripped)
                continue
            cue = _CUE.match(stripped)
            if cue is not None:
                current = cue.group(1).strip().upper()
                roles.setdefault(current, [])
                if cue.group(2):
                    roles[current].append(cue.group(2).strip())
                continue
            if current is None or stripped.startswith("["):
                discarded += len(stripped)
                continue
            roles[current].append(stripped)

        roles = OrderedDict((role, said) for role, said in roles.items() if said)
        if len(roles) > 1:
            plays.append((title, roles))
    return plays, discarded


def user_id(play: str, character: str) -> str:
    """LEAF's ``play_and_character``: ``<PLAY>_<ROLE>``, non-alphanumerics -> ``_``."""
    return _NON_WORD.sub("_", (play + "_" + character).replace(" ", "_"))


def role_text(utterances: Sequence[str]) -> str:
    """
    LEAF's ``txt_to_data`` preprocessing of one role's stream.

    The utterances are joined by newlines (one per line, as LEAF writes them),
    newlines become spaces and runs of two or more spaces collapse to one.
    """
    raw = "\n".join(utterances) + "\n"
    raw = raw.replace("\n", " ")
    return _MULTI_SPACE.sub(" ", raw)


def encode(text: str) -> Tuple[np.ndarray, int]:
    """
    Map characters onto :data:`ALL_LETTERS` indices.

    Characters outside the alphabet are dropped (LEAF's ``ALL_LETTERS.find``
    would silently map them to ``-1``, i.e. onto the last symbol).  Returns the
    ``uint8`` token stream and the number of dropped characters.
    """
    tokens = [_LETTER_INDEX[char] for char in text if char in _LETTER_INDEX]
    return np.asarray(tokens, dtype=np.uint8), len(text) - len(tokens)


# --------------------------------------------------------------- preparation
def build_users(
    text: str, seq_length: int = SEQ_LENGTH, min_samples: int = 2
) -> Tuple[List[str], List[np.ndarray], dict]:
    """
    Turn the raw corpus into per-user token streams.

    Args:
        text: The raw corpus.
        seq_length: Input window length (LEAF uses 80).
        min_samples: Users with fewer sequences are dropped, following LEAF's
            ``-k`` minimum-samples filter.  The default of 2 reproduces LEAF's
            rule of discarding roles with fewer than two samples.

    Returns:
        ``(user_ids, token_streams, statistics)``.
    """
    plays, discarded_chars = parse_plays(text)

    users: List[str] = []
    streams: List[np.ndarray] = []
    dropped_users = 0
    dropped_chars = 0

    for play, roles in plays:
        for character, utterances in roles.items():
            tokens, lost = encode(role_text(utterances))
            dropped_chars += lost
            if tokens.size - seq_length < min_samples:
                dropped_users += 1
                continue
            users.append(user_id(play, character))
            streams.append(tokens)

    stats = {
        "num_plays": len(plays),
        "num_users": len(users),
        "num_sequences": int(sum(stream.size - seq_length for stream in streams)),
        "num_characters": int(sum(stream.size for stream in streams)),
        "dropped_users": dropped_users,
        "dropped_out_of_vocabulary_characters": dropped_chars,
        "discarded_characters": discarded_chars,
    }
    return users, streams, stats


def prepare(
    raw_path: Path,
    out_dir: Optional[Path] = None,
    seq_length: int = SEQ_LENGTH,
    min_samples: int = 2,
    log=print,
) -> dict:
    """
    Build ``shakespeare.npz`` and ``shakespeare_index.json``.

    The result is deterministic: the same corpus always produces byte-identical
    files.
    """
    raw_path = Path(raw_path)
    out_dir = Path(out_dir) if out_dir else config.SHAKESPEARE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    log(f"reading {raw_path}")
    text = read_raw_text(raw_path)
    users, streams, stats = build_users(text, seq_length=seq_length, min_samples=min_samples)
    if not users:
        raise ValueError("The corpus produced no usable users")

    lengths = np.asarray([stream.size for stream in streams], dtype=np.int64)
    starts = np.zeros(len(streams), dtype=np.int64)
    np.cumsum(lengths[:-1], out=starts[1:])
    tokens = np.concatenate(streams) if streams else np.empty(0, dtype=np.uint8)

    seq_counts = lengths - seq_length
    seq_starts = np.zeros(len(streams), dtype=np.int64)
    np.cumsum(seq_counts[:-1], out=seq_starts[1:])

    npz_path = out_dir / config.SHAKESPEARE_NPZ_NAME
    index_path = out_dir / config.SHAKESPEARE_INDEX_NAME
    np.savez(
        npz_path,
        tokens=tokens,
        user_starts=starts,
        user_lengths=lengths,
        seq_starts=seq_starts,
        seq_counts=seq_counts,
        seq_length=np.int64(seq_length),
        vocab_size=np.int64(NUM_LETTERS),
    )

    index = {
        "dataset": "shakespeare",
        "source": raw_path.name,
        "source_sha256": _sha256(raw_path),
        "seq_length": int(seq_length),
        "min_samples": int(min_samples),
        "vocabulary": ALL_LETTERS,
        "num_classes": NUM_LETTERS,
        "users": users,
        "counts": seq_counts.tolist(),
        "array_file": npz_path.name,
        **stats,
    }
    with open(index_path, "w") as handle:
        json.dump(index, handle)

    summary = {
        "npz": str(npz_path),
        "index": str(index_path),
        "npz_bytes": npz_path.stat().st_size,
        **stats,
    }
    log(
        f"{stats['num_users']} users, {stats['num_sequences']} sequences from "
        f"{stats['num_plays']} plays -> {npz_path}"
    )
    return summary


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


#: Sentinel for "the book has not been looked for yet", distinct from "looked
#: for and absent" - which is a real answer and must not be retried on every
#: client of every round.
_UNSET = object()


class WindowCache:
    """
    The row-addressed face of a Shakespeare dataset.

    Mirrors the three attributes the fold-book builder reads off the packed NIST
    cache - ``len``, ``writer_ids``, ``writer_index`` - over the dense window
    index.  It holds no data of its own: every row is a view into the character
    stream the dataset already has.
    """

    def __init__(self, data: "ShakespeareData"):
        self._data = data
        self.writer_ids: List[str] = list(data.users)
        counts = np.asarray(data._seq_counts, dtype=np.int64)
        # Row -> owning user, for all four million windows at once: each user
        # owns one contiguous block, so this is a repeat rather than a loop.
        self.writer_index: np.ndarray = np.repeat(
            np.arange(counts.size, dtype=np.int64), counts
        )
        self._rows = int(counts.sum())

    def __len__(self) -> int:
        return self._rows

    @property
    def resolution(self):
        """No such thing here; present so a caller can ask and get an answer."""
        return None


class _LazyWriterSamples(Mapping):
    """
    ``{user: [(row, label), ...]}`` computed per user, on demand.

    The book builder asks for the whole mapping and then uses a handful of its
    keys.  Answering that literally would build four million tuples to split
    ten users.
    """

    def __init__(self, data: "ShakespeareData", writers=None):
        self._data = data
        self._keys = list(data.users) if writers is None else list(writers)

    def __getitem__(self, user: str):
        if user not in self._data._row_of:
            raise KeyError(user)
        rows = self._data.window_rows(user)
        labels = self._data.row_labels(rows)
        return list(zip(rows.tolist(), labels.tolist()))

    def __iter__(self):
        return iter(self._keys)

    def __len__(self) -> int:
        return len(self._keys)

    def __contains__(self, user) -> bool:
        return user in self._data._row_of


# ------------------------------------------------------------------- access
class ShakespeareSequenceDataset(Dataset):
    """
    Windows of the character stream, addressed by their start offset.

    ``__getitem__`` returns ``(x, y)`` with ``x`` a ``LongTensor`` of
    ``seq_length`` character ids and ``y`` the id of the following character.
    """

    def __init__(self, tokens: np.ndarray, offsets: np.ndarray, seq_length: int = SEQ_LENGTH):
        self.tokens = tokens
        self.offsets = np.asarray(offsets, dtype=np.int64)
        self.seq_length = int(seq_length)

    def __len__(self) -> int:
        return int(self.offsets.size)

    def __getitem__(self, item):
        start = int(self.offsets[item])
        window = self.tokens[start : start + self.seq_length]
        target = int(self.tokens[start + self.seq_length])
        return torch.from_numpy(np.asarray(window, dtype=np.int64)), target

    def eval_payload(self):
        """
        Compact form of the dataset for the evaluation cache.

        The windows are gathered once into a ``[N, seq_length]`` integer tensor,
        which is already what the model consumes, so no decoding is needed.
        """
        if self.offsets.size == 0:
            return None
        columns = np.arange(self.seq_length + 1, dtype=np.int64)
        gathered = np.asarray(self.tokens, dtype=np.int64)[
            self.offsets[:, None] + columns[None, :]
        ]
        windows = torch.from_numpy(np.ascontiguousarray(gathered[:, : self.seq_length]))
        targets = torch.from_numpy(np.ascontiguousarray(gathered[:, self.seq_length]))
        return windows, targets.to(torch.long), IDENTITY


class ShakespeareData:
    """
    Read-only view over a prepared Shakespeare dataset.

    Attributes:
        users: User (``play_role``) ids in storage order.
        tokens: Concatenated ``uint8`` character stream of all users.
    """

    def __init__(self, npz_path: Optional[Path] = None, index_path: Optional[Path] = None):
        self.npz_path = Path(npz_path) if npz_path else config.SHAKESPEARE_NPZ
        self.index_path = Path(index_path) if index_path else config.SHAKESPEARE_INDEX_JSON
        with open(self.index_path) as handle:
            self.index = json.load(handle)

        with np.load(self.npz_path) as archive:
            self.tokens: np.ndarray = archive["tokens"]
            self._user_starts: np.ndarray = archive["user_starts"]
            self._user_lengths: np.ndarray = archive["user_lengths"]
            self._seq_starts: np.ndarray = archive["seq_starts"]
            self._seq_counts: np.ndarray = archive["seq_counts"]
            self.seq_length = int(archive["seq_length"])
            self.vocab_size = int(archive["vocab_size"])

        self.users: List[str] = list(self.index["users"])
        self._row_of: Dict[str, int] = {user: row for row, user in enumerate(self.users)}
        self._offset_cache: Dict[str, np.ndarray] = {}
        self._cache: Optional["WindowCache"] = None
        self._fold_book: Any = _UNSET

    # ------------------------------------------------------------- metadata
    def __len__(self) -> int:
        return len(self.users)

    @property
    def num_sequences(self) -> int:
        return int(self._seq_counts.sum())

    @property
    def num_classes(self) -> int:
        return self.vocab_size

    def all_users(self) -> List[str]:
        return list(self.users)

    # The population commands were written against the NIST dataset's names.
    # A client is a writer there and a (play, role) here; the vocabulary
    # differs, the question does not, so both names answer it.
    def all_writers(self) -> List[str]:
        """Alias of :meth:`all_users`, for the writer-shaped call sites."""
        return self.all_users()

    def get_sample_count(self, writer_id) -> int:
        """Alias of :meth:`sample_count`, for the writer-shaped call sites."""
        return self.sample_count(writer_id)

    def sample_count(self, user: str) -> int:
        """Number of 80-character windows the user contributes."""
        return int(self._seq_counts[self._row_of[user]])

    def user_tokens(self, user: str) -> np.ndarray:
        row = self._row_of[user]
        start = int(self._user_starts[row])
        return self.tokens[start : start + int(self._user_lengths[row])]

    def user_text(self, user: str) -> str:
        return "".join(ALL_LETTERS[token] for token in self.user_tokens(user))

    # -------------------------------------------------------------- windows
    def window_offsets(self, user: str) -> np.ndarray:
        """Absolute start offsets of the user's windows inside ``tokens``."""
        cached = self._offset_cache.get(user)
        if cached is None:
            row = self._row_of[user]
            start = int(self._user_starts[row])
            count = int(self._seq_counts[row])
            cached = np.arange(start, start + count, dtype=np.int64)
            self._offset_cache[user] = cached
        return cached

    def targets(self, offsets: np.ndarray) -> np.ndarray:
        """Next-character label of every window start in ``offsets``."""
        return self.tokens[np.asarray(offsets, dtype=np.int64) + self.seq_length]

    def sequences(self, user: str) -> Tuple[np.ndarray, np.ndarray]:
        """``(X, y)`` of one user: ``X`` is ``[n, seq_length]``, ``y`` is ``[n]``."""
        offsets = self.window_offsets(user)
        windows = np.lib.stride_tricks.sliding_window_view(self.tokens, self.seq_length)
        return windows[offsets], self.targets(offsets)

    @property
    def X(self) -> np.ndarray:
        """All windows as a ``[Nseq, seq_length]`` ``uint8`` array."""
        return np.concatenate([self.sequences(user)[0] for user in self.users])

    @property
    def y(self) -> np.ndarray:
        """All targets as a ``[Nseq]`` ``uint8`` array."""
        return np.concatenate([self.targets(self.window_offsets(u)) for u in self.users])

    # ------------------------------------------------------ the cache facade
    #
    # Everything below exists so that the fold-book machinery - which was built
    # for the packed NIST cache and addresses samples by ROW - can address
    # Shakespeare windows too.  The book is the study's split record: it is what
    # lets two runs of one fold see the same rows, and what lets a reviewer
    # check a split rather than only a result.  Re-deriving that machinery for a
    # second dataset would have meant two split rules to keep in agreement, so
    # instead this presents the shape the existing one already reads.
    #
    # THE ROW SPACE IS THE DENSE WINDOW INDEX, 0 .. num_sequences-1, not the
    # token offset.  Token offsets are the natural address here - a window IS a
    # position in the character stream - but they leave gaps: the last 80
    # positions of every user start no window, and no user owns them.  The book
    # maps every row to a writer through ``writer_index``, and a row owned by
    # nobody would be attributed to whichever writer sat at index -1.  A dense
    # index has no such holes, so the question cannot arise.

    @property
    def cache_dir(self) -> Path:
        """Where the packed arrays live; recorded in a book's metadata."""
        return self.npz_path.parent

    @property
    def classes(self) -> str:
        """The label space, for a book's metadata: the character vocabulary."""
        return f"chars{self.vocab_size}"

    @property
    def cache(self) -> "WindowCache":
        """The row-addressed view of this dataset the fold book builds against."""
        if self._cache is None:
            self._cache = WindowCache(self)
        return self._cache

    def window_rows(self, user: str) -> np.ndarray:
        """The user's dense window rows, ``[n]``."""
        row = self._row_of[user]
        start = int(self._seq_starts[row])
        return np.arange(start, start + int(self._seq_counts[row]), dtype=np.int64)

    def offsets_of_rows(self, rows) -> np.ndarray:
        """
        Token offsets of dense window rows.

        The two spaces differ by each user's own displacement, so the map is a
        per-row lookup rather than a constant shift.
        """
        rows = np.asarray(rows, dtype=np.int64)
        if rows.size == 0:
            return rows
        owner = self.cache.writer_index[rows]
        return self._user_starts[owner] + (rows - self._seq_starts[owner])

    def row_labels(self, rows) -> np.ndarray:
        """Next-character label of each dense window row."""
        return self.targets(self.offsets_of_rows(rows))

    def writer_samples(self, writers=None) -> Mapping:
        """
        ``{user: [(row, label), ...]}`` over the dense window rows.

        Lazy on purpose.  The corpus holds four million windows, and
        materialising every one as a Python tuple costs hundreds of megabytes to
        build a book that usually covers ten users.  This computes a user's list
        only when it is asked for, while still answering ``in``, ``len`` and
        iteration over the full population - which is all the book builder needs
        to decide what it covers.
        """
        return _LazyWriterSamples(self, writers)

    def rows_dataset(self, rows, labels=None):
        """
        A torch dataset over explicit dense window rows.

        The generic evaluation loader asks the dataset for this rather than
        constructing an image dataset itself, which is the single change that
        lets every book-addressed command - scoring, evaluation, the isolated
        and centralized arms - run on a modality that is not images.

        Args:
            rows: Dense window rows.
            labels: Ignored; the label of a window is the character that
                follows it and is recovered from the stream. Accepted so the
                call site does not need to know that.
        """
        return ShakespeareSequenceDataset(
            self.tokens, self.offsets_of_rows(rows), self.seq_length
        )

    # ------------------------------------------------------------- the book
    @property
    def fold_book(self):
        """The configured fold book, or ``None``; loaded once."""
        if self._fold_book is _UNSET:
            path = config.fold_book()
            if not path:
                self._fold_book = None
            else:
                from federated_outlier_adaptation.data.fold_book import FoldBook

                self._fold_book = FoldBook.load(path)
        return self._fold_book

    def book_split(self, writers, fold):
        """
        The three parts of ``writers`` in ``fold``, from the book.

        ``None`` when there is no book, no fold, or the book does not cover
        every requested writer - in which case the caller falls back to the
        seeded split, exactly as it does for NIST.
        """
        book = self.fold_book
        if book is None or fold is None:
            return None
        if not all(book.covers(writer) for writer in writers):
            return None
        return book.split(fold, writers)

    # -------------------------------------------------------------- loaders
    def build_dataset(
        self,
        users,
        seed: Optional[int] = 42,
        train_rate: float = 0.6,
        eval_rate: float = 0.2,
        is_stratified: bool = True,
        batch_size: int = 64,
        loader_seed: Optional[int] = None,
    ):
        """
        Train/validation/test DataLoaders for one or more users.

        The split rule is the one NIST uses: the users' samples are pooled,
        grouped by label (the next character), shuffled with ``seed`` and cut at
        ``train_rate`` / ``eval_rate``.
        """
        if isinstance(users, str):
            users = [users]
        known = [user for user in users if user in self._row_of]
        if not known:
            return None, None, None

        def loader_of(offsets_part: np.ndarray):
            if offsets_part is None or np.asarray(offsets_part).size == 0:
                return None
            dataset = ShakespeareSequenceDataset(
                self.tokens, np.asarray(offsets_part, dtype=np.int64), self.seq_length
            )
            generator = make_generator(loader_seed)
            if generator is None:
                return DataLoader(dataset, batch_size=batch_size, shuffle=True)
            return DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)

        # A fold book, when one is configured and covers these users, *is* the
        # split: the rows it recorded are the split, and no rate or seed here
        # can move them. This is what makes one fold mean the same thing in
        # every process that reads it - and, for this study specifically, what
        # keeps g-0's training rows out of the partitions its preservation is
        # measured on. Without it g-0 would train on a seeded cut and be scored
        # on a booked one, and the two would overlap.
        from federated_outlier_adaptation.utils.seeding import configured_fold

        booked = self.book_split(known, configured_fold())
        if booked is not None:
            return tuple(
                loader_of(self.offsets_of_rows(booked[part]))
                for part in ("train", "val", "test")
            )

        offsets = np.concatenate([self.window_offsets(user) for user in known])
        if offsets.size == 0:
            return None, None, None
        labels = self.targets(offsets)

        train, val, test = split_indices(
            offsets,
            labels,
            train_rate=train_rate,
            eval_rate=eval_rate,
            seed=seed,
            is_stratified=is_stratified,
        )

        def loader(part: np.ndarray):
            if part.size == 0:
                return None
            dataset = ShakespeareSequenceDataset(self.tokens, part, self.seq_length)
            generator = make_generator(loader_seed)
            if generator is None:
                return DataLoader(dataset, batch_size=batch_size, shuffle=True)
            return DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)

        return loader(train), loader(val), loader(test)


# ---------------------------------------------------------------- entry point
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare the LEAF Shakespeare dataset from the raw Gutenberg corpus."
    )
    parser.add_argument(
        "--raw",
        default=None,
        help="Raw corpus (.txt or .zip; a zip is read in memory, never extracted).",
    )
    parser.add_argument("--out-dir", default=None, help="Destination of the npz and the index.")
    parser.add_argument("--seq-length", type=int, default=SEQ_LENGTH)
    parser.add_argument(
        "--min-samples",
        type=int,
        default=2,
        help="Drop users with fewer sequences (LEAF's -k).",
    )
    args = parser.parse_args(argv)

    raw = Path(args.raw) if args.raw else config.SHAKESPEARE_RAW_TXT
    summary = prepare(
        raw_path=raw,
        out_dir=args.out_dir,
        seq_length=args.seq_length,
        min_samples=args.min_samples,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
