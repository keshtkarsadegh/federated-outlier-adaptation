"""
The qualitative outlier figure: what the low-accuracy writers actually look like.

One row per selected client, one column per character class, and a top row
holding the *reference* population's mean image of that class.  Which writers
count as the reference is the caller's choice and is recorded in the sidecar -
see :func:`_default_source` for why it is not a default.  The point of the
figure is the comparison: the source mean is what the global model was trained
to expect, and a writer whose glyphs sit far from it is what "outlier" means in
this study - a statement about handwriting, not only about an accuracy number.

Why this is not
:func:`~federated_outlier_adaptation.plotting.outlier_samples.render_default_figure`
-----------------------------------------------------------------------------
That function belongs to the published 128x128 digit pipeline: it reads the
digit manifest directly and lays out one column per digit, which is a readable
figure for ten classes and an unreadable one for sixty-two.  It is left
untouched.  This renderer goes through the provider instead, so it works at any
resolution, for any character set and for any dataset that satisfies the
provider contract, and it selects the columns rather than drawing all of them.

Column selection
----------------
With 62 classes a full grid is 62 columns wide.  ``max_columns`` keeps the
classes the selected clients have the most examples of - the ones the figure can
actually say something about - and the chosen labels are recorded in the JSON
sidecar next to the image, so the figure is reproducible and its selection is
explicit rather than implied.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from federated_outlier_adaptation.logging_utils import NistLogger

#: Columns drawn by default; 62 would be unreadable.
DEFAULT_MAX_COLUMNS = 16

#: Rows drawn per client by default.
DEFAULT_EXAMPLES = 1

#: The transform's normalisation, undone for display.
TRANSFORM_MEAN = 0.5
TRANSFORM_STD = 0.5


def _to_display(array: np.ndarray) -> np.ndarray:
    """Undo ``Normalize((0.5,), (0.5,))`` and clip into [0, 1]."""
    return np.clip(array * TRANSFORM_STD + TRANSFORM_MEAN, 0.0, 1.0)


def collect_by_label(loader, limit: Optional[int] = None) -> Dict[int, List[np.ndarray]]:
    """
    Group a loader's images by label.

    Args:
        loader: A ``DataLoader`` yielding ``(images, labels)``; ``None`` gives an
            empty mapping, so a client without data does not need special casing
            at the call site.
        limit: Keep at most this many images per label (``None`` keeps all).

    Returns:
        ``{label: [image, ...]}`` with images as ``[H, W]`` float arrays.
    """
    grouped: Dict[int, List[np.ndarray]] = defaultdict(list)
    if loader is None:
        return grouped
    for images, labels in loader:
        for image, label in zip(images, labels):
            key = int(label)
            if limit is not None and len(grouped[key]) >= limit:
                continue
            grouped[key].append(np.asarray(image).squeeze())
    return grouped


def class_names(provider) -> Dict[int, str]:
    """The dataset's label -> character map, or an empty mapping."""
    dataset = getattr(provider, "dataset", None)
    accessor = getattr(dataset, "class_names", None)
    if callable(accessor):
        try:
            return dict(accessor())
        except (OSError, ValueError, KeyError):  # pragma: no cover - broken cache
            return {}
    return {}


def choose_labels(
    client_images: Dict[str, Dict[int, List[np.ndarray]]],
    source_images: Dict[int, List[np.ndarray]],
    max_columns: int = DEFAULT_MAX_COLUMNS,
    labels: Optional[Sequence[int]] = None,
) -> List[int]:
    """
    The classes the figure draws, most-represented first and then sorted.

    Only classes the source population also has are eligible, because the top
    row is the source mean and a column without one says nothing.
    """
    eligible = set(source_images)
    if labels is not None:
        return [int(label) for label in labels if int(label) in eligible]

    totals: Dict[int, int] = defaultdict(int)
    for grouped in client_images.values():
        for label, images in grouped.items():
            if label in eligible:
                totals[label] += len(images)
    ranked = sorted(totals, key=lambda label: (-totals[label], label))
    return sorted(ranked[: max(1, int(max_columns))])


def _default_source(provider) -> List[str]:
    """
    The reference population when the caller names none.

    This used to be ``provider.global_client_ids()``, which reads the frozen
    ``writer_split.json`` of the published pipeline.  A study that never writes
    one - every study since the fold books - could not draw this figure at all,
    and the failure came from a file the figure has no business knowing about.

    The reference is now the caller's to choose, and rightly so: "what a typical
    writer looks like" is a claim about *which* writers you consider typical, so
    it belongs in the task line where it can be recorded, not in a default. The
    legacy split is still honoured when it happens to exist, so the published
    figure is unchanged; otherwise this says what is missing and asks for it.
    """
    try:
        source = list(provider.global_client_ids())
    except (FileNotFoundError, AttributeError, KeyError) as missing:
        raise ValueError(
            "The gallery needs a reference population to compare against, and "
            "this results root has no writer split to fall back on "
            f"({missing}). Name one with --source-clients-file (a JSON list of "
            "writers) or --source-clients."
        ) from None
    if not source:
        raise ValueError(
            "The writer split names no source writers, so there is nothing to "
            "compare against. Name a reference population with "
            "--source-clients-file."
        )
    return source


def render_outlier_gallery(
    provider,
    clients: Sequence[str],
    out_path,
    source_clients: Optional[Sequence[str]] = None,
    labels: Optional[Sequence[int]] = None,
    n_examples: int = DEFAULT_EXAMPLES,
    max_columns: int = DEFAULT_MAX_COLUMNS,
    batch_size: int = 256,
) -> Dict[str, Any]:
    """
    Draw the source-versus-clients gallery and write it next to a JSON sidecar.

    Args:
        provider: Dataset provider.
        clients: The writers to draw, one row block each.
        out_path: Destination ``.png``; the sidecar takes the same stem.
        source_clients: The source population; defaults to the provider's.
        labels: Explicit column labels; ``None`` selects them (see
            :func:`choose_labels`).
        n_examples: Rows drawn per client.
        max_columns: Upper bound on the number of columns.
        batch_size: Evaluation batch size of the loaders.

    Returns:
        A summary dict: the figure path, the clients and the labels drawn.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    clients = list(clients)
    if not clients:
        raise ValueError("The outlier gallery needs at least one client to draw.")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    source = list(source_clients) if source_clients else _default_source(provider)
    NistLogger.info(
        f"Outlier gallery: {len(clients)} client(s) against {len(source)} source writer(s)"
    )
    source_images = collect_by_label(
        provider.build_dataset(
            source, train_rate=0.0, eval_rate=0.0, batch_size=batch_size
        )[2]
    )
    client_images = {
        client: collect_by_label(
            provider.build_dataset(
                client, train_rate=0.0, eval_rate=0.0, batch_size=batch_size
            )[2],
            limit=n_examples,
        )
        for client in clients
    }

    drawn = choose_labels(client_images, source_images, max_columns, labels)
    if not drawn:
        raise ValueError(
            "No class is present in both the source population and the selected "
            "clients, so there is nothing to compare."
        )

    names = class_names(provider)
    rows = 1 + len(clients) * max(1, int(n_examples))
    figure, axes = plt.subplots(
        rows, len(drawn), figsize=(1.1 * len(drawn), 1.2 * rows), squeeze=False
    )

    for column, label in enumerate(drawn):
        mean_image = np.mean(np.stack(source_images[label]), axis=0)
        axis = axes[0][column]
        axis.imshow(_to_display(mean_image), cmap="gray", vmin=0.0, vmax=1.0)
        axis.set_title(str(names.get(label, label)), fontsize=9)
        axis.set_xticks([])
        axis.set_yticks([])
        if column == 0:
            axis.set_ylabel("source\nmean", fontsize=7, rotation=0, ha="right", va="center")

    for index, client in enumerate(clients):
        for example in range(max(1, int(n_examples))):
            row = 1 + index * max(1, int(n_examples)) + example
            for column, label in enumerate(drawn):
                axis = axes[row][column]
                axis.set_xticks([])
                axis.set_yticks([])
                available = client_images[client].get(label) or []
                if example < len(available):
                    axis.imshow(
                        _to_display(available[example]), cmap="gray", vmin=0.0, vmax=1.0
                    )
                else:
                    # The writer has no sample of this class; an empty cell says
                    # so, which is itself part of what makes them an outlier.
                    axis.imshow(np.ones_like(_to_display(np.zeros((2, 2)))), cmap="gray",
                                vmin=0.0, vmax=1.0)
                if column == 0 and example == 0:
                    axis.set_ylabel(client, fontsize=7, rotation=0, ha="right", va="center")

    figure.tight_layout()
    figure.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(figure)

    summary = {
        "figure": str(out_path),
        "clients": clients,
        "source_clients": len(source),
        "labels": [int(label) for label in drawn],
        "class_names": [str(names.get(label, label)) for label in drawn],
        "n_examples": int(n_examples),
        "max_columns": int(max_columns),
    }
    sidecar = out_path.with_suffix(".json")
    with open(sidecar, "w") as handle:
        json.dump(summary, handle, indent=2)
    NistLogger.info(f"Outlier gallery -> {out_path} (and {sidecar.name})")
    return summary
