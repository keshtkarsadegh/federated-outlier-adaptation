"""
Fixtures for the additional datasets (LEAF Shakespeare, CIFAR-10).

Everything here is synthetic and tiny: a three-role toy corpus and a handful of
random 32x32 images, prepared into the real on-disk format inside ``tmp_path``.
No test touches the real corpora, the real caches or the published results.
"""

from __future__ import annotations

import io
import json
import pickle
import tarfile
from pathlib import Path

import numpy as np
import pytest
import torch

from federated_outlier_adaptation.data import cifar10 as cifar10_data
from federated_outlier_adaptation.data import shakespeare as shakespeare_data
from federated_outlier_adaptation.providers.cifar10 import Cifar10Provider
from federated_outlier_adaptation.providers.shakespeare import ShakespeareProvider

# --------------------------------------------------------------- Shakespeare
ROLE_LINES = {
    "ALPHA": [
        "Now is the winter of our discontent made glorious summer by this sun of York,",
        "And all the clouds that lourd upon our house in the deep bosom of the ocean buried.",
        "Now are our brows bound with victorious wreaths, our bruised arms hung up for monuments.",
    ],
    "BETA": [
        "But I, that am not shaped for sportive tricks, nor made to court an amorous glass,",
        "I, that am rudely stamped, and want loves majesty to strut before a wanton ambling nymph,",
        "Why, I, in this weak piping time of peace, have no delight to pass away the time.",
    ],
    "GAMMA": [
        "And therefore, since I cannot prove a lover to entertain these fair well spoken days,",
        "I am determined to prove a villain and hate the idle pleasures of these days.",
        "Plots have I laid, inductions dangerous, by drunken prophecies, libels and dreams.",
    ],
    "DELTA": [
        "Grim visaged war hath smoothed his wrinkled front, and now, instead of mounting barbed steeds,",
        "He capers nimbly in a ladys chamber to the lascivious pleasing of a lute.",
        "But I, that am not shaped for sportive tricks, nor made to court an amorous looking glass.",
    ],
}

TOY_PLAY_TITLE = "A TRIAL PLAY"
SECOND_PLAY_TITLE = "ANOTHER TRIAL PLAY"


def _play_body(title: str, roles) -> str:
    lines = [
        title,
        "",
        "Contents",
        "",
        "ACT I",
        "Scene I. A hall.",
        "",
        "Dramatis Personae",
        "",
    ]
    lines += [f"{role}, a person of the play." for role in roles]
    lines += ["", "ACT I", "", "SCENE I. A hall.", "", " Enter the players.", ""]
    for role in roles:
        lines.append(f"{role}.")
        lines.extend(ROLE_LINES[role])
        lines.append("")
        lines.append("[_A stage direction that must be discarded._]")
        lines.append("")
    return "\n".join(lines)


def toy_corpus() -> str:
    """A miniature Gutenberg-style corpus with two plays and one poem."""
    head = [
        "The Toy Works of a Playwright",
        "",
        "*** START OF THE PROJECT GUTENBERG EBOOK THE TOY WORKS ***",
        "",
        "                    Contents",
        "",
        f"    THE SONNETS",
        f"    {TOY_PLAY_TITLE}",
        f"    {SECOND_PLAY_TITLE}",
        "",
        "",
        "",
        "THE SONNETS",
        "",
        "A verse with no speaking roles at all, which must not become a user.",
        "",
    ]
    tail = ["", "*** END OF THE PROJECT GUTENBERG EBOOK THE TOY WORKS ***", ""]
    return "\n".join(
        head
        + [_play_body(TOY_PLAY_TITLE, ["ALPHA", "BETA"]), ""]
        + [_play_body(SECOND_PLAY_TITLE, ["GAMMA", "DELTA"]), ""]
        + tail
    )


@pytest.fixture(scope="session")
def toy_corpus_text() -> str:
    return toy_corpus()


@pytest.fixture(scope="session")
def shakespeare_dir(tmp_path_factory, toy_corpus_text) -> Path:
    """A prepared toy Shakespeare dataset (npz + index) in a temp directory."""
    directory = tmp_path_factory.mktemp("shakespeare_data")
    raw = directory / "toy100.txt"
    raw.write_text(toy_corpus_text, encoding="utf-8")
    shakespeare_data.prepare(raw_path=raw, out_dir=directory, min_samples=2, log=lambda *a: None)
    return directory


# ------------------------------------------------------------------ CIFAR-10
CIFAR_IMAGES = 600
CIFAR_CLIENTS = 6


@pytest.fixture(scope="session")
def cifar_archive(tmp_path_factory) -> Path:
    """A synthetic ``cifar-10-python.tar.gz`` with six 20-image batches."""
    directory = tmp_path_factory.mktemp("cifar_raw")
    archive_path = directory / "cifar-10-python.tar.gz"
    rng = np.random.default_rng(7)

    with tarfile.open(archive_path, "w:gz") as archive:
        for index, name in enumerate((*cifar10_data._TRAIN_BATCHES, cifar10_data._TEST_BATCH)):
            rows = 20
            payload = {
                b"data": rng.integers(0, 256, size=(rows, 3072), dtype=np.uint8),
                b"labels": [(index + row) % 10 for row in range(rows)],
            }
            blob = pickle.dumps(payload)
            info = tarfile.TarInfo(f"cifar-10-batches-py/{name}")
            info.size = len(blob)
            archive.addfile(info, io.BytesIO(blob))
    return archive_path


@pytest.fixture(scope="session")
def cifar_dir(tmp_path_factory) -> Path:
    """A prepared toy CIFAR-10 dataset (npz + client index) in a temp directory."""
    directory = tmp_path_factory.mktemp("cifar_data")
    rng = np.random.default_rng(11)

    images = rng.integers(0, 256, size=(CIFAR_IMAGES, 32, 32, 3), dtype=np.uint8)
    labels = np.tile(np.arange(10, dtype=np.uint8), CIFAR_IMAGES // 10)
    is_test = np.zeros(CIFAR_IMAGES, dtype=bool)
    is_test[int(CIFAR_IMAGES * 0.8) :] = True
    np.savez(directory / "cifar10.npz", images=images, labels=labels, is_test=is_test)

    client_ids, pools, proxy_rows, stats = cifar10_data.build_clients(
        labels, is_test, num_clients=CIFAR_CLIENTS, alpha=1.0, seed=3
    )
    index = {
        "dataset": "cifar10",
        "num_classes": 10,
        "class_names": list(cifar10_data.CLASS_NAMES),
        "clients": {cid: pool.tolist() for cid, pool in zip(client_ids, pools)},
        "proxy": proxy_rows.tolist(),
        **stats,
    }
    with open(directory / "cifar10_clients.json", "w") as handle:
        json.dump(index, handle)
    return directory


# ----------------------------------------------------------------- artefacts
def write_artefacts(provider, participants) -> None:
    """Create the frozen artefacts the runners and trainers read."""
    results_dir = Path(provider.results_dir)
    (results_dir / "global_results").mkdir(parents=True, exist_ok=True)
    (results_dir / "global_clients_results").mkdir(parents=True, exist_ok=True)
    (results_dir / "outliers").mkdir(parents=True, exist_ok=True)
    provider.fisher_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    model = provider.make_model()
    torch.save(model.state_dict(), provider.global_model_path)

    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    torch.save({n: torch.ones_like(p) * 1e-3 for n, p in trainable}, provider.fisher_dir / "fisher.pt")
    torch.save({n: p.detach().clone() for n, p in trainable}, provider.fisher_dir / "global_params.pt")

    with open(results_dir / "writer_split.json", "w") as handle:
        json.dump(
            {
                "local_writers": provider.local_client_ids(),
                "global_writers": provider.global_client_ids(),
            },
            handle,
        )
    with open(results_dir / "outliers" / "selected_outliers.json", "w") as handle:
        json.dump(list(participants), handle)

    for name in (
        results_dir / "global_results" / "global_metrics.json",
        results_dir / "global_clients_results" / "global_clients_global_metrics.json",
        results_dir / "global_clients_results" / "global_clients_all_outliers_metrics.json",
    ):
        with open(name, "w") as handle:
            json.dump({"test_accuracy": 0.5}, handle)


@pytest.fixture
def shakespeare_provider(shakespeare_dir, tmp_path) -> ShakespeareProvider:
    """Toy Shakespeare provider with its artefacts on disk."""
    provider = ShakespeareProvider(
        results_dir=tmp_path / "results_shakespeare",
        data_dir=shakespeare_dir,
        global_fraction=0.5,
        min_selection_samples=1,
        model_kwargs={"embed": 4, "hidden": 16, "layers": 1},
    )
    write_artefacts(provider, provider.local_client_ids()[:2])
    provider._outliers_file = provider.results_dir / "outliers" / "selected_outliers.json"
    return provider


@pytest.fixture
def cifar10_provider(cifar_dir, tmp_path) -> Cifar10Provider:
    """Toy CIFAR-10 provider with its artefacts on disk."""
    provider = Cifar10Provider(
        results_dir=tmp_path / "results_cifar10",
        data_dir=cifar_dir,
        num_global_clients=4,
        min_selection_samples=1,
        model_kwargs={"widths": (8, 16), "fc_dim": 16},
    )
    write_artefacts(provider, provider.local_client_ids()[:2])
    provider._outliers_file = provider.results_dir / "outliers" / "selected_outliers.json"
    return provider
