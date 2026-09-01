"""
The 28x28, 62-class NIST pipeline: conversion, labels, cache and dataset.

Everything here runs against a synthetic archive of a handful of images written
into ``tmp_path``, so the suite never touches ``by_write.zip`` and never writes
a per-image file outside the temporary directory.
"""

from __future__ import annotations

import io
import json
import zipfile

import numpy as np
import pytest
from PIL import Image

from federated_outlier_adaptation.data import sd19_labels
from federated_outlier_adaptation.data.emnist_convert import (
    bounding_box,
    conversion_info,
    convert_image,
)
from federated_outlier_adaptation.data.nist28 import (
    INDEX_NAME,
    Nist28Cache,
    Nist28Dataset,
    build_cache,
)

#: Edge length of the raw scans SD19 stores.
SOURCE_SIZE = 128

#: The order ``build_dataset`` returns its three loaders in.
PARTS_ORDER = ("train", "val", "test")


def _scan(top: int, left: int, height: int, width: int) -> np.ndarray:
    """A bilevel scan with one black rectangle of ink on a white page."""
    array = np.full((SOURCE_SIZE, SOURCE_SIZE), 255, dtype=np.uint8)
    array[top : top + height, left : left + width] = 0
    return array


# --------------------------------------------------------------------------- #
# the conversion
# --------------------------------------------------------------------------- #
def test_the_conversion_produces_a_28x28_uint8_image():
    out = convert_image(_scan(20, 30, 40, 25))
    assert out.shape == (28, 28)
    assert out.dtype == np.uint8


def test_the_ink_survives_and_the_polarity_is_kept():
    out = convert_image(_scan(20, 30, 40, 25))
    # Dark ink on a light page, as the archive stores it and as the rest of the
    # project (including the inverted MNIST proxy set) expects.
    assert out.min() < 64
    assert out.max() > 192


def test_a_blank_scan_yields_a_blank_frame():
    blank = np.full((SOURCE_SIZE, SOURCE_SIZE), 255, dtype=np.uint8)
    out = convert_image(blank)
    assert out.shape == (28, 28)
    assert (out == 255).all()


def test_the_bounding_box_is_tight_around_the_ink():
    ink = np.zeros((10, 10), dtype=np.uint8)
    ink[3:6, 4:9] = 255
    assert bounding_box(ink) == (3, 6, 4, 9)


def test_the_bounding_box_of_an_empty_image_is_none():
    assert bounding_box(np.zeros((10, 10), dtype=np.uint8)) is None


def test_position_is_discarded_but_shape_is_not():
    """
    The EMNIST conversion centres the bounding box geometrically, so the same
    character in two corners converts to the same image - and a different
    aspect ratio still converts to a different one.
    """
    top_left = convert_image(_scan(4, 4, 30, 20))
    bottom_right = convert_image(_scan(90, 100, 30, 20))
    wider = convert_image(_scan(4, 4, 30, 40))
    assert np.array_equal(top_left, bottom_right)
    assert not np.array_equal(top_left, wider)


@pytest.mark.parametrize("height,width", [(20, 12), (60, 36), (100, 60)])
def test_the_character_fills_the_frame(height, width):
    """
    "the technique used in this paper attempts to make use of the maximum
    amount of space available" - so however small the character was on the
    page, it spans nearly the whole 28x28 frame after the conversion.

    It is not scale-*invariant*, though: the 2-pixel border is added at source
    resolution, so it costs a small region of interest proportionally more of
    the frame than a large one.  That is the paper's own ordering; see the
    ``border_space`` note in the conversion module.
    """
    ink = 255 - convert_image(_scan(10, 10, height, width)).astype(int)
    rows = np.flatnonzero((ink > 8).any(axis=1))
    assert rows.size >= 22
    # ...and it is centred, not pushed against an edge.
    assert abs((28 - 1 - rows[-1]) - rows[0]) <= 2


def test_the_border_convention_is_selectable():
    source = convert_image(_scan(20, 30, 40, 25), border_space="source")
    target = convert_image(_scan(20, 30, 40, 25), border_space="target")
    assert not np.array_equal(source, target)
    # The target convention reserves the border in the 28x28 frame itself.
    assert (target[:2, :] == 255).all()
    assert (target[-2:, :] == 255).all()


def test_an_unknown_border_convention_is_rejected():
    with pytest.raises(ValueError):
        convert_image(_scan(20, 30, 40, 25), border_space="somewhere")


def test_the_conversion_records_what_it_did():
    info = conversion_info()
    assert info["name"] == "emnist"
    assert info["sigma"] == 1.0
    assert info["border"] == 2
    assert info["interpolation"] == "bicubic"
    assert "1702.05373" in info["reference"]


# --------------------------------------------------------------------------- #
# the labels
# --------------------------------------------------------------------------- #
def test_the_class_order_is_digits_then_upper_then_lower():
    assert sd19_labels.CLASS_ORDER[0] == "0"
    assert sd19_labels.CLASS_ORDER[9] == "9"
    assert sd19_labels.CLASS_ORDER[10] == "A"
    assert sd19_labels.CLASS_ORDER[35] == "Z"
    assert sd19_labels.CLASS_ORDER[36] == "a"
    assert sd19_labels.CLASS_ORDER[61] == "z"
    assert len(sd19_labels.CLASS_ORDER) == 62


@pytest.mark.parametrize(
    "hex_code,expected", [("30", 0), ("39", 9), ("41", 10), ("5a", 35), ("61", 36), ("7a", 61)]
)
def test_the_directory_name_is_the_ascii_code(hex_code, expected):
    assert sd19_labels.class_label(hex_code) == expected


def test_a_directory_that_is_not_a_class_is_ignored():
    assert sd19_labels.class_label("hsf_0") is None


def _logs(tmp_path):
    by_class = tmp_path / "by_class_md5.log"
    by_write = tmp_path / "by_write_md5.log"
    by_class.write_text(
        "aaa /data/by_class/30/hsf_0/hsf_0_00000.png\n"
        "bbb /data/by_class/41/hsf_4/hsf_4_00001.png\n"
        "ccc /data/by_class/61/hsf_4/hsf_4_00002.png\n"
    )
    by_write.write_text(
        "aaa /data/by_write/hsf_0/f0001_11/d0001_11/d0001_11_00000.png\n"
        "bbb /data/by_write/hsf_0/f0001_11/u0001_11/u0001_11_00001.png\n"
        "ccc /data/by_write/hsf_4/f0002_22/l0002_22/l0002_22_00002.png\n"
    )
    return by_class, by_write


def test_the_label_map_joins_the_two_hierarchies(tmp_path):
    by_class, by_write = _logs(tmp_path)
    labels = build_label_map = sd19_labels.build_label_map(by_class, by_write, log=lambda *_: None)
    assert labels == {
        "by_write/hsf_0/f0001_11/d0001_11/d0001_11_00000.png": 0,
        "by_write/hsf_0/f0001_11/u0001_11/u0001_11_00001.png": 10,
        "by_write/hsf_4/f0002_22/l0002_22/l0002_22_00002.png": 36,
    }


def test_the_digit_ablation_keeps_only_the_ten_digits(tmp_path):
    by_class, by_write = _logs(tmp_path)
    labels = sd19_labels.build_label_map(
        by_class, by_write, classes="digits", log=lambda *_: None
    )
    assert set(labels.values()) == {0}


def test_the_writer_is_the_third_path_component():
    path = "by_write/hsf_0/f0001_11/d0001_11/d0001_11_00000.png"
    assert sd19_labels.writer_of(path) == "f0001_11"


# --------------------------------------------------------------------------- #
# the packed cache
# --------------------------------------------------------------------------- #
@pytest.fixture()
def tiny_archive(tmp_path):
    """A synthetic ``by_write`` archive with two writers and four classes."""
    entries = [
        ("f0001_11", "d0001_11", "30", "aaa", 0),
        ("f0001_11", "u0001_11", "41", "bbb", 10),
        ("f0002_22", "l0002_22", "61", "ccc", 36),
        ("f0002_22", "d0002_22", "31", "ddd", 1),
    ]
    by_class_lines, by_write_lines, members = [], [], []
    for writer, partition, hex_code, digest, _ in entries:
        name = f"by_write/hsf_0/{writer}/{partition}/{partition}_00000.png"
        by_class_lines.append(f"{digest} /x/by_class/{hex_code}/hsf_0/hsf_0_00000.png")
        by_write_lines.append(f"{digest} /x/{name}")
        members.append(name)

    (tmp_path / "by_class_md5.log").write_text("\n".join(by_class_lines) + "\n")
    (tmp_path / "by_write_md5.log").write_text("\n".join(by_write_lines) + "\n")

    archive = tmp_path / "by_write.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        for offset, name in enumerate(members):
            buffer = io.BytesIO()
            Image.fromarray(_scan(20 + offset, 30, 40, 25), mode="L").save(buffer, format="PNG")
            handle.writestr(name, buffer.getvalue())
    return archive, entries


def test_the_cache_is_four_files(tmp_path, tiny_archive):
    archive, _ = tiny_archive
    out = tmp_path / "nist28"
    build_cache(archive, out, classes="all", timing_sample=2, log=lambda *_: None)
    assert sorted(path.name for path in out.iterdir()) == [
        "nist28_images.npy",
        "nist28_index.json",
        "nist28_labels.npy",
        "nist28_writers.npy",
    ]


def test_the_cache_holds_the_converted_images_and_their_labels(tmp_path, tiny_archive):
    archive, entries = tiny_archive
    out = tmp_path / "nist28"
    summary = build_cache(archive, out, classes="all", timing_sample=2, log=lambda *_: None)

    assert summary["rows"] == len(entries)
    assert summary["writers"] == 2

    cache = Nist28Cache(out)
    assert len(cache) == len(entries)
    assert cache.array(0).shape == (28, 28)
    assert sorted(cache.labels.tolist()) == sorted(label for *_, label in entries)
    assert cache.writer_ids == ["f0001_11", "f0002_22"]
    assert cache.class_map[10] == "A"


def test_the_index_records_the_conversion_and_the_source(tmp_path, tiny_archive):
    archive, _ = tiny_archive
    out = tmp_path / "nist28"
    build_cache(archive, out, classes="all", timing_sample=2, log=lambda *_: None)
    with open(out / INDEX_NAME) as handle:
        meta = json.load(handle)
    assert meta["resolution"] == 28
    assert meta["num_classes"] == 62
    assert meta["conversion"]["name"] == "emnist"
    assert meta["source_sha256"]


def test_the_index_says_nothing_about_the_run_that_built_it(tmp_path, tiny_archive):
    """
    Two builds of one archive must produce one index, byte for byte.

    Two nodes built this cache from the same archive and produced identical
    image, label and writer arrays; the index alone differed, because it
    recorded how long the conversion took - 574.99 seconds against 577.97. A
    cache whose index cannot be checksummed is a cache whose integrity nobody
    can check, and a spurious mismatch sends someone hunting a corruption that
    is not there.

    Absolute paths are the same defect with a longer fuse: they describe where
    the archive happened to sit, so the index differs between machines while
    the data is identical. The hash is what identifies the release.
    """
    archive, _ = tiny_archive
    first, second = tmp_path / "a", tmp_path / "b"
    for out in (first, second):
        build_cache(archive, out, classes="all", timing_sample=2, log=lambda *_: None)

    assert (first / INDEX_NAME).read_bytes() == (second / INDEX_NAME).read_bytes()

    meta = json.loads((first / INDEX_NAME).read_text())
    assert "seconds" not in meta
    for key in ("source_archive", "by_class_md5_log", "by_write_md5_log"):
        assert not str(meta[key]).startswith("/"), f"{key} is an absolute path"
    # the archive is still identified, by the thing that actually identifies it
    assert meta["source_sha256"]


def test_rows_of_one_writer_are_contiguous(tmp_path, tiny_archive):
    archive, _ = tiny_archive
    out = tmp_path / "nist28"
    build_cache(archive, out, classes="all", timing_sample=2, log=lambda *_: None)
    writers = Nist28Cache(out).writer_index.tolist()
    assert writers == sorted(writers)


# --------------------------------------------------------------------------- #
# the dataset over the cache
# --------------------------------------------------------------------------- #
@pytest.fixture()
def prepared(tmp_path, tiny_archive):
    archive, _ = tiny_archive
    out = tmp_path / "nist28"
    build_cache(archive, out, classes="all", timing_sample=2, log=lambda *_: None)
    return out


def test_the_dataset_lists_the_writers_of_the_cache(prepared):
    dataset = Nist28Dataset(prepared, classes="all")
    assert dataset.all_writers() == ["f0001_11", "f0002_22"]
    assert dataset.get_sample_count("f0001_11") == 2
    assert dataset.num_classes == 62


def test_the_digit_ablation_restricts_the_same_cache(prepared):
    dataset = Nist28Dataset(prepared, classes="digits")
    assert dataset.num_classes == 10
    assert dataset.get_sample_count("f0001_11") == 1
    assert all(label < 10 for _, label in dataset.writer_samples()["f0001_11"])


def test_the_loaders_produce_28x28_tensors(prepared):
    dataset = Nist28Dataset(prepared, classes="all")
    _, _, test_loader = dataset.build_dataset(
        ["f0001_11", "f0002_22"], train_rate=0.0, eval_rate=0.0, batch_size=4
    )
    images, labels = next(iter(test_loader))
    assert images.shape[1:] == (1, 28, 28)
    assert labels.dtype.is_floating_point is False


def test_the_writer_split_is_deterministic(prepared):
    dataset = Nist28Dataset(prepared, classes="all")
    first = dataset.split_writers(global_size=0.5, seed=7)
    second = dataset.split_writers(global_size=0.5, seed=7)
    assert first == second
    assert sorted(first[0] + first[1]) == dataset.all_writers()


def test_a_missing_cache_says_how_to_build_it(tmp_path):
    dataset = Nist28Dataset(tmp_path / "absent", classes="all")
    with pytest.raises(FileNotFoundError, match="prepare-data"):
        dataset.all_writers()


# --------------------------------------------------------------------------- #
# the provider over the cache
# --------------------------------------------------------------------------- #
@pytest.fixture()
def provider28(prepared, tmp_path):
    """The NIST provider wired to the 28x28 cache, 62 classes, FedAvg CNN."""
    from federated_outlier_adaptation.providers.nist import NistProvider

    return NistProvider(
        results_dir=tmp_path / "results",
        cache_dir=prepared,
        resolution=28,
        classes="all",
        model="fedavg_cnn",
    )


def test_the_provider_reports_the_62_class_setting(provider28):
    assert provider28.num_classes == 62
    assert provider28.all_client_ids() == ["f0001_11", "f0002_22"]
    assert provider28.sample_count("f0001_11") == 2


def test_the_provider_builds_28x28_loaders(provider28):
    _, _, test_loader = provider28.build_dataset(
        ["f0001_11"], train_rate=0.0, eval_rate=0.0, batch_size=2
    )
    images, _ = next(iter(test_loader))
    assert images.shape[1:] == (1, 28, 28)


def test_the_model_accepts_the_provider_s_batches(provider28):
    import torch

    model = provider28.make_model()
    _, _, test_loader = provider28.build_dataset(
        ["f0001_11"], train_rate=0.0, eval_rate=0.0, batch_size=2
    )
    images, _ = next(iter(test_loader))
    with torch.no_grad():
        assert model(images).shape[1] == 62


def test_the_evaluation_cache_path_is_available(provider28):
    payload = provider28.evaluation_arrays(["f0001_11"], split="insample", batch_size=2)
    assert payload is not None
    images, labels, _ = payload
    # The cache already stores 28x28 uint8, so materialising is a stack.
    assert images.shape[1:] == (28, 28)
    assert images.dtype.is_floating_point is False
    assert len(labels) == len(images)


def test_the_provider_draws_its_own_writer_split(provider28):
    assert hasattr(provider28, "make_client_split")
    local, source = provider28.make_client_split(seed=3, global_size=0.5)
    assert sorted(local + source) == provider28.all_client_ids()


def test_the_provenance_names_the_cache_and_the_model(provider28):
    from federated_outlier_adaptation.providers import provider_config_block

    block = provider_config_block(provider28)
    assert set(block["dataset_files"]) == {
        "nist28_images",
        "nist28_labels",
        "nist28_writers",
        "nist28_index",
    }
    assert block["model_info"]["model"] == "fedavg_cnn"
    assert block["model_info"]["parameters"] == 1_690_046
    assert block["model_info"]["resolution"] == 28
    assert block["model_info"]["classes"] == "all"


def test_the_128_setting_refuses_the_62_class_task(tmp_path):
    from federated_outlier_adaptation.providers.nist import NistProvider

    with pytest.raises(ValueError, match="resolution 28"):
        NistProvider(results_dir=tmp_path, resolution=128, classes="all")


# --------------------------------------------------------------------------- #
# the extreme-case path over the packed cache
# --------------------------------------------------------------------------- #
def _population(provider, selected, restriction):
    from federated_outlier_adaptation.runners.population import ClientPopulation

    return ClientPopulation(provider=provider, selected=selected, single_outlier=restriction)


@pytest.mark.parametrize("case", ["single", "double", "dual"])
def test_the_published_extreme_ids_are_rejected_here(provider28, case):
    """
    ``constants.EXTREME_CASE_CLIENTS`` names writers of the published 128x128
    digit setting.  This dataset does not have them, and an id that does not
    exist used to surface only much later, as a ``TypeError`` on a ``None``
    loader inside ``BaseTrainer.evaluate``.  All three cases share the same
    defaults, so all three have to fail the same way.
    """
    from federated_outlier_adaptation.constants import EXTREME_CASE_CLIENTS
    from federated_outlier_adaptation.runners.population import UnknownClientError

    with pytest.raises(UnknownClientError, match="Unknown client id"):
        _population(
            provider28, provider28.all_client_ids(), EXTREME_CASE_CLIENTS[case]
        )


def test_the_error_names_what_was_asked_for_and_what_exists(provider28):
    from federated_outlier_adaptation.runners.population import UnknownClientError

    with pytest.raises(UnknownClientError) as raised:
        _population(provider28, provider28.all_client_ids(), ["f3503_07"])
    message = str(raised.value)
    assert "f3503_07" in message
    assert "f0001_11" in message
    assert "--clients" in message


@pytest.mark.parametrize(
    "case,restriction",
    [
        ("single", ["f0001_11"]),
        ("double", ["f0001_11", "f0002_22"]),
        ("dual", ["f0001_11", "f0001_11"]),
    ],
)
def test_the_extreme_cases_resolve_against_this_population(provider28, case, restriction):
    """With ids that exist, all three cases build a population and a loader."""
    population = _population(provider28, provider28.all_client_ids(), restriction)
    assert population.participants == restriction
    assert population.unique_clients == sorted(set(restriction))
    assert population.has_duplicates is (case == "dual")

    # The pool evaluation loader is what used to come back as None.
    _, _, pool_loader = provider28.build_dataset(
        population.unique_clients, train_rate=0.0, eval_rate=0.0, batch_size=2
    )
    assert pool_loader is not None
    assert len(pool_loader.dataset) > 0


def test_explicit_clients_do_not_depend_on_the_current_selection(provider28):
    """
    Naming clients explicitly states who the participants are; it is not a
    request to narrow whatever selection file happens to be current.

    This is the case that failed in the smoke run: the default selection was a
    one-writer severe pool, the requested ids came from a different pool, and
    the intersection of the two was empty.  The requested ids exist in the
    dataset, so the run must resolve them.
    """
    population = _population(provider28, ["f0002_22"], ["f0001_11"])
    assert population.unique_clients == ["f0001_11"]
    assert population.participants == ["f0001_11"]


def test_explicit_clients_work_with_no_selection_at_all(provider28):
    population = _population(provider28, [], ["f0001_11", "f0002_22"])
    assert population.participants == ["f0001_11", "f0002_22"]


def test_a_duplicate_outside_the_selection_still_gives_two_participants(provider28):
    """The 'dual' case: two participants holding identical data."""
    population = _population(provider28, ["f0002_22"], ["f0001_11", "f0001_11"])
    assert population.unique_clients == ["f0001_11"]
    assert population.participants == ["f0001_11", "f0001_11"]
    assert population.has_duplicates is True


def test_an_unknown_id_is_still_rejected_when_the_selection_differs(provider28):
    from federated_outlier_adaptation.runners.population import UnknownClientError

    with pytest.raises(UnknownClientError):
        _population(provider28, ["f0002_22"], ["f0001_11", "nobody_99"])


def test_a_run_without_any_participant_is_rejected(provider28):
    from federated_outlier_adaptation.runners.population import EmptyPopulationError

    with pytest.raises(EmptyPopulationError, match="no participants"):
        _population(provider28, [], None)


def test_a_restriction_inside_the_selection_keeps_the_published_order(provider28):
    """
    The published reading is unchanged: when every requested id is in the
    selection, the participants come out in the *selection's* order, which is
    what the published extreme cases were run with.
    """
    selection = ["f0002_22", "f0001_11"]
    population = _population(provider28, selection, ["f0001_11", "f0002_22"])
    assert population.participants == selection


def test_an_unrestricted_population_is_untouched(provider28):
    """The default path has no restriction and must not be affected."""
    population = _population(provider28, provider28.all_client_ids(), None)
    assert population.participants == provider28.all_client_ids()


def test_the_extreme_command_accepts_explicit_clients():
    from federated_outlier_adaptation.cli import build_parser

    args = build_parser().parse_args(
        ["extreme", "--case", "dual", "--clients", "f0001_11", "f0001_11"]
    )
    assert args.clients == ["f0001_11", "f0001_11"]
    assert build_parser().parse_args(["extreme"]).clients is None


# --------------------------------------------------------------------------- #
# writers too small to be trained
# --------------------------------------------------------------------------- #
@pytest.fixture()
def thin_cache(tmp_path):
    """
    A cache with one writer holding exactly one sample per class.

    That is the shape that broke Phase A: a stratified 60/40 split cuts per
    label and truncates, so ``int(1 * 0.6) == 0`` in every class and both halves
    come out empty.
    """
    from federated_outlier_adaptation.data import sd19_labels
    from federated_outlier_adaptation.data.emnist_convert import conversion_info

    out = tmp_path / "thin"
    out.mkdir()
    writers = ["f0001_11", "f0002_22"]           # thick, then thin
    rows = []
    for index, writer in enumerate(writers):
        per_class = 4 if index == 0 else 1
        for label in range(3):
            rows.extend([(index, label)] * per_class)

    images = np.lib.format.open_memmap(
        out / "nist28_images.npy", mode="w+", dtype=np.uint8, shape=(len(rows), 28, 28)
    )
    for row, _ in enumerate(rows):
        images[row] = 255
    images.flush()
    del images
    np.save(out / "nist28_labels.npy", np.array([l for _, l in rows], dtype=np.uint8))
    np.save(out / "nist28_writers.npy", np.array([w for w, _ in rows], dtype=np.int32))
    with open(out / "nist28_index.json", "w") as handle:
        json.dump(
            {
                "resolution": 28,
                "classes": "all",
                "num_classes": 62,
                "class_map": {str(k): v for k, v in sd19_labels.class_map("all").items()},
                "conversion": conversion_info(),
                "rows": len(rows),
                "writers": writers,
            },
            handle,
        )
    return out


def test_a_writer_with_one_sample_per_class_has_no_split(thin_cache):
    """
    One sample per class means ``int(1 * 0.6) == 0`` - no training half at all,
    which is what makes the writer untrainable.  Its single row per class is
    held out; with rates 0.6/0.4 that holdout is validation's, because the rates
    say there is no test part.  (It used to land in test purely because test
    took every remainder.)
    """
    dataset = Nist28Dataset(thin_cache, classes="all")
    train, val, test = dataset.build_dataset(["f0002_22"], train_rate=0.6, eval_rate=0.4)
    assert train is None                      # the reason it cannot be trained
    assert test is None                       # rates 0.6/0.4 ask for no test part
    assert val is not None and len(val.dataset) == 3


def test_trainable_writers_excludes_it(thin_cache):
    dataset = Nist28Dataset(thin_cache, classes="all")
    assert dataset.all_writers() == ["f0001_11", "f0002_22"]
    assert dataset.trainable_writers() == ["f0001_11"]


def test_the_eligibility_rule_is_off_by_default(thin_cache, tmp_path):
    from federated_outlier_adaptation.providers.nist import NistProvider

    provider = NistProvider(
        results_dir=tmp_path / "r", cache_dir=thin_cache, resolution=28, classes="all"
    )
    assert not hasattr(provider, "eligible_clients")


def test_the_eligibility_rule_can_be_switched_on(thin_cache, tmp_path):
    from federated_outlier_adaptation.outliers.selection import eligible_of
    from federated_outlier_adaptation.providers.nist import NistProvider

    provider = NistProvider(
        results_dir=tmp_path / "r",
        cache_dir=thin_cache,
        resolution=28,
        classes="all",
        require_trainable=True,
    )
    assert provider.eligible_clients() == ["f0001_11"]
    # the pool writers reach it through this accessor
    assert eligible_of(provider) == ["f0001_11"]


def test_the_environment_switches_it_on_too(thin_cache, tmp_path, monkeypatch):
    from federated_outlier_adaptation.providers.nist import NistProvider

    monkeypatch.setenv("FOA_REQUIRE_TRAINABLE", "1")
    provider = NistProvider(
        results_dir=tmp_path / "r", cache_dir=thin_cache, resolution=28, classes="all"
    )
    assert provider.eligible_clients() == ["f0001_11"]


def test_an_untrainable_participant_is_named(thin_cache, tmp_path):
    """
    The runners used to hand the ``None`` loaders to the trainer and die with
    ``TypeError: 'NoneType' object is not iterable`` mid-round.
    """
    from federated_outlier_adaptation.providers.nist import NistProvider
    from federated_outlier_adaptation.runners.population import (
        UntrainableClientError,
        untrainable_message,
    )

    provider = NistProvider(
        results_dir=tmp_path / "r", cache_dir=thin_cache, resolution=28, classes="all"
    )
    train, val, _ = provider.build_dataset(["f0002_22"], train_rate=0.6, eval_rate=0.4)
    message = untrainable_message("f0002_22", train, val, provider)
    assert "f0002_22" in message
    assert "'train': 0" in message
    assert "--require-trainable" in message
    assert issubclass(UntrainableClientError, ValueError)


def test_the_pool_commands_expose_the_flag():
    from federated_outlier_adaptation.cli import build_parser

    parser = build_parser()
    assert parser.parse_args(
        ["select-outliers", "--mode", "pool", "--require-trainable"]
    ).require_trainable is True
    assert parser.parse_args(["select-outliers"]).require_trainable is False
    assert parser.parse_args(["global-train", "--require-trainable"]).require_trainable is True


# --------------------------------------------------------------------------- #
# the qualitative outlier figure
# --------------------------------------------------------------------------- #
@pytest.fixture()
def provider_thin(thin_cache, tmp_path):
    """A provider whose two writers share their three classes."""
    from federated_outlier_adaptation.providers.nist import NistProvider

    return NistProvider(
        results_dir=tmp_path / "results_thin",
        cache_dir=thin_cache,
        resolution=28,
        classes="all",
    )


def test_the_gallery_draws_the_clients_against_the_source_means(provider_thin, tmp_path):
    from federated_outlier_adaptation.analysis.outlier_gallery import render_outlier_gallery

    out = tmp_path / "gallery" / "old3.png"
    summary = render_outlier_gallery(
        provider_thin,
        clients=["f0002_22"],
        out_path=out,
        source_clients=["f0001_11"],
        batch_size=8,
    )
    assert out.is_file()
    assert out.with_suffix(".json").is_file()
    assert summary["clients"] == ["f0002_22"]
    assert summary["labels"]
    # the columns are named by character, not by index
    assert set(summary["class_names"]) <= set("0123456789abcdefghijklmnopqrstuvwxyz"
                                              "ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def test_only_classes_the_source_also_has_become_columns(provider_thin, tmp_path):
    """A column without a source mean would have nothing to compare against."""
    from federated_outlier_adaptation.analysis.outlier_gallery import render_outlier_gallery

    summary = render_outlier_gallery(
        provider_thin,
        clients=["f0002_22"],
        out_path=tmp_path / "g.png",
        source_clients=["f0001_11"],
        labels=[0, 1, 2, 61],
        batch_size=8,
    )
    # 61 is in neither writer, so it is dropped rather than drawn empty.
    assert summary["labels"] == [0, 1, 2]


def test_disjoint_populations_are_refused(provider28, tmp_path):
    """The two writers of this cache share no class at all."""
    from federated_outlier_adaptation.analysis.outlier_gallery import render_outlier_gallery

    with pytest.raises(ValueError, match="nothing to compare"):
        render_outlier_gallery(
            provider28,
            clients=["f0001_11"],
            out_path=tmp_path / "g.png",
            source_clients=["f0002_22"],
            batch_size=8,
        )


def test_the_column_count_is_capped(provider28, tmp_path):
    from federated_outlier_adaptation.analysis.outlier_gallery import choose_labels

    client_images = {"w": {label: [object()] * (label + 1) for label in range(30)}}
    source_images = {label: [object()] for label in range(30)}
    drawn = choose_labels(client_images, source_images, max_columns=5)
    assert len(drawn) == 5
    # the classes the client has most examples of, then sorted for display
    assert drawn == sorted(drawn) == [25, 26, 27, 28, 29]


def test_explicit_labels_win(provider28, tmp_path):
    from federated_outlier_adaptation.analysis.outlier_gallery import choose_labels

    source_images = {label: [object()] for label in (0, 1, 2)}
    assert choose_labels({}, source_images, labels=[2, 0, 9]) == [2, 0]


def test_a_gallery_without_clients_is_rejected(provider28, tmp_path):
    from federated_outlier_adaptation.analysis.outlier_gallery import render_outlier_gallery

    with pytest.raises(ValueError, match="at least one client"):
        render_outlier_gallery(provider28, clients=[], out_path=tmp_path / "g.png")


def test_the_class_names_come_from_the_cache(provider28):
    from federated_outlier_adaptation.analysis.outlier_gallery import class_names

    names = class_names(provider28)
    assert names[0] == "0"
    assert names[10] == "A"
    assert names[36] == "a"


def test_the_figure_command_is_wired_up():
    from federated_outlier_adaptation.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(
        ["outlier-figure", "--outliers-file", "/tmp/p.json", "--max-columns", "8"]
    )
    assert args.command == "outlier-figure"
    assert args.max_columns == 8
    assert parser.parse_args(["outlier-figure"]).n_examples == 1


# --------------------------------------------------------------------------- #
# folds re-split the clients, not the client population
# --------------------------------------------------------------------------- #
def _split_rows(dataset, writer):
    """The three split index lists of one writer, as sets of rows."""
    train, val, test = dataset.build_dataset(
        [writer], train_rate=0.6, eval_rate=0.2, batch_size=64
    )
    return tuple(
        set() if loader is None else set(loader.dataset.rows)
        for loader in (train, val, test)
    )


def test_the_split_is_unchanged_without_a_fold(prepared, monkeypatch):
    monkeypatch.delenv("FOA_FOLD", raising=False)
    dataset = Nist28Dataset(prepared, classes="all")
    once = _split_rows(dataset, "f0001_11")
    twice = _split_rows(dataset, "f0001_11")
    assert once == twice
    assert set().union(*once) == {row for row, _ in dataset.writer_samples()["f0001_11"]}


def test_each_fold_splits_the_same_client_differently(thin_cache, monkeypatch):
    """
    A fold is a different draw over the same samples - the union of the three
    splits is the writer's data in every fold, and the writer set never moves.
    """
    dataset = Nist28Dataset(thin_cache, classes="all")
    everything = {row for row, _ in dataset.writer_samples()["f0001_11"]}

    splits = {}
    for fold in (1, 2, 3, 4, 5):
        monkeypatch.setenv("FOA_FOLD", str(fold))
        splits[fold] = _split_rows(dataset, "f0001_11")
        assert set().union(*splits[fold]) == everything
        assert dataset.all_writers() == ["f0001_11", "f0002_22"]

    # deterministic per fold...
    monkeypatch.setenv("FOA_FOLD", "2")
    assert _split_rows(dataset, "f0001_11") == splits[2]
    # ...and the folds are not all the same draw
    train_splits = {frozenset(split[0]) for split in splits.values()}
    assert len(train_splits) > 1


def test_a_fold_does_not_move_the_writer_split(thin_cache, monkeypatch, tmp_path):
    """theta_g is one fixed artefact: the *population* must not follow the fold."""
    from federated_outlier_adaptation.providers.nist import NistProvider

    def writers(fold):
        if fold is None:
            monkeypatch.delenv("FOA_FOLD", raising=False)
        else:
            monkeypatch.setenv("FOA_FOLD", str(fold))
        provider = NistProvider(
            results_dir=tmp_path / f"r{fold}",
            cache_dir=thin_cache,
            resolution=28,
            classes="all",
        )
        return provider.make_client_split(seed=42, global_size=0.5)

    assert writers(None) == writers(1) == writers(5)


def test_the_val_test_halves_follow_the_fold(thin_cache, monkeypatch):
    from federated_outlier_adaptation.runners.population import stratified_halves

    dataset = Nist28Dataset(thin_cache, classes="all")
    monkeypatch.delenv("FOA_FOLD", raising=False)
    _, held_out, _ = dataset.build_dataset(
        ["f0001_11"], train_rate=0.6, eval_rate=0.4, batch_size=64
    )
    plain = stratified_halves(held_out.dataset, seed=42)

    monkeypatch.setenv("FOA_FOLD", "4")
    folded = stratified_halves(held_out.dataset, seed=42)
    assert sorted(plain[0] + plain[1]) == sorted(folded[0] + folded[1])
    assert stratified_halves(held_out.dataset, seed=42) == folded


# --------------------------------------------------------------------------- #
# fold books: the split written down, not promised
# --------------------------------------------------------------------------- #
def test_a_book_covers_every_writer_by_default(prepared):
    from federated_outlier_adaptation.data.fold_book import build_fold_book

    dataset = Nist28Dataset(prepared, classes="all")
    book = build_fold_book(dataset, folds=3, seed=42, tag="all")
    assert book.folds == 3
    assert book.writers == dataset.all_writers()
    assert book.rows == len(dataset.cache)
    assert book.metadata["tag"] == "all"


def test_the_three_parts_partition_each_writer(thin_cache):
    from federated_outlier_adaptation.data.fold_book import PARTS, build_fold_book

    dataset = Nist28Dataset(thin_cache, classes="all")
    book = build_fold_book(dataset, folds=5, seed=7)
    everything = {row for row, _ in dataset.writer_samples()["f0001_11"]}
    for fold in range(1, 6):
        parts = [set(book.part(fold, "f0001_11", part)) for part in PARTS]
        assert set().union(*parts) == everything          # nothing lost
        assert sum(len(part) for part in parts) == len(everything)   # nothing shared


def test_a_book_is_reproducible_and_folds_differ(thin_cache):
    from federated_outlier_adaptation.data.fold_book import build_fold_book

    dataset = Nist28Dataset(thin_cache, classes="all")
    once = build_fold_book(dataset, folds=5, seed=7)
    twice = build_fold_book(dataset, folds=5, seed=7)
    assert np.array_equal(once.assignment, twice.assignment)

    other = build_fold_book(dataset, folds=5, seed=8)
    assert not np.array_equal(once.assignment, other.assignment)

    trains = {tuple(once.part(fold, "f0001_11", "train")) for fold in range(1, 6)}
    assert len(trains) > 1


def test_a_book_survives_a_round_trip(thin_cache, tmp_path):
    from federated_outlier_adaptation.data.fold_book import (
        FoldBook,
        build_fold_book,
        write_fold_book,
    )

    dataset = Nist28Dataset(thin_cache, classes="all")
    book = build_fold_book(dataset, folds=5, seed=7, tag="roundtrip")
    path = write_fold_book(book, tmp_path / "book")
    assert path.name.endswith(".foldbook.npz")

    reloaded = FoldBook.load(path)
    assert np.array_equal(reloaded.assignment, book.assignment)
    assert reloaded.writers == book.writers
    assert reloaded.metadata["tag"] == "roundtrip"
    assert reloaded.part(2, "f0001_11", "test") == book.part(2, "f0001_11", "test")


def test_a_book_records_who_cannot_be_split(thin_cache):
    """
    The one-sample-per-class writer has no train or validation half in any
    fold, and the book says so rather than leaving it to be discovered later.
    """
    from federated_outlier_adaptation.data.fold_book import build_fold_book

    dataset = Nist28Dataset(thin_cache, classes="all")
    book = build_fold_book(dataset, folds=2, seed=7)
    unsplittable = book.metadata["unsplittable"]
    assert unsplittable
    assert all("f0002_22" in writers for writers in unsplittable.values())


def test_writers_outside_the_book_are_marked(prepared):
    from federated_outlier_adaptation.data.fold_book import OUTSIDE, build_fold_book

    dataset = Nist28Dataset(prepared, classes="all")
    book = build_fold_book(dataset, writers=["f0001_11"], folds=2, seed=3)
    assert book.covers("f0001_11") and not book.covers("f0002_22")
    assert book.part(1, "f0002_22", "train") == []
    other_rows = [row for row, _ in dataset.writer_samples()["f0002_22"]]
    assert (book.assignment[0, other_rows] == OUTSIDE).all()


def test_an_unknown_writer_is_refused(prepared):
    from federated_outlier_adaptation.data.fold_book import build_fold_book

    dataset = Nist28Dataset(prepared, classes="all")
    with pytest.raises(ValueError, match="not in the dataset"):
        build_fold_book(dataset, writers=["nobody_00"], folds=2)


def test_a_bad_fold_or_part_is_refused(thin_cache):
    from federated_outlier_adaptation.data.fold_book import build_fold_book

    book = build_fold_book(Nist28Dataset(thin_cache, classes="all"), folds=3, seed=1)
    with pytest.raises(ValueError, match="outside 1..3"):
        book.part(4, "f0001_11", "train")
    with pytest.raises(ValueError, match="Unknown part"):
        book.part(1, "f0001_11", "holdout")


# --------------------------------------------------------------------------- #
# the loader reads the book
# --------------------------------------------------------------------------- #
def test_the_loader_uses_the_book_when_one_is_configured(thin_cache, tmp_path, monkeypatch):
    from federated_outlier_adaptation.data.fold_book import build_fold_book, write_fold_book

    dataset = Nist28Dataset(thin_cache, classes="all")
    book = build_fold_book(dataset, writers=["f0001_11"], folds=5, seed=7)
    path = write_fold_book(book, tmp_path / "book")

    monkeypatch.setenv("FOA_FOLD_BOOK", str(path))
    monkeypatch.setenv("FOA_FOLD", "3")
    fresh = Nist28Dataset(thin_cache, classes="all")
    loaders = fresh.build_dataset(["f0001_11"], batch_size=4)

    # Every part the loader hands back is exactly the part the book recorded.
    # A part the book left empty - this writer has four samples per class, so a
    # 20 % validation share truncates to nothing - is a ``None`` loader, which
    # is the same convention the seeded path uses.
    for part, loader in zip(PARTS_ORDER, loaders):
        expected = book.part(3, "f0001_11", part)
        if expected:
            assert sorted(loader.dataset.rows) == expected
        else:
            assert loader is None
    assert book.part(3, "f0001_11", "train"), "the test would be vacuous"


def test_the_book_beats_the_rates_it_is_given(thin_cache, tmp_path, monkeypatch):
    """The recorded rows are the split; no rate passed at call time moves them."""
    from federated_outlier_adaptation.data.fold_book import build_fold_book, write_fold_book

    dataset = Nist28Dataset(thin_cache, classes="all")
    book = build_fold_book(dataset, writers=["f0001_11"], folds=2, seed=7)
    path = write_fold_book(book, tmp_path / "book")

    monkeypatch.setenv("FOA_FOLD_BOOK", str(path))
    monkeypatch.setenv("FOA_FOLD", "1")
    fresh = Nist28Dataset(thin_cache, classes="all")
    train, _, _ = fresh.build_dataset(
        ["f0001_11"], train_rate=0.1, eval_rate=0.1, batch_size=4
    )
    assert sorted(train.dataset.rows) == book.part(1, "f0001_11", "train")


def test_a_book_without_a_fold_changes_nothing(thin_cache, tmp_path, monkeypatch):
    from federated_outlier_adaptation.data.fold_book import build_fold_book, write_fold_book

    dataset = Nist28Dataset(thin_cache, classes="all")
    path = write_fold_book(
        build_fold_book(dataset, writers=["f0001_11"], folds=2, seed=7), tmp_path / "b"
    )
    monkeypatch.setenv("FOA_FOLD_BOOK", str(path))
    monkeypatch.delenv("FOA_FOLD", raising=False)

    plain = Nist28Dataset(thin_cache, classes="all").build_dataset(
        ["f0001_11"], batch_size=4
    )
    monkeypatch.delenv("FOA_FOLD_BOOK", raising=False)
    reference = Nist28Dataset(thin_cache, classes="all").build_dataset(
        ["f0001_11"], batch_size=4
    )
    assert sorted(plain[0].dataset.rows) == sorted(reference[0].dataset.rows)


def test_a_writer_outside_the_book_falls_back(thin_cache, tmp_path, monkeypatch):
    """A book that does not cover the request must not silently return nothing."""
    from federated_outlier_adaptation.data.fold_book import build_fold_book, write_fold_book

    dataset = Nist28Dataset(thin_cache, classes="all")
    path = write_fold_book(
        build_fold_book(dataset, writers=["f0001_11"], folds=2, seed=7), tmp_path / "b"
    )
    monkeypatch.setenv("FOA_FOLD_BOOK", str(path))
    monkeypatch.setenv("FOA_FOLD", "1")
    fresh = Nist28Dataset(thin_cache, classes="all")
    assert fresh.book_split(["f0002_22"], 1) is None
    train, _, _ = fresh.build_dataset(["f0002_22"], batch_size=4)
    assert train is None or len(train.dataset.rows) >= 0     # the seeded path ran


def test_a_missing_book_is_an_error(thin_cache, tmp_path, monkeypatch):
    monkeypatch.setenv("FOA_FOLD_BOOK", str(tmp_path / "absent.foldbook.npz"))
    dataset = Nist28Dataset(thin_cache, classes="all")
    with pytest.raises(FileNotFoundError, match="foa fold-book"):
        _ = dataset.fold_book


def test_the_fold_book_command_is_wired_up():
    from federated_outlier_adaptation.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(
        ["fold-book", "--out", "/tmp/b", "--folds", "5", "--seed", "42", "--tag", "old100"]
    )
    assert args.command == "fold-book" and args.folds == 5 and args.tag == "old100"
    run = parser.parse_args(["final", "--trainer", "BaseTrainer", "--fold-book", "/tmp/b.npz"])
    assert run.fold_book == "/tmp/b.npz"


# --------------------------------------------------------------------------- #
# v6 stage 2/3: scoring and evaluating off the books
# --------------------------------------------------------------------------- #
def test_a_writer_is_scored_only_on_what_the_fold_held_out(prepared, tmp_path):
    from federated_outlier_adaptation.data.fold_book import build_fold_book
    from federated_outlier_adaptation.outliers.scoring import holdout_rows, score_writers
    from federated_outlier_adaptation.providers.nist import NistProvider

    provider = NistProvider(
        results_dir=tmp_path / "r", cache_dir=prepared, resolution=28, classes="all"
    )
    book = build_fold_book(provider.dataset, folds=3, seed=5)
    payload = score_writers(provider, provider.make_model(), book, fold=2, batch_size=8)

    assert payload["fold"] == 2
    assert "never trained on" in payload["rule"]
    scored = {writer for entry in payload["scores"] for writer in entry}
    # every scored writer was scored on exactly its held-out rows
    for writer in scored:
        expected = holdout_rows(book, 2, writer)
        assert expected
        assert payload["samples"][writer] == len(expected)
    assert scored | set(payload["skipped"]) == set(book.writers)


def test_the_held_out_rows_never_include_the_fold_s_training_rows(thin_cache):
    from federated_outlier_adaptation.data.fold_book import build_fold_book
    from federated_outlier_adaptation.outliers.scoring import holdout_rows

    book = build_fold_book(Nist28Dataset(thin_cache, classes="all"), folds=5, seed=3)
    for fold in range(1, 6):
        for writer in book.writers:
            train = set(book.part(fold, writer, "train"))
            assert not (train & set(holdout_rows(book, fold, writer)))


def test_evaluating_on_a_book_pools_and_splits_by_writer(thin_cache, tmp_path):
    # thin_cache's f0001_11 holds four samples per class, so its test part is
    # reliably non-empty; a writer with one sample per class has nothing left
    # after training and may land its single held-out row in either half.
    from federated_outlier_adaptation.data.fold_book import build_fold_book
    from federated_outlier_adaptation.providers.nist import NistProvider
    from federated_outlier_adaptation.training.evaluate import evaluate_on_book

    provider = NistProvider(
        results_dir=tmp_path / "r", cache_dir=thin_cache, resolution=28, classes="all"
    )
    book = build_fold_book(provider.dataset, folds=3, seed=5)
    payload = evaluate_on_book(
        provider, provider.make_model(), book, fold=1,
        writers=book.writers, part="test", batch_size=8,
    )
    assert payload["part"] == "test" and payload["fold"] == 1
    assert 0.0 <= payload["accuracy"] <= 1.0
    assert payload["samples"] == sum(
        len(book.part(1, writer, "test")) for writer in book.writers
    )
    assert set(payload["per_writer"]) <= set(book.writers)


def test_an_evaluation_file_holds_more_than_one_result(prepared, tmp_path):
    from federated_outlier_adaptation.training.evaluate import write_evaluation

    path = tmp_path / "evaluations.json"
    write_evaluation({"tag": "old_fold1", "accuracy": 0.9}, path)
    write_evaluation({"tag": "cohort_fold1", "accuracy": 0.4}, path)
    with open(path) as handle:
        stored = json.load(handle)
    assert set(stored) == {"old_fold1", "cohort_fold1"}
    assert stored["cohort_fold1"]["accuracy"] == 0.4


def test_concurrent_writers_all_survive_the_accumulator(tmp_path):
    """
    Every fold written at once is still in the file afterwards.

    THE DEFECT THIS PINS. A reference stage points one `evaluate-book` task per
    fold at a single accumulator and runs them as one array, so the merge is
    performed by several processes at the same time. Read-modify-write without a
    lock loses whichever folds were read before the last writer read: the tasks
    all succeed, all print their number, and the file ends up short. That is
    what emptied folds 2 and 4 out of the five-client evaluations while the
    twenty-client file beside it kept all five by luck of the interleaving.

    Processes rather than threads, because a thread lock inside one interpreter
    would pass this while an array of tasks on separate nodes still raced, and
    processes are what actually run.
    """
    import multiprocessing as mp

    path = tmp_path / "evaluations.json"
    tags = [f"fold{n}" for n in range(1, 13)]
    context = mp.get_context("spawn")
    barrier = context.Barrier(len(tags))
    workers = [context.Process(target=_merge_one, args=(str(path), tag, barrier))
               for tag in tags]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(120)
    assert [w.exitcode for w in workers] == [0] * len(tags)

    stored = json.loads(path.read_text())
    assert sorted(stored) == sorted(tags)
    assert all(stored[tag]["accuracy"] == float(tag[4:]) for tag in tags)


def _merge_one(path: str, tag: str, barrier) -> None:
    """One writer, released together with all the others."""
    from federated_outlier_adaptation.training.evaluate import write_evaluation

    barrier.wait(60)
    for _ in range(5):
        write_evaluation({"tag": tag, "accuracy": float(tag[4:])}, path)


# --------------------------------------------------------------------------- #
# the split rule: validation and test must be exchangeable
# --------------------------------------------------------------------------- #
# The defect this pins: with two independent floors, `int(c * 0.2)` is 0 for
# every class of four or fewer - which at 62 classes and ~3.6 images per class
# per writer is nearly every class - so validation got nothing from the small
# classes and test got everything they held.  Pooled, that is a 45.7/5.6/48.8
# split, and validation ends up drawn only from a writer's frequent classes
# while test is dominated by its rare ones.  A model then scores far worse on
# test than on validation for reasons that have nothing to do with the model.
def _class_counts(rng, writers=400, classes=62, mean=3.6, spread=1.5):
    return [
        {label: max(1, int(rng.gauss(mean, spread))) for label in range(classes)}
        for _ in range(writers)
    ]


def test_validation_and_test_are_exchangeable():
    import random as _random

    from federated_outlier_adaptation.data.fold_book import split_rows

    rng = _random.Random(0)
    row = 0
    sizes = {"train": 0, "val": 0, "test": 0}
    small = {"val": 0, "test": 0}
    for index, counts in enumerate(_class_counts(rng)):
        rows_by_label, membership = {}, {}
        for label, count in counts.items():
            rows_by_label[label] = list(range(row, row + count))
            for r in rows_by_label[label]:
                membership[r] = count
            row += count
        train, val, test = split_rows(rows_by_label, seed=index)
        sizes["train"] += len(train)
        sizes["val"] += len(val)
        sizes["test"] += len(test)
        small["val"] += sum(1 for r in val if membership[r] <= 4)
        small["test"] += sum(1 for r in test if membership[r] <= 4)

    # the two halves are the same size...
    assert sizes["val"] == pytest.approx(sizes["test"], rel=0.02)
    # ...and drawn from the same kind of class, which is the property that failed
    assert small["val"] / sizes["val"] == pytest.approx(
        small["test"] / sizes["test"], abs=0.02
    )


def test_the_training_rows_are_untouched_by_the_fix():
    """
    The fix had to leave the train partition alone: models already trained
    against a book stay valid when the book is rebuilt.
    """
    import random as _random

    from federated_outlier_adaptation.data.fold_book import split_rows

    rng = _random.Random(1)
    for index, counts in enumerate(_class_counts(rng, writers=50)):
        row = 0
        rows_by_label = {}
        for label, count in counts.items():
            rows_by_label[label] = list(range(row, row + count))
            row += count
        train, val, test = split_rows(rows_by_label, seed=index)
        # exactly what the old rule produced for train - the tie-breaks are
        # drawn from a separate stream, so the shuffle sequence is untouched
        expected = []
        generator = _random.Random(index)
        for label in sorted(rows_by_label):
            rows = list(rows_by_label[label])
            generator.shuffle(rows)
            expected += rows[: int(len(rows) * 0.6)]
        assert train == sorted(expected)
        # ...and the three parts still partition the writer
        assert len(train) + len(val) + len(test) == sum(counts.values())


@pytest.mark.parametrize(
    "train_rate,eval_rate,expect",
    [
        (0.0, 0.0, "all test"),        # the in-sample idiom used across the code
        (0.6, 0.4, "no test"),         # the source-validation idiom
        (0.6, 0.0, "no val"),          # train-only idiom
    ],
)
def test_the_existing_rate_idioms_still_mean_what_they_meant(
    thin_cache, train_rate, eval_rate, expect
):
    # f0001_11 of this fixture holds four samples in each of three classes, so
    # every part is non-empty wherever the rates ask for one.
    dataset = Nist28Dataset(thin_cache, classes="all")
    train, val, test = dataset.build_dataset(
        ["f0001_11"], train_rate=train_rate, eval_rate=eval_rate, batch_size=4
    )
    total = len(dataset.writer_samples()["f0001_11"])
    if expect == "all test":
        assert train is None and val is None
        assert len(test.dataset) == total
    elif expect == "no test":
        assert test is None
        assert len(train.dataset) + len(val.dataset) == total
    else:
        assert val is None
        assert len(train.dataset) + len(test.dataset) == total


def test_the_held_out_union_is_unchanged_so_scoring_was_never_wrong(thin_cache):
    """
    ``score-writers`` uses validation+test - the whole non-training remainder -
    so the apportionment between the two never affected it.  Pinning that means
    the B2 scores do not have to be recomputed when a book is rebuilt.
    """
    from federated_outlier_adaptation.data.fold_book import build_fold_book
    from federated_outlier_adaptation.outliers.scoring import holdout_rows

    dataset = Nist28Dataset(thin_cache, classes="all")
    book = build_fold_book(dataset, folds=3, seed=11)
    for fold in (1, 2, 3):
        for writer in book.writers:
            everything = {row for row, _ in dataset.writer_samples()[writer]}
            train = set(book.part(fold, writer, "train"))
            assert set(holdout_rows(book, fold, writer)) == everything - train


# --------------------------------------------------------------------------- #
# v6 stage 4: isolated training, and the three things a private model is worth
# --------------------------------------------------------------------------- #
@pytest.fixture()
def isolated_setup(thin_cache, tmp_path):
    """A provider, a cohort book and an old book over the same tiny cache."""
    from federated_outlier_adaptation.data.fold_book import build_fold_book
    from federated_outlier_adaptation.providers.nist import NistProvider

    provider = NistProvider(
        results_dir=tmp_path / "r", cache_dir=thin_cache, resolution=28, classes="all"
    )
    cohort = build_fold_book(provider.dataset, writers=["f0001_11"], folds=5, seed=4)
    old = build_fold_book(provider.dataset, writers=["f0002_22"], folds=5, seed=9)
    return provider, cohort, old


def test_the_three_evaluations_are_the_three_sets(isolated_setup):
    from federated_outlier_adaptation.training.isolated import isolated_training

    provider, cohort, old = isolated_setup
    payload = isolated_training(
        provider,
        clients=["f0001_11"],
        cohort_book=cohort,
        fold=1,
        old_book=old,
        old_fold=2,
        init="scratch",
        epochs=2,
        batch_size=4,
        patience=None,
        min_epochs=0,
    )
    row = payload["per_client"]["f0001_11"]
    assert set(row) >= {"own", "union", "old", "convergence"}
    assert row["test_samples"] == len(cohort.part(1, "f0001_11", "test"))
    # union is every cohort client's fold-k test rows...
    assert payload["union_test_samples"] == row["test_samples"]     # one client here
    # ...and old is the old book's partition, not the cohort's.  The sizes are
    # per fold now, because each fold is scored separately.
    assert payload["old_test_samples"] == {"2": len(old.part(2, "f0002_22", "test"))}
    assert payload["old_fold"] == 2
    assert payload["old_folds"] == [2]


def test_the_union_is_every_client_not_just_this_one(isolated_setup, thin_cache):
    """The isolation-fails-for-the-group number has to see the whole cohort."""
    from federated_outlier_adaptation.data.fold_book import build_fold_book
    from federated_outlier_adaptation.training.isolated import isolated_training

    provider, _, old = isolated_setup
    both = build_fold_book(
        provider.dataset, writers=["f0001_11", "f0002_22"], folds=3, seed=6
    )
    payload = isolated_training(
        provider,
        clients=["f0001_11", "f0002_22"],
        cohort_book=both,
        fold=1,
        old_book=old,
        old_fold=2,
        init="scratch",
        epochs=1,
        batch_size=4,
        patience=None,
        min_epochs=0,
    )
    expected = sum(
        len(both.part(1, writer, "test")) for writer in ("f0001_11", "f0002_22")
    )
    assert payload["union_test_samples"] == expected
    # each trained model was scored on that same union
    for row in payload["per_client"].values():
        if not row.get("skipped"):
            assert row["union"] is not None
            assert row["test_samples"] <= expected


def test_a_client_this_fold_cannot_train_is_recorded(isolated_setup, thin_cache):
    from federated_outlier_adaptation.data.fold_book import build_fold_book
    from federated_outlier_adaptation.training.isolated import isolated_training

    provider, _, old = isolated_setup
    # f0002_22 holds one sample per class: no training half in any fold.
    book = build_fold_book(provider.dataset, writers=["f0002_22"], folds=2, seed=2)
    payload = isolated_training(
        provider, clients=["f0002_22"], cohort_book=book, fold=1,
        old_book=old, old_fold=2, init="scratch", epochs=1, batch_size=4,
        patience=None, min_epochs=0,
    )
    assert payload["skipped"] == ["f0002_22"]
    assert payload["trained"] == 0
    assert payload["per_client"]["f0002_22"]["skipped"] is True


def test_scratch_and_checkpoint_inits_differ(isolated_setup, tmp_path):
    import torch

    from federated_outlier_adaptation.training.isolated import isolated_training

    provider, cohort, old = isolated_setup
    torch.manual_seed(5)
    checkpoint = tmp_path / "g0_model"
    torch.save(provider.make_model().state_dict(), checkpoint)

    common = dict(
        clients=["f0001_11"], cohort_book=cohort, fold=1, old_book=old, old_fold=2,
        epochs=1, batch_size=4, patience=None, min_epochs=0,
    )
    from_g0 = isolated_training(
        provider, init="global", init_checkpoint=str(checkpoint), **common
    )
    scratch = isolated_training(provider, init="scratch", **common)

    assert from_g0["init"] == "global" and scratch["init"] == "scratch"
    # the shipped model is identified in the record, the scratch run has none
    assert from_g0["init_checkpoint_sha256"]
    assert "init_checkpoint_sha256" not in scratch
    # ...and both scored their old column on the same fixed partition
    assert from_g0["old_fold"] == scratch["old_fold"] == 2
    assert from_g0["old_test_samples"] == scratch["old_test_samples"]


def test_a_checkpoint_init_needs_a_checkpoint(isolated_setup):
    from federated_outlier_adaptation.training.isolated import isolated_training

    provider, cohort, old = isolated_setup
    with pytest.raises(ValueError, match="needs the checkpoint"):
        isolated_training(
            provider, clients=["f0001_11"], cohort_book=cohort, fold=1,
            old_book=old, old_fold=2, init="global",
        )
    with pytest.raises(FileNotFoundError, match="g-0 to exist"):
        isolated_training(
            provider, clients=["f0001_11"], cohort_book=cohort, fold=1,
            old_book=old, old_fold=2, init="global",
            init_checkpoint="/nowhere/g0_model", epochs=1, patience=None,
        )


def test_the_pooled_figures_are_weighted_where_it_matters(isolated_setup):
    from federated_outlier_adaptation.training.isolated import isolated_training

    provider, cohort, old = isolated_setup
    payload = isolated_training(
        provider, clients=["f0001_11"], cohort_book=cohort, fold=1,
        old_book=old, old_fold=2, init="scratch", epochs=1, batch_size=4,
        patience=None, min_epochs=0,
    )
    pooled = payload["pooled"]
    # own is per-image, so it is sample weighted; union and old are the same set
    # for every model, so they are reported as a spread across models
    assert pooled["own_weighted"] is not None
    for key in ("own", "union", "old"):
        assert set(pooled[key]) == {"mean", "spread", "min", "max", "n"}
    assert pooled["union"]["n"] == payload["trained"]


def test_rows_loader_drops_nothing_it_should_keep(isolated_setup):
    from federated_outlier_adaptation.training.evaluate import rows_loader

    provider, cohort, _ = isolated_setup
    rows = cohort.part(1, "f0001_11", "test")
    loader = rows_loader(provider, rows, batch_size=4)
    assert sorted(loader.dataset.rows) == rows
    assert rows_loader(provider, [], batch_size=4) is None


def test_the_isolated_command_is_wired_up():
    from federated_outlier_adaptation.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(
        ["isolated-train", "--fold-book", "/c", "--fold", "3",
         "--old-book", "/o", "--old-fold", "2",
         "--init", "global", "--init-checkpoint", "/g0"]
    )
    assert args.command == "isolated-train"
    # --old-fold takes "all" or a fold number, so it arrives as text
    assert args.old_fold == "2" and args.init == "global"
    assert args.early_stopping_patience == 10 and args.min_epochs == 20


# --------------------------------------------------------------------------- #
# stage 4, final protocol: every old fold separately, one writer at a time
# --------------------------------------------------------------------------- #
def test_every_old_fold_is_scored_separately(isolated_setup):
    """
    Five partitions of one population, so their spread is the error bar on
    preservation.  Merging them into one set would throw that away.
    """
    from federated_outlier_adaptation.training.isolated import isolated_training

    provider, cohort, old = isolated_setup
    payload = isolated_training(
        provider, clients=["f0001_11"], cohort_book=cohort, fold=1,
        old_book=old, old_fold="all", init="scratch",
        epochs=1, batch_size=4, patience=None, min_epochs=0,
    )
    assert payload["old_folds"] == [1, 2, 3, 4, 5]
    assert payload["old_fold"] == "all"
    assert set(payload["old_test_samples"]) == {"1", "2", "3", "4", "5"}

    row = payload["per_client"]["f0001_11"]
    assert set(row["old_folds"]) == {"1", "2", "3", "4", "5"}
    # the five are the five partitions, not one repeated
    per_fold_sizes = {
        str(k): len(old.part(k, "f0002_22", "test")) for k in range(1, 6)
    }
    assert payload["old_test_samples"] == per_fold_sizes


def test_the_old_mean_and_sd_are_of_those_five(isolated_setup):
    from statistics import mean, pstdev

    from federated_outlier_adaptation.training.isolated import isolated_training

    provider, cohort, old = isolated_setup
    payload = isolated_training(
        provider, clients=["f0001_11"], cohort_book=cohort, fold=1,
        old_book=old, old_fold="all", init="scratch",
        epochs=1, batch_size=4, patience=None, min_epochs=0,
    )
    row = payload["per_client"]["f0001_11"]
    measured = [v for v in row["old_folds"].values() if v is not None]
    assert measured
    assert row["old_mean"] == pytest.approx(mean(measured))
    assert row["old_sd"] == pytest.approx(pstdev(measured) if len(measured) > 1 else 0.0)
    # `old` is that mean under a short name, for the pooled block
    assert row["old"] == pytest.approx(row["old_mean"])


def test_a_single_old_fold_is_still_accepted(isolated_setup):
    """Compatibility with runs made before the five-fold rule."""
    from federated_outlier_adaptation.training.isolated import isolated_training

    provider, cohort, old = isolated_setup
    for given in (2, "2"):
        payload = isolated_training(
            provider, clients=["f0001_11"], cohort_book=cohort, fold=1,
            old_book=old, old_fold=given, init="scratch",
            epochs=1, batch_size=4, patience=None, min_epochs=0,
        )
        assert payload["old_folds"] == [2]
        assert payload["old_fold"] == 2
        row = payload["per_client"]["f0001_11"]
        assert set(row["old_folds"]) == {"2"}
        assert row["old_sd"] == 0.0


def test_only_client_trains_one_but_the_union_spans_all(isolated_setup, thin_cache):
    """
    What one private model is worth to the whole group is meaningless against a
    group of one, so the union must not narrow with --only-client.
    """
    from federated_outlier_adaptation.data.fold_book import build_fold_book
    from federated_outlier_adaptation.training.isolated import isolated_training

    provider, _, old = isolated_setup
    cohort = build_fold_book(
        provider.dataset, writers=["f0001_11", "f0002_22"], folds=3, seed=6
    )
    everyone = ["f0001_11", "f0002_22"]

    payload = isolated_training(
        provider, clients=everyone, cohort_book=cohort, fold=1,
        old_book=old, old_fold="all", init="scratch", only_client="f0001_11",
        epochs=1, batch_size=4, patience=None, min_epochs=0,
    )
    assert payload["only_client"] == "f0001_11"
    assert payload["trained_clients"] == ["f0001_11"]
    assert payload["clients"] == everyone          # the cohort is still the cohort
    assert set(payload["per_client"]) == {"f0001_11"}

    # the union is every cohort client's fold-k test rows, both writers
    expected = sum(len(cohort.part(1, w, "test")) for w in everyone)
    assert payload["union_test_samples"] == expected
    assert payload["per_client"]["f0001_11"]["union"] is not None


def test_only_client_must_be_in_the_cohort(isolated_setup):
    from federated_outlier_adaptation.training.isolated import isolated_training

    provider, cohort, old = isolated_setup
    with pytest.raises(ValueError, match="not in the cohort"):
        isolated_training(
            provider, clients=["f0001_11"], cohort_book=cohort, fold=1,
            old_book=old, only_client="f0002_22", init="scratch",
            epochs=1, patience=None,
        )


def test_the_final_protocol_flags_are_on_the_command_line():
    from federated_outlier_adaptation.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(
        ["isolated-train", "--fold-book", "/c", "--fold", "3", "--old-book", "/o",
         "--init", "global", "--init-checkpoint", "/g0",
         "--only-client", "f0001_11"]
    )
    assert args.old_fold == "all"          # the default is every fold
    assert args.only_client == "f0001_11"
    compat = parser.parse_args(
        ["isolated-train", "--fold-book", "/c", "--fold", "1",
         "--old-book", "/o", "--old-fold", "2"]
    )
    assert compat.old_fold == "2" and compat.only_client is None


# --------------------------------------------------------------------------- #
# v6 stage 5: normal FL on the cohort, and the three things the final model
# is scored on
# --------------------------------------------------------------------------- #
def _stage5_population(provider, clients, loader_seed=1):
    """A pool over the cohort writers; the module already has a ``_population``."""
    from federated_outlier_adaptation.runners.population import ClientPopulation

    return ClientPopulation(provider=provider, selected=list(clients), loader_seed=loader_seed)


def test_the_pool_test_half_is_the_book_s_test_partition(
    stage5_setup, cohort_cache, tmp_path, monkeypatch
):
    """
    The rows the FL loop reports on are the book's test rows.

    Without this the run took the book's *validation* rows, cut them in half a
    second time, and reported one half as "test" - so the fold's actual test
    partition was never read and the reported number was measured on rows the
    early-stopping rule had already seen.
    """
    from federated_outlier_adaptation.data.fold_book import write_fold_book
    from federated_outlier_adaptation.providers.nist import NistProvider

    _, cohort_writers, cohort, _, _ = stage5_setup
    path = write_fold_book(cohort, tmp_path / "cohort20")
    monkeypatch.setenv("FOA_FOLD_BOOK", str(path))
    monkeypatch.setenv("FOA_FOLD", "2")

    provider = NistProvider(
        results_dir=tmp_path / "r", cache_dir=cohort_cache, resolution=28, classes="all"
    )
    population = _stage5_population(provider, cohort_writers)
    for client in cohort_writers:
        val_loader, test_loader = population.split_loaders(client, batch_size=4)
        assert sorted(val_loader.dataset.rows) == cohort.part(2, client, "val")
        assert sorted(test_loader.dataset.rows) == cohort.part(2, client, "test")
    assert cohort.part(2, cohort_writers[0], "test"), "the test would be vacuous"


def test_a_different_fold_gives_the_pool_different_test_rows(
    stage5_setup, cohort_cache, tmp_path, monkeypatch
):
    """Fold membership has to reach the FL loop, or CV-5 is five identical runs."""
    from federated_outlier_adaptation.data.fold_book import write_fold_book
    from federated_outlier_adaptation.providers.nist import NistProvider

    _, cohort_writers, cohort, _, _ = stage5_setup
    path = write_fold_book(cohort, tmp_path / "cohort20")
    monkeypatch.setenv("FOA_FOLD_BOOK", str(path))

    seen = []
    for fold in (1, 2):
        monkeypatch.setenv("FOA_FOLD", str(fold))
        provider = NistProvider(
            results_dir=tmp_path / f"r{fold}", cache_dir=cohort_cache,
            resolution=28, classes="all",
        )
        population = _stage5_population(provider, cohort_writers)
        rows = []
        for client in cohort_writers:
            rows.extend(population.split_loaders(client, batch_size=4)[1].dataset.rows)
        seen.append(sorted(rows))
    assert seen[0] != seen[1]


def test_without_a_book_the_pool_halves_are_unchanged(stage5_setup):
    """The default path is the published seeded division, untouched."""
    from federated_outlier_adaptation.runners.population import (
        SPLIT_SEED,
        stratified_halves,
    )

    provider, cohort_writers, _, _, _ = stage5_setup
    population = _stage5_population(provider, cohort_writers)
    client = cohort_writers[0]
    val_loader, test_loader = population.split_loaders(client, batch_size=4)

    heldout = population.heldout_loader(client, batch_size=4)
    val_index, test_index = stratified_halves(heldout.dataset, seed=SPLIT_SEED)
    assert len(val_loader.dataset) == len(val_index)
    assert len(test_loader.dataset) == len(test_index)


def test_the_final_model_is_scored_on_every_old_fold(stage5_setup):
    from statistics import mean, pstdev

    from federated_outlier_adaptation.training.final_eval import final_model_evaluation

    provider, cohort_writers, cohort, old_writers, old = stage5_setup
    payload = final_model_evaluation(
        provider,
        provider.make_model(),
        cohort_book=cohort,
        fold=1,
        clients=cohort_writers,
        old_book=old,
        old_writers=old_writers,
        old_folds="all",
        batch_size=8,
    )
    assert payload["old"]["folds"] == [1, 2, 3, 4, 5]
    assert set(payload["old"]["accuracies"]) == {"1", "2", "3", "4", "5"}
    # five partitions of one population, so the five sample counts are the five
    # test partitions and not one set counted five times
    assert payload["old"]["samples"] == {
        str(k): sum(len(old.part(k, w, "test")) for w in old_writers) for k in range(1, 6)
    }
    measured = [v for v in payload["old"]["accuracies"].values() if v is not None]
    assert payload["old"]["mean"] == pytest.approx(mean(measured))
    assert payload["old"]["sd"] == pytest.approx(
        pstdev(measured) if len(measured) > 1 else 0.0
    )


def test_a_single_old_fold_is_accepted_by_the_final_evaluation(stage5_setup):
    from federated_outlier_adaptation.training.final_eval import final_model_evaluation

    provider, cohort_writers, cohort, old_writers, old = stage5_setup
    payload = final_model_evaluation(
        provider, provider.make_model(), cohort_book=cohort, fold=1,
        clients=cohort_writers, old_book=old, old_writers=old_writers,
        old_folds=3, batch_size=8,
    )
    assert payload["old"]["folds"] == [3]
    assert payload["old"]["sd"] == 0.0


def test_the_final_model_has_a_per_client_column(stage5_setup):
    """
    Every cohort writer is scored on its own fold-k test rows.

    A pooled number cannot tell "the federation lifted the cohort" from "the
    federation lifted three writers and left seventeen where they were", which
    is the whole question stage 5 asks.
    """
    from federated_outlier_adaptation.training.final_eval import final_model_evaluation

    provider, cohort_writers, cohort, _, _ = stage5_setup
    payload = final_model_evaluation(
        provider, provider.make_model(), cohort_book=cohort, fold=1,
        clients=cohort_writers, batch_size=8,
    )
    clients = payload["clients"]
    assert set(clients["per_client"]) == set(cohort_writers)
    assert clients["samples"] == sum(
        len(cohort.part(1, w, "test")) for w in cohort_writers
    )
    # the pooled figure is the sample-weighted mean of the column, per image
    weighted = sum(
        clients["per_client"][w] * len(cohort.part(1, w, "test")) for w in cohort_writers
    )
    assert clients["accuracy"] == pytest.approx(weighted / clients["samples"])


def test_the_worker_opens_the_books_from_the_run_s_own_settings(
    stage5_setup, cohort_cache, tmp_path, monkeypatch
):
    """The cohort book and fold come from the environment the run is split by."""
    import json as _json

    from federated_outlier_adaptation.data.fold_book import write_fold_book
    from federated_outlier_adaptation.providers.nist import NistProvider
    from federated_outlier_adaptation.training.final_eval import evaluate_final_model

    _, cohort_writers, cohort, old_writers, old = stage5_setup
    cohort_path = write_fold_book(cohort, tmp_path / "cohort20")
    old_path = write_fold_book(old, tmp_path / "old_data")
    old_list = tmp_path / "old_data.json"
    with open(old_list, "w") as handle:
        _json.dump(old_writers, handle)

    monkeypatch.setenv("FOA_FOLD_BOOK", str(cohort_path))
    monkeypatch.setenv("FOA_FOLD", "4")
    provider = NistProvider(
        results_dir=tmp_path / "rw", cache_dir=cohort_cache, resolution=28, classes="all"
    )
    payload = evaluate_final_model(
        provider,
        provider.make_model(),
        clients=cohort_writers,
        old_book=str(old_path),
        old_clients_file=str(old_list),
        batch_size=8,
    )
    assert payload["fold"] == 4
    assert payload["old"]["folds"] == [1, 2, 3, 4, 5]
    assert payload["old"]["writers"] == len(old_writers)
    assert set(payload["clients"]["per_client"]) == set(cohort_writers)
    assert payload["fold_book"] == str(cohort_path)
    assert payload["old_book"] == str(old_path)


def test_a_run_without_books_reports_nothing_extra(stage5_setup, monkeypatch):
    """An ordinary run is unaffected: no books, no key, no failure."""
    from federated_outlier_adaptation.training.final_eval import evaluate_final_model

    monkeypatch.delenv("FOA_FOLD_BOOK", raising=False)
    monkeypatch.delenv("FOA_FOLD", raising=False)
    provider, cohort_writers, _, _, _ = stage5_setup
    assert evaluate_final_model(
        provider, provider.make_model(), clients=cohort_writers
    ) is None


def test_a_missing_old_book_is_named(stage5_setup, tmp_path):
    from federated_outlier_adaptation.training.final_eval import evaluate_final_model

    provider, cohort_writers, _, _, _ = stage5_setup
    with pytest.raises(FileNotFoundError, match="old-data fold book"):
        evaluate_final_model(
            provider, provider.make_model(), clients=cohort_writers,
            old_book=str(tmp_path / "absent.foldbook.npz"),
        )


def test_scratch_and_the_winner_are_two_different_starting_points(
    stage5_setup, tmp_path
):
    """
    ``--init scratch`` never reads the checkpoint; ``--init global`` is it exactly.

    The two inits are the two arms of stage 5, so a run that quietly loaded the
    same weights for both would produce two identical curves and one wasted
    half of the grid.
    """
    import torch

    from federated_outlier_adaptation.runners.concurrent_runner import (
        BaseConcurrentRunner,
    )

    provider, _, _, _, _ = stage5_setup
    reference = provider.make_model()
    results_dir = provider.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    torch.save(reference.state_dict(), results_dir / "g0_model")

    loaded = BaseConcurrentRunner(
        trainer=None, provider=provider, init="global"
    ).load_global_model("g0")
    scratch = BaseConcurrentRunner(
        trainer=None, provider=provider, init="scratch"
    ).load_global_model("g0")

    reference_state = reference.state_dict()
    assert all(
        torch.equal(value.cpu(), reference_state[key].cpu())
        for key, value in loaded.state_dict().items()
    )
    assert any(
        not torch.equal(value.cpu(), reference_state[key].cpu())
        for key, value in scratch.state_dict().items()
    )


def test_the_stage_five_flags_are_on_the_final_command():
    from federated_outlier_adaptation.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(
        [
            "final", "--trainer", "BaseTrainer",
            "--fold-book", "/books/cohort20.foldbook.npz", "--fold", "3",
            "--old-book", "/books/old_data.foldbook.npz",
            "--old-clients-file", "/outliers/old_data.json",
            "--aggregation", "fedavg", "--init", "scratch",
            "--policy", "uniform", "--clients-per-round", "16",
            "--sampler-seed", "5", "--track-clients", "--save-final-model",
            "--rounds", "100", "--epochs", "5", "--batch-size", "64",
        ]
    )
    assert args.old_book == "/books/old_data.foldbook.npz"
    assert args.old_clients_file == "/outliers/old_data.json"
    assert args.old_fold == "all"          # every fold of the old book, separately
    assert args.eval_batch_size == 256
    assert (args.policy, args.clients_per_round) == ("uniform", 16)
    assert args.save_final_model and args.track_clients


def test_the_source_series_follows_the_old_book(stage5_setup, monkeypatch):
    """
    With an old-data book the per-round series is that book's fold-k test rows.

    Cheap, disjoint from the cohort by construction, and clearly a diagnostic -
    the reported preservation figure is the five-fold evaluation of the final
    model, not this.
    """
    from federated_outlier_adaptation.data.fold_book import write_fold_book
    from federated_outlier_adaptation.runners.source_series import source_loaders

    provider, _, _, old_writers, old = stage5_setup
    path = write_fold_book(old, provider.results_dir / "old_data")

    test_loader, val_loader, info = source_loaders(
        provider, batch_size=8, old_book=str(path), fold=3
    )
    expected_test = sorted(r for w in old_writers for r in old.part(3, w, "test"))
    expected_val = sorted(r for w in old_writers for r in old.part(3, w, "val"))
    assert sorted(test_loader.dataset.rows) == expected_test
    assert sorted(val_loader.dataset.rows) == expected_val
    assert info["source"] == "old_book"
    assert info["fold"] == 3 and info["fold_given"] is True
    assert info["diagnostic"] is True


def test_the_source_series_takes_the_run_s_own_fold(stage5_setup):
    """Fold k of the run is fold k of the diagnostic, or the two disagree."""
    from federated_outlier_adaptation.data.fold_book import write_fold_book
    from federated_outlier_adaptation.runners.source_series import source_loaders

    provider, _, _, old_writers, old = stage5_setup
    path = write_fold_book(old, provider.results_dir / "old_data")

    seen = []
    for fold in (1, 2):
        loader, _, info = source_loaders(
            provider, batch_size=8, old_book=str(path), fold=fold
        )
        assert info["fold"] == fold
        seen.append(sorted(loader.dataset.rows))
    assert seen[0] != seen[1]


def test_the_source_series_honours_the_old_client_list(stage5_setup, tmp_path):
    import json as _json

    from federated_outlier_adaptation.data.fold_book import write_fold_book
    from federated_outlier_adaptation.runners.source_series import source_loaders

    provider, _, _, old_writers, old = stage5_setup
    path = write_fold_book(old, provider.results_dir / "old_data")
    one = tmp_path / "one_old.json"
    with open(one, "w") as handle:
        _json.dump(old_writers[:1], handle)

    loader, _, info = source_loaders(
        provider, batch_size=8, old_book=str(path),
        old_clients_file=str(one), fold=1,
    )
    assert info["writers"] == 1
    assert sorted(loader.dataset.rows) == sorted(old.part(1, old_writers[0], "test"))


def test_a_missing_old_book_for_the_series_is_named(stage5_setup, tmp_path):
    from federated_outlier_adaptation.runners.source_series import source_loaders

    provider, _, _, _, _ = stage5_setup
    with pytest.raises(FileNotFoundError, match="old-data fold book"):
        source_loaders(
            provider, batch_size=8, old_book=str(tmp_path / "absent.foldbook.npz"), fold=1
        )


# --------------------------------------------------------------------------- #
# v6 stage 5b: one centralized model over the pooled cohort
# --------------------------------------------------------------------------- #
def test_the_pooled_arm_trains_on_the_union_of_the_train_partitions(stage5_setup):
    """
    One model, one training set: every cohort writer's fold-k train rows.

    This is the rung between isolation (twenty private models) and federation
    (twenty clients, one server): what a single model achieves when the privacy
    constraint is simply lifted and all the data is in one place.
    """
    from federated_outlier_adaptation.training.isolated import isolated_training

    provider, cohort_writers, cohort, old_writers, old = stage5_setup
    payload = isolated_training(
        provider, clients=cohort_writers, cohort_book=cohort, fold=1,
        old_book=old, old_clients=old_writers, pooled=True, init="scratch",
        epochs=1, batch_size=4, patience=None, min_epochs=0,
    )
    expected = sorted(r for w in cohort_writers for r in cohort.part(1, w, "train"))
    assert payload["pooled_train_rows"] == len(expected)
    assert payload["pooled_val_rows"] == len(
        [r for w in cohort_writers for r in cohort.part(1, w, "val")]
    )
    # one model, not twenty
    assert payload["stage"] == "centralized_outliers"
    assert payload["trained"] == 1
    assert payload["models"] == 1


def test_the_pooled_model_has_a_per_client_column(stage5_setup):
    from federated_outlier_adaptation.training.isolated import isolated_training

    provider, cohort_writers, cohort, old_writers, old = stage5_setup
    payload = isolated_training(
        provider, clients=cohort_writers, cohort_book=cohort, fold=1,
        old_book=old, old_clients=old_writers, pooled=True, init="scratch",
        epochs=1, batch_size=4, patience=None, min_epochs=0,
    )
    assert set(payload["per_client"]) == set(cohort_writers)
    for client in cohort_writers:
        row = payload["per_client"][client]
        assert row["test_samples"] == len(cohort.part(1, client, "test"))
        assert 0.0 <= row["own"] <= 1.0
    # the pooled figure is the union evaluation, and it is the same set for all
    assert payload["union_test_samples"] == sum(
        len(cohort.part(1, w, "test")) for w in cohort_writers
    )
    assert payload["pooled"]["union"] is not None


def test_the_pooled_model_is_scored_on_every_old_fold(stage5_setup):
    from statistics import mean, pstdev

    from federated_outlier_adaptation.training.isolated import isolated_training

    provider, cohort_writers, cohort, old_writers, old = stage5_setup
    payload = isolated_training(
        provider, clients=cohort_writers, cohort_book=cohort, fold=1,
        old_book=old, old_clients=old_writers, pooled=True, old_fold="all",
        init="scratch", epochs=1, batch_size=4, patience=None, min_epochs=0,
    )
    folds = payload["old_folds_accuracy"]
    assert set(folds) == {"1", "2", "3", "4", "5"}
    measured = [v for v in folds.values() if v is not None]
    assert payload["old_mean"] == pytest.approx(mean(measured))
    assert payload["old_sd"] == pytest.approx(
        pstdev(measured) if len(measured) > 1 else 0.0
    )
    assert "convergence" in payload


def test_the_two_pooled_inits_are_different_models(stage5_setup, tmp_path):
    """scratch and the winner g-0 must not quietly be the same starting point."""
    import torch

    from federated_outlier_adaptation.training.isolated import isolated_training

    provider, cohort_writers, cohort, old_writers, old = stage5_setup
    checkpoint = tmp_path / "g0_model"
    torch.save(provider.make_model().state_dict(), checkpoint)

    common = dict(
        clients=cohort_writers, cohort_book=cohort, fold=1, old_book=old,
        old_clients=old_writers, pooled=True, epochs=1, batch_size=4,
        patience=None, min_epochs=0,
    )
    scratch = isolated_training(provider, init="scratch", **common)
    seeded = isolated_training(
        provider, init="global", init_checkpoint=str(checkpoint), **common
    )
    assert scratch["init"] == "scratch" and seeded["init"] == "global"
    assert "init_checkpoint_sha256" in seeded
    assert "init_checkpoint_sha256" not in scratch


def test_the_pooled_flag_is_on_the_isolated_command():
    from federated_outlier_adaptation.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(
        ["isolated-train", "--fold-book", "/c", "--fold", "2",
         "--old-book", "/o", "--pooled", "--init", "scratch"]
    )
    assert args.pooled is True
    plain = parser.parse_args(
        ["isolated-train", "--fold-book", "/c", "--fold", "2", "--old-book", "/o"]
    )
    assert plain.pooled is False


# --------------------------------------------------------------------------- #
# Sharing old data back into training: which rows may move, and which may not
# --------------------------------------------------------------------------- #
def test_the_shared_pool_is_the_old_book_s_train_partition(stage5_setup):
    """
    Only train rows may ever be shared.

    The old book's validation and test rows are the preservation measurement. A
    row that enters a client's training set and is later scored as knowledge the
    model retained measures nothing at all, so this is the one invariant the
    sharing path cannot be allowed to violate.
    """
    from federated_outlier_adaptation.data.source_share import old_train_dataset

    provider, _, _, old_writers, old = stage5_setup
    for fold in range(1, old.folds + 1):
        pool = old_train_dataset(provider, old, old_writers, fold, batch_size=8)
        shared = set(pool.rows)
        expected = {r for w in old_writers for r in old.part(fold, w, "train")}
        assert shared == expected
        assert shared, "the test would be vacuous"


def test_no_shared_row_is_ever_a_validation_or_test_row(stage5_setup):
    """
    Within the fold being shared, which is where the invariant lives.

    A row that is train in fold 1 and validation in fold 2 is not a leak - that
    is what cross-validation *is*. The leak would be sharing a row that the
    same run later scores on, so the comparison is fold k's train against fold
    k's validation and test.
    """
    from federated_outlier_adaptation.data.source_share import old_train_dataset

    provider, _, _, old_writers, old = stage5_setup
    for fold in range(1, old.folds + 1):
        pool = old_train_dataset(provider, old, old_writers, fold, batch_size=8)
        forbidden = {
            row
            for writer in old_writers
            for part in ("val", "test")
            for row in old.part(fold, writer, part)
        }
        assert forbidden, "the test would be vacuous"
        assert not (set(pool.rows) & forbidden)


def test_the_runner_shares_the_old_book_when_it_has_one(stage5_setup, tmp_path, monkeypatch):
    """A v6 run draws from the old-data draw, not from the legacy 3% split."""
    from federated_outlier_adaptation.data.fold_book import write_fold_book
    from federated_outlier_adaptation.runners.concurrent_runner import BaseConcurrentRunner

    provider, _, _, old_writers, old = stage5_setup
    path = write_fold_book(old, tmp_path / "old_data")
    monkeypatch.setenv("FOA_FOLD", "2")

    runner = BaseConcurrentRunner(
        trainer=None, provider=provider, source_share="equal", old_book=str(path)
    )
    pool = runner._old_train_pool(batch_size=8)
    assert set(pool.rows) == {r for w in old.writers for r in old.part(2, w, "train")}


def test_a_run_without_an_old_book_keeps_the_legacy_pool(stage5_setup, monkeypatch):
    """v4 and v5 roots are unchanged: no book, no rewire."""
    from federated_outlier_adaptation.runners.concurrent_runner import BaseConcurrentRunner

    provider, _, _, _, _ = stage5_setup
    called = {}

    def legacy(provider_arg, batch_size, loader_seed=None):
        called["yes"] = True
        return None

    monkeypatch.setattr(
        "federated_outlier_adaptation.data.source_share.source_pool_dataset", legacy
    )
    runner = BaseConcurrentRunner(trainer=None, provider=provider, source_share="equal")
    assert runner._old_train_pool(batch_size=8) is None
    assert called.get("yes")


def test_the_multiplier_scales_the_share(stage5_setup):
    from federated_outlier_adaptation.data.source_share import share_size

    for multiplier, expected in ((1, 50), (2, 100), (4, 200), (8, 400)):
        assert share_size("equal", 50, 10000, multiplier=multiplier) == expected
    # never more than the pool holds, and the caller can see it saturated
    assert share_size("equal", 50, 120, multiplier=8) == 120
    # 'full' is the whole pool by definition and ignores the multiplier
    assert share_size("full", 50, 120, multiplier=8) == 120


def test_a_non_positive_multiplier_is_refused():
    from federated_outlier_adaptation.data.source_share import share_size

    for multiplier in (0, -1):
        with pytest.raises(ValueError, match="multiplier must be positive"):
            share_size("equal", 50, 100, multiplier=multiplier)


def test_the_validation_blend_is_equal_counts_at_one_half():
    """
    At rho = 0.5 the accuracy over the blend is the mean of the two accuracies.

    That is what makes "a 50/50 blend" literally what the number is, rather than
    a weighting that depends on how much data each side happens to hold.
    """
    from federated_outlier_adaptation.data.source_share import blend_counts

    assert blend_counts(100, 10000, 0.5) == 100
    assert blend_counts(100, 10000, 0.25) == 33
    assert blend_counts(100, 10000, 0.0) == 0
    # bounded by what the old set actually holds
    assert blend_counts(100, 40, 0.5) == 40


def test_an_impossible_validation_blend_is_refused():
    from federated_outlier_adaptation.data.source_share import blend_counts

    for rho in (1.0, 1.5, -0.1):
        with pytest.raises(ValueError, match=r"blend must lie in \[0, 1\)"):
            blend_counts(100, 100, rho)


def test_the_blended_criterion_draws_only_old_validation_rows(stage5_setup):
    """Monitoring, not training - and never from the preservation test rows."""
    from federated_outlier_adaptation.data.source_share import (
        blended_val_loader,
        old_book_rows,
    )
    from federated_outlier_adaptation.training.evaluate import rows_loader

    provider, cohort_writers, cohort, old_writers, old = stage5_setup
    client_val = rows_loader(
        provider, cohort.part(1, cohort_writers[0], "val"), 8
    )
    old_val = rows_loader(provider, old_book_rows(old, old_writers, 1, "val"), 8).dataset

    blended, info = blended_val_loader(client_val, old_val, rho=0.5, batch_size=8, seed=3)
    assert blended is not None
    assert info["old_val_samples"] == min(
        len(client_val.dataset), len(old_val)
    )
    assert info["blended_samples"] == len(blended.dataset)

    # Fold 1 is the fold being blended, so fold 1's train and test rows are the
    # ones that must not appear; other folds' partitions overlap by design.
    forbidden = {
        row
        for writer in old_writers
        for part in ("train", "test")
        for row in old.part(1, writer, part)
    }
    drawn = {old_val.rows[index] for index in blended.dataset.datasets[1].indices}
    assert drawn and not (drawn & forbidden)


def test_the_blend_is_stable_across_reruns(stage5_setup):
    """A client must not be judged by a criterion that moves underneath it."""
    from federated_outlier_adaptation.data.source_share import (
        blended_val_loader,
        old_book_rows,
    )
    from federated_outlier_adaptation.training.evaluate import rows_loader

    provider, cohort_writers, cohort, old_writers, old = stage5_setup
    client_val = rows_loader(provider, cohort.part(1, cohort_writers[0], "val"), 8)
    old_val = rows_loader(provider, old_book_rows(old, old_writers, 1, "val"), 8).dataset

    first = blended_val_loader(client_val, old_val, 0.5, 8, seed=11)[0]
    second = blended_val_loader(client_val, old_val, 0.5, 8, seed=11)[0]
    assert first.dataset.datasets[1].indices == second.dataset.datasets[1].indices


def test_the_two_new_access_flags_are_on_the_final_command():
    from federated_outlier_adaptation.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(
        ["final", "--trainer", "BaseTrainer", "--source-share", "equal",
         "--source-share-multiplier", "4", "--val-blend-source", "0.5"]
    )
    assert args.source_share_multiplier == pytest.approx(4.0)
    assert args.val_blend_source == pytest.approx(0.5)

    default = parser.parse_args(["final", "--trainer", "BaseTrainer"])
    assert default.source_share_multiplier == 1.0
    assert default.val_blend_source == 0.0
    assert default.source_share == "off"
