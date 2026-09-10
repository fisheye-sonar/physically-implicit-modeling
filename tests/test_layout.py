"""pim.environments.layout — the one place dataset paths are built (layout v2, 2026-09-10).

Pure path logic on a fake ``datasets/`` tree under tmp_path: the v1 fallback before an
instance carries ``layout.json``, the v2 resolution after, the logical probe-cache key and
the maps from the paths older callers pass onto it. No real data is touched.
"""
from __future__ import annotations

import json

import pytest

from pim.environments import layout


@pytest.fixture
def fake_datasets(tmp_path, monkeypatch):
    root = tmp_path / "datasets"
    (root / "discworld" / "dw-x").mkdir(parents=True)
    (root / "othello" / "oth-x").mkdir(parents=True)
    monkeypatch.setattr(layout, "DATASETS", root)
    return root


def test_v1_fallback_then_v2_after_marker(fake_datasets):
    r = fake_datasets / "discworld" / "dw-x"
    assert not layout.is_migrated("discworld", "dw-x")
    assert layout.probe_file("discworld", "dw-x", "120k") == r / "probe" / "test.h5"
    assert layout.probe_file("discworld", "dw-x", "250k") == r / "probe_250k" / "test.h5"
    assert layout.probe_manifest("discworld", "dw-x", "250k") == r / "probe_250k" / "dataset.json"
    assert layout.edits_file("discworld", "dw-x") == r / "eval" / "edits.h5"
    assert layout.edits_selection("discworld", "dw-x") == r / "edits_selection.json"
    assert layout.eval_file("discworld", "dw-x") == r / "eval" / "test.h5"
    assert layout.othello_split_file("oth-x", "probe_large", 170_000) == (
        fake_datasets / "othello" / "oth-x" / "corpus" / "probe_large_170000.npz")

    layout.write_marker("discworld", "dw-x", moves=[["a", "b"]])
    layout.write_marker("othello", "oth-x")
    assert layout.is_migrated("discworld", "dw-x")
    assert json.loads(layout.layout_file("discworld", "dw-x").read_text())["moves"] == [["a", "b"]]
    assert layout.probe_file("discworld", "dw-x", "120k") == r / "probe" / "probe_120k.h5"
    assert layout.probe_file("discworld", "dw-x", "250k") == r / "probe" / "probe_250k.h5"
    assert layout.probe_manifest("discworld", "dw-x", "250k") == r / "probe" / "probe_250k.json"
    assert layout.edits_file("discworld", "dw-x") == r / "edits" / "v1" / "edits.h5"
    assert layout.edits_manifest("discworld", "dw-x") == r / "edits" / "v1" / "edits.json"
    assert layout.edits_selection("discworld", "dw-x") == r / "edits" / "v1" / "selection.json"
    assert layout.edits_dir("discworld", "dw-x", "v2") == r / "edits" / "v2"
    assert layout.eval_manifest("discworld", "dw-x") == r / "eval" / "test.json"
    o = fake_datasets / "othello" / "oth-x"
    assert layout.othello_split_file("oth-x", "train", 20_000_000) == o / "train" / "train_20000000.npz"
    assert layout.othello_split_file("oth-x", "test", 10_000) == o / "eval" / "test_10000.npz"
    assert layout.othello_split_file("oth-x", "probe_large", 170_000) == o / "probe" / "probe_large_170000.npz"
    assert layout.othello_cases_file("oth-x") == o / "edits" / "v1" / "cases_1001.pkl"
    # eval/test.h5 and train/ do not move
    assert layout.eval_file("discworld", "dw-x") == r / "eval" / "test.h5"
    assert layout.train_dir("discworld", "dw-x") == r / "train"


def test_ensure_marker_refuses_an_unmigrated_v1_instance(fake_datasets):
    r = fake_datasets / "discworld" / "dw-x"
    (r / "probe").mkdir()
    (r / "probe" / "test.h5").write_bytes(b"")
    with pytest.raises(RuntimeError):
        layout.ensure_marker("discworld", "dw-x")
    # a genuinely new instance is stamped at birth
    (fake_datasets / "discworld" / "dw-new").mkdir()
    layout.ensure_marker("discworld", "dw-new")
    assert layout.is_migrated("discworld", "dw-new")


def test_probe_key_is_logical_and_legacy_paths_map_onto_it(fake_datasets):
    assert layout.probe_key("discworld", "dw-x", "120k") == ("discworld/dw-x", "probe_120k")
    assert layout.probe_key("discworld", "dw-x", "250k") == ("discworld/dw-x", "probe_250k")
    with pytest.raises(KeyError):
        layout.probe_key("discworld", "dw-x", "1k")
    r = fake_datasets / "discworld" / "dw-x"
    # the forms older callers pass, absolute or datasets/-relative, v1 or v2
    assert layout.legacy_probe_key(r / "probe") == ("discworld", "dw-x", "120k")
    assert layout.legacy_probe_key(r / "probe_250k") == ("discworld", "dw-x", "250k")
    assert layout.legacy_probe_key(r / "probe" / "test.h5") == ("discworld", "dw-x", "120k")
    assert layout.legacy_probe_key(r / "probe" / "probe_250k.h5") == ("discworld", "dw-x", "250k")
    assert layout.legacy_probe_key("datasets/discworld/dw-x/probe") == ("discworld", "dw-x", "120k")
    # anything outside the instances keeps its path-based key
    assert layout.legacy_probe_key("/somewhere/else/probe") is None
    assert layout.legacy_probe_key("experiments/blink_ablation/pilot/data") is None
    assert layout.legacy_probe_key(r / "eval") is None


def test_legacy_edits_dir_maps_onto_the_instance(fake_datasets):
    r = fake_datasets / "discworld" / "dw-x"
    assert layout.legacy_edits_instance(r / "eval") == ("discworld", "dw-x")
    assert layout.legacy_edits_instance(r / "edits" / "v1") == ("discworld", "dw-x")
    assert layout.legacy_edits_instance("datasets/discworld/dw-x/eval") == ("discworld", "dw-x")
    assert layout.legacy_edits_instance(r / "probe") is None
    assert layout.legacy_edits_instance("/elsewhere/eval") is None


def test_parse_dataset_path(fake_datasets):
    r = fake_datasets / "discworld" / "dw-x"
    assert layout.parse_dataset_path(r) == {"cls": "discworld", "inst": "dw-x", "rel": ()}
    assert layout.parse_dataset_path(r / "probe" / "x.h5")["rel"] == ("probe", "x.h5")
    assert layout.parse_dataset_path("datasets/othello/oth-x/corpus")["cls"] == "othello"
    assert layout.parse_dataset_path("/other/datasets/discworld/dw-x") is None   # another tree
    assert layout.parse_dataset_path(fake_datasets / "archive" / "x") is None      # not a class
