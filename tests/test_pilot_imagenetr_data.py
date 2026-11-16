"""Verified original-split exception must never become a generic bypass."""

import shutil

import pytest

from cl27 import data as data_module
from cl27 import pilot_imagenetr
from cl27.common import file_hash
from cl27.data import ProtocolDataManager, folder_arrays


@pytest.fixture
def folder_root(tmp_path):
    # ImageFolder inventories files without decoding them in _setup_data.
    # Two train entries per class permit train-only development splitting.
    for split, per_class in (("train", 2), ("test", 1)):
        for cls in range(200):
            directory = tmp_path / split / f"n{cls:08d}"
            directory.mkdir(parents=True)
            for index in range(per_class):
                (directory / f"image_{index}.jpg").write_bytes(
                    f"{split}-{cls}-{index}".encode()
                )
    return tmp_path


def arguments(root, **updates):
    return {
        "data_root": str(root),
        "model_name": "shared_basis",
        "phase": "final",
        "backbone_type": "vit_base_patch16_224",
        **updates,
    }


def manager(root, **updates):
    return ProtocolDataManager(
        "imagenetr", True, 1993, 20, 20, arguments(root, **updates)
    )


def add_overlap(root):
    source = root / "train/n00000000/image_0.jpg"
    target = root / "test/n00000000/image_0.jpg"
    shutil.copyfile(source, target)
    return file_hash(source)


def pin_fixture_identity(root, monkeypatch, overlap_hashes):
    inventory = {}
    identities = {
        split: folder_arrays(root, split, inventory)[2]
        for split in ("train", "test")
    }
    monkeypatch.setattr(pilot_imagenetr, "SPLIT_SHA256", identities)
    monkeypatch.setattr(
        pilot_imagenetr, "SPLIT_COUNTS",
        {split: len(rows) for split, rows in inventory.items()},
    )
    monkeypatch.setattr(pilot_imagenetr, "KNOWN_OVERLAP_SHA256", frozenset(overlap_hashes))
    return identities


def test_default_nonoverlapping_preparation_unchanged(folder_root):
    dm = manager(folder_root)
    assert dm.nb_tasks == 10
    assert len(dm._train_data) == 400 and len(dm._test_data) == 200
    report = dm.manifest["prepared_split_provenance"]
    assert report["policy"] == "reject"
    assert report["observed_cross_split_duplicate_hash_count"] == 0
    assert report["verified_splits"] == []
    assert "source_archive_sha256" not in report


def test_default_rejects_unknown_identical_content(folder_root):
    add_overlap(folder_root)
    with pytest.raises(ValueError, match="contain identical image files"):
        manager(folder_root)


def test_user_archive_hash_cannot_authorize_unverified_tree(folder_root):
    add_overlap(folder_root)
    with pytest.raises(ValueError, match="train tree differs"):
        manager(
            folder_root,
            prepared_split_overlap_policy="preserve_verified_pilot",
            source_archive_sha256=pilot_imagenetr.SOURCE_ARCHIVE_SHA256,
        )


def test_verified_tree_allows_only_explicit_known_overlap(folder_root, monkeypatch):
    digest = add_overlap(folder_root)
    identities = pin_fixture_identity(folder_root, monkeypatch, [digest])
    dm = manager(folder_root, prepared_split_overlap_policy="preserve_verified_pilot")
    report = dm.manifest["prepared_split_provenance"]
    assert report["verified_splits"] == ["train", "test"]
    assert report["observed_split_manifest_sha256"] == identities
    assert report["source_archive_sha256"] == pilot_imagenetr.SOURCE_ARCHIVE_SHA256
    assert report["observed_cross_split_duplicate_hash_count"] == 1
    assert report["observed_cross_split_duplicate_pair_count"] == 1
    assert report["original_split_preserved"] and not report["deduplicated"]
    assert len(dm._train_data) == 400 and len(dm._test_data) == 200
    # Even a recognized preparation still needs the explicit policy.
    with pytest.raises(ValueError, match="contain identical image files"):
        manager(folder_root)


@pytest.mark.parametrize("split", ["train", "test"])
@pytest.mark.parametrize("change", ["bytes", "path"])
def test_changed_tree_cannot_use_exception(folder_root, monkeypatch, split, change):
    digest = add_overlap(folder_root)
    pin_fixture_identity(folder_root, monkeypatch, [digest])
    path = folder_root / split / "n00000001/image_0.jpg"
    if change == "bytes":
        path.write_bytes(b"changed-content")
    else:
        path.rename(path.with_name("renamed.jpg"))
    with pytest.raises(ValueError, match=f"{split} tree differs"):
        manager(folder_root, prepared_split_overlap_policy="preserve_verified_pilot")


def test_matching_digests_do_not_allow_unknown_overlap(folder_root, monkeypatch):
    add_overlap(folder_root)
    pin_fixture_identity(folder_root, monkeypatch, ["some-other-known-hash"])
    with pytest.raises(ValueError, match="overlap differs"):
        manager(folder_root, prepared_split_overlap_policy="preserve_verified_pilot")


def test_matching_digests_do_not_allow_changed_duplicate_multiplicity(folder_root, monkeypatch):
    digest = add_overlap(folder_root)
    shutil.copyfile(
        folder_root / "train/n00000000/image_0.jpg",
        folder_root / "train/n00000000/image_1.jpg",
    )
    pin_fixture_identity(folder_root, monkeypatch, [digest])
    with pytest.raises(ValueError, match="overlap differs"):
        manager(folder_root, prepared_split_overlap_policy="preserve_verified_pilot")


def test_verified_development_reads_only_train(folder_root, monkeypatch):
    digest = add_overlap(folder_root)
    identities = pin_fixture_identity(folder_root, monkeypatch, [digest])
    shutil.rmtree(folder_root / "test")
    actual_loader = folder_arrays
    calls = []

    def train_only(root, split, inventory):
        calls.append(split)
        assert split == "train", "Development must not open designated test"
        return actual_loader(root, split, inventory)

    monkeypatch.setattr(data_module, "folder_arrays", train_only)
    dm = manager(
        folder_root,
        phase="development",
        validation_only=True,
        prepared_split_overlap_policy="preserve_verified_pilot",
    )
    assert calls == ["train"]
    assert not dm.manifest["official_test_read"]
    report = dm.manifest["prepared_split_provenance"]
    assert report["verified_splits"] == ["train"]
    assert report["observed_split_manifest_sha256"] == {"train": identities["train"]}
    assert report["observed_cross_split_duplicate_hash_count"] is None
    assert report["observed_cross_split_duplicate_pair_count"] is None


def test_verified_development_still_rejects_changed_train(folder_root, monkeypatch):
    digest = add_overlap(folder_root)
    pin_fixture_identity(folder_root, monkeypatch, [digest])
    (folder_root / "train/n00000001/image_0.jpg").write_bytes(b"changed")
    with pytest.raises(ValueError, match="train tree differs"):
        manager(
            folder_root,
            phase="development",
            validation_only=True,
            prepared_split_overlap_policy="preserve_verified_pilot",
        )


def test_unknown_policy_is_rejected(folder_root):
    with pytest.raises(ValueError, match="Unknown prepared_split_overlap_policy"):
        manager(folder_root, prepared_split_overlap_policy="allow_all")
