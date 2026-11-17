"""PILOT DataManager subclass: local roots, stable IDs and explicit validation."""

from __future__ import annotations
import hashlib
import json
from pathlib import Path
import pickle
import random

import numpy as np
from torchvision import transforms
from torchvision.datasets import ImageFolder
from utils.data import iCIFAR224, iImageNetR
from utils.data_manager import DataManager
from .common import file_hash
from .pilot_imagenetr import validate_prepared_split


def split_training(labels, fraction=0.1, seed=0):
    if not 0 < fraction < 1:
        raise ValueError("Invalid development fraction")
    labels = np.asarray(labels)
    rng = random.Random(seed)
    val = set()
    for c in sorted(set(labels.tolist())):
        ids = np.flatnonzero(labels == c).tolist()
        rng.shuffle(ids)
        if len(ids) < 2:
            raise ValueError("Each development class needs two training images")
        n = max(1, min(len(ids) - 1, int(len(ids) * fraction)))
        val.update(ids[:n])
    return (
        np.asarray([i for i in range(len(labels)) if i not in val], dtype=np.int64),
        np.asarray(sorted(val), dtype=np.int64),
    )


def cifar_arrays(root, split, download=False):
    if split not in {"train", "test"}:
        raise ValueError("Unknown CIFAR-100 split")
    root = Path(root)
    path = root / "cifar-100-python" / split
    if not path.is_file() and download:
        from torchvision.datasets import CIFAR100

        CIFAR100(str(root), train=True, download=True)
    from torchvision.datasets import CIFAR100
    from torchvision.datasets.utils import check_integrity

    expected_md5 = dict(
        CIFAR100.train_list if split == "train" else CIFAR100.test_list
    )[split]
    if not check_integrity(str(path), expected_md5):
        raise ValueError("CIFAR-100 file does not match the official split checksum")
    with path.open("rb") as stream:
        value = pickle.load(stream, encoding="latin1")
    images = value["data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    labels = np.asarray(value["fine_labels"], dtype=np.int64)
    expected = 50000 if split == "train" else 10000
    if len(labels) != expected or len(set(labels)) != 100:
        raise ValueError("Unexpected official CIFAR split")
    return images, labels, file_hash(path), False, None


def folder_arrays(root, split, inventory=None):
    root = Path(root).resolve()
    source = ImageFolder(str(root / split))
    if len(source.classes) != 200:
        raise ValueError("Prepared ImageNet-R split must contain 200 classes")
    images = np.asarray([p for p, _ in source.samples])
    labels = np.asarray(source.targets, dtype=np.int64)
    manifest = [(str(Path(p).relative_to(root)), file_hash(p)) for p in images]
    if inventory is not None:
        inventory[split] = [digest for _, digest in manifest]
    digest = hashlib.sha256(
        json.dumps(manifest, separators=(",", ":")).encode()
    ).hexdigest()
    return images, labels, digest, True, source.class_to_idx


class ProtocolDataManager(DataManager):
    """Reuse upstream task construction, DummyDataset and preprocessing.

    Formal mode uses complete designated training data. Development and smoke
    never read the official test pickle/folder. They use a fixed train holdout.
    """

    def _setup_data(self, dataset_name, shuffle, seed):
        args = self.args
        phase = args.get("phase", "final")
        smoke = bool(args.get("smoke", False))
        folder_inventory = {}
        prepared_split_report = None
        if phase not in {"final", "development", "smoke"}:
            raise ValueError("Unknown data phase")
        if smoke != (phase == "smoke") or smoke != (dataset_name == "synthetic"):
            raise ValueError("Synthetic-only runs must be explicitly labeled smoke")
        development = phase != "final"
        if dataset_name == "synthetic":
            classes = int(args.get("synthetic_classes", 6))
            per = int(args.get("synthetic_per_class", 12))
            size = int(args.get("image_size", 32))
            rng = np.random.default_rng(101)
            labels = np.repeat(np.arange(classes), per)
            images = rng.integers(0, 35, (len(labels), size, size, 3), dtype=np.uint8)
            for i, c in enumerate(labels):
                images[i, :, :, int(c) % 3] += np.uint8(70 + (int(c) * 13) % 100)
            source_hash = hashlib.sha256(
                images.tobytes() + labels.tobytes()
            ).hexdigest()
            use_path = False
            mapping = None
            train_trsf = [
                transforms.RandomResizedCrop(size, scale=(0.05, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
            test_trsf = [
                transforms.Resize(int(size * 256 / 224), interpolation=3),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
            ]
            common = []
            loader = None
        elif dataset_name == "cifar224":
            loader = lambda split: cifar_arrays(
                args.get("data_root", "data"), split, bool(args.get("download", False))
            )
            images, labels, source_hash, use_path, mapping = loader("train")
            owner = iCIFAR224(args)
            train_trsf = owner.train_trsf
            test_trsf = owner.test_trsf
            common = owner.common_trsf
        elif dataset_name == "imagenetr":
            loader = lambda split: folder_arrays(
                args.get("data_root", "data/imagenet-r"), split, folder_inventory
            )
            images, labels, source_hash, use_path, mapping = loader("train")
            owner = iImageNetR(args)
            train_trsf = owner.train_trsf
            test_trsf = owner.test_trsf
            common = owner.common_trsf
        else:
            raise ValueError(
                "This comparison supports cifar224 and imagenetr; smoke uses synthetic"
            )
        original = np.asarray(labels, dtype=np.int64)
        classes = len(set(original.tolist()))
        if sorted(set(original.tolist())) != list(range(classes)):
            raise ValueError("Noncontiguous original labels")
        if development:
            fit, val = split_training(
                original,
                float(args.get("validation_fraction", 0.1)),
                int(args.get("split_seed", 0)),
            )
            train_images, train_y = images[fit], original[fit]
            eval_images, eval_y = images[val], original[val]
            train_ids = fit
            eval_ids = val
            eval_hash = source_hash
        else:
            fit = np.arange(len(original))
            train_images, train_y = images, original
            train_ids = fit
            eval_images, eval_y, eval_hash, _, eval_mapping = loader("test")
            if eval_mapping != mapping:
                raise ValueError("Train/test class-to-index mappings differ")
            eval_ids = np.arange(len(eval_y)) + 10**9
        if dataset_name == "imagenetr":
            split_digests = {"train": source_hash}
            if not development:
                split_digests["test"] = eval_hash
            prepared_split_report = validate_prepared_split(
                args.get("prepared_split_overlap_policy", "reject"),
                split_digests,
                folder_inventory,
                development,
            )
        # Only explicitly marked validation jobs may restrict data per class.
        limit = args.get("validation_max_per_class")
        if limit is not None:
            if not args.get("validation_only") or not development:
                raise ValueError("Subset restriction is validation-only")
            limit = int(limit)
            if limit < 1:
                raise ValueError("Invalid validation subset limit")
            ti = np.concatenate(
                [np.flatnonzero(train_y == c)[:limit] for c in range(classes)]
            )
            ei = np.concatenate(
                [np.flatnonzero(eval_y == c)[:limit] for c in range(classes)]
            )
            train_images, train_y, train_ids = (
                train_images[ti],
                train_y[ti],
                train_ids[ti],
            )
            eval_images, eval_y, eval_ids = eval_images[ei], eval_y[ei], eval_ids[ei]
        if set(map(int, train_ids)) & set(map(int, eval_ids)):
            raise ValueError("Training/evaluation ID overlap")
        if shuffle:
            # This intentionally follows upstream's MT19937 and global state.
            np.random.seed(seed)
            order = np.random.permutation(classes).tolist()
        else:
            order = list(range(classes))
        inverse = np.argsort(order)
        self._train_data, self._train_targets = train_images, inverse[train_y]
        self._test_data, self._test_targets = eval_images, inverse[eval_y]
        self._train_ids, self._test_ids = np.asarray(train_ids), np.asarray(eval_ids)
        self.use_path = use_path
        self._class_order = order
        self._train_trsf, self._test_trsf, self._common_trsf = (
            train_trsf,
            test_trsf,
            common,
        )
        self.manifest = {
            "dataset": dataset_name,
            "phase": phase,
            "class_order": order,
            "order_rule": "PILOT np.random.seed and np.random.permutation",
            "train_count": len(train_y),
            "evaluation_count": len(eval_y),
            "evaluation_source": (
                "training_holdout" if development else "designated_test"
            ),
            "official_test_read": not development,
            "train_source_sha256": source_hash,
            "evaluation_source_sha256": eval_hash,
            "split_seed": int(args.get("split_seed", 0)),
            "fit_ids_sha256": hashlib.sha256(self._train_ids.tobytes()).hexdigest(),
            "evaluation_ids_sha256": hashlib.sha256(
                self._test_ids.tobytes()
            ).hexdigest(),
            "validation_only": bool(args.get("validation_only", False)),
            "cross_split_content_overlap_checked": bool(folder_inventory)
            and not development,
            "preprocessing_source": "upstream utils/data.py; synthetic scaled geometry for smoke",
            "normalization": "no additional pixel mean/std normalization",
            "class_mapping": mapping,
        }
        if prepared_split_report is not None:
            self.manifest["prepared_split_provenance"] = prepared_split_report
        self.manifest["fingerprint"] = hashlib.sha256(
            json.dumps(self.manifest, sort_keys=True).encode()
        ).hexdigest()

    def get_dataset(
        self, indices, source, mode, appendent=None, ret_data=False, m_rate=None
    ):
        result = super().get_dataset(indices, source, mode, appendent, ret_data, m_rate)
        dataset = result[-1] if ret_data else result
        if appendent is None and m_rate is None:
            labels = self._train_targets if source == "train" else self._test_targets
            ids = self._train_ids if source == "train" else self._test_ids
            chosen = (
                np.concatenate([np.flatnonzero(labels == int(c)) for c in indices])
                if len(indices)
                else np.empty(0, dtype=np.int64)
            )
            dataset.ids = np.asarray(ids[chosen], dtype=np.int64)
            if len(dataset.ids) != len(dataset):
                raise RuntimeError("PILOT view and stable IDs disagree")
        return result
