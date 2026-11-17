"""Native PILOT exemplar rules, with bounded original-image access."""

from __future__ import annotations
from dataclasses import dataclass
import copy
import operator
import numpy as np
from PIL import Image
from utils.data_manager import DummyDataset


@dataclass
class RawRecord:
    image: object
    label: int
    sample_id: int


class ReplayMemory:
    def __init__(self, args):
        self.fixed = bool(args.get("fixed_memory", False))
        self.memory_size = int(args.get("memory_size", 0))
        self.memory_per_class = int(args.get("memory_per_class", 0))
        self.selection = args.get("replay_selection", "herding")
        if self.selection not in {"herding", "random"}:
            raise ValueError("Unknown replay selection")
        if min(self.memory_size, self.memory_per_class) < 0:
            raise ValueError("Negative replay budget")
        self.records = []
        self.classes = []
        self.seen = 0
        self.rng = np.random.Generator(np.random.PCG64(int(args.get("seed", 0)) + 71))

    def __len__(self):
        return len(self.records)

    def quota(self, n_classes):
        if isinstance(n_classes, (bool, np.bool_)):
            raise ValueError("Invalid seen class count")
        n_classes = operator.index(n_classes)
        if n_classes < 1:
            raise ValueError("Replay quota requires at least one seen class")
        return self.memory_per_class if self.fixed else self.memory_size // n_classes

    def update(self, current_dataset, current_raw_features, seen_classes):
        raw_y = np.asarray(current_dataset.labels)
        raw_ids = np.asarray(current_dataset.ids)
        if (
            raw_y.ndim != 1
            or raw_ids.shape != raw_y.shape
            or raw_y.dtype.kind not in "iu"
            or raw_ids.dtype.kind not in "iu"
            or not len(raw_y)
        ):
            raise ValueError(
                "Replay arrivals need aligned, nonempty integer labels and IDs"
            )
        y = raw_y.astype(np.int64, copy=False)
        ids = raw_ids.astype(np.int64, copy=False)
        x = np.asarray(current_raw_features, dtype=np.float64)
        if x.ndim != 2 or len(x) != len(y) or not np.isfinite(x).all():
            raise ValueError("Invalid replay-selection features")
        if len(np.unique(ids)) != len(ids) or set(ids) & {
            r.sample_id for r in self.records
        }:
            raise ValueError("Duplicate replay arrivals")
        if set(y) & set(self.classes):
            raise ValueError("A class may be committed only once")
        classes = list(seen_classes)
        if any(
            isinstance(c, (bool, np.bool_)) or not isinstance(c, (int, np.integer))
            for c in classes
        ):
            raise ValueError("Registered classes must be integer labels")
        classes = list(map(int, classes))
        if classes != sorted(set(self.classes) | set(map(int, y))):
            raise ValueError(
                "Registered classes must preserve exactly the old and arriving vocabulary"
            )
        if len(current_dataset.images) != len(y):
            raise ValueError("Replay images and labels are not aligned")
        for image in current_dataset.images:
            self._validate_image(image)
        m = self.quota(len(classes))
        if m < 1:
            raise ValueError("Budget does not cover every seen class")
        records = [
            r
            for c in self.classes
            for r in [z for z in self.records if z.label == c][:m]
        ]
        for c in sorted(set(y)):
            pool = np.flatnonzero(y == c)
            k = min(m, len(pool))
            if self.selection == "random":
                chosen = self.rng.choice(pool, size=k, replace=False)
            else:
                values = x[pool]
                values = values / np.maximum(
                    np.linalg.norm(values, axis=1, keepdims=True), 1e-12
                )
                mean = values.mean(0)
                running = np.zeros(values.shape[1])
                available = np.ones(len(pool), dtype=bool)
                selected = []
                for rank in range(1, k + 1):
                    distance = ((mean - (values + running) / rank) ** 2).sum(1)
                    distance[~available] = np.inf
                    j = int(distance.argmin())
                    available[j] = False
                    selected.append(pool[j])
                    running += values[j]
                chosen = np.asarray(selected, dtype=np.int64)
            for i in chosen:
                image = current_dataset.images[int(i)]
                records.append(
                    RawRecord(
                        image.copy() if isinstance(image, np.ndarray) else str(image),
                        int(c),
                        int(ids[i]),
                    )
                )
        self.records = records
        self.classes = classes
        self.seen += len(y)
        if len(self.records) > m * len(classes):
            raise RuntimeError("Replay exceeds protocol budget")

    def dataset(self, transform):
        if not self.records:
            raise ValueError("Empty replay dataset")
        use_path = isinstance(self.records[0].image, str)
        images = np.asarray([r.image for r in self.records])
        labels = np.asarray([r.label for r in self.records], dtype=np.int64)
        result = DummyDataset(images, labels, transform, use_path)
        result.ids = np.asarray([r.sample_id for r in self.records], dtype=np.int64)
        return result

    def sample_indices(self, n, rng):
        return (
            rng.choice(
                len(self.records), size=min(int(n), len(self.records)), replace=False
            )
            if self.records
            else np.empty(0, dtype=np.int64)
        )

    def payload_bytes(self):
        total = 0
        for r in self.records:
            if isinstance(r.image, np.ndarray):
                total += r.image.nbytes
            else:
                with Image.open(r.image) as image:
                    total += image.width * image.height * 3
            total += 16
        return int(total)

    def state_dict(self):
        return copy.deepcopy(
            {
                "version": 1,
                "fixed": self.fixed,
                "memory_size": self.memory_size,
                "memory_per_class": self.memory_per_class,
                "selection": self.selection,
                "records": self.records,
                "classes": self.classes,
                "seen": self.seen,
                "rng": self.rng.bit_generator.state,
            }
        )

    def load_state_dict(self, state):
        expected = (
            1,
            self.fixed,
            self.memory_size,
            self.memory_per_class,
            self.selection,
        )
        actual = tuple(
            state[k]
            for k in [
                "version",
                "fixed",
                "memory_size",
                "memory_per_class",
                "selection",
            ]
        )
        if actual != expected:
            raise ValueError("Replay checkpoint protocol differs")
        records = copy.deepcopy(state["records"])
        classes = list(state["classes"])
        if any(
            isinstance(c, (bool, np.bool_)) or not isinstance(c, (int, np.integer))
            for c in classes
        ):
            raise ValueError("Invalid replay checkpoint vocabulary")
        classes = list(map(int, classes))
        if classes != sorted(set(classes)):
            raise ValueError("Invalid replay checkpoint vocabulary")
        seen = state["seen"]
        if (
            isinstance(seen, (bool, np.bool_))
            or not isinstance(seen, (int, np.integer))
            or seen < 0
        ):
            raise ValueError("Invalid replay historical count")
        seen = int(seen)
        if not isinstance(records, list) or any(
            not isinstance(r, RawRecord) for r in records
        ):
            raise ValueError("Invalid replay checkpoint records")
        for r in records:
            if (
                isinstance(r.sample_id, (bool, np.bool_))
                or not isinstance(r.sample_id, (int, np.integer))
                or isinstance(r.label, (bool, np.bool_))
                or not isinstance(r.label, (int, np.integer))
                or r.label not in classes
            ):
                raise ValueError("Invalid replay checkpoint record identity")
            self._validate_image(r.image)
        if len({r.sample_id for r in records}) != len(records):
            raise ValueError("Duplicate replay checkpoint IDs")
        if not classes:
            if records or seen:
                raise ValueError("Empty replay vocabulary has historical records")
        else:
            m = self.quota(len(classes))
            sizes = [sum(r.label == c for r in records) for c in classes]
            if m < 1 or any(n < 1 or n > m for n in sizes):
                raise ValueError("Invalid replay per-class checkpoint quota")
            if seen < max(len(records), len(classes)):
                raise ValueError("Invalid replay historical count")
        checked_rng = np.random.Generator(np.random.PCG64(0))
        checked_rng.bit_generator.state = copy.deepcopy(state["rng"])
        # Commit only after every invariant and the generator state have passed.
        self.records, self.classes, self.seen = records, classes, seen
        self.rng = checked_rng

    @staticmethod
    def _validate_image(image):
        if isinstance(image, (str, np.str_)):
            if not str(image):
                raise ValueError("Replay path is empty")
        elif isinstance(image, np.ndarray):
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
                raise ValueError("Replay arrays must contain raw uint8 RGB images")
        else:
            raise ValueError("Replay supports raw RGB arrays or original file paths")
