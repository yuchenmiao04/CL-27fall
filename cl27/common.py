"""Native PILOT interfaces shared by the three additional learners."""

from __future__ import annotations

import copy
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from models.base import BaseLearner

UPSTREAM_COMMIT = "7a6e904c5bc5cb7a4e1823b3434020be27469b63"
NEW_METHODS = {"shared_basis", "history_transport", "current_adapted"}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = False


def state_hash(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for value in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(value)
    return digest.hexdigest()


class TinyPatchEmbed(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.proj = nn.Conv2d(3, width, 8, 8)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class TinyBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.norm1 = nn.LayerNorm(width, eps=1e-6)
        self.attn = nn.MultiheadAttention(width, 3, dropout=0.0, batch_first=True)
        self.norm2 = nn.LayerNorm(width, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width)
        )

    def forward(self, x):
        y = self.norm1(x)
        x = x + self.attn(y, y, y, need_weights=False)[0]
        return x + self.mlp(self.norm2(x))


class TinyBackbone(nn.Module):
    """Explicit synthetic-only token ViT; not a pretrained benchmark surrogate."""

    def __init__(self, width=24, depth=3, image_size=32):
        super().__init__()
        if width % 3 or image_size % 8:
            raise ValueError("Invalid synthetic backbone dimensions")
        self.out_dim = self.num_features = width
        self.image_size = image_size
        self.patch_embed = TinyPatchEmbed(width)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, width))
        self.pos_embed = nn.Parameter(
            torch.randn(1, (image_size // 8) ** 2 + 1, width) * 0.02
        )
        self.blocks = nn.Sequential(*[TinyBlock(width) for _ in range(depth)])
        self.norm = nn.LayerNorm(width, eps=1e-6)

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = torch.cat([self.cls_token.expand(len(x), -1, -1), x], 1) + self.pos_embed
        return self.norm(self.blocks(x))

    def forward(self, x):
        return self.forward_features(x)[:, 0]


def build_backbone(args):
    """Reuse PILOT's native timm backbone; no parameter-name conversion."""
    if args.get("smoke", False):
        if args.get("dataset") != "synthetic" or args.get("pretrained", False):
            raise ValueError(
                "Tiny backbone is only permitted in labeled synthetic smoke runs"
            )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(args.get("backbone_seed", args.get("seed", 0))))
            model = TinyBackbone(
                int(args.get("smoke_feature_dim", 24)),
                int(args.get("smoke_depth", 3)),
                int(args.get("image_size", 32)),
            )
        model.cl27_provenance = {
            "kind": "synthetic_random",
            "initial_state_sha256": state_hash(model),
        }
        return model
    import timm

    if timm.__version__ != "0.6.12":
        raise RuntimeError(
            "The historical PILOT protocol requires timm==0.6.12; untagged newer weights are rejected"
        )
    if args.get("backbone_type") not in {
        "vit_base_patch16_224",
        "pretrained_vit_b16_224",
    }:
        raise ValueError("Main comparison requires PILOT IN21K->IN1K ViT-B/16")
    from utils.inc_net import get_backbone

    model = get_backbone(args, pretrained=True)
    url = model.default_cfg["url"]
    weight = Path(torch.hub.get_dir()) / "checkpoints" / url.rsplit("/", 1)[1]
    if not weight.is_file():
        raise FileNotFoundError("Cannot bind the exact pretrained weight file")
    model.cl27_provenance = {
        "kind": "pretrained",
        "timm_version": timm.__version__,
        "checkpoint_family": "augreg_in21k_ft_in1k",
        "public_weight_url": url,
        "weight_file_sha256": file_hash(weight),
        "initial_state_sha256": state_hash(model),
        "upstream_commit": UPSTREAM_COMMIT,
    }
    return model


def feature_normalization(args):
    """Resolve one feature coordinate convention; reject ambiguous aliases."""
    canonical = args.get("feature_normalization")
    legacy = args.get("normalization")
    if canonical is not None and legacy is not None and canonical != legacy:
        raise ValueError("Conflicting feature_normalization and normalization")
    value = (
        canonical if canonical is not None else legacy if legacy is not None else "l2"
    )
    if value not in {"l2", "none"}:
        raise ValueError("Unknown analytic feature normalization")
    return value


def normalize_features(x, normalization="l2"):
    if normalization == "l2":
        return torch.nn.functional.normalize(x, dim=-1)
    if normalization == "none":
        return x
    raise ValueError("Unknown analytic feature normalization")


@torch.no_grad()
def extract_features(backbone, dataset, device, batch_size=64, normalization="l2"):
    backbone.eval()
    features = []
    labels = []
    for _, images, target in DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0
    ):
        raw = backbone(images.to(device))
        features.append(
            normalize_features(raw, normalization).detach().cpu().double().numpy()
        )
        labels.append(target.numpy())
    if not features:
        raise ValueError("Cannot encode an empty dataset")
    ids = np.asarray(getattr(dataset, "ids", np.arange(len(dataset))), dtype=np.int64)
    if ids.shape != (len(dataset),):
        raise ValueError("Invalid stable dataset IDs")
    return np.concatenate(features), np.concatenate(labels).astype(np.int64), ids.copy()


class ScoreNetwork(nn.Module):
    def __init__(self, backbone, predict_scores, normalization="l2"):
        super().__init__()
        self.backbone = backbone
        self.predict_scores = predict_scores
        self.normalization = normalization
        self.feature_dim = backbone.out_dim

    def extract_vector(self, x):
        return self.backbone(x)

    def forward(self, x):
        raw = self.backbone(x)
        value = (
            normalize_features(raw, self.normalization).detach().cpu().double().numpy()
        )
        scores = self.predict_scores(value)
        if not np.isfinite(scores).all():
            raise FloatingPointError("Nonfinite analytic scores")
        return {"logits": torch.as_tensor(scores, device=x.device, dtype=raw.dtype)}


class IntegratedLearner(BaseLearner):
    """PILOT task lifecycle plus raw evaluation records and resumable state."""

    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.task_diagnostics = []
        self._evaluation_record = None

    def start_task(self, data_manager):
        if self._known_classes != self._total_classes:
            raise RuntimeError(
                "after_task must complete before beginning the next task"
            )
        self._cur_task += 1
        if self._cur_task >= data_manager.nb_tasks:
            raise ValueError("No remaining task")
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        current = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="test",
        )
        if not hasattr(current, "ids"):
            raise ValueError(
                "The protocol data manager must supply stable original IDs"
            )
        test = data_manager.get_dataset(
            np.arange(self._total_classes), source="test", mode="test"
        )
        self.topk = min(5, self._total_classes)
        self.test_loader = DataLoader(
            test,
            batch_size=int(self.args.get("eval_batch_size", 64)),
            shuffle=False,
            num_workers=int(self.args.get("num_workers", 0)),
        )
        return current, self.test_loader

    def after_task(self):
        self._known_classes = self._total_classes

    def eval_task(self):
        predictions, labels = self._eval_cnn(self.test_loader)
        self._evaluation_record = {
            "predictions": predictions[:, 0].astype(int).tolist(),
            "labels": labels.astype(int).tolist(),
        }
        return self._evaluate(predictions, labels), None

    def checkpoint_core(self):
        raise NotImplementedError

    def restore_core(self, state):
        raise NotImplementedError

    def checkpoint_state(self):
        if self._known_classes != self._total_classes:
            raise RuntimeError("Save only after a completed task")
        return {
            "version": 1,
            "method": self.args["model_name"],
            "cur_task": self._cur_task,
            "known_classes": self._known_classes,
            "total_classes": self._total_classes,
            "task_diagnostics": copy.deepcopy(self.task_diagnostics),
            "core": self.checkpoint_core(),
        }

    def restore_checkpoint(self, state):
        if state["version"] != 1 or state["method"] != self.args["model_name"]:
            raise ValueError("Checkpoint method/version mismatch")
        if state["known_classes"] != state["total_classes"]:
            raise ValueError("Partial task checkpoint")
        self.restore_core(state["core"])
        self._cur_task = int(state["cur_task"])
        self._known_classes = int(state["known_classes"])
        self._total_classes = int(state["total_classes"])
        self.task_diagnostics = copy.deepcopy(state["task_diagnostics"])


def jsonable(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def numeric_bytes(value):
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, np.ndarray):
        return value.nbytes
    if isinstance(value, dict):
        return sum(numeric_bytes(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(numeric_bytes(v) for v in value)
    return 0
