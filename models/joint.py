"""Offline all-class CE reference with exactly one fit and final evaluation.

The PILOT class order still defines output columns, but all selected training
classes are available together. This is a different data-access reference from
continual learning, not a theorem establishing an accuracy upper bound.
"""

from __future__ import annotations

import copy
import hashlib
import operator

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from cl27.common import IntegratedLearner, build_backbone, numeric_bytes, state_hash
from cl27.replay import ReplayMemory
from cl27.supervised import CEClassifier
from cl27.transport_training import adapt_backbone, expand_head


def positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(name + " must be a positive integer")
    try:
        result = operator.index(value)
    except TypeError as error:
        raise ValueError(name + " must be a positive integer") from error
    if result < 1:
        raise ValueError(name + " must be a positive integer")
    return int(result)


def cpu_state(module):
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def check_module_state(module, state, name):
    expected = module.state_dict()
    if set(state) != set(expected) or any(
        not isinstance(state[k], torch.Tensor)
        or state[k].shape != value.shape
        or not torch.isfinite(state[k]).all()
        for k, value in expected.items()
    ):
        raise ValueError("Invalid " + name + " checkpoint tensors")


class Learner(IntegratedLearner):
    offline_joint = True

    def __init__(self, args):
        super().__init__(args)
        if (
            int(args.get("memory_size", 0)) != 0
            or int(args.get("memory_per_class", 0)) != 0
        ):
            raise ValueError(
                "Offline Joint has no replay memory; both budgets must be zero"
            )
        if args.get("max_tasks") is not None:
            raise ValueError(
                "Use explicit validation-only joint_max_classes, not max_tasks"
            )
        self.backbone = build_backbone(args).to(self._device)
        self.backbone_provenance = copy.deepcopy(
            getattr(self.backbone, "cl27_provenance", None)
        )
        self._initial_backbone_hash = state_hash(self.backbone)
        self.head = None
        self._network = CEClassifier(self.backbone, self.head)
        self._fit_started = False
        self._fit_complete = False
        self.selection = None

    @property
    def exemplar_size(self):
        return 0

    def _selected_prefix(self, data_manager):
        widths = [
            positive_integer(data_manager.get_task_size(t), "protocol task width")
            for t in range(data_manager.nb_tasks)
        ]
        available = positive_integer(data_manager.nb_classes, "protocol class count")
        if sum(widths) != available:
            raise ValueError("Protocol task sizes disagree with class count")
        limit = self.args.get("joint_max_classes")
        if limit is not None:
            if not self.args.get("validation_only", False) or self.args.get(
                "phase"
            ) not in {"development", "smoke"}:
                raise ValueError(
                    "joint_max_classes is permitted only in explicit development/smoke validation"
                )
            selected = positive_integer(limit, "joint_max_classes")
        else:
            selected = available
        endings = np.cumsum(widths).tolist()
        if selected not in endings:
            raise ValueError(
                "Joint class selection must end on a protocol task boundary"
            )
        return {
            "selected_classes": selected,
            "protocol_endpoint_task": endings.index(selected),
            "protocol_task_sizes": widths,
            "protocol_total_classes": available,
        }

    def incremental_train(self, data_manager):
        if self._fit_started or self._fit_complete:
            raise RuntimeError("Offline Joint permits exactly one fresh all-class fit")
        selection = self._selected_prefix(data_manager)
        classes = selection["selected_classes"]
        train = data_manager.get_dataset(
            np.arange(classes), source="train", mode="train"
        )
        labels = np.asarray(train.labels)
        ids = np.asarray(getattr(train, "ids", None))
        if (
            labels.ndim != 1
            or labels.dtype.kind not in "iu"
            or sorted(set(map(int, labels))) != list(range(classes))
        ):
            raise ValueError(
                "Joint training must contain every selected remapped class"
            )
        if (
            ids.shape != (len(train),)
            or ids.dtype.kind not in "iu"
            or len(np.unique(ids)) != len(ids)
        ):
            raise ValueError("Joint training requires unique stable integer source IDs")
        ids = ids.astype(np.int64, copy=True)
        selection.update(
            {
                "unique_training_images": len(train),
                "fitting_ids_sha256": hashlib.sha256(ids.tobytes()).hexdigest(),
            }
        )
        before = state_hash(self.backbone)
        self._fit_started = True
        self.head = expand_head(None, self.backbone.out_dim, classes, self._device)
        self._network = CEClassifier(self.backbone, self.head)
        # Reuse exactly the plastic baseline's CE optimizer, LR groups and
        # cosine loop. The empty object contains no old images and is never
        # updated; every selected class is ordinary current offline data.
        empty = ReplayMemory(dict(self.args, memory_size=0, memory_per_class=0))
        settings = dict(self.args, replay_batch_size=0)
        training = adapt_backbone(
            self.backbone,
            self.head,
            train,
            empty,
            settings,
            self._device,
            np.random.default_rng(int(self.args.get("seed", 0)) + 113),
        )
        if any(not torch.isfinite(p).all() for p in self._network.parameters()):
            raise FloatingPointError("Nonfinite final Joint parameter")
        # Final-only access: no evaluation image or loader is requested during
        # optimization, model selection or an intermediate epoch.
        test = data_manager.get_dataset(np.arange(classes), source="test", mode="test")
        test_ids = np.asarray(getattr(test, "ids", None))
        if test_ids.shape != (len(test),) or test_ids.dtype.kind not in "iu":
            raise ValueError("Joint evaluation requires stable integer source IDs")
        if set(map(int, ids)) & set(map(int, test_ids)):
            raise ValueError("Joint fitting/evaluation source IDs overlap")
        self._cur_task = selection["protocol_endpoint_task"]
        self._total_classes = classes
        self.topk = min(5, classes)
        self.test_loader = DataLoader(
            test,
            batch_size=int(self.args.get("eval_batch_size", 64)),
            shuffle=False,
            num_workers=int(self.args.get("num_workers", 0)),
        )
        self.selection = selection
        self._fit_complete = True
        after = state_hash(self.backbone)
        self.task_diagnostics.append(
            {
                "method": "joint",
                "evaluation_mode": "offline_joint_final",
                **copy.deepcopy(selection),
                "offline_fit_count": 1,
                "training": training,
                "backbone_changed": before != after,
                "backbone_sha256_before": before,
                "backbone_sha256_after": after,
                "raw_replay_images": 0,
                "replay_images": 0,
                "historical_statistics_bytes": 0,
                "data_access": "all selected training classes jointly available before a single offline fit",
                "evaluation_during_training": False,
                "aia_applicable": False,
                "forgetting_applicable": False,
                "resources": self.resource_report(),
            }
        )

    def after_task(self):
        if not self._fit_complete:
            raise RuntimeError("Offline Joint fit has not completed")
        super().after_task()

    def eval_task(self):
        if not self._fit_complete:
            raise RuntimeError("Evaluate Offline Joint only after its full fit")
        if not hasattr(self, "test_loader"):
            raise RuntimeError(
                "A restored completed Joint checkpoint has no new evaluation loader"
            )
        return super().eval_task()

    def resource_report(self):
        return {
            "deployment_backbone_parameter_bytes": sum(
                p.numel() * p.element_size() for p in self.backbone.parameters()
            ),
            "deployment_ce_head_bytes": (
                numeric_bytes(self.head.state_dict()) if self.head is not None else 0
            ),
            "inference_backbones": 1,
            "training_backbones": 1,
            "replay_raw_payload_bytes": 0,
            "historical_statistics_bytes": 0,
            "offline_training_union_access": True,
        }

    def checkpoint_core(self):
        if not self._fit_complete:
            raise RuntimeError("Joint checkpoints are completed-fit only")
        return {
            "version": 1,
            "kind": "offline_joint",
            "fit_complete": True,
            "selection": copy.deepcopy(self.selection),
            "initial_backbone_hash": self._initial_backbone_hash,
            "backbone_provenance": copy.deepcopy(self.backbone_provenance),
            "backbone": cpu_state(self.backbone),
            "head": cpu_state(self.head),
            "resume_policy": "completed offline fit only; no optimizer or mid-epoch continuation",
        }

    def restore_core(self, state):
        if (
            state.get("version") != 1
            or state.get("kind") != "offline_joint"
            or state.get("fit_complete") is not True
        ):
            raise ValueError("Invalid completed Joint checkpoint")
        if (
            state.get("initial_backbone_hash") != self._initial_backbone_hash
            or state.get("backbone_provenance") != self.backbone_provenance
        ):
            raise ValueError("Joint checkpoint pretrained source differs")
        selected = copy.deepcopy(state["selection"])
        classes = positive_integer(
            selected["selected_classes"], "checkpoint selected classes"
        )
        widths = [
            positive_integer(n, "checkpoint protocol width")
            for n in selected["protocol_task_sizes"]
        ]
        if isinstance(selected["protocol_endpoint_task"], (bool, np.bool_)):
            raise ValueError("Invalid Joint checkpoint task endpoint")
        endpoint = operator.index(selected["protocol_endpoint_task"])
        if (
            not 0 <= endpoint < len(widths)
            or sum(widths[: endpoint + 1]) != classes
            or sum(widths)
            != positive_integer(
                selected["protocol_total_classes"], "checkpoint protocol classes"
            )
        ):
            raise ValueError("Joint checkpoint class prefix and task endpoint disagree")
        if (
            positive_integer(
                selected["unique_training_images"], "checkpoint fitting images"
            )
            < classes
        ):
            raise ValueError(
                "Joint checkpoint fitting count cannot cover its vocabulary"
            )
        if len(selected.get("fitting_ids_sha256", "")) != 64:
            raise ValueError("Missing Joint training source hash")
        if (
            self.args.get("joint_max_classes") is not None
            and self.args["joint_max_classes"] != classes
        ):
            raise ValueError("Joint checkpoint selected class count differs")
        candidate = copy.deepcopy(self.backbone)
        check_module_state(candidate, state["backbone"], "Joint backbone")
        candidate.load_state_dict(state["backbone"], strict=True)
        with torch.random.fork_rng(devices=[]):
            head = nn.Linear(self.backbone.out_dim, classes).to(self._device)
        check_module_state(head, state["head"], "Joint head")
        head.load_state_dict(state["head"], strict=True)
        self.backbone = candidate.eval()
        self.head = head.eval()
        self._network = CEClassifier(self.backbone, self.head)
        self.selection = selected
        self._fit_started = self._fit_complete = True
        if hasattr(self, "test_loader"):
            del self.test_loader
        self._evaluation_record = None

    def restore_checkpoint(self, state):
        core = state.get("core", {})
        selection = core.get("selection", {})
        if (
            state.get("known_classes") != selection.get("selected_classes")
            or state.get("total_classes") != selection.get("selected_classes")
            or state.get("cur_task") != selection.get("protocol_endpoint_task")
            or len(state.get("task_diagnostics", [])) != 1
        ):
            raise ValueError(
                "Joint outer checkpoint must describe one complete offline fit"
            )
        super().restore_checkpoint(state)
