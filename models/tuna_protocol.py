"""Protocol records around the unmodified PILOT TUNA learner.

The original optimizer, adapter updates/merging, classifier alignment and
inference branches remain in models.tuna and backbone.vit_tuna.  This wrapper
binds their existing IN21K->IN1K construction branch to an already verified
cache file, records one native evaluation, and saves task-boundary state.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import re

import numpy as np
import torch
from torch import nn

from cl27.common import UPSTREAM_COMMIT, file_hash, numeric_bytes, state_hash
from models import tuna as native


WEIGHT_SHA256 = "ac104c0df8158c46754510e50495536fa6de9647e572e38b5659dc54f260b124"
INITIAL_BODY_SHA256 = "1ece119a218036ffacbd3b576e1eb00a483915c88b21d81980c5e4e4a52e66e9"
WEIGHT_FILENAME = (
    "B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0"
    "--imagenet2012-steps_20k-lr_0.01-res_224.npz"
)
ALGORITHM_RNG_POLICY = "native_tuna_loaders_and_one_raw_eval_v1"
RECIPE_KEYS = (
    "backbone_type", "nb_classes", "init_cls", "increment", "tuned_epoch",
    "init_lr", "batch_size", "weight_decay", "min_lr", "optimizer",
    "scheduler", "reinit_optimizer", "init_milestones", "init_lr_decay",
    "reg", "use_orth", "crct_epochs", "ca_lr",
    "ca_storage_efficient_method", "ca_storage_efficient_method_choices",
    "decay", "pretrained", "drop", "drop_path", "r", "scale", "m",
)
PREFIX_STATE = (
    "down_weight_sum", "down_bias_sum", "up_weight_sum", "up_bias_sum",
)


def _need(ok, message):
    if not ok:
        raise ValueError(message)


def _cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {copy.deepcopy(k): _cpu_copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_cpu_copy(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_cpu_copy(v) for v in value)
    return copy.deepcopy(value)


def _state_hash(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def canonical_body_state(backbone_state):
    """Invert only native TUNA's byte-preserving QKV/MLP name conversion."""
    result = {}
    for name, value in backbone_state.items():
        if name.startswith(("head.", "head_dist.", "cur_adapter.",
                            "adapter_list.", "merged_adapter.")):
            continue
        match = re.fullmatch(r"(.+\.attn\.)q_proj\.(weight|bias)", name)
        if match:
            prefix, kind = match.groups()
            _need(all(prefix + part + "_proj." + kind in backbone_state for part in ("q", "k", "v")),
                  "Incomplete native TUNA QKV projection state")
            pieces = [backbone_state[prefix + part + "_proj." + kind]
                      for part in ("q", "k", "v")]
            converted = prefix + "qkv." + kind
            result[converted] = torch.cat(pieces, dim=0)
            continue
        if re.fullmatch(r".+\.attn\.[kv]_proj\.(weight|bias)", name):
            continue
        converted = re.sub(r"^(blocks\.\d+)\.(fc[12])\.", r"\1.mlp.\2.", name)
        _need(converted not in result, "Duplicate canonical TUNA body key")
        result[converted] = value
    _need(bool(result), "No original ViT state found")
    return result


def canonical_body_hash(backbone):
    return _state_hash(canonical_body_state(backbone.state_dict()))


def verified_pretrained_cache(args):
    """Fail before native construction if its exact local cache is unavailable.

    The unmodified native non-in21k TUNA builder subsequently resolves this same
    timm URL/cache basename.  No downloader, second model, or forward is used.
    """
    _need(args.get("backbone_type") == "vit_base_patch16_224_tuna",
          "Protocol TUNA requires its native IN21K->IN1K construction branch")
    _need(args.get("pretrained") is True and not args.get("smoke", False),
          "Protocol TUNA has no synthetic/pretrained=False construction path")
    import timm
    from timm.models.vision_transformer import default_cfgs

    _need(timm.__version__ == "0.6.12", "Protocol TUNA requires timm==0.6.12")
    url = default_cfgs["vit_base_patch16_224"]["url"]
    _need(url.rsplit("/", 1)[-1] == WEIGHT_FILENAME,
          "Native timm checkpoint URL differs from the pinned AugReg NPZ")
    path = Path(torch.hub.get_dir()) / "checkpoints" / WEIGHT_FILENAME
    if not path.is_file():
        raise FileNotFoundError("Pinned TUNA NPZ is not cached; downloads are disabled: " + str(path))
    _need(file_hash(path) == WEIGHT_SHA256, "Pinned TUNA NPZ SHA256 differs")
    return {
        "kind": "pretrained",
        "timm_version": timm.__version__,
        "checkpoint_family": "augreg_in21k_ft_in1k",
        "public_weight_url": url,
        "weight_file_sha256": WEIGHT_SHA256,
        "weight_cache_file": str(path),
        "upstream_commit": UPSTREAM_COMMIT,
        "algorithm_rng_policy": ALGORITHM_RNG_POLICY,
        "network_construction": "unmodified native vit_base_patch16_224_tuna",
    }


def native_source_hashes():
    root = Path(__file__).resolve().parents[1]
    return {
        name: file_hash(root / name)
        for name in ("models/tuna.py", "backbone/vit_tuna.py",
                     "utils/inc_net.py", "backbone/linears.py",
                     "models/tuna_protocol.py")
    }


class Learner(native.Learner):
    """Native TUNA training/inference plus the structured runner interface."""

    def __init__(self, args):
        _need(len(args.get("device", [])) == 1,
              "Protocol TUNA supports independent single-device runs only")
        _need(args.get("model_name") == "tuna", "Wrong protocol TUNA method")
        _need(args.get("memory_size", 0) == 0
              and args.get("memory_per_class", 0) == 0,
              "Native TUNA does not use an image replay buffer")
        provenance = verified_pretrained_cache(args)
        super().__init__(args)
        _need(canonical_body_hash(self._network.backbone) == INITIAL_BODY_SHA256,
              "Loaded TUNA original ViT body differs from the pinned initial model")
        provenance["initial_state_sha256"] = INITIAL_BODY_SHA256
        provenance["canonical_body_mapping"] = "concat native q/k/v; restore mlp.fc names; exclude adapters/head"
        self.backbone_provenance = provenance
        self.algorithm_rng_policy = ALGORITHM_RNG_POLICY
        self._recipe = {key: copy.deepcopy(args.get(key)) for key in RECIPE_KEYS}
        self._native_sources = native_source_hashes()
        self.task_diagnostics = []
        self._evaluation_record = None
        self._arrived_ids = []
        self._class_counts = {}
        self._in_task = False
        self._main_steps = 0
        self._ca_backward_calls = 0
        self._optimizer_names = []

    def get_optimizer(self, model):
        optimizer = super().get_optimizer(model)
        by_id = {id(p): name for name, p in self._network.named_parameters()}
        self._optimizer_names = [by_id[id(p)] for group in optimizer.param_groups
                                 for p in group["params"]]

        def observed_step(_optimizer, _args, _kwargs):
            self._main_steps += 1

        optimizer.register_step_post_hook(observed_step)
        return optimizer

    def classifer_align(self, model):
        # A no-op gradient observer counts native first-head backward calls.
        # It returns None and does not inspect/replace gradient values.
        parameter = next(self._network.fc.parameters())
        handle = None
        if parameter.requires_grad:
            def observed_backward(_gradient):
                self._ca_backward_calls += 1
            handle = parameter.register_hook(observed_backward)
        try:
            return super().classifer_align(model)
        finally:
            if handle is not None:
                handle.remove()

    def incremental_train(self, data_manager):
        _need(not self._in_task and self._known_classes == self._total_classes,
              "Complete after_task before beginning the next TUNA task")
        old_current = state_hash(self._network.backbone.cur_adapter)
        old_adapters = [state_hash(a) for a in self._network.backbone.adapter_list]
        self._main_steps = self._ca_backward_calls = 0
        self._optimizer_names = []
        self._evaluation_record = None
        self._in_task = True
        try:
            super().incremental_train(data_manager)
        finally:
            self._in_task = False
        current = self.train_dataset
        _need(hasattr(current, "ids"), "Protocol TUNA requires stable training IDs")
        ids = np.asarray(current.ids, dtype=np.int64).copy()
        labels = np.asarray(current.labels, dtype=np.int64)
        _need(ids.shape == labels.shape == (len(current),), "Invalid current TUNA rows")
        _need(len(np.unique(ids)) == len(ids), "Duplicate current TUNA training IDs")
        previous_ids = np.concatenate(self._arrived_ids) if self._arrived_ids else np.empty(0, np.int64)
        _need(not np.intersect1d(previous_ids, ids).size, "Training rows arrived in more than one TUNA task")
        self._arrived_ids.append(ids)
        for c in range(self._known_classes, self._total_classes):
            self._class_counts[c] = int(np.count_nonzero(labels == c))
        body_hash = canonical_body_hash(self._network.backbone)
        _need(body_hash == INITIAL_BODY_SHA256, "Original TUNA ViT weights changed")
        body_frozen = all(not p.requires_grad for name, p in self._network.backbone.named_parameters()
                          if "adapter" not in name and "head" not in name)
        _need(body_frozen, "An original TUNA ViT parameter became trainable")
        parameters = dict(self._network.named_parameters())
        eligible = set(self._optimizer_names)
        task_count = self._cur_task + 1
        self.task_diagnostics.append({
            "task": self._cur_task,
            "arrivals": len(ids),
            "arrived_training_rows": sum(map(len, self._arrived_ids)),
            "arrival_ids_sha256": hashlib.sha256(ids.tobytes()).hexdigest(),
            "class_counts": copy.deepcopy(self._class_counts),
            "raw_replay_images": self.exemplar_size,
            "original_vit_frozen": body_frozen,
            "original_vit_sha256": body_hash,
            "current_adapter_changed": old_current != state_hash(self._network.backbone.cur_adapter),
            "historical_adapter_changed": [before != state_hash(after)
                                           for before, after in zip(old_adapters, self._network.backbone.adapter_list)],
            "native_adapter_sets": {"historical": len(self._network.backbone.adapter_list),
                                    "current_layers": len(self._network.backbone.cur_adapter),
                                    "merged_layers": len(self._network.backbone.merged_adapter)},
            "algorithm_rng_policy": ALGORITHM_RNG_POLICY,
            "native_source_sha256": copy.deepcopy(self._native_sources),
            "training": {
                "image_epochs": int(self.args["tuned_epoch"]),
                "main_optimizer_steps_observed": self._main_steps,
                "class_alignment_epochs": int(self.crct_epochs) if self._cur_task else 0,
                "ca_first_head_backward_calls_observed": self._ca_backward_calls,
                "ca_scheduler_T_max_native": 5 if self._cur_task else None,
                "synthetic_features_per_class_per_ca_epoch": self.batch_size * 5 if self._cur_task else 0,
                "optimizer_parameter_names": list(self._optimizer_names),
            },
            "resources": {
                "gradient_eligible_nn_parameters": sum(parameters[n].numel() for n in eligible),
                "resident_nn_parameters": sum(p.numel() for p in parameters.values()),
                "network_parameter_bytes": sum(p.numel() * p.element_size() for p in parameters.values()),
                "historical_statistics_bytes": numeric_bytes((self.cls_mean, self.cls_cov)),
                "replay_raw_payload_bytes": 0,
                "training_backbones": 1,
                "inference_backbones": 1,
                "native_eval_backbone_invocations_per_batch": 1 if task_count == 1 else task_count + 1,
                "inference_invocation_count_scope": "Derived from unchanged native _eval_cnn branches, not a profiler measurement",
                "persisted_optimizer_bytes": 0,
            },
            "native_behaviors_preserved": {
                "historical_adapters_not_explicitly_frozen": True,
                "current_adapter_not_reset_after_task": True,
                "ca_schedule_period_is_5": True,
                "decay_task_size_is_known_minus_total": True,
                "second_ensemble_branch_uses_current_adapter": True,
            },
        })

    def eval_task(self):
        predictions, labels = self._eval_cnn(self.test_loader)
        self._evaluation_record = {
            "predictions": predictions[:, 0].astype(int).tolist(),
            "labels": labels.astype(int).tolist(),
        }
        return self._evaluate(predictions, labels), None

    def checkpoint_core(self):
        _need(not self._in_task and self._known_classes == self._total_classes,
              "TUNA supports completed task-boundary checkpoints only")
        network = self._network
        _need(network.fc is not None and self._cur_task >= 0, "No completed TUNA task")
        state = _cpu_copy(network.state_dict())
        return {
            "version": 1,
            "algorithm_rng_policy": ALGORITHM_RNG_POLICY,
            "recipe": copy.deepcopy(self._recipe),
            "native_source_sha256": copy.deepcopy(self._native_sources),
            "backbone_provenance": copy.deepcopy(self.backbone_provenance),
            "network": state,
            "network_sha256": _state_hash(state),
            "fc_head_sizes": [int(h[0].out_features) for h in network.fc.heads],
            "adapter_sets": len(network.backbone.adapter_list),
            "merged_layers": len(network.backbone.merged_adapter),
            "requires_grad": {n: bool(p.requires_grad) for n, p in network.named_parameters()},
            "module_training": {n: bool(m.training) for n, m in network.named_modules()},
            "cls_mean": _cpu_copy(self.cls_mean),
            "cls_cov": _cpu_copy(self.cls_cov),
            "cls2task": copy.deepcopy(self.cls2task),
            "native_prefix_state": {k: _cpu_copy(getattr(network.backbone, k)) for k in PREFIX_STATE},
            "arrived_ids": _cpu_copy(self._arrived_ids),
            "class_counts": copy.deepcopy(self._class_counts),
            "last_task_instrumentation": {"main_optimizer_steps": self._main_steps,
                                          "ca_backward_calls": self._ca_backward_calls,
                                          "optimizer_names": list(self._optimizer_names)},
            "optimizer_state_policy": "Native main and CA optimizers are rebuilt per task; no mid-task resume",
            "raw_replay_images": self.exemplar_size,
        }

    def checkpoint_state(self):
        return {
            "version": 1,
            "method": "tuna",
            "cur_task": self._cur_task,
            "known_classes": self._known_classes,
            "total_classes": self._total_classes,
            "task_diagnostics": copy.deepcopy(self.task_diagnostics),
            "core": self.checkpoint_core(),
        }

    def restore_checkpoint(self, state):
        _need(state.get("version") == 1 and state.get("method") == "tuna", "Wrong TUNA checkpoint version/method")
        task, known, total = (state[k] for k in ("cur_task", "known_classes", "total_classes"))
        _need(type(task) is int and task >= 0 and type(known) is int
              and type(total) is int and known == total and 0 < total <= self.args["nb_classes"],
              "TUNA supports completed task-boundary restore only")
        core = state["core"]
        _need(core.get("version") == 1 and core.get("algorithm_rng_policy") == ALGORITHM_RNG_POLICY,
              "TUNA native RNG/lifecycle policy changed")
        _need(core["recipe"] == self._recipe and core["native_source_sha256"] == self._native_sources,
              "TUNA recipe or executed source changed on restore")
        _need(core.get("raw_replay_images") == 0, "TUNA image replay state is not supported")
        _need(core["backbone_provenance"]["weight_file_sha256"] == self.backbone_provenance["weight_file_sha256"]
              and core["backbone_provenance"]["initial_state_sha256"] == INITIAL_BODY_SHA256,
              "TUNA pretrained identity changed")
        sizes = core["fc_head_sizes"]
        _need(len(sizes) == core["adapter_sets"] == task + 1
              and all(type(n) is int and n > 0 for n in sizes) and sum(sizes) == total,
              "TUNA dynamic classifier/adapter topology differs from task counters")
        _need(sizes == [int(self.args["init_cls"])] + [int(self.args["increment"])] * task,
              "TUNA saved class increments differ")
        layers = len(self._network.backbone.cur_adapter)
        _need(core["merged_layers"] == (0 if task == 0 else layers), "TUNA merged topology differs")
        expected_classes = set(range(total))
        _need(set(core["cls_mean"]) == set(core["cls_cov"]) == set(core["cls2task"])
              == set(core["class_counts"]) == expected_classes, "TUNA class statistics are incomplete")
        width = self._network.feature_dim
        covariance_shape = ((width, width) if self.args["ca_storage_efficient_method"] == "covariance" else (width,))
        for c in range(total):
            mean, cov = core["cls_mean"][c], core["cls_cov"][c]
            _need(isinstance(mean, torch.Tensor) and isinstance(cov, torch.Tensor)
                  and tuple(mean.shape) == (width,) and tuple(cov.shape) == covariance_shape
                  and torch.isfinite(mean).all().item() and torch.isfinite(cov).all().item(),
                  "Invalid saved TUNA Gaussian statistics")
            _need(type(core["class_counts"][c]) is int and core["class_counts"][c] > 0,
                  "Invalid TUNA training class count")
            boundary = np.cumsum(sizes)
            _need(core["cls2task"][c] == int(np.searchsorted(boundary, c, side="right")),
                  "Saved TUNA class-to-task map differs")
        ids = core["arrived_ids"]
        _need(len(ids) == task + 1 and all(isinstance(x, np.ndarray) and x.dtype == np.int64 and x.ndim == 1 for x in ids),
              "Invalid TUNA arrival ID state")
        joined = np.concatenate(ids)
        _need(len(np.unique(joined)) == len(joined) == sum(core["class_counts"].values()),
              "TUNA arrival IDs/counts disagree")
        weights = core["network"]
        _need(_state_hash(weights) == core["network_sha256"], "TUNA saved network digest differs")
        _need(all(isinstance(v, torch.Tensor) and torch.isfinite(v).all().item() for v in weights.values()),
              "Nonfinite saved TUNA weights")
        body_state = {k.removeprefix("backbone."): v for k, v in weights.items() if k.startswith("backbone.")}
        _need(_state_hash(canonical_body_state(body_state)) == INITIAL_BODY_SHA256,
              "Saved original TUNA ViT body changed")
        # Reconstruct on a candidate. A rejected restore cannot mutate live state.
        candidate = copy.deepcopy(self._network)
        candidate.backbone.adapter_list = nn.ModuleList([
            copy.deepcopy(candidate.backbone.cur_adapter) for _ in range(task + 1)
        ])
        candidate.backbone.merged_adapter = (
            copy.deepcopy(candidate.backbone.cur_adapter) if task else nn.ModuleList()
        )
        candidate.fc = None
        with torch.random.fork_rng(devices=[]):
            for size in sizes:
                candidate.update_fc(size)
        candidate.to(self._device)
        candidate.load_state_dict(weights, strict=True)
        parameters, modules = dict(candidate.named_parameters()), dict(candidate.named_modules())
        _need(set(core["requires_grad"]) == set(parameters)
              and all(type(v) is bool for v in core["requires_grad"].values()),
              "Saved TUNA parameter qualification differs")
        _need(set(core["module_training"]) == set(modules)
              and all(type(v) is bool for v in core["module_training"].values()),
              "Saved TUNA module modes differ")
        for name, parameter in parameters.items():
            parameter.requires_grad_(core["requires_grad"][name])
            if name.startswith("backbone.") and "adapter" not in name and "head" not in name:
                _need(not parameter.requires_grad, "Original ViT trainable in saved TUNA state")
        for name, module in modules.items():
            module.training = core["module_training"][name]
        _need(set(core["native_prefix_state"]) == set(PREFIX_STATE), "Missing native TUNA owned state")
        for name in PREFIX_STATE:
            setattr(candidate.backbone, name, _cpu_copy(core["native_prefix_state"][name]))
        _need(len(state["task_diagnostics"]) == task + 1, "TUNA task diagnostics/counters disagree")
        instrument = core["last_task_instrumentation"]
        _need(all(type(instrument[k]) is int and instrument[k] >= 0
                  for k in ("main_optimizer_steps", "ca_backward_calls")),
              "Invalid saved TUNA observation counters")
        _need(len(instrument["optimizer_names"]) == len(set(instrument["optimizer_names"]))
              and set(instrument["optimizer_names"]) <= set(parameters),
              "Saved TUNA optimizer parameter qualification differs")
        # All validation succeeded; commit the restored candidate.
        self._network = candidate
        # Tensor.to(cpu) may return its input unchanged.  Own these statistics
        # even on CPU, so a caller retaining the checkpoint cannot mutate the
        # resumed learner (or vice versa) through shared tensor storage.
        self.cls_mean = {c: v.to(self._device).clone() for c, v in core["cls_mean"].items()}
        self.cls_cov = {c: v.to(self._device).clone() for c, v in core["cls_cov"].items()}
        self.cls2task = copy.deepcopy(core["cls2task"])
        self._class_counts = copy.deepcopy(core["class_counts"])
        self._arrived_ids = _cpu_copy(ids)
        self._cur_task, self._known_classes, self._total_classes = task, known, total
        self.task_diagnostics = copy.deepcopy(state["task_diagnostics"])
        self._main_steps = int(instrument["main_optimizer_steps"])
        self._ca_backward_calls = int(instrument["ca_backward_calls"])
        self._optimizer_names = list(instrument["optimizer_names"])
        self._in_task = False
        self._evaluation_record = None
