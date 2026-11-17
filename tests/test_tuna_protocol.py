"""TUNA protocol fidelity with the native twelve-block algorithm on tiny data.

Only model width/image size, fixture data, worker count and epoch counts are
reduced.  Native TUNA's training, orthogonal loss, EMR merge, Gaussian sampling,
classifier alignment and inference implementations execute unchanged.  These
tests establish implementation/lifecycle correctness, not benchmark accuracy.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from backbone.linears import TunaLinear
from backbone.vit_tuna import VisionTransformer
from cl27 import runner
from cl27.common import state_hash
from models import tuna as native
from models import tuna_protocol as protocol


class TinyTUNANet(nn.Module):
    """Use the real TUNA backbone/adapters/head without a pretrained download."""

    def __init__(self, args, pretrained):
        super().__init__()
        assert pretrained
        self._device = args["device"][0]
        config = SimpleNamespace(d_model=8, _device=self._device,
                                 ffn_adapt=True, ffn_option="parallel")
        # As with a loaded checkpoint, the original body is independent of the
        # current run's RNG.  Do not consume the algorithm's global RNG here.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(17)
            self.backbone = VisionTransformer(
                img_size=8, patch_size=4, embed_dim=8, depth=12,
                num_heads=2, mlp_ratio=2, num_classes=10,
                tuning_config=config,
            )
        self.backbone.out_dim = 8
        self.fc = None

    @property
    def feature_dim(self):
        return self.backbone.out_dim

    def update_fc(self, size):
        if self.fc is None:
            self.fc = TunaLinear(self.feature_dim, size)
        else:
            self.fc.update(size, freeze_old=False)


class Rows(Dataset):
    def __init__(self, manager, indices, source):
        self.manager = manager
        self.indices = np.asarray(indices, dtype=np.int64)
        self.labels = manager.y[self.indices]
        self.ids = self.indices + (10000 if source == "test" else 0)
        self.source = source

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        self.manager.reads[self.source] += 1
        row = int(self.indices[index])
        return index, self.manager.x[row], int(self.manager.y[row])


class Manager:
    nb_tasks = 2

    def __init__(self):
        # At least five classes at the first task are needed by native top-5.
        self.y = np.repeat(np.arange(10), 3)
        generator = torch.Generator().manual_seed(13)
        self.x = torch.randn(30, 3, 8, 8, generator=generator)
        self.reads = {"train": 0, "test": 0}

    def get_task_size(self, task):
        return 5

    def get_dataset(self, classes, source, mode, ret_data=False):
        rows = Rows(self, np.flatnonzero(np.isin(self.y, classes)), source)
        if ret_data:
            return self.x[rows.indices], self.y[rows.indices], rows
        return rows


def args_for_test():
    return {
        "model_name": "tuna", "device": [torch.device("cpu")],
        "backbone_type": "vit_base_patch16_224_tuna", "pretrained": True,
        "memory_size": 0, "memory_per_class": 0, "fixed_memory": False,
        "nb_classes": 10, "init_cls": 5, "increment": 5,
        "use_orth": False, "batch_size": 5, "init_lr": 0.01,
        "weight_decay": 0.0005, "min_lr": 1e-8, "ca_lr": 0.01,
        "tuned_epoch": 1, "crct_epochs": 1, "optimizer": "sgd",
        "scheduler": "cosine", "ca_storage_efficient_method": "covariance",
        "reg": 100, "scale": 20, "m": 0, "decay": False,
        "evaluation_head": "cnn", "seed": 1993,
    }


def same(first, second):
    if isinstance(first, torch.Tensor):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    elif isinstance(first, np.ndarray):
        np.testing.assert_array_equal(first, second)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            same(first[key], second[key])
    elif isinstance(first, (list, tuple)):
        assert len(first) == len(second)
        for one, two in zip(first, second):
            same(one, two)
    else:
        assert first == second


@pytest.fixture
def tiny_native(monkeypatch):
    args = args_for_test()
    body_hash = protocol.canonical_body_hash(TinyTUNANet(args, True).backbone)
    monkeypatch.setattr(protocol, "INITIAL_BODY_SHA256", body_hash)
    monkeypatch.setattr(native, "TUNANet", TinyTUNANet)
    # Only worker count is replaced, equally on native and wrapped paths.
    def loader(*positional, **keywords):
        keywords["num_workers"] = 0
        return DataLoader(*positional, **keywords)
    monkeypatch.setattr(native, "DataLoader", loader)
    monkeypatch.setattr(protocol, "verified_pretrained_cache", lambda _args: {
        "weight_file_sha256": protocol.WEIGHT_SHA256,
        "kind": "tiny_native_fixture_not_pretrained",
    })
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield args
    torch.set_num_threads(old_threads)


@pytest.mark.parametrize("orth,decay,storage", [
    (False, False, "covariance"),
    (True, True, "variance"),
])
def test_native_and_wrapped_two_tasks_have_identical_state_predictions_and_rng(
    tiny_native, orth, decay, storage,
):
    args = copy.deepcopy(tiny_native)
    args.update(use_orth=orth, decay=decay,
                ca_storage_efficient_method=storage)
    outcomes = []
    for learner_type in (native.Learner, protocol.Learner):
        torch.manual_seed(1993)
        np.random.seed(1993)
        manager = Manager()
        learner = learner_type(copy.deepcopy(args))
        stages = []
        for task in range(2):
            learner.incremental_train(manager)
            report, nme = learner.eval_task()
            if isinstance(learner, protocol.Learner):
                before_reads = manager.reads["test"]
                before_rng = torch.get_rng_state().clone()
                record = runner.stage_record(learner, report, nme, manager, task)
                assert record["extra_evaluation_pass_for_raw_matrix"] is False
                assert record["prediction_space"] == "all_seen_classes_without_task_id"
                assert record["evaluation_images"] == (task + 1) * 15
                assert record["task_counts"] == ([15, 0] if task == 0 else [15, 15])
                assert manager.reads["test"] == before_reads
                same(torch.get_rng_state(), before_rng)
                stages.append(copy.deepcopy(learner._evaluation_record))
            else:
                # Native eval_task already made exactly one evaluation pass.
                stages.append(report)
            learner.after_task()
        outcomes.append({
            "network": copy.deepcopy(learner._network.state_dict()),
            "means": copy.deepcopy(learner.cls_mean),
            "covariances": copy.deepcopy(learner.cls_cov),
            "cls2task": copy.deepcopy(learner.cls2task),
            "rng": torch.get_rng_state().clone(),
            "reads": copy.deepcopy(manager.reads),
            "reports": stages,
        })
        if isinstance(learner, protocol.Learner):
            assert learner.task_diagnostics[0]["training"]["main_optimizer_steps_observed"] == 3
            assert learner.task_diagnostics[1]["training"]["ca_first_head_backward_calls_observed"] == 10
            assert learner.task_diagnostics[1]["training"]["ca_scheduler_T_max_native"] == 5
            assert learner.task_diagnostics[1]["raw_replay_images"] == 0
            changed = learner.task_diagnostics[1]["historical_adapter_changed"]
            assert changed == [orth]
            # Registered historical adapters retain native gradient eligibility.
            assert all(p.requires_grad for p in learner._network.backbone.adapter_list.parameters())
    for key in ("network", "means", "covariances", "cls2task", "rng", "reads"):
        same(outcomes[0][key], outcomes[1][key])
    # Compare raw labels/top-1 against an explicit native eval using the saved
    # final weights, with RNG isolated so this extra assertion changes nothing.
    with torch.random.fork_rng(devices=[]):
        reference = native.Learner(copy.deepcopy(args))
        reference._network = copy.deepcopy(learner._network)
        reference._cur_task = 1
        reference._known_classes = reference._total_classes = 10
        p, y = reference._eval_cnn(learner.test_loader)
    assert outcomes[1]["reports"][-1] == {
        "predictions": p[:, 0].tolist(), "labels": y.tolist(),
    }


def test_task_boundary_restore_and_next_task_match_uninterrupted_native_rng(tiny_native):
    args = copy.deepcopy(tiny_native)
    args.update(use_orth=True, decay=True)
    torch.manual_seed(1993)
    manager = Manager()
    continuous = protocol.Learner(copy.deepcopy(args))
    continuous.incremental_train(manager)
    continuous.eval_task()
    continuous.after_task()
    saved = continuous.checkpoint_state()
    saved_rng = torch.get_rng_state().clone()
    saved_copy = copy.deepcopy(saved)
    continuous.incremental_train(manager)
    expected_reports = continuous.eval_task()
    expected_raw = copy.deepcopy(continuous._evaluation_record)
    continuous.after_task()
    expected_state = continuous.checkpoint_state()
    expected_rng = torch.get_rng_state().clone()

    torch.manual_seed(73)
    restored = protocol.Learner(copy.deepcopy(args))
    before_restore = torch.get_rng_state().clone()
    restored.restore_checkpoint(saved)
    same(torch.get_rng_state(), before_restore)
    same(restored.checkpoint_state(), saved_copy)
    assert restored.cls_mean[0].data_ptr() != saved["core"]["cls_mean"][0].data_ptr()
    assert restored.cls_cov[0].data_ptr() != saved["core"]["cls_cov"][0].data_ptr()
    # Runner owns the global RNG snapshot and restores it after model loading.
    torch.set_rng_state(saved_rng)
    restored.incremental_train(Manager())
    actual_reports = restored.eval_task()
    restored.after_task()
    same(actual_reports, expected_reports)
    same(restored.checkpoint_state(), expected_state)
    same(torch.get_rng_state(), expected_rng)
    same(saved, saved_copy)  # Checkpoint input was never modified or aliased.
    # Restoring after task two must also reconstruct the saved merged adapter.
    final = protocol.Learner(copy.deepcopy(args))
    before_restore = torch.get_rng_state().clone()
    final.restore_checkpoint(expected_state)
    same(torch.get_rng_state(), before_restore)
    same(final.checkpoint_state(), expected_state)
    assert len(final._network.backbone.merged_adapter) == 12
    final.test_loader = restored.test_loader
    final_reports = final.eval_task()
    same(final._evaluation_record, expected_raw)
    # Native grouped old/new reports use _known_classes.  A task-boundary
    # checkpoint is saved after after_task, so all classes are now "old";
    # those two administrative groups differ from the pre-boundary report.
    # The raw predictions and the task-independent accuracy metrics must not.
    same(final_reports[0]["top1"], expected_reports[0]["top1"])
    same(final_reports[0]["top5"], expected_reports[0]["top5"])
    assert final_reports[1] is expected_reports[1] is None


def test_invalid_restore_does_not_mutate_live_model_or_rng(tiny_native):
    torch.manual_seed(1993)
    model = protocol.Learner(copy.deepcopy(tiny_native))
    model.incremental_train(Manager())
    model.eval_task()
    model.after_task()
    valid = model.checkpoint_state()
    for field, replacement in (
        ("algorithm_rng_policy", "different"),
        ("fc_head_sizes", [10]),
        ("requires_grad", {}),
        ("module_training", {}),
        ("network_sha256", "incorrect"),
    ):
        invalid = copy.deepcopy(valid)
        invalid["core"][field] = replacement
        before_rng = torch.get_rng_state().clone()
        with pytest.raises(ValueError):
            model.restore_checkpoint(invalid)
        same(model.checkpoint_state(), valid)
        same(torch.get_rng_state(), before_rng)
    assert state_hash(model._network) == valid["core"]["network_sha256"]
    invalid = copy.deepcopy(valid)
    invalid["total_classes"] = float(valid["total_classes"])
    with pytest.raises(ValueError, match="completed task-boundary"):
        model.restore_checkpoint(invalid)
    same(model.checkpoint_state(), valid)


def test_weight_cache_rejects_missing_wrong_and_different_native_branch(monkeypatch, tmp_path):
    args = args_for_test()
    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path))
    constructor_calls = []
    monkeypatch.setattr(native, "TUNANet", lambda *args: constructor_calls.append(args))
    with pytest.raises(FileNotFoundError, match="downloads are disabled"):
        protocol.Learner(args)
    path = tmp_path / "checkpoints" / protocol.WEIGHT_FILENAME
    path.parent.mkdir()
    path.write_bytes(b"wrong local cached weight")
    with pytest.raises(ValueError, match="SHA256 differs"):
        protocol.Learner(args)
    assert constructor_calls == []
    with pytest.raises(ValueError, match="IN21K->IN1K construction branch"):
        protocol.verified_pretrained_cache(dict(args, backbone_type="vit_base_patch16_224_in21k_tuna"))
    monkeypatch.setattr(protocol, "file_hash", lambda _path: protocol.WEIGHT_SHA256)
    provenance = protocol.verified_pretrained_cache(args)
    assert provenance["weight_cache_file"] == str(path)
    assert provenance["algorithm_rng_policy"] == protocol.ALGORITHM_RNG_POLICY
    assert constructor_calls == []  # Cache checking does not construct a model.


def test_canonical_body_mapping_matches_standard_vit_names_and_bytes():
    from timm.models.vision_transformer import VisionTransformer as StandardViT
    torch.manual_seed(31)
    reference = StandardViT(img_size=8, patch_size=4, embed_dim=8,
                            depth=2, num_heads=2, mlp_ratio=2,
                            qkv_bias=True, num_classes=10)
    original = {n: v for n, v in reference.state_dict().items() if not n.startswith("head.")}
    native_body = {}
    for name, value in original.items():
        if ".attn.qkv." in name:
            for part, tensor in zip(("q", "k", "v"), value.chunk(3, dim=0)):
                native_body[name.replace("qkv", part + "_proj")] = tensor.clone()
        else:
            native_body[name.replace(".mlp.fc", ".fc")] = value.clone()
    native_body["cur_adapter.0.up_proj.weight"] = torch.zeros(8, 16)
    native_body["adapter_list.0.0.up_proj.weight"] = torch.zeros(8, 16)
    native_body["merged_adapter.0.up_proj.weight"] = torch.zeros(8, 16)
    native_body["head.weight"] = torch.zeros(10, 8)
    same(protocol.canonical_body_state(native_body), original)
    del native_body["blocks.0.attn.k_proj.weight"]
    with pytest.raises(ValueError, match="Incomplete native TUNA QKV"):
        protocol.canonical_body_state(native_body)
