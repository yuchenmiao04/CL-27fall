"""Fixed-baseline entry-point policy, without training or formal data access."""

import copy
import json
from pathlib import Path

import pytest

from cl27.ce_trajectory import CEStageCache
from scripts import run_fixed_er_record as entry


ROOT = Path(__file__).resolve().parents[1]


def config(dataset="cifar224"):
    return json.loads((ROOT / "exps/pilot_figure2" / f"baseline_er_{dataset}.json").read_text())


def test_fixed_baseline_role_and_exact_parameters_reach_native_record_runner(tmp_path, monkeypatch):
    original = config()
    # The wrapper preserves supplied fixed values; it does not substitute its
    # own epochs, learning rates, batches, paths or numerical settings.
    original.update(epochs=17, backbone_lr=7e-6, head_lr=8e-4, batch_size=48,
                    replay_batch_size=32, data_root="unchanged/data/root",
                    output_dir="unchanged/output/root", device=[1], cpu_threads=3)
    path = tmp_path / "er.json"
    path.write_text(json.dumps(original))
    seen = {}

    def native(args, *, ce_training_executor):
        seen.update(args=copy.deepcopy(args), executor=ce_training_executor)
        assert isinstance(ce_training_executor, CEStageCache)
        assert ce_training_executor.mode == "record"
        assert ce_training_executor.directory == (tmp_path / "trajectory").resolve()
        binding = ce_training_executor.for_run(args, {"fingerprint": "fixture-data"})
        assert binding.report()["execution_role"] == "fixed_baseline"
        # Persist the exact arguments as the native runner does, so the role is
        # checked on disk rather than only at this call boundary.
        (tmp_path / "config.json").write_text(json.dumps(args))
        return {}

    monkeypatch.setattr(entry, "run", native)
    assert entry.main(["--config", str(path), "--trajectory-dir", str(tmp_path / "trajectory")]) == 0
    saved = json.loads((tmp_path / "config.json").read_text())
    assert saved["execution_role"] == "fixed_baseline"
    assert saved["validation_only"] is False and saved["seed"] == 1993
    assert saved["config"] == str(path.resolve())
    for key, value in original.items():
        if key != "seed":
            assert saved[key] == value


def test_development_internal_subset_or_other_methods_fail_before_runner(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("An invalid policy reached formal execution")

    monkeypatch.setattr(entry, "run", forbidden)
    for changes in (
        {"phase": "development"}, {"validation_only": True},
        {"execution_role": "internal_shared_ce_driver"}, {"max_tasks": 3},
        {"model_name": "history_transport"}, {"cl27_protocol": "other"},
        {"dataset": "synthetic", "smoke": True}, {"seed": [1994]},
    ):
        path = tmp_path / "invalid.json"
        path.write_text(json.dumps({**config(), **changes}))
        with pytest.raises(ValueError):
            entry.main(["--config", str(path), "--trajectory-dir", str(tmp_path / "never")])
    assert not (tmp_path / "never").exists()


def test_both_pilot_data_policies_are_checked_without_rewriting_them(tmp_path):
    for dataset in ("cifar224", "imagenetr"):
        path = tmp_path / (dataset + ".json")
        value = config(dataset)
        path.write_text(json.dumps(value))
        loaded = entry.load_fixed_config(path)
        for key in ("dataset", "fixed_memory", "memory_size", "memory_per_class", "epochs"):
            assert loaded[key] == value[key]
        invalid = dict(value, memory_size=1999) if dataset == "cifar224" else dict(value, fixed_memory=False)
        path.write_text(json.dumps(invalid))
        with pytest.raises(ValueError, match="CIFAR-100|ImageNet-R"):
            entry.load_fixed_config(path)
