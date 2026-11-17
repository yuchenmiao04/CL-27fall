"""CPU-only tests for the strict checkpoint equivalence gate.

These are small trusted serialization fixtures, not training/evaluation results
or evidence that two physical GPUs produce equal features. CUDA initialization
is prohibited within each test. Unknown metadata and nested time fields remain
part of the comparison; only root stages[index].wall_seconds may vary.
"""

from __future__ import annotations

import copy
import hashlib
import json

import numpy as np
import pytest
import torch

from cl27.replay import RawRecord
from scripts.compare_checkpoint_state import main


CASES = (
    "same",
    "only_stage_time",
    "missing_stage_time",
    "python_signed_zero",
    "tensor_one_bit",
    "object_array_dataclass_pixel",
    "nested_object_array",
    "stage_other_field",
    "nested_stage_time_not_ignored",
    "other_metadata_time_not_ignored",
    "cuda_rng_changed",
    "data_identity_changed",
    "dataclass_dynamic_attribute",
)


@pytest.fixture(autouse=True)
def prohibit_cuda_initialization(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("A CPU checkpoint comparison must not initialize CUDA")

    monkeypatch.setattr(torch.cuda, "_lazy_init", fail)


@pytest.fixture
def reference_checkpoint():
    objects = np.empty(2, dtype=object)
    objects[0] = RawRecord(np.arange(12, dtype=np.uint8).reshape(2, 2, 3), 3, 17)
    objects[1] = {
        "nested": np.array(["bird", "plane"], dtype=object),
        "scalar": np.float64(-0.0),
    }
    return {
        "version": 1,
        "random_stream_policy": "canonical_loader_isolated_v1",
        "config_fingerprint": "fixed-config",
        "data_fingerprint": "fixed-data",
        "learner": {
            "core": {
                "weight": torch.tensor([1.0, -0.0], dtype=torch.float32),
                "scalar_weight": torch.tensor(2.0),
                "float": 0.0,
                "objects": objects,
                "records": [objects[0]],
            }
        },
        "rng": {
            "python": (3, (1, 2, 3), None),
            "numpy": ("MT19937", np.array([17, 23], dtype=np.uint32), 0, 0, 0.0),
            "torch": torch.tensor([1, 3, 4], dtype=torch.uint8),
            # Serialized RNG payloads are CPU byte tensors; no CUDA is needed.
            "cuda": [
                torch.tensor([7, 8], dtype=torch.uint8),
                torch.tensor([9, 10], dtype=torch.uint8),
            ],
        },
        "stages": [
            {
                "task": 0,
                "wall_seconds": 1.5,
                "accuracy": 0.5,
                "metadata": {"wall_seconds": 4.0},
            }
        ],
        "extra_metadata": {"wall_seconds": 8.0, "unrecognized_but_checked": True},
    }


def change_candidate(candidate, case):
    """Return expected pass status and, on rejection, the changed state path."""
    core = candidate["learner"]["core"]
    if case == "same":
        return True, None
    if case == "only_stage_time":
        candidate["stages"][0]["wall_seconds"] = 99.0
        return True, None
    if case == "missing_stage_time":
        del candidate["stages"][0]["wall_seconds"]
        return True, None
    if case == "python_signed_zero":
        core["float"] = -0.0
        return False, "$.learner.core.float"
    if case == "tensor_one_bit":
        core["weight"].view(torch.uint8)[0] ^= 1
        return False, "$.learner.core.weight"
    if case == "object_array_dataclass_pixel":
        core["objects"][0].image[0, 0, 0] ^= 1
        return False, "$.learner.core.objects[0].image"
    if case == "nested_object_array":
        core["objects"][1]["nested"][1] = "changed"
        return False, "$.learner.core.objects[1].nested[1]"
    if case == "stage_other_field":
        candidate["stages"][0]["accuracy"] = 0.75
        return False, "$.stages[0].accuracy"
    if case == "nested_stage_time_not_ignored":
        candidate["stages"][0]["metadata"]["wall_seconds"] = 6.0
        return False, "$.stages[0].metadata.wall_seconds"
    if case == "other_metadata_time_not_ignored":
        candidate["extra_metadata"]["wall_seconds"] = 12.0
        return False, "$.extra_metadata.wall_seconds"
    if case == "cuda_rng_changed":
        candidate["rng"]["cuda"][1][0] ^= 1
        return False, "$.rng.cuda[1]"
    if case == "data_identity_changed":
        candidate["data_fingerprint"] = "different-data"
        return False, "$.data_fingerprint"
    if case == "dataclass_dynamic_attribute":
        core["records"][0].extra_attribute = "new state"
        return False, "$.learner.core.records[0].extra_attribute"
    raise AssertionError("Unknown case: " + case)


@pytest.mark.parametrize("case", CASES)
def test_strict_state_comparison_cli(tmp_path, reference_checkpoint, case):
    candidate = copy.deepcopy(reference_checkpoint)
    expected_pass, changed_path = change_candidate(candidate, case)
    reference_path = tmp_path / "reference.pt"
    candidate_path = tmp_path / "candidate.pt"
    report_path = tmp_path / "comparison.json"
    torch.save(reference_checkpoint, reference_path)
    torch.save(candidate, candidate_path)

    code = main([
        "--reference", str(reference_path),
        "--candidate", str(candidate_path),
        "--report", str(report_path),
    ])
    report = json.loads(report_path.read_text())
    assert "error" not in report
    assert report["passed"] is expected_pass
    assert code == (0 if expected_pass else 1)
    assert report["reference_file_sha256"] == hashlib.sha256(reference_path.read_bytes()).hexdigest()
    assert report["candidate_file_sha256"] == hashlib.sha256(candidate_path.read_bytes()).hexdigest()
    assert report["compared_nodes"] > 0
    assert report["difference_count"] == len(report["differences"])
    assert report["ignored_paths"] == ["$.stages[0].wall_seconds"]
    assert report["ignored_path_count"] == 1
    if expected_pass:
        assert report["difference_count"] == 0
        assert report["reference_state_sha256"] == report["candidate_state_sha256"]
    else:
        assert report["difference_count"] > 0
        assert report["reference_state_sha256"] != report["candidate_state_sha256"]
        assert any(item["path"] == changed_path for item in report["differences"])
    if case == "python_signed_zero":
        assert report["differences"][0]["reference_bits_hex"] == "0000000000000000"
        assert report["differences"][0]["candidate_bits_hex"] == "8000000000000000"
    if case == "tensor_one_bit":
        assert report["difference_count"] == 1
        assert report["differences"][0]["different_bytes"] == 1
