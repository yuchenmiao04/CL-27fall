"""The bounded gate must compare matched updates and fail without enabling."""

import pytest
import torch

from scripts.probe_current_prefix import (
    TOLERANCES,
    compare_tensors,
    parser,
    run_probe,
    validate_options,
)


@pytest.fixture(autouse=True)
def restore_thread_count():
    previous = torch.get_num_threads()
    yield
    torch.set_num_threads(previous)


def options(tmp_path, *extra):
    return parser().parse_args(
        [
            "--tiny",
            "--device",
            "-1",
            "--classes",
            "3",
            "--samples-per-class",
            "4",
            "--support-shots",
            "1",
            "2",
            "--background-shots",
            "1",
            "--query-per-class",
            "1",
            "--warmup",
            "1",
            "--steps",
            "1",
            "--cpu-threads",
            "1",
            "--output-json",
            str(tmp_path / "unused.json"),
            *extra,
        ]
    )


def test_tiny_controller_checks_two_steps_views_gradients_and_counters(tmp_path):
    result = run_probe(options(tmp_path))
    assert result["status"] == "complete"
    assert result["correctness_passed"] and result["resource_limits_passed"]
    assert result["genuine_pretrained_encoder"] is False
    assert result["enable_recommended"] is False and result["speed_passed"] is None
    assert result["prefix_replay_calls"] > 0
    assert result["frozen_main_versions_unchanged"]
    assert len(result["correctness_steps"]) == 2
    assert [row["kind"] for row in result["correctness_steps"]] == [
        "transfer",
        "anchored",
    ]
    assert all(row["identical_augmented_views"] for row in result["correctness_steps"])
    assert all(
        row["comparison"]["gradient"]["passed"] for row in result["correctness_steps"]
    )
    assert all(
        row["comparison"]["parameter"]["passed"] for row in result["correctness_steps"]
    )
    assert result["final_comparison"]["passed"]
    assert result["optimizer_updates"] == 4


def test_a_failed_runtime_gate_does_not_recommend_enabling(tmp_path):
    result = run_probe(options(tmp_path, "--max-seconds", ".000001"))
    assert result["status"] == "failed"
    assert result["enable_recommended"] is False
    assert "Runtime budget" in result["error"]


def test_difference_checks_reject_wrong_values_and_gradient_presence():
    a = {"weight": torch.tensor([1.0])}
    b = {"weight": torch.tensor([1.1])}
    result = compare_tensors(a, b, TOLERANCES["parameter"])
    assert result["passed"] is False and result["max_tolerance_ratio"] > 1
    with pytest.raises(RuntimeError, match="presence"):
        compare_tensors({"weight": None}, b, TOLERANCES["gradient"])


def test_oversized_microbatch_is_rejected_before_model_loading(tmp_path):
    with pytest.raises(ValueError, match="microbatch"):
        validate_options(options(tmp_path, "--microbatch", "17"))


def test_production_microbatch_is_allowed_without_changing_the_default(tmp_path):
    assert options(tmp_path).microbatch == 2
    production = options(tmp_path, "--microbatch", "16")
    validate_options(production)
    assert production.microbatch == 16
