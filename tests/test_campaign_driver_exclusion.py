import json

from scripts.summarize_campaign import inspect_run


def test_internal_ce_trajectory_driver_is_not_a_formal_er_baseline(tmp_path):
    job = tmp_path / "trajectory_driver"
    job.mkdir()
    (job / "config.json").write_text(json.dumps({
        "dataset": "cifar224", "model_name": "er", "seed": 1993,
        "phase": "final", "execution_role": "internal_shared_ce_driver",
    }))
    output = job / "results.json"
    output.write_text(json.dumps({"status": "complete", "phase": "final"}))
    record, reason = inspect_run(output, tmp_path)
    assert record is None
    assert "not the fixed ER baseline" in reason
