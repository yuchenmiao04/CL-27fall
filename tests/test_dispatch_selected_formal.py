"""Selected-job dispatch guards with fake processes/NVML; no CUDA or training."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import types

import pytest

from scripts import dispatch_selected_formal as dispatch


JOBS = (
    "formal_cifar_transport_selected_v2", "formal_cifar_plastic_shared_selected_v5",
    "formal_imagenetr_transport_selected_v3", "formal_imagenetr_plastic_shared_selected_v5",
    "formal_imagenetr_current_selected_v5",
)
EXTERNALS = (
    "formal_cifar_er_fixed_v1", "formal_imagenetr_er_fixed_v1",
    "cifar_current_bounded64_full_v4", "cifar_current_mb2_full_v5",
)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def pinned(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def files(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


class Fixture:
    def __init__(self, root, jobs=JOBS):
        self.root = root
        root.mkdir()
        self.probes, self.calls = {}, []
        self.now = 0.0
        self.handshake_missing = False
        self.snapshot = {gpu: dict(total_mib=32607, used_mib=0, free_mib=32000, processes={})
                         for gpu in (0, 1)}
        launcher = root / "launch_jobs.py"
        launcher.write_text("# pinned launcher fixture, never executed\n")
        self.cache = {name: root / "cache" / name for name in EXTERNALS[:2]}
        external = []
        for index, name in enumerate(EXTERNALS):
            gpu = (1, 0, 1, 0)[index]
            receipt = {"job": name, "pid": 100 + index, "proc_start_ticks": str(1000 + index),
                       "command": [str(root / "python"), "external.py", "--job", name],
                       "cwd": str(root), "metadata": {"gpu": gpu}}
            path = root / "external" / (name + ".json")
            save(path, receipt)
            result_path = root / "external" / name / "results.json"
            result = self.result("er" if index < 2 else "current_adapted")
            if index < 2:
                directory = self.cache[name]
                directory.mkdir(parents=True)
                stages = []
                for task in range(10):
                    stage = directory / f"stage_{task:03d}.pt"
                    stage.write_bytes(b"fake immutable atomic cache checkpoint")
                    stages.append(dict(task=task, path=str(stage)))
                result["execution"] = {"ce_training_reuse": dict(mode="record", execution_role="fixed_baseline",
                    trajectory_directory=str(directory), stages=stages)}
            save(result_path, result)
            saved_config = result_path.parent / "config.json"
            save(saved_config, {"device": [gpu]})
            external.append(dict(job=name, gpu=gpu, receipt=pinned(path), config=pinned(saved_config), result=str(result_path)))
        self.value = dict(version=1, runtime_root=str(root), python=str(root / "python"),
                          launcher=pinned(launcher), state_dir=str(root / "dispatch-state"), externals=external, jobs=[])
        for name in jobs:
            self.add_job(name)
        self.path = root / "queue.json"
        self.write_queue()

    @staticmethod
    def result(method):
        return dict(status="complete", phase="final", method=method,
                    metrics={"completed_tasks": 10}, stages=[{"task": task} for task in range(10)])

    def add_job(self, name):
        module = name == JOBS[4]
        imagenetr = "imagenetr" in name
        gpu = 1 if module or not imagenetr else 0
        method = "current_adapted" if module else "plastic_shared_basis" if "plastic" in name else "history_transport"
        formal = self.root / "campaign_20261117" / "formal"
        config = formal / "configs" / (name + ".json")
        width = 20 if imagenetr else 10
        save(config, dict(model_name=method, dataset="imagenetr" if imagenetr else "cifar224", phase="final",
                          cl27_protocol="pilot_figure2", seed=[1993], device=[gpu], validation_only=False,
                          shuffle=True, backbone_type="vit_base_patch16_224", pretrained=True,
                          init_cls=width, increment=width, fixed_memory=imagenetr,
                          memory_size=width * 200, memory_per_class=20, replay_selection="herding",
                          epochs=5 if module else 20, outer_updates=4,
                          schedule=["modules", "modules", "backbone", "modules"],
                          output_dir=str(formal / "runs" / name)))
        predecessor = EXTERNALS[0] if module or not imagenetr else EXTERNALS[1]
        command = [str(self.root / "python"), "main.py", "--config", str(config)] if module else [
            str(self.root / "python"), "scripts/run_ce_cohort.py", "--replay-only", "--configs", str(config),
            "--trajectory-dir", str(self.cache[predecessor]), "--report",
            str(self.root / "campaign_20261117" / "formal" / "cohorts" / (name + ".json"))]
        spec = self.root / "specs" / (name + ".json")
        cwd = self.root / ("runtime_bounded_v4" if module else "runtime_tuning_v1")
        cwd.mkdir(exist_ok=True)
        metadata = {"gpu": gpu}
        if module:
            metadata["selected_development_trial"] = "imagenetr_current_bounded64_full_v5"
        save(spec, {"jobs": [dict(name=name, area="formal", cwd=str(cwd), command=command, metadata=metadata)]})
        entry = dict(id=name, spec=pinned(spec), config=pinned(config), predecessor=predecessor)
        if module:
            decision = self.root / "D0054.json"
            save(decision, dict(id="D0054", hpo_closed=True, method=method, dataset="imagenetr",
                formal_refit={"job": name, "preferred_device": [1]},
                launch_constraints={"required_predecessor": {"job": EXTERNALS[0]}},
                selected_parameters={"epochs": 5, "outer_updates": 4,
                    "schedule": ["modules", "modules", "backbone", "modules"]},
                selected_development_trial=metadata["selected_development_trial"]))
            entry["decision"] = pinned(decision)
        else:
            entry["cache_dir"] = str(self.cache[predecessor])
        self.value["jobs"].append(entry)

    def write_queue(self):
        save(self.path, self.value)

    def dispatcher(self, **kwargs):
        options = dict(probe=self.probes.get, memory=lambda: copy.deepcopy(self.snapshot), spawn=self.spawn,
                       monotonic=lambda: self.now, sleep=self.sleep, launch_timeout=.2)
        return dispatch.Dispatcher(self.path, **{**options, **kwargs})

    def sleep(self, seconds):
        self.now += seconds

    def spawn(self, argv, **kwargs):
        self.calls.append((list(argv), kwargs))
        spec = json.loads(Path(argv[-1]).read_text())["jobs"][0]
        pid = 7000 + len(self.calls)
        receipt = dict(job=spec["name"], pid=pid, proc_start_ticks=str(pid * 10), command=spec["command"],
                       cwd=spec["cwd"], metadata=spec["metadata"])
        save(self.receipt(spec["name"]), receipt)
        if not self.handshake_missing:
            self.probes[pid] = dict(state="R", startticks=pid * 10, cmd=spec["command"], exit_code=None)
        class Child:
            pid = 9999  # Wrapper identity must never become the training PID.
            def poll(self):
                return 0
        return Child()

    def receipt(self, name):
        return self.root / "campaign_20261117" / "formal" / "receipts" / (name + ".json")

    def live_external(self, name, *, used=0, state="R", ticks=None, exit_code=None):
        entry = next(v for v in self.value["externals"] if v["job"] == name)
        receipt = json.loads(Path(entry["receipt"]["path"]).read_text())
        self.probes[receipt["pid"]] = dict(state=state, startticks=ticks or int(receipt["proc_start_ticks"]),
                                            cmd=receipt["command"], exit_code=exit_code)
        if used:
            self.snapshot[entry["gpu"]]["processes"][receipt["pid"]] = used
        return entry, receipt


@pytest.fixture
def f(tmp_path, monkeypatch):
    fixture = Fixture(tmp_path / "runtime")
    # Production approval hashes stay untouched. Only CPU fixture files receive
    # a temporary approval table, fixed before any mutation under test.
    monkeypatch.setattr(dispatch, "APPROVED_CONFIG", {
        item["id"]: item["config"]["sha256"] for item in fixture.value["jobs"]
    })
    return fixture


def test_plan_is_readonly_and_never_spawns(f):
    before = files(f.root)
    plan = f.dispatcher().plan()
    assert set(plan) == set(JOBS)
    assert set(plan.values()) == {"ready"}
    assert files(f.root) == before
    assert not f.calls


@pytest.mark.parametrize("mode", [[], ["--plan"]])
def test_cli_default_and_plan_mode_never_enter_execute(f, monkeypatch, capsys, mode):
    d = f.dispatcher()
    monkeypatch.setattr(dispatch, "Dispatcher", lambda path: d)
    before = files(f.root)
    dispatch.main(["--queue", str(f.path), *mode])
    assert set(json.loads(capsys.readouterr().out).values()) == {"ready"}
    assert files(f.root) == before
    assert not f.calls


@pytest.mark.parametrize("change", ["launcher", "spec", "config", "unknown_job"])
def test_unpinned_or_non_whitelisted_input_fails_closed(f, change):
    if change == "unknown_job":
        f.value["jobs"][0]["id"] = "formal_extra_hpo_not_authorized"
        f.write_queue()
    else:
        record = f.value["launcher"] if change == "launcher" else f.value["jobs"][0][change]
        Path(record["path"]).write_text("changed after pinning")
    with pytest.raises((ValueError, RuntimeError)):
        f.dispatcher().plan()
    assert not f.calls


@pytest.mark.parametrize("which", ["cifar", "imagenetr"])
def test_complete_result_does_not_release_a_live_original_er(f, which):
    name = EXTERNALS[which == "imagenetr"]
    f.live_external(name)
    plan = f.dispatcher().plan()
    selected = [job for job in JOBS if ("imagenetr" in job) == (which == "imagenetr") and job != JOBS[4]]
    assert all(plan[job] == "waiting_dependency" for job in selected)
    assert not f.calls


def test_reused_original_pid_is_not_natural_terminal_success(f):
    f.live_external(EXTERNALS[0], ticks=1234567)
    with pytest.raises((ValueError, RuntimeError)):
        f.dispatcher().plan()
    assert not f.calls


def test_nine_stages_and_uncommitted_tmp_never_release_readers(f):
    cache = f.cache[EXTERNALS[0]]
    (cache / "stage_009.pt").rename(cache / "stage_009.tmp")
    plan = f.dispatcher().plan()
    assert plan[JOBS[0]] == plan[JOBS[1]] == "waiting_cache"
    assert not f.calls


def test_same_round_and_later_unallocated_workers_keep_budget_ledger(f):
    f.snapshot[1].update(used_mib=0, free_mib=7000)
    d = f.dispatcher()
    with d.lock():
        d.cycle(True)
        on_gpu1 = [name for name, record in d.owned.items() if record["gpu"] == 1]
        assert len(on_gpu1) == 1
        assert len([call for call in f.calls if "cifar" in call[0][-1]]) == 1
        before = len(f.calls)
        d.cycle(True)
        assert len(f.calls) == before
    # The mocked NVML snapshot never accounts for the fresh processes: only
    # the dispatcher's reservation ledger prevents admitting the next reader.
    assert f.snapshot[1]["processes"] == {}


def test_unknown_gpu_process_blocks_only_its_card(f):
    f.snapshot[1]["processes"] = {98765: 1}
    d = f.dispatcher()
    with d.lock():
        plan = d.cycle(True)
    assert all("cifar" not in call[0][-1] and "current" not in call[0][-1] for call in f.calls)
    assert len(f.calls) == 2
    assert plan[JOBS[0]] == "blocked_unknown"


@pytest.mark.parametrize("artifact", ["receipt", "log", "output", "cohort_report"])
def test_existing_artifact_never_launches_again(f, artifact):
    name = JOBS[0]
    entry = f.value["jobs"][0]
    if artifact == "receipt":
        save(f.receipt(name), {"previous": "attempt"})
    elif artifact == "log":
        path = f.root / "campaign_20261117" / "formal" / "logs" / (name + ".log")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    elif artifact == "output":
        Path(json.loads(Path(entry["config"]["path"]).read_text())["output_dir"]).mkdir(parents=True)
    else:
        save(f.root / "campaign_20261117" / "formal" / "cohorts" / (name + ".json"), {"status": "old output"})
    d = f.dispatcher()
    with d.lock():
        with pytest.raises((ValueError, RuntimeError), match="Existing|existing|duplicate"):
            d.cycle(True)
    assert all(name not in call[0][-1] for call in f.calls)


def test_worker_handshake_records_worker_not_launcher_wrapper(f):
    d = f.dispatcher()
    with d.lock():
        d.cycle(True)
    assert len(d.owned) == 5
    assert all(record["identity"]["pid"] != 9999 for record in d.owned.values())
    assert {record["identity"]["pid"] for record in d.owned.values()} == set(f.probes)
    for name, record in d.owned.items():
        proof = json.loads((f.root / "dispatch-state" / (name + ".handshake.json")).read_text())
        assert proof["verified"] is True
        assert proof["launcher_pid"] == 9999
        assert proof["receipt"] == record["identity"]
        assert proof["observed_process"] == f.probes[record["identity"]["pid"]]


def test_execute_requires_exclusive_lock_and_duplicate_instance_fails(f):
    first, second = f.dispatcher(), f.dispatcher()
    with pytest.raises((ValueError, RuntimeError)):
        first.cycle(True)
    with first.lock():
        with pytest.raises((OSError, RuntimeError, ValueError)):
            with second.lock():
                pytest.fail("Second dispatcher unexpectedly acquired execution lock")
    assert not f.calls


def test_handshake_failure_is_durable_and_never_restarted(f, monkeypatch):
    monkeypatch.setattr(dispatch.os, "kill", lambda *args: pytest.fail("Dispatcher must never kill a process"))
    f.handshake_missing = True
    d = f.dispatcher()
    with d.lock():
        with pytest.raises((ValueError, RuntimeError)):
            d.cycle(True)
    assert len(f.calls) == 1
    proof = json.loads((f.root / "dispatch-state" / (JOBS[0] + ".handshake.json")).read_text())
    assert proof["verified"] is False
    assert proof["launcher_pid"] == 9999
    assert proof["receipt"]["job"] == JOBS[0]
    assert proof["receipt"]["pid"] == 7001
    assert int(proof["receipt"]["proc_start_ticks"]) == 70010
    assert proof["observed_process"] is None
    assert (f.root / "dispatch-state" / "failure.json").is_file()
    events = (f.root / "dispatch-state" / "events.jsonl").read_text().splitlines()
    assert any(json.loads(line).get("error") for line in events)
    second = f.dispatcher()
    with second.lock():
        with pytest.raises((ValueError, RuntimeError)):
            second.cycle(True)
    assert len(f.calls) == 1


def test_module_requires_closed_d54_and_stays_on_gpu1(f):
    entry = next(job for job in f.value["jobs"] if job["id"] == JOBS[4])
    decision = Path(entry["decision"]["path"])
    value = json.loads(decision.read_text())
    value["hpo_closed"] = False
    save(decision, value)
    entry["decision"] = pinned(decision)
    f.write_queue()
    with pytest.raises((ValueError, RuntimeError)):
        f.dispatcher().plan()
    assert not f.calls


def test_unallocated_external_modules_reserve_their_full_high_water_budgets(f):
    for name in EXTERNALS[2:]:
        f.live_external(name, used=100)
    for gpu in (0, 1):
        f.snapshot[gpu].update(used_mib=100, free_mib=10000)
    d = f.dispatcher()
    report = d.plan()
    assert report[JOBS[0]] == report[JOBS[1]] == report[JOBS[4]] == "waiting_memory"
    # Both fixed cards reserve their own 6440 MiB module, even before its peak.
    assert all(d.headroom(f.snapshot)[gpu] == 10000 - 1024 - (6440 - 100) for gpu in (0, 1))
    assert not f.calls


def test_external_er_future_128_batch_budget_is_reserved_even_at_low_observed_usage(f):
    f.live_external(EXTERNALS[1], used=100)
    f.snapshot[0].update(used_mib=100, free_mib=31900)
    assert f.dispatcher().headroom(f.snapshot)[0] == 31900 - 1024 - (24576 - 100)


def test_observed_allocation_above_budget_is_not_subtracted_twice(f):
    f.live_external(EXTERNALS[2], used=10000)
    f.snapshot[1].update(used_mib=10000, free_mib=6000)
    assert f.dispatcher().headroom(f.snapshot)[1] == 6000 - 1024


def test_owned_worker_failure_stops_dispatch_and_remains_closed_on_restart(f):
    f.value["jobs"] = [f.value["jobs"][0]]
    f.write_queue()
    d = f.dispatcher()
    with d.lock():
        d.cycle(True)
        identity = next(iter(d.owned.values()))["identity"]
        del f.probes[identity["pid"]]
        with pytest.raises((ValueError, RuntimeError), match="failed|failure|complete"):
            d.cycle(True)
    assert len(f.calls) == 1
    second = f.dispatcher()
    with second.lock():
        with pytest.raises((ValueError, RuntimeError)):
            second.cycle(True)
    assert len(f.calls) == 1


@pytest.mark.parametrize("placement", ["inherited", "inherited_empty", "spec"])
def test_cuda_visible_devices_permutation_is_refused(f, monkeypatch, placement):
    if placement.startswith("inherited"):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "" if placement == "inherited_empty" else "1,0")
    else:
        entry = f.value["jobs"][0]
        path = Path(entry["spec"]["path"])
        spec = json.loads(path.read_text())
        spec["jobs"][0]["environment"] = {"CUDA_VISIBLE_DEVICES": "1,0"}
        save(path, spec)
        entry["spec"] = pinned(path)
        f.write_queue()
    with pytest.raises((ValueError, RuntimeError), match="remap"):
        f.dispatcher().plan()
    assert not f.calls


def test_each_actual_launch_rechecks_unknown_processes_after_previous_handshake(f):
    def memory():
        current = copy.deepcopy(f.snapshot)
        if f.calls:
            current[1]["processes"] = {98765: 1}
        return current

    d = f.dispatcher(memory=memory)
    with d.lock():
        d.cycle(True)
    on_gpu1 = [record for record in d.owned.values() if record["gpu"] == 1]
    assert len(on_gpu1) == 1


@pytest.mark.parametrize("field,value", [("device", [0]), ("seed", [2000])])
def test_repinning_does_not_override_fixed_selected_device_or_seed(f, field, value):
    entry = f.value["jobs"][0]
    path = Path(entry["config"]["path"])
    config = json.loads(path.read_text())
    config[field] = value
    save(path, config)
    entry["config"] = pinned(path)
    f.write_queue()
    with pytest.raises((ValueError, RuntimeError), match="approved identity|Config identity|placement|seed"):
        f.dispatcher().plan()
    assert not f.calls


@pytest.mark.parametrize("code,allowed", [(0, True), (1, False)])
def test_zombie_terminal_requires_original_identity_and_success_exit_code(f, code, allowed):
    f.live_external(EXTERNALS[0], state="Z", exit_code=code)
    if allowed:
        assert f.dispatcher().plan()[JOBS[0]] == "ready"
    else:
        with pytest.raises((ValueError, RuntimeError), match="nonzero|exit"):
            f.dispatcher().plan()
    assert not f.calls


@pytest.mark.parametrize("field,value", [("pid", None), ("proc_start_ticks", "-1"), ("command", [])])
def test_malformed_original_process_identity_cannot_become_terminal_success(f, field, value):
    external = f.value["externals"][0]
    path = Path(external["receipt"]["path"])
    receipt = json.loads(path.read_text())
    receipt[field] = value
    save(path, receipt)
    external["receipt"] = pinned(path)
    f.write_queue()
    with pytest.raises((ValueError, RuntimeError)):
        f.dispatcher().plan()
    assert not f.calls


def test_distinct_state_directories_do_not_bypass_single_runtime_lock(f):
    first = f.dispatcher()
    value = copy.deepcopy(f.value)
    value["state_dir"] = str(f.root / "another-state-directory")
    other_path = f.root / "another-queue.json"
    save(other_path, value)
    second = dispatch.Dispatcher(other_path, probe=f.probes.get, memory=lambda: f.snapshot, spawn=f.spawn)
    with first.lock():
        with pytest.raises((OSError, ValueError, RuntimeError)):
            with second.lock():
                pytest.fail("Different state_dir bypassed the runtime execution lock")
    assert not f.calls


def test_receipt_appearing_during_final_resource_check_is_never_overwritten(f):
    f.value["jobs"] = [f.value["jobs"][0]]
    f.write_queue()
    observations = 0
    existing = {"previous_attempt": "must survive without a second launch"}

    def memory():
        nonlocal observations
        observations += 1
        if observations == 2:
            save(f.receipt(JOBS[0]), existing)
        return copy.deepcopy(f.snapshot)

    d = f.dispatcher(memory=memory)
    with d.lock():
        with pytest.raises((ValueError, RuntimeError), match="Existing|existing|duplicate"):
            d.cycle(True)
    assert not f.calls
    assert json.loads(f.receipt(JOBS[0]).read_text()) == existing


@pytest.mark.parametrize("field,value", [
    ("validation_only", True), ("smoke", True), ("max_tasks", 3),
    ("resume", "old_checkpoint.pt"), ("validation_max_per_class", 25), ("joint_max_classes", 20),
])
def test_partial_or_resumed_limits_are_never_formal(f, field, value):
    config = json.loads(Path(f.value["jobs"][0]["config"]["path"]).read_text())
    config[field] = value
    before = files(f.root)
    with pytest.raises(ValueError, match="Partial|resumed|smoke"):
        dispatch.formal_config(config, "cifar224")
    assert files(f.root) == before
    assert not f.calls


@pytest.mark.parametrize("field,value", [
    ("cl27_protocol", "other_protocol"), ("shuffle", False),
    ("backbone_type", "vit_small_patch16_224"), ("pretrained", False),
    ("init_cls", 20), ("increment", 20), ("fixed_memory", True),
    ("memory_size", 1999), ("memory_per_class", 10), ("replay_selection", "random"),
])
def test_fixed_formal_protocol_and_replay_budget_cannot_drift(f, field, value):
    config = json.loads(Path(f.value["jobs"][0]["config"]["path"]).read_text())
    config[field] = value
    with pytest.raises(ValueError, match="PILOT|pretrained|protocol|replay budget"):
        dispatch.formal_config(config, "cifar224")
    assert not f.calls


@pytest.mark.parametrize("index", range(4))
def test_repinning_external_receipt_cannot_move_its_fixed_gpu(f, index):
    item = f.value["externals"][index]
    item["gpu"] = 1 - item["gpu"]
    path = Path(item["receipt"]["path"])
    receipt = json.loads(path.read_text())
    receipt["metadata"]["gpu"] = item["gpu"]
    save(path, receipt)
    item["receipt"] = pinned(path)
    f.write_queue()
    with pytest.raises(ValueError, match="External placement"):
        f.dispatcher().plan()
    assert not f.calls


@pytest.mark.parametrize("change", ["method", "dataset", "preferred", "predecessor", "parameters", "empty_parameters", "trial"])
def test_d54_selection_fields_are_bound_to_this_exact_module(f, change):
    entry = next(item for item in f.value["jobs"] if item["id"] == JOBS[4])
    path = Path(entry["decision"]["path"])
    decision = json.loads(path.read_text())
    if change == "preferred":
        decision["formal_refit"]["preferred_device"] = [0]
    elif change == "predecessor":
        decision["launch_constraints"]["required_predecessor"]["job"] = EXTERNALS[1]
    elif change == "parameters":
        decision["selected_parameters"]["outer_updates"] = 2
    elif change == "empty_parameters":
        decision["selected_parameters"] = {}
    elif change == "trial":
        decision["selected_development_trial"] = "an_unselected_development_candidate"
    else:
        decision[change] = "history_transport" if change == "method" else "cifar224"
    save(path, decision)
    entry["decision"] = pinned(path)
    f.write_queue()
    with pytest.raises(ValueError, match="D0054"):
        f.dispatcher().plan()
    assert not f.calls


def test_known_live_process_on_another_physical_gpu_stops_dispatch(f):
    _, receipt = f.live_external(EXTERNALS[2])  # Fixed GPU1.
    f.snapshot[0]["processes"][receipt["pid"]] = 100
    d = f.dispatcher()
    with d.lock():
        with pytest.raises(ValueError, match="GPU|placement|physical|card"):
            d.cycle(True)
    assert not f.calls


def test_repinning_shape_changed_config_cannot_reapprove_its_memory_budget(f):
    entry = f.value["jobs"][0]
    path = Path(entry["config"]["path"])
    config = json.loads(path.read_text())
    config["feature_batch_size"] = 1024
    save(path, config)
    entry["config"] = pinned(path)
    f.write_queue()
    with pytest.raises(ValueError, match="approved identity/budget"):
        f.dispatcher().plan()
    assert not f.calls


def test_non_atomic_nvml_samples_never_refund_used_memory_credit(f):
    f.live_external(EXTERNALS[2], used=8000)
    # memInfo was sampled before the per-PID allocation rose to 8000 MiB.
    f.snapshot[1].update(used_mib=1000, free_mib=9000)
    d = f.dispatcher()
    assert d.headroom(f.snapshot)[1] == 10000 - 8000 - 1024
    report = d.plan()
    assert all(report[name] == "waiting_memory" for name in (JOBS[0], JOBS[1], JOBS[4]))
    assert not f.calls


def test_starting_worker_growth_between_nvml_samples_cannot_admit_second_reader(f):
    # Only GPU0 is available; its capacity admits one 4096 MiB reader + margin.
    f.snapshot[1]["processes"] = {98765: 1}
    f.snapshot[0].update(used_mib=0, free_mib=8500)
    d = f.dispatcher()
    with d.lock():
        d.cycle(True)
        assert len(f.calls) == 1
        identity = next(iter(d.owned.values()))["identity"]
        # The next memInfo query sees 2 GiB used, then the per-PID query sees
        # 4 GiB. The pinned 4 GiB reservation cannot become a negative credit.
        f.snapshot[0].update(used_mib=2000, free_mib=6500, processes={identity["pid"]: 4000})
        report = d.cycle(True)
    assert len(f.calls) == 1
    assert report[JOBS[3]] == "waiting_memory"
    assert d.headroom(f.snapshot)[0] == 8500 - 4096 - 1024


def test_fresh_resource_check_revalidates_owned_pid_before_another_launch(f, monkeypatch):
    monkeypatch.setattr(dispatch.os, "kill", lambda *args: pytest.fail("Do not kill a reused PID"))

    def memory():
        current = copy.deepcopy(f.snapshot)
        if f.calls:
            pid = 7001
            f.probes[pid] = dict(state="R", startticks=70011, cmd=["unrelated_process"], exit_code=None)
            current[1]["processes"] = {pid: 100}
        return current

    d = f.dispatcher(memory=memory)
    with d.lock():
        with pytest.raises(ValueError, match="PID.*reused|identity"):
            d.cycle(True)
    assert len(f.calls) == 1
    assert JOBS[0] in f.calls[0][0][-1]
    # The original launch record remains available for recovery; no receipt
    # mutation or unknown-process relabelling authorizes a second dispatch.
    assert json.loads(f.receipt(JOBS[0]).read_text())["proc_start_ticks"] == "70010"
    assert (f.root / "dispatch-state" / "failure.json").exists()


def test_external_saved_run_device_is_checked_before_it_appears_in_nvml(f):
    item, _ = f.live_external(EXTERNALS[3])  # Fixed GPU0, with no allocation yet.
    assert all(not card["processes"] for card in f.snapshot.values())
    path = Path(item["config"]["path"])
    save(path, {"device": [1]})
    item["config"] = pinned(path)
    f.write_queue()
    with pytest.raises(ValueError, match="device|Device|GPU|placement"):
        f.dispatcher().plan()
    assert not f.calls


@pytest.mark.parametrize("state", ["R", "S"])
def test_same_starttick_empty_cmd_keeps_er_dependency_and_future_budget_reserved(f, state):
    entry, identity = f.live_external(EXTERNALS[0], state=state)
    f.probes[identity["pid"]]["cmd"] = []
    receipt_before = Path(entry["receipt"]["path"]).read_bytes()
    before = files(f.root)
    d = f.dispatcher()

    report = d.plan()
    assert report[JOBS[0]] == report[JOBS[1]] == report[JOBS[4]] == "waiting_dependency"
    assert d.headroom(f.snapshot)[1] == 32000 - 24576 - 1024
    assert files(f.root) == before
    assert not f.calls

    # A complete result/cache and an empty cmdline are not terminal evidence.
    # Only the next observation that the original PID is absent releases it.
    del f.probes[identity["pid"]]
    report = d.plan()
    assert report[JOBS[0]] == report[JOBS[1]] == report[JOBS[4]] == "ready"
    assert d.headroom(f.snapshot)[1] == 32000 - 1024
    assert Path(entry["receipt"]["path"]).read_bytes() == receipt_before
    assert files(f.root) == before
    assert not f.calls


def test_completed_owned_reader_with_empty_cmd_stays_reserved_until_actual_absence(f, monkeypatch):
    monkeypatch.setattr(dispatch.os, "kill", lambda *args: pytest.fail("Observer must not kill a process"))
    f.value["jobs"] = [f.value["jobs"][0]]
    f.write_queue()
    d = f.dispatcher()
    with d.lock():
        assert d.cycle(True)[JOBS[0]] == "running"
        owned = d.owned[JOBS[0]]["identity"]
        receipt_before = f.receipt(JOBS[0]).read_bytes()
        save(Path(d.jobs[JOBS[0]]["config_data"]["output_dir"]) / "results.json", f.result("history_transport"))
        f.probes[owned["pid"]].update(state="S", cmd=[])

        assert d.plan()[JOBS[0]] == "running"
        assert JOBS[0] in d.owned and JOBS[0] not in d.finished
        assert d.headroom(f.snapshot)[1] == 32000 - 4096 - 1024

        del f.probes[owned["pid"]]
        assert d.plan()[JOBS[0]] == "complete"
        assert JOBS[0] not in d.owned and JOBS[0] in d.finished
        assert d.headroom(f.snapshot)[1] == 32000 - 1024
        assert f.receipt(JOBS[0]).read_bytes() == receipt_before
    assert len(f.calls) == 1


def test_nonempty_wrong_command_still_fails_with_locatable_process_diagnostics(f):
    _, identity = f.live_external(EXTERNALS[0])
    f.probes[identity["pid"]]["cmd"] = ["unexpected_worker_command", "--wrong-job"]
    with pytest.raises(ValueError) as caught:
        f.dispatcher().plan()
    message = str(caught.value)
    assert str(identity["pid"]) in message
    assert "unexpected_worker_command" in message
    assert "external.py" in message
    assert "cmd" in message and "state" in message
    assert not f.calls


def fake_nvml_module(monkeypatch, used_bytes, *, unavailable=None, define_constant=False):
    module = types.ModuleType("pynvml")
    calls = {"init": 0, "shutdown": 0}

    def count(name):
        calls[name] += 1

    module.nvmlInit = lambda: count("init")
    module.nvmlShutdown = lambda: count("shutdown")
    module.nvmlDeviceGetHandleByIndex = lambda index: index
    module.nvmlDeviceGetMemoryInfo = lambda handle: types.SimpleNamespace(
        total=32 * 2**30, used=2 * 2**30, free=30 * 2**30)
    module.nvmlDeviceGetComputeRunningProcesses = lambda handle: [
        types.SimpleNamespace(pid=100 + handle, usedGpuMemory=used_bytes)]
    module.nvmlDeviceGetGraphicsRunningProcesses = lambda handle: []
    if define_constant:
        module.NVML_VALUE_NOT_AVAILABLE = unavailable
    monkeypatch.setitem(sys.modules, "pynvml", module)
    return module, calls


@pytest.mark.parametrize("used_bytes,expected_mib", [(0, 0), (2**20 + 1, 2)])
def test_nvml_without_optional_constant_accepts_normal_allocations(monkeypatch, used_bytes, expected_mib):
    module, calls = fake_nvml_module(monkeypatch, used_bytes)
    assert not hasattr(module, "NVML_VALUE_NOT_AVAILABLE")
    snapshot = dispatch.nvml_snapshot()
    assert set(snapshot) == {0, 1}
    for gpu in (0, 1):
        assert snapshot[gpu] == dict(total_mib=32768, used_mib=2048, free_mib=30720,
                                    processes={100 + gpu: expected_mib})
    assert calls == {"init": 1, "shutdown": 1}


@pytest.mark.parametrize("value,define_constant", [(None, False), (2**64 - 1, False), (2**32 - 1, True)])
def test_nvml_unavailable_memory_fails_closed_and_always_shuts_down(monkeypatch, value, define_constant):
    _, calls = fake_nvml_module(monkeypatch, value, unavailable=value, define_constant=define_constant)
    with pytest.raises(ValueError, match="allocation|unavailable"):
        dispatch.nvml_snapshot()
    assert calls == {"init": 1, "shutdown": 1}
