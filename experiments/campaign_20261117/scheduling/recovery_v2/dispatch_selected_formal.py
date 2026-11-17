"""Admission control for five already selected formal jobs; default is --plan.

Queue v1: runtime_root, python, state_dir, launcher={path,sha256}, externals=[
{job,gpu,receipt:{path,sha256},config:{path,sha256},result}], jobs=[{id,spec:{path,sha256},
config:{path,sha256},predecessor,cache_dir? ,decision?:{path,sha256}}].
Only the R module uses decision (D0054); only HT/PSB use cache_dir. Existing
specifications are passed byte-for-byte to the original launch_jobs.py.
Scope: this campaign's five approved original configurations. A configuration
or path migration requires renewed identity and VRAM-budget approval.
No Torch import, CUDA initialization, parameter selection, killing or retries.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time


ALLOWED = {
    "formal_cifar_transport_selected_v2": (1, "cifar224", "history_transport"),
    "formal_cifar_plastic_shared_selected_v5": (1, "cifar224", "plastic_shared_basis"),
    "formal_imagenetr_transport_selected_v3": (0, "imagenetr", "history_transport"),
    "formal_imagenetr_plastic_shared_selected_v5": (0, "imagenetr", "plastic_shared_basis"),
    "formal_imagenetr_current_selected_v5": (1, "imagenetr", "current_adapted"),
}
APPROVED_CONFIG = {
    "formal_cifar_transport_selected_v2": "77c1b36cde65d0eeec688f6c7005333b0adcbe5355afda73670c5c05b8aa519e",
    "formal_cifar_plastic_shared_selected_v5": "70976433b18c1d7a38d3ab2b572783518fdccf742fed585dfe1f526feae2db31",
    "formal_imagenetr_transport_selected_v3": "7a90eee367afef3606f0be7b180ff2ef8d2edf5b74ffd85094e5f05bcc5cca8a",
    "formal_imagenetr_plastic_shared_selected_v5": "ef4eb5db5329bf262ba306361cd89ee567804c03be194e1c3a50f7ca81d0d1c9",
    "formal_imagenetr_current_selected_v5": "1579a138db77f313d729c04063a05fb2b991164d8c25d23cf0ce2c1d14274553",
}
EXTERNAL = {"formal_cifar_er_fixed_v1": 24576, "formal_imagenetr_er_fixed_v1": 24576,
            "cifar_current_bounded64_full_v4": 6440, "cifar_current_mb2_full_v5": 6440}
EXTERNAL_GPU = dict(zip(EXTERNAL, (1, 0, 1, 0)))
DEAD = {"Z", "X", "x"}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(condition, message):
    if not condition:
        raise ValueError(message)


def pinned(reference):
    check(sha(reference["path"]) == reference["sha256"], "Pinned file changed: " + reference["path"])
    return Path(reference["path"])


def command(args):
    check(isinstance(args, list) and bool(args) and all(isinstance(x, str) and x for x in args), "Missing original command identity")
    args = list(args)
    check(not any("CUDA_VISIBLE_DEVICES" in item for item in args), "Device remapping is forbidden")
    if args and Path(args[0]).name == "env":
        args.pop(0)
        while args and "=" in args[0]:
            check(args[0] in {"CL27_CURRENT_FROZEN_PREFIX_REUSE=0", "CL27_CURRENT_FROZEN_PREFIX_REUSE=1"},
                  "Unexpected env wrapper")
            args.pop(0)
    return args


def probe_process(pid):
    path = Path("/proc") / str(pid)
    try:
        text = (path / "stat").read_text()
        fields = text[text.rfind(")") + 2:].split()
        args = [x.decode() for x in (path / "cmdline").read_bytes().split(b"\0") if x]
        latest = (path / "stat").read_text()
        after = latest[latest.rfind(")") + 2:].split()
        check(fields[19] == after[19], "PID changed during process observation")
        fields = after
        return dict(state=fields[0], startticks=int(fields[19]),
                    cmd=args,
                    exit_code=os.waitstatus_to_exitcode(int(fields[49])) if fields[0] in DEAD else None)
    except (FileNotFoundError, ProcessLookupError):
        return None


def nvml_snapshot():
    import pynvml as nv  # NVML observes the driver; it does not create a CUDA context.
    nv.nvmlInit()
    try:
        result = {}
        for gpu in (0, 1):
            handle = nv.nvmlDeviceGetHandleByIndex(gpu)
            mem = nv.nvmlDeviceGetMemoryInfo(handle)
            processes = {}
            for process in nv.nvmlDeviceGetComputeRunningProcesses(handle) + nv.nvmlDeviceGetGraphicsRunningProcesses(handle):
                allocation = process.usedGpuMemory
                sentinel = getattr(nv, "NVML_VALUE_NOT_AVAILABLE", (1 << 64) - 1)
                check(isinstance(allocation, (int, float)) and not isinstance(allocation, bool)
                      and math.isfinite(allocation) and 0 <= allocation < (1 << 64) - 1
                      and allocation != sentinel, "NVML process allocation unavailable")
                processes[process.pid] = max(processes.get(process.pid, 0), math.ceil(allocation / 2**20))
            result[gpu] = dict(total_mib=mem.total / 2**20, used_mib=mem.used / 2**20,
                               free_mib=mem.free / 2**20, processes=processes)
        return result
    finally:
        nv.nvmlShutdown()


def formal_config(config, dataset):
    width = 10 if dataset == "cifar224" else 20
    check(config.get("validation_only", False) is False and config.get("smoke", False) is False
          and all(config.get(k) is None for k in ("max_tasks", "resume", "validation_max_per_class", "joint_max_classes")),
          "Partial/resumed/smoke configurations cannot become formal jobs")
    check(config.get("cl27_protocol") == "pilot_figure2" and config.get("shuffle") is True
          and config.get("backbone_type") == "vit_base_patch16_224" and config.get("pretrained") is True
          and (config.get("init_cls"), config.get("increment")) == (width, width), "Wrong PILOT/pretrained/class protocol")
    check(config.get("fixed_memory") is (dataset == "imagenetr") and config.get("memory_size") == width * 200
          and config.get("memory_per_class") == 20 and config.get("replay_selection") == "herding", "Wrong formal replay budget")


class Dispatcher:
    def __init__(self, queue_path, *, probe=probe_process, memory=nvml_snapshot, spawn=subprocess.Popen,
                 monotonic=time.monotonic, sleep=time.sleep, launch_timeout=10):
        self.path, self.q = Path(queue_path), read(queue_path)
        self.queue_sha, self.owned, self.finished, self.last_event = sha(queue_path), {}, set(), None
        self.probe, self.memory, self.spawn, self.clock, self.sleep = probe, memory, spawn, monotonic, sleep
        self.timeout, self.locked = launch_timeout, False
        self.root = Path(self.q["runtime_root"]) / "campaign_20261117" / "formal"
        self.state = Path(self.q["state_dir"])
        self.jobs, self.refs = {}, {}
        check(self.q["version"] == 1 and math.isfinite(launch_timeout) and launch_timeout > 0, "Invalid queue version/timeout")
        for item in self.q["externals"]:
            name = item["job"]
            check(name in EXTERNAL and name not in self.refs, "Unexpected/duplicate external job")
            receipt = read(pinned(item["receipt"]))
            check(receipt["job"] == name and receipt["metadata"]["gpu"] == item["gpu"] == EXTERNAL_GPU[name], "External placement differs")
            check(item["gpu"] in (0, 1) and type(receipt["pid"]) is int and receipt["pid"] > 0
                  and str(receipt["proc_start_ticks"]).isdigit() and int(receipt["proc_start_ticks"]) > 0, "Invalid external process identity")
            check(Path(item["result"]).parent.name == name and Path(item["result"]).name == "results.json",
                  "External result belongs to another job")
            command(receipt["command"])
            check("CUDA_VISIBLE_DEVICES" not in receipt.get("environment", {}), "External device remapping")
            self.refs[name] = dict(item, identity=receipt, budget=EXTERNAL[name])
        check(set(self.refs) == set(EXTERNAL), "All four external reservations must be declared")
        for item in self.q["jobs"]:
            name = item["id"]
            check(name in ALLOWED and name not in self.jobs, "Job outside the five selected formal jobs")
            self.jobs[name] = dict(item, budget=6440 if ALLOWED[name][2] == "current_adapted" else 4096)
        check(bool(self.jobs), "Empty selected queue")
        self.validate()

    def validate(self):
        check(sha(self.path) == self.queue_sha, "Queue changed")
        check(os.environ.get("CUDA_VISIBLE_DEVICES") in (None, "0,1"), "Inherited GPU remapping is forbidden")
        pinned(self.q["launcher"])
        for ref in self.refs.values():
            pinned(ref["receipt"])
            config = read(pinned(ref["config"]))
            check(Path(ref["config"]["path"]) == Path(ref["result"]).with_name("config.json")
                  and config.get("device") == [EXTERNAL_GPU[ref["job"]]], "External saved config path/device differs")
        for name, item in self.jobs.items():
            check(item["config"]["sha256"] == APPROVED_CONFIG[name], "Config is outside the approved identity/budget table")
            spec = read(pinned(item["spec"]))
            config = read(pinned(item["config"]))
            check(len(spec["jobs"]) == 1, "Each original spec must contain exactly one job")
            job, (gpu, dataset, method) = spec["jobs"][0], ALLOWED[name]
            formal_config(config, dataset)
            check(job["name"] == name and job["area"] == "formal" and job["metadata"]["gpu"] == gpu,
                  "Spec identity/placement differs")
            source = "runtime_bounded_v4" if method == "current_adapted" else "runtime_tuning_v1"
            check(Path(item["config"]["path"]) == self.root / "configs" / (name + ".json")
                  and Path(config["output_dir"]) == self.root / "runs" / name
                  and Path(job["cwd"]) == Path(self.q["runtime_root"]) / source
                  and Path(job["cwd"]).is_dir(), "Runtime/config/output root differs")
            check(config["device"] == [gpu] and config["dataset"] == dataset and config["model_name"] == method
                  and config["phase"] == "final" and config["seed"] == [1993]
                  and config.get("validation_only", False) is False and config.get("max_tasks") is None
                  and Path(config["output_dir"]).name == name, "Config identity/placement differs")
            check("CUDA_VISIBLE_DEVICES" not in job.get("environment", {}), "Job device remapping is forbidden")
            predecessor = "formal_imagenetr_er_fixed_v1" if dataset == "imagenetr" and method != "current_adapted" else "formal_cifar_er_fixed_v1"
            check(item["predecessor"] == predecessor, "Wrong fixed formal ER dependency")
            argv = [self.q["python"], "main.py", "--config", item["config"]["path"]]
            if method != "current_adapted":
                argv = [self.q["python"], "scripts/run_ce_cohort.py", "--replay-only", "--configs", item["config"]["path"],
                        "--trajectory-dir", item["cache_dir"], "--report", str(self.root / "cohorts" / (name + ".json"))]
            else:
                decision = read(pinned(item["decision"]))
                check(decision["id"] == "D0054" and decision["hpo_closed"] is True
                      and decision["formal_refit"]["job"] == name
                      and decision.get("method") == method and decision.get("dataset") == dataset
                      and decision["formal_refit"].get("preferred_device") == [gpu]
                      and decision["launch_constraints"]["required_predecessor"]["job"] == predecessor
                      and bool(decision.get("selected_parameters"))
                      and all(config.get(k) == v for k, v in decision["selected_parameters"].items())
                      and decision.get("selected_development_trial") == job["metadata"].get("selected_development_trial"),
                      "Module selection is not locked by D0054")
            check(command(job["command"]) == argv and job["command"] == argv, "Original job command differs")
            item.update(job=job, config_data=config, gpu=gpu, receipt=str(self.root / "receipts" / (name + ".json")),
                        log=str(self.root / "logs" / (name + ".log")), outputs=[config["output_dir"]] +
                        ([] if method == "current_adapted" else [argv[-1]]))

    def live(self, identity):
        check(type(identity["pid"]) is int and identity["pid"] > 0 and str(identity["proc_start_ticks"]).isdigit()
              and int(identity["proc_start_ticks"]) > 0, "Invalid process identity")
        expected = command(identity["command"])
        info = self.probe(identity["pid"])
        if info is None:
            return False
        check(info["startticks"] == int(identity["proc_start_ticks"]), "Original PID was reused")
        if info["state"] in DEAD:
            check(info.get("exit_code") == 0, "Original process exited nonzero")
            return False
        check(info["cmd"] == expected, "Original process command differs")
        return True

    def complete(self, path, method):
        if not Path(path).is_file():
            return False
        value = read(path)
        return (value.get("status") == "complete" and value.get("phase") == "final" and value.get("method") == method
                and value.get("metrics", {}).get("completed_tasks") == 10
                and [stage.get("task") for stage in value.get("stages", [])] == list(range(10)))

    def dependency(self, item):
        ref = self.refs[item["predecessor"]]
        if self.live(ref["identity"]):
            return "waiting_dependency"
        check(self.complete(ref["result"], "er"), "Fixed ER terminated without a complete ten-stage result")
        if "cache_dir" in item:
            cache = Path(item["cache_dir"])
            reuse = read(ref["result"]).get("execution", {}).get("ce_training_reuse", {})
            expected = [cache / f"stage_{i:03d}.pt" for i in range(10)]
            check(reuse.get("mode") == "record" and reuse.get("execution_role") == "fixed_baseline"
                  and reuse.get("trajectory_directory") == str(cache)
                  and [s.get("task") for s in reuse.get("stages", [])] == list(range(10))
                  and [s.get("path") for s in reuse["stages"]] == [str(p) for p in expected], "Wrong formal ER cache provenance")
            if any(not p.is_file() or p.is_symlink() or p.stat().st_size == 0 for p in expected) or list(cache.glob("*.tmp")):
                return "waiting_cache"
        return None

    def headroom(self, snapshot):
        self.poll_owned()
        live = [(r["identity"], r["gpu"], r["budget"]) for r in self.refs.values() if self.live(r["identity"])]
        live += [(r["identity"], r["gpu"], r["budget"]) for r in self.owned.values()]
        for identity, card, _ in live:
            check(not any(identity["pid"] in snapshot[gpu]["processes"] for gpu in (0, 1) if gpu != card),
                  "Known live PID appears on a different physical GPU")
        result = {}
        for gpu in (0, 1):
            mem = snapshot[gpu]
            known = {r["pid"]: budget for r, card, budget in live if card == gpu}
            observed = mem["processes"]
            check(all(math.isfinite(v) and v >= 0 for v in (*observed.values(), mem["used_mib"], mem["free_mib"])), "Invalid NVML memory")
            capacity = mem["used_mib"] + mem["free_mib"]
            # NVML memory/PID queries are not atomic: a newer allocation must
            # never earn admission credit against an older free-memory sample.
            reserved = sum(max(b, observed.get(pid, 0)) for pid, b in known.items())
            overhead = max(0, mem["used_mib"] - sum(observed.values()))
            result[gpu] = None if set(observed) - set(known) else capacity - reserved - overhead - 1024
        return result

    def launch(self, name, item):
        check(not any(Path(p).exists() for p in (item["receipt"], item["log"], *item["outputs"])),
              "Artifact appeared before launch; refusing duplicate: " + name)
        self.state.mkdir(parents=True, exist_ok=True)
        intent = self.state / (name + ".intent.json")
        with intent.open("x") as stream:
            json.dump(dict(job=name, spec_sha256=item["spec"]["sha256"], queue_sha256=self.queue_sha), stream)
        with (self.state / (name + ".launcher.log")).open("x") as log:
            launcher = self.spawn([self.q["python"], self.q["launcher"]["path"], item["spec"]["path"]],
                                  cwd=self.q["runtime_root"], stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        handshake = dict(job=name, launcher_pid=launcher.pid, expected_command=item["job"]["command"],
                         receipt=None, observed_process=None, verified=False)
        write(self.state / (name + ".handshake.json"), handshake)
        deadline = self.clock() + self.timeout
        while True:
            code = launcher.poll()
            try:
                receipt = read(item["receipt"])
            except (FileNotFoundError, json.JSONDecodeError):
                receipt = None
            if receipt is not None:
                info = self.probe(receipt["pid"])
                handshake.update(receipt=receipt, observed_process=info)
                write(self.state / (name + ".handshake.json"), handshake)
                check(all(receipt[k] == item["job"][k] for k in ("cwd", "command", "metadata"))
                      and receipt["job"] == name and receipt.get("environment", {}) == item["job"].get("environment", {}), "New receipt differs from original spec")
                if info is not None:
                    check(info["startticks"] == int(receipt["proc_start_ticks"]), "New worker PID reused")
                    check(info["state"] not in DEAD, "New worker exited during handshake")
                if info is not None and info["cmd"] == command(receipt["command"]) and code == 0:
                    self.owned[name] = dict(identity=receipt, gpu=item["gpu"], budget=item["budget"])
                    with (self.state / (name + ".identity.json")).open("x") as stream:
                        json.dump(dict(receipt, observed_process=info, launcher_pid=launcher.pid), stream, indent=2)
                    write(self.state / (name + ".handshake.json"), dict(handshake, verified=True))
                    return
            check(code in (None, 0), "Original launcher failed; no retry")
            check(self.clock() < deadline, "Worker identity handshake timed out; no retry or kill")
            self.sleep(.05)

    def plan(self):
        return self.cycle(False)

    def cycle(self, execute=False):
        check(not execute or self.locked, "Execute requires the single dispatcher lock")
        try:
            check(not execute or not (self.state / "failure.json").exists(), "Prior dispatcher failure requires review")
            return self._cycle(execute)
        except BaseException as error:
            if execute:
                value = dict(error=type(error).__name__, message=str(error))
                if not (self.state / "failure.json").exists():
                    with (self.state / "failure.json").open("x") as stream:
                        json.dump(value, stream)
                self.event(value)
            raise

    def _cycle(self, execute):
        self.validate()
        room, report = self.headroom(self.memory()), {}
        for name, item in self.jobs.items():
            if name in self.finished or name in self.owned:
                report[name] = "complete" if name in self.finished else "running"
                continue
            existing = any(Path(p).exists() for p in (item["receipt"], item["log"], str(self.state / (name + ".intent.json")), *item["outputs"]))
            if existing:
                check(not execute, "Existing receipt/output/log/intent blocks duplicate dispatch: " + name)
                report[name] = "existing"
                continue
            reason, free = self.dependency(item), room[item["gpu"]]
            report[name] = reason or ("blocked_unknown" if free is None else "waiting_memory" if free < item["budget"] else "ready")
            if report[name] == "ready":
                room[item["gpu"]] -= item["budget"]  # Reserve even before the allocator appears in NVML.
                if execute:
                    self.validate()
                    fresh = self.headroom(self.memory())[item["gpu"]]
                    reason = self.dependency(item)
                    if reason or fresh is None or fresh < item["budget"]:
                        report[name] = reason or ("blocked_unknown" if fresh is None else "waiting_memory")
                        continue
                    self.launch(name, item)
                    report[name] = "running"
        return report

    def poll_owned(self):
        for name, owned in list(self.owned.items()):
            check(read(self.jobs[name]["receipt"]) == owned["identity"], "Owned receipt changed")
            if not self.live(owned["identity"]):
                check(self.complete(Path(self.jobs[name]["config_data"]["output_dir"]) / "results.json", ALLOWED[name][2]), "Owned job failed; no retry")
                self.finished.add(name)
                del self.owned[name]

    @contextmanager
    def lock(self):
        self.state.mkdir(parents=True, exist_ok=True)
        lock_path = self.root.parent / "scheduling" / "selected_formal_dispatch.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.locked = True
            try:
                yield
            finally:
                self.locked = False
                fcntl.flock(stream, fcntl.LOCK_UN)

    def event(self, value):
        if value != self.last_event:
            with (self.state / "events.jsonl").open("a") as stream:
                stream.write(json.dumps(dict(observed_unix=time.time(), **value)) + "\n")
            self.last_event = value

    def run(self, poll_seconds=5):
        with self.lock():
            try:
                while True:
                    report = self.cycle(True)
                    self.event(dict(jobs=report))
                    if all(state == "complete" for state in report.values()):
                        return report
                    self.sleep(poll_seconds)
            except BaseException as error:
                self.event(dict(error=type(error).__name__, message=str(error)))
                raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    dispatcher = Dispatcher(args.queue)
    print(json.dumps(dispatcher.run() if args.execute else dispatcher.plan(), indent=2))


if __name__ == "__main__":
    main()
