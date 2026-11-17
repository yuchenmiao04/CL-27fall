"""Observe current20 on the original host or run TUNA on an added two-GPU host.

The complete prepared manifest is unchanged. This user-authorized partition
overrides only the old same-card current20 -> TUNA dependency: current20 is
adoption-only, while TUNA may start concurrently on the other server.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import types


GROUPS = {
    'current20': ('formal_cifar_current_epochs20_v1', 'formal_imagenetr_current_epochs20_v1'),
    'tuna': ('formal_cifar_tuna_native_v1', 'formal_imagenetr_tuna_native_v1'),
}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def load_guarded_queue(manifest_path):
    """Import exactly the manifest's original queue after checking its bytes.

    An isolated package namespace makes relative imports resolve to that same
    verified checkout rather than any ambient PYTHONPATH or module cache.
    """
    manifest = json.loads(Path(manifest_path).read_text())
    repo = Path(manifest['repo'])
    need(repo.is_absolute() and repo.is_dir(), 'Manifest repository must be absolute and present')
    inventory = manifest['source_files_sha256']
    paths = {name: repo / 'scripts' / (name + '.py') for name in
             ('prepare_followup_campaign', 'run_followup_queue')}
    for name, path in paths.items():
        relative = 'scripts/' + name + '.py'
        need(path.is_file() and inventory.get(relative) == file_sha(path),
             'Original manifest-bound controller source changed: ' + relative)
    identity = hashlib.sha256((str(repo.resolve()) + json.dumps(
        {name: file_sha(path) for name, path in paths.items()}, sort_keys=True)).encode()).hexdigest()
    package_name = '_cl27_partition_' + identity
    if package_name + '.run_followup_queue' in sys.modules:
        return sys.modules[package_name + '.run_followup_queue']
    package = types.ModuleType(package_name)
    package.__path__ = [str(repo / 'scripts')]
    sys.modules[package_name] = package
    for name, path in paths.items():
        spec = importlib.util.spec_from_file_location(package_name + '.' + name, path)
        need(spec is not None and spec.loader is not None, 'Cannot load original queue controller')
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(spec.name, None)
            raise
    return module


class PartitionController:
    def __init__(self, manifest, partition, *, base=None):
        need(partition in GROUPS, 'Unknown follow-up partition')
        self.path = Path(manifest).resolve()
        self.partition, self.jobs = partition, GROUPS[partition]
        if base is None:
            api = load_guarded_queue(self.path)
            base = api.FollowupQueue(self.path)
        else:
            # Dependency injection is used by local process/NVML contract tests.
            api = sys.modules[type(base).__module__]
            need(base.path == self.path, 'Injected base belongs to another manifest')
        self.base, self.api = base, api
        self.controller_path = Path(__file__).resolve()
        self.controller_sha = file_sha(self.controller_path)
        self.latest = base.state / ('partition_' + partition + '_latest.json')
        self.failure = base.state / ('partition_' + partition + '_failure.json')

    def observe_assigned(self):
        base = self.base
        full = base.entries
        try:
            # The old receipt/live/completion implementation is unchanged.
            # Excluded host-local receipts, intents and absent outputs have no
            # bearing on this partition's ownership or completion predicate.
            base.entries = {job: full[job] for job in self.jobs}
            if self.partition == 'current20':
                for job in self.jobs:
                    need(base.receipt_path(job).is_file(),
                         'current20 is adoption-only; missing original receipt: ' + job)
            return base.observe()
        finally:
            base.entries = full

    def cycle(self, execute=False):
        base, api = self.base, self.api
        need(not execute or base.locked, 'Execute requires the original campaign lock')
        need(file_sha(self.controller_path) == self.controller_sha, 'Partition controller changed after admission')
        base.validate()  # Admit all four immutable recipes and source identities.
        if execute:
            need(not self.failure.exists() and not (base.state / 'failure.json').exists(),
                 'Recorded failure requires review; no automatic retry')
        statuses, identities = self.observe_assigned()
        snapshot = base.memory()
        free = base.disk(base.root).free
        actions = []
        if self.partition == 'tuna' and all(v == 'waiting' for v in statuses.values()):
            need(free >= base.manifest['minimum_initial_free_bytes'], 'Insufficient initial disk budget')
        for job in self.jobs:
            row, status = base.entries[job], statuses[job]
            gpu = row['gpu']
            known = identities.get(job)
            if known is not None:
                need(known['pid'] not in set(snapshot[1 - gpu]['pids']),
                     'Known worker appears on another physical GPU')
                continue
            if status == 'complete':
                continue
            need(self.partition == 'tuna' and status == 'waiting',
                 'current20 is adoption-only; never relaunch a missing worker')
            if snapshot[gpu]['pids']:
                actions.append(dict(job=job, gpu=gpu, state='waiting_unknown_gpu_process'))
                continue
            if snapshot[gpu]['free_mib'] < 12288 + 1024:
                actions.append(dict(job=job, gpu=gpu, state='waiting_gpu_memory'))
                continue
            need(free >= 4 * 2**30, 'Insufficient atomic-save disk margin for next job')
            actions.append(dict(job=job, gpu=gpu, state='ready'))
            if execute:
                receipt = base.launch(job)
                identities[job], statuses[job] = receipt, 'running'
                actions[-1]['state'] = 'launched'
        report = dict(utc=api.utc(), partition=self.partition,
            partition_policy='user_authorized_separate_servers_current20_adoption_only_tuna_independent',
            partition_controller_path=str(self.controller_path), partition_controller_sha256=self.controller_sha,
            manifest_sha256=base.manifest_sha, assigned_jobs=list(self.jobs),
            statuses=statuses, actions=actions, all_complete=all(v == 'complete' for v in statuses.values()),
            disk_free_bytes=free, gpu_snapshot=snapshot)
        if execute:
            api.write(self.latest, report)
        return report

    def run(self, poll_seconds=30):
        need(1 <= poll_seconds <= 60, 'Polling interval must be between 1 and 60 seconds')
        with self.base.exclusive():
            try:
                while True:
                    report = self.cycle(True)
                    print(json.dumps(report, allow_nan=False), flush=True)
                    if report['all_complete']:
                        return
                    time.sleep(poll_seconds)
            except Exception as error:
                self.api.write(self.failure, dict(utc=self.api.utc(), partition=self.partition,
                    partition_controller_sha256=self.controller_sha,
                    type=type(error).__name__, message=str(error)))
                raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--partition', choices=GROUPS, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--plan', action='store_true', help='Read-only report (default)')
    mode.add_argument('--execute', action='store_true')
    parser.add_argument('--poll-seconds', type=float, default=30)
    args = parser.parse_args()
    controller = PartitionController(args.manifest, args.partition)
    if args.execute:
        controller.run(args.poll_seconds)
    else:
        print(json.dumps(controller.cycle(False), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
