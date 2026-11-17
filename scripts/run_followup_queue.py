"""Run the four predeclared follow-up jobs, one process per physical GPU.

The default is read-only planning. Execution never selects by accuracy, kills a
worker, retries failed work, or treats observation expiry as process termination.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

if __package__:
    from .prepare_followup_campaign import NPZ_NAME, NPZ_SHA
else:
    from prepare_followup_campaign import NPZ_NAME, NPZ_SHA

JOBS = {
    'formal_cifar_current_epochs20_v1': (0, 'cifar224', 'current_adapted', 'epochs20'),
    'formal_imagenetr_current_epochs20_v1': (1, 'imagenetr', 'current_adapted', 'epochs20'),
    'formal_cifar_tuna_native_v1': (0, 'cifar224', 'tuna', 'pilot_native_same_protocol'),
    'formal_imagenetr_tuna_native_v1': (1, 'imagenetr', 'tuna', 'pilot_native_same_protocol'),
}
ORDER = {
    '0': ['formal_cifar_current_epochs20_v1', 'formal_cifar_tuna_native_v1'],
    '1': ['formal_imagenetr_current_epochs20_v1', 'formal_imagenetr_tuna_native_v1'],
}
DEAD = {'Z', 'X', 'x'}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def probe_process(pid):
    path = Path('/proc') / str(pid)
    try:
        text = (path / 'stat').read_text()
        before = text[text.rfind(')') + 2:].split()
        command = [x.decode() for x in (path / 'cmdline').read_bytes().split(b'\0') if x]
        text = (path / 'stat').read_text()
        after = text[text.rfind(')') + 2:].split()
    except (FileNotFoundError, ProcessLookupError):
        return None
    need(before[19] == after[19], 'PID changed during observation')
    return {'startticks': int(after[19]), 'state': after[0], 'command': command,
            'exit_code': os.waitstatus_to_exitcode(int(after[49])) if after[0] in DEAD else None}


def gpu_snapshot():
    import pynvml as nv
    nv.nvmlInit()
    try:
        result = {}
        for gpu in (0, 1):
            handle = nv.nvmlDeviceGetHandleByIndex(gpu)
            memory = nv.nvmlDeviceGetMemoryInfo(handle)
            pids = {p.pid for p in nv.nvmlDeviceGetComputeRunningProcesses(handle)}
            pids.update(p.pid for p in nv.nvmlDeviceGetGraphicsRunningProcesses(handle))
            result[gpu] = {'free_mib': memory.free / 2**20,
                           'total_mib': memory.total / 2**20, 'pids': sorted(pids)}
        return result
    finally:
        nv.nvmlShutdown()


class FollowupQueue:
    def __init__(self, manifest, *, probe=probe_process, memory=gpu_snapshot,
                 disk=shutil.disk_usage, spawn=subprocess.Popen):
        self.path = Path(manifest).resolve()
        self.manifest = read(self.path)
        self.manifest_sha = sha(self.path)
        self.root = Path(self.manifest['campaign_root'])
        self.repo = Path(self.manifest['repo'])
        self.state = self.root / 'queue'
        self.formal = self.root / 'formal'
        self.probe, self.memory, self.disk, self.spawn = probe, memory, disk, spawn
        self.locked = False
        self.handles = {}
        self.entries = {row['job']: row for row in self.manifest['jobs']}
        self.source_inventory_sha = hashlib.sha256(json.dumps(
            self.manifest['source_files_sha256'], sort_keys=True).encode()).hexdigest()
        need(len(self.entries) == len(self.manifest['jobs']) == 4 and set(self.entries) == set(JOBS),
             'Manifest must predeclare exactly the four requested jobs')
        need(self.manifest['queue'] == ORDER, 'Queue order or physical GPU assignment changed')
        need(self.path == self.root / 'manifest.json', 'Manifest is outside its campaign')
        self.validate()
        self.check_cache()

    def validate(self):
        need(sha(self.path) == self.manifest_sha, 'Manifest changed after admission')
        need(self.manifest['version'] == 1 and self.repo.is_dir(), 'Invalid prepared campaign')
        need(self.root.is_absolute() and self.repo.is_absolute()
             and Path(self.manifest['runtime_root']).is_absolute()
             and Path(self.manifest['python']).is_absolute()
             and Path(self.manifest['python']).is_file(), 'Runtime paths must be absolute and present')
        need(os.environ.get('CUDA_VISIBLE_DEVICES') in (None, '0,1'), 'Physical GPU remapping is unsupported')
        runtime = Path(self.manifest['runtime_root'])
        need(self.manifest['environment'] == dict(
            TORCH_HOME=str(runtime / 'torch'), OMP_NUM_THREADS='4',
            OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4', PYTHONUNBUFFERED='1',
            CUBLAS_WORKSPACE_CONFIG=':4096:8'), 'Prepared runtime environment differs')
        for name, digest in self.manifest['source_files_sha256'].items():
            path = (self.repo / name).resolve()
            need(path.is_relative_to(self.repo.resolve()) and sha(path) == digest,
                 'Pinned execution source changed: ' + name)
        for job, row in self.entries.items():
            gpu, dataset, method, variant = JOBS[job]
            path = Path(row['config']['path'])
            need(path == self.formal / 'configs' / (job + '.json') and sha(path) == row['config']['sha256'],
                 'Pinned job configuration changed: ' + job)
            cfg = read(path)
            need((row['gpu'], row['dataset'], row['method'], row['comparison_variant'])
                 == (gpu, dataset, method, variant), 'Manifest job identity differs')
            need((cfg['model_name'], cfg['dataset'], cfg['comparison_variant'], cfg['device'])
                 == (method, dataset, variant, [gpu]) and cfg['seed'] == [1993]
                 and cfg['phase'] == 'final' and cfg['validation_only'] is False,
                 'Prepared formal identity differs')
            width = 10 if dataset == 'cifar224' else 20
            need((cfg['init_cls'], cfg['increment']) == (width, width)
                 and cfg['cl27_protocol'] == 'pilot_figure2'
                 and cfg['download'] is False and cfg['pretrained'] is True
                 and cfg['save_checkpoints'] is True and cfg['evaluation_head'] == 'cnn',
                 'Incomplete or wrong formal protocol')
            expected_data = runtime / 'data' / ('cifar100' if dataset == 'cifar224' else 'imagenetr')
            need(cfg['data_root'] == str(expected_data) and expected_data.is_dir(),
                 'Prepared dataset path differs or is missing')
            need(not any(cfg.get(k) is not None for k in ('resume', 'max_tasks', 'validation_max_per_class', 'joint_max_classes')),
                 'Partial or resumed job in follow-up queue')
            need(cfg['output_dir'] == row['output_dir'] == str(self.formal / 'runs' / job),
                 'Prepared output path differs')
            need(row['command'] == [self.manifest['python'], str(self.repo / 'main.py'), '--config', str(path)],
                 'Prepared command differs')
            expected_env = {'CL27_CURRENT_FROZEN_PREFIX_REUSE': '1'} if method == 'current_adapted' else {}
            need(row['environment'] == expected_env, 'Unexpected per-job execution switch')
        need(self.manifest['minimum_initial_free_bytes'] >= 13 * 2**30, 'Initial disk budget reduced')

    def check_cache(self):
        """Admit only the existing exact checkpoint; never trigger a download."""
        expected = Path(self.manifest['runtime_root']) / 'torch/hub/checkpoints' / NPZ_NAME
        need(self.manifest['pretrained_npz'] == dict(path=str(expected), sha256=NPZ_SHA)
             and expected.is_file() and sha(expected) == NPZ_SHA,
             'Missing or changed prebound pretrained cache; no download')

    def receipt_path(self, job):
        return self.formal / 'receipts' / (job + '.json')

    def intent_path(self, job):
        return self.state / (job + '.intent.json')

    def check_receipt(self, job, receipt):
        row = self.entries[job]
        need(receipt['job'] == job and receipt['manifest_sha256'] == self.manifest_sha
             and receipt['config_sha256'] == row['config']['sha256']
             and receipt['cwd'] == str(self.repo) and receipt['command'] == row['command']
             and receipt['gpu'] == row['gpu'] and receipt['environment'] == self.environment(row)
             and receipt['source_commit'] == self.manifest.get('source_commit')
             and receipt['source_inventory_sha256'] == self.source_inventory_sha,
             'Existing worker receipt differs from the prebound job')
        need(type(receipt['pid']) is int and receipt['pid'] > 0
             and type(receipt['proc_start_ticks']) is int and receipt['proc_start_ticks'] > 0,
             'Worker process identity is invalid')

    def live(self, job, receipt):
        info = self.probe(receipt['pid'])
        if info is None:
            return False, 'absent', None
        if info['startticks'] != receipt['proc_start_ticks']:
            return False, 'original_pid_reused', None
        if info['state'] in DEAD:
            need(info['exit_code'] in (None, 0), 'Worker exited nonzero: ' + job)
            handle = self.handles.get(job)
            if handle is not None:
                # Reap owned terminal children only after observing their
                # original Linux process identity and exit status.
                code = handle.poll()
                need(code in (None, 0), 'Worker exited nonzero: ' + job)
            return False, info['state'], info['exit_code']
        # Linux may clear argv while the same original process is shutting down.
        if info['command']:
            need(info['command'] == receipt['command'], 'Same-PID worker command changed: ' + job)
        return True, info['state'], None

    def complete(self, job):
        row = self.entries[job]
        root = Path(row['output_dir'])
        result = root / 'results.json'
        need(result.is_file(), 'Terminated worker has no complete result: ' + job)
        value, cfg = read(result), read(root / 'config.json')
        declared = read(row['config']['path'])
        expected_cfg = dict(declared, seed=1993)
        # main.py adds its config-file argument; this does not affect training.
        actual_cfg = {k: v for k, v in cfg.items() if k != 'config'}
        need(actual_cfg == expected_cfg, 'Saved worker config differs: ' + job)
        need(value['status'] == 'complete' and value['phase'] == 'final'
             and value['method'] == row['method'] and value['metrics']['completed_tasks'] == 10
             and len(value['stages']) == 10 and [s['task'] for s in value['stages']] == list(range(10)),
             'Worker did not finish the full stream: ' + job)
        width, evaluation = (10, 10000) if row['dataset'] == 'cifar224' else (20, 6000)
        need([s['seen_classes'] for s in value['stages']] == [width * (i + 1) for i in range(10)]
             and all(s['prediction_space'] == 'all_seen_classes_without_task_id' for s in value['stages'])
             and value['stages'][-1]['evaluation_images'] == evaluation,
             'Wrong complete all-class evaluation: ' + job)
        need((root / 'checkpoint.pt').is_file(), 'Complete result lacks final checkpoint: ' + job)
        return {'result_sha256': sha(result), 'config_sha256': sha(root / 'config.json'),
                'data_manifest_sha256': sha(root / 'data_manifest.json')}

    def environment(self, row):
        result = dict(self.manifest['environment'])
        result.update(row['environment'])
        return result

    def observe(self):
        statuses, identities = {}, {}
        for job in self.entries:
            receipt_file = self.receipt_path(job)
            if receipt_file.exists():
                receipt = read(receipt_file)
                self.check_receipt(job, receipt)
                live, state, code = self.live(job, receipt)
                if live:
                    statuses[job] = 'running'
                    identities[job] = receipt
                else:
                    evidence = self.complete(job)
                    statuses[job] = 'complete'
                    if self.locked:
                        seal = self.state / (job + '.complete.json')
                        if not seal.exists():
                            write(seal, dict(job=job, utc=utc(), original_process_state=state,
                                             observed_exit_code=code, **evidence))
                        else:
                            previous = read(seal)
                            need(all(previous[k] == v for k, v in evidence.items()), 'Completed output changed: ' + job)
            elif self.intent_path(job).exists():
                raise ValueError('Unresolved launch intent; never retry blindly: ' + job)
            else:
                output = Path(self.entries[job]['output_dir'])
                log = self.formal / 'logs' / (job + '.log')
                need(not output.exists() and not log.exists(), 'Unreceipted output/log blocks launch: ' + job)
                statuses[job] = 'waiting'
        return statuses, identities

    def launch(self, job):
        need(self.locked, 'Launching requires the exclusive dispatcher lock')
        self.check_cache()
        row = self.entries[job]
        for directory in ('receipts', 'logs'):
            (self.formal / directory).mkdir(parents=True, exist_ok=True)
        intent = self.intent_path(job)
        with intent.open('x') as stream:
            json.dump(dict(job=job, manifest_sha256=self.manifest_sha,
                           config_sha256=row['config']['sha256'], utc=utc()), stream)
            stream.flush()
            os.fsync(stream.fileno())
        environment = os.environ.copy()
        environment.update(self.environment(row))
        with (self.formal / 'logs' / (job + '.log')).open('xb') as log:
            worker = self.spawn(row['command'], cwd=self.repo, env=environment,
                                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
        self.handles[job] = worker
        info = self.probe(worker.pid)
        need(info is not None and info['state'] not in DEAD
             and info['command'] in ([], row['command']), 'New process identity handshake failed; do not retry')
        receipt = dict(job=job, pid=worker.pid, proc_start_ticks=info['startticks'],
                       launch_utc=utc(), cwd=str(self.repo), command=row['command'],
                       environment=self.environment(row), gpu=row['gpu'],
                       manifest_sha256=self.manifest_sha, config_sha256=row['config']['sha256'],
                       source_commit=self.manifest.get('source_commit'),
                       source_inventory_sha256=self.source_inventory_sha)
        with self.receipt_path(job).open('x') as stream:
            json.dump(receipt, stream, indent=2)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        return receipt

    def cycle(self, execute=False):
        need(not execute or self.locked, 'Execute requires exclusive lock')
        self.validate()
        if execute:
            need(not (self.state / 'failure.json').exists(), 'Recorded failure requires review; no automatic retry')
        statuses, identities = self.observe()
        snapshot = self.memory()
        any_started = bool(identities) or any(v == 'complete' for v in statuses.values())
        free = self.disk(self.root).free
        if not any_started:
            need(free >= self.manifest['minimum_initial_free_bytes'], 'Insufficient initial disk budget')
        actions = []
        for gpu in (0, 1):
            ordered = ORDER[str(gpu)]
            known = {r['pid'] for r in identities.values() if r['gpu'] == gpu}
            observed = set(snapshot[gpu]['pids'])
            need(not any(r['pid'] in set(snapshot[1 - gpu]['pids']) for r in identities.values() if r['gpu'] == gpu),
                 'Known worker appears on another physical GPU')
            if known:
                need(len(known) == 1, 'Two owned workers occupy the same GPU')
                continue
            next_job = next((job for job in ordered if statuses[job] != 'complete'), None)
            if next_job is None:
                continue
            need(statuses[next_job] == 'waiting', 'Nonwaiting job cannot be admitted')
            if observed:
                actions.append({'job': next_job, 'gpu': gpu, 'state': 'waiting_unknown_gpu_process'})
                continue
            minimum_mib = 8192 if self.entries[next_job]['method'] == 'current_adapted' else 12288
            if snapshot[gpu]['free_mib'] < minimum_mib + 1024:
                actions.append({'job': next_job, 'gpu': gpu, 'state': 'waiting_gpu_memory'})
                continue
            need(free >= 4 * 2**30, 'Insufficient atomic-save disk margin for next job')
            actions.append({'job': next_job, 'gpu': gpu, 'state': 'ready'})
            if execute:
                receipt = self.launch(next_job)
                identities[next_job] = receipt
                statuses[next_job] = 'running'
                actions[-1]['state'] = 'launched'
        report = {'utc': utc(), 'manifest_sha256': self.manifest_sha,
                  'statuses': statuses, 'actions': actions, 'all_complete': all(x == 'complete' for x in statuses.values()),
                  'disk_free_bytes': free, 'gpu_snapshot': snapshot}
        if execute:
            write(self.state / 'latest.json', report)
        return report

    @contextmanager
    def exclusive(self):
        self.state.mkdir(parents=True, exist_ok=True)
        with (self.state / 'dispatcher.lock').open('a') as stream:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.locked = True
            try:
                yield
            finally:
                self.locked = False
                fcntl.flock(stream, fcntl.LOCK_UN)

    def run(self, poll_seconds=30):
        need(1 <= poll_seconds <= 60, 'Polling interval must be between1 and60 seconds')
        with self.exclusive():
            try:
                while True:
                    report = self.cycle(True)
                    print(json.dumps(report, allow_nan=False), flush=True)
                    if report['all_complete']:
                        return
                    time.sleep(poll_seconds)
            except Exception as error:
                write(self.state / 'failure.json', {'utc': utc(), 'type': type(error).__name__, 'message': str(error)})
                raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--plan', action='store_true', help='Read-only admission/ownership report (default)')
    mode.add_argument('--execute', action='store_true')
    p.add_argument('--poll-seconds', type=float, default=30)
    a = p.parse_args()
    queue = FollowupQueue(a.manifest)
    if a.execute:
        queue.run(a.poll_seconds)
    else:
        print(json.dumps(queue.cycle(False), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
