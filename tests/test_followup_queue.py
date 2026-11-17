"""Follow-up admission and process-ownership contracts; no CUDA or training."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import prepare_followup_campaign as prepare
from scripts import run_followup_queue as queue


REPO = Path(__file__).resolve().parents[1]
CF20, R20, CFTUNA, RTUNA = tuple(queue.JOBS)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def files(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file()}


class Fixture:
    def __init__(self, root, monkeypatch):
        self.root, self.repo = root, root / 'repo'
        self.repo.mkdir()
        (self.repo / 'main.py').write_text('# never executed\n')
        for path in ('data/cifar100', 'data/imagenetr', 'torch/hub/checkpoints'):
            (root / path).mkdir(parents=True)
        self.cache = root / 'torch/hub/checkpoints' / prepare.NPZ_NAME
        self.cache.write_bytes(b'existing checkpoint fixture')
        monkeypatch.setattr(prepare, 'NPZ_SHA', prepare.sha(self.cache))
        monkeypatch.setattr(queue, 'NPZ_SHA', prepare.sha(self.cache))
        monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
        self.python = root / 'venv/bin/python'
        self.python.parent.mkdir(parents=True)
        self.python.symlink_to(REPO.parent / '.venv-pilot/bin/python')
        self.campaign = root / 'extension'
        self.probes, self.calls = {}, []
        self.free = 20 * 2**30
        self.snapshot = {gpu: dict(free_mib=24000, total_mib=24576, pids=[]) for gpu in (0, 1)}
        entries = []
        for job, (gpu, dataset, method, variant) in queue.JOBS.items():
            cfg = json.loads((REPO / 'experiments/extension_20261117/configs' / (job + '.json')).read_text())
            cfg['data_root'] = str(root / 'data' / ('cifar100' if dataset == 'cifar224' else 'imagenetr'))
            cfg['output_dir'] = str(self.campaign / 'formal/runs' / job)
            path = self.campaign / 'formal/configs' / (job + '.json')
            save(path, cfg)
            entries.append(dict(job=job, gpu=gpu, dataset=dataset, method=method,
                comparison_variant=variant, config=dict(path=str(path), sha256=queue.sha(path)),
                output_dir=cfg['output_dir'],
                command=[str(self.python), str(self.repo / 'main.py'), '--config', str(path)],
                environment={'CL27_CURRENT_FROZEN_PREFIX_REUSE': '1'} if method == 'current_adapted' else {}))
        self.path = self.campaign / 'manifest.json'
        save(self.path, dict(version=1, runtime_root=str(root), repo=str(self.repo),
            python=str(self.python), campaign_root=str(self.campaign), source_commit='fixture',
            source_files_sha256={'main.py': queue.sha(self.repo / 'main.py')},
            pretrained_npz=dict(path=str(self.cache), sha256=queue.NPZ_SHA),
            minimum_initial_free_bytes=13 * 2**30, jobs=entries, queue=queue.ORDER,
            environment=dict(TORCH_HOME=str(root / 'torch'), OMP_NUM_THREADS='4',
                OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4', PYTHONUNBUFFERED='1',
                CUBLAS_WORKSPACE_CONFIG=':4096:8')))

    def create(self):
        return queue.FollowupQueue(self.path, probe=lambda pid: self.probes.get(pid),
            memory=lambda: copy.deepcopy(self.snapshot), disk=lambda _: SimpleNamespace(free=self.free),
            spawn=self.spawn)

    def spawn(self, command, **kwargs):
        pid = 200 + len(self.calls)
        cfg = json.loads(Path(command[-1]).read_text())
        self.calls.append((command, kwargs))
        self.probes[pid] = dict(startticks=pid * 100, state='R', command=command, exit_code=None)
        self.snapshot[cfg['device'][0]]['pids'] = [pid]
        return SimpleNamespace(pid=pid, poll=lambda: self.probes[pid]['exit_code'])

    def finish(self, q, job, *, terminal=True, exit_code=0, stages=10):
        receipt = queue.read(q.receipt_path(job))
        cfg = queue.read(q.entries[job]['config']['path'])
        root = Path(cfg['output_dir'])
        width, evaluation = (10, 10000) if cfg['dataset'] == 'cifar224' else (20, 6000)
        save(root / 'config.json', dict(cfg, seed=1993, config=q.entries[job]['config']['path']))
        save(root / 'data_manifest.json', {'fingerprint': 'same prepared data'})
        save(root / 'results.json', dict(status='complete', phase='final', method=cfg['model_name'],
            metrics={'completed_tasks': stages}, stages=[dict(task=i, seen_classes=width * (i + 1),
            evaluation_images=evaluation, prediction_space='all_seen_classes_without_task_id') for i in range(stages)]))
        (root / 'checkpoint.pt').write_bytes(b'server-local checkpoint fixture')
        if terminal:
            self.probes[receipt['pid']].update(state='Z', command=[], exit_code=exit_code)
            self.snapshot[receipt['gpu']]['pids'] = []


def test_plan_is_read_only_and_starts_only_the_two_long_jobs(tmp_path, monkeypatch):
    fixture = Fixture(tmp_path, monkeypatch)
    q = fixture.create()
    before = files(tmp_path)
    report = q.cycle()
    assert [(x['job'], x['gpu'], x['state']) for x in report['actions']] == [
        (CF20, 0, 'ready'), (R20, 1, 'ready')]
    assert files(tmp_path) == before
    assert fixture.calls == []


def test_dispatcher_restart_adopts_workers_and_empty_argv_stays_reserved(tmp_path, monkeypatch):
    fixture = Fixture(tmp_path, monkeypatch)
    first = fixture.create()
    with first.exclusive():
        first.cycle(True)
    assert len(fixture.calls) == 2
    restarted = fixture.create()
    pid = queue.read(restarted.receipt_path(CF20))['pid']
    fixture.probes[pid]['command'] = []
    fixture.finish(restarted, CF20, terminal=False)
    with restarted.exclusive():
        report = restarted.cycle(True)
    assert report['statuses'][CF20] == 'running'
    assert len(fixture.calls) == 2  # Even a complete JSON cannot advance a live PID.
    fixture.finish(restarted, CF20)
    with restarted.exclusive():
        report = restarted.cycle(True)
    assert report['statuses'][CF20] == 'complete'
    assert report['statuses'][CFTUNA] == 'running'
    assert report['statuses'][RTUNA] == 'waiting'
    assert len(fixture.calls) == 3
    command, kwargs = fixture.calls[-1]
    assert command == restarted.entries[CFTUNA]['command']
    assert kwargs['start_new_session'] is True
    assert kwargs['env']['TORCH_HOME'] == str(tmp_path / 'torch')


@pytest.mark.parametrize('failure', ['nonzero', 'partial', 'missing_result', 'lost_receipt'])
def test_failed_or_ambiguous_worker_never_retries_or_advances(tmp_path, monkeypatch, failure):
    fixture = Fixture(tmp_path, monkeypatch)
    q = fixture.create()
    with q.exclusive():
        q.cycle(True)
    fixture.finish(q, CF20, exit_code=1 if failure == 'nonzero' else 0,
                   stages=9 if failure == 'partial' else 10)
    if failure == 'missing_result':
        (Path(q.entries[CF20]['output_dir']) / 'results.json').unlink()
    elif failure == 'lost_receipt':
        q.receipt_path(CF20).unlink()
    with q.exclusive(), pytest.raises(ValueError):
        q.cycle(True)
    assert len(fixture.calls) == 2


def test_admission_rejects_changed_sources_config_cache_and_low_disk(tmp_path, monkeypatch):
    fixture = Fixture(tmp_path, monkeypatch)
    q = fixture.create()
    fixture.free = 13 * 2**30 - 1
    with q.exclusive(), pytest.raises(ValueError, match='disk budget'):
        q.cycle(True)
    assert fixture.calls == []
    fixture.free = 20 * 2**30
    source = fixture.repo / 'main.py'
    old = source.read_bytes()
    source.write_bytes(old + b'# changed')
    with pytest.raises(ValueError, match='source changed'):
        fixture.create()
    source.write_bytes(old)
    config = Path(q.entries[CF20]['config']['path'])
    old = config.read_bytes()
    config.write_bytes(old + b' ')
    with pytest.raises(ValueError, match='configuration changed'):
        fixture.create()
    config.write_bytes(old)
    fixture.cache.write_bytes(b'other weights')
    with pytest.raises(ValueError, match='pretrained cache'):
        fixture.create()


def test_unknown_gpu_occupant_blocks_that_card_only_and_receipts_bind_source(tmp_path, monkeypatch):
    fixture = Fixture(tmp_path, monkeypatch)
    fixture.snapshot[0]['pids'] = [999]
    q = fixture.create()
    with q.exclusive():
        report = q.cycle(True)
    assert report['actions'][0]['state'] == 'waiting_unknown_gpu_process'
    assert report['statuses'][R20] == 'running' and len(fixture.calls) == 1
    receipt = queue.read(q.receipt_path(R20))
    receipt['source_inventory_sha256'] = 'different sources'
    save(q.receipt_path(R20), receipt)
    with pytest.raises(ValueError, match='receipt differs'):
        q.cycle()


def test_observation_interruption_does_not_mark_workers_failed(tmp_path, monkeypatch):
    fixture = Fixture(tmp_path, monkeypatch)
    q = fixture.create()
    def interrupt(_):
        raise KeyboardInterrupt()
    monkeypatch.setattr(queue.time, 'sleep', interrupt)
    with pytest.raises(KeyboardInterrupt):
        q.run()
    assert not (q.state / 'failure.json').exists()
    with fixture.create().exclusive():
        pass
    restarted = fixture.create()
    with restarted.exclusive():
        report = restarted.cycle(True)
    assert len(fixture.calls) == 2 and set(report['statuses'].values()) == {'running', 'waiting'}


def test_preparation_preserves_python_symlink_and_refuses_overwrite(tmp_path, monkeypatch):
    fixture = Fixture(tmp_path, monkeypatch)
    # Use real immutable recipe checks, but the tiny fixture's source inventory.
    monkeypatch.setattr(prepare, 'source_inventory', lambda _: {'main.py': queue.sha(fixture.repo / 'main.py')})
    target = tmp_path / 'new_campaign'
    manifest = prepare.prepare(tmp_path, REPO, fixture.python, target, 'fixture')
    value = queue.read(manifest)
    assert value['python'] == str(fixture.python)  # Do not resolve a venv symlink.
    assert value['jobs'][0]['config']['sha256'] == queue.sha(value['jobs'][0]['config']['path'])
    before = files(target)
    with pytest.raises(ValueError, match='existing campaign'):
        prepare.prepare(tmp_path, REPO, fixture.python, target, 'fixture')
    assert files(target) == before
    for job, _, dataset, method in prepare.JOBS:
        cfg = queue.read(REPO / 'experiments/extension_20261117/configs' / (job + '.json'))
        cfg['module_lr' if method == 'current_adapted' else 'reg'] = -1
        with pytest.raises(ValueError, match='recipe changed|hyperparameter changed'):
            prepare.check_template(REPO, job, cfg, dataset, method)
