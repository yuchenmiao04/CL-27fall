"""Task-boundary recovery contracts with tiny CPU checkpoints; no training."""
import copy
import json
from pathlib import Path

import pytest
import torch

from cl27 import runner
from scripts import resume_followup_after_reboot as recovery
from scripts import run_followup_queue as queue
from scripts.run_followup_partition import PartitionController
from test_followup_queue import Fixture, save, files, CF20, R20, CFTUNA, RTUNA


def fixture(root, monkeypatch, partition='current20'):
    f = Fixture(root, monkeypatch)
    base = f.create()
    if partition == 'current20':
        with base.exclusive():
            base.cycle(True)
    else:
        with base.exclusive():
            PartitionController(f.path, 'tuna', base=base).cycle(True)
    for job in recovery.GROUPS[partition]:
        row = base.entries[job]
        cfg = queue.read(row['config']['path'])
        output = Path(row['output_dir'])
        data = {'fingerprint': recovery.DATA_SHA[row['dataset']], 'phase': 'final', 'rows': 123}
        prefix = [{'task': i, 'seen_classes': (i + 1) * cfg['increment'], 'wall_seconds': 3.25,
                   'accuracy': .73, 'heads': {}, 'evaluation_images': 123,
                   'prediction_space': 'all_seen_classes_without_task_id'} for i in range(2)]
        save(output / 'config.json', dict(cfg, seed=1993, config=row['config']['path']))
        save(output / 'data_manifest.json', data)
        save(output / 'progress.json', {'status': 'running', 'stages': prefix})
        state = {'version': 1, 'random_stream_policy': recovery.RNG_POLICY,
                 'config_fingerprint': runner.configuration_fingerprint(dict(cfg, seed=1993)),
                 'data_fingerprint': data['fingerprint'], 'stages': prefix,
                 'learner': {'version': 1, 'method': row['method'], 'cur_task': 1,
                             'known_classes': 2 * cfg['increment'], 'total_classes': 2 * cfg['increment'],
                             'task_diagnostics': [{}, {}], 'core': {}},
                 'rng': {'python': (), 'numpy': (), 'torch': torch.zeros(4, dtype=torch.uint8),
                         'cuda': [torch.zeros(4, dtype=torch.uint8), torch.ones(4, dtype=torch.uint8)]}}
        torch.save(state, output / 'checkpoint.pt')
    f.probes.clear()
    for snapshot in f.snapshot.values():
        snapshot['pids'] = []
    class DataManager:
        def __init__(self, dataset, *args):
            self.manifest = {'fingerprint': recovery.DATA_SHA[dataset], 'phase': 'final', 'rows': 123}
    monkeypatch.setattr(recovery, '_context', lambda path, role: (base, queue, runner, DataManager))
    return f, base


@pytest.mark.parametrize('partition', ['current20', 'tuna'])
def test_prepare_retains_science_rng_and_exact_prefix_in_immutable_snapshot(tmp_path, monkeypatch, partition):
    f, base = fixture(tmp_path, monkeypatch, partition)
    old = files(f.campaign)
    path = recovery.prepare(f.path, partition, tmp_path / 'recovery')
    admitted = recovery.validate_recovery_manifest(path)
    assert files(f.campaign) == old
    assert len(admitted['manifest']['jobs']) == 2
    for row in admitted['manifest']['jobs']:
        cfg = admitted['configs'][row['job']]
        original = queue.read(base.entries[row['job']]['config']['path'])
        expected = dict(original, resume=row['lineage']['resume_checkpoint']['path'], output_dir=row['output_dir'])
        assert cfg == expected and cfg['seed'] == [1993]
        assert runner.configuration_fingerprint(dict(cfg, seed=1993)) == row['lineage']['config_fingerprint']
        snapshot = Path(cfg['resume'])
        assert snapshot.stat().st_mode & 0o222 == 0
        state = torch.load(snapshot, weights_only=False)
        assert state['stages'] == row['lineage']['prefix_stages']
        assert row['lineage']['prefix_stages_sha256'] == recovery.canonical_sha(state['stages'])
        assert row['lineage']['source_checkpoint']['sha256'] == recovery.sha(snapshot)
    with pytest.raises(ValueError, match='existing recovery'):
        recovery.prepare(f.path, partition, tmp_path / 'recovery')


def test_live_original_and_partial_checkpoint_never_create_recovery(tmp_path, monkeypatch):
    f, base = fixture(tmp_path, monkeypatch)
    receipt = queue.read(base.receipt_path(CF20))
    f.probes[receipt['pid']] = dict(startticks=receipt['proc_start_ticks'], state='R', command=[], exit_code=None)
    with pytest.raises(ValueError, match='still live'):
        recovery.prepare(f.path, 'current20', tmp_path / 'recovery')
    assert not (tmp_path / 'recovery').exists()
    f.probes.clear()
    partial = Path(base.entries[CF20]['output_dir']) / 'checkpoint.tmp'
    partial.write_bytes(b'incomplete save')
    with pytest.raises(ValueError, match='Partial-save'):
        recovery.prepare(f.path, 'current20', tmp_path / 'recovery')


def test_checkpoint_counter_mismatch_and_scientific_drift_rejected(tmp_path, monkeypatch):
    f, base = fixture(tmp_path, monkeypatch)
    cp = Path(base.entries[CF20]['output_dir']) / 'checkpoint.pt'
    state = torch.load(cp, weights_only=False)
    state['learner']['known_classes'] = 10
    torch.save(state, cp)
    with pytest.raises(ValueError, match='counters disagree'):
        recovery.prepare(f.path, 'current20', tmp_path / 'recovery')
    state['learner']['known_classes'] = state['learner']['total_classes']
    torch.save(state, cp)
    path = recovery.prepare(f.path, 'current20', tmp_path / 'recovery')
    manifest = queue.read(path)
    row = manifest['jobs'][0]
    cfg = queue.read(row['config']['path'])
    cfg['epochs'] = 1
    save(Path(row['config']['path']), cfg)
    row['config']['sha256'] = recovery.sha(row['config']['path'])
    save(path, manifest)
    with pytest.raises(ValueError, match='Only resume/output_dir'):
        recovery.validate_recovery_manifest(path)


def test_manual_standard_receipts_adopt_and_only_terminal_exact_prefix_can_seal(tmp_path, monkeypatch):
    f, _ = fixture(tmp_path, monkeypatch)
    path = recovery.prepare(f.path, 'current20', tmp_path / 'recovery')
    q, _ = recovery.recovery_queue(path)
    # Match the root operator's native launch contract, before observer startup.
    with q.exclusive():
        for job in (CF20, R20):
            q.launch(job)
    controller = recovery.RecoveryController(path)
    with controller.q.exclusive():
        report = controller.cycle(True)
    assert set(report['statuses'].values()) == {'running'}
    assert len(f.calls) == 4  # Two original stopped workers and two resumed.
    for job in (CF20, R20):
        f.finish(controller.q, job)
        root = Path(controller.q.entries[job]['output_dir'])
        lineage = controller.q.entries[job]['lineage']
        result = queue.read(root / 'results.json')
        result['stages'][:2] = copy.deepcopy(lineage['prefix_stages'])
        result.update(config_fingerprint=lineage['config_fingerprint'], data_fingerprint=lineage['data_fingerprint'])
        save(root / 'results.json', result)
        save(root / 'data_manifest.json', queue.read(lineage['original_data_manifest']['path']))
    # A still-live empty-argv PID remains reserved even with complete results.
    receipt = queue.read(controller.q.receipt_path(CF20))
    f.probes[receipt['pid']].update(state='R', command=[], exit_code=None)
    with controller.q.exclusive():
        report = controller.cycle(True)
    assert report['statuses'][CF20] == 'running' and not report['all_complete']
    f.probes[receipt['pid']].update(state='Z', exit_code=0)
    with controller.q.exclusive():
        report = controller.cycle(True)
    assert report['all_complete'] and len(f.calls) == 4
    seal = queue.read(controller.q.state / (CF20 + '.complete.json'))
    assert seal['recovery_manifest_sha256'] == recovery.sha(path)
    assert seal['lineage_sha256'] == recovery.canonical_sha(controller.q.entries[CF20]['lineage'])


def test_recovery_observer_refuses_changed_saved_prefix(tmp_path, monkeypatch):
    f, _ = fixture(tmp_path, monkeypatch)
    path = recovery.prepare(f.path, 'current20', tmp_path / 'recovery')
    c = recovery.RecoveryController(path)
    with c.q.exclusive():
        c.cycle(True)
    f.finish(c.q, CF20)
    with c.q.exclusive(), pytest.raises(ValueError, match='changed the saved checkpoint prefix'):
        c.cycle(True)
    assert len(f.calls) == 4
