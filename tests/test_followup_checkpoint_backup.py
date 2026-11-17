"""Follow-up backup ownership, no-overwrite and exact-restore audit contracts."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch

from scripts import backup_followup_checkpoint as backup
from scripts import compare_checkpoint_state as comparator

REPO = Path(__file__).resolve().parents[1]
JOB = 'formal_cifar_current_epochs20_v1'


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def file_bytes(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file()}


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    repo, campaign = tmp_path / 'repo', tmp_path / 'campaign'
    repo.mkdir()
    (repo / 'main.py').write_text('# source inventory fixture; never executed\n')
    inventory = {'main.py': backup.sha(repo / 'main.py')}
    npz = tmp_path / 'torch/hub/checkpoints' / backup.WEIGHT_NAME
    npz.parent.mkdir(parents=True)
    npz.write_bytes(b'cached fixture')
    monkeypatch.setattr(backup, 'WEIGHT_SHA', backup.sha(npz))
    config_path = campaign / 'formal/configs' / (JOB + '.json')
    run_dir = campaign / 'formal/runs' / JOB
    cfg = json.loads((REPO / 'experiments/extension_20261117/configs' / (JOB + '.json')).read_text())
    cfg.update(data_root=str(tmp_path / 'data'), output_dir=str(run_dir))
    save(config_path, cfg)
    executed = dict(cfg, seed=1993, config=str(config_path))
    save(run_dir / 'config.json', executed)
    result = dict(status='complete', method='current_adapted', phase='final',
                  metrics={'completed_tasks': 10},
                  stages=[dict(task=i, seen_classes=10 * (i + 1), evaluation_images=10000,
                               prediction_space='all_seen_classes_without_task_id') for i in range(10)])
    save(run_dir / 'results.json', result)
    save(run_dir / 'data_manifest.json', dict(phase='final', evaluation_source='designated_test',
                                             official_test_read=True, train_count=50000, evaluation_count=10000))
    (run_dir / 'checkpoint.pt').write_bytes(b'trusted checkpoint fixture; never unpickled')
    command = ['/fixture/python', str(repo / 'main.py'), '--config', str(config_path)]
    row = dict(job=JOB, method='current_adapted', dataset='cifar224', comparison_variant='epochs20',
               gpu=0, output_dir=str(run_dir), command=command,
               config=dict(path=str(config_path), sha256=backup.sha(config_path)))
    manifest = campaign / 'manifest.json'
    save(manifest, dict(campaign_root=str(campaign), repo=str(repo), source_commit='fixed',
                        source_files_sha256=inventory, jobs=[row],
                        pretrained_npz=dict(path=str(npz), sha256=backup.WEIGHT_SHA)))
    receipt = campaign / 'formal/receipts' / (JOB + '.json')
    save(receipt, dict(job=JOB, pid=12345, proc_start_ticks=67890, command=command,
                       cwd=str(repo), gpu=0, source_commit='fixed',
                       manifest_sha256=backup.sha(manifest), config_sha256=backup.sha(config_path),
                       source_inventory_sha256=hashlib.sha256(json.dumps(inventory, sort_keys=True).encode()).hexdigest()))
    seal = campaign / 'queue' / (JOB + '.complete.json')
    save(seal, dict(job=JOB, original_process_state='absent', observed_exit_code=None,
                    config_sha256=backup.sha(run_dir / 'config.json'),
                    result_sha256=backup.sha(run_dir / 'results.json'),
                    data_manifest_sha256=backup.sha(run_dir / 'data_manifest.json')))
    return dict(root=tmp_path, repo=repo, campaign=campaign, manifest=manifest,
                receipt=receipt, seal=seal, run_dir=run_dir, npz=npz, config=config_path)


@pytest.mark.parametrize('command', [[], ['expected']])
def test_live_original_process_blocks_even_with_completed_result(fixture, command):
    receipt = backup.read(fixture['receipt'])
    if command:
        command = receipt['command']
    before = file_bytes(fixture['root'])
    with pytest.raises(ValueError, match='still active'):
        backup.admit(fixture['manifest'], JOB, probe=lambda _pid: dict(
            startticks=receipt['proc_start_ticks'], state='R', command=command, exit_code=None))
    assert file_bytes(fixture['root']) == before


def test_terminal_receipt_records_unknown_exit_and_rejects_nonzero_or_wrong_command(fixture):
    receipt = backup.read(fixture['receipt'])
    seal = backup.read(fixture['seal'])
    proof = backup.terminal_process(receipt, probe=lambda _pid: None, seal=seal)
    assert proof['observed_state'] == 'absent'
    assert proof['observed_exit_code'] is None and proof['exit_code_known'] is False
    for code in (1, None):
        with pytest.raises(ValueError, match='successful observed exit'):
            backup.terminal_process(receipt, seal=seal, probe=lambda _pid: dict(
                startticks=receipt['proc_start_ticks'], state='Z', command=[], exit_code=code))
    with pytest.raises(ValueError, match='command differs'):
        backup.terminal_process(receipt, seal=seal, probe=lambda _pid: dict(
            startticks=receipt['proc_start_ticks'], state='Z', command=['different'], exit_code=0))


def test_changed_sealed_metadata_or_source_is_rejected_before_copy(fixture):
    original = (fixture['run_dir'] / 'results.json').read_bytes()
    result = backup.read(fixture['run_dir'] / 'results.json')
    result['extra'] = 'changed after queue completion'
    save(fixture['run_dir'] / 'results.json', result)
    with pytest.raises(ValueError, match='completion metadata changed'):
        backup.admit(fixture['manifest'], JOB, probe=lambda _pid: None)
    (fixture['run_dir'] / 'results.json').write_bytes(original)
    (fixture['repo'] / 'main.py').write_text('# changed scientific entry\n')
    with pytest.raises(ValueError, match='Executed source changed'):
        backup.admit(fixture['manifest'], JOB, probe=lambda _pid: None)


def test_atomic_checkpoint_copy_never_replaces_an_existing_target(tmp_path):
    original, target = tmp_path / 'source.pt', tmp_path / 'backup.pt'
    original.write_bytes(b'new checkpoint')
    target.write_bytes(b'existing checkpoint')
    with pytest.raises(FileExistsError):
        backup.atomic_copy(original, target, backup.sha(original))
    assert target.read_bytes() == b'existing checkpoint'
    assert sorted(p.name for p in tmp_path.iterdir()) == ['backup.pt', 'source.pt']
    target.unlink()
    with pytest.raises(ValueError, match='Source SHA differs'):
        backup.atomic_copy(original, target, 'not-the-source-sha')
    assert not target.exists()
    backup.atomic_copy(original, target, backup.sha(original))
    assert target.read_bytes() == original.read_bytes()


def test_exact_learner_comparison_covers_optimizer_bits_and_module_modes():
    reference = dict(core=dict(optimizers={'module': {'step': torch.tensor(1.), 'exp_avg': torch.tensor([0.])}},
                               module_training={'backbone': False}, requires_grad={'adapter': True}),
                     diagnostics=['native state'])
    report = backup.compare_learner(reference, copy.deepcopy(reference), comparator)
    assert report['exact_typed_bitwise_comparison'] and report['ignored_paths'] == []
    for change in ('optimizer', 'mode', 'qualification', 'string'):
        candidate = copy.deepcopy(reference)
        if change == 'optimizer':
            candidate['core']['optimizers']['module']['exp_avg'][0] = -0.
        elif change == 'mode':
            candidate['core']['module_training']['backbone'] = True
        elif change == 'qualification':
            candidate['core']['requires_grad']['adapter'] = False
        else:
            candidate['diagnostics'][0] = 'changed native state'
        with pytest.raises(ValueError, match='Native learner restore differs'):
            backup.compare_learner(reference, candidate, comparator)


def test_backup_publishes_sha_bound_evidence_and_refuses_repeat(fixture):
    admitted = backup.admit(fixture['manifest'], JOB, probe=lambda _pid: None)
    calls = []
    def restore(admission, checkpoint, loader):
        assert checkpoint.read_bytes() == (fixture['run_dir'] / 'checkpoint.pt').read_bytes()
        assert admission['checkpoint_sha256'] == backup.sha(checkpoint)
        calls.append(checkpoint)
        return dict(native_cpu_restore_executed=True, fixture_backend=True)
    backup_root, evidence_root = fixture['root'] / 'backups', fixture['root'] / 'evidence'
    result = backup.backup(admitted, backup_root, evidence_root, restore=restore, probe=lambda _pid: None)
    assert len(calls) == 1
    assert result['native_restore_verified'] and result['source_process_terminal_verified']
    assert result['formal_checkpoint_evidence'] is True and result['validation_only'] is False
    assert result['checkpoint_sha256'] == result['backup_sha256'] == backup.sha(fixture['run_dir'] / 'checkpoint.pt')
    assert result['source_metadata_sha256'] == {name: backup.sha(fixture['run_dir'] / (name + '.json')) for name in backup.METADATA}
    assert backup.read(evidence_root / (JOB + '.json')) == result
    before = file_bytes(fixture['root'])
    with pytest.raises(ValueError, match='Refusing an existing backup'):
        backup.backup(admitted, backup_root, evidence_root, restore=restore, probe=lambda _pid: None)
    assert file_bytes(fixture['root']) == before


def test_process_or_source_change_during_restore_cannot_publish_evidence(fixture):
    admitted = backup.admit(fixture['manifest'], JOB, probe=lambda _pid: None)
    before = (fixture['run_dir'] / 'checkpoint.pt').read_bytes()
    def restore(*args):
        (fixture['run_dir'] / 'checkpoint.pt').write_bytes(b'changed after terminal admission')
        return {'fixture_backend': True}
    backup_root, evidence_root = fixture['root'] / 'backups', fixture['root'] / 'evidence'
    with pytest.raises(ValueError, match='source files changed'):
        backup.backup(admitted, backup_root, evidence_root, restore=restore, probe=lambda _pid: None)
    assert not (backup_root / JOB).exists()
    assert not (evidence_root / (JOB + '.json')).exists()
    assert not list(backup_root.glob('*.preparing.*'))
    (fixture['run_dir'] / 'checkpoint.pt').write_bytes(before)
    with pytest.raises(ValueError, match='still active'):
        backup.backup(admitted, backup_root, evidence_root, restore=lambda *args: {},
                       probe=lambda _pid: dict(startticks=67890, state='R', command=[], exit_code=None))
    assert not (backup_root / JOB).exists()
    assert not (evidence_root / (JOB + '.json')).exists()


def test_validation_receipt_is_explicit_and_never_becomes_formal_evidence(fixture):
    cfg = backup.read(fixture['config'])
    cfg.update(phase='development', validation_only=True, max_tasks=2, validation_max_per_class=4)
    save(fixture['config'], cfg)
    save(fixture['run_dir'] / 'config.json', dict(cfg, seed=1993, config=str(fixture['config'])))
    result = backup.read(fixture['run_dir'] / 'results.json')
    result.update(phase='development', status='validation_partial', performance_claim=False,
                  stages=result['stages'][:2], metrics={'completed_tasks': 2})
    save(fixture['run_dir'] / 'results.json', result)
    data = backup.read(fixture['run_dir'] / 'data_manifest.json')
    data.update(phase='development', evaluation_source='training_holdout', official_test_read=False)
    save(fixture['run_dir'] / 'data_manifest.json', data)
    receipt = backup.read(fixture['receipt'])
    receipt.update(job='two_stage_validation', start_ticks=receipt.pop('proc_start_ticks'),
                   source_root=receipt.pop('cwd'), config=str(fixture['config']), output_dir=str(fixture['run_dir']))
    save(fixture['receipt'], receipt)
    admitted = backup.admit(None, 'two_stage_validation', validation_only=True,
                            run_dir=fixture['run_dir'], receipt_path=fixture['receipt'],
                            repo=fixture['repo'], pretrained_npz=fixture['npz'], probe=lambda _pid: None)
    result = backup.backup(admitted, fixture['root'] / 'backups', fixture['root'] / 'evidence',
                           validation_only=True, restore=lambda *args: {'fixture_backend': True}, probe=lambda _pid: None)
    assert result['validation_only'] is True and result['formal_checkpoint_evidence'] is False
    with pytest.raises(ValueError, match='Formal backup requires'):
        backup.admit(None, 'two_stage_validation', run_dir=fixture['run_dir'])
