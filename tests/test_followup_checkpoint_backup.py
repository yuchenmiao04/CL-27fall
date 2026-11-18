"""Follow-up backup ownership, no-overwrite and exact-restore audit contracts."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from scripts import backup_followup_checkpoint as backup
from scripts import compare_checkpoint_state as comparator
from cl27.current.engine import ResidualLibrary

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


def test_current_restored_resource_counts_are_partitioned_without_state_changes():
    engine = nn.Module()
    encoder = nn.Module()
    encoder.main = nn.Linear(2, 2, bias=False)
    encoder.anchor = nn.Linear(2, 2, bias=False).double().requires_grad_(False)
    encoder.libraries = nn.ModuleList([ResidualLibrary(width=2, rank=1, count=2)])
    encoder.positions = (0,)
    engine.encoder = encoder
    engine.statistics = SimpleNamespace(G=torch.zeros(3, 3, dtype=torch.float64),
                                       C=torch.zeros(3, 2, dtype=torch.float64))
    engine.W0, engine.V = (torch.zeros(3, 2, dtype=torch.float64) for _ in range(2))
    engine.memory_base = {1: torch.ones(3, dtype=torch.float64), 2: torch.ones(3, dtype=torch.float64)}
    engine.rng = torch.Generator().manual_seed(7)
    engine.kind_rng = torch.Generator().manual_seed(11)
    engine.optimizers = dict(modules=torch.optim.AdamW(encoder.libraries.parameters()),
                             backbone=torch.optim.AdamW(encoder.main.parameters()))
    for optimizer in engine.optimizers.values():
        for group in optimizer.param_groups:
            for parameter in group['params']:
                optimizer.state[parameter] = dict(step=torch.tensor(3.), exp_avg=torch.zeros_like(parameter),
                                                   exp_avg_sq=torch.ones_like(parameter))
    network = nn.Module()
    network.engine = engine
    memory = SimpleNamespace(records=[SimpleNamespace(image=np.zeros((2, 2, 3), dtype=np.uint8)) for _ in range(2)])
    model = SimpleNamespace(_network=network, core=engine, memory=memory)
    before = copy.deepcopy((network.state_dict(), {k: v.state_dict() for k, v in engine.optimizers.items()},
                            engine.rng.get_state(), engine.kind_rng.get_state()))
    measured = backup.restored_resources(model, 'current_adapted')
    neural = measured['neural_network']
    assert neural['resident_nn_parameters'] == 22
    assert neural['resident_nn_parameter_bytes'] == 104
    assert neural['gradient_eligible_nn_parameters'] == 18
    assert neural['gradient_eligible_nn_parameter_bytes'] == 72
    assert neural['registered_nn_buffer_bytes'] == 4
    assert {k: v['parameters'] for k, v in neural['components'].items()} == dict(
        predictive_backbone=4, frozen_anchor=4, module_libraries=14)
    assert measured['architecture'] == dict(library_positions=[0], library_count=1,
                                            experts_per_library=[2], expert_ranks=[1],
                                            learned_expert_count=2, fixed_null_options=1)
    assert measured['optimizer']['persisted_optimizer_state_tensor_bytes'] == 164
    assert measured['optimizer']['per_optimizer']['modules']['state_tensor_bytes'] == 128
    assert measured['optimizer']['per_optimizer']['backbone']['state_tensor_bytes'] == 36
    assert measured['auxiliary']['frozen_anchor_statistics_bytes'] == 120
    assert measured['auxiliary']['analytic_readout_bytes'] == 96
    assert measured['auxiliary']['resident_anchor_feature_cache_bytes'] == 48
    assert measured['replay']['images'] == 2 and measured['replay']['inline_rgb_array_bytes'] == 24
    assert measured['replay']['decoded_rgb_plus_id_label_payload_bytes'] == 56
    after = (network.state_dict(), {k: v.state_dict() for k, v in engine.optimizers.items()},
             engine.rng.get_state(), engine.kind_rng.get_state())
    comparison = comparator.Comparison()
    comparison.walk(before, after)
    assert comparison.differences == []


def test_tuna_restored_resources_include_all_adapter_sets_and_both_head_types():
    backbone = nn.Module()
    backbone.body = nn.Linear(2, 2, bias=False).requires_grad_(False)
    backbone.register_buffer('body_buffer', torch.zeros(2, dtype=torch.float64))
    backbone.head = nn.Linear(2, 3)
    backbone.cur_adapter = nn.ModuleList([nn.Linear(2, 1, bias=False) for _ in range(2)])
    backbone.adapter_list = nn.ModuleList([copy.deepcopy(backbone.cur_adapter) for _ in range(3)])
    backbone.adapter_list[0].requires_grad_(False)
    backbone.merged_adapter = copy.deepcopy(backbone.cur_adapter)
    for key in ('down_weight_sum', 'down_bias_sum', 'up_weight_sum', 'up_bias_sum'):
        setattr(backbone, key, [])
    network = nn.Module()
    network.backbone = backbone
    network.fc = nn.Module()
    network.fc.heads = nn.ModuleList([nn.Linear(2, 2, bias=False) for _ in range(2)])
    model = SimpleNamespace(_network=network, exemplar_size=0,
                            cls_mean={i: torch.zeros(2) for i in range(2)},
                            cls_cov={i: torch.zeros(2, 2) for i in range(2)},
                            _arrived_ids=[np.arange(4, dtype=np.int64)])
    measured = backup.restored_resources(model, 'tuna')
    neural = measured['neural_network']
    assert neural['resident_nn_parameters'] == 41 and neural['resident_nn_parameter_bytes'] == 164
    assert neural['gradient_eligible_nn_parameters'] == 33
    assert neural['gradient_eligible_nn_parameter_bytes'] == 132
    assert neural['registered_nn_buffer_bytes'] == 16
    assert {k: v['parameters'] for k, v in neural['components'].items()} == dict(
        original_backbone_body=4, native_backbone_heads=9, current_adapters=4,
        historical_adapters=12, merged_adapters=4, class_predictor_heads=8)
    assert measured['architecture'] == dict(current_adapter_layers=2, historical_adapter_sets=3,
                                            historical_adapter_layers_per_set=[2, 2, 2], historical_adapter_layers=6,
                                            merged_adapter_layers=2, class_predictor_heads=2)
    assert measured['auxiliary']['class_mean_bytes'] == 16
    assert measured['auxiliary']['class_covariance_bytes'] == 32
    assert measured['auxiliary']['historical_class_statistics_bytes'] == 48
    assert measured['auxiliary']['training_arrival_id_bytes'] == 32
    assert measured['optimizer']['persisted_optimizer_count'] == 0
    assert measured['optimizer']['persisted_optimizer_state_tensor_bytes'] == 0
    assert measured['replay']['images'] == 0


def test_replay_resources_distinguish_encoded_paths_from_inline_or_decoded_bytes(tmp_path):
    from PIL import Image
    path = tmp_path / 'image.png'
    Image.fromarray(np.zeros((3, 4, 3), dtype=np.uint8)).save(path)
    memory = SimpleNamespace(records=[SimpleNamespace(image=str(path))])
    measured = backup.replay_resources(memory)
    assert measured['images'] == measured['referenced_images'] == 1
    assert measured['inline_rgb_array_bytes'] == 0
    assert measured['referenced_encoded_image_file_bytes'] == path.stat().st_size
    assert measured['image_path_utf8_bytes'] == len(str(path).encode('utf-8'))
    assert measured['decoded_rgb_plus_id_label_payload_bytes'] == 3 * 4 * 3 + 16


def test_numeric_payload_count_deduplicates_objects_and_ignores_python_metadata():
    value = torch.ones(3, dtype=torch.float64)
    array = np.zeros(2, dtype=np.int16)
    assert backup.numeric_payload_bytes(dict(one=value, alias=value, array=array,
                                             metadata=[123456, 'not tensor bytes', 2.5])) == 28


@pytest.fixture
def recovered_fixture(tmp_path, monkeypatch):
    """Use the real recovery preparer/validator with existing tiny queue fixtures."""
    from test_resume_followup_after_reboot import fixture as prepare_fixture
    from scripts import resume_followup_after_reboot as recovery
    from scripts import run_followup_queue as queue
    from cl27 import runner
    f, base = prepare_fixture(tmp_path, monkeypatch)
    data_by_name = {}
    for job in recovery.GROUPS['current20']:
        row = base.entries[job]
        run = Path(row['output_dir'])
        n, evaluation = (50000, 10000) if row['dataset'] == 'cifar224' else (24000, 6000)
        data = dict(fingerprint=recovery.DATA_SHA[row['dataset']], phase='final',
                    evaluation_source='designated_test', official_test_read=True,
                    train_count=n, evaluation_count=evaluation)
        data_by_name[row['dataset']] = data
        save(run / 'data_manifest.json', data)
        state = torch.load(run / 'checkpoint.pt', map_location='cpu', weights_only=False)
        for stage in state['stages']:
            stage.update(prediction_space='all_seen_classes_without_task_id', evaluation_images=evaluation)
        state['rng'] = dict(python=random.Random(7).getstate(), numpy=np.random.RandomState(7).get_state(),
                            torch=torch.Generator().manual_seed(7).get_state(),
                            cuda=[torch.zeros(4, dtype=torch.uint8), torch.ones(4, dtype=torch.uint8)])
        state['learner']['core'] = {'latent': torch.arange(3, dtype=torch.float64)}
        torch.save(state, run / 'checkpoint.pt')
        save(run / 'progress.json', dict(status='running', stages=state['stages']))
    class DataManager:
        def __init__(self, dataset, *args):
            self.manifest = copy.deepcopy(data_by_name[dataset])
    monkeypatch.setattr(recovery, '_context', lambda *args: (base, queue, runner, DataManager))
    monkeypatch.setattr(backup, 'WEIGHT_SHA', backup.sha(f.cache))
    recovery_path = recovery.prepare(f.path, 'current20', tmp_path / 'recovery')
    manifest = recovery.read(recovery_path)
    entry = next(row for row in manifest['jobs'] if row['job'] == JOB)
    cfg = recovery.read(entry['config']['path'])
    run = Path(entry['output_dir'])
    save(run / 'config.json', dict(cfg, seed=1993, config=entry['config']['path']))
    save(run / 'data_manifest.json', data_by_name['cifar224'])
    parent = torch.load(cfg['resume'], map_location='cpu', weights_only=False)
    state = copy.deepcopy(parent)
    state['stages'] += [dict(task=i, seen_classes=10 * (i + 1), wall_seconds=3.25,
                             prediction_space='all_seen_classes_without_task_id', evaluation_images=10000)
                        for i in range(2, 10)]
    state['learner'].update(cur_task=9, known_classes=100, total_classes=100, task_diagnostics=[{}] * 10)
    torch.save(state, run / 'checkpoint.pt')
    result = dict(status='complete', method='current_adapted', phase='final',
                  stages=copy.deepcopy(state['stages']), metrics={'completed_tasks': 10},
                  config_fingerprint=state['config_fingerprint'], data_fingerprint=state['data_fingerprint'],
                  task_diagnostics=[{}] * 10, execution={'wall_seconds': 30.})
    save(run / 'results.json', result)
    receipt = dict(job=JOB, pid=12345, proc_start_ticks=67890, command=entry['command'],
                    cwd=str(f.repo), gpu=0, source_commit=manifest['source_commit'],
                    manifest_sha256=backup.sha(recovery_path), config_sha256=entry['config']['sha256'],
                    source_inventory_sha256=hashlib.sha256(json.dumps(manifest['source_files_sha256'], sort_keys=True).encode()).hexdigest())
    save(recovery_path.parent / 'receipts' / (JOB + '.json'), receipt)
    seal = dict(job=JOB, original_process_state='absent', observed_exit_code=None,
                 config_sha256=backup.sha(run / 'config.json'), result_sha256=backup.sha(run / 'results.json'),
                 data_manifest_sha256=backup.sha(run / 'data_manifest.json'),
                 recovery_manifest_sha256=backup.sha(recovery_path),
                 lineage_sha256=recovery.canonical_sha(entry['lineage']),
                 prefix_stages_sha256=entry['lineage']['prefix_stages_sha256'], saved_stage_count=2)
    seal_path = recovery_path.parent / 'queue' / (JOB + '.complete.json')
    save(seal_path, seal)
    admitted = backup.admit(f.path, JOB, recovery_manifest_path=recovery_path, probe=lambda _pid: None)
    return dict(root=tmp_path, base=f, path=recovery_path, run=run, state=state,
                admission=admitted, lineage=entry['lineage'], seal_path=seal_path)


def test_recovery_admission_requires_exact_lineage_and_binds_new_terminal_seal(recovered_fixture):
    f = recovered_fixture
    admitted = f['admission']
    assert admitted['manifest_sha256'] == backup.sha(f['base'].path)
    assert admitted['recovery_manifest_sha256'] == backup.sha(f['path'])
    assert admitted['run_dir'] == f['run']
    assert admitted['metadata']['config']['resume'] == f['lineage']['resume_checkpoint']['path']
    with pytest.raises(ValueError, match='fresh complete stream'):
        backup.metadata_guard(admitted['metadata'], validation_only=False)
    seal = backup.read(f['seal_path'])
    seal['lineage_sha256'] = 'wrong'
    save(f['seal_path'], seal)
    with pytest.raises(ValueError, match='bound to the original checkpoint lineage'):
        backup.admit(f['base'].path, JOB, recovery_manifest_path=f['path'], probe=lambda _pid: None)


def test_recovery_prefix_proof_reopens_pickle_and_compares_typed_stage_bits(recovered_fixture):
    f = recovered_fixture
    report = backup.recovery_prefix_proof(f['admission'], f['state'], comparator)
    assert report['recovery_prefix_verified'] is True
    assert report['saved_stage_count'] == 2 and report['recovery_prefix_ignored_paths'] == []
    assert report['source_checkpoint_stages_sha256'] == report['restored_final_prefix_sha256'] == f['lineage']['prefix_stages_sha256']
    assert report['original_checkpoint_sha256'] == report['resume_checkpoint_sha256'] == f['lineage']['source_checkpoint']['sha256']
    assert report['recovery_execution_scope']['complete_lineage_training_cost_claimed'] is False
    changed = copy.deepcopy(f['state'])
    changed['stages'][0]['wall_seconds'] += 1e-12
    with pytest.raises(ValueError, match='exact typed source prefix'):
        backup.recovery_prefix_proof(f['admission'], changed, comparator)
    changed = copy.deepcopy(f['state'])
    changed['config_fingerprint'] = 'different science'
    with pytest.raises(ValueError, match='config/data/RNG identity differs'):
        backup.recovery_prefix_proof(f['admission'], changed, comparator)


def test_recovery_backup_requires_pickle_prefix_proof_before_publication(recovered_fixture):
    f = recovered_fixture
    backup_root, evidence_root = f['root'] / 'backups', f['root'] / 'evidence'
    with pytest.raises(ValueError, match='independent native pickle-prefix proof'):
        backup.backup(f['admission'], backup_root, evidence_root, restore=lambda *args: {}, probe=lambda _pid: None)
    assert not (backup_root / JOB).exists() and not (evidence_root / (JOB + '.json')).exists()
    def restore(admission, checkpoint, loader):
        state, _ = comparator.load_trusted_checkpoint(checkpoint)
        return backup.recovery_prefix_proof(admission, state, comparator)
    proof = backup.backup(f['admission'], backup_root, evidence_root, restore=restore, probe=lambda _pid: None)
    assert proof['source_manifest_sha256'] == backup.sha(f['base'].path)
    assert proof['recovery_manifest_sha256'] == backup.sha(f['path'])
    assert proof['recovery_lineage'] == dict(
        original_checkpoint_sha256=f['lineage']['source_checkpoint']['sha256'],
        resume_checkpoint_sha256=f['lineage']['resume_checkpoint']['sha256'], saved_stage_count=2,
        prefix_stages_sha256=f['lineage']['prefix_stages_sha256'], config_fingerprint=f['lineage']['config_fingerprint'],
        data_fingerprint=f['lineage']['data_fingerprint'], random_stream_policy=f['lineage']['random_stream_policy'],
        prefix_verified=True)
    assert (backup_root / JOB / 'source_recovery_manifest.json').is_file()
    assert proof['recovery_execution_scope']['wall_seconds'] == 30.


def test_recovery_source_checkpoint_mutation_never_produces_native_prefix_proof(recovered_fixture):
    f = recovered_fixture
    snapshot = Path(f['lineage']['resume_checkpoint']['path'])
    snapshot.chmod(0o644)
    snapshot.write_bytes(b'changed retained checkpoint')
    with pytest.raises(ValueError, match='checkpoint bytes changed'):
        backup.recovery_prefix_proof(f['admission'], f['state'], comparator)


def test_second_recovery_reopens_both_checkpoints_against_final_typed_history(recovered_fixture, tmp_path):
    f = recovered_fixture
    parent = f['lineage']
    count = 8
    latest_path = tmp_path / 'later_original.pt'
    snapshot = tmp_path / 'immutable_copy.pt'
    state = copy.deepcopy(f['state'])
    state['stages'] = state['stages'][:count]
    state['learner'].update(cur_task=7, known_classes=80, total_classes=80, task_diagnostics=[{}] * count)
    def bind_intermediate():
        torch.save(state, latest_path)
        snapshot.write_bytes(latest_path.read_bytes())
        def binding(p):
            return dict(path=str(p), sha256=backup.sha(p), bytes=p.stat().st_size)
        prefix_sha = hashlib.sha256(json.dumps(
            state['stages'], sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        return dict(parent, saved_stage_count=count, prefix_stages=state['stages'], prefix_stages_sha256=prefix_sha,
                    source_checkpoint=binding(latest_path), resume_checkpoint=binding(snapshot))
    admission = dict(f['admission'], recovery_lineage=bind_intermediate(),
                     parent_recovery_lineage=parent, parent_recovery_manifest=dict(path=str(f['path']), sha256=backup.sha(f['path'])))
    final = copy.deepcopy(f['state'])
    proof = backup.recovery_prefix_proof(admission, final, comparator)
    assert proof['saved_stage_count'] == count
    assert proof['parent_recovery_prefix_verification']['saved_stage_count'] == 2
    # A common mutation in the intermediate and final stages passes their
    # mutual comparison but contradicts the immutable original two stages.
    state['stages'][0]['wall_seconds'] += .25
    final['stages'][0]['wall_seconds'] += .25
    admission['metadata']['results']['stages'][0]['wall_seconds'] += .25
    admission['recovery_lineage'] = bind_intermediate()
    with pytest.raises(ValueError, match='exact typed source prefix'):
        backup.recovery_prefix_proof(admission, final, comparator)
