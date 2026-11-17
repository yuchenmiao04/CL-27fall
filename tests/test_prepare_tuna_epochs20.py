"""Budget-overlay contracts; no GPU, training, or real checkpoint payload."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import prepare_tuna_epochs20 as overlay
from scripts import prepare_followup_campaign as original_prepare
from scripts import run_followup_queue as queue
from scripts.run_followup_partition import PartitionController
from test_followup_queue import Fixture, REPO, files, save, CF20, R20, CFTUNA, RTUNA


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    fixture = Fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(overlay, 'BASE_SHA256', queue.sha(fixture.path))
    monkeypatch.setattr(overlay, 'load_guarded_queue', lambda _: SimpleNamespace(
        FollowupQueue=lambda _: fixture.create(), __package__='scripts'))
    monkeypatch.setattr(original_prepare, 'source_inventory',
        lambda repo: {'main.py': queue.sha(repo / 'main.py')})
    check = original_prepare.check_template
    calls = []
    def check_original(repo, job, cfg, dataset, method):
        assert repo == fixture.repo
        calls.append(job)
        return check(REPO, job, cfg, dataset, method)
    monkeypatch.setattr(original_prepare, 'check_template', check_original)
    return fixture, calls


def test_only_tuna_budget_changes_and_original_controller_accepts_overlay(prepared):
    fixture, calls = prepared
    before = files(fixture.campaign)
    old = queue.read(fixture.path)
    path = overlay.prepare(fixture.path, fixture.root / 'campaign_tuna20')
    new = queue.read(path)
    assert files(fixture.campaign) == before
    assert calls == list(queue.JOBS)
    assert new['user_requested_image_epochs'] == 20
    assert new['base_manifest_path'] == str(fixture.path)
    assert new['base_manifest_sha256'] == queue.sha(fixture.path)
    assert hashlib.sha256(new['base_manifest_text'].encode()).hexdigest() == new['base_manifest_sha256']
    for key in old:
        if key not in {'campaign_root', 'jobs'}:
            assert new[key] == old[key]
    for oldrow, newrow in zip(old['jobs'], new['jobs']):
        cfg = queue.read(newrow['config']['path'])
        expected = queue.read(oldrow['config']['path'])
        expected['output_dir'] = newrow['output_dir']
        if newrow['method'] == 'tuna':
            expected.update(tuned_epoch=20, training_budget_variant='epochs20', prefix='followup_tuna_epochs20')
            assert cfg['crct_epochs'] == 30
            assert cfg['comparison_variant'] == 'pilot_native_same_protocol'
        assert cfg == expected
        assert newrow['config']['sha256'] == queue.sha(newrow['config']['path'])
        assert newrow['command'][-1] == newrow['config']['path']
        assert {k: v for k, v in newrow.items() if k not in {'config', 'command', 'output_dir'}} == {
            k: v for k, v in oldrow.items() if k not in {'config', 'command', 'output_dir'}}
    # Reuse the original controller APIs against the complete new manifest.
    q = queue.FollowupQueue(path, probe=lambda pid: None, memory=lambda: copy.deepcopy(fixture.snapshot),
                           disk=lambda _: SimpleNamespace(free=fixture.free), spawn=fixture.spawn)
    partition = PartitionController(path, 'tuna', base=q)
    report = partition.cycle()
    assert set(report['statuses']) == {CFTUNA, RTUNA}
    assert [a['job'] for a in report['actions']] == [CFTUNA, RTUNA]
    assert not q.receipt_path(CF20).exists() and not q.receipt_path(R20).exists()
    assert fixture.calls == []  # Preparation and planning never launch a worker.


def test_refuses_existing_overlay_and_changed_base_manifest(prepared):
    fixture, _ = prepared
    target = fixture.root / 'campaign_tuna20'
    overlay.prepare(fixture.path, target)
    before = files(target)
    with pytest.raises(ValueError, match='existing campaign'):
        overlay.prepare(fixture.path, target)
    assert files(target) == before
    fixture.path.write_text(fixture.path.read_text() + '\n')
    with pytest.raises(ValueError, match='manifest SHA differs'):
        overlay.prepare(fixture.path, fixture.root / 'another_campaign')
    assert not (fixture.root / 'another_campaign').exists()


def test_original_native_recipe_is_validated_before_user_override(prepared, monkeypatch):
    fixture, _ = prepared
    manifest = queue.read(fixture.path)
    row = next(r for r in manifest['jobs'] if r['job'] == CFTUNA)
    cfg_path = Path(row['config']['path'])
    cfg = queue.read(cfg_path)
    cfg['reg'] = .123
    save(cfg_path, cfg)
    row['config']['sha256'] = queue.sha(cfg_path)
    save(fixture.path, manifest)
    # Even if an alternative base hash were explicitly admitted, recipe drift
    # still cannot be disguised as the requested image-epoch-only override.
    monkeypatch.setattr(overlay, 'BASE_SHA256', queue.sha(fixture.path))
    with pytest.raises(ValueError, match='Native TUNA hyperparameter changed'):
        overlay.prepare(fixture.path, fixture.root / 'campaign_tuna20')
    assert not (fixture.root / 'campaign_tuna20').exists()


@pytest.mark.parametrize('field,value', [('crct_epochs', 20), ('init_lr', .5), ('batch_size', 4)])
def test_exact_delta_rejects_unrequested_scientific_changes(field, value):
    original = queue.read(REPO / 'experiments/extension_20261117/configs/formal_cifar_tuna_native_v1.json')
    modified = dict(original, output_dir='/fresh', tuned_epoch=20,
                    training_budget_variant='epochs20', prefix='followup_tuna_epochs20')
    modified[field] = value
    with pytest.raises(ValueError, match='Scientific delta exceeds'):
        overlay.assert_config_delta(original, modified, 'tuna', '/fresh')
