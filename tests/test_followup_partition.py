"""Two-server partition ownership, using fake processes and GPU snapshots."""
from __future__ import annotations

from pathlib import Path

import pytest

from scripts import run_followup_partition as partition
from scripts import run_followup_queue as queue
from test_followup_queue import Fixture, files, save, REPO, CF20, R20, CFTUNA, RTUNA


def controller(fixture, role):
    return partition.PartitionController(fixture.path, role, base=fixture.create())


def test_tuna_partition_ignores_missing_or_unresolved_excluded_jobs_and_plan_is_read_only(tmp_path, monkeypatch):
    fixture = Fixture(tmp_path, monkeypatch)
    c = controller(fixture, 'tuna')
    save(c.base.intent_path(CF20), {'excluded_host_local_intent': True})
    before = files(tmp_path)
    report = c.cycle()
    assert files(tmp_path) == before and fixture.calls == []
    assert set(report['statuses']) == {CFTUNA, RTUNA}
    assert [(a['job'], a['gpu'], a['state']) for a in report['actions']] == [
        (CFTUNA, 0, 'ready'), (RTUNA, 1, 'ready')]
    with c.base.exclusive():
        report = c.cycle(True)
    assert len(fixture.calls) == 2 and set(report['statuses'].values()) == {'running'}
    assert [call[0] for call in fixture.calls] == [c.base.entries[j]['command'] for j in (CFTUNA, RTUNA)]
    assert not c.base.receipt_path(CF20).exists() and not c.base.receipt_path(R20).exists()
    assert c.latest.is_file() and not (c.base.state / 'latest.json').exists()


def test_current_partition_adopts_existing_workers_only_and_finishes_without_tuna(tmp_path, monkeypatch):
    fixture = Fixture(tmp_path, monkeypatch)
    original = fixture.create()
    with original.exclusive():
        original.cycle(True)
    c = controller(fixture, 'current20')
    with c.base.exclusive():
        report = c.cycle(True)
    assert set(report['statuses']) == {CF20, R20}
    assert not report['all_complete'] and len(fixture.calls) == 2
    fixture.finish(c.base, CF20)
    fixture.finish(c.base, R20)
    with c.base.exclusive():
        report = c.cycle(True)
    assert report['all_complete'] and len(fixture.calls) == 2
    assert not c.base.receipt_path(CFTUNA).exists() and not c.base.receipt_path(RTUNA).exists()


def test_current_partition_never_recreates_missing_original_worker(tmp_path, monkeypatch):
    fixture = Fixture(tmp_path, monkeypatch)
    c = controller(fixture, 'current20')
    with c.base.exclusive(), pytest.raises(ValueError, match='adoption-only'):
        c.cycle(True)
    assert fixture.calls == []


def test_tuna_restart_preserves_empty_argv_reservation_and_requires_terminal_result(tmp_path, monkeypatch):
    fixture = Fixture(tmp_path, monkeypatch)
    c = controller(fixture, 'tuna')
    with c.base.exclusive():
        c.cycle(True)
    c = controller(fixture, 'tuna')
    pid = queue.read(c.base.receipt_path(CFTUNA))['pid']
    fixture.probes[pid]['command'] = []
    fixture.finish(c.base, CFTUNA, terminal=False)
    fixture.finish(c.base, RTUNA)
    with c.base.exclusive():
        report = c.cycle(True)
    assert report['statuses'] == {CFTUNA: 'running', RTUNA: 'complete'}
    assert not report['all_complete'] and len(fixture.calls) == 2
    fixture.finish(c.base, CFTUNA)
    with c.base.exclusive():
        report = c.cycle(True)
    assert report['all_complete'] and len(fixture.calls) == 2


@pytest.mark.parametrize('failure', ['nonzero', 'lost_receipt'])
def test_tuna_failure_or_ambiguous_intent_does_not_retry(tmp_path, monkeypatch, failure):
    fixture = Fixture(tmp_path, monkeypatch)
    c = controller(fixture, 'tuna')
    with c.base.exclusive():
        c.cycle(True)
    if failure == 'nonzero':
        fixture.finish(c.base, CFTUNA, exit_code=1)
    else:
        c.base.receipt_path(CFTUNA).unlink()
    c = controller(fixture, 'tuna')
    with c.base.exclusive(), pytest.raises(ValueError):
        c.cycle(True)
    assert len(fixture.calls) == 2


def test_foreign_gpu_occupancy_blocks_only_its_tuna_job_and_low_disk_blocks_all(tmp_path, monkeypatch):
    fixture = Fixture(tmp_path, monkeypatch)
    c = controller(fixture, 'tuna')
    fixture.free = 13 * 2**30 - 1
    with c.base.exclusive(), pytest.raises(ValueError, match='disk budget'):
        c.cycle(True)
    assert fixture.calls == []
    fixture.free = 20 * 2**30
    fixture.snapshot[0]['pids'] = [999]
    with c.base.exclusive():
        report = c.cycle(True)
    assert report['actions'][0]['state'] == 'waiting_unknown_gpu_process'
    assert report['statuses'] == {CFTUNA: 'waiting', RTUNA: 'running'}
    assert len(fixture.calls) == 1


def test_observation_interrupt_keeps_receipts_adoptable_without_failure(tmp_path, monkeypatch):
    fixture = Fixture(tmp_path, monkeypatch)
    c = controller(fixture, 'tuna')
    def interrupt(_):
        raise KeyboardInterrupt()
    monkeypatch.setattr(partition.time, 'sleep', interrupt)
    with pytest.raises(KeyboardInterrupt):
        c.run()
    assert not c.failure.exists()
    restarted = controller(fixture, 'tuna')
    with restarted.base.exclusive():
        restarted.cycle(True)
    assert len(fixture.calls) == 2


def test_runtime_import_uses_manifest_checkout_and_rejects_changed_original_controller(tmp_path, monkeypatch):
    fixture = Fixture(tmp_path, monkeypatch)
    manifest = queue.read(fixture.path)
    for name in ('prepare_followup_campaign.py', 'run_followup_queue.py'):
        target = fixture.repo / 'scripts' / name
        target.parent.mkdir(exist_ok=True)
        target.write_bytes((REPO / 'scripts' / name).read_bytes())
        manifest['source_files_sha256']['scripts/' + name] = queue.sha(target)
    save(fixture.path, manifest)
    module = partition.load_guarded_queue(fixture.path)
    assert Path(module.__file__) == fixture.repo / 'scripts/run_followup_queue.py'
    path = fixture.repo / 'scripts/prepare_followup_campaign.py'
    path.write_text(path.read_text() + '\n# changed after manifest\n')
    with pytest.raises(ValueError, match='source changed'):
        partition.load_guarded_queue(fixture.path)
