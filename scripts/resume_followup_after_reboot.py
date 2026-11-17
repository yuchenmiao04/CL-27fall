"""Prepare task-boundary recovery in new directories without scientific changes.

Original campaigns and raw results are retained. Only ``resume`` and
``output_dir`` are added/changed in the prepared configurations. Checkpoints
are copied server-local, SHA-bound, and read-only; no model download occurs.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

if __package__:
    from .run_followup_partition import GROUPS, load_guarded_queue
else:
    from run_followup_partition import GROUPS, load_guarded_queue

SOURCE_SHA = {
    'current20': '8a70c7e5a6011bb293c833387d71046173503d4ba94efca3c3ee2587b82e4c43',
    'tuna': '04fda502b11e670bb3f13b35b31fef7545d2d47728b4947220210c6b3bd0c2c6',
}
RNG_POLICY = 'canonical_loader_isolated_v1'
DATA_SHA = {'cifar224': '98b4aa3a9e6bfd167c52eb391f6dc1b028f2fb920f8a20c8c48751282b8f7ed4',
            'imagenetr': '014b448f085423b32b6ca0468e7165f896460ff3e4341d1f7bcc305d96a2de76'}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def bound(path):
    path = Path(path)
    return {'path': str(path), 'sha256': sha(path)}


def _context(path, partition):
    path = Path(path).resolve()
    need(partition in GROUPS and sha(path) == SOURCE_SHA[partition], 'Wrong selected source campaign')
    api = load_guarded_queue(path)
    base = api.FollowupQueue(path)  # Original source/config/cache guards; no GPU context.
    sys.path.insert(0, str(base.repo))
    runner = importlib.import_module('cl27.runner')
    data = importlib.import_module('cl27.data')
    need(Path(runner.__file__).resolve() == base.repo / 'cl27/runner.py'
         and Path(data.__file__).resolve() == base.repo / 'cl27/data.py', 'Wrong imported scientific source')
    return base, api, runner, data.ProtocolDataManager


def old_worker_observation(base, job):
    path = base.receipt_path(job)
    receipt = read(path)
    base.check_receipt(job, receipt)
    info = base.probe(receipt['pid'])
    if info is None:
        state = 'original_pid_absent'
    elif info['startticks'] != receipt['proc_start_ticks']:
        state = 'original_pid_reused'
    else:
        need(info['state'] in {'Z', 'X', 'x'}, 'Original worker identity is still live: ' + job)
        state = 'original_process_terminal_' + info['state']
    return {**bound(path), 'pid': receipt['pid'], 'proc_start_ticks': receipt['proc_start_ticks'],
            'observation': state}


def prepared_fingerprint(runner, cfg):
    return runner.configuration_fingerprint(dict(cfg, seed=1993))


def check_checkpoint(state, cfg, data, runner):
    runner.validate_checkpoint_random_stream(state)
    need(state.get('version') == 1 and state.get('random_stream_policy') == RNG_POLICY,
         'Wrong checkpoint version/RNG policy')
    need(state['config_fingerprint'] == prepared_fingerprint(runner, cfg)
         and state['data_fingerprint'] == data['fingerprint'] == DATA_SHA[cfg['dataset']],
         'Checkpoint configuration or data identity changed')
    stages, learner = state['stages'], state['learner']
    n, width = len(stages), cfg['increment']
    need(0 < n < 10 and [s['task'] for s in stages] == list(range(n))
         and [s['seen_classes'] for s in stages] == [width * (i + 1) for i in range(n)],
         'Checkpoint stages are not a proper task-boundary prefix')
    need(learner['version'] == 1 and learner['method'] == cfg['model_name']
         and learner['cur_task'] == n - 1 and learner['known_classes'] == learner['total_classes'] == n * width
         and len(learner['task_diagnostics']) == n, 'Checkpoint learner counters disagree')
    rng = state['rng']
    need(set(rng) == {'python', 'numpy', 'torch', 'cuda'} and rng['cuda'] is not None
         and len(rng['cuda']) == 2, 'Checkpoint full two-device RNG state is missing')
    return runner.jsonable(stages)


def prepare(manifest_path, partition, recovery_root):
    manifest_path, recovery_root = Path(manifest_path).resolve(), Path(recovery_root).resolve()
    need(not recovery_root.exists(), 'Refusing existing recovery output')
    base, api, runner, data_manager = _context(manifest_path, partition)
    need(all(not row['pids'] for row in base.memory().values()), 'GPU occupancy blocks reboot recovery')
    staging = recovery_root.with_name(recovery_root.name + '.preparing')
    need(not staging.exists(), 'Unresolved previous recovery preparation')
    staging.mkdir(parents=True, exist_ok=False)
    try:
        (staging / 'configs').mkdir()
        entries = []
        for job in GROUPS[partition]:
            old = base.entries[job]
            cfg = read(old['config']['path'])
            output = Path(old['output_dir'])
            need(not (output / 'checkpoint.tmp').exists() and not (output / 'results.json').exists(),
                 'Partial-save or already-complete source requires review: ' + job)
            observation = old_worker_observation(base, job)
            saved = read(output / 'config.json')
            need({k: v for k, v in saved.items() if k != 'config'} == dict(cfg, seed=1993),
                 'Original executed config differs')
            data = read(output / 'data_manifest.json')
            dm = data_manager(cfg['dataset'], cfg['shuffle'], 1993, cfg['init_cls'], cfg['increment'], dict(cfg, seed=1993))
            need(dm.manifest == data and data['fingerprint'] == DATA_SHA[cfg['dataset']],
                 'Actual full dataset identity changed')
            checkpoint = output / 'checkpoint.pt'
            checkpoint_sha, size = sha(checkpoint), checkpoint.stat().st_size
            need(shutil.disk_usage(recovery_root.parent).free > size + 4 * 2**30,
                 'Insufficient disk for immutable checkpoint snapshot')
            snapshot = staging / 'resume_sources' / job / 'checkpoint.pt'
            snapshot.parent.mkdir(parents=True)
            shutil.copyfile(checkpoint, snapshot)
            with snapshot.open('rb') as stream:
                os.fsync(stream.fileno())
            snapshot.chmod(0o444)
            need(sha(snapshot) == checkpoint_sha == sha(checkpoint), 'Checkpoint changed while snapshotting')
            state = runner.torch.load(snapshot, map_location='cpu', weights_only=False)
            prefix = check_checkpoint(state, cfg, data, runner)
            progress = read(output / 'progress.json')
            need(progress['stages'][:len(prefix)] == prefix, 'Checkpoint prefix differs from source progress')
            final_snapshot = recovery_root / 'resume_sources' / job / 'checkpoint.pt'
            new_cfg = dict(cfg, resume=str(final_snapshot), output_dir=str(recovery_root / 'runs' / job))
            need(prepared_fingerprint(runner, new_cfg) == state['config_fingerprint'], 'Recovery scientific fingerprint changed')
            config_path = recovery_root / 'configs' / (job + '.json')
            staged_config = staging / 'configs' / config_path.name
            write_value = json.dumps(new_cfg, indent=2, allow_nan=False) + '\n'
            staged_config.write_text(write_value)
            lineage = {
                'original_manifest': bound(manifest_path), 'original_prepared_config': copy.deepcopy(old['config']),
                'original_saved_config': bound(output / 'config.json'), 'original_output_dir': str(output),
                'original_progress': bound(output / 'progress.json'), 'original_data_manifest': bound(output / 'data_manifest.json'),
                'source_checkpoint': dict(path=str(checkpoint), sha256=checkpoint_sha, bytes=size),
                'resume_checkpoint': dict(path=str(final_snapshot), sha256=checkpoint_sha, bytes=size),
                'config_fingerprint': state['config_fingerprint'], 'data_fingerprint': state['data_fingerprint'],
                'random_stream_policy': state['random_stream_policy'], 'saved_stage_count': len(prefix),
                'prefix_stages': prefix, 'prefix_stages_sha256': canonical_sha(prefix), 'original_receipt': observation,
            }
            old_log = base.formal / 'logs' / (job + '.log')
            if old_log.is_file():
                lineage['original_log'] = bound(old_log)
            entries.append(dict(job=job, gpu=old['gpu'], dataset=old['dataset'], method=old['method'],
                comparison_variant=old['comparison_variant'], config=dict(path=str(config_path), sha256=sha(staged_config)),
                output_dir=new_cfg['output_dir'], command=[base.manifest['python'], str(base.repo / 'main.py'), '--config', str(config_path)],
                environment=copy.deepcopy(old['environment']), lineage=lineage))
            del state, dm
        manifest = {key: copy.deepcopy(base.manifest[key]) for key in
                    ('repo', 'python', 'environment', 'pretrained_npz', 'source_files_sha256', 'source_commit')}
        manifest.update(schema_version=1, kind='followup_task_boundary_recovery', partition=partition,
            source_manifest=bound(manifest_path), recovery_root=str(recovery_root), jobs=entries,
            preparation_helper_sha256=sha(__file__), prepared_unix=time.time(),
            resource_scope='Raw resumed execution wall/peak cover only the resumed segment; saved stage records remain unchanged.')
        (staging / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
        base.validate()
        need(not recovery_root.exists(), 'Recovery path appeared during preparation')
        os.rename(staging, recovery_root)
    except BaseException:
        shutil.rmtree(staging)
        raise
    return recovery_root / 'manifest.json'


def validate_recovery_manifest(path):
    """Validate JSON/source/hash lineage without deserializing model payloads."""
    path = Path(path).resolve()
    manifest = read(path)
    need(manifest['schema_version'] == 1 and manifest['kind'] == 'followup_task_boundary_recovery', 'Wrong recovery schema')
    base, api, runner, _ = _context(manifest['source_manifest']['path'], manifest['partition'])
    need(manifest['source_manifest'] == bound(base.path), 'Recovery source manifest changed')
    root = Path(manifest['recovery_root'])
    need(path == root / 'manifest.json' and {r['job'] for r in manifest['jobs']} == set(GROUPS[manifest['partition']])
         and len(manifest['jobs']) == 2, 'Wrong recovery roots/jobs')
    for key in ('repo', 'python', 'environment', 'pretrained_npz', 'source_files_sha256', 'source_commit'):
        need(manifest[key] == base.manifest[key], 'Recovery source/runtime identity changed: ' + key)
    configs = {}
    for row in manifest['jobs']:
        old, lineage = base.entries[row['job']], row['lineage']
        need((row['gpu'], row['dataset'], row['method'], row['comparison_variant'], row['environment'])
             == tuple(old[k] for k in ('gpu', 'dataset', 'method', 'comparison_variant', 'environment')), 'Recovery job identity differs')
        cfg_path = root / 'configs' / (row['job'] + '.json')
        need(row['config'] == bound(cfg_path) and row['output_dir'] == str(root / 'runs' / row['job']), 'Recovery config/output differs')
        cfg = read(cfg_path)
        expected = dict(read(old['config']['path']), resume=str(root / 'resume_sources' / row['job'] / 'checkpoint.pt'), output_dir=row['output_dir'])
        need(cfg == expected and cfg['seed'] == [1993], 'Only resume/output_dir may differ from the original scientific config')
        need(row['command'] == [manifest['python'], str(base.repo / 'main.py'), '--config', str(cfg_path)], 'Recovery command changed')
        need(lineage['original_manifest'] == bound(base.path) and lineage['original_prepared_config'] == old['config']
             and lineage['original_output_dir'] == old['output_dir'], 'Recovery original bindings changed')
        for key, name in [('original_saved_config', 'config.json'), ('original_progress', 'progress.json'), ('original_data_manifest', 'data_manifest.json')]:
            need(lineage[key] == bound(Path(old['output_dir']) / name), 'Recovery original metadata changed: ' + key)
        for key, expected_path in [('source_checkpoint', Path(old['output_dir']) / 'checkpoint.pt'),
                                   ('resume_checkpoint', Path(cfg['resume']))]:
            value = lineage[key]
            need(value['path'] == str(expected_path) and value['sha256'] == sha(expected_path)
                 and value['bytes'] == expected_path.stat().st_size, 'Retained checkpoint bytes changed: ' + key)
        need(lineage['source_checkpoint']['sha256'] == lineage['resume_checkpoint']['sha256']
             and lineage['source_checkpoint']['bytes'] == lineage['resume_checkpoint']['bytes'], 'Snapshot/source checkpoint differs')
        receipt = lineage['original_receipt']
        need({k: receipt[k] for k in ('path', 'sha256')} == bound(base.receipt_path(row['job'])), 'Original process receipt changed')
        raw_receipt = read(receipt['path'])
        base.check_receipt(row['job'], raw_receipt)
        need((receipt['pid'], receipt['proc_start_ticks']) == (raw_receipt['pid'], raw_receipt['proc_start_ticks'])
             and receipt['observation'] in {'original_pid_absent', 'original_pid_reused',
                                           'original_process_terminal_Z', 'original_process_terminal_X', 'original_process_terminal_x'},
             'Original process observation differs')
        if 'original_log' in lineage:
            need(lineage['original_log'] == bound(base.formal / 'logs' / (row['job'] + '.log')), 'Original log changed')
        prefix, n = lineage['prefix_stages'], lineage['saved_stage_count']
        need(0 < n < 10 and len(prefix) == n and [s['task'] for s in prefix] == list(range(n))
             and lineage['prefix_stages_sha256'] == canonical_sha(prefix), 'Recovery prefix metadata changed')
        need(lineage['random_stream_policy'] == RNG_POLICY and lineage['data_fingerprint'] == DATA_SHA[row['dataset']]
             and lineage['config_fingerprint'] == prepared_fingerprint(runner, cfg), 'Recovery fingerprint/RNG differs')
        configs[row['job']] = cfg
    return {'manifest': manifest, 'configs': configs, 'source_manifest': base.manifest}


def recovery_queue(path):
    """Reuse only the old queue's ownership/launch/sealing primitives.

    Its fresh-run validator is replaced by explicit recovery validation. This
    does not change the pinned runner, checkpoint, configuration, or results.
    Standard receipts produced by an operator are adoptable without rewriting.
    """
    path = Path(path).resolve()
    admitted = validate_recovery_manifest(path)
    manifest = admitted['manifest']
    base, api, _, _ = _context(manifest['source_manifest']['path'], manifest['partition'])
    q = copy.copy(base)
    q.path, q.manifest_sha = path, sha(path)
    q.manifest = manifest
    q.root = Path(manifest['recovery_root'])
    q.formal, q.state = q.root, q.root / 'queue'
    q.entries = {r['job']: r for r in manifest['jobs']}
    q.handles, q.locked = {}, False
    q.check_cache = base.check_cache
    def validate():
        need(sha(path) == q.manifest_sha, 'Recovery manifest changed after admission')
        validate_recovery_manifest(path)
    q.validate = validate
    def complete(job):
        evidence = api.FollowupQueue.complete(q, job)
        row = q.entries[job]
        result = read(Path(row['output_dir']) / 'results.json')
        lineage = row['lineage']
        need(result['stages'][:lineage['saved_stage_count']] == lineage['prefix_stages'],
             'Resumed result changed the saved checkpoint prefix')
        need(result['config_fingerprint'] == lineage['config_fingerprint']
             and result['data_fingerprint'] == lineage['data_fingerprint']
             and read(Path(row['output_dir']) / 'data_manifest.json') == read(lineage['original_data_manifest']['path']),
             'Completed resumed result fingerprint changed')
        return evidence
    q.complete = complete
    return q, api


class RecoveryController:
    def __init__(self, path):
        self.q, self.api = recovery_queue(path)
        self.helper_sha = sha(__file__)

    def cycle(self, execute=False):
        q, api = self.q, self.api
        need(not execute or q.locked, 'Execute requires exclusive recovery lock')
        need(sha(__file__) == self.helper_sha, 'Recovery controller changed after admission')
        q.validate()
        if execute:
            need(not (q.state / 'failure.json').exists(), 'Recorded failure requires review; no automatic retry')
        statuses, identities = q.observe()
        snapshot = q.memory()
        free = q.disk(q.root).free
        actions = []
        for job, row in q.entries.items():
            gpu = row['gpu']
            if job in identities:
                need(identities[job]['pid'] not in set(snapshot[1 - gpu]['pids']), 'Known resumed worker appears on wrong GPU')
                continue
            if statuses[job] == 'complete':
                if execute:
                    seal = q.state / (job + '.complete.json')
                    value = read(seal)
                    additions = dict(recovery_manifest_sha256=q.manifest_sha,
                        lineage_sha256=canonical_sha(row['lineage']),
                        prefix_stages_sha256=row['lineage']['prefix_stages_sha256'],
                        saved_stage_count=row['lineage']['saved_stage_count'])
                    need(all(k not in value or value[k] == v for k, v in additions.items()), 'Recovery completion seal changed')
                    api.write(seal, dict(value, **additions))
                continue
            need(statuses[job] == 'waiting', 'Only a waiting recovery job can launch')
            if snapshot[gpu]['pids']:
                actions.append(dict(job=job, gpu=gpu, state='waiting_unknown_gpu_process'))
                continue
            minimum = 8192 if row['method'] == 'current_adapted' else 12288
            if snapshot[gpu]['free_mib'] < minimum + 1024:
                actions.append(dict(job=job, gpu=gpu, state='waiting_gpu_memory'))
                continue
            need(free >= 4 * 2**30, 'Insufficient atomic-save disk margin')
            actions.append(dict(job=job, gpu=gpu, state='ready'))
            if execute:
                q.launch(job)  # Durable intent, PID/startticks/cmd/source/config receipt.
                statuses[job] = 'running'
                actions[-1]['state'] = 'launched'
        report = dict(utc=api.utc(), partition=q.manifest['partition'],
            recovery_manifest_sha256=q.manifest_sha, source_manifest=q.manifest['source_manifest'],
            helper_sha256=self.helper_sha, statuses=statuses, actions=actions,
            all_complete=all(v == 'complete' for v in statuses.values()), disk_free_bytes=free,
            gpu_snapshot=snapshot)
        if execute:
            api.write(q.state / 'latest.json', report)
        return report

    def run(self, poll_seconds=30):
        need(1 <= poll_seconds <= 60, 'Polling interval must be between 1 and 60 seconds')
        with self.q.exclusive():
            try:
                while True:
                    report = self.cycle(True)
                    print(json.dumps(report, allow_nan=False), flush=True)
                    if report['all_complete']:
                        return
                    time.sleep(poll_seconds)
            except Exception as error:
                self.api.write(self.q.state / 'failure.json', dict(utc=self.api.utc(),
                    type=type(error).__name__, message=str(error),
                    recovery_manifest_sha256=self.q.manifest_sha))
                raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--partition', choices=GROUPS, required=True)
    parser.add_argument('--recovery-root', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--prepare', action='store_true', help='Prepare only (default)')
    mode.add_argument('--plan', action='store_true', help='Read-only observation of an already prepared recovery')
    mode.add_argument('--execute', action='store_true', help='Launch/adopt only the already prepared recovery')
    parser.add_argument('--poll-seconds', type=float, default=30)
    args = parser.parse_args()
    if args.execute or args.plan:
        path = args.recovery_root.resolve() / 'manifest.json'
        manifest = read(path)
        need(manifest['partition'] == args.partition and manifest['source_manifest'] == bound(args.manifest.resolve()),
             'CLI source/partition does not match prepared recovery')
        controller = RecoveryController(path)
        if args.execute:
            controller.run(args.poll_seconds)
        else:
            print(json.dumps(controller.cycle(False), indent=2, allow_nan=False))
        return
    path = prepare(args.manifest, args.partition, args.recovery_root)
    print(json.dumps(dict(status='prepared_recovery_no_training', manifest=str(path), sha256=sha(path))))


if __name__ == '__main__':
    main()
