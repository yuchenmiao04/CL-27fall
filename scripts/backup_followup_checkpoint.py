#!/usr/bin/env python3
"""Back up one terminal follow-up run and execute its native CPU restore.

This utility runs on the experiment server. It does not train, evaluate test
accuracy, apply saved CUDA RNG, download weights, or transfer model files.
Formal jobs are bound to their prepared queue manifest and completion seal.
Small development fixtures require --validation-only and produce explicitly
non-formal evidence. Checkpoints are trusted local pickle files.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import datetime
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import tempfile
import time

WEIGHT_SHA = 'ac104c0df8158c46754510e50495536fa6de9647e572e38b5659dc54f260b124'
INITIAL_SHA = '1ece119a218036ffacbd3b576e1eb00a483915c88b21d81980c5e4e4a52e66e9'
WEIGHT_NAME = ('B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0'
               '--imagenet2012-steps_20k-lr_0.01-res_224.npz')
METADATA = ('config', 'results', 'data_manifest')
FORMAL_JOBS = {
    'formal_cifar_current_epochs20_v1': ('cifar224', 'current_adapted', 'epochs20'),
    'formal_imagenetr_current_epochs20_v1': ('imagenetr', 'current_adapted', 'epochs20'),
    'formal_cifar_tuna_native_v1': ('cifar224', 'tuna', 'pilot_native_same_protocol'),
    'formal_imagenetr_tuna_native_v1': ('imagenetr', 'tuna', 'pilot_native_same_protocol'),
}
DEAD = {'Z', 'X', 'x'}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def atomic_copy(source, target, expected_sha):
    """Publish a same-filesystem copy with link(2)'s no-overwrite guarantee."""
    source, target = Path(source), Path(target)
    need(source.is_file() and sha(source) == expected_sha, 'Source SHA differs before copy: ' + str(source))
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.' + target.name + '.', dir=target.parent)
    temporary = Path(temporary)
    try:
        with os.fdopen(descriptor, 'wb') as destination, source.open('rb') as original:
            shutil.copyfileobj(original, destination, 4 << 20)
            destination.flush()
            os.fsync(destination.fileno())
        need(sha(temporary) == expected_sha == sha(source), 'Source changed during copy: ' + str(source))
        os.link(temporary, target)  # Atomically fail rather than replace a target.
        need(sha(target) == expected_sha, 'Published backup SHA differs')
    finally:
        temporary.unlink(missing_ok=True)


def write_new_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    temporary = Path(temporary)
    try:
        with os.fdopen(descriptor, 'w') as stream:
            json.dump(value, stream, indent=2, ensure_ascii=True, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def probe_process(pid):
    root = Path('/proc') / str(pid)
    try:
        first = (root / 'stat').read_text().rsplit(')', 1)[1].split()
        command = [part.decode() for part in (root / 'cmdline').read_bytes().split(b'\0') if part]
        last = (root / 'stat').read_text().rsplit(')', 1)[1].split()
    except (FileNotFoundError, ProcessLookupError):
        return None
    need(first[19] == last[19], 'PID identity changed during observation')
    return dict(startticks=int(last[19]), state=last[0], command=command,
                exit_code=os.waitstatus_to_exitcode(int(last[49])) if last[0] in DEAD else None)


def terminal_process(receipt, *, probe=probe_process, seal=None, validation_only=False):
    pid = receipt.get('pid')
    ticks = receipt.get('proc_start_ticks', receipt.get('start_ticks'))
    command = receipt.get('command')
    need(type(pid) is int and pid > 0 and type(ticks) is int and ticks > 0
         and isinstance(command, list) and bool(command)
         and all(isinstance(part, str) and part for part in command), 'Invalid launch process identity')
    observed = probe(pid)
    if observed is None:
        state, code = 'absent', None
    elif observed['startticks'] != ticks:
        state, code = 'original_pid_reused', None
    elif observed['state'] in DEAD:
        state, code = observed['state'], observed['exit_code']
        need(code == 0, 'Original process has no successful observed exit code')
        need(not observed['command'] or observed['command'] == command,
             'Terminal original PID command differs from its receipt')
    else:
        # Empty argv is possible during teardown of a still-live process.
        # Neither it nor a completed JSON result permits early finalization.
        need(not observed['command'] or observed['command'] == command,
             'Live original PID command differs from its receipt')
        raise ValueError('Original source process is still active; no backup finalization')
    if not validation_only:
        need(isinstance(seal, dict) and seal.get('job') == receipt.get('job'),
             'Formal source requires its metadata-bound queue completion seal')
        need(seal.get('original_process_state') in DEAD | {'absent', 'original_pid_reused'},
             'Queue did not record original process termination')
        need(seal.get('observed_exit_code') in (None, 0), 'Queue recorded a failed source exit')
    return dict(pid=pid, proc_start_ticks=ticks, command=command,
                observed_state=state, observed_exit_code=code,
                exit_code_known=code is not None,
                original_process_terminal_verified=True,
                queue_completion_seal_required=not validation_only,
                observed_utc=utc())


def metadata_guard(values, *, validation_only):
    cfg, result, data = (values[name] for name in METADATA)
    need(cfg['model_name'] in {'current_adapted', 'tuna'} and cfg['dataset'] in {'cifar224', 'imagenetr'},
         'Only the two follow-up methods and two specified datasets are supported')
    need(type(cfg['seed']) is int and cfg['seed'] == 1993 and cfg.get('shuffle') is True,
         'Use the exact executed configuration with scalar seed1993')
    need(cfg.get('pretrained') is True and not cfg.get('smoke', False)
         and cfg.get('download') is False and cfg.get('save_checkpoints') is True,
         'Expected existing pretrained weights and saved native checkpoints')
    need(result['method'] == cfg['model_name'] and result['phase'] == cfg['phase'] == data['phase'],
         'Method/data/phase metadata disagree')
    width = 10 if cfg['dataset'] == 'cifar224' else 20
    need((cfg['init_cls'], cfg['increment']) == (width, width), 'Class stream widths differ')
    expected_backbone = 'vit_base_patch16_224_tuna' if cfg['model_name'] == 'tuna' else 'vit_base_patch16_224'
    need(cfg['backbone_type'] == expected_backbone, 'Wrong pretrained backbone branch')
    if validation_only:
        need(cfg.get('validation_only') is True and cfg['phase'] == 'development'
             and result['status'] in {'validation_partial', 'complete'}
             and result.get('performance_claim') is False
             and data['evaluation_source'] == 'training_holdout' and data['official_test_read'] is False,
             '--validation-only requires an explicitly labeled training-holdout fixture')
    else:
        need(cfg['phase'] == 'final' and result['status'] == 'complete'
             and cfg.get('validation_only') is False
             and data['evaluation_source'] == 'designated_test' and data['official_test_read'] is True,
             'Only a complete final run can produce formal checkpoint evidence')
        need(all(cfg.get(key) is None for key in ('resume', 'max_tasks', 'validation_max_per_class', 'joint_max_classes')),
             'Formal follow-up must be a fresh complete stream')
        need(len(result['stages']) == result['metrics']['completed_tasks'] == 10, 'Formal stream is incomplete')
        n, evaluation = (50000, 10000) if cfg['dataset'] == 'cifar224' else (24000, 6000)
        need(data['train_count'] == n and data['evaluation_count'] == evaluation
             and result['stages'][-1]['evaluation_images'] == evaluation, 'Full-data counts differ')
        if cfg['model_name'] == 'current_adapted':
            need(cfg['epochs'] == 20 and cfg['comparison_variant'] == 'epochs20', 'Not the requested20-epoch variant')
        else:
            need(cfg['comparison_variant'] == 'pilot_native_same_protocol', 'Not the native TUNA protocol variant')
    stages = result['stages']
    need(stages and len(stages) == result['metrics']['completed_tasks']
         and [row['task'] for row in stages] == list(range(len(stages)))
         and all(row['seen_classes'] == width * (i + 1)
                 and row['prediction_space'] == 'all_seen_classes_without_task_id'
                 for i, row in enumerate(stages)), 'Malformed completed stage records')


def source_guard(repo, inventory):
    repo = Path(repo).resolve(strict=True)
    need(bool(inventory), 'Missing executed-source inventory')
    for name, expected in inventory.items():
        path = (repo / name).resolve(strict=True)
        need(path.is_relative_to(repo) and sha(path) == expected, 'Executed source changed: ' + name)


def admit(manifest_path, job, *, validation_only=False, run_dir=None,
          receipt_path=None, repo=None, pretrained_npz=None, probe=probe_process):
    """Bind metadata to launch identity before copying any model payload."""
    need(isinstance(job, str) and job and Path(job).name == job and job not in {'.', '..'}, 'Invalid job name')
    manifest = read(manifest_path) if manifest_path else None
    manifest_digest = sha(manifest_path) if manifest_path else None
    row, seal, inventory = None, None, None
    if not validation_only:
        need(manifest is not None and job in FORMAL_JOBS, 'Formal backup requires a prepared manifest and requested job')
        rows = [r for r in manifest['jobs'] if r['job'] == job]
        need(len(rows) == 1, 'Missing or ambiguous manifest job')
        row = rows[0]
        need((row['dataset'], row['method'], row['comparison_variant']) == FORMAL_JOBS[job], 'Manifest job identity differs')
        root = Path(manifest['campaign_root']).resolve(strict=True)
        repo = Path(manifest['repo']).resolve(strict=True)
        run_dir = Path(row['output_dir']).resolve(strict=True)
        need(run_dir == root / 'formal/runs' / job, 'Manifest output escaped its campaign')
        receipt_path = root / 'formal/receipts' / (job + '.json')
        seal = read(root / 'queue' / (job + '.complete.json'))
        pretrained_npz = Path(manifest['pretrained_npz']['path'])
        need(manifest['pretrained_npz']['sha256'] == WEIGHT_SHA, 'Manifest weight identity differs')
        inventory = manifest['source_files_sha256']
        source_guard(repo, inventory)
    else:
        need(manifest is None, 'Validation mode uses explicit paths, never a formal campaign manifest')
        need(all(x is not None for x in (run_dir, receipt_path, repo, pretrained_npz)), 'Validation requires run, receipt, source and NPZ paths')
        repo, run_dir = Path(repo).resolve(strict=True), Path(run_dir).resolve(strict=True)
    receipt_digest = sha(receipt_path)
    receipt = read(receipt_path)
    need(sha(receipt_path) == receipt_digest, 'Receipt changed during admission')
    need(receipt.get('job') == job, 'Launch receipt job differs')
    launch_repo = receipt.get('cwd', receipt.get('source_root'))
    need(launch_repo is not None and Path(launch_repo).resolve(strict=True) == repo, 'Launch source root differs')
    values = {name: read(run_dir / (name + '.json')) for name in METADATA}
    digests = {name: sha(run_dir / (name + '.json')) for name in METADATA}
    metadata_guard(values, validation_only=validation_only)
    if row is not None:
        need(receipt['manifest_sha256'] == manifest_digest
             and receipt['config_sha256'] == row['config']['sha256']
             and receipt['command'] == row['command']
             and receipt['gpu'] == row['gpu']
             and receipt['source_commit'] == manifest.get('source_commit')
             and receipt['source_inventory_sha256'] == hashlib.sha256(json.dumps(inventory, sort_keys=True).encode()).hexdigest(),
             'Receipt is not bound to the prepared command/config/source')
        need(sha(row['config']['path']) == row['config']['sha256'], 'Prepared configuration changed')
        declared = dict(read(row['config']['path']), seed=1993)
        saved = {k: v for k, v in values['config'].items() if k != 'config'}
        need(saved == declared, 'Saved configuration differs from prepared recipe')
        need(seal['config_sha256'] == digests['config'] and seal['result_sha256'] == digests['results']
             and seal['data_manifest_sha256'] == digests['data_manifest'], 'Queue completion metadata changed')
    else:
        if receipt.get('output_dir') is not None:
            need(Path(receipt['output_dir']).resolve(strict=True) == run_dir, 'Validation output path differs')
        config_path = receipt.get('config', values['config'].get('config'))
        need(config_path is not None and str(config_path) in receipt['command'], 'Validation launch config is not bound to command')
        declared = read(config_path)
        if isinstance(declared.get('seed'), list):
            need(declared['seed'] == [1993], 'Wrong validation seed list')
            declared['seed'] = 1993
        need({k: v for k, v in values['config'].items() if k != 'config'} == declared,
             'Saved validation config differs from launch recipe')
    terminal = terminal_process(receipt, probe=probe, seal=seal, validation_only=validation_only)
    pretrained_npz = Path(os.path.abspath(pretrained_npz))
    need(pretrained_npz.name == WEIGHT_NAME and pretrained_npz.parent.name == 'checkpoints'
         and pretrained_npz.is_file() and sha(pretrained_npz) == WEIGHT_SHA,
         'Exact local native NPZ cache is missing or changed; no download')
    checkpoint = run_dir / 'checkpoint.pt'
    need(checkpoint.is_file(), 'Completed run has no final checkpoint')
    return dict(job=job, repo=repo, run_dir=run_dir, receipt_path=Path(receipt_path), receipt=receipt,
                receipt_sha256=receipt_digest,
                seal=seal, manifest_path=Path(manifest_path) if manifest_path else None,
                manifest_sha256=manifest_digest, source_inventory=inventory,
                metadata=values, metadata_sha256=digests, pretrained_npz=pretrained_npz,
                checkpoint=checkpoint, checkpoint_sha256=sha(checkpoint),
                checkpoint_bytes=checkpoint.stat().st_size, source_terminal=terminal)


def load_comparator(path, repo):
    spec = importlib.util.spec_from_file_location('followup_independent_state_comparator', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # The independent comparator inserts its own repository for its CLI. Native
    # deserialization/construction must instead use the actual executed snapshot.
    sys.path[:] = [str(repo)] + [item for item in sys.path if item != str(repo)]
    return module


def compare_learner(reference, candidate, comparator):
    comparison = comparator.Comparison()
    comparison.walk(reference, candidate, ('learner',))
    need(not comparison.differences, 'Native learner restore differs: ' + json.dumps(comparison.differences[:3]))
    return dict(compared_nodes=comparison.compared_nodes,
                reference_semantic_sha256=comparator.hash_state(reference, ('learner',)).hex(),
                restored_semantic_sha256=comparator.hash_state(candidate, ('learner',)).hex(),
                ignored_paths=sorted(comparison.ignored_paths), exact_typed_bitwise_comparison=True)


def finite_scan(value, np, torch, counts=None):
    counts = dict(tensors=0, arrays=0, scalars=0) if counts is None else counts
    if isinstance(value, torch.Tensor):
        need(value.device.type == 'cpu', 'Checkpoint tensor escaped CPU map_location')
        if value.is_floating_point() or value.is_complex():
            need(bool(torch.isfinite(value).all()), 'Nonfinite checkpoint tensor')
        counts['tensors'] += 1
    elif isinstance(value, np.ndarray):
        if value.dtype.kind in 'fc':
            need(bool(np.isfinite(value).all()), 'Nonfinite checkpoint array')
        elif value.dtype.hasobject:
            for item in value.flat:
                finite_scan(item, np, torch, counts)
        counts['arrays'] += 1
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        for child in vars(value).values():
            finite_scan(child, np, torch, counts)
    elif isinstance(value, dict):
        for child in value.values():
            finite_scan(child, np, torch, counts)
    elif isinstance(value, (list, tuple)):
        for child in value:
            finite_scan(child, np, torch, counts)
    elif isinstance(value, (float, np.floating)):
        need(math.isfinite(float(value)), 'Nonfinite checkpoint scalar')
        counts['scalars'] += 1
    return counts


def native_restore(admission, checkpoint, comparator_path):
    """Instantiate the exact method on CPU and compare every persisted leaf."""
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    sys.dont_write_bytecode = True
    repo = admission['repo']
    sys.path[:] = [str(repo)] + [item for item in sys.path if item != str(repo)]
    comparator = load_comparator(comparator_path, repo)
    import numpy as np
    import torch
    from cl27.data import ProtocolDataManager
    from cl27.runner import configuration_fingerprint, validate_checkpoint_random_stream
    from utils import factory
    need(not torch.cuda.is_initialized(), 'CPU audit must start without an initialized CUDA context')
    torch.set_num_threads(4)
    torch.hub.set_dir(str(admission['pretrained_npz'].parent.parent))
    state, digest = comparator.load_trusted_checkpoint(checkpoint)
    need(digest == admission['checkpoint_sha256'], 'Loaded backup SHA differs')
    finite_counts = finite_scan(state, np, torch)
    values = admission['metadata']
    cfg, result, saved_manifest = (values[name] for name in METADATA)
    validate_checkpoint_random_stream(state)
    need(state['config_fingerprint'] == result['config_fingerprint'] == configuration_fingerprint(cfg),
         'Saved configuration fingerprint differs')
    need(state['data_fingerprint'] == result['data_fingerprint'] == saved_manifest['fingerprint'], 'Data fingerprint differs')
    # Parse global RNG snapshots on independent CPU generators only. Do not call
    # runner.restore_rng: it would apply CUDA states and this is a CPU audit.
    rng = state['rng']
    random.Random().setstate(rng['python'])
    np.random.RandomState().set_state(rng['numpy'])
    torch.Generator(device='cpu').set_state(rng['torch'])
    need(rng['cuda'] is None or (isinstance(rng['cuda'], list) and bool(rng['cuda'])
         and all(isinstance(v, torch.Tensor) and v.device.type == 'cpu' and v.dtype == torch.uint8 and v.numel() > 0
                 for v in rng['cuda'])), 'Malformed saved CUDA RNG payload')
    data_args = copy.deepcopy(cfg)
    data_args['download'] = False
    dm = ProtocolDataManager(cfg['dataset'], cfg['shuffle'], 1993, cfg['init_cls'], cfg['increment'], data_args)
    need(dm.manifest == saved_manifest, 'Manifest rebuilt from source bytes differs')
    learner = state['learner']
    tasks = len(result['stages'])
    seen = sum(dm.get_task_size(i) for i in range(tasks))
    need(learner['method'] == cfg['model_name'] and learner['cur_task'] == tasks - 1
         and learner['known_classes'] == learner['total_classes'] == seen
         and len(state['stages']) == len(learner['task_diagnostics']) == tasks, 'Checkpoint endpoint differs')
    from cl27.common import jsonable
    need(jsonable(state['stages']) == result['stages']
         and jsonable(learner['task_diagnostics']) == result['task_diagnostics'], 'Checkpoint/result stage records differ')
    expected_counts = np.bincount(dm._train_targets, minlength=dm.nb_classes)[:seen]
    expected_rows = [int(expected_counts[i * cfg['increment']:(i + 1) * cfg['increment']].sum()) for i in range(tasks)]
    core = learner['core']
    if cfg['model_name'] == 'tuna':
        need(core['raw_replay_images'] == 0, 'TUNA checkpoint contains image replay')
        need(core['class_counts'] == {i: int(n) for i, n in enumerate(expected_counts)}, 'TUNA class arrival counts differ')
        need(len(core['arrived_ids']) == tasks, 'TUNA arrival ledger is incomplete')
        for task, recorded in enumerate(core['arrived_ids']):
            labels = range(task * cfg['increment'], (task + 1) * cfg['increment'])
            expected = np.concatenate([dm._train_ids[dm._train_targets == c] for c in labels])
            need(np.array_equal(recorded, expected), 'TUNA arrivals differ from actual ordered training IDs')
    else:
        engine, memory = core['engine'], core['memory']
        need(engine['completed_tasks'] == tasks and engine['task_sizes'] == expected_rows
             and engine['statistics']['count'] == int(expected_counts.sum()), 'Fixed-library task/history counts differ')
        need(np.array_equal(engine['statistics']['C'][-1].numpy(), expected_counts), 'Fixed-library class moment counts differ')
        need(memory['seen'] == int(expected_counts.sum()) and memory['classes'] == list(range(seen)), 'Fixed-library replay history differs')
        actual_labels = {int(i): int(c) for i, c in zip(dm._train_ids, dm._train_targets) if c < seen}
        row_index = {int(i): position for position, i in enumerate(dm._train_ids)}
        ids = [int(row.sample_id) for row in memory['records']]
        need(len(ids) == len(set(ids)) and set(ids) == set(engine['memory_base'])
             and all(actual_labels.get(int(row.sample_id)) == int(row.label) for row in memory['records']),
             'Fixed-library replay membership/cache differs')
        for record in memory['records']:
            original = dm._train_data[row_index[int(record.sample_id)]]
            if isinstance(record.image, np.ndarray):
                need(record.image.dtype == np.uint8 and np.array_equal(record.image, original),
                     'Replay RGB bytes differ from their original training row')
            else:
                recorded = Path(str(record.image))
                actual = Path(str(original))
                if not recorded.is_absolute():
                    recorded = repo / recorded
                if not actual.is_absolute():
                    actual = repo / actual
                need(recorded.resolve(strict=True) == actual.resolve(strict=True),
                     'Replay image path differs from its original training row')
    random.seed(1993)
    np.random.seed(1993)
    torch.random.default_generator.manual_seed(1993)
    native_args = copy.deepcopy(cfg)
    native_args.update(device=[torch.device('cpu')], seed=1993, nb_classes=dm.nb_classes, nb_tasks=dm.nb_tasks)
    model = factory.get_model(cfg['model_name'], native_args)
    need(Path(sys.modules[type(model).__module__].__file__).resolve().is_relative_to(repo), 'Wrong native learner source snapshot')
    provenance = model.backbone_provenance
    need(provenance['weight_file_sha256'] == WEIGHT_SHA and provenance['initial_state_sha256'] == INITIAL_SHA,
         'Fresh native initial model did not use the specified NPZ')
    if cfg['model_name'] == 'current_adapted':
        model.canonical_transform.transform = dm.get_dataset(np.arange(seen), source='train', mode='test').trsf
        model.training_transform.transform = dm.get_dataset(np.arange(seen), source='train', mode='train').trsf
    model.restore_checkpoint(learner)
    restored = model.checkpoint_state()
    comparison = compare_learner(learner, restored, comparator)
    need(all(parameter.device.type == 'cpu' for parameter in model._network.parameters()), 'Native learner escaped CPU')
    optimizer_bindings = {}
    if cfg['model_name'] == 'current_adapted':
        names = {id(parameter): name for name, parameter in model.core.encoder.named_parameters()}
        need(not any(p.requires_grad for p in model.core.encoder.anchor.parameters()), 'Restored anchor is trainable')
        for name, optimizer in model.core.optimizers.items():
            bound = []
            for group in optimizer.param_groups:
                for parameter in group['params']:
                    key = names.get(id(parameter), '')
                    need(key.startswith('libraries.' if name == 'modules' else 'main.'), 'Optimizer is bound to the wrong parameter family')
                    moment = optimizer.state[parameter]
                    need(moment['exp_avg'].shape == moment['exp_avg_sq'].shape == parameter.shape
                         and moment['exp_avg'].device.type == moment['exp_avg_sq'].device.type == 'cpu',
                         'Restored optimizer moments/parameter binding differ')
                    bound.append(key)
            optimizer_bindings[name] = bound
    # File hashes also cover global RNG and serialization data which a learner
    # does not own. Native restore comparison never exempts optimizer or modes.
    need(sha(checkpoint) == digest and not torch.cuda.is_initialized(), 'Read-only CPU audit changed input/CUDA state')
    imported = {}
    for name, module in list(sys.modules.items()):
        if name.split('.')[0] not in {'cl27', 'models', 'utils', 'backbone'}:
            continue
        path = vars(module).get('__file__') if hasattr(module, '__dict__') else None
        if isinstance(path, str) and Path(path).is_absolute():
            path = Path(path).resolve()
            need(path.is_relative_to(repo), 'Native module imported outside executed source: ' + name)
            if path.suffix == '.py':
                imported[str(path.relative_to(repo))] = sha(path)
    if admission['source_inventory'] is not None:
        source_guard(repo, admission['source_inventory'])
        need(all(admission['source_inventory'].get(name) == digest for name, digest in imported.items()),
             'Imported native source was absent from the prebound inventory')
    return dict(native_cpu_restore_executed=True, **comparison,
                actual_training_rows=int(expected_counts.sum()), actual_seen_classes=seen,
                source_data_manifest_rebuilt=True, source_training_arrivals_verified=True,
                native_optimizer_parameter_bindings=optimizer_bindings,
                persisted_mode_map_compared=cfg['model_name'] == 'tuna',
                mode_scope=('All stored module_training/requires_grad leaves compared exactly' if cfg['model_name'] == 'tuna'
                            else 'Current native checkpoint has no module-mode map; no unrecorded mode recovery claim'),
                global_rng_payloads_validated_on_cpu=True,
                saved_cuda_rng_payload_lengths=None if rng['cuda'] is None else [v.numel() for v in rng['cuda']],
                cuda_rng_applied=False, extra_training_steps=0, accuracy_evaluation_performed=False,
                finite_scan=finite_counts,
                runtime_source_files_sha256=imported,
                state_comparator_sha256=sha(comparator_path), pretrained_npz_sha256=WEIGHT_SHA)


def backup(admission, backup_root, evidence_root, *, validation_only=False,
           comparator_path=None, restore=native_restore, probe=probe_process):
    began = time.perf_counter()
    job = admission['job']
    backup_root, evidence_root = Path(backup_root), Path(evidence_root)
    destination, evidence = backup_root / job, evidence_root / (job + '.json')
    need(not destination.exists() and not evidence.exists(), 'Refusing an existing backup/evidence; never overwrite')
    backup_root.mkdir(parents=True, exist_ok=True)
    evidence_root.mkdir(parents=True, exist_ok=True)
    with (backup_root / ('.' + job + '.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        need(not destination.exists() and not evidence.exists(), 'Backup appeared after admission')
        # Keep an incomplete attempt visibly separate from a finalized job.
        temporary = Path(tempfile.mkdtemp(prefix='.' + job + '.preparing.', dir=backup_root))
        try:
            for name in METADATA:
                atomic_copy(admission['run_dir'] / (name + '.json'), temporary / (name + '.json'), admission['metadata_sha256'][name])
            atomic_copy(admission['receipt_path'], temporary / 'source_receipt.json', admission['receipt_sha256'])
            if admission['manifest_path']:
                atomic_copy(admission['manifest_path'], temporary / 'source_manifest.json', admission['manifest_sha256'])
            if admission['seal'] is not None:
                write_new_json(temporary / 'source_completion_seal.json', admission['seal'])
            atomic_copy(admission['checkpoint'], temporary / 'checkpoint.pt', admission['checkpoint_sha256'])
            comparator_path = Path(comparator_path or Path(__file__).with_name('compare_checkpoint_state.py')).resolve(strict=True)
            report = restore(admission, temporary / 'checkpoint.pt', comparator_path)
            after = terminal_process(admission['receipt'], probe=probe, seal=admission['seal'], validation_only=validation_only)
            need(all(sha(admission['run_dir'] / (name + '.json')) == value for name, value in admission['metadata_sha256'].items())
                 and sha(admission['checkpoint']) == admission['checkpoint_sha256']
                 and sha(admission['receipt_path']) == admission['receipt_sha256'], 'Terminal source files changed during audit')
            write_new_json(temporary / 'native_restore.json', report)
            # The exclusive per-job lock and both existence checks protect the
            # directory publication; each model/metadata file was itself linked
            # atomically with a strict no-replacement operation above.
            need(not destination.exists(), 'Backup destination appeared during audit')
            os.rename(temporary, destination)
            payload = dict(schema_version=1, job=job, validation_only=validation_only,
                           formal_checkpoint_evidence=not validation_only,
                           checkpoint_sha256=admission['checkpoint_sha256'],
                           checkpoint_bytes=admission['checkpoint_bytes'],
                           backup_sha256=sha(destination / 'checkpoint.pt'),
                           native_restore_verified=True, source_process_terminal_verified=True,
                           source_metadata_sha256=admission['metadata_sha256'],
                           backup_checkpoint_path=str(destination / 'checkpoint.pt'),
                           source_run_dir=str(admission['run_dir']),
                           source_receipt_sha256=admission['receipt_sha256'],
                           source_manifest_sha256=admission['manifest_sha256'],
                           source_terminal_before=admission['source_terminal'], source_terminal_after=after,
                           restore_report=report, restore_report_path=str(destination / 'native_restore.json'),
                           restore_report_sha256=sha(destination / 'native_restore.json'),
                           auditor_sha256=sha(Path(__file__)), audit_wall_seconds=time.perf_counter() - began,
                           created_utc=utc())
            need(payload['checkpoint_sha256'] == payload['backup_sha256'], 'Final backup identity differs')
            write_new_json(evidence, payload)
            return payload
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--job', required=True)
    parser.add_argument('--backup-root', required=True, type=Path)
    parser.add_argument('--evidence-root', required=True, type=Path)
    parser.add_argument('--validation-only', action='store_true')
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--receipt', type=Path)
    parser.add_argument('--repo', type=Path)
    parser.add_argument('--pretrained-npz', type=Path)
    parser.add_argument('--state-loader', type=Path)
    args = parser.parse_args()
    admission = admit(args.manifest, args.job, validation_only=args.validation_only,
                      run_dir=args.run_dir, receipt_path=args.receipt, repo=args.repo,
                      pretrained_npz=args.pretrained_npz)
    result = backup(admission, args.backup_root, args.evidence_root,
                    validation_only=args.validation_only, comparator_path=args.state_loader)
    print(json.dumps(dict(job=result['job'], native_restore_verified=True,
                         source_process_terminal_verified=True, checkpoint_sha256=result['checkpoint_sha256'],
                         validation_only=args.validation_only,
                         evidence=str(Path(args.evidence_root) / (args.job + '.json'))), allow_nan=False))


if __name__ == '__main__':
    main()
