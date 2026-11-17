"""CPU-only native restore audit of one explicitly finalized SHA-bound backup.

Use the executed model/data snapshot via --repo and the independent trusted
checkpoint comparator via --state-loader. No training, CUDA restore, checkpoint
write, accuracy evaluation or data download is performed. A zero-image forward
checks deployed output shape/finiteness only. Final campaign audit is separate.
"""
from __future__ import annotations
import argparse
import copy
import dataclasses
import hashlib
import importlib.util
from importlib.machinery import ModuleSpec
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from types import ModuleType
from urllib.parse import urlsplit

WEIGHT_SHA = 'ac104c0df8158c46754510e50495536fa6de9647e572e38b5659dc54f260b124'
POLICY = 'canonical_loader_isolated_v1'


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text())


def runtime_source_inventory(repo, modules):
    """Read static module metadata; never ask a dynamic namespace for __file__.

    Relative/virtual module filenames do not become campaign source paths just
    because the audit cwd happens to be the runtime root. An absolute campaign
    Python source claim must still exist; disappearance is an audit failure.
    """
    repo = Path(repo).resolve()
    sources, excluded = {}, []
    for name, imported in list(modules.items()):
        if not isinstance(imported, ModuleType):
            excluded.append({'module':name, 'reason':'non_module_or_virtual_entry'})
            continue
        namespace = vars(imported)
        filename = namespace.get('__file__')
        spec = namespace.get('__spec__')
        origin = vars(spec).get('origin') if isinstance(spec, ModuleSpec) else None
        absolute = {Path(value) for value in (filename, origin)
                    if isinstance(value, str) and value and Path(value).is_absolute()}
        if not absolute:
            excluded.append({'module':name, 'reason':'relative_virtual_or_no_file_origin',
                             'file':filename if isinstance(filename,str) else None,
                             'origin':origin if isinstance(origin,str) else None})
            continue
        for path in sorted(absolute):
            # abspath is lexical: retain a real REPO claim even if its target
            # has gone missing, instead of silently dropping it after resolve.
            path = Path(os.path.abspath(path))
            if not (path.is_relative_to(repo) or path.resolve().is_relative_to(repo)):
                if not path.is_file():
                    excluded.append({'module':name, 'reason':'external_missing_non_campaign_file', 'path':str(path)})
                continue
            if path.suffix == '.pyc':
                try:
                    path = Path(importlib.util.source_from_cache(str(path)))
                except ValueError:
                    path = path.with_suffix('.py')
            if path.suffix != '.py':
                excluded.append({'module':name, 'reason':'not_python_source', 'path':str(path)})
                continue
            need(path.is_file(), 'Imported runtime Python source is missing: '+str(path))
            resolved = path.resolve(strict=True)
            need(resolved.is_relative_to(repo), 'Imported runtime source escapes its snapshot: '+str(path))
            sources[str(resolved.relative_to(repo))] = sha(resolved)
    return sources, excluded


def config_guard(cfg, result):
    need(cfg['model_name'] in {'er', 'history_transport', 'plastic_shared_basis', 'current_adapted'}, 'Not one of the remaining eight formal cells')
    need(cfg['dataset'] in {'cifar224', 'imagenetr'}, 'Unexpected formal dataset')
    need(cfg.get('phase') == result.get('phase') == 'final' and result.get('status') == 'complete', 'Only a completed final result is admissible')
    need(cfg.get('seed') == 1993 and cfg.get('shuffle') is True, 'Use exact executed config with scalar seed1993')
    width = 10 if cfg['dataset'] == 'cifar224' else 20
    need(cfg.get('pretrained') is True and cfg.get('backbone_type') in {'vit_base_patch16_224', 'pretrained_vit_b16_224'}, 'Wrong pretrained backbone')
    need((cfg.get('init_cls'), cfg.get('increment')) == (width, width), 'Wrong ten-stage class widths')
    need(not cfg.get('validation_only', False) and not cfg.get('smoke', False), 'Development/smoke checkpoint is inadmissible')
    need(all(cfg.get(k) is None for k in ('max_tasks', 'resume', 'validation_max_per_class', 'joint_max_classes')), 'Formal audit forbids truncated/resumed/capped runs')


def global_rng_guard(state, np, torch):
    rng = state['rng']
    random.Random().setstate(rng['python'])
    np.random.RandomState().set_state(rng['numpy'])
    torch.Generator(device='cpu').set_state(rng['torch'])
    need(isinstance(rng['cuda'], list) and len(rng['cuda']) == 2, 'Expected two saved logical CUDA RNG payloads')
    need(all(isinstance(x, torch.Tensor) and x.device.type == 'cpu' and x.dtype == torch.uint8 and x.numel() > 0 for x in rng['cuda']), 'Malformed saved CUDA RNG bytes')
    return [x.numel() for x in rng['cuda']]


def finite_scan(value, np, torch, counts=None):
    counts = {'floating_tensors': 0, 'floating_arrays': 0} if counts is None else counts
    if isinstance(value, torch.Tensor) and (value.is_floating_point() or value.is_complex()):
        need(value.device.type == 'cpu' and bool(torch.isfinite(value).all()), 'Nonfinite/non-CPU checkpoint tensor')
        counts['floating_tensors'] += 1
    elif isinstance(value, np.ndarray) and value.dtype.kind in 'fc':
        need(bool(np.isfinite(value).all()), 'Nonfinite checkpoint array')
        counts['floating_arrays'] += 1
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        for field in dataclasses.fields(value):
            finite_scan(getattr(value, field.name), np, torch, counts)
    elif isinstance(value, dict):
        for child in value.values():
            finite_scan(child, np, torch, counts)
    elif isinstance(value, (list, tuple)):
        for child in value:
            finite_scan(child, np, torch, counts)
    elif isinstance(value, (float, np.floating)):
        need(math.isfinite(float(value)), 'Nonfinite checkpoint scalar')
    return counts


def checkpoint_facts(state, cfg, result, manifest, np, torch):
    method, dataset = cfg['model_name'], cfg['dataset']
    learner, core = state['learner'], state['learner']['core']
    n, c, replay_total = (50000, 100, 2000) if dataset == 'cifar224' else (24000, 200, 4000)
    need(state['version'] == 1 and state['random_stream_policy'] == POLICY, 'Checkpoint version/RNG policy mismatch')
    need(state['data_fingerprint'] == result['data_fingerprint'] == manifest['fingerprint'], 'Data fingerprints differ')
    ignored = {'output_dir', 'resume', 'max_tasks', 'config', 'device', 'num_workers', 'cpu_threads'}
    expected = hashlib.sha256(json.dumps({k:v for k,v in cfg.items() if k not in ignored}, sort_keys=True).encode()).hexdigest()
    need(state['config_fingerprint'] == expected == result['config_fingerprint'], 'Executed configuration fingerprint differs')
    need(learner['method'] == method and learner['cur_task'] == 9 and learner['known_classes'] == learner['total_classes'] == c, 'Checkpoint is not the full formal endpoint')
    need(len(state['stages']) == len(learner['task_diagnostics']) == 10, 'Need ten actual completed CIL stages')
    width = c // 10
    stage_replay = []
    for i, diagnostic in enumerate(learner['task_diagnostics']):
        seen = (i + 1) * width
        quota = 20 * seen if dataset == 'imagenetr' else seen * (2000 // seen)
        need(diagnostic['replay_images'] == quota, 'Recorded per-stage replay quota differs')
        need(state['stages'][i]['seen_classes'] == seen, 'Recorded per-stage output vocabulary differs')
        stage_replay.append(quota)
    replay = core['memory'] if method == 'current_adapted' else core['replay']
    need(replay['seen'] == n and replay['classes'] == list(range(c)), 'Replay/history count/vocabulary mismatch')
    need(len(replay['records']) == replay_total, 'Wrong final replay budget')
    ids = [int(row.sample_id) for row in replay['records']]
    need(len(set(ids)) == len(ids) and all(sum(row.label == label for row in replay['records']) == 20 for label in range(c)), 'Replay IDs/quota mismatch')
    generator = np.random.default_rng(); generator.bit_generator.state = replay['rng']
    facts = {'method':method, 'dataset':dataset, 'history_rows':n, 'classes':c, 'final_replay_rows':len(ids),
             'stage_replay_rows':stage_replay,
             'cuda_rng_payload_lengths':global_rng_guard(state, np, torch), 'finite_scan':finite_scan(state, np, torch)}
    if method == 'er':
        need(core['completed_tasks'] == len(core['task_arrivals']) == 10 and sum(core['task_arrivals']) == n, 'ER arrival/task counts differ')
        need(sum(core['task_class_sizes']) == c, 'ER class/task counts differ')
        generator.bit_generator.state = core['rng']
        facts['task_arrivals'] = core['task_arrivals']
    elif method in {'history_transport', 'plastic_shared_basis'}:
        analytic = core['analytic']; transport = analytic['transport'] if method == 'plastic_shared_basis' else analytic
        need(transport['tasks'] == 10 and int(np.asarray(transport['counts']).sum()) == n and transport['classes'] == list(range(c)), 'Transport counts differ')
        generator.bit_generator.state = core['rng']
        if method == 'plastic_shared_basis':
            need(analytic['basis']['n_samples'] == n and analytic['basis']['tasks'] == 10 and np.array_equal(analytic['basis']['counts'], transport['counts']), 'Basis snapshot counts differ')
            basis = np.asarray(analytic['basis']['D'])
            facts['basis_activity'] = {'D_shape':list(basis.shape), 'D_norm':float(np.linalg.norm(basis)), 'actual_rank':int(np.linalg.matrix_rank(basis)), 'interpretation':'Activity only; legitimate collapse is reported, not excluded.'}
    else:
        engine = core['engine']
        need(engine['completed_tasks'] == len(engine['task_sizes']) == 10 and engine['statistics']['count'] == sum(engine['task_sizes']) == n, 'Current task/history counts differ')
        need(set(engine['memory_base']) == set(ids), 'Resident anchor cache/replay IDs differ')
        need(all(x.shape == (769,) and torch.isfinite(x).all() and float(x[-1]) == 1 for x in engine['memory_base'].values()), 'Resident anchor cache malformed')
        need(engine['updates'] == engine['module_updates'] + engine['backbone_updates'], 'Current update counters differ')
        for key in ('rng','kind_rng'):
            torch.Generator(device='cpu').set_state(engine[key])
        facts['task_arrivals'] = engine['task_sizes']
        facts['successful_updates'] = {k:engine[k] for k in ('updates','module_updates','backbone_updates','episode_draws')}
        facts['optimizers'] = {}
        need(set(engine['optimizers']) == {'modules','backbone'}, 'Current needs both persistent AdamW states')
        for name, optimizer in engine['optimizers'].items():
            parameters = [pid for group in optimizer['param_groups'] for pid in group['params']]
            need(parameters and len(parameters) == len(set(parameters)) and set(parameters) == set(optimizer['state']), 'Incomplete optimizer parameter states')
            steps = []
            for value in optimizer['state'].values():
                need({'step','exp_avg','exp_avg_sq'} <= set(value), 'Missing AdamW state')
                need(torch.isfinite(value['exp_avg']).all() and torch.isfinite(value['exp_avg_sq']).all() and value['exp_avg'].shape == value['exp_avg_sq'].shape, 'Invalid optimizer moments')
                step = float(value['step']); need(step > 0 and math.isfinite(step), 'Invalid AdamW step'); steps.append(step)
            facts['optimizers'][name] = {'parameters':len(parameters), 'min_step':min(steps), 'max_step':max(steps)}
    reuse = result.get('execution', {}).get('ce_training_reuse')
    if reuse:
        rows = reuse['stages']
        need([r['task'] for r in rows] == list(range(10)) and reuse['mode'] == ('record' if method == 'er' else 'replay'), 'CE reuse mode/stage scope differs')
        need(reuse['execution_role'] == ('fixed_baseline' if method == 'er' else 'candidate_method'), 'Wrong native CE driver role')
        facts['ce_cost_sources'] = {'trajectory_directory':reuse['trajectory_directory'], 'shared_native_ce_seconds':sum(r['shared_native_ce_seconds'] for r in rows),
            'executor_wall_seconds':sum(r['executor_wall_seconds'] for r in rows), 'actual_run_wall_seconds':result['execution']['wall_seconds'],
            'meaning':'Shared CE is driver work, not additional reader-executed gradient training; audit performs zero training.'}
    return facts, replay


def rebuild_data(cfg, saved_manifest, state, replay, data_root, runtime_cwd, facts, np):
    from cl27.data import ProtocolDataManager
    need(Path(sys.modules[ProtocolDataManager.__module__].__file__).resolve().is_relative_to(REPO), 'DataManager imported from a different source snapshot')
    original_root = Path(cfg['data_root'])
    if not original_root.is_absolute():
        original_root = runtime_cwd / original_root
    original_root = original_root.resolve()
    data_args = copy.deepcopy(cfg); data_args['download'] = False
    data_args['data_root'] = str(data_root or original_root)
    dm = ProtocolDataManager(cfg['dataset'], cfg['shuffle'], 1993, cfg['init_cls'], cfg['increment'], data_args)
    need(dm.manifest == saved_manifest and dm.manifest['fingerprint'] == state['data_fingerprint'], 'Manifest rebuilt from actual full source bytes differs')
    n, c = facts['history_rows'], facts['classes']
    ids, labels = np.asarray(dm._train_ids), np.asarray(dm._train_targets)
    need(len(ids) == n and len(np.unique(ids)) == n and np.array_equal(ids, np.arange(n, dtype=ids.dtype)), 'Full natural-order training IDs differ')
    need(dm.nb_tasks == 10 and dm.nb_classes == c and dm._class_order == np.random.RandomState(1993).permutation(c).tolist(), 'Class/stage protocol differs')
    need(hashlib.sha256(ids.tobytes()).hexdigest() == saved_manifest['fit_ids_sha256'], 'Natural-order ID hash differs')
    counts = np.bincount(labels, minlength=c)
    task_rows = [int(counts[t*cfg['increment']:(t+1)*cfg['increment']].sum()) for t in range(10)]
    if 'task_arrivals' in facts:
        need(facts['task_arrivals'] == task_rows, 'Actual per-task arrivals do not match official remapped labels')
    core = state['learner']['core']
    if cfg['model_name'] == 'current_adapted':
        need(np.array_equal(core['engine']['statistics']['C'][-1].numpy(), counts), 'Current frozen-anchor class-count row differs from official labels')
    elif cfg['model_name'] != 'er':
        analytic = core['analytic']; transport = analytic['transport'] if cfg['model_name'] == 'plastic_shared_basis' else analytic
        need(np.array_equal(np.asarray(transport['counts']), counts), 'Transport class counts differ from official labels')
    row_index = {int(sample):i for i,sample in enumerate(ids)}
    mapped_paths = 0
    for record in replay['records']:
        need(int(record.sample_id) in row_index, 'Replay sample ID is not a formal training member')
        i = row_index[int(record.sample_id)]
        need(int(record.label) == int(labels[i]), 'Replay label disagrees with actual training row')
        actual = dm._train_data[i]
        if isinstance(record.image, np.ndarray):
            need(record.image.dtype == np.uint8 and np.array_equal(record.image, actual), 'Replay RGB bytes disagree with actual training row')
        else:
            recorded = Path(str(record.image))
            if not recorded.is_absolute(): recorded = runtime_cwd / recorded
            relative = recorded.resolve().relative_to(original_root)
            resolved = (Path(data_args['data_root']) / relative).resolve()
            need(resolved == Path(str(actual)).resolve() and resolved.is_file(), 'Replay original training path/mapping differs or is missing')
            mapped_paths += int(resolved != recorded.resolve())
    return dm, {'manifest_rebuilt_from_actual_source_bytes':True, 'natural_training_ids_sha256':hashlib.sha256(ids.tobytes()).hexdigest(),
        'training_rows':n, 'per_class_counts':counts.tolist(), 'per_task_rows':task_rows,
        'replay_training_members_and_labels_verified':len(replay['records']), 'mapped_replay_paths':mapped_paths,
        'evaluation_performed':False, 'note':'DataManager hashes designated test source for identity only; no test prediction/selection is run.'}


def normalize_execution_metadata(value, path=(), notes=None):
    notes = [] if notes is None else notes
    if isinstance(value, dict):
        out = {}
        for key, child in value.items():
            if key in {'device','devices','device_type','logical_device'} and any(p in {'execution','numeric_policy','training_configuration'} for p in path):
                notes.append('.'.join(map(str,path+(key,)))); out[key] = '<execution-device-in-CPU-audit>'
            else:
                out[key] = normalize_execution_metadata(child, path+(key,), notes)
        return out
    if isinstance(value, list): return [normalize_execution_metadata(x,path+(i,),notes) for i,x in enumerate(value)]
    if isinstance(value, tuple): return tuple(normalize_execution_metadata(x,path+(i,),notes) for i,x in enumerate(value))
    return value


def native_restore(cfg, state, dm, npz, comparator, np, torch):
    import timm
    from timm.models.vision_transformer import default_cfgs
    from cl27.common import state_hash
    from utils import factory
    need(timm.__version__ == '0.6.12', 'Native restore requires actual fixed timm0.6.12')
    url = default_cfgs['vit_base_patch16_224']['url']
    need(npz.parent.name == 'checkpoints' and npz.name == Path(urlsplit(url).path).name, 'NPZ must already occupy the exact native hub/checkpoints cache filename')
    torch.hub.set_dir(str(npz.parent.parent))
    random.seed(1993); np.random.seed(1993); torch.random.default_generator.manual_seed(1993)
    native_args = copy.deepcopy(cfg)
    native_args.update(device=[torch.device('cpu')], seed=1993, nb_classes=dm.nb_classes, nb_tasks=dm.nb_tasks)
    model = factory.get_model(cfg['model_name'], native_args)
    imported = sys.modules[model.__class__.__module__]
    need(Path(imported.__file__).resolve().is_relative_to(REPO), 'Learner imported from a different runtime snapshot')
    provenance = copy.deepcopy(model.provenance if cfg['model_name']=='current_adapted' else model.backbone_provenance)
    need(provenance['weight_file_sha256'] == WEIGHT_SHA, 'Native initial model was not bound to the verified NPZ')
    if cfg['model_name']=='current_adapted':
        current = dm.get_dataset(np.arange(dm.nb_classes), source='train', mode='test')
        training = dm.get_dataset(np.arange(dm.nb_classes), source='train', mode='train')
        model.canonical_transform.transform = current.trsf
        model.training_transform.transform = training.trsf
    model.restore_checkpoint(state['learner'])
    optimizer_bindings = {}
    if cfg['model_name'] == 'current_adapted':
        names = {id(parameter):name for name,parameter in model.core.encoder.named_parameters()}
        for name, optimizer in model.core.optimizers.items():
            bound = []
            for group in optimizer.param_groups:
                for parameter in group['params']:
                    key = names.get(id(parameter), '')
                    need(key.startswith('libraries.' if name == 'modules' else 'main.'), 'Optimizer is bound to the wrong native parameter family')
                    saved = optimizer.state[parameter]
                    need(saved['exp_avg'].shape == saved['exp_avg_sq'].shape == parameter.shape, 'Restored optimizer moment/parameter shapes differ')
                    need(saved['exp_avg'].device.type == saved['exp_avg_sq'].device.type == 'cpu', 'Restored optimizer escaped CPU')
                    bound.append(key)
            optimizer_bindings[name] = {'parameter_count':len(bound), 'parameter_names':bound}
    actual = model.checkpoint_state()
    notes = []; reference = normalize_execution_metadata(state['learner'], notes=notes)
    candidate = normalize_execution_metadata(actual)
    comparison = comparator.Comparison(); comparison.walk(reference,candidate)
    need(not comparison.differences, 'Native restored checkpoint_state differs: '+json.dumps(comparison.differences[:3]))
    diagnostics = state['learner']['task_diagnostics']; initial = provenance['initial_state_sha256']
    anchor = model.core.encoder.anchor if cfg['model_name']=='current_adapted' else getattr(model,'anchor',None)
    anchor_hash = state_hash(anchor) if anchor is not None else None
    if anchor is not None:
        need(anchor_hash == initial and all(x.get('anchor_sha256') == initial for x in diagnostics), 'Actual restored anchor / initial NPZ / diagnostic hashes differ')
        need(not any(p.requires_grad for p in anchor.parameters()), 'Native anchor restored as trainable')
    body = model.core.encoder.main if cfg['model_name']=='current_adapted' else model.backbone
    body_hash = state_hash(body)
    need(diagnostics[0]['backbone_sha256_before'] == initial and diagnostics[-1]['backbone_sha256_after'] == body_hash and body_hash != initial, 'Initial/final plastic body tensor identity differs')
    need(all(diagnostics[i]['backbone_sha256_after'] == diagnostics[i+1]['backbone_sha256_before'] for i in range(9)), 'Broken task body hash chain')
    model._network.eval()
    need(all(p.device.type == 'cpu' for p in model._network.parameters()), 'Native model escaped CPU')
    with torch.inference_mode():
        output = model._network(torch.zeros(1,3,224,224,device='cpu'))
    logits = output['logits'] if isinstance(output,dict) else output
    need(tuple(logits.shape) == (1,dm.nb_classes) and bool(torch.isfinite(logits).all()), 'Invalid all-class native deployment output')
    after = comparator.Comparison(); after.walk(reference,normalize_execution_metadata(model.checkpoint_state()))
    need(not after.differences, 'Read-only zero-input forward changed persisted learner state')
    return {'native_cpu_restore_executed':True, 'compared_state_nodes':comparison.compared_nodes,
        'execution_device_metadata_normalized':sorted(set(notes)), 'raw_checkpoint_numerically_restored':True,
        'initial_backbone_sha256_from_verified_npz':initial, 'actual_anchor_sha256':anchor_hash,
        'native_optimizer_parameter_bindings':optimizer_bindings,
        'actual_final_body_sha256':body_hash, 'body_hash_chain_verified':True,
        'output_shape':list(logits.shape), 'output_is_finite':True, 'audit_forward_images':1,
        'extra_training_steps':0, 'accuracy_evaluation_performed':False, 'cuda_rng_applied':False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,required=True,help='Executed model/data runtime snapshot; never modified')
    p.add_argument('--state-loader',type=Path,required=True,help='Independent compare_checkpoint_state.py path, possibly outside --repo')
    p.add_argument('--checkpoint-backup',type=Path,required=True)
    p.add_argument('--expected-sha256',required=True)
    p.add_argument('--metadata-dir',type=Path,required=True)
    p.add_argument('--pretrained-npz',type=Path,required=True,help='Existing SHA-pinned native hub/checkpoints NPZ; never downloaded')
    p.add_argument('--runtime-cwd',type=Path,help='Original launch cwd for relative data/replay paths; defaults to --repo')
    p.add_argument('--data-root',type=Path,help='Optional equivalent original-data mapping; full reconstructed manifest must match')
    p.add_argument('--confirmed-final-backup',action='store_true',required=True)
    args=p.parse_args(); began=time.perf_counter()
    global REPO
    REPO=args.repo.resolve(strict=True)
    checkpoint=args.checkpoint_backup.resolve(strict=True); metadata=args.metadata_dir.resolve(strict=True)
    loader=args.state_loader.resolve(strict=True)
    # Keep the native cache filename if it is a verified symlink to another
    # partition; native timm resolves its URL under this hub/checkpoints name.
    npz=Path(os.path.abspath(args.pretrained_npz))
    need(npz.is_file(), 'Existing pretrained NPZ is missing')
    cwd=(args.runtime_cwd or REPO).resolve(strict=True)
    data_root=args.data_root.resolve(strict=True) if args.data_root else None
    need(sha(checkpoint)==args.expected_sha256,'Finalized backup SHA mismatch')
    need(sha(npz)==WEIGHT_SHA,'Pretrained NPZ SHA mismatch')
    values={n:load_json(metadata/(n+'.json')) for n in ('config','results','data_manifest')}
    cfg,result,manifest=(values[n] for n in ('config','results','data_manifest'))
    config_guard(cfg,result)
    os.environ['CUDA_VISIBLE_DEVICES']=''; sys.dont_write_bytecode=True
    sys.path.insert(0,str(REPO)); os.chdir(cwd)
    spec=importlib.util.spec_from_file_location('independent_cl27_cpu_state_loader',loader)
    comparator=importlib.util.module_from_spec(spec); spec.loader.exec_module(comparator)
    sys.path=[str(REPO)]+[x for x in sys.path if x!=str(REPO)]
    import numpy as np
    import torch
    torch.set_num_threads(4)
    state,digest=comparator.load_trusted_checkpoint(checkpoint)
    need(digest==args.expected_sha256,'Loaded checkpoint SHA differs')
    facts,replay=checkpoint_facts(state,cfg,result,manifest,np,torch)
    dm,data_proof=rebuild_data(cfg,manifest,state,replay,data_root,cwd,facts,np)
    restored=native_restore(cfg,state,dm,npz,comparator,np,torch)
    need(sha(checkpoint)==args.expected_sha256 and sha(npz)==WEIGHT_SHA,'Read-only input bytes changed')
    source_files, source_exclusions = runtime_source_inventory(REPO, sys.modules)
    print(json.dumps({'status':'native_cpu_checkpoint_audit_checks_passed_not_campaign_certificate','final_audit_complete':False,
        'auditor_sha256':sha(Path(__file__)),
        'checkpoint_backup':str(checkpoint),'checkpoint_sha256':digest,'state_loader':str(loader),'state_loader_sha256':sha(loader),
        'executed_model_source_root':str(REPO),'runtime_source_files_sha256':source_files,
        'runtime_source_inventory_exclusions':source_exclusions,
        'metadata_sha256':{n:sha(metadata/(n+'.json')) for n in values},'pretrained_npz_sha256':WEIGHT_SHA,
        'facts':facts,'data_proof':data_proof,'native_restore':restored,'cpu_audit_wall_seconds':time.perf_counter()-began,
        'limits':['Original process termination/selection/runtime source binding and backup receipt remain caller prerequisites.',
            'CE trajectory bytes and full live request preconditions need the separate existing evidence checks.',
            'No GPU continuation or accuracy evaluation; zero-input shape/finiteness only; no extra training.',
            'Count/moment agreement is not an independent full-history membership proof for methods without a historical-ID ledger.']},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
